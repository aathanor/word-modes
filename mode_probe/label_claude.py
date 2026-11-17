"""D2 model labels with Claude through the Message Batches API.

One request per sentence; the system prompt is LABELING.md in full; the output is
restricted to the five labels by a JSON schema. Claude Opus 5.5 does not accept a
temperature setting, so the labels are one recorded run, not a temperature-0 run.

python -m mode_probe.label_claude --pilot 20     # synchronous check on non-gold sentences
python -m mode_probe.label_claude --submit       # submit the batch (idempotent: reuses a pending batch)
python -m mode_probe.label_claude --collect      # wait, collect, retry failures, write natural_labeled.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import anthropic
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from .common import DATA, LABELS, RESULTS, config
from .label import GOLD, LABELED, instruction, user_message

STATE = RESULTS / "claude_batch.json"
RAW_OUT = DATA / "claude_labels.csv"
SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": LABELS}},
          "required": ["label"], "additionalProperties": False}


def _cfg() -> dict:
    return config()["labeler"]["claude"]


def _params(word: str, sentence: str) -> dict:
    c = _cfg()
    return {
        "model": c["model"],
        "max_tokens": c["max_tokens"],
        "system": [{"type": "text", "text": instruction(), "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user_message(word, sentence)}],
        "output_config": {"effort": c["effort"], "format": {"type": "json_schema", "schema": SCHEMA}},
    }


def _parse(msg) -> str | None:
    if msg.stop_reason == "refusal":
        return None
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        lab = json.loads(text)["label"]
    except (ValueError, KeyError, TypeError):
        return None
    return lab if lab in LABELS else None


def _sync_label(client, word: str, sentence: str) -> tuple[str | None, dict]:
    """Synchronous call, used for the pilot and to retry batch failures (with refusal fallback)."""
    p = _params(word, sentence)
    for attempt in range(4):
        try:
            msg = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                              fallbacks="default", **p)
            return _parse(msg), {"in": msg.usage.input_tokens, "out": msg.usage.output_tokens,
                                 "cache_read": msg.usage.cache_read_input_tokens or 0,
                                 "model": msg.model, "stop": msg.stop_reason}
        except anthropic.RateLimitError:
            time.sleep(30 * (attempt + 1))
        except anthropic.APIConnectionError:
            time.sleep(10 * (attempt + 1))
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                time.sleep(10 * (attempt + 1))
            else:
                raise
    return None, {}


def pilot(natural: pd.DataFrame, n: int) -> None:
    gold = set(pd.read_csv(GOLD, keep_default_na=False)["sid"])
    sample = natural[~natural["sid"].isin(gold)].sample(n, random_state=config()["seed"] + 7)
    client = anthropic.Anthropic()
    tin = tout = 0
    for r in sample.itertuples():
        lab, u = _sync_label(client, r.word, r.sentence)
        tin += u.get("in", 0) + u.get("cache_read", 0)
        tout += u.get("out", 0)
        print(f"{lab or '??':8} [{r.word}] {r.sentence[:100]}")
    print(f"\n[pilot] mean input {tin / n:.0f} tokens, mean output {tout / n:.0f} tokens per sentence")


def submit(natural: pd.DataFrame) -> str:
    if STATE.exists():
        st = json.loads(STATE.read_text())
        print(f"[claude] batch {st['id']} already submitted; use --collect")
        return st["id"]
    client = anthropic.Anthropic()
    reqs = [Request(custom_id=r.sid, params=MessageCreateParamsNonStreaming(**_params(r.word, r.sentence)))
            for r in natural.itertuples()]
    batch = client.messages.batches.create(requests=reqs)
    STATE.write_text(json.dumps({"id": batch.id, "n": len(reqs), "model": _cfg()["model"],
                                 "effort": _cfg()["effort"], "submitted": time.strftime("%F %T")}))
    print(f"[claude] submitted batch {batch.id} with {len(reqs)} requests")
    return batch.id


def collect(natural: pd.DataFrame, poll: int = 60) -> pd.DataFrame:
    st = json.loads(STATE.read_text())
    client = anthropic.Anthropic()
    while True:
        b = client.messages.batches.retrieve(st["id"])
        if b.processing_status == "ended":
            break
        c = b.request_counts
        print(f"[claude] {b.processing_status}: processing {c.processing}, succeeded {c.succeeded}, "
              f"errored {c.errored}", flush=True)
        time.sleep(poll)
    labels, usage = {}, {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}
    for res in client.messages.batches.results(st["id"]):
        if res.result.type == "succeeded":
            m = res.result.message
            labels[res.custom_id] = _parse(m)
            usage["in"] += m.usage.input_tokens
            usage["out"] += m.usage.output_tokens
            usage["cache_read"] += m.usage.cache_read_input_tokens or 0
            usage["cache_write"] += m.usage.cache_creation_input_tokens or 0
        else:
            labels[res.custom_id] = None
    failed = [r for r in natural.itertuples() if labels.get(r.sid) is None]
    print(f"[claude] {len(labels) - len(failed)} labeled, {len(failed)} to retry synchronously")
    for r in failed:
        lab, _ = _sync_label(client, r.word, r.sentence)
        labels[r.sid] = lab or "X"
        if lab is None:
            print(f"[claude] no label after retry, set to X: {r.sid}")
    out = natural.copy()
    out["label"] = out["sid"].map(labels)
    out.to_csv(LABELED, index=False, quoting=csv.QUOTE_NONNUMERIC)
    out[["sid", "label"]].to_csv(RAW_OUT, index=False)
    st.update({"collected": time.strftime("%F %T"), "usage": usage, "retried": len(failed)})
    STATE.write_text(json.dumps(st, indent=2))
    (RESULTS / "labeler.json").write_text(json.dumps({"labeler": {"backend": "anthropic_batch", **_cfg()},
                                                      "n": len(out), "usage": usage}, indent=2))
    print(f"[claude] wrote {LABELED}; usage {usage}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args(argv)
    natural = pd.read_csv(DATA / "natural.csv")
    if a.pilot:
        pilot(natural, a.pilot)
    if a.submit:
        submit(natural)
    if a.collect:
        collect(natural)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
