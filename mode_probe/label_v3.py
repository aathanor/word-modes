"""Version-3 labels (LABELING_v3.md): eight labels, reference x occasion.

Same machinery as version 2, with its own criterion, label set and files under data/v3/:

  natural_v3.csv           the sentences (built by harvest_v3.py)
  natural_v3_labeled.csv   Claude Opus 5.5, Message Batches API (main labels)
  gold_v3.csv              the 300-sentence sample, relabeled by Claude Opus 5.5 in separate
                           synchronous requests that never see the batch labels (the gate)
  qwen_v3.csv              the local model, cross-check

python -m mode_probe.label_v3 --submit | --collect | --gold | --qwen
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

from .common import DATA, PKG, RESULTS, config

V3 = DATA / "v3"
NATURAL = V3 / "natural_v3.csv"
LABELED = V3 / "natural_v3_labeled.csv"
GOLD = V3 / "gold_v3.csv"
QWEN = V3 / "qwen_v3.csv"
STATE = RESULTS / "claude_batch_v3.json"
LABELS_V3 = ["P-ep", "P-gen", "K-ep", "K-gen", "A-idiom", "A-simile", "A-def", "X"]
SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": LABELS_V3}},
          "required": ["label"], "additionalProperties": False}


def instruction() -> str:
    return (PKG / "LABELING_v3.md").read_text()


def user_message(word: str, sentence: str) -> str:
    return (f"Target noun: {word}\nSentence: {sentence}\n\n"
            "Apply step 1, then steps 2 and 3, and answer with exactly one label: "
            + ", ".join(LABELS_V3) + ".")


def _params(word: str, sentence: str) -> dict:
    c = config()["labeler"]["claude"]
    return {"model": c["model"], "max_tokens": c["max_tokens"],
            "system": [{"type": "text", "text": instruction(), "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user_message(word, sentence)}],
            "output_config": {"effort": c["effort"], "format": {"type": "json_schema", "schema": SCHEMA}}}


def _parse(msg) -> str | None:
    if msg.stop_reason == "refusal":
        return None
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        lab = json.loads(text)["label"]
    except (ValueError, KeyError, TypeError):
        return None
    return lab if lab in LABELS_V3 else None


def _sync(client, word: str, sentence: str) -> str | None:
    p = _params(word, sentence)
    for attempt in range(4):
        try:
            msg = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                              fallbacks="default", **p)
            return _parse(msg)
        except anthropic.RateLimitError:
            time.sleep(30 * (attempt + 1))
        except anthropic.APIConnectionError:
            time.sleep(10 * (attempt + 1))
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                time.sleep(10 * (attempt + 1))
            else:
                raise
    return None


def submit() -> str:
    if STATE.exists():
        st = json.loads(STATE.read_text())
        print(f"[v3] batch {st['id']} already submitted; use --collect")
        return st["id"]
    nat = pd.read_csv(NATURAL)
    client = anthropic.Anthropic()
    reqs = [Request(custom_id=r.sid, params=MessageCreateParamsNonStreaming(**_params(r.word, r.sentence)))
            for r in nat.itertuples()]
    batch = client.messages.batches.create(requests=reqs)
    STATE.write_text(json.dumps({"id": batch.id, "n": len(reqs), "submitted": time.strftime("%F %T")}))
    print(f"[v3] submitted batch {batch.id} with {len(reqs)} requests")
    return batch.id


def collect(poll: int = 60) -> pd.DataFrame:
    st = json.loads(STATE.read_text())
    nat = pd.read_csv(NATURAL)
    client = anthropic.Anthropic()
    while (b := client.messages.batches.retrieve(st["id"])).processing_status != "ended":
        c = b.request_counts
        print(f"[v3] {b.processing_status}: processing {c.processing}, succeeded {c.succeeded}", flush=True)
        time.sleep(poll)
    labels, usage = {}, {"in": 0, "out": 0, "cache_read": 0}
    for res in client.messages.batches.results(st["id"]):
        if res.result.type == "succeeded":
            m = res.result.message
            labels[res.custom_id] = _parse(m)
            usage["in"] += m.usage.input_tokens
            usage["out"] += m.usage.output_tokens
            usage["cache_read"] += m.usage.cache_read_input_tokens or 0
    failed = [r for r in nat.itertuples() if labels.get(r.sid) is None]
    for r in failed:
        labels[r.sid] = _sync(client, r.word, r.sentence) or "X"
    out = nat.assign(label=nat["sid"].map(labels))
    out.to_csv(LABELED, index=False, quoting=csv.QUOTE_NONNUMERIC)
    st.update({"collected": time.strftime("%F %T"), "usage": usage, "retried": len(failed)})
    STATE.write_text(json.dumps(st, indent=2))
    print(f"[v3] wrote {LABELED}; {len(failed)} retried; usage {usage}")
    return out


def gold() -> pd.DataFrame:
    """The gate sample, relabeled in separate requests. Never reads LABELED."""
    sample = pd.read_csv(DATA / "gold_to_label.csv", keep_default_na=False)[["sid", "word", "sentence"]]
    done = pd.read_csv(GOLD) if GOLD.exists() else pd.DataFrame(columns=["sid", "label"])
    have = dict(zip(done["sid"], done["label"]))
    client = anthropic.Anthropic()
    for r in sample.itertuples():
        if r.sid not in have:
            have[r.sid] = _sync(client, r.word, r.sentence) or "X"
    out = sample.assign(label=sample["sid"].map(have))
    out.to_csv(GOLD, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"[v3] gold sample: {out['label'].value_counts().to_dict()}")
    return out


def qwen() -> pd.DataFrame:
    """Cross-check with the local model: the same modifier step as version 2, then the v3 criterion."""
    import requests

    from .label import modifier_candidate, modifier_check

    lc = config()["labeler"]
    nat = pd.read_csv(NATURAL)
    have = dict(pd.read_csv(QWEN)[["sid", "label"]].itertuples(index=False)) if QWEN.exists() else {}
    t0, todo = time.time(), [r for r in nat.itertuples() if r.sid not in have]
    for i, r in enumerate(todo):
        lab = "X" if modifier_candidate(r.word, r.sentence) and modifier_check(r.word, r.sentence) else None
        if lab is None:
            body = {"model": lc["model"], "stream": False, "think": False, "format": SCHEMA,
                    "options": {"temperature": 0, "seed": config()["seed"], "num_ctx": 6144},
                    "messages": [{"role": "system", "content": instruction()},
                                 {"role": "user", "content": user_message(r.word, r.sentence)}]}
            for attempt in range(3):
                try:
                    resp = requests.post(f"{lc['url']}/api/chat", json=body, timeout=300)
                    resp.raise_for_status()
                    lab = json.loads(resp.json()["message"]["content"])["label"]
                    if lab in LABELS_V3:
                        break
                except (requests.RequestException, KeyError, ValueError):
                    time.sleep(3 * (attempt + 1))
            lab = lab if lab in LABELS_V3 else "X"
        have[r.sid] = lab
        if (i + 1) % 100 == 0 or i + 1 == len(todo):
            nat.assign(label=nat["sid"].map(have)).dropna(subset=["label"]).to_csv(
                QWEN, index=False, quoting=csv.QUOTE_NONNUMERIC)
            print(f"[v3 qwen] {i + 1}/{len(todo)}  {(i + 1) / (time.time() - t0):.2f}/s", flush=True)
    return nat.assign(label=nat["sid"].map(have))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    for f in ("submit", "collect", "gold", "qwen"):
        ap.add_argument(f"--{f}", action="store_true")
    a = ap.parse_args(argv)
    V3.mkdir(parents=True, exist_ok=True)
    if a.gold:
        gold()
    if a.submit:
        submit()
    if a.collect:
        collect()
    if a.qwen:
        qwen()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
