"""Bring the Wikipedia sentences reused from the 2025 study under the same rules as the rest.

Those 633 sentences were collected before the five-per-article cap and without article ids.
Their articles were recovered afterwards (attribution.py -> data/legacy_sources.csv). This step:

1. gives every recovered sentence the source id `wiki:<title>`, so the cap and the
   within-Wikipedia baseline (E3) group by real article;
2. replaces sentences whose article could not be recovered (the CC BY-SA licence needs
   attribution, and the repository publishes the text);
3. replaces sentences beyond five per article and word, keeping check-sample sentences first;
4. draws replacements from the same 2023 dump pool the harvest uses, under the same filters
   and cap, and labels them with the same labelers (Claude for the main labels, the local
   model for the cross-check).

Row order is kept: a replacement takes the place of the sentence it replaces. Every change is
logged in data/rebalance_log.csv. Idempotent: a second run finds nothing to replace.

python -m mode_probe.rebalance --plan     # show what would change
python -m mode_probe.rebalance --apply    # replace, label, write the data files
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np
import pandas as pd

from .common import DATA, config
from .label import GOLD, LABELED, QWEN_V2

LEGACY = DATA / "legacy_sources.csv"
LOG = DATA / "rebalance_log.csv"
CLAUDE_RAW = DATA / "claude_labels.csv"


def _title(src: str) -> str | None:
    for p in ("wiki:", "wikidump:"):
        if src.startswith(p):
            return src[len(p):]
    return None


def plan(nat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (nat with legacy source ids resolved, rows to replace with a reason)."""
    nat = nat.copy()
    titles = dict(pd.read_csv(LEGACY, keep_default_na=False)[["sid", "title"]].itertuples(index=False))
    legacy = nat["source_id"].str.startswith("legacy:")
    nat.loc[legacy, "source_id"] = [f"wiki:{titles[s]}" if titles.get(s) else src
                                    for s, src in zip(nat.loc[legacy, "sid"], nat.loc[legacy, "source_id"])]
    out = [{"sid": s, "reason": "article not recovered"}
           for s in nat.loc[nat["source_id"].str.startswith("legacy:"), "sid"]]

    cap = config()["harvest"]["max_per_source"]["wikipedia"]
    gold = set(pd.read_csv(GOLD, keep_default_na=False)["sid"])
    rng = np.random.default_rng(config()["seed"] + 21)
    wiki = nat[(nat["genre"] == "wikipedia") & ~nat["source_id"].str.startswith("legacy:")].copy()
    wiki["article"] = wiki["source_id"].map(_title)
    for (w, a), g in wiki.groupby(["word", "article"]):
        if len(g) <= cap:
            continue
        keep_first = g[g["sid"].isin(gold)]
        rest = g[~g["sid"].isin(gold)]
        rest = rest.iloc[rng.permutation(len(rest))]
        drop = rest.iloc[max(cap - len(keep_first), 0):]
        out += [{"sid": s, "reason": f"over cap: {a}"} for s in drop["sid"]]
    return nat, pd.DataFrame(out, columns=["sid", "reason"])


def replacements(nat: pd.DataFrame, drop: pd.DataFrame) -> dict[str, dict]:
    """sid to replace -> new row, drawn from the dump pool under the harvest's filters and cap."""
    from .harvest import dump_wikipedia

    cap = config()["harvest"]["max_per_source"]["wikipedia"]
    rng = np.random.default_rng(config()["seed"] + 22)
    by_word = nat[nat["sid"].isin(drop["sid"])].groupby("word")["sid"].apply(list).to_dict()
    pool = dump_wikipedia(sorted(by_word))
    kept = nat[~nat["sid"].isin(drop["sid"])]
    seen = set(kept["sentence"].str.lower()) | set(nat["sentence"].str.lower())
    out, next_id = {}, max(int(s[1:]) for s in nat["sid"]) + 1
    for w, sids in by_word.items():
        per_art = (kept[(kept.word == w) & (kept.genre == "wikipedia")]["source_id"]
                   .map(_title).value_counts().to_dict())
        cand = [pool[w][i] for i in rng.permutation(len(pool[w]))]
        it = iter(cand)
        for sid in sids:
            for r in it:
                a = _title(r["source_id"])
                if r["sentence"].lower() in seen or per_art.get(a, 0) >= cap:
                    continue
                seen.add(r["sentence"].lower())
                per_art[a] = per_art.get(a, 0) + 1
                out[sid] = {**r, "sid": f"s{next_id:05d}"}
                next_id += 1
                break
            else:
                raise RuntimeError(f"dump pool exhausted for {w}")
    return out


def _claude_labels(rows: pd.DataFrame) -> dict[str, str]:
    import anthropic

    from .label_claude import _sync_label

    client, out = anthropic.Anthropic(), {}
    for r in rows.itertuples():
        lab, _ = _sync_label(client, r.word, r.sentence)
        out[r.sid] = lab or "X"
        print(f"[rebalance] claude {r.sid} {out[r.sid]:8} {r.word}: {r.sentence[:80]}", flush=True)
    return out


def _qwen_labels(rows: pd.DataFrame) -> dict[str, str]:
    from .label import label_v2

    config()["labeler"]["backend"] = "ollama"
    return {r.sid: label_v2(r.word, r.sentence) for r in rows.itertuples()}


def apply() -> int:
    nat0 = pd.read_csv(DATA / "natural.csv")
    nat, drop = plan(nat0)
    if drop.empty:
        print("[rebalance] nothing to replace")
        nat[nat0.columns].to_csv(DATA / "natural.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
        return 0
    rep = replacements(nat, drop)
    new = pd.DataFrame([{**rep[s], "half": nat.loc[nat.sid == s, "half"].iloc[0]} for s in drop["sid"]])
    labels_c = _claude_labels(new)
    labels_q = _qwen_labels(new) if QWEN_V2.exists() else {}

    idx = nat.set_index("sid")
    for old, r in zip(drop["sid"], new.itertuples()):
        for col in ("word", "sentence", "genre", "source_id", "half"):
            idx.loc[old, col] = getattr(r, col)
    idx = idx.rename(index=dict(zip(drop["sid"], new["sid"])))
    out = idx.reset_index()[nat0.columns]
    out.to_csv(DATA / "natural.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

    def relabel(path, new_labels):
        lab = pd.read_csv(path)
        m = dict(zip(lab["sid"], lab["label"]))
        m.update(new_labels)
        o = out.copy()
        o["label"] = o["sid"].map(m)
        assert o["label"].notna().all(), path
        return o

    c = relabel(LABELED, labels_c)
    c.to_csv(LABELED, index=False, quoting=csv.QUOTE_NONNUMERIC)
    c[["sid", "label"]].to_csv(CLAUDE_RAW, index=False)
    if labels_q:
        relabel(QWEN_V2, labels_q).to_csv(QWEN_V2, index=False, quoting=csv.QUOTE_NONNUMERIC)
    log = drop.assign(replaced_by=new["sid"].values, word=new["word"].values,
                      new_source=new["source_id"].values,
                      claude_label=[labels_c[s] for s in new["sid"]],
                      qwen_label=[labels_q.get(s, "") for s in new["sid"]])
    log.to_csv(LOG, index=False)
    print(f"[rebalance] replaced {len(drop)} sentences; log in {LOG}")
    print(log["reason"].str.split(":").str[0].value_counts().to_string())
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    if a.plan:
        _, drop = plan(pd.read_csv(DATA / "natural.csv"))
        print(drop["reason"].value_counts().to_string() if len(drop) else "nothing to replace")
    if a.apply:
        return apply()
    return 0


if __name__ == "__main__":
    sys.exit(main())
