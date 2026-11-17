"""Build data/v3/natural_v3.csv: the sentences labeled under version 3.

  set "natural"   the 5,900 sentences of E1-E7, unchanged (Wikipedia and fiction)
  set "encounter" diaries, letters and travel accounts (E10): up to 150 per word,
                  at most six per book, same filters as the novels
  set "targeted"  Wikipedia sentences collected on purpose: candidate definitions
                  ("A/An/The X is a ...", "Xs are ...") and candidate episodes about the kind
                  ("were introduced", "reached", "became extinct" ...), up to 30 of each per
                  word, at most five per article counting the natural set. Whether a candidate is
                  a definition or a kind-episode is decided by the labels, not by the pattern.
  set "quads"     the 400 written quadruple sentences (quads.py); labeled too, as a check that
                  the labelers read the cells as intended

python -m mode_probe.harvest_v3
"""
from __future__ import annotations

import csv
import re

import numpy as np
import pandas as pd

from .common import DATA, RAW, all_words, config, half_of, word_forms
from .label_v3 import NATURAL, V3

PER_WORD_ENCOUNTER = 150
PER_WORD_TARGETED = 30

KIND_EPISODE = re.compile(
    r"\b(were|was)\s+(first\s+)?(introduced|domesticated|brought|imported|cultivated|bred|invented|"
    r"developed|recorded|mentioned|adopted|banned|exported|reintroduced|patented)\b"
    r"|\b(became|were declared)\s+extinct\b|\b(reached|arrived in|spread to|spread throughout)\b")


def is_definition(word: str, s: str) -> bool:
    sg, pl = word_forms(word)
    return bool(re.match(rf"^(A|An|The)\s+{sg}\s+is\s+(a|an|any|the|one)\b", s)
                or re.match(rf"^{pl.capitalize()}\s+are\s+(a|an|the)?\s*\w+", s))


def is_kind_episode(word: str, s: str) -> bool:
    sg, pl = word_forms(word)
    return bool(KIND_EPISODE.search(s)) and bool(
        re.search(rf"(?<![\w-])({re.escape(pl)}|the {re.escape(sg)})(?![\w-])", s, re.I))


def _title(src: str) -> str:
    return re.sub(r"^wiki(?:dump)?:", "", src)


def _capped(cands: pd.DataFrame, n: int, cap: int, used: dict, key, rng) -> list[dict]:
    out = []
    for r in cands.iloc[rng.permutation(len(cands))].to_dict("records"):
        k = key(r)
        if used.get(k, 0) >= cap:
            continue
        used[k] = used.get(k, 0) + 1
        out.append(r)
        if len(out) >= n:
            break
    return out


def build() -> pd.DataFrame:
    rng = np.random.default_rng(config()["seed"] + 41)
    nat = pd.read_csv(DATA / "natural.csv").assign(set="natural")
    seen = set(nat["sentence"].str.lower())
    rows = []

    enc = pd.read_parquet(RAW / "gutenberg" / "encounter_pool.parquet")
    enc = enc[~enc["sentence"].str.lower().isin(seen)]
    used: dict = {}
    for w in all_words():
        rows += [{**r, "set": "encounter"} for r in _capped(enc[enc.word == w], PER_WORD_ENCOUNTER,
                                                            config()["harvest"]["max_per_source"]["fiction"],
                                                            used, lambda r: r["source_id"], rng)]

    pool = pd.read_parquet(RAW / "wikipedia" / "pool_v3.parquet")
    pool = pool[~pool["sentence"].str.lower().isin(seen)]
    wiki = nat[nat.genre == "wikipedia"]
    used = {(w, _title(s)): n for (w, s), n in wiki.groupby(["word", "source_id"]).size().items()}
    cap = config()["harvest"]["max_per_source"]["wikipedia"]
    for w in all_words():
        pw = pool[pool.word == w]
        for test in (is_definition, is_kind_episode):
            c = pw[[test(w, s) for s in pw["sentence"]]]
            picked = _capped(c, PER_WORD_TARGETED, cap, used, lambda r: (r["word"], _title(r["source_id"])), rng)
            rows += [{**r, "set": "targeted"} for r in picked]
            pw = pw[~pw["sentence"].isin({r["sentence"] for r in picked})]

    from .quads import quad_frame
    q = quad_frame()
    rows += [{"word": r.word, "sentence": r.sentence, "genre": "written", "source_id": f"quad:{r.qid}:{r.cell}",
              "set": "quads"} for r in q.itertuples()]

    add = pd.DataFrame(rows).drop_duplicates("sentence")
    add["half"] = add["word"].map(half_of)
    start = max(int(s[1:]) for s in nat["sid"]) + 1
    add.insert(0, "sid", [f"s{start + i:05d}" for i in range(len(add))])
    cols = ["sid", "word", "sentence", "genre", "source_id", "half", "set"]
    out = pd.concat([nat[cols], add[cols]], ignore_index=True)
    V3.mkdir(parents=True, exist_ok=True)
    out.to_csv(NATURAL, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(out.groupby(["set", "genre"]).size().to_string())
    return out


if __name__ == "__main__":
    build()
