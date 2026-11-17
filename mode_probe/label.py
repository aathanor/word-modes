"""D2: gold-set sampling, model labeling (one sentence per call, temperature 0), Cohen's kappa."""
from __future__ import annotations

import csv
import json
import time

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import cohen_kappa_score

from .common import DATA, LABELS, PKG, RESULTS, all_words, config, set_seed

GOLD = DATA / "gold_to_label.csv"
LABELED = DATA / "natural_labeled.csv"


# ---------------------------------------------------------------- gold set
def make_gold(natural: pd.DataFrame) -> pd.DataFrame:
    """15 sentences per word, split as evenly as possible between the two genres."""
    if GOLD.exists():
        print(f"[gold] {GOLD} exists; not resampling")
        return pd.read_csv(GOLD, keep_default_na=False)
    set_seed()
    rng = np.random.default_rng(config()["seed"] + 1)
    k = config()["gold"]["per_word"]
    picks = []
    for i, w in enumerate(all_words()):
        sub = natural[natural["word"] == w]
        first, second = ("wikipedia", "fiction") if i % 2 == 0 else ("fiction", "wikipedia")
        n_first = (k + 1) // 2
        for genre, n in ((first, n_first), (second, k - n_first)):
            g = sub[sub["genre"] == genre]
            idx = rng.choice(len(g), size=min(n, len(g)), replace=False)
            picks.append(g.iloc[idx])
    gold = pd.concat(picks)[["sid", "word", "genre", "half", "sentence"]].copy()
    gold = gold.iloc[rng.permutation(len(gold))]          # hide word/genre order from the labeler
    gold["label"] = ""
    gold.to_csv(GOLD, index=False, quoting=csv.QUOTE_NONNUMERIC)
    return gold


def gold_status() -> tuple[pd.DataFrame | None, int]:
    if not GOLD.exists():
        return None, -1
    gold = pd.read_csv(GOLD, keep_default_na=False)
    gold["label"] = gold["label"].astype(str).str.strip()
    bad = gold[~gold["label"].isin(LABELS)]
    return gold, len(bad)


def gold_ready_message() -> str:
    return (f"Gold set ready for hand-labeling at {GOLD}. Fill `label` with "
            "O / A-gen / A-idiom / A-def / X, then rerun with `--labels-ready`.")


# ---------------------------------------------------------------- model labeling
def instruction() -> str:
    return (PKG / "LABELING.md").read_text()


def user_message(word: str, sentence: str) -> str:
    return (f"Target noun: {word}\nSentence: {sentence}\n\n"
            "Apply the decision tests in order and answer with exactly one label: "
            "O, A-gen, A-idiom, A-def or X.")


SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": LABELS}},
          "required": ["label"]}


def _ollama(model: str, url: str, word: str, sentence: str) -> str:
    body = {"model": model, "stream": False, "think": False, "format": SCHEMA,
            "options": {"temperature": 0, "seed": config()["seed"], "num_ctx": 4096},
            "messages": [{"role": "system", "content": instruction()},
                         {"role": "user", "content": user_message(word, sentence)}]}
    r = requests.post(f"{url}/api/chat", json=body, timeout=300)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])["label"]


def _openai_compatible(model: str, url: str, word: str, sentence: str) -> str:
    body = {"model": model, "temperature": 0, "seed": config()["seed"],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "label", "schema": SCHEMA}},
            "messages": [{"role": "system", "content": instruction()},
                         {"role": "user", "content": user_message(word, sentence)}]}
    r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=300)
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])["label"]


def label_one(word: str, sentence: str) -> str:
    lc = config()["labeler"]
    fn = _ollama if lc["backend"] == "ollama" else _openai_compatible
    model = lc["model"] if lc["backend"] == "ollama" else lc["alt_model"]
    url = lc["url"] if lc["backend"] == "ollama" else lc["alt_url"]
    for attempt in range(3):
        try:
            lab = fn(model, url, word, sentence)
            if lab in LABELS:
                return lab
        except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError):
            time.sleep(3 * (attempt + 1))
    return "X"


# ---------------------------------------------------------------- modifier/material step (v2)
YESNO = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
         "required": ["answer"]}


def modifier_candidate(word: str, sentence: str) -> bool:
    """Parse-based candidates for LABELING.md v2 'modifier or material': the target modifies
    a following noun, or is a bare singular (a possible substance)."""
    from .grammar import _det_type, _target_token, nlp

    doc = nlp()(sentence)
    t = _target_token(doc, word, sentence)
    if t.dep_ in ("compound", "amod", "nmod") and t.head.pos_ in ("NOUN", "PROPN") and t.head.i > t.i:
        return True
    return t.tag_ == "NN" and _det_type(t) == "none" and t.dep_ != "compound"


def modifier_check(word: str, sentence: str) -> bool:
    lc = config()["labeler"]
    q = (f"Sentence: {sentence}\n\nIn this sentence, is the word \"{word}\" used only to modify "
         f"another noun (as in \"stone wall\", \"cow shed\", \"apple pie\") or to name a material "
         f"or substance (as in \"built of stone\"), so that it refers to no individual {word}? "
         "Answer yes or no.")
    body = {"model": lc["model"], "stream": False, "think": False, "format": YESNO,
            "options": {"temperature": 0, "seed": config()["seed"], "num_ctx": 2048},
            "messages": [{"role": "user", "content": q}]}
    for attempt in range(3):
        try:
            r = requests.post(f"{lc['url']}/api/chat", json=body, timeout=300)
            r.raise_for_status()
            return json.loads(r.json()["message"]["content"])["answer"] == "yes"
        except (requests.RequestException, KeyError, ValueError):
            time.sleep(3 * (attempt + 1))
    return False


def label_v2(word: str, sentence: str) -> str:
    if modifier_candidate(word, sentence) and modifier_check(word, sentence):
        return "X"
    return label_one(word, sentence)


def model_label(natural: pd.DataFrame, out=None) -> pd.DataFrame:
    """Label every D1 sentence with the local model; resumable (already-labeled sids are skipped)."""
    out = out or LABELED
    done = {}
    if out.exists():
        prev = pd.read_csv(out)
        done = dict(zip(prev["sid"], prev["label"]))
    todo = natural[~natural["sid"].isin(done)]
    lc = config()["labeler"]
    print(f"[label] {len(done)} done, {len(todo)} to label with {lc['backend']}:{lc['model']}")
    t0 = time.time()
    for i, row in enumerate(todo.itertuples()):
        done[row.sid] = label_v2(row.word, row.sentence)
        if (i + 1) % 100 == 0 or i + 1 == len(todo):
            _write_labeled(natural, done, out)
            rate = (i + 1) / (time.time() - t0)
            print(f"[label] {i + 1}/{len(todo)}  {rate:.2f}/s")
    return _write_labeled(natural, done, out)


def _write_labeled(natural: pd.DataFrame, done: dict, path) -> pd.DataFrame:
    out = natural.copy()
    out["label"] = out["sid"].map(done)
    out = out.dropna(subset=["label"])
    out.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    return out


QWEN_V2 = DATA / "labeling_qwen_v2" / "natural_labeled.csv"


def labeler_name() -> str:
    lc = config()["labeler"]
    return lc["claude"]["model"] if lc["backend"] == "anthropic_batch" else lc["model"]


def qwen_crosscheck() -> None:
    """Cross-family check: the local model relabels everything under LABELING.md v2."""
    config()["labeler"]["backend"] = "ollama"
    QWEN_V2.parent.mkdir(parents=True, exist_ok=True)
    model_label(pd.read_csv(DATA / "natural.csv"), out=QWEN_V2)


# ---------------------------------------------------------------- agreement
def kappa_report(gold: pd.DataFrame, labeled: pd.DataFrame, name: str = "kappa") -> dict:
    """Cohen's kappa of model labels against the gold set, overall and per label (one-vs-rest)."""
    m = gold[["sid", "label"]].merge(labeled[["sid", "label"]], on="sid",
                                     suffixes=("_gold", "_model"))
    rows = [{"label": "overall", "n_gold": len(m),
             "kappa": cohen_kappa_score(m["label_gold"], m["label_model"]),
             "agreement": float((m["label_gold"] == m["label_model"]).mean())}]
    for lab in LABELS:
        g, p = m["label_gold"] == lab, m["label_model"] == lab
        k = cohen_kappa_score(g, p) if g.any() or p.any() else float("nan")
        rows.append({"label": lab, "n_gold": int(g.sum()), "n_model": int(p.sum()),
                     "kappa": k})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"{name}.csv", index=False)
    pd.crosstab(m["label_gold"], m["label_model"]).to_csv(RESULTS / f"{name}_confusion.csv")
    return {"overall": float(rows[0]["kappa"]), "table": df}


if __name__ == "__main__":
    import sys

    if "--qwen-crosscheck" in sys.argv:
        qwen_crosscheck()
