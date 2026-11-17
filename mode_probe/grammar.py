"""Surface grammar of the target noun (spaCy) and the rule-based labeling checker."""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from .common import target_span, word_forms

DET_TYPES = ["definite", "indefinite", "demonstrative", "possessive", "quantifier", "none"]
TENSES = ["past", "present", "modal", "none"]


@lru_cache(maxsize=1)
def nlp():
    import spacy

    return spacy.load("en_core_web_sm")


def _target_token(doc, word: str, sentence: str):
    a, b = target_span(sentence, word)
    for t in doc:
        if t.idx <= a < t.idx + len(t) or a <= t.idx < b:
            return t
    return doc[0]


def _det_type(tok) -> str:
    for c in tok.children:
        low = c.lower_
        if c.dep_ == "poss" or c.tag_ in ("PRP$", "WP$") or (c.dep_ == "poss" and c.tag_ == "POS"):
            return "possessive"
        if c.dep_ in ("det", "predet"):
            if low == "the":
                return "definite"
            if low in ("a", "an"):
                return "indefinite"
            if low in ("this", "that", "these", "those"):
                return "demonstrative"
            return "quantifier"
        if c.dep_ == "nummod":
            return "quantifier"
    return "none"


def _main_verb(doc):
    root = next((t for t in doc if t.dep_ == "ROOT"), doc[0])
    return root


def _tense(root) -> str:
    toks = [root] + [c for c in root.children if c.dep_ in ("aux", "auxpass")]
    if any(t.tag_ == "MD" for t in toks):
        return "modal"
    finite = [t for t in toks if "Fin" in t.morph.get("VerbForm") or t.tag_ in ("VBD", "VBZ", "VBP")]
    tenses = {m for t in finite for m in t.morph.get("Tense")}
    if not finite or not tenses:
        tenses = {"Past" if t.tag_ == "VBD" else "Pres" for t in finite if t.tag_ in ("VBD", "VBZ", "VBP")}
    if "Past" in tenses:
        return "past"
    if "Pres" in tenses:
        return "present"
    return "none"


def features(word: str, sentence: str) -> dict:
    doc = nlp()(sentence)
    tok = _target_token(doc, word, sentence)
    root = _main_verb(doc)
    n = len(doc)
    return {
        "det": _det_type(tok),
        "plural": int(tok.tag_ in ("NNS", "NNPS") or "Plur" in tok.morph.get("Number")),
        "tense": _tense(root),
        "n_pronouns": sum(t.pos_ == "PRON" for t in doc),
        "has_pronoun": int(any(t.pos_ == "PRON" for t in doc)),
        "length": n,
        "position": tok.i / max(n - 1, 1),
        "is_subject": int(tok.dep_ in ("nsubj", "nsubjpass")),
    }


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot surface-grammar design matrix for E4a (spec features only)."""
    rows = [features(w, s) for w, s in zip(df["word"], df["sentence"])]
    f = pd.DataFrame(rows, index=df.index)
    out = pd.DataFrame(index=df.index)
    for d in DET_TYPES:
        out[f"det_{d}"] = (f["det"] == d).astype(int)
    for t in TENSES:
        out[f"tense_{t}"] = (f["tense"] == t).astype(int)
    for c in ("plural", "has_pronoun", "n_pronouns", "length", "position"):
        out[c] = f[c]
    return out


# ---------------------------------------------------------------- labeling checker
_DEF_RE = r"\b(a|an|the)\s+{w}\s+is\s+"


def check_label(word: str, sentence: str, label: str) -> list[str]:
    """Flag labels that violate the decision tests of LABELING.md.

    Rules: an O whose target is a bare plural subject of a generic simple present;
    an A-def not of the form "a/the X is (a) Y"; an O or A-gen that *is* of the
    "a/the X is a Y" form (test 3 would have made it A-def unless it is about one X).
    """
    flags = []
    doc = nlp()(sentence)
    tok = _target_token(doc, word, sentence)
    root = _main_verb(doc)
    det = _det_type(tok)
    plural = tok.tag_ == "NNS" or "Plur" in tok.morph.get("Number")
    simple_present = (_tense(root) == "present" and not any(
        c.dep_ == "aux" and c.lemma_ == "be" and root.tag_ == "VBG" for c in root.children))
    subj = tok.dep_ in ("nsubj", "nsubjpass")
    # en_core_web_sm often misparses a sentence-initial bare plural as a compound noun
    # ("Dogs bark at strangers" -> compound "Dogs bark"); fall back on surface form.
    if tok.i == 0 and det == "none" and tok.lower_ == word_forms(word)[1].lower():
        plural, subj = True, True
        verbs = [t for t in doc if t.tag_ in ("VBP", "VBZ", "VBD", "MD")]
        simple_present = bool(verbs) and verbs[0].tag_ in ("VBP", "VBZ") and not any(
            t.tag_ == "VBG" for t in doc[verbs[0].i:verbs[0].i + 2])
        if not verbs and root.tag_ == "NN" and root.i == 1:    # "Dogs bark ..." parsed as NP
            simple_present = True
    def_form = re.search(_DEF_RE.format(w=re.escape(word)), sentence, re.IGNORECASE)

    if label == "O" and plural and det == "none" and subj and simple_present:
        flags.append("O_bare_plural_generic_present")
    if label == "A-def" and not def_form:
        flags.append("A-def_not_X_is_Y")
    if label == "A-gen" and def_form and re.search(
            _DEF_RE.format(w=re.escape(word)) + r"(a|an)\s", sentence, re.IGNORECASE):
        flags.append("A-gen_has_X_is_a_Y_form")
    return flags


def check_frame(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    out = df.copy()
    out["flags"] = [";".join(check_label(w, s, l)) for w, s, l in
                    zip(df["word"], df["sentence"], df[label_col])]
    return out
