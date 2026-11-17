"""Brysbaert, Warriner & Kuperman (2014) concreteness norms; mean over content words."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

from .common import DATA, word_forms
from .grammar import nlp

PATH = DATA / "brysbaert.csv"
URLS = [
    "https://raw.githubusercontent.com/ArtsEngine/concreteness/master/Concreteness_ratings_Brysbaert_et_al_BRM.txt",
    "http://crr.ugent.be/papers/Concreteness_ratings_Brysbaert_et_al_BRM.txt",
]
INSTRUCTIONS = (
    "Concreteness norms not found. Download 'Concreteness_ratings_Brysbaert_et_al_BRM.txt' "
    "(Brysbaert, Warriner & Kuperman 2014, Behavior Research Methods 46:904-911, supplementary "
    f"material) and save it as {PATH} with at least the columns Word, Conc.M."
)
CONTENT = {"NOUN", "VERB", "ADJ", "ADV"}


def load_norms() -> dict[str, float] | None:
    if not PATH.exists():
        for url in URLS:
            try:
                r = requests.get(url, timeout=60)
                if r.ok and "Conc.M" in r.text[:500]:
                    df = pd.read_csv(io.StringIO(r.text), sep="\t")
                    df.to_csv(PATH, index=False)
                    break
            except requests.RequestException:
                continue
    if not PATH.exists():
        print(INSTRUCTIONS)
        return None
    df = pd.read_csv(PATH)
    df = df[df["Word"].astype(str).str.match(r"^[A-Za-z'-]+$")]
    return dict(zip(df["Word"].str.lower(), df["Conc.M"].astype(float)))


def mean_concreteness(df: pd.DataFrame, norms: dict[str, float]) -> np.ndarray:
    out = np.full(len(df), np.nan)
    for i, (w, s) in enumerate(zip(df["word"], df["sentence"])):
        forms = {f.lower() for f in word_forms(w)}
        vals = []
        for t in nlp()(s):
            if t.pos_ not in CONTENT or t.lower_ in forms or t.lemma_.lower() in forms:
                continue
            v = norms.get(t.lower_, norms.get(t.lemma_.lower()))
            if v is not None:
                vals.append(v)
        if vals:
            out[i] = float(np.mean(vals))
    return out
