"""Linear probe (O = 0, A-gen = 1), transfer protocols, permutation null, bootstrap CIs.

A higher probe score means the embedding sits further toward the abstract-mode side.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import assert_no_half_b, config, select_rows

warnings.filterwarnings("ignore", category=ConvergenceWarning)
N_JOBS = 8

BINARY = ["O", "A-gen"]


def y_of(labels) -> np.ndarray:
    return (np.asarray(labels) == "A-gen").astype(int)


def fit(X: np.ndarray, y: np.ndarray, C: float, seed: int | None = None):
    seed = config()["seed"] if seed is None else seed
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=C, max_iter=3000, random_state=seed))
    return clf.fit(X, y)


def score(clf, X: np.ndarray) -> np.ndarray:
    return clf.decision_function(X)


def auc(y, s) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def boot_ci(y, s, n: int | None = None, seed: int | None = None) -> tuple[float, float]:
    n = n or config()["probe"]["n_bootstrap"]
    rng = np.random.default_rng(config()["seed"] if seed is None else seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(roc_auc_score(y[i], s[i]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def binary_mask(labels) -> np.ndarray:
    return np.isin(np.asarray(labels), BINARY)


# ---------------------------------------------------------------- hyperparameters (Half A only)
def lowo_auc_within(df: pd.DataFrame, X: np.ndarray, C: float) -> float:
    """Mean held-out-word AUC, leave-one-word-out inside the given rows."""
    words = df["word"].to_numpy()
    y = y_of(df["label"])
    aucs = []
    for w in np.unique(words):
        tr, te = words != w, words == w
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        aucs.append(auc(y[te], score(fit(X[tr], y[tr], C), X[te])))
    return float(np.mean(aucs)) if aucs else float("nan")


def select_hyperparams(df: pd.DataFrame, emb: dict[str, np.ndarray],
                       layer_keys: list[str]) -> dict:
    """Choose layer and L2 strength by LOWO AUC inside Half A. Half B never enters."""
    tune = select_rows(df, halves=("A",), tuning=True)
    tune = tune[binary_mask(tune["label"])]
    assert_no_half_b(tune)
    pos = df.index.get_indexer(tune.index)
    rows = []
    for lk in layer_keys:
        for C in config()["probe"]["C_grid"]:
            rows.append({"layer": lk, "C": C, "auc": lowo_auc_within(tune, emb[lk][pos], C)})
    grid = pd.DataFrame(rows)
    best = grid.sort_values(["auc", "C"], ascending=[False, True]).iloc[0]
    return {"layer": best["layer"], "C": float(best["C"]), "grid": grid}


# ---------------------------------------------------------------- protocols
def train_test(df, X, train_mask, test_mask, C):
    """Train on binary rows of train_mask; return scores for *all* rows of test_mask."""
    tr = train_mask & binary_mask(df["label"])
    clf = fit(X[tr], y_of(df["label"][tr]), C)
    return clf, score(clf, X[test_mask])


def lowo_scores(df, X, C, words=None) -> np.ndarray:
    """Out-of-fold score for every row: each word scored by a probe that never saw it."""
    out = np.full(len(df), np.nan)
    wcol = df["word"].to_numpy()
    for w in (words or np.unique(wcol)):
        te = wcol == w
        _, s = train_test(df, X, ~te, te, C)
        out[te] = s
    return out


def per_word_auc(df, s, label_col="label") -> pd.DataFrame:
    rows = []
    for w, g in df.groupby("word", sort=False):
        m = binary_mask(g[label_col])
        sw = s[df.index.get_indexer(g.index)]
        yw = y_of(g[label_col])
        rows.append({"word": w, "auc": auc(yw[m], sw[m]),
                     "n_O": int((g[label_col] == "O").sum()),
                     "n_Agen": int((g[label_col] == "A-gen").sum())})
    return pd.DataFrame(rows)


def permutation_null(df, X, train_mask, test_mask, C, n=None, seed=None):
    """Shuffle training labels within word; return pooled and per-word AUC null draws."""
    n = n or config()["probe"]["n_permutations"]
    rng = np.random.default_rng(config()["seed"] if seed is None else seed)
    tr = train_mask & binary_mask(df["label"])
    te = test_mask & binary_mask(df["label"])
    Xtr, Xte = X[tr], X[te]
    ytr = y_of(df["label"][tr])
    yte = y_of(df["label"][te])
    wtr = df["word"][tr].to_numpy()
    wte = df["word"][te].to_numpy()
    groups = [np.where(wtr == w)[0] for w in np.unique(wtr)]
    perms = []
    for _ in range(n):
        yp = ytr.copy()
        for g in groups:
            yp[g] = rng.permutation(yp[g])
        perms.append(yp)
    scaler = StandardScaler().fit(Xtr)          # scaling does not depend on labels
    Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)

    def one(yp):
        clf = LogisticRegression(C=C, max_iter=3000, random_state=config()["seed"]).fit(Ztr, yp)
        return clf.decision_function(Zte)

    scores = Parallel(n_jobs=N_JOBS)(delayed(one)(yp) for yp in perms)
    words = np.unique(wte)
    pooled = np.array([auc(yte, s) for s in scores])
    per_word = {w: np.array([auc(yte[wte == w], s[wte == w]) for s in scores]) for w in words}
    return pooled, per_word
