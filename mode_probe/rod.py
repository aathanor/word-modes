"""E11: rod and cap. Do general uses spread more widely than object uses once anisotropy and
topic are controlled? Measure and prediction as fixed in PREREGISTRATION_v3.md (amendment of
17 November 2026), on the version-2 labels.

python -m mode_probe.rod     -> results/e11_rod_cap.csv, results/e11_rod_cap_by_word.csv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

from .common import config, detect_device, rdir, set_seed, target_span
from .embed import embed

K_TOPICS = 8
MIN_PER_CELL = 5
ALPHA = 0.01


def _cosdist_sum(Z: np.ndarray) -> tuple[float, int]:
    """Sum of pairwise cosine distances and the number of pairs."""
    if len(Z) < 2:
        return 0.0, 0
    U = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    G = U @ U.T
    n = len(Z)
    s = (n * n - G.sum()) / 2          # sum over i<j of (1 - cos)
    return float(s), n * (n - 1) // 2


def run(device: str | None = None) -> pd.DataFrame:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from .run_all import labeled_frame

    set_seed()
    device = device or detect_device()
    nat = labeled_frame()
    sel = pd.read_csv(rdir() / "selected.csv").set_index("model")
    masked = nat.copy()
    masked["sentence"] = [s[:a] + "something" + s[b:] for s, w in zip(nat.sentence, nat.word)
                          for a, b in [target_span(s, w)]]
    topic = embed("mpnet", masked, "masked", device)["S"]
    lab, words = nat["label"].to_numpy(), nat["word"].to_numpy()
    readouts = {"bert": ("bert", sel.loc["bert", "layer"]),
                "decoder (at the word)": ("decoder", sel.loc["decoder_target", "layer"]),
                "decoder": ("decoder", sel.loc["decoder", "layer"]),
                "mpnet": ("mpnet", "S")}
    rows = []
    for name, (src, layer) in readouts.items():
        X = embed(src, nat, "natural", device)[layer].astype(np.float64)
        for w in np.unique(words):
            wm = (words == w) & np.isin(lab, ["O", "A-gen"])
            Xw = X[wm]
            Xw = (Xw - Xw.mean(0)) / np.where(Xw.std(0) > 0, Xw.std(0), 1)
            lw = lab[wm]
            raw = {}
            for c in ("O", "A-gen"):
                s, n = _cosdist_sum(Xw[lw == c])
                raw[c] = s / n if n else np.nan
            km = KMeans(n_clusters=K_TOPICS, n_init=10, random_state=config()["seed"]).fit(topic[wm])
            ctrl = {}
            for c in ("O", "A-gen"):
                tot, pairs = 0.0, 0
                for k in range(K_TOPICS):
                    in_k = km.labels_ == k
                    if (in_k & (lw == "O")).sum() < MIN_PER_CELL or (in_k & (lw == "A-gen")).sum() < MIN_PER_CELL:
                        continue
                    s, n = _cosdist_sum(Xw[in_k & (lw == c)])
                    tot, pairs = tot + s, pairs + n
                ctrl[c] = tot / pairs if pairs else np.nan
            rows.append({"model": name, "word": w, "n O": int((lw == "O").sum()), "n A-gen": int((lw == "A-gen").sum()),
                         "raw ratio": raw["A-gen"] / raw["O"], "topic-controlled ratio": ctrl["A-gen"] / ctrl["O"]})
    by_word = pd.DataFrame(rows)
    by_word.to_csv(rdir() / "e11_rod_cap_by_word.csv", index=False)
    summ = []
    for name, g in by_word.groupby("model", sort=False):
        r = g["topic-controlled ratio"].dropna()
        k_up, n = int((r > 1).sum()), len(r)
        p_up = stats.binomtest(k_up, n, 0.5, alternative="greater").pvalue if n else np.nan
        p_dn = stats.binomtest(n - k_up, n, 0.5, alternative="greater").pvalue if n else np.nan
        verdict = ("rod and cap" if p_up < ALPHA else "no rod" if p_dn < ALPHA else "no difference")
        if name not in ("bert", "decoder (at the word)"):
            verdict += " (not a primary readout)"
        summ.append({"model": name, "words": n, "words with A-gen wider": k_up,
                     "median raw ratio": float(g["raw ratio"].median()),
                     "median topic-controlled ratio": float(r.median()) if n else np.nan,
                     "binomial p (wider)": p_up, "binomial p (narrower)": p_dn, "verdict": verdict})
    out = pd.DataFrame(summ)
    out.to_csv(rdir() / "e11_rod_cap.csv", index=False)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    run()
