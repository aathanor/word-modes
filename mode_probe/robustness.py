"""Does the transfer result (E2) depend on which labeler made the labels?

Uses the layer and L2 strength already chosen on Half A, and the cached embeddings.
Three label sets for the same sentences:
  claude     the main labels (claude-opus-5-5)
  qwen       the cross-family relabel under the same criterion (qwen3.8:27b)
  agreed     only sentences on which both labelers agree
Two scorings: probes trained and tested on that label set; and the main probes'
(Claude-trained) scores tested against that label set.

python -m mode_probe.robustness      -> results/label_robustness.csv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import probe as P
from .common import RESULTS, config, rdir
from .embed import embed
from .label import LABELED, QWEN_V2


def frame() -> pd.DataFrame:
    c = pd.read_csv(LABELED)
    q = pd.read_csv(QWEN_V2)[["sid", "label"]].rename(columns={"label": "qwen"})
    nat = c.merge(q, on="sid", how="left").rename(columns={"label": "claude"})
    nat["agreed"] = np.where(nat["claude"] == nat["qwen"], nat["claude"], "X")
    return nat.reset_index(drop=True)


def available(n: int) -> bool:
    return QWEN_V2.exists() and len(pd.read_csv(QWEN_V2)) == n


def from_ctx(ctx) -> pd.DataFrame:
    """Inside run_all, after E2: reuse its rows, embeddings, choices and scores."""
    q = pd.read_csv(QWEN_V2)[["sid", "label"]].rename(columns={"label": "qwen"})
    nat = ctx.nat.merge(q, on="sid", how="left").rename(columns={"label": "claude"})
    nat["agreed"] = np.where(nat["claude"] == nat["qwen"], nat["claude"], "X")
    sel = pd.DataFrame(ctx.sel).T
    main = {m: ctx.scores[m]["swap"] for m in ctx.models if m in ctx.scores}
    return table(nat, ctx.emb, sel, main, rdir())


def run(device: str = "cpu") -> pd.DataFrame:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    nat = frame()
    sel = pd.read_csv(RESULTS / "selected.csv").set_index("model")
    sc = pd.read_csv(RESULTS / "scores.csv").set_index("sid").reindex(nat["sid"])
    emb = {m: embed(m, nat, "natural", device) for m in config()["main_models"]}
    emb["decoder_target"] = emb["decoder"]
    main = {m: sc[f"{m}_swap"].to_numpy(float) for m in sel.index}
    return table(nat, emb, sel, main, RESULTS)


def table(nat, emb, sel, main_scores, out_dir) -> pd.DataFrame:
    halfA, halfB = (nat["half"] == "A").to_numpy(), (nat["half"] == "B").to_numpy()
    rows = []
    for m in sel.index:
        X, C = emb[m][sel.loc[m, "layer"]], float(sel.loc[m, "C"])
        for lab in ("claude", "qwen", "agreed"):
            df = nat.assign(label=nat[lab])
            binm = P.binary_mask(df["label"])
            swap = np.full(len(df), np.nan)
            for tr, te in ((halfA, halfB), (halfB, halfA)):
                _, s = P.train_test(df, X, tr, te, C)
                swap[te] = s
            lowo = P.lowo_scores(df, X, C)
            y = P.y_of(df["label"][binm])
            main = main_scores[m]
            rows.append({
                "model": m, "labels": lab,
                "n_O": int((df["label"] == "O").sum()), "n_Agen": int((df["label"] == "A-gen").sum()),
                "A->B auc": P.auc(P.y_of(df["label"][halfB & binm]), swap[halfB & binm]),
                "B->A auc": P.auc(P.y_of(df["label"][halfA & binm]), swap[halfA & binm]),
                "LOWO pooled auc": P.auc(y, lowo[binm]),
                "LOWO median word auc": float(np.nanmedian(P.per_word_auc(df, lowo).auc)),
                "main probe auc": P.auc(y, main[binm]),
            })
            print(f"[robust] {m:14} {lab:7} " + "  ".join(
                f"{k} {v:.3f}" for k, v in rows[-1].items() if "auc" in k), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "label_robustness.csv", index=False)
    return out


if __name__ == "__main__":
    from .common import detect_device

    run(detect_device())
