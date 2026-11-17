"""Supplementary: does the mark grow with model size? One family (Qwen3), four sizes.

For each size the readout (target word, last token, mean after the target) and the layer are
chosen on Half A exactly as for the main decoder. Reported per size: transfer between the halves
(E2), the best unlabeled direction pooled over words (E1), and the two E1 variants (E1b, E1c).
Run after run_all (it reads the labels and writes results/scale.csv, which the summary picks up):

python -m mode_probe.scale
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from . import experiments as E
from . import probe as P
from .common import config, detect_device, rdir, set_seed
from .embed import embed


def params(name: str) -> float:
    return {"Qwen/Qwen3-0.6B": 0.6, "Qwen/Qwen3-1.7B": 1.7, "Qwen/Qwen3-4B": 4.0, "Qwen/Qwen3-8B": 8.0}[name]


def run(device: str | None = None) -> pd.DataFrame:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from .run_all import grammar_features, labeled_frame

    set_seed()
    device = device or detect_device()
    nat = labeled_frame()
    grammar = grammar_features(nat)
    thr = config()["verdicts"]["e1_aligned_auc"]
    binm = P.binary_mask(nat["label"])
    y = P.y_of(nat["label"][binm])
    halfA, halfB = (nat["half"] == "A").to_numpy(), (nat["half"] == "B").to_numpy()
    rows = []
    for key in config()["scale_models"]:
        emb = embed(key, nat, "natural", device)
        sel = P.select_hyperparams(nat, emb, sorted(emb))
        X, C = emb[sel["layer"]], sel["C"]
        rec = {"model": key, "name": config()["models"][key]["name"],
               "params_b": params(config()["models"][key]["name"]), "readout": sel["layer"], "C": C}
        for name, tr, te in (("A->B", halfA, halfB), ("B->A", halfB, halfA)):
            _, s = P.train_test(nat, X, tr, te, C)
            t = te & binm
            sv = np.full(len(nat), np.nan); sv[te] = s
            rec[f"{name} auc"] = P.auc(P.y_of(nat["label"][t]), sv[t])
        Xc = X.astype(np.float64).copy()
        for w in nat["word"].unique():
            wm = (nat["word"] == w).to_numpy()
            Xc[wm] -= Xc[wm].mean(0)
        Z = PCA(n_components=10, random_state=config()["seed"]).fit_transform(Xc[binm])
        aucs, rank = E._aligned(Z, y, thr)
        rec.update({"E1 rank": rank, "E1 best auc": max(aucs)})
        ctx = E.Ctx(nat=nat, emb={key: emb}, models=[key], grammar=grammar)
        ctx.sel[key] = {"layer": sel["layer"], "C": C}
        ext = E.e1_extensions(ctx, write=False)
        rec.update({"E1b rank": ext["residual"]["rank_first_aligned"].iloc[0],
                    "E1b best auc": ext["residual"]["best_auc"].iloc[0],
                    "E1c rank": ext["shared"]["rank_first_aligned"].iloc[0],
                    "E1c best auc": ext["shared"]["best_auc"].iloc[0]})
        rows.append(rec)
        print("[scale] " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                     for k, v in rec.items()), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(rdir() / "scale.csv", index=False)
    return out


if __name__ == "__main__":
    run()
