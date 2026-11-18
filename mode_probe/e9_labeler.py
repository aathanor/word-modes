"""E9 by labeler: does the reference/occasion result depend on who labeled? Added after the
version-3 results, after the Qwen cross-check showed the two labelers disagree mostly on P-gen.
Same procedure as E9 (layer and C from results/v3/selected_v3.csv, Half A -> B and back), with
the labels taken from Claude, from Qwen, or only where the two agree. Reported, no verdicts.

python -m mode_probe.e9_labeler     -> results/v3/e9_by_labeler.csv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import probe as P
from .common import config, detect_device, set_seed
from .embed import embed
from .experiments_v3 import MODELS, OUT, _as_binary, _t_index
from .label_v3 import LABELED, QWEN


def run(device: str | None = None) -> pd.DataFrame:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    set_seed()
    device = device or detect_device()
    nat = pd.read_csv(LABELED).reset_index(drop=True)
    q = pd.read_csv(QWEN)[["sid", "label"]].rename(columns={"label": "qwen"})
    nat = nat.merge(q, on="sid", how="left")
    assert nat["qwen"].notna().all(), "Qwen labels incomplete"
    sel = pd.read_csv(OUT / "selected_v3.csv").set_index("model")
    sets, half = nat["set"].to_numpy(), nat["half"].to_numpy()
    wf = np.isin(sets, ["natural", "targeted"])
    variants = {
        "Claude": (nat["label"], np.ones(len(nat), bool)),
        "Qwen": (nat["qwen"], np.ones(len(nat), bool)),
        "both agree": (nat["label"], (nat["label"] == nat["qwen"]).to_numpy()),
    }
    rows = []
    for m in MODELS:
        emb = embed(m, nat.drop(columns="qwen"), "v3", device)
        X, C = emb[sel.loc[m, "layer"]], float(sel.loc[m, "C"])
        for name, (labels, ok) in variants.items():
            rng = np.random.default_rng(config()["seed"] + 51)
            lab = labels.to_numpy()
            binary = nat.assign(label=_as_binary(labels.where(ok, "X"), "K-gen", "P-ep"))
            s = np.full(len(nat), np.nan)
            for tr_half, te_half in (("A", "B"), ("B", "A")):
                _, sc = P.train_test(binary, X, wf & ok & (half == tr_half), half == te_half, C)
                s[half == te_half] = sc
            keep = wf & ok
            rec = {"model": m, "labels": name}
            for cell in ("P-gen", "K-ep"):
                t, lo, hi, n = _t_index(s[keep], lab[keep], cell, rng)
                rec.update({f"t({cell})": t, f"t({cell}) lo": lo, f"t({cell}) hi": hi, f"n {cell}": n})
            rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "e9_by_labeler.csv", index=False)
    print(out.round(3).to_string(index=False))
    return out


if __name__ == "__main__":
    run()
