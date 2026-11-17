"""Transfer between this experiment's direction and the genericity annotations of UDS-G
(Govindarajan, Van Durme & White 2019), in both directions. Added after the first results;
not part of the pre-registration; reported without verdicts.

UDS-G rates every argument of the English Web Treebank for how particular, kind-like and abstract
it is, and every predicate for how particular (episodic) it is. Ratings are normalised around 0;
a noun is taken as particular if arg-particular > 0 and arg-kind <= 0, as kind if the reverse,
and its predicate as episodic if pred-particular > 0, general if < 0. Ambiguous cases are left out.

Readouts: BERT and the decoder at the noun itself (the only readout both data sets share: UDS
targets are arguments inside sentences, so the last-token readout would describe the sentence,
not the noun), each at the layer chosen on Half A for the main experiment.

  (a) UDS -> here: a probe trained on UDS particular-vs-kind nouns scores this experiment's
      sentences; AUC for O vs A-gen (version 2) and for each version-3 contrast if available.
  (b) here -> UDS: the E2 probe (O vs A-gen, all twenty words) scores the UDS nouns; AUC for
      particular vs kind and for episodic vs general predicates.
  (c) the cosine between the two directions.

The UDS data must first be exported (python 3.12 env with `decomp`):
data/raw/uds/uds_genericity_nouns.csv.

python -m mode_probe.uds_transfer     -> results/uds_transfer.csv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import probe as P
from .common import RAW, RESULTS, config, detect_device, set_seed
from .embed import embed

UDS = RAW / "uds" / "uds_genericity_nouns.csv"


def uds_frame() -> pd.DataFrame:
    d = pd.read_csv(UDS, keep_default_na=False, na_values={"pred_particular": [""], "pred_dynamic": [""],
                                                          "pred_hypothetical": [""]})
    d = d[(d["form"] != "") & (d["sentence"] != "")]
    # the readouts truncate at 128 tokens; this experiment's own sentences have at most 40 words
    d = d[d["sentence"].str.split(" ").str.len() <= 60].reset_index(drop=True)
    starts = []
    for sent, idx in zip(d["sentence"], d["token_index"]):
        toks = sent.split(" ")
        starts.append(sum(len(t) + 1 for t in toks[:idx - 1]))
    d["span_a"] = starts
    d["span_b"] = d["span_a"] + d["form"].str.len().astype(int)
    ok = [s[a:b] == f for s, a, b, f in zip(d["sentence"], d["span_a"], d["span_b"], d["form"])]
    d = d[ok].reset_index(drop=True)
    d["word"] = d["form"] + "@" + d["token_index"].astype(str)
    d["sid"] = [f"u{i:06d}" for i in range(len(d))]
    ref = np.where((d.arg_particular > 0) & (d.arg_kind <= 0), "P",
                   np.where((d.arg_kind > 0) & (d.arg_particular <= 0), "K", "-"))
    occ = np.where(d.pred_particular > 0, "ep", np.where(d.pred_particular < 0, "gen", "-"))
    d["ref"], d["occ"] = ref, occ
    return d


def run(device: str | None = None) -> pd.DataFrame:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from .run_all import labeled_frame

    set_seed()
    device = device or detect_device()
    sel = pd.read_csv(RESULTS / "selected.csv").set_index("model")
    nat = labeled_frame()
    u = uds_frame()
    train = (u["split"] == "train").to_numpy()
    rows = []
    for m, readout in (("bert", sel.loc["bert", "layer"]), ("decoder", sel.loc["decoder_target", "layer"])):
        Xn = embed(m, nat, "natural", device)[readout]
        Xu = embed(m, u, "uds", device)[readout]
        C = float(sel.loc["bert" if m == "bert" else "decoder_target", "C"])
        # (a) UDS reference probe -> here (trained on UDS train split)
        ulab = u.assign(label=np.where(u["ref"] == "K", "A-gen", np.where(u["ref"] == "P", "O", "-")))
        tr = train & P.binary_mask(ulab["label"])
        clf_u = P.fit(Xu[tr], P.y_of(ulab["label"][tr]), C)
        s_n = P.score(clf_u, Xn)
        b = P.binary_mask(nat["label"])
        rec = {"model": m, "readout": readout,
               "UDS->here AUC (O vs A-gen)": P.auc(P.y_of(nat["label"][b]), s_n[b])}
        # held-out UDS check of the UDS probe itself
        te = (~train) & P.binary_mask(ulab["label"])
        rec["UDS->UDS AUC (particular vs kind, dev+test)"] = P.auc(P.y_of(ulab["label"][te]), P.score(clf_u, Xu[te]))
        # (b) here -> UDS: E2-style probe on all twenty words
        clf_h = P.fit(Xn[b], P.y_of(nat["label"][b]), C)
        s_u = P.score(clf_h, Xu)
        refm = u["ref"].isin(["P", "K"]).to_numpy()
        occm = u["occ"].isin(["ep", "gen"]).to_numpy()
        rec["here->UDS AUC (particular vs kind)"] = P.auc((u["ref"][refm] == "K").astype(int), s_u[refm])
        rec["here->UDS AUC (episodic vs general predicate)"] = P.auc((u["occ"][occm] == "gen").astype(int), s_u[occm])
        for r_ in ("P", "K"):
            for o_ in ("ep", "gen"):
                mk = ((u["ref"] == r_) & (u["occ"] == o_)).to_numpy()
                rec[f"here->UDS median score {r_}-{o_}"] = float(np.median(s_u[mk]))
                rec[f"n {r_}-{o_}"] = int(mk.sum())
        # (c) directions in the raw embedding space
        du = clf_u[-1].coef_[0] / clf_u[0].scale_
        dh = clf_h[-1].coef_[0] / clf_h[0].scale_
        rec["cosine(UDS reference direction, this direction)"] = P.cosine(du, dh)
        rows.append(rec)
        print("[uds] " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()),
              flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "uds_transfer.csv", index=False)
    return out


if __name__ == "__main__":
    run()
