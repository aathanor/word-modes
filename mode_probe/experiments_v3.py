"""E9 (reference or occasion), the minimal quadruples, E10 (encounter writing) and the E6 rerun,
exactly as fixed in PREREGISTRATION_v3.md. Run after the version-3 labels pass their gate:

python -m mode_probe.experiments_v3      -> results/v3/*.csv, results/v3/summary_v3.md
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from . import probe as P
from .common import DATA, RESULTS, config, detect_device, set_seed
from .embed import embed
from .label_v3 import GOLD, LABELED, LABELS_V3, QWEN

OUT = RESULTS / "v3"
PRIMARY = ["bert", "decoder"]
MODELS = ["bert", "decoder", "mpnet"]
N_BOOT = 1000
MIN_CELL = 20


def _as_binary(labels: pd.Series, pos: str, neg: str) -> pd.Series:
    """Map a pair of v3 cells onto the probe's O / A-gen convention (A-gen = 1)."""
    return labels.map({neg: "O", pos: "A-gen"}).fillna("other")


def _boot_median(x: np.ndarray, rng) -> np.ndarray:
    return np.array([np.median(rng.choice(x, len(x))) for _ in range(N_BOOT)])


def _t_index(s: np.ndarray, lab: np.ndarray, cell: str, rng) -> tuple[float, float, float, int]:
    a, b, c = s[lab == "P-ep"], s[lab == "K-gen"], s[lab == cell]
    if min(len(a), len(b), len(c)) == 0:
        return np.nan, np.nan, np.nan, len(c)
    t = (np.median(c) - np.median(a)) / (np.median(b) - np.median(a))
    bt = (_boot_median(c, rng) - _boot_median(a, rng)) / (_boot_median(b, rng) - _boot_median(a, rng))
    return float(t), float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5)), len(c)


def gate() -> dict:
    lab = pd.read_csv(LABELED)
    gold = pd.read_csv(GOLD, keep_default_na=False)
    m = gold[["sid", "label"]].merge(lab[["sid", "label"]], on="sid", suffixes=("_gold", "_full"))
    k = {"n": len(m), "kappa": float(cohen_kappa_score(m.label_gold, m.label_full)),
         "agreement": float((m.label_gold == m.label_full).mean())}
    if QWEN.exists():
        q = pd.read_csv(QWEN)
        mq = gold[["sid", "label"]].merge(q[["sid", "label"]], on="sid", suffixes=("_gold", "_q"))
        if len(mq) == len(gold):
            k["qwen_kappa"] = float(cohen_kappa_score(mq.label_gold, mq.label_q))
        mm = lab[["sid", "label"]].merge(q[["sid", "label"]], on="sid", suffixes=("_c", "_q"))
        k["qwen_vs_claude_kappa_all"] = float(cohen_kappa_score(mm.label_c, mm.label_q))
        k["qwen_n"] = len(mm)
    pd.crosstab(m.label_gold, m.label_full).to_csv(OUT / "kappa_v3_confusion.csv")
    k["passed"] = k["kappa"] >= config()["labeler"]["kappa_min"]
    (OUT / "kappa_v3.json").write_text(json.dumps(k, indent=2))
    return k


def _frame():
    lab = pd.read_csv(LABELED)
    return lab.reset_index(drop=True)


def _select(df: pd.DataFrame, emb: dict, m: str) -> dict:
    keys = sorted(emb) if m != "decoder" else sorted(emb)
    return P.select_hyperparams(df, emb, keys)


def run(device: str | None = None) -> dict:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    set_seed()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config()["seed"] + 51)
    device = device or detect_device()
    g = gate()
    print(f"[v3] gate: κ = {g['kappa']:.3f} on {g['n']} ({'passed' if g['passed'] else 'FAILED'})")
    if not g["passed"]:
        return {"gate": g}
    nat = _frame()
    emb = {m: embed(m, nat, "v3", device) for m in MODELS}
    lab = nat["label"].to_numpy()
    sets, genre, half = nat["set"].to_numpy(), nat["genre"].to_numpy(), nat["half"].to_numpy()
    wf = np.isin(sets, ["natural", "targeted"])         # Wikipedia and fiction
    binary = nat.assign(label=_as_binary(nat["label"], "K-gen", "P-ep"))
    res = {"gate": g}

    # ---------------------------------------------- choices on Half A (P-ep vs K-gen, Wikipedia+fiction)
    sel = {}
    for m in MODELS:
        sub = binary[wf].reset_index(drop=True)
        e = {k: v[wf] for k, v in emb[m].items()}
        r = _select(sub, e, m)
        sel[m] = {"layer": r["layer"], "C": r["C"]}
    pd.DataFrame([{"model": m, **v} for m, v in sel.items()]).to_csv(OUT / "selected_v3.csv", index=False)

    # ---------------------------------------------- held-out scores, both directions
    scores = {}
    for m in MODELS:
        X, C = emb[m][sel[m]["layer"]], sel[m]["C"]
        s = np.full(len(nat), np.nan)
        for tr_half, te_half in (("A", "B"), ("B", "A")):
            tr = wf & (half == tr_half)
            _, sc = P.train_test(binary, X, tr, half == te_half, C)
            s[half == te_half] = sc
        scores[m] = s

    # ---------------------------------------------- E9 primary
    rows, v9 = [], {}
    for m in MODELS:
        s = scores[m]
        keep = wf
        rec = {"model": m}
        for cell in ("P-gen", "K-ep"):
            t, lo, hi, n = _t_index(s[keep], lab[keep], cell, rng)
            rec.update({f"t({cell})": t, f"t({cell}) lo": lo, f"t({cell}) hi": hi, f"n {cell}": n})
        rec["n P-ep"], rec["n K-gen"] = int((lab[keep] == "P-ep").sum()), int((lab[keep] == "K-gen").sum())
        tp, tk = rec["t(P-gen)"], rec["t(K-ep)"]
        if rec["n P-gen"] < MIN_CELL or rec["n K-ep"] < MIN_CELL:
            v = "underpowered"
        elif tp <= 0.33 and tk >= 0.67 and rec["t(P-gen) hi"] < 0.5 and rec["t(K-ep) lo"] > 0.5:
            v = "follows reference"
        elif tp >= 0.67 and tk <= 0.33 and rec["t(P-gen) lo"] > 0.5 and rec["t(K-ep) hi"] < 0.5:
            v = "follows occasion"
        elif (0.33 < tp < 0.67 and 0.33 < tk < 0.67) or (rec["t(P-gen) lo"] <= 0.5 <= rec["t(P-gen) hi"]) \
                or (rec["t(K-ep) lo"] <= 0.5 <= rec["t(K-ep) hi"]):
            v = "both"
        else:
            v = "no verdict (pattern outside the pre-registered cases)"
        rec["verdict"] = v
        v9[m] = v
        rows.append(rec)
    e9 = pd.DataFrame(rows)
    e9.to_csv(OUT / "e9_reference_occasion.csv", index=False)
    res["e9"] = {"table": e9, "verdicts": v9}

    # ---------------------------------------------- E9 secondary: reference and occasion probes
    v2 = pd.read_csv(DATA / "natural_labeled.csv")[["sid", "label"]].rename(columns={"label": "v2"})
    nv = nat.merge(v2, on="sid", how="left")
    four = np.isin(lab, ["P-ep", "P-gen", "K-ep", "K-gen"]) & wf
    ref = nat.assign(label=np.where(four, np.where(np.char.startswith(lab.astype(str), "K"), "A-gen", "O"), "other"))
    occ = nat.assign(label=np.where(four, np.where(np.char.endswith(lab.astype(str), "gen"), "A-gen", "O"), "other"))
    e2lab = nat.assign(label=nv["v2"].fillna("other").where(nv["set"] == "natural", "other"))
    srows = []
    for m in MODELS:
        X, C = emb[m][sel[m]["layer"]], sel[m]["C"]
        dirs = {}
        for name, df in (("reference", ref), ("occasion", occ), ("E2 (v2 labels)", e2lab)):
            clf, s = P.train_test(df, X, half == "A", half == "B", C)
            sc = clf[0].scale_
            dirs[name] = clf[-1].coef_[0] / sc
            t = (half == "B") & P.binary_mask(df["label"])
            srows.append({"model": m, "probe": name,
                          "A->B auc": P.auc(P.y_of(df["label"][t]), s[P.binary_mask(df["label"][half == "B"])])})
        for name in ("reference", "occasion"):
            srows.append({"model": m, "probe": f"cos({name}, E2)", "value": P.cosine(dirs[name], dirs["E2 (v2 labels)"])})
    pd.DataFrame(srows).to_csv(OUT / "e9_secondary.csv", index=False)

    # ---------------------------------------------- minimal quadruples
    qmask = sets == "quads"
    qmeta = nat.loc[qmask, "source_id"].str.extract(r"quad:(\d+):(.+)")
    qrows, vq = [], {}
    for m in MODELS:
        s = scores[m][qmask]
        q = pd.DataFrame({"qid": qmeta[0].astype(int).to_numpy(), "cell": qmeta[1].to_numpy(), "s": s})
        p = q.pivot(index="qid", columns="cell", values="s")
        refx = (p["K-ep"] + p["K-gen"]) / 2 - (p["P-ep"] + p["P-gen"]) / 2
        occx = (p["P-gen"] + p["K-gen"]) / 2 - (p["P-ep"] + p["K-ep"]) / 2
        k = int((refx > occx).sum())
        p_ref = stats.binomtest(k, len(p), 0.5, alternative="greater").pvalue
        p_occ = stats.binomtest(len(p) - k, len(p), 0.5, alternative="greater").pvalue
        alpha = 0.01
        vq[m] = ("reference larger" if p_ref < alpha else "occasion larger" if p_occ < alpha else "neither")
        qrows.append({"model": m, "n quads": len(p), "reference > occasion": k,
                      "binomial p (reference)": p_ref, "binomial p (occasion)": p_occ,
                      "mean reference effect": float(refx.mean()), "mean occasion effect": float(occx.mean()),
                      "verdict": vq[m]})
    qcheck = nat.loc[qmask].assign(intended=qmeta[1].to_numpy())
    qrows_df = pd.DataFrame(qrows)
    qrows_df.to_csv(OUT / "e9_quads.csv", index=False)
    pd.crosstab(qcheck["intended"], qcheck["label"]).to_csv(OUT / "e9_quads_labels.csv")
    res["quads"] = {"table": qrows_df, "verdicts": vq,
                    "label_agreement": float((qcheck["intended"] == qcheck["label"]).mean())}

    # ---------------------------------------------- E10: encounter writing
    rows10, v10 = [], {}
    for m in MODELS:
        s = scores[m]
        rec, aucs, boots = {"model": m}, {}, {}
        for src, mask in (("wikipedia", wf & (genre == "wikipedia")), ("fiction", wf & (genre == "fiction")),
                          ("encounter", sets == "encounter")):
            t = mask & np.isin(lab, ["P-ep", "K-gen"])
            y, sc = (lab[t] == "K-gen").astype(int), s[t]
            aucs[src] = P.auc(y, sc)
            idx = np.arange(len(y))
            b = []
            for _ in range(N_BOOT):
                i = rng.choice(idx, len(idx))
                if len(np.unique(y[i])) == 2:
                    b.append(P.auc(y[i], sc[i]))
            boots[src] = np.array(b)
            rec.update({f"auc {src}": aucs[src], f"n P-ep {src}": int((lab[t] == "P-ep").sum()),
                        f"n K-gen {src}": int((lab[t] == "K-gen").sum())})
        n = min(len(v) for v in boots.values())
        d_enc = boots["encounter"][:n] - np.maximum(boots["fiction"][:n], boots["wikipedia"][:n])
        pairs = {f"{a}-{b}": boots[a][:n] - boots[b][:n]
                 for a, b in (("encounter", "fiction"), ("encounter", "wikipedia"), ("fiction", "wikipedia"))}
        rec["encounter - max(other)"] = aucs["encounter"] - max(aucs["fiction"], aucs["wikipedia"])
        rec["lo"], rec["hi"] = float(np.percentile(d_enc, 2.5)), float(np.percentile(d_enc, 97.5))
        for k2, v in pairs.items():
            rec[f"{k2} lo"], rec[f"{k2} hi"] = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
        if rec["n P-ep encounter"] < MIN_CELL or rec["n K-gen encounter"] < MIN_CELL:
            v = "underpowered"
        elif rec["encounter - max(other)"] >= 0.03 and rec["lo"] > 0:
            v = "encounter stands out"
        elif all(-0.03 <= rec[f"{k2} lo"] and rec[f"{k2} hi"] <= 0.03 for k2 in pairs):
            v = "mark of particular reference"
        else:
            v = "inconclusive"
        rec["verdict"] = v
        v10[m] = v
        rows10.append(rec)
    e10 = pd.DataFrame(rows10)
    e10.to_csv(OUT / "e10_encounter.csv", index=False)
    res["e10"] = {"table": e10, "verdicts": v10}

    # ---------------------------------------------- E6 rerun: definitions, original rule
    alpha, rows6, v6 = config()["verdicts"]["e6_alpha"], [], {}
    nv2 = nv.assign(label=nv["v2"].fillna("other").where(nv["set"] == "natural", "other"))
    for m in MODELS:
        X, C = emb[m][sel_v2(m)], c_v2(m)
        s = P.lowo_scores(nv2, X, C)
        defs = s[np.isin(sets, ["natural", "targeted"]) & (lab == "A-def")]
        sg = s[(nv2["label"] == "A-gen").to_numpy()]
        med_g = np.median(sg)
        p = float(stats.mannwhitneyu(defs, sg, alternative="greater").pvalue) if len(defs) else np.nan
        frac = float((defs > med_g).mean()) if len(defs) else np.nan
        vd = ("supported" if (p < alpha and frac > 0.5) else "not supported") if len(defs) else "not tested"
        if 0 < len(defs) < config()["verdicts"]["min_per_class"]:
            vd += " (underpowered)"
        v6[m] = vd
        rows6.append({"model": m, "n A-def": len(defs), "fraction beyond A-gen median": frac,
                      "MW p (greater)": p, "verdict": vd})
    e6 = pd.DataFrame(rows6)
    e6.to_csv(OUT / "e6_rerun.csv", index=False)
    res["e6"] = {"table": e6, "verdicts": v6}
    (OUT / "verdicts_v3.json").write_text(json.dumps(
        {"gate": g, "e9": v9, "quads": vq, "quads_label_agreement": res["quads"]["label_agreement"],
         "e10": v10, "e6": v6, "selected": sel}, indent=2, default=str))
    return res


def sel_v2(m: str) -> str:
    return pd.read_csv(RESULTS / "selected.csv").set_index("model").loc[m, "layer"]


def c_v2(m: str) -> float:
    return float(pd.read_csv(RESULTS / "selected.csv").set_index("model").loc[m, "C"])


if __name__ == "__main__":
    run()
