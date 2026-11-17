"""E1-E8. Each function writes its numbers to results/ and one figure to figures/,
and returns a dict used by summary.py. Verdict rules come from config.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

from . import figures as F
from . import probe as P
from .common import all_words, config, rdir

MODEL_LABEL = {"bert": "BERT", "decoder": "Decoder", "decoder_final": "Decoder (final token)",
               "mpnet": "mpnet", "multilingual": "mpnet-multilingual"}


@dataclass
class Ctx:
    nat: pd.DataFrame                       # all natural rows; label = model label; gold = "" or gold label
    emb: dict[str, dict[str, np.ndarray]]   # model -> layer_key -> (n, d)
    models: list[str]
    sel: dict[str, dict] = field(default_factory=dict)
    scores: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    pairs: pd.DataFrame | None = None
    pair_emb: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    grammar: pd.DataFrame | None = None
    conc: np.ndarray | None = None

    def X(self, m: str) -> np.ndarray:
        return self.emb[m][self.sel[m]["layer"]]

    def C(self, m: str) -> float:
        return self.sel[m]["C"]

    @property
    def V(self) -> dict:
        return config()["verdicts"]


def layer_keys(m: str, emb: dict) -> list[str]:
    if m == "decoder_final":
        return sorted(k for k in emb if k.startswith("final_"))
    return sorted(k for k in emb if not k.startswith("final_"))


def _mask(ctx, **kw) -> np.ndarray:
    m = np.ones(len(ctx.nat), bool)
    for col, val in kw.items():
        v = ctx.nat[col].to_numpy()
        m &= np.isin(v, val if isinstance(val, (list, tuple)) else [val])
    return m


def _bin(labels) -> np.ndarray:
    return P.binary_mask(labels)


def _gold_auc(ctx, s_all: np.ndarray, test_mask: np.ndarray) -> tuple[float, int]:
    g = test_mask & _bin(ctx.nat["gold"])
    if g.sum() == 0:
        return float("nan"), 0
    return P.auc(P.y_of(ctx.nat["gold"][g]), s_all[g]), int(g.sum())


def _underpowered(ctx, mask, col="label") -> bool:
    lab = ctx.nat[col][mask]
    n = config()["verdicts"]["min_per_class"]
    return (lab == "O").sum() < n or (lab == "A-gen").sum() < n


def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


# ================================================================ selection
def select_all(ctx: Ctx) -> pd.DataFrame:
    """Choose layer and L2 strength for every model on Half A only."""
    rows = []
    for m in ctx.models:
        src = "decoder" if m == "decoder_final" else m
        keys = layer_keys(m, ctx.emb[src])
        res = P.select_hyperparams(ctx.nat, ctx.emb[src], keys)
        ctx.sel[m] = {"layer": res["layer"], "C": res["C"]}
        if m == "decoder_final":
            ctx.emb[m] = ctx.emb[src]
        g = res["grid"].assign(model=m)
        rows.append(g)
        print(f"[select] {m}: layer {res['layer']}, C {res['C']}")
    grid = pd.concat(rows)
    grid.to_csv(rdir() / "hyperparameters_halfA.csv", index=False)
    pd.DataFrame([{"model": m, **v} for m, v in ctx.sel.items()]).to_csv(
        rdir() / "selected.csv", index=False)
    return grid


# ================================================================ E1
def e1_prominence(ctx: Ctx) -> dict:
    """Settles whether the mode is a main axis of variation of a word's embeddings or a minor one."""
    thr = ctx.V["e1_aligned_auc"]
    rows, pooled_rows, pcs = [], [], {}
    for m in ctx.models:
        X = ctx.X(m)
        centered = np.zeros_like(X)
        for w in all_words():
            wm = _mask(ctx, word=w)
            ref = wm & _bin(ctx.nat["label"])
            if wm.any():
                centered[wm] = X[wm] - (X[ref] if ref.any() else X[wm]).mean(0)
        for w in all_words() + ["(pooled)"]:
            base = _bin(ctx.nat["label"]) & (_mask(ctx, word=w) if w != "(pooled)" else True)
            Xi = (centered if w == "(pooled)" else X)[base]
            y = P.y_of(ctx.nat["label"][base])
            if len(y) < 5 or len(np.unique(y)) < 2:
                continue
            k = min(10, len(y) - 1)
            pca = PCA(n_components=k, random_state=config()["seed"]).fit(Xi)
            Z = pca.transform(Xi)
            aucs = [max(a, 1 - a) for a in (P.auc(y, Z[:, i]) for i in range(k))]
            rank = next((i + 1 for i, a in enumerate(aucs) if a >= thr), None)
            # gold: same components, gold labels on the gold rows of this set
            gm = _bin(ctx.nat["gold"]) & (_mask(ctx, word=w) if w != "(pooled)" else True)
            Zg = pca.transform((centered if w == "(pooled)" else X)[gm]) if gm.any() else None
            yg = P.y_of(ctx.nat["gold"][gm])
            g_aucs = [max(a, 1 - a) for a in (P.auc(yg, Zg[:, i]) for i in range(k))] \
                if gm.any() and len(np.unique(yg)) == 2 else []
            gpos = np.where(gm)[0]
            g_rank = next((i + 1 for i, a in enumerate(g_aucs) if a >= thr), None)
            rec = {"model": m, "word": w, "n": len(y), "rank_first_aligned": rank,
                   "ev_share": float(pca.explained_variance_ratio_[rank - 1]) if rank else np.nan,
                   "best_pc": int(np.argmax(aucs)) + 1, "best_auc": float(max(aucs)),
                   "gold_n": len(gpos), "gold_rank_first_aligned": g_rank,
                   **{f"auc_pc{i + 1}": aucs[i] for i in range(k)}}
            (pooled_rows if w == "(pooled)" else rows).append(rec)
            if w in ("cat", "hammer"):
                allw = _mask(ctx, word=w) & (ctx.nat["label"] != "X").to_numpy()
                pcs[(m, w)] = (pca.transform(X[allw])[:, :2], ctx.nat["label"][allw].to_numpy(),
                               pca.explained_variance_ratio_[:2])
    df = pd.DataFrame(rows + pooled_rows)
    df.to_csv(rdir() / "e1_prominence.csv", index=False)
    F.fig1(pcs, df, ctx.models)

    verdicts = {}
    for m in ctx.models:
        p = df[(df.model == m) & (df.word == "(pooled)")]
        r = p["rank_first_aligned"].iloc[0] if len(p) else None
        r = None if r is None or (isinstance(r, float) and np.isnan(r)) else int(r)
        if r is not None and r <= ctx.V["e1_supported_rank"]:
            verdicts[m] = "supported"
        elif r is not None:
            verdicts[m] = "mixed"
        else:
            verdicts[m] = "not supported"
    return {"table": df, "verdicts": verdicts}


# ================================================================ E2
def e2_transfer(ctx: Ctx) -> dict:
    """Settles whether the mark is one direction shared across words."""
    halfA, halfB = _mask(ctx, half="A"), _mask(ctx, half="B")
    binm = _bin(ctx.nat["label"])
    word_rows, pooled_rows, verdicts, nulls = [], [], {}, {}
    for m in ctx.models:
        X, C = ctx.X(m), ctx.C(m)
        swap = np.full(len(ctx.nat), np.nan)
        rec = {"model": m}
        null_word = {}
        for name, tr, te in (("A->B", halfA, halfB), ("B->A", halfB, halfA)):
            _, s = P.train_test(ctx.nat, X, tr, te, C)
            swap[te] = s
            null_p, null_w = P.permutation_null(ctx.nat, X, tr, te, C)
            null_word.update({w: np.nanpercentile(v, 95) for w, v in null_w.items()})
            t = te & binm
            a = P.auc(P.y_of(ctx.nat["label"][t]), swap[t])
            lo, hi = P.boot_ci(P.y_of(ctx.nat["label"][t]), swap[t])
            ga, gn = _gold_auc(ctx, swap, te)
            rec.update({f"{name} auc": a, f"{name} ci_lo": lo, f"{name} ci_hi": hi,
                        f"{name} null95": float(np.nanpercentile(null_p, 95)),
                        f"{name} null_mean": float(np.nanmean(null_p)),
                        f"{name} gold_auc": ga, f"{name} gold_n": gn,
                        f"{name} underpowered": _underpowered(ctx, te)})
            nulls[(m, name)] = null_p
        lowo = P.lowo_scores(ctx.nat, X, C)
        ctx.scores[m] = {"swap": swap, "lowo": lowo}
        pw = P.per_word_auc(ctx.nat, lowo)
        pw_gold = P.per_word_auc(ctx.nat, lowo, label_col="gold")
        pw_swap = P.per_word_auc(ctx.nat, swap)
        for _, r in pw.iterrows():
            g = pw_gold[pw_gold.word == r.word].iloc[0]
            sw = pw_swap[pw_swap.word == r.word].iloc[0]
            word_rows.append({"model": m, "word": r.word, "half": "A" if r.word in config()["words"]["A"] else "B",
                              "lowo_auc": r.auc, "swap_auc": sw.auc, "null95": null_word.get(r.word),
                              "n_O": r.n_O, "n_Agen": r.n_Agen, "gold_auc": g.auc,
                              "gold_n_O": g.n_O, "gold_n_Agen": g.n_Agen})
        t = binm
        rec["LOWO pooled auc"] = P.auc(P.y_of(ctx.nat["label"][t]), lowo[t])
        rec["LOWO median word auc"] = float(np.nanmedian(pw.auc))
        rec["LOWO words above null95"] = int(sum(
            r["lowo_auc"] > (r["null95"] or 1) for r in word_rows if r["model"] == m))
        ga, gn = _gold_auc(ctx, lowo, np.ones(len(ctx.nat), bool))
        rec["LOWO gold auc"], rec["LOWO gold n"] = ga, gn
        pooled_rows.append(rec)

        both = rec["A->B auc"] > rec["A->B null95"] and rec["B->A auc"] > rec["B->A null95"]
        if both and rec["LOWO median word auc"] >= ctx.V["e2_supported_auc"]:
            verdicts[m] = "supported"
        elif both or rec["A->B auc"] > rec["A->B null95"] or rec["B->A auc"] > rec["B->A null95"]:
            verdicts[m] = "mixed"
        else:
            verdicts[m] = "not supported"
        if rec["A->B underpowered"] or rec["B->A underpowered"]:
            verdicts[m] += " (underpowered)"
    wdf, pdf_ = pd.DataFrame(word_rows), pd.DataFrame(pooled_rows)
    wdf.to_csv(rdir() / "e2_transfer_by_word.csv", index=False)
    pdf_.to_csv(rdir() / "e2_transfer.csv", index=False)
    F.fig2(ctx, [m for m in ctx.models if m in ("bert", "decoder", "mpnet")])
    return {"by_word": wdf, "pooled": pdf_, "verdicts": verdicts}


# ================================================================ E3
def e3_genre(ctx: Ctx, e2: dict) -> dict:
    """Settles whether the mark is genre."""
    binm = _bin(ctx.nat["label"])
    wiki, fic = _mask(ctx, genre="wikipedia"), _mask(ctx, genre="fiction")
    halfA, halfB = _mask(ctx, half="A"), _mask(ctx, half="B")
    rows, verdicts = [], {}
    balance = ctx.nat[binm].groupby(["genre", "label"]).size().unstack(fill_value=0)
    balance.to_csv(rdir() / "e3_class_balance.csv")
    for m in ctx.models:
        X, C = ctx.X(m), ctx.C(m)
        rec = {"model": m}
        for name, tr, te in (("wiki->fiction", wiki, fic), ("fiction->wiki", fic, wiki)):
            _, s = P.train_test(ctx.nat, X, tr, te, C)
            full = np.full(len(ctx.nat), np.nan)
            full[te] = s
            t = te & binm
            y = P.y_of(ctx.nat["label"][t])
            rec[f"{name} auc"] = P.auc(y, full[t])
            rec[f"{name} ci_lo"], rec[f"{name} ci_hi"] = P.boot_ci(y, full[t])
            rec[f"{name} gold_auc"], rec[f"{name} gold_n"] = _gold_auc(ctx, full, te)
            rec[f"{name} underpowered"] = _underpowered(ctx, te)
        for gname, gm in (("wiki", wiki), ("fiction", fic)):
            # same-genre baseline over all words, folds grouped by source document
            idx = np.where(gm & binm)[0]
            groups = ctx.nat["source_id"].to_numpy()[idx]
            y = P.y_of(ctx.nat["label"].to_numpy()[idx])
            oof = np.full(len(idx), np.nan)
            n_splits = min(5, len(np.unique(groups)))
            for tr_i, te_i in GroupKFold(n_splits=n_splits).split(idx, y, groups):
                clf = P.fit(X[idx[tr_i]], y[tr_i], C)
                oof[te_i] = P.score(clf, X[idx[te_i]])
            rec[f"within-{gname} auc"] = P.auc(y, oof)
            # within-genre cross-word transfer (E2 inside the genre)
            vals = []
            for tr, te in ((halfA, halfB), (halfB, halfA)):
                _, s = P.train_test(ctx.nat, X, tr & gm, te & gm, C)
                full = np.full(len(ctx.nat), np.nan)
                full[te & gm] = s
                t = te & gm & binm
                vals.append(P.auc(P.y_of(ctx.nat["label"][t]), full[t]))
            rec[f"within-{gname} cross-word auc"] = float(np.nanmean(vals))
        rec["drop to fiction"] = rec["within-fiction auc"] - rec["wiki->fiction auc"]
        rec["drop to wiki"] = rec["within-wiki auc"] - rec["fiction->wiki auc"]
        e2p = e2["pooled"][e2["pooled"].model == m].iloc[0]
        rec["E2 A<->B mean auc"] = float(np.nanmean([e2p["A->B auc"], e2p["B->A auc"]]))
        rows.append(rec)
        cross = [rec["wiki->fiction auc"], rec["fiction->wiki auc"]]
        drops = [rec["drop to fiction"], rec["drop to wiki"]]
        if min(cross) >= ctx.V["e3_supported_auc"] and max(drops) <= ctx.V["e3_max_drop"]:
            verdicts[m] = "supported"
        elif min(cross) >= 0.6:
            verdicts[m] = "mixed"
        else:
            verdicts[m] = "not supported"
        if rec["wiki->fiction underpowered"] or rec["fiction->wiki underpowered"]:
            verdicts[m] += " (underpowered)"
    df = pd.DataFrame(rows)
    df.to_csv(rdir() / "e3_genre.csv", index=False)
    F.fig3(e2["by_word"], e2["pooled"], df, [m for m in ctx.models if m != "decoder_final"])
    return {"table": df, "balance": balance, "verdicts": verdicts}


# ================================================================ E4
def _lrt(y, sg, se) -> tuple[float, float]:
    import statsmodels.api as sm

    X0 = sm.add_constant(np.column_stack([sg]))
    X1 = sm.add_constant(np.column_stack([sg, se]))
    try:
        ll0 = sm.Logit(y, X0).fit(disp=0, maxiter=200).llf
        ll1 = sm.Logit(y, X1).fit(disp=0, maxiter=200).llf
    except Exception:            # perfect separation or singular design
        return float("nan"), float("nan")
    stat = 2 * (ll1 - ll0)
    return float(stat), float(stats.chi2.sf(stat, 1))


def _resid_auc(y, se, cov) -> float:
    A = np.column_stack([np.ones(len(cov)), cov])
    beta, *_ = np.linalg.lstsq(A, se, rcond=None)
    return P.auc(y, se - A @ beta)


def e4_grammar(ctx: Ctx, pair_words_by_half: bool = True) -> dict:
    """Settles whether the mark is more than the determiner and the tense."""
    binm = _bin(ctx.nat["label"])
    G = ctx.grammar.to_numpy(dtype=float)
    gsel = P.select_hyperparams(ctx.nat, {"G": G}, ["G"])
    Cg = gsel["C"]
    g_lowo = P.lowo_scores(ctx.nat, G, Cg)
    halfA, halfB = _mask(ctx, half="A"), _mask(ctx, half="B")
    g_swap = np.full(len(ctx.nat), np.nan)
    for tr, te in ((halfA, halfB), (halfB, halfA)):
        g_swap[te] = P.train_test(ctx.nat, G, tr, te, Cg)[1]
    g_pw = P.per_word_auc(ctx.nat, g_lowo)

    rows, word_rows, pair_rows, verdicts, grammar_wins = [], [], [], {}, {}
    y = P.y_of(ctx.nat["label"][binm])
    gm = _bin(ctx.nat["gold"])
    yg = P.y_of(ctx.nat["gold"][gm])
    for m in ctx.models:
        se = ctx.scores[m]["lowo"]
        e_pw = P.per_word_auc(ctx.nat, se)
        for (_, er), (_, gr) in zip(e_pw.iterrows(), g_pw.iterrows()):
            word_rows.append({"model": m, "word": er.word, "embedding_auc": er.auc,
                              "grammar_auc": gr.auc})
        diff = e_pw.auc.to_numpy() - g_pw.auc.to_numpy()
        diff = diff[~np.isnan(diff)]
        wil_p = float(stats.wilcoxon(diff, alternative="greater").pvalue) if len(diff) > 5 else np.nan
        stat, p = _lrt(y, g_lowo[binm], se[binm])
        sw = ctx.scores[m]["swap"]
        stat_s, p_s = _lrt(y, g_swap[binm], sw[binm])
        rec = {"model": m, "embedding LOWO auc": P.auc(y, se[binm]),
               "grammar LOWO auc": P.auc(y, g_lowo[binm]),
               "embedding A<->B auc": P.auc(y, sw[binm]), "grammar A<->B auc": P.auc(y, g_swap[binm]),
               "words embedding > grammar": int((diff > 0).sum()), "n words": len(diff),
               "wilcoxon p": wil_p, "LRT stat (LOWO)": stat, "LRT p (LOWO)": p,
               "LRT stat (A<->B)": stat_s, "LRT p (A<->B)": p_s,
               "embedding residual auc (LOWO)": _resid_auc(y, se[binm], g_lowo[binm]),
               "gold embedding auc": P.auc(yg, se[gm]) if gm.any() else np.nan,
               "gold grammar auc": P.auc(yg, g_lowo[gm]) if gm.any() else np.nan,
               "gold n": int(gm.sum())}
        if gm.sum() >= 30 and len(np.unique(yg)) == 2:
            rec["gold LRT stat"], rec["gold LRT p"] = _lrt(yg, g_lowo[gm], se[gm])

        # (b) minimal pairs: Half-A probe on Half-B pairs and back
        pdx, pe = ctx.pairs, ctx.pair_emb[m]
        Xp = pe[ctx.sel[m]["layer"]]
        X = ctx.X(m)
        diffs = np.full(len(pdx) // 2, np.nan)
        meta = pdx.iloc[::2].reset_index(drop=True)
        for tr_half, te_half in (("A", "B"), ("B", "A")):
            tr = _mask(ctx, half=tr_half) & binm
            clf = P.fit(X[tr], P.y_of(ctx.nat["label"][tr]), ctx.C(m))
            s = P.score(clf, Xp)
            sO, sA = s[0::2], s[1::2]
            sel = (meta["half"] == te_half).to_numpy()
            diffs[sel] = (sA - sO)[sel]
        ties = int((np.abs(diffs) < 1e-6).sum())
        k = int((diffs > 1e-6).sum())
        bt = stats.binomtest(k, len(diffs), 0.5, alternative="greater")
        rec.update({"pairs n": len(diffs), "pairs positive": k, "pairs ties": ties,
                    "pairs fraction positive": k / len(diffs), "pairs binomial p": bt.pvalue})
        for sub in ("A-gen", "A-def"):
            mm = (meta["a_sublabel"] == sub).to_numpy()
            rec[f"pairs fraction positive ({sub})"] = float((diffs[mm] > 1e-6).mean()) if mm.any() else np.nan
        for i, d in enumerate(diffs):
            pair_rows.append({"model": m, "pair": i, "word": meta.word[i],
                              "a_sublabel": meta.a_sublabel[i], "diff": d})
        rows.append(rec)

        beats = (rec["LRT p (LOWO)"] < ctx.V["e4_alpha"] and
                 rec["embedding LOWO auc"] > rec["grammar LOWO auc"] and
                 (np.isnan(wil_p) or wil_p < 0.05))
        pairs_ok = bt.pvalue < ctx.V["e4_alpha"]
        grammar_wins[m] = not (rec["embedding LOWO auc"] > rec["grammar LOWO auc"]
                               and rec["LRT p (LOWO)"] < ctx.V["e4_alpha"])
        verdicts[m] = "supported" if beats and pairs_ok else ("mixed" if beats or pairs_ok
                                                             else "not supported")
    df = pd.DataFrame(rows)
    df.to_csv(rdir() / "e4_grammar.csv", index=False)
    wdf = pd.DataFrame(word_rows)
    wdf.to_csv(rdir() / "e4_grammar_by_word.csv", index=False)
    pdf_ = pd.DataFrame(pair_rows)
    pdf_.to_csv(rdir() / "e4_minimal_pairs.csv", index=False)
    pd.DataFrame({"feature": ctx.grammar.columns}).to_csv(rdir() / "e4_grammar_features.csv", index=False)
    F.fig4(wdf, pdf_, [m for m in ctx.models if m != "decoder_final"])
    return {"table": df, "by_word": wdf, "pairs": pdf_, "verdicts": verdicts,
            "grammar_wins": grammar_wins, "grammar_C": Cg}


# ================================================================ E5
def e5_concreteness(ctx: Ctx) -> dict:
    """Settles whether the mark is concreteness."""
    if ctx.conc is None:
        return {"skipped": "concreteness norms unavailable", "verdicts": {}}
    binm = _bin(ctx.nat["label"])
    conc = ctx.conc
    ok = binm & ~np.isnan(conc)
    y = P.y_of(ctx.nat["label"][ok])
    rows, verdicts, fig_data = [], {}, {}
    words = ctx.nat["word"].to_numpy()
    for m in ctx.models:
        X, C = ctx.X(m), ctx.C(m)
        se = ctx.scores[m]["lowo"]
        rho, rp = stats.spearmanr(se[ok], conc[ok])
        before = P.auc(y, se[ok])
        after_score = _resid_auc(y, se[ok], conc[ok])
        # feature-level: remove the part of every embedding dimension predicted by concreteness
        resid_oof = np.full(len(ctx.nat), np.nan)
        cfill = np.where(np.isnan(conc), np.nanmean(conc), conc)
        for w in np.unique(words):
            te = words == w
            tr = ~te & binm
            A = np.column_stack([np.ones(tr.sum()), cfill[tr]])
            beta, *_ = np.linalg.lstsq(A, X[tr], rcond=None)
            Xtr = X[tr] - A @ beta
            Xte = X[te] - np.column_stack([np.ones(te.sum()), cfill[te]]) @ beta
            clf = P.fit(Xtr, P.y_of(ctx.nat["label"][tr]), C)
            resid_oof[te] = P.score(clf, Xte)
        after_feat = P.auc(y, resid_oof[ok])
        gm = _bin(ctx.nat["gold"]) & ~np.isnan(conc)
        yg = P.y_of(ctx.nat["gold"][gm])
        rec = {"model": m, "n": int(ok.sum()), "spearman score~concreteness": rho, "p": rp,
               "concreteness alone auc": 1 - P.auc(y, conc[ok]),
               "auc before": before, "auc after (score residualized)": after_score,
               "auc after (features residualized)": after_feat,
               "drop": before - after_feat,
               "gold auc before": P.auc(yg, se[gm]) if gm.any() else np.nan,
               "gold auc after (features)": P.auc(yg, resid_oof[gm]) if gm.any() else np.nan}
        for lab in ("O", "A-gen"):
            mm = ok & (ctx.nat["label"] == lab).to_numpy()
            rec[f"spearman within {lab}"] = stats.spearmanr(se[mm], conc[mm])[0]
        rows.append(rec)
        drop = rec["drop"]
        verdicts[m] = ("supported" if drop <= ctx.V["e5_max_drop"] else
                       "mixed" if drop <= ctx.V["e5_mixed_drop"] else "not supported")
        fig_data[m] = (se[ok], conc[ok], ctx.nat["label"][ok].to_numpy(), before, after_feat)
    df = pd.DataFrame(rows)
    df.to_csv(rdir() / "e5_concreteness.csv", index=False)
    if "mpnet" in fig_data:
        F.fig5(*fig_data["mpnet"])
    return {"table": df, "verdicts": verdicts}


# ================================================================ E6
def _placement(med_o, med_g, med_i, p_vs_o, p_vs_g, alpha):
    lo, hi = sorted([med_o, med_g])
    if p_vs_g >= alpha and p_vs_o < alpha:
        return "with abstract-mode (A-gen)"
    if p_vs_o >= alpha and p_vs_g < alpha:
        return "with object-mode (O)"
    if med_i > max(med_o, med_g):
        return "beyond A-gen"
    if med_i < min(med_o, med_g):
        return "beyond O"
    if lo <= med_i <= hi:
        return "between O and A-gen"
    return "undetermined"


def e6_predictions(ctx: Ctx) -> dict:
    """Settles whether definitions are the far end of abstract-mode, and whether idioms sit with abstract-mode or elsewhere."""
    alpha = ctx.V["e6_alpha"]
    rows, verdicts = [], {}
    for m in ctx.models:
        s = ctx.scores[m]["lowo"]
        for src, col in (("model labels", "label"), ("gold labels", "gold")):
            lab = ctx.nat[col].to_numpy()
            words = ctx.nat["word"].to_numpy()
            sg, so = s[lab == "A-gen"], s[lab == "O"]
            med_g = np.median(sg) if len(sg) else np.nan
            med_o = np.median(so) if len(so) else np.nan
            rec = {"model": m, "labels": src, "n_O": len(so), "n_Agen": len(sg),
                   "median O": med_o, "median A-gen": med_g}
            # within-word reference: distance from that word's A-gen median
            wmed = {w: np.median(s[(words == w) & (lab == "A-gen")])
                    for w in np.unique(words) if ((words == w) & (lab == "A-gen")).any()}
            for cat in ("A-def", "A-idiom"):
                sc = s[lab == cat]
                rec[f"n {cat}"] = len(sc)
                if len(sc) == 0 or len(sg) == 0:
                    continue
                rec[f"median {cat}"] = float(np.median(sc))
                rec[f"{cat} fraction beyond A-gen median"] = float((sc > med_g).mean())
                cw = [(sv > wmed[w]) for sv, w in zip(sc, words[lab == cat]) if w in wmed]
                rec[f"{cat} fraction beyond own-word A-gen median"] = float(np.mean(cw)) if cw else np.nan
                rec[f"{cat} MW vs A-gen p (greater)"] = float(stats.mannwhitneyu(sc, sg, alternative="greater").pvalue)
                rec[f"{cat} MW vs A-gen p (two-sided)"] = float(stats.mannwhitneyu(sc, sg).pvalue)
                if len(so):
                    rec[f"{cat} MW vs O p (two-sided)"] = float(stats.mannwhitneyu(sc, so).pvalue)
            if rec.get("n A-idiom", 0) and len(so) and len(sg):
                rec["idiom placement"] = _placement(med_o, med_g, rec["median A-idiom"],
                                                    rec["A-idiom MW vs O p (two-sided)"],
                                                    rec["A-idiom MW vs A-gen p (two-sided)"], alpha)
            rows.append(rec)
        r = rows[-2]   # model labels
        n_min = ctx.V["min_per_class"]
        if r.get("n A-def", 0) == 0:
            vd = "not tested (no A-def)"
        else:
            ok = (r["A-def MW vs A-gen p (greater)"] < alpha and
                  r["A-def fraction beyond A-gen median"] > 0.5)
            vd = "supported" if ok else "not supported"
            if r["n A-def"] < n_min:
                vd += " (underpowered)"
        vi = r.get("idiom placement", "not tested (no A-idiom)")
        if r.get("n A-idiom", 0) and r["n A-idiom"] < n_min:
            vi += " (underpowered)"
        verdicts[m] = {"definitions": vd, "idioms": vi}
    df = pd.DataFrame(rows)
    df.to_csv(rdir() / "e6_predictions.csv", index=False)
    F.fig6(ctx, [m for m in ctx.models if m in ("bert", "decoder", "mpnet")])
    return {"table": df, "verdicts": verdicts}


# ================================================================ E7
def e7_agreement(ctx: Ctx) -> dict:
    """Settles whether the models order the same sentences the same way."""
    te = _mask(ctx, half="B") & _bin(ctx.nat["label"])
    gm = te & _bin(ctx.nat["gold"])
    models = [m for m in ctx.models if m in ("bert", "decoder", "mpnet")]
    rows = []
    for a, b in combinations(models, 2):
        sa, sb = ctx.scores[a]["swap"], ctx.scores[b]["swap"]
        rho, p = stats.spearmanr(sa[te], sb[te])
        within = []
        for lab in ("O", "A-gen"):
            mm = te & (ctx.nat["label"] == lab).to_numpy()
            within.append(stats.spearmanr(sa[mm], sb[mm])[0])
        grho = stats.spearmanr(sa[gm], sb[gm])[0] if gm.sum() > 3 else np.nan
        rows.append({"pair": f"{a} vs {b}", "spearman": rho, "p": p, "n": int(te.sum()),
                     "spearman within O": within[0], "spearman within A-gen": within[1],
                     "gold spearman": grho, "gold n": int(gm.sum())})
    df = pd.DataFrame(rows)
    df.to_csv(rdir() / "e7_agreement.csv", index=False)
    F.fig7(ctx, models, te)
    if df.empty:
        return {"table": df, "verdicts": {}, "overall": "not tested"}
    lo = df["spearman"].min()
    overall = ("supported" if lo >= ctx.V["e7_supported_rho"] else
               "mixed" if lo >= ctx.V["e7_mixed_rho"] else "not supported")
    return {"table": df, "verdicts": {r["pair"]: r["spearman"] for r in rows}, "overall": overall}


# ================================================================ E8
def e8_romanian(ctx: Ctx, ro: pd.DataFrame, ro_emb: dict, en_emb: dict) -> dict:
    """Settles whether the mark survives a change of language and of the grammar of definiteness."""
    keys = ["S"]
    sel = P.select_hyperparams(ctx.nat, en_emb, keys)
    Xen, C = en_emb["S"], sel["C"]
    binm = _bin(ctx.nat["label"])
    clf = P.fit(Xen[binm], P.y_of(ctx.nat["label"][binm]), C)
    rb = P.binary_mask(ro["label"])
    s = P.score(clf, ro_emb["S"][rb])
    y = P.y_of(ro["label"][rb])
    a = P.auc(y, s)
    lo, hi = P.boot_ci(y, s)
    # null: English training labels shuffled within word
    rng = np.random.default_rng(config()["seed"])
    words = ctx.nat["word"].to_numpy()[binm]
    ytr = P.y_of(ctx.nat["label"][binm])
    null = []
    for _ in range(config()["probe"]["n_permutations"]):
        yp = ytr.copy()
        for w in np.unique(words):
            g = np.where(words == w)[0]
            yp[g] = rng.permutation(yp[g])
        null.append(P.auc(y, P.score(P.fit(Xen[binm], yp, C), ro_emb["S"][rb])))
    rec = {"model": "multilingual", "C": C, "n": int(rb.sum()), "auc": a, "ci_lo": lo,
           "ci_hi": hi, "null95": float(np.percentile(null, 95)),
           "n words": ro["word_en"].nunique()}
    df = pd.DataFrame([rec])
    df.to_csv(rdir() / "e8_romanian.csv", index=False)
    F.fig8(s, y, np.array(null), a)
    verdict = "supported" if a > rec["null95"] and a >= 0.7 else (
        "mixed" if a > rec["null95"] else "not supported")
    return {"table": df, "verdicts": {"multilingual": verdict}}
