"""One figure per experiment. PNG (200 dpi) and PDF, white background, no titles,
font 11, sized for an 8.5 cm column (panels are stacked vertically)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from .common import config, fdir  # noqa: E402

W = 8.5 / 2.54
C_O, C_A = "#1f4e79", "#d9822b"
NAMES = {"bert": "BERT", "decoder": "Decoder", "decoder_target": "Decoder (at the word)",
         "mpnet": "mpnet", "multilingual": "mpnet-multi"}

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                     "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def _style(lab):
    if lab == "O":
        return dict(c=C_O, marker="o", s=10, alpha=0.6, linewidths=0)
    if lab == "A-gen":
        return dict(c=C_A, marker="o", s=10, alpha=0.6, linewidths=0)
    if lab == "A-def":
        return dict(facecolors="none", edgecolors="black", marker="o", s=22, linewidths=0.8)
    return dict(c="grey", marker="^", s=20, alpha=0.9, linewidths=0)


ORDER = ["O", "A-gen", "A-def", "A-idiom"]


def _tag(ax, text):
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left", fontsize=9)


def _save(fig, name):
    fig.savefig(fdir() / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(fdir() / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(fdir() / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def _legend(fig, labels=ORDER):
    handles = [plt.scatter([], [], **_style(l)) for l in labels]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, columnspacing=0.8)


# ---------------------------------------------------------------- Fig 1
def fig1(pcs, table, models):
    models = [m for m in models if any(k[0] == m for k in pcs)]
    n = len(models)
    fig = plt.figure(figsize=(W, 1.75 * n + 2.0))
    gs = fig.add_gridspec(n + 1, 2, height_ratios=[1] * n + [1.1], hspace=0.45, wspace=0.3)
    for i, m in enumerate(models):
        for j, w in enumerate(("cat", "hammer")):
            ax = fig.add_subplot(gs[i, j])
            if (m, w) in pcs:
                Z, lab, ev = pcs[(m, w)]
                for l in ORDER:
                    k = lab == l
                    if k.any():
                        ax.scatter(Z[k, 0], Z[k, 1], **_style(l))
                ax.set_xlabel(f"PC1 ({ev[0]:.0%})", fontsize=9, labelpad=1)
                ax.set_ylabel(f"PC2 ({ev[1]:.0%})", fontsize=9, labelpad=1)
            ax.set_xticks([]); ax.set_yticks([])
            _tag(ax, f"{NAMES[m]}, {w}")
    ax = fig.add_subplot(gs[n, :])
    t = table[table.word != "(pooled)"]
    words = list(dict.fromkeys(t.word))
    x = np.arange(len(words))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        r = t[t.model == m].set_index("word").reindex(words)["rank_first_aligned"]
        vals = r.fillna(11).to_numpy()
        ax.bar(x + i * width, vals, width, label=NAMES[m],
               color=["#555555", "#999999", "#cccccc", "#777777"][i % 4])
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(words, rotation=90, fontsize=8)
    ax.set_ylabel("rank of first\nmode-aligned PC", fontsize=9)
    ax.set_yticks([1, 3, 5, 11]); ax.set_yticklabels(["1", "3", "5", ">10"])
    ax.legend(frameon=False, fontsize=8, ncol=len(models), loc="lower left",
              bbox_to_anchor=(0, 1.0))
    _legend(fig)
    _save(fig, "fig1_e1_prominence")


# ---------------------------------------------------------------- Fig 2
def fig2(ctx, models):
    B = config()["words"]["B"]
    rng = np.random.default_rng(config()["seed"])
    fig, axes = plt.subplots(len(models), 1, figsize=(W, 2.9 * len(models)), squeeze=False)
    lab = ctx.nat["label"].to_numpy()
    words = ctx.nat["word"].to_numpy()
    for ax, m in zip(axes[:, 0], models):
        s = ctx.scores[m]["swap"]
        for i, w in enumerate(B):
            for l in ORDER:
                k = (words == w) & (lab == l) & ~np.isnan(s)
                if k.any():
                    ax.scatter(s[k], i + rng.uniform(-0.25, 0.25, k.sum()), **_style(l))
        ax.set_yticks(range(len(B))); ax.set_yticklabels(B, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_xlabel("probe score (Half-A probe)", fontsize=9)
        _tag(ax, NAMES[m])
    fig.tight_layout()
    _legend(fig)
    _save(fig, "fig2_e2_strip")


# ---------------------------------------------------------------- Fig 3
def fig3(by_word, pooled, e3, models):
    words = config()["words"]["A"] + config()["words"]["B"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 5.2), gridspec_kw={"height_ratios": [1.6, 1]})
    width = 0.8 / max(len(models), 1)
    shades = ["#1f4e79", "#d9822b", "#7f7f7f", "#b0b0b0"]
    x = np.arange(len(words))
    for i, m in enumerate(models):
        t = by_word[by_word.model == m].set_index("word").reindex(words)
        a1.bar(x + i * width, t["lowo_auc"], width, color=shades[i % 4], label=NAMES[m])
        nulls = t["null95"].to_numpy(dtype=float)
        for xi, nv in zip(x, nulls):
            if not np.isnan(nv):
                a1.plot([xi + i * width - width / 2, xi + i * width + width / 2], [nv, nv],
                        color="k", ls="--", lw=0.6)
    a1.axhline(0.5, color="k", ls=":", lw=0.8)
    a1.axvline(len(config()["words"]["A"]) - 0.5 + width * (len(models) - 1) / 2, color="grey", lw=0.5)
    a1.set_xticks(x + width * (len(models) - 1) / 2)
    a1.set_xticklabels(words, rotation=90, fontsize=8)
    a1.set_ylabel("held-out-word AUC", fontsize=9)
    a1.set_ylim(0.3, 1.0)
    a1.legend(frameon=False, fontsize=8, ncol=len(models), loc="lower left", bbox_to_anchor=(0, 1.0))
    conds = ["wiki->fiction", "fiction->wiki", "within-wiki", "within-fiction"]
    xs = np.arange(len(conds))
    for i, m in enumerate(models):
        r = e3[e3.model == m].iloc[0]
        vals = [r[f"{c} auc"] for c in conds]
        a2.bar(xs + i * width, vals, width, color=shades[i % 4])
    a2.axhline(0.5, color="k", ls=":", lw=0.8)
    a2.set_xticks(xs + width * (len(models) - 1) / 2)
    a2.set_xticklabels(["Wiki→Fic", "Fic→Wiki", "Wiki (CV)", "Fic (CV)"], fontsize=8)
    a2.set_ylabel("AUC", fontsize=9)
    a2.set_ylim(0.3, 1.0)
    fig.tight_layout()
    _save(fig, "fig3_e2e3_transfer")


# ---------------------------------------------------------------- Fig 4
def fig4(by_word, pairs, models):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 6.0))
    shades = ["#1f4e79", "#d9822b", "#7f7f7f"]
    for i, m in enumerate(models):
        t = by_word[by_word.model == m]
        off = (i - (len(models) - 1) / 2) * 0.25
        for gx, ex in zip(t["grammar_auc"], t["embedding_auc"]):
            a1.plot([0 + off, 1 + off], [gx, ex], color=shades[i % 3], lw=0.6, alpha=0.6)
        a1.scatter(np.zeros(len(t)) + off, t["grammar_auc"], s=10, color=shades[i % 3])
        a1.scatter(np.ones(len(t)) + off, t["embedding_auc"], s=10, color=shades[i % 3],
                   label=NAMES[m])
    a1.set_xticks([0, 1]); a1.set_xticklabels(["grammar", "embedding"])
    a1.set_xlim(-0.6, 1.6)
    a1.axhline(0.5, color="k", ls=":", lw=0.8)
    a1.set_ylabel("held-out-word AUC", fontsize=9)
    a1.legend(frameon=False, fontsize=8, loc="lower right")
    words = list(dict.fromkeys(pairs.word))
    rng = np.random.default_rng(config()["seed"])
    for i, m in enumerate(models):
        t = pairs[pairs.model == m]
        off = (i - (len(models) - 1) / 2) * 0.25
        xs = np.array([words.index(w) for w in t.word]) + off + rng.uniform(-0.06, 0.06, len(t))
        d = t["diff"].to_numpy()
        sd = np.nanstd(d) or 1.0
        a2.scatter(xs, d / sd, s=6, color=shades[i % 3], alpha=0.7)
    a2.axhline(0, color="k", lw=0.8)
    a2.set_xticks(range(len(words))); a2.set_xticklabels(words, rotation=90, fontsize=8)
    a2.set_ylabel("probe(A) − probe(O)\n(per-model SD units)", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig4_e4_grammar")


# ---------------------------------------------------------------- Fig 5
def fig5(score, conc, labels, auc_before, auc_after):
    fig, ax = plt.subplots(figsize=(W, W * 0.85))
    for l in ("O", "A-gen"):
        k = labels == l
        ax.scatter(conc[k], score[k], **_style(l), label=l)
    b = np.polyfit(conc, score, 1)
    xx = np.linspace(np.nanmin(conc), np.nanmax(conc), 50)
    ax.plot(xx, np.polyval(b, xx), color="k", lw=1)
    ax.set_xlabel("mean concreteness of context words")
    ax.set_ylabel("probe score (mpnet)")
    ax.text(0.98, 0.02, f"AUC before {auc_before:.2f}\nAUC after {auc_after:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="grey", lw=0.5))
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _save(fig, "fig5_e5_concreteness")


# ---------------------------------------------------------------- Fig 6
def fig6(ctx, models):
    B = ctx.nat["half"].to_numpy() == "B"
    lab = ctx.nat["label"].to_numpy()
    rng = np.random.default_rng(config()["seed"])
    fig, axes = plt.subplots(len(models), 1, figsize=(W, 1.9 * len(models)), squeeze=False)
    for ax, m in zip(axes[:, 0], models):
        s = ctx.scores[m]["swap"]
        for i, l in enumerate(ORDER):
            k = B & (lab == l) & ~np.isnan(s)
            if k.any():
                ax.scatter(s[k], i + rng.uniform(-0.2, 0.2, k.sum()), **_style(l))
                ax.plot([np.median(s[k])] * 2, [i - 0.35, i + 0.35], color="k", lw=1.2)
        ax.set_yticks(range(4)); ax.set_yticklabels(ORDER, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("probe score (Half-A probe, Half-B sentences)", fontsize=9)
        _tag(ax, NAMES[m])
        ax.texts[-1].set_position((0.98, 0.98)); ax.texts[-1].set_ha("right")
    fig.tight_layout()
    _save(fig, "fig6_e6_predictions")


# ---------------------------------------------------------------- Fig 7
def fig7(ctx, models, mask):
    from itertools import combinations

    pairs = list(combinations(models, 2))
    if not pairs:
        return
    lab = ctx.nat["label"].to_numpy()
    fig, axes = plt.subplots(len(pairs), 1, figsize=(W, 2.6 * len(pairs)), squeeze=False)
    for ax, (a, b) in zip(axes[:, 0], pairs):
        sa, sb = ctx.scores[a]["swap"], ctx.scores[b]["swap"]
        for l in ("O", "A-gen"):
            k = mask & (lab == l)
            ax.scatter(sa[k], sb[k], **_style(l))
        rho = stats.spearmanr(sa[mask], sb[mask])[0]
        ax.text(0.02, 0.98, f"ρ = {rho:.2f}", transform=ax.transAxes, va="top", fontsize=9)
        ax.set_xlabel(NAMES[a], fontsize=9); ax.set_ylabel(NAMES[b], fontsize=9)
    fig.tight_layout()
    _legend(fig, ["O", "A-gen"])
    _save(fig, "fig7_e7_agreement")


# ---------------------------------------------------------------- Fig 8
def fig8(scores, y, null, auc):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 4.2))
    a1.scatter(scores[y == 0], np.random.default_rng(0).uniform(-.2, .2, (y == 0).sum()), **_style("O"))
    a1.scatter(scores[y == 1], 1 + np.random.default_rng(1).uniform(-.2, .2, (y == 1).sum()), **_style("A-gen"))
    a1.set_yticks([0, 1]); a1.set_yticklabels(["O", "A-gen"])
    a1.set_xlabel("probe score (English-trained), Romanian sentences", fontsize=9)
    a2.hist(null, bins=30, color="#bbbbbb")
    a2.axvline(auc, color="k", lw=1.2)
    a2.axvline(np.percentile(null, 95), color="k", ls="--", lw=0.8)
    a2.set_xlabel("AUC (null: shuffled English labels)", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig8_e8_romanian")
