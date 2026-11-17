"""results/summary.md, written in ladder order. The only permitted claim:
the embedding carries a mark of the mode of use."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .common import PERMITTED_PHRASE, config, rdir

NAMES = {"bert": "BERT", "decoder": "Decoder", "decoder_final": "Decoder, final-token readout (supplementary)",
         "mpnet": "mpnet", "multilingual": "mpnet-multilingual"}


def _f(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, (float, np.floating)):
        return "n/a" if np.isnan(x) else f"{x:.{nd}f}"
    return str(x)


def _p(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.2g}" if x >= 1e-4 else "<1e-4"


def _table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(_f(c) if not isinstance(c, str) else c for c in r) + " |" for r in rows]
    return "\n".join(out)


def _verdict_lines(verdicts: dict, models) -> str:
    return "\n".join(f"- {NAMES.get(m, m)}: **{verdicts[m]}**" for m in models if m in verdicts)


def write(ctx, res: dict, meta: dict) -> str:
    V = config()["verdicts"]
    models = ctx.models
    primary = [m for m in config()["primary_models"] if m in models]
    L = []
    stops: list[str] = []

    # ---------------------------------------------------------------- first lines
    e2v = res["e2"]["verdicts"]
    failed = [m for m in primary if e2v[m].startswith("not supported")]
    if failed:
        L.append(f"**E2 fails for {', '.join(NAMES[m] for m in failed)}: a probe trained on Half A "
                 "does not transfer to unseen words above the permutation null. For "
                 f"{'this model' if len(failed) == 1 else 'these models'} the argument stops at rung 2.**\n")
    if meta.get("tag"):
        L.append(f"> **{meta['tag'].upper()} RUN.** {meta.get('tag_note', '')}\n")

    L.append("# Mode-of-use probe: summary\n")
    L.append(f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}. The claim tested is that "
             f"{PERMITTED_PHRASE}: the text sediment carries a mark of the mode of use, and models "
             "trained on it pick that mark up. Nothing below says that a model has the distinction "
             "or understands anything.\n")
    L.append("**Data.** " + meta["data_line"] + "\n")
    L.append("**Labels.** " + meta["label_line"] + "\n")
    sel = [[NAMES.get(m, m), ctx.sel[m]["layer"], ctx.sel[m]["C"]] for m in models]
    L.append("**Choices made on Half A only** (layer and L2 strength by leave-one-word-out AUC "
             "inside Half A; Half B was not read):\n\n" + _table(["model", "layer", "C"], sel) + "\n")
    gcfg = config()["gold"]
    L.append("Every number is given twice: *full* = model labels on all sentences; *gold* = the same "
             "probe scores evaluated only on the gold sentences with the gold labels. Probes are "
             f"always trained on model labels. Gold labeler: {gcfg['labeler']}.\n")
    if not gcfg.get("human", True):
        stops.append("No human gold set: the gold labels come from a second model "
                     f"({gcfg['labeler']}), so κ measures agreement between two models, not "
                     "between a model and a person.")

    # ---------------------------------------------------------------- E1
    t = res["e1"]["table"]
    L.append("## 1. Prominence (E1)\n")
    L.append("*Question.* Without labels, is the mode one of the main directions along which a "
             "word's contextual embeddings vary?\n")
    rows = []
    for m in models:
        p = t[(t.model == m) & (t.word == "(pooled)")]
        w = t[(t.model == m) & (t.word != "(pooled)")]
        ranks = w["rank_first_aligned"]
        rows.append([NAMES.get(m, m),
                     _f(p["rank_first_aligned"].iloc[0], 0) if len(p) else "n/a",
                     _f(p["ev_share"].iloc[0]) if len(p) else "n/a",
                     _f(p["best_auc"].iloc[0]) if len(p) else "n/a",
                     f"{int((ranks <= V['e1_supported_rank']).sum())}/{len(w)}",
                     f"{int(ranks.isna().sum())}/{len(w)}",
                     _f(p["gold_rank_first_aligned"].iloc[0], 0) if len(p) else "n/a"])
    L.append(f"Rank = first principal component whose score separates O from A-gen with AUC ≥ "
             f"{V['e1_aligned_auc']} (sign-free). Pooled = all words, each centred on its own mean.\n")
    L.append(_table(["model", "pooled rank", "its variance share", "best pooled PC AUC",
                     f"words with rank ≤ {V['e1_supported_rank']}", "words with none in top 10",
                     "gold pooled rank"], rows) + "\n")
    L.append("*Verdict* (supported: pooled rank ≤ "
             f"{V['e1_supported_rank']}; mixed: within the first 10; not supported: none):\n\n"
             + _verdict_lines(res["e1"]["verdicts"], models) + "\n")
    L.append("*What it does not show.* A mode-aligned component can be prominent because mode "
             "co-varies with something larger (genre, sentence type); rungs 3–5 test that. A "
             "non-prominent mode can still be linearly readable (rung 2).\n")
    for m in models:
        if not res["e1"]["verdicts"][m].startswith("supported"):
            stops.append(f"E1 ({NAMES.get(m, m)}): {res['e1']['verdicts'][m]} — the mode is not among "
                         "the top three directions of variation.")

    # ---------------------------------------------------------------- E2
    p = res["e2"]["pooled"]
    L.append("## 2. Shared direction (E2)\n")
    L.append("*Question.* With labels, does a linear probe trained on some words transfer to "
             "words it never saw?\n")
    rows = []
    for m in models:
        r = p[p.model == m].iloc[0]
        rows.append([NAMES.get(m, m),
                     f"{_f(r['A->B auc'])} [{_f(r['A->B ci_lo'], 2)}, {_f(r['A->B ci_hi'], 2)}]",
                     _f(r["A->B null95"]),
                     f"{_f(r['B->A auc'])} [{_f(r['B->A ci_lo'], 2)}, {_f(r['B->A ci_hi'], 2)}]",
                     _f(r["B->A null95"]),
                     _f(r["LOWO median word auc"]),
                     f"{int(r['LOWO words above null95'])}/20",
                     f"{_f(r['A->B gold_auc'])} / {_f(r['B->A gold_auc'])} (n={int(r['A->B gold_n'])}/{int(r['B->A gold_n'])})",
                     f"{_f(r['LOWO gold auc'])} (n={int(r['LOWO gold n'])})"])
    L.append(_table(["model", "A→B AUC [95% CI]", "null 95th", "B→A AUC [95% CI]", "null 95th",
                     "LOWO median word AUC", "words above own null", "gold A→B / B→A",
                     "gold LOWO pooled"], rows) + "\n")
    L.append(f"Null: training labels shuffled within word, {config()['probe']['n_permutations']} "
             "permutations. Per-word AUCs: `results/e2_transfer_by_word.csv`.\n")
    L.append("*Verdict* (supported: both directions above the null 95th percentile and median "
             f"held-out-word AUC ≥ {V['e2_supported_auc']}; mixed: above the null but weaker):\n\n"
             + _verdict_lines(e2v, models) + "\n")
    rob = res.get("robust")
    if rob is not None and len(rob):
        def _r(m, lab):
            r = rob[(rob.model == m) & (rob.labels == lab)]
            return r.iloc[0] if len(r) else None
        rr = []
        for m in models:
            for lab, name in (("claude", "Claude (main)"), ("qwen", "Qwen"), ("agreed", "both agree")):
                r = _r(m, lab)
                if r is not None:
                    rr.append([NAMES.get(m, m), name, str(int(r["n_O"])), str(int(r["n_Agen"])),
                               _f(r["A->B auc"]), _f(r["B->A auc"]), _f(r["LOWO median word auc"]),
                               _f(r["main probe auc"])])
        L.append("**Does it depend on the labeler?** The same probes (same layer and L2 strength) "
                 "were refitted and retested on the cross-family labels, and on only the sentences "
                 "both labelers put in the same class. The last column scores the main probes, "
                 "trained on Claude's labels, against each label set.\n")
        L.append(_table(["model", "labels", "n O", "n A-gen", "A→B AUC", "B→A AUC",
                         "LOWO median word AUC", "main probe vs these labels"], rr) + "\n")
        L.append("The local model calls fewer sentences A-gen, keeping the clearer cases, and the "
                 "agreed subset drops the cases the two labelers read differently, so higher AUCs "
                 "there are expected. What the table shows is that the transfer does not hinge on one "
                 "labeler's reading of the borderline sentences. Both labelers are language models; "
                 "this does not replace human labels.\n")
    L.append("*What it does not show.* A shared direction may be shared because a shared confound "
             "(genre, grammar, concreteness) carries it. Rungs 3–5 test each.\n")
    for m in models:
        if not e2v[m].startswith("supported"):
            stops.append(f"E2 ({NAMES.get(m, m)}): {e2v[m]}.")

    # ---------------------------------------------------------------- E3
    t3 = res["e3"]["table"]
    L.append("## 3. Not genre (E3)\n")
    L.append("*Question.* Does the probe transfer from encyclopedic text to fiction and back?\n")
    bal = res["e3"]["balance"]
    L.append("Class balance (model labels): " + "; ".join(
        f"{g}: " + ", ".join(f"{c} {int(bal.loc[g, c])}" for c in bal.columns) for g in bal.index) + ".\n")
    rows = []
    for m in models:
        r = t3[t3.model == m].iloc[0]
        rows.append([NAMES.get(m, m), _f(r["wiki->fiction auc"]), _f(r["fiction->wiki auc"]),
                     _f(r["within-wiki auc"]), _f(r["within-fiction auc"]),
                     _f(r["within-wiki cross-word auc"]), _f(r["within-fiction cross-word auc"]),
                     _f(r["E2 A<->B mean auc"]),
                     f"{_f(r['wiki->fiction gold_auc'])} / {_f(r['fiction->wiki gold_auc'])}"])
    L.append(_table(["model", "Wiki→Fiction", "Fiction→Wiki", "within Wiki (CV by source)",
                     "within Fiction (CV by source)", "within Wiki, A↔B words",
                     "within Fiction, A↔B words", "E2 A↔B (both genres)", "gold W→F / F→W"],
                    rows) + "\n")
    L.append(f"*Verdict* (supported: both cross-genre AUCs ≥ {V['e3_supported_auc']} and each at most "
             f"{V['e3_max_drop']} below the within-genre baseline):\n\n"
             + _verdict_lines(res["e3"]["verdicts"], models) + "\n")
    L.append("*What it does not show.* Cross-genre training uses all twenty words, so it is not also "
             "a cross-word test. Wikipedia and fiction differ in their O/A-gen balance; AUC is "
             "insensitive to that, the fitted probe is not. Legacy Wikipedia sentences have no "
             "article id, so the within-Wikipedia baseline may share articles across folds.\n")
    for m in models:
        if not res["e3"]["verdicts"][m].startswith("supported"):
            stops.append(f"E3 ({NAMES.get(m, m)}): {res['e3']['verdicts'][m]}.")

    # ---------------------------------------------------------------- E4
    t4 = res["e4"]["table"]
    L.append("## 4. Not grammar (E4)\n")
    L.append("*Question.* Does the probe beat a surface-grammar classifier, and does it survive "
             "minimal pairs with identical surface form?\n")
    gw = [m for m in primary if res["e4"]["grammar_wins"].get(m)]
    if gw:
        L.append(f"**For {', '.join(NAMES[m] for m in gw)} the embedding probe does not beat the "
                 "grammar classifier on held-out words: the mark is carried mainly by grammar "
                 "(determiner, number, tense, pronouns, length, position), and this changes the paper.**\n")
    rows = []
    for m in models:
        r = t4[t4.model == m].iloc[0]
        rows.append([NAMES.get(m, m), _f(r["grammar LOWO auc"]), _f(r["embedding LOWO auc"]),
                     f"{int(r['words embedding > grammar'])}/{int(r['n words'])}",
                     _p(r["wilcoxon p"]), f"{_f(r['LRT stat (LOWO)'], 1)} (p {_p(r['LRT p (LOWO)'])})",
                     _f(r["embedding residual auc (LOWO)"]),
                     f"{_f(r['gold grammar auc'])} / {_f(r['gold embedding auc'])}"])
    L.append("(a) Held-out-word (LOWO) scores. LRT: does the embedding score add to the grammar "
             "score in a logistic model of the label (df = 1)?\n")
    L.append(_table(["model", "grammar AUC", "embedding AUC", "words emb > gram", "Wilcoxon p",
                     "LRT χ² (p)", "embedding AUC on residual", "gold gram / emb"], rows) + "\n")
    rows = []
    for m in models:
        r = t4[t4.model == m].iloc[0]
        rows.append([NAMES.get(m, m), f"{int(r['pairs positive'])}/{int(r['pairs n'])}",
                     _f(r["pairs fraction positive"]), str(int(r["pairs ties"])),
                     _p(r["pairs binomial p"]), _f(r["pairs fraction positive (A-gen)"]),
                     _f(r["pairs fraction positive (A-def)"])])
    L.append("(b) Minimal pairs (Half-A probe on Half-B pairs and back; score = probe(A) − probe(O); "
             "ties count as failures):\n")
    L.append(_table(["model", "positive", "fraction", "ties", "binomial p (>0.5)",
                     "fraction, A-gen pairs", "fraction, A-def pairs"], rows) + "\n")
    L.append(f"*Verdict* (supported: LRT p < {V['e4_alpha']}, embedding AUC above grammar AUC, and "
             f"minimal pairs binomial p < {V['e4_alpha']}; mixed: one of the two):\n\n"
             + _verdict_lines(res["e4"]["verdicts"], models) + "\n")
    L.append("*What it does not show.* The grammar classifier uses only the listed surface features; "
             "a richer syntax model could absorb more. The minimal pairs are written, not natural. "
             "For the decoder's target-token readout both members of a pair share the prefix up to "
             "the target, so its states are identical and every pair is a tie: (b) cannot test that "
             "readout, and the supplementary final-token readout is reported for that reason.\n")
    for m in models:
        if not res["e4"]["verdicts"][m].startswith("supported"):
            stops.append(f"E4 ({NAMES.get(m, m)}): {res['e4']['verdicts'][m]}.")

    # ---------------------------------------------------------------- E5
    L.append("## 5. Not concreteness (E5)\n")
    L.append("*Question.* Does the probe survive regressing out word concreteness?\n")
    if "table" not in res["e5"]:
        L.append(f"Not run: {res['e5'].get('skipped')}.\n")
        stops.append("E5: not run (no concreteness norms).")
    else:
        t5 = res["e5"]["table"]
        rows = [[NAMES.get(r.model, r.model), _f(r["spearman score~concreteness"]),
                 _f(r["concreteness alone auc"]), _f(r["auc before"]),
                 _f(r["auc after (score residualized)"]), _f(r["auc after (features residualized)"]),
                 _f(r["drop"]), f"{_f(r['gold auc before'])} → {_f(r['gold auc after (features)'])}"]
                for _, r in t5.iterrows()]
        L.append("Mean Brysbaert concreteness of the content words, target excluded. LOWO scores.\n")
        L.append(_table(["model", "Spearman ρ (score, concreteness)", "concreteness alone AUC",
                         "AUC before", "after (score resid.)", "after (feature resid.)", "drop",
                         "gold before → after"], rows) + "\n")
        L.append(f"*Verdict* (supported: feature-residualized AUC drops by ≤ {V['e5_max_drop']}; "
                 f"mixed: ≤ {V['e5_mixed_drop']}):\n\n" + _verdict_lines(res["e5"]["verdicts"], models) + "\n")
        for m in models:
            if not res["e5"]["verdicts"][m].startswith("supported"):
                stops.append(f"E5 ({NAMES.get(m, m)}): {res['e5']['verdicts'][m]}.")
    L.append("*What it does not show.* Only a linear effect of mean context concreteness is removed; "
             "concreteness of particular words, or non-linear effects, could remain.\n")

    # ---------------------------------------------------------------- E6
    t6 = res["e6"]["table"]
    L.append("## 6. Predictions (E6)\n")
    L.append("*Question.* Do definitions land at the far abstract end, and where do idioms land?\n")
    rows = []
    for _, r in t6.iterrows():
        rows.append([NAMES.get(r.model, r.model), r["labels"], str(int(r.get("n A-def", 0) or 0)),
                     _f(r.get("A-def fraction beyond A-gen median")),
                     _p(r.get("A-def MW vs A-gen p (greater)")),
                     str(int(r.get("n A-idiom", 0) or 0)),
                     _f(r.get("A-idiom fraction beyond A-gen median")),
                     _p(r.get("A-idiom MW vs A-gen p (two-sided)")),
                     _p(r.get("A-idiom MW vs O p (two-sided)")),
                     str(r.get("idiom placement", "n/a"))])
    L.append("Probe trained on O vs A-gen only (LOWO scores). Definitions and idioms are reported "
             "separately and never pooled.\n")
    L.append(_table(["model", "labels", "n A-def", "A-def beyond A-gen median",
                     "MW A-def > A-gen p", "n A-idiom", "idiom beyond A-gen median",
                     "MW idiom vs A-gen p", "MW idiom vs O p", "idiom placement"], rows) + "\n")
    L.append("*Verdict* (definitions supported: Mann–Whitney p < "
             f"{V['e6_alpha']} and more than half beyond the A-gen median):\n\n" + "\n".join(
                 f"- {NAMES.get(m, m)}: definitions **{v['definitions']}**; idioms **{v['idioms']}**"
                 for m, v in res["e6"]["verdicts"].items()) + "\n")
    L.append("*What it does not show.* Natural definitions are mostly Wikipedia lead sentences, so "
             "their position also reflects that register. Idioms are rare in this sample.\n")
    for m, v in res["e6"]["verdicts"].items():
        if not v["definitions"].startswith("supported"):
            stops.append(f"E6 definitions ({NAMES.get(m, m)}): {v['definitions']}.")
        if "underpowered" in v["idioms"] or "not tested" in v["idioms"]:
            stops.append(f"E6 idioms ({NAMES.get(m, m)}): {v['idioms']}.")

    # ---------------------------------------------------------------- E7
    t7 = res["e7"]["table"]
    L.append("## 7. Same direction across models (E7)\n")
    L.append("*Question.* Do an encoder, a sentence model and a decoder order the same sentences "
             "the same way?\n")
    if len(t7):
        rows = [[r.pair, _f(r.spearman), _f(r["spearman within O"]), _f(r["spearman within A-gen"]),
                 f"{_f(r['gold spearman'])} (n={int(r['gold n'])})"] for _, r in t7.iterrows()]
        L.append("Half-B sentences, scores from the Half-A probe of each model.\n")
        L.append(_table(["pair", "Spearman ρ", "within O", "within A-gen", "gold ρ"], rows) + "\n")
    L.append(f"*Verdict* (supported: every pair ρ ≥ {V['e7_supported_rho']}; mixed: ≥ "
             f"{V['e7_mixed_rho']}): **{res['e7']['overall']}**\n")
    L.append("*What it does not show.* Agreement on the overall ordering is driven largely by the "
             "O/A-gen split itself; the within-label correlations show whether the models also agree "
             "inside a class.\n")
    if not res["e7"]["overall"].startswith("supported"):
        stops.append(f"E7: {res['e7']['overall']}.")

    # ---------------------------------------------------------------- E8
    L.append("## 8. Not English syntax (E8, optional)\n")
    if "e8" in res:
        r = res["e8"]["table"].iloc[0]
        L.append(f"English-trained probe on {int(r['n'])} Romanian sentences ({int(r['n words'])} words): "
                 f"AUC {_f(r['auc'])} [{_f(r['ci_lo'], 2)}, {_f(r['ci_hi'], 2)}], null 95th "
                 f"{_f(r['null95'])}. Verdict: **{res['e8']['verdicts']['multilingual']}**.\n")
        L.append("*What it does not show.* One language pair, one multilingual sentence model.\n")
        if not res["e8"]["verdicts"]["multilingual"].startswith("supported"):
            stops.append(f"E8: {res['e8']['verdicts']['multilingual']}.")
    else:
        L.append("Not run (no `data/romanian_labeled.csv`, or `--romanian` not set).\n")

    # ---------------------------------------------------------------- stops
    L.append("## Where the argument stops\n")
    L.append("\n".join(f"- {s}" for s in stops) if stops else
             "- Every rung was supported for every model tested.")
    L.append("")
    text = "\n".join(L)
    (rdir() / "summary.md").write_text(text)
    write_verdicts(ctx, res, meta, stops)
    write_sections(text)
    return text


def write_sections(text: str) -> None:
    """Split summary.md at its '## ' headings so the website embeds the same text."""
    import json
    import re

    parts = re.split(r"(?m)^## ", text)
    out = {"intro": parts[0].strip()}
    for part in parts[1:]:
        head, _, body = part.partition("\n")
        m = re.match(r"(\d)\.", head)
        key = f"e{m.group(1)}" if m else "stops"
        out[key] = {"heading": head.strip(), "body": body.strip()}
    (rdir() / "sections.json").write_text(json.dumps(out, indent=2))


RUNGS = [("e1", "Prominence"), ("e2", "Shared direction"), ("e3", "Not genre"),
         ("e4", "Not grammar"), ("e5", "Not concreteness"), ("e6", "Predictions"),
         ("e7", "Same direction across models"), ("e8", "Not English syntax")]


def write_verdicts(ctx, res, meta, stops) -> None:
    """Machine-readable verdicts for the website; mirrors summary.md."""
    import json

    rungs = []
    for key, title in RUNGS:
        r = res.get(key)
        entry = {"id": key, "title": title, "run": r is not None and "skipped" not in (r or {})}
        if entry["run"]:
            if key == "e7":
                entry["verdicts"] = {"all": r["overall"]}
                entry["pairs"] = {k: float(v) for k, v in r["verdicts"].items()}
            elif key == "e6":
                entry["verdicts"] = {m: f"definitions: {v['definitions']}; idioms: {v['idioms']}"
                                     for m, v in r["verdicts"].items()}
                entry["detail"] = r["verdicts"]
            else:
                entry["verdicts"] = dict(r["verdicts"])
        if key == "e4" and r:
            entry["grammar_wins"] = {m: bool(v) for m, v in r["grammar_wins"].items()}
        rungs.append(entry)
    out = {"tag": meta.get("tag"), "generated": dt.datetime.now().isoformat(timespec="seconds"),
           "models": ctx.models, "primary": config()["primary_models"],
           "selected": {m: {"layer": str(v["layer"]), "C": float(v["C"])} for m, v in ctx.sel.items()},
           "data_line": meta["data_line"], "label_line": meta["label_line"],
           "gold": config()["gold"], "rungs": rungs, "stops": stops}
    (rdir() / "verdicts.json").write_text(json.dumps(out, indent=2, default=str))
