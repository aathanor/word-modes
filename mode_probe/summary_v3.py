"""results/summary_further.md: the tests added after the first results (E9, quadruples, E10, E6
rerun, E11 rod/cap, UDS-G transfer). Pre-registered parts say so; the rest are marked as added
after the first results. Reads whatever result files exist.

python -m mode_probe.summary_v3
"""
from __future__ import annotations

import json

import pandas as pd

from .common import RESULTS

V3 = RESULTS / "v3"
NAMES = {"bert": "BERT", "decoder": "Decoder", "mpnet": "mpnet", "decoder (at the word)": "Decoder, at the word"}


def _f(x, nd=2):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def write() -> str:
    L = ["# Further tests\n",
         "These tests were added after the first results, to separate this experiment's distinction "
         "from the linguistic one between episodic and generic sentences, and to test the explanation "
         "of the mark offered on the home page. Their criterion, measures and predictions were "
         "committed to the repository before their data existed "
         "([PREREGISTRATION_v3.md](https://github.com/aathanor/word-modes/blob/main/mode_probe/PREREGISTRATION_v3.md)), "
         "except the UDS-G transfer, which is reported without verdicts.\n"]
    kp = V3 / "kappa_v3.json"
    if kp.exists():
        k = json.loads(kp.read_text())
        line = (f"**Labels (version 3).** Claude Opus 5.5 labeled every sentence under "
                f"[LABELING_v3.md](https://github.com/aathanor/word-modes/blob/main/mode_probe/LABELING_v3.md); "
                f"the 300-sentence sample was relabeled in separate requests. Gate: κ = {k['kappa']:.2f} "
                f"(n = {k['n']}, eight labels; required 0.7) — {'passed' if k['passed'] else 'failed'}. Both "
                f"labelings come from the same model with the same instruction, so this measures how "
                f"consistently it applies the criterion, not agreement between independent judges.")
        if "qwen_kappa" in k:
            line += f" Cross-check, Qwen3.8 27B against the sample: κ = {k['qwen_kappa']:.2f}."
        L.append(line + "\n")
        if not k["passed"]:
            L.append("The gate failed, so E9, E10 and the E6 rerun were not run.\n")

    p = V3 / "e9_reference_occasion.csv"
    if p.exists():
        t = pd.read_csv(p)
        L.append("## Reference or occasion? (E9)\n")
        L.append("The object/general difference could follow **reference** (a particular thing or the "
                 "kind; the PCF reading) or **occasion** (one episode or what holds in general; the "
                 "reading from the linguistics of genericity). The two agree on most sentences and "
                 "disagree on two kinds: habits and standing properties of a particular thing "
                 "(\"My cat always sleeps by the fire\", P-gen) and episodes in the history of a kind "
                 "(\"Cats reached Australia in the seventeenth century\", K-ep). A probe trained only on "
                 "the cells both readings agree on (P-ep against K-gen) places the two disputed cells "
                 "on a scale from 0 (with particular episodes) to 1 (with general statements about "
                 "the kind), on words it never saw.\n")
        L.append(_table(["model", "P-gen position [95% CI]", "K-ep position [95% CI]", "n P-gen / K-ep", "verdict"],
                        [[NAMES.get(r.model, r.model),
                          f"{_f(r['t(P-gen)'])} [{_f(r['t(P-gen) lo'])}, {_f(r['t(P-gen) hi'])}]",
                          f"{_f(r['t(K-ep)'])} [{_f(r['t(K-ep) lo'])}, {_f(r['t(K-ep) hi'])}]",
                          f"{int(r['n P-gen'])} / {int(r['n K-ep'])}", f"**{r.verdict}**"] for _, r in t.iterrows()]) + "\n")
        L.append("Reference predicts P-gen near 0 and K-ep near 1; occasion predicts the reverse. "
                 "BERT and the decoder decide; mpnet is reported.\n")
        b = t.set_index("model")
        if {"bert", "decoder"} <= set(b.index):
            L.append("**What this shows.** Neither reading fits alone. For both primary readouts the "
                     "disputed cells sit between the poles, so the direction responds to reference and "
                     f"to occasion. BERT leans toward reference: the episodes about kinds sit near the "
                     f"general statements ({_f(b.loc['bert', 't(K-ep)'])}), and the habits of particular "
                     f"things sit nearer the particular episodes ({_f(b.loc['bert', 't(P-gen)'])}), though "
                     "not as near as the reference reading predicted; that combination was not one of the "
                     "pre-registered cases, so it gets no verdict. The decoder weighs the two about "
                     f"equally ({_f(b.loc['decoder', 't(P-gen)'])} and {_f(b.loc['decoder', 't(K-ep)'])}). "
                     "The claim that the mark is PCF's distinction rather than genericity is not "
                     "supported as stated; the mark carries both.\n")
        q = V3 / "e9_quads.csv"
        if q.exists():
            tq = pd.read_csv(q)
            vj = json.loads((V3 / "verdicts_v3.json").read_text())
            L.append("**Minimal quadruples.** Five sets of four written sentences per word, differing "
                     "only in reference and occasion. Within each set, the reference effect is how far the "
                     "kind sentences sit from the particular ones; the occasion effect, how far the general "
                     "sentences sit from the episodes.\n")
            L.append(_table(["model", "sets where reference > occasion", "mean reference effect",
                             "mean occasion effect", "verdict"],
                            [[NAMES.get(r.model, r.model), f"{int(r['reference > occasion'])} / {int(r['n quads'])}",
                              _f(r["mean reference effect"]), _f(r["mean occasion effect"]), f"**{r.verdict}**"]
                             for _, r in tq.iterrows()]) + "\n")
            L.append(f"The labeler put {vj['quads_label_agreement']:.0%} of the written sentences in the "
                     "cell they were written for. The quadruples were written by the same model that wrote "
                     "the code.\n")
        s = V3 / "e9_secondary.csv"
        if s.exists():
            ts = pd.read_csv(s)
            cos = ts[ts.probe.str.startswith("cos")]
            if len(cos):
                L.append("**Which way does the original direction point?** Cosine between the E2 direction "
                         "(object against general, version-2 labels) and a direction trained on reference "
                         "alone or on occasion alone:\n")
                piv = cos.pivot(index="model", columns="probe", values="value")
                L.append(_table(["model", "with reference", "with occasion"],
                                [[NAMES.get(m, m), _f(r.get("cos(reference, E2)")), _f(r.get("cos(occasion, E2)"))]
                                 for m, r in piv.iterrows()]) + "\n")

    p = V3 / "e10_encounter.csv"
    if p.exists():
        t = pd.read_csv(p)
        L.append("## Writing from encounter (E10)\n")
        L.append("The home page explains the mark as the way people write when they write from an "
                 "encounter. If so, particular sentences in diaries, letters and travel accounts should "
                 "carry it more strongly than particular sentences in novels or in an encyclopedia. The "
                 "measure is how well the probe (trained on Wikipedia and fiction) separates particular "
                 "episodes from general statements within each source, on unseen words (AUC).\n")
        L.append(_table(["model", "Wikipedia", "fiction", "encounter writing", "encounter − best other [95% CI]", "verdict"],
                        [[NAMES.get(r.model, r.model), _f(r["auc wikipedia"], 3), _f(r["auc fiction"], 3),
                          _f(r["auc encounter"], 3), f"{_f(r['encounter - max(other)'], 3)} [{_f(r['lo'], 3)}, {_f(r['hi'], 3)}]",
                          f"**{r.verdict}**"] for _, r in t.iterrows()]) + "\n")
        if (t["hi"] < 0).all():
            L.append("**What this shows.** In all three models the mark is *less* sharp in writing from "
                     "encounter than in the novels and the encyclopedia, by 0.03–0.04, and the intervals "
                     "exclude zero. That is the opposite of what the encounter explanation predicts. The "
                     "pre-registered rule planned for \"stands out\" and for \"the same\", not for "
                     "\"weaker\", so its verdict is \"inconclusive\"; the data do not support the "
                     "encounter explanation. One reason may be that diaries and letters mix reports of "
                     "encounters with reflection, and that the probe was trained on the other two genres.\n")

    p = V3 / "e6_rerun.csv"
    if p.exists():
        t = pd.read_csv(p)
        L.append("## Definitions, with definitions collected on purpose (E6 rerun)\n")
        L.append("Definitions were too few to test (11). Wikipedia sentences of the form \"A spoon is "
                 "a …\" were collected on purpose and labeled with the rest; the original rule is "
                 "unchanged: definitions should sit beyond the general statements on a direction "
                 "that never saw them.\n")
        L.append("Collecting on purpose did not help much: Wikipedia sentences that state what a "
                 "concrete noun is are rare outside each article's first line, and only 8 of the "
                 "candidates were definitions.\n")
        L.append(_table(["model", "definitions", "beyond the general-statement median", "Mann–Whitney p", "verdict"],
                        [[NAMES.get(r.model, r.model), int(r["n A-def"]), _f(r["fraction beyond A-gen median"]),
                          _f(r["MW p (greater)"], 4), f"**{r.verdict}**"] for _, r in t.iterrows()]) + "\n")

    p = RESULTS / "e11_rod_cap.csv"
    if p.exists():
        t = pd.read_csv(p)
        L.append("## Rod and cap (E11)\n")
        L.append("The 2025 study read object uses as a tight \"rod\" and general uses as a wide "
                 "\"cap\". Here the spread of each class is measured after correcting for the "
                 "embeddings' uneven use of directions, and within groups of sentences on the same "
                 "topic (found with the noun masked), because general sentences come from more varied "
                 "topics, which alone would spread them. A ratio above 1 means general uses spread "
                 "wider.\n")
        L.append(_table(["model", "words where general uses spread wider", "median ratio, raw",
                         "median ratio, same topic", "verdict"],
                        [[NAMES.get(r.model, r.model), f"{int(r['words with A-gen wider'])} / {int(r['words'])}",
                          _f(r["median raw ratio"], 3), _f(r["median topic-controlled ratio"], 3), f"**{r.verdict}**"]
                         for _, r in t.iterrows()]) + "\n")

    p = RESULTS / "uds_transfer.csv"
    if p.exists():
        t = pd.read_csv(p)
        L.append("## Against an independent annotation of genericity (UDS-G)\n")
        L.append("Govindarajan, Van Durme and White (2019) had people rate every noun in an English "
                 "treebank for how particular and how kind-like it is, and every verb for how episodic. "
                 "Their annotators knew nothing of this experiment. Probes trained on one data set are "
                 "tested on the other, reading the noun itself. Added after the first results; no "
                 "verdicts.\n")
        L.append(_table(["model", "their particular/kind probe on our object/general sentences",
                         "our probe on their particular vs kind nouns", "our probe on their episodic vs general verbs",
                         "cosine of the two directions"],
                        [[NAMES.get(r.model, r.model), _f(r["UDS->here AUC (O vs A-gen)"], 3),
                          _f(r["here->UDS AUC (particular vs kind)"], 3),
                          _f(r["here->UDS AUC (episodic vs general predicate)"], 3),
                          _f(r["cosine(UDS reference direction, this direction)"], 2)] for _, r in t.iterrows()]) + "\n")
    text = "\n".join(L)
    (RESULTS / "summary_further.md").write_text(text)
    return text


if __name__ == "__main__":
    print(write())
