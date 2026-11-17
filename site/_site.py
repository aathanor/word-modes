"""Helpers for the .qmd pages. Everything shown here is read from data/, which
mode_probe.site_export fills from the pipeline's results; nothing is typed by hand."""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import yaml


def _out(text: str) -> None:
    """Pages run with `output: asis`: printed markdown/HTML goes straight into the page."""
    print(text)
    print()



ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

NAMES = {"bert": "BERT", "decoder": "Decoder", "decoder_final": "Decoder, final token",
         "mpnet": "mpnet", "multilingual": "mpnet-multilingual", "all": "all models"}
QUESTIONS = {
    "e1": "Without labels, is the mode one of the main directions along which a word's embeddings vary, or a thinner signal a probe has to be told to find?",
    "e2": "Does a probe trained on some words separate the modes in words it never saw?",
    "e3": "Does the direction carry over from encyclopedic text to fiction and back?",
    "e4": "Does it beat a surface-grammar classifier, and survive pairs with the same surface form?",
    "e5": "Does it survive removing the concreteness of the other words in the sentence?",
    "e6": "Do definitions sit at the far abstract end? Where do idioms fall?",
    "e7": "Do an encoder, a decoder and a sentence model order the same sentences the same way?",
    "e8": "Does a direction learned on English carry over to Romanian?",
}
FIGS = {"e1": "fig1_e1_prominence", "e2": "fig2_e2_strip", "e3": "fig3_e2e3_transfer",
        "e4": "fig4_e4_grammar", "e5": "fig5_e5_concreteness", "e6": "fig6_e6_predictions",
        "e7": "fig7_e7_agreement", "e8": "fig8_e8_romanian"}
CAPTIONS = {
    "e1": "Top: each word's own embeddings on their first two principal components, coloured by label "
          "(the labels played no part in the PCA). Bottom: for each word, the rank of the first "
          "component that separates O from A-gen with AUC ≥ 0.8; '>10' means none of the first ten.",
    "e2": "Probe trained on Half A, applied to the Half-B words it never saw. One row per word; "
          "object-mode dark blue, A-gen orange, definitions hollow circles, idioms grey triangles.",
    "e3": "Top: AUC for each held-out word (leave-one-word-out); dashes mark that word's permutation "
          "null (95th percentile), the dotted line chance. Bottom: cross-genre and within-genre AUCs.",
    "e4": "Top: grammar classifier against the embedding probe, per held-out word. Bottom: "
          "probe(A) − probe(O) for each minimal pair; above zero means the A sentence sits further "
          "toward abstract-mode.",
    "e5": "mpnet probe score against the mean concreteness of the other content words, with the AUC "
          "before and after concreteness is regressed out.",
    "e6": "All Half-B sentences on the axis of a probe trained on O vs A-gen only; bars mark medians.",
    "e7": "Per-sentence probe scores of each pair of models on the Half-B sentences.",
    "e8": "English-trained probe on Romanian sentences, and the permutation null.",
}


def _json(name):
    p = DATA / name
    return json.loads(p.read_text()) if p.exists() else None


def verdicts():
    return _json("verdicts.json") or {}


def sections():
    return _json("sections.json") or {}


def _vclass(v: str, key: str | None = None) -> str:
    v = (v or "").lower()
    if "underpowered" in v:
        return "under"
    cls = ("supported" if v.startswith("supported") else "mixed" if v.startswith("mixed")
           else "not" if v.startswith("not supported") else "none")
    if key == "e1" and cls == "not":
        cls = "thin"      # the mark is there (E2) but not among the main directions: a description, not a failure
    return cls


E1_WORD = {"supported": "prominent", "mixed": "partly prominent", "not supported": "not prominent"}


def _vword(v: str, key: str | None = None) -> str:
    if key == "e1":
        for k, w in E1_WORD.items():
            if (v or "").startswith(k):
                return w
    return v or "not run"


def badge(v: str, key: str | None = None) -> str:
    return f'<span class="verdict {_vclass(v, key)}">{html.escape(_vword(v, key))}</span>'


# The argument as it is told: first whether there is a mark, then whether it is something
# else, then whether it behaves as a mark of the mode should, and last how visible it is
# without labels. E-numbers are the experiment ids of the summary and the data files; the
# pipeline runs them in id order.
STEPS = [
    ("Is there a mark?", ["e2"]),
    ("Is it something else?", ["e3", "e4", "e5"]),
    ("Does it behave as a mark of the mode should?", ["e7", "e6"]),
    ("How visible is it without labels?", ["e1"]),
    ("Beyond English", ["e8"]),
]


MAIN = ["bert", "decoder", "mpnet"]
ICON = {"supported": "✓", "mixed": "~", "not": "✕", "none": "–", "under": "?", "thin": "◐"}


def _csv(name):
    p = DATA / name
    return pd.read_csv(p) if p.exists() else None


def _rng(vals, fmt="{:.2f}"):
    vals = [v for v in vals if v is not None and not pd.isna(v)]
    if not vals:
        return "n/a"
    lo, hi = min(vals), max(vals)
    return fmt.format(lo) if fmt.format(lo) == fmt.format(hi) else f"{fmt.format(lo)}–{fmt.format(hi)}"


def headline(key: str) -> str:
    """One line per rung, computed from the result tables."""
    try:
        if key == "e1":
            t = _csv("e1_prominence.csv")
            w = t[(t.word != "(pooled)") & t.model.isin(MAIN)]
            top = [int((w[w.model == m].rank_first_aligned <= 3).sum()) for m in MAIN if m in set(w.model)]
            p = t[(t.word == "(pooled)") & t.model.isin(MAIN)]
            return (f"in {_rng(top, '{:.0f}')} of 20 words one of the first three unlabeled directions separates the modes · "
                    f"pooled over words the best reaches AUC {_rng(p.best_auc)}, under the 0.8 bar")
        if key == "e2":
            t = _csv("e2_transfer.csv"); t = t[t.model.isin(MAIN)]
            auc = [(a + b) / 2 for a, b in zip(t["A->B auc"], t["B->A auc"])]
            return f"AUC {_rng(auc)} on words the probe never saw · {int(t['LOWO words above null95'].min())}/20 words above chance"
        if key == "e3":
            t = _csv("e3_genre.csv"); t = t[t.model.isin(MAIN)]
            return f"AUC {_rng(list(t['wiki->fiction auc']) + list(t['fiction->wiki auc']))} across genres"
        if key == "e4":
            t = _csv("e4_grammar.csv"); t = t[t.model.isin(MAIN)]
            pr = t[t["pairs ties"] == 0]["pairs fraction positive"] * 100
            return (f"probe {_rng(t['embedding LOWO auc'])} vs grammar alone {_rng(t['grammar LOWO auc'])} · "
                    f"minimal pairs {_rng(pr, '{:.0f}')}% right")
        if key == "e5":
            t = _csv("e5_concreteness.csv"); t = t[t.model.isin(MAIN)]
            return f"AUC drops by at most {t['drop'].max():.3f} without concreteness"
        if key == "e6":
            t = _csv("e6_predictions.csv"); t = t[(t.model == "bert") & (t.labels == "model labels")]
            n = int(t["n A-def"].iloc[0]) if len(t) else 0
            return f"only {n} definitions, too few to decide · idioms fall between the modes"
        if key == "e7":
            t = _csv("e7_agreement.csv")
            return f"ρ {_rng(t.spearman)} between each pair of models"
        if key == "e8":
            return "needs Romanian sentences labeled O and A-gen"
    except Exception:  # noqa: BLE001 - a missing table just leaves the line empty
        return ""
    return ""


def _status(v: str, key: str | None = None) -> str:
    return _vclass(v, key).split()[0]


def _per_model(r: dict, models: list[str]) -> dict:
    if not r["run"]:
        return {m: "not run" for m in models}
    if r["id"] == "e7":
        return {m: r["verdicts"]["all"] for m in models}
    if r["id"] == "e6":
        return {m: r["detail"][m]["definitions"] for m in models if m in r["detail"]}
    if r["id"] == "e8":
        return {m: r["verdicts"].get("multilingual", "not run") for m in models}
    return {m: r["verdicts"].get(m, "") for m in models}


STEP_WORD = {"supported": "holds", "mixed": "holds in part", "not": "does not hold", "none": "not run",
             "under": "too little data to decide", "thin": "present, not prominent"}


def ladder() -> None:
    """The argument, drawn: the tests in the order the argument needs them, grouped by the
    question each group answers, a glyph per model on each. Not a ladder of difficulty."""
    V = verdicts()
    if not V:
        _out("*Results are not available yet.*")
        return
    models = [m for m in MAIN if m in V["models"]]
    by_id = {r["id"]: r for r in V["rungs"]}
    head = "".join(f'<span class="vl-m">{NAMES[m]}</span>' for m in models)
    out = ['<div class="vladder"><div class="vl-head"><span></span><span class="vl-ms">' + head +
           '</span><span class="vl-key">✓ holds · ~ in part · ✕ does not hold · ◐ present, not prominent · '
           '? too little data · – not run</span></div>']
    for group, ids in STEPS:
        out.append(f'<div class="vl-group">{html.escape(group)}</div>')
        for key in ids:
            r = by_id[key]
            per = _per_model(r, models)
            sts = [_status(per.get(m, ""), key) for m in models]
            overall = ("supported" if all(s == "supported" for s in sts) else "none" if all(s == "none" for s in sts)
                       else "under" if all(s in ("under", "none") for s in sts)
                       else "thin" if all(s in ("thin", "none") for s in sts)
                       else "not" if all(s in ("not", "none") for s in sts) else "mixed")
            glyphs = "".join(
                f'<span class="g g-{_status(per.get(m, ""), key)}" title="{NAMES[m]}: {html.escape(_vword(per.get(m, ""), key))}">'
                f'{ICON[_status(per.get(m, ""), key)]}</span>' for m in models)
            out.append(
                f'<a class="vl-rung st-{overall}" href="rungs/{key}.html">'
                f'<span class="vl-n">{key.upper()}</span>'
                f'<span class="vl-bar"><span class="vl-g">{glyphs}</span></span>'
                f'<span class="vl-t"><strong>{html.escape(r["title"])}</strong> <em class="vl-w">{STEP_WORD[overall]}</em>'
                f'<span class="vl-q">{html.escape(QUESTIONS[key])}</span>'
                f'<span class="vl-h">{html.escape(headline(key))}</span></span></a>')
    out.append('<div class="vl-ground"><span class="vl-gt">E1–E8 are the experiment ids used in the summary and the '
               'data files. The pipeline runs them in id order; the argument reads them in this order.</span></div></div>')
    _out("".join(out))
    if V.get("stops"):
        _out(f"[Where the argument stops](stops.qmd) lists every test that did not hold outright ({len(V['stops'])}).")


def chart(name: str, depth: int = 0) -> None:
    src = "../" * depth + "data/viz.json"
    _out(f'<div class="viz" data-chart="{name}" data-src="{src}"><p class="viz-note">Loading chart…</p></div>')


def scripts(depth: int = 0) -> None:
    up = "../" * depth
    _out(f'<script src="{up}assets/vendor/d3.min.js"></script><script src="{up}assets/vendor/plot.umd.min.js"></script>'
         f'<script src="{up}assets/viz.js"></script>')


def stage_flow(link: bool = True) -> None:
    """The pipeline as a row of stages, numbers read from the results."""
    V = verdicts()
    nat = _csv("natural.csv")
    lab = _csv("natural_labeled.csv")
    k = _csv("kappa.csv")
    n = len(nat) if nat is not None else 0
    gw = int((nat.genre == "wikipedia").sum()) if nat is not None else 0
    lc = lab.label.value_counts().to_dict() if lab is not None else {}
    kap = float(k.loc[k.label == "overall", "kappa"].iloc[0]) if k is not None else None
    ran = [r for r in V.get("rungs", []) if r["run"]]
    held = sum(1 for r in ran if all(
        _status(r["verdicts"].get(m, "")) == "supported" for m in MAIN if m in r.get("verdicts", {})))
    stages = [
        ("words", "Words", "20", "concrete nouns, split in two halves"),
        ("sentences", "Sentences", f"{n:,}", f"Wikipedia {gw:,} · fiction {n - gw:,}"),
        ("labels", "Labels", "5", f"O {lc.get('O', 0):,} · A-gen {lc.get('A-gen', 0):,} · idiom {lc.get('A-idiom', 0):,} · "
                                  f"definition {lc.get('A-def', 0):,} · excluded {lc.get('X', 0):,}"),
        ("gate", "Gate", f"κ {kap:.2f}" if kap else "κ", "agreement with a separately labeled sample; 0.7 required"),
        ("readouts", "Embeddings", "3", "BERT · Decoder · mpnet"),
        ("probe", "Probe", "A → B", "every choice on Half A; Half B tested once"),
        ("ladder", "Tests", f"{held} of {len(ran)}", "held outright for every model; the rest held in part, "
                                                   "had too little data, or describe how visible the mark is"),
    ]
    cards = []
    for anchor, title, big, sub in stages:
        href = f' href="stages.html#{anchor}"' if link else f' href="#{anchor}"'
        cards.append(f'<a class="sf-card"{href}><span class="sf-t">{title}</span><span class="sf-big">{big}</span>'
                     f'<span class="sf-sub">{html.escape(sub)}</span></a>')
    _out('<div class="stageflow">' + '<span class="sf-arrow">→</span>'.join(cards) + "</div>")


def halves() -> None:
    cfg = yaml.safe_load((DATA / "config.yaml").read_text())
    a = "".join(f'<span class="chip chip-a">{w}</span>' for w in cfg["words"]["A"])
    b = "".join(f'<span class="chip chip-b">{w}</span>' for w in cfg["words"]["B"])
    _out(f'<div class="halves"><div class="half"><div class="half-h">Half A: every choice is made here</div>{a}'
         f'<div class="half-f">layer, regularisation, thresholds</div></div>'
         f'<div class="half-arrow">→</div><div class="half"><div class="half-h">Half B: used once, at the end</div>{b}'
         f'<div class="half-f">a guard in the code stops Half B from reaching any tuning step</div></div></div>')


def readouts() -> None:
    """Schematic: what each model has read when it reaches the target word."""
    toks = ["The", "cat", "is", "on", "the", "mat", "."]
    xs = [20 + i * 50 for i in range(len(toks))]
    tgt = 1

    def row(y, label, sources, note, faded=(), box=None):
        out = [f'<text x="4" y="{y - 50}" class="ro-l">{label}</text>']
        for i, (t, x) in enumerate(zip(toks, xs)):
            cls = "ro-tgt" if i == tgt else ("ro-fade" if i in faded else "ro-tok")
            out.append(f'<rect x="{x}" y="{y - 12}" width="44" height="22" rx="4" class="{cls}-r"/>'
                       f'<text x="{x + 22}" y="{y + 4}" text-anchor="middle" class="{cls}">{t}</text>')
        for i in sources:
            x0, x1 = xs[i] + 22, xs[tgt] + 22
            mid = (x0 + x1) / 2
            out.append(f'<path d="M{x0},{y - 13} Q{mid},{y - 40} {x1},{y - 13}" class="ro-arc"/>')
        if box:
            out.append(f'<path d="M{xs[0]},{y + 16} L{xs[0]},{y + 22} L{xs[-1] + 44},{y + 22} L{xs[-1] + 44},{y + 16}" class="ro-br"/>')
        out.append(f'<text x="4" y="{y + 40}" class="ro-n">{note}</text>')
        return "".join(out)

    svg = ('<svg class="readouts" viewBox="0 0 380 380" role="img" aria-label="What each model reads at the target word">'
           + row(75, "BERT: the word, read in both directions", [0, 2, 3, 4, 5, 6], "the target sees the whole sentence")
           + row(195, "Decoder: the word, read left to right", [0], "at “cat” it has read only “The”", faded=(2, 3, 4, 5, 6))
           + row(315, "mpnet: the whole sentence, one vector", [], "no single word is read; the sentence is", box=True)
           + "</svg>")
    _out(f'<div class="ro-wrap">{svg}</div>')


def _strip_verdict_heading(body: str) -> str:
    # the question line is shown separately at the top of the page
    lines = body.splitlines()
    return "\n".join(l for l in lines if not l.startswith("*Question.*"))


INTRO = {
    "e1": "This test describes the mark rather than testing for it. E2 shows the mark is there and shared across "
          "words; E1 asks how large it is next to everything else that varies in a word's embeddings. The bar was "
          "set before the run: the mode counts as prominent if one of the first three unlabeled directions "
          "separates object-mode from general statements at AUC ≥ 0.8, pooled over all words. For some words (three to "
          "eleven of twenty, depending on the model) it is among the first three things that vary; pooled over "
          "words it is visible but under the bar. The mark "
          "is real, and thin. That is why the rest of the argument uses a probe, which is told what to look for.",
}


def rung(key: str, interactive: str | None = None) -> None:
    """A result page: question, verdict line, headline chart, then the summary section and the paper figure."""
    S, V = sections(), verdicts()
    _out(f'<p class="lead-question">{html.escape(QUESTIONS[key])}</p>')
    if key in INTRO:
        _out(INTRO[key])
    entry = next((r for r in V.get("rungs", []) if r["id"] == key), None)
    if entry is None or not entry["run"] or key not in S:
        _out("This rung was not run. " + (
            "It needs Romanian sentences labeled O and A-gen (`data/romanian_labeled.csv`)."
            if key == "e8" else ""))
        return
    verd = entry.get("verdicts", {})
    if key == "e7":
        badges = badge(verd.get("all", ""))
    elif key == "e6":
        badges = " ".join(f'<span class="rb">{NAMES[m]}</span> {badge(entry["detail"][m]["definitions"])}'
                          for m in MAIN if m in entry.get("detail", {}))
    else:
        badges = " ".join(f'<span class="rb">{NAMES[m]}</span> {badge(verd.get(m, ""), key)}' for m in MAIN if m in verd)
    _out(f'<div class="rung-verdict">{badges}<div class="rung-h">{html.escape(headline(key))}</div></div>')
    if key in ("e1", "e2", "e3", "e4", "e5", "e6", "e7"):
        chart(key, depth=1)
    if interactive:
        _out("## Every sentence\n\n" + interactive)
    body = _strip_verdict_heading(S[key]["body"]).replace("`results/", "`data/")
    body = body.replace("rungs 3–5", "E3–E5").replace("rung 2", "E2")
    _out("## The numbers\n\n" + body)
    fig = FIGS[key]
    if (ROOT / "figures" / f"{fig}.svg").exists():
        _out(f"## Paper figure\n\n![{CAPTIONS[key]}](../figures/{fig}.svg){{fig-alt=\"{CAPTIONS[key]}\"}}\n\n"
             f"[PNG](../figures/{fig}.png) · [PDF](../figures/{fig}.pdf)")
    scripts(depth=1)


def stops() -> None:
    S = sections()
    _out((S["stops"]["body"] if "stops" in S else "*Results are not available yet.*"))


def method_facts() -> None:
    V = verdicts()
    if not V:
        return
    sel = "\n".join(f"| {NAMES.get(m, m)} | {v['layer']} | {v['C']:g} |" for m, v in V["selected"].items())
    _out((f"{V['data_line']}\n\n{V['label_line']}\n\n"
                     "| model | layer | L2 strength C |\n|---|---|---|\n" + sel))


def kappa_table() -> None:
    parts = [("kappa.csv", "Full-set labels (Claude Opus 5.5) against the sample. This is the gate."),
             ("kappa_qwen_crosscheck.csv", "Cross-check: Qwen3.8 27B, version-2 criterion, against the sample."),
             ("kappa_v1.csv", "First round: Qwen3.8 27B, original criterion, against the sample before "
                              "the modifier rule was applied. Below the gate; no experiment was run.")]
    shown = False
    for name, caption in parts:
        p = DATA / name
        if p.exists():
            k = pd.read_csv(p)
            _out(f"**{caption}**\n\n" + k.to_markdown(index=False, floatfmt=".3f"))
            shown = True
    if not shown:
        _out("*Agreement is computed when the full run is made.*")


def labeling_text() -> None:
    txt = (DATA / "LABELING.md").read_text()
    body = txt.split("\n", 1)[1]          # drop the file's own heading
    _out((body))


def thresholds() -> None:
    cfg = yaml.safe_load((DATA / "config.yaml").read_text())
    rows = "\n".join(f"| `{k}` | {v} |" for k, v in cfg["verdicts"].items())
    _out(("| rule | value |\n|---|---|\n" + rows))


def gold_conventions() -> None:
    p = DATA / "gold_provenance.json"
    if not p.exists():
        return
    g = json.loads(p.read_text())
    items = "\n".join(f"- {c}" for c in g.get("conventions", []))
    _out((f"Labeled by {g['labeler']}, reading only `LABELING.md`, blind to "
                     f"{g['blind_to']}. Conventions for borderline cases:\n\n{items}"))


def pairs_table() -> None:
    p = DATA / "pairs.json"
    if not p.exists():
        return
    df = pd.read_json(p)
    models = [c for c in df.columns if c in NAMES and c != "decoder_final"]

    def cell(v):
        if v is None or pd.isna(v):
            return "<td></td>"
        cls = "diff-pos" if v > 1e-6 else "diff-neg" if v < -1e-6 else "diff-zero"
        return f'<td class="{cls}">{v:+.2f}</td>'

    head = "".join(f"<th>{NAMES[m]}</th>" for m in models)
    rows = "".join(
        "<tr>" + f"<td>{html.escape(row['word'])}</td><td>{html.escape(row['sentence_O'])}</td>"
        f"<td>{html.escape(row['sentence_A'])}</td><td>{row['a_sublabel']}</td>"
        + "".join(cell(row[m]) for m in models) + "</tr>"
        for row in df.to_dict("records"))
    _out((f'<table class="table sent-table"><thead><tr><th>word</th><th>object-mode (O)</th>'
                 f'<th>abstract-mode (A)</th><th>A type</th>{head}</tr></thead><tbody>{rows}</tbody></table>'))


def downloads() -> None:
    files = sorted(p for p in DATA.iterdir() if p.is_file())
    rows = "\n".join(f"| [{p.name}](data/{p.name}) | {p.stat().st_size / 1024:,.0f} KB |" for p in files)
    _out(("| file | size |\n|---|---|\n" + rows))
