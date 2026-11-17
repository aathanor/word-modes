"""Copy pipeline outputs into site/ for Quarto. Run before every render.

MODE_PROBE_SITE_SOURCE=smoke  -> read results/smoke (a banner marks every page)
MODE_PROBE_PUBLISH=1          -> refuse anything but a full, gated run
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from .common import DATA, FIGURES, PKG, RESULTS, config
from .label import labeler_name

SITE = PKG.parent / "site"


def _src() -> tuple:
    tag = os.environ.get("MODE_PROBE_SITE_SOURCE", "").strip()
    return (RESULTS / tag, FIGURES / tag, tag) if tag else (RESULTS, FIGURES, "")


def _source_label(row, legacy: dict) -> tuple[str, str]:
    sid, src = row["sid"], str(row["source_id"])
    if src.startswith("gutenberg:"):
        bid = src.split(":", 1)[1]
        return f"Project Gutenberg #{bid}", f"https://www.gutenberg.org/ebooks/{bid}"
    if src.startswith(("wikidump:", "wiki:")):
        title = src.split(":", 1)[1]
        return f"Wikipedia: {title}", "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
    title = legacy.get(sid, "")
    if title:
        return f"Wikipedia: {title}", "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
    return "Wikipedia (article not recovered)", ""


def _num(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    return str(x)


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def _pick(rows: pd.DataFrame, col: str, q: float) -> dict | None:
    """A short sentence near quantile q of col, preferring ones where both labelers agree."""
    r = rows[rows["sentence"].str.split().str.len() <= 16]
    agree = r[(r["gold"] == "") | (r["gold"] == r["label"])]
    r = agree if len(agree) else r
    if r.empty:
        return None
    target = r[col].quantile(q)
    best = r.iloc[(r[col] - target).abs().argsort().iloc[0]]
    return {"s": best["sentence"], "w": best["word"], "l": best["label"], "x": float(best[col])}


def _viz(res, sc: pd.DataFrame, models: list[str], verdicts: dict) -> dict:
    """Every number the site's charts draw, in one file."""
    v: dict = {"models": models}
    hc = RESULTS / "harvest_counts.csv"
    if hc.exists():
        v["counts"] = _records(pd.read_csv(hc))
    lab = sc[["genre", "label"]].value_counts().reset_index(name="n")
    v["labels"] = _records(lab)
    rounds = []
    for name, path, who, crit in (
            ("Round 1", DATA / "labeling_v1" / "kappa.csv", "Qwen3.8 27B", "criterion v1"),
            ("Round 2", res / "kappa.csv", labeler_name(), "criterion v2"),
            ("Cross-check", res / "kappa_qwen_crosscheck.csv", "Qwen3.8 27B", "criterion v2")):
        if path.exists():
            k = pd.read_csv(path)
            rounds.append({"round": name, "labeler": who, "criterion": crit,
                           "kappa": float(k.loc[k.label == "overall", "kappa"].iloc[0])})
    v["kappa"] = rounds
    v["kappa_min"] = config()["labeler"]["kappa_min"]
    for key, fname in (("e1", "e1_prominence.csv"), ("e2w", "e2_transfer_by_word.csv"),
                       ("e2", "e2_transfer.csv"), ("e3", "e3_genre.csv"), ("e4", "e4_grammar.csv"),
                       ("e4w", "e4_grammar_by_word.csv"), ("e5", "e5_concreteness.csv"),
                       ("e7", "e7_agreement.csv")):
        if (res / fname).exists():
            v[key] = _records(pd.read_csv(res / fname))
    # score distributions per model and label (leave-one-word-out scores)
    dist = []
    for m in models:
        col = f"{m}_lowo"
        if col not in sc:
            continue
        for l, g in sc.groupby("label"):
            x = pd.to_numeric(g[col], errors="coerce").dropna()
            if l == "X" or x.empty:
                continue
            q = x.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            dist.append({"model": m, "label": l, "n": int(len(x)), "q10": q[0.1], "q25": q[0.25],
                         "q50": q[0.5], "q75": q[0.75], "q90": q[0.9]})
    v["dist"] = dist
    # the picture on the home page: BERT scores on unseen (Half-B) words, with examples
    m0 = "bert" if "bert" in models else models[0]
    hb = sc[(sc["half"] == "B") & sc["label"].isin(["O", "A-gen", "A-idiom", "A-def"])].copy()
    hb["x"] = pd.to_numeric(hb[f"{m0}_swap"], errors="coerce")
    hb = hb.dropna(subset=["x"])
    v["hero"] = {"model": m0,
                 "points": [{"x": round(float(r.x), 3), "l": r.label, "w": r.word, "s": r.sentence}
                            for r in hb.itertuples()],
                 "examples": [e for e in (_pick(hb[hb.label == "O"], "x", 0.08),
                                          _pick(hb[hb.label == "A-idiom"], "x", 0.5),
                                          _pick(hb[hb.label == "A-gen"], "x", 0.92)) if e]}
    v["rungs"] = verdicts.get("rungs", [])
    return v


def export() -> int:
    res, fig, tag = _src()
    publish = os.environ.get("MODE_PROBE_PUBLISH") == "1"
    verdicts_path = res / "verdicts.json"
    if not verdicts_path.exists():
        print(f"[site] no results at {res}; run the pipeline first", file=sys.stderr)
        return 1
    verdicts = json.loads(verdicts_path.read_text())
    if publish and (tag or verdicts.get("tag")):
        print(f"[site] refusing to publish a '{tag or verdicts.get('tag')}' run", file=sys.stderr)
        return 2

    out = SITE / "data"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for p in res.glob("*.csv"):
        shutil.copy(p, out / p.name)
    for name in ("verdicts.json", "sections.json", "env.txt", "summary.md"):
        if (res / name).exists():
            shutil.copy(res / name, out / name)
    shutil.copy(PKG / "LABELING.md", out / "LABELING.md")
    shutil.copy(PKG / "config.yaml", out / "config.yaml")
    for name in ("natural.csv", "control_pairs.csv", "gold_to_label.csv"):
        shutil.copy(DATA / name, out / name)
    if (DATA / "natural_labeled.csv").exists() and not tag:
        shutil.copy(DATA / "natural_labeled.csv", out / "natural_labeled.csv")
    if (DATA / "labeling_qwen_v2" / "natural_labeled.csv").exists() and not tag:
        shutil.copy(DATA / "labeling_qwen_v2" / "natural_labeled.csv", out / "natural_labeled_qwen.csv")
    if (DATA / "labeling_v1" / "kappa.csv").exists():
        shutil.copy(DATA / "labeling_v1" / "kappa.csv", out / "kappa_v1.csv")
    if (DATA / "gold_claude" / "provenance.json").exists():
        shutil.copy(DATA / "gold_claude" / "provenance.json", out / "gold_provenance.json")

    figs = SITE / "figures"
    shutil.rmtree(figs, ignore_errors=True)
    figs.mkdir()
    for p in list(fig.glob("*.svg")) + list(fig.glob("*.png")) + list(fig.glob("*.pdf")):
        shutil.copy(p, figs / p.name)

    # sentences for the explorer and the interactive strips
    sc = pd.read_csv(res / "scores.csv", keep_default_na=False)
    nat = pd.read_csv(DATA / "natural.csv")[["sid", "sentence"]]
    sc = sc.merge(nat, on="sid", how="left")
    legacy = {}
    if (DATA / "legacy_sources.csv").exists():
        ls = pd.read_csv(DATA / "legacy_sources.csv", keep_default_na=False)
        legacy = dict(zip(ls["sid"], ls["title"]))
    src = [_source_label(r, legacy) for _, r in sc.iterrows()]
    models = verdicts["models"]
    rows = []
    for (_, r), (label, url) in zip(sc.iterrows(), src):
        rec = {"sid": r.sid, "w": r.word, "g": r.genre, "h": r.half, "l": r.label,
               "gl": r.gold, "s": r.sentence, "src": label, "url": url}
        for m in models:
            for kind in ("swap", "lowo"):
                v = r.get(f"{m}_{kind}", "")
                rec[f"{m}_{kind}"] = None if v == "" or pd.isna(v) else round(float(v), 4)
        rows.append(rec)
    (out / "sentences.json").write_text(json.dumps({"models": models, "rows": rows}))

    # minimal pairs with each model's difference
    pairs = pd.read_csv(DATA / "control_pairs.csv")
    diffs = pd.read_csv(res / "e4_minimal_pairs.csv")
    for m in diffs["model"].unique():
        d = diffs[diffs.model == m].set_index("pair")["diff"]
        pairs[m] = [None if i not in d or pd.isna(d[i]) else round(float(d[i]), 4)
                    for i in range(len(pairs))]
    pairs.to_json(out / "pairs.json", orient="records")

    (out / "viz.json").write_text(json.dumps(_viz(res, sc, models, verdicts), default=_num))

    # run stamp and banner
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PKG.parent,
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    gold = config()["gold"]
    stamp = (f'<div class="run-stamp">Results generated {verdicts["generated"]} · code '
             f'{commit or "uncommitted"} · labels: {labeler_name()} (all sentences), '
             f'{gold["labeler"]} (gold sample)</div>')
    (SITE / "_stamp.html").write_text(stamp)
    banner = ""
    if tag or verdicts.get("tag"):
        banner = ('<div class="smoke-banner">Preview built from a <strong>'
                  f'{tag or verdicts.get("tag")}</strong> run. Labels are random and the numbers '
                  'say nothing about the hypothesis.</div>')
    (SITE / "_banner.html").write_text(banner)
    print(f"[site] exported {len(rows)} sentences, {len(pairs)} pairs from {res}")
    return 0


if __name__ == "__main__":
    sys.exit(export())
