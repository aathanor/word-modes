"""End-to-end pipeline.

python -m mode_probe.run_all                 harvest, write minimal pairs and the gold set, stop
python -m mode_probe.run_all --label         model-label every sentence (resumable)
python -m mode_probe.run_all --labels-ready  check gold, kappa gate, embed, E1-E7, figures, summary
    --quick      BERT and mpnet only, 50 sentences per word (smoke-testing)
    --romanian   also run E8 if data/romanian_labeled.csv exists
    --decoder N  override the decoder model name
    --smoke      code check on random labels; writes to results/smoke and figures/smoke only
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import os
import platform
import sys

# Hugging Face's Xet backend returned 404s for public models on this machine; plain HTTP works.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import pandas as pd

from . import experiments as E
from . import label as LB
from . import summary as S
from .common import (CACHE, DATA, LABELS, PERMITTED_PHRASE, config, detect_device, ensure_dirs,
                     half_of, rdir, set_output, set_seed)
from .embed import embed
from .harvest import harvest, load_natural
from .pairs import write_pairs


def write_env(device: str, models: list[str]) -> None:
    pkgs = ["torch", "transformers", "sentence-transformers", "scikit-learn", "scipy", "numpy",
            "pandas", "statsmodels", "spacy", "en-core-web-sm", "matplotlib", "pyyaml"]
    lines = [f"python {sys.version.split()[0]}", f"platform {platform.platform()}",
             f"device {device}"]
    try:
        import torch

        lines.append(f"torch hip {getattr(torch.version, 'hip', None)} cuda {torch.version.cuda}")
    except ImportError:
        pass
    for p in pkgs:
        try:
            lines.append(f"{p}=={md.version(p)}")
        except md.PackageNotFoundError:
            lines.append(f"{p} not installed")
    for m in models:
        lines.append(f"model {m}: {config()['models'][m]['name']}")
    lc = config()["labeler"]
    lines.append(f"labeler {lc['backend']}:{lc['model']}")
    lines.append(f"seed {config()['seed']}")
    (rdir() / "env.txt").write_text("\n".join(lines) + "\n")


def grammar_features(nat: pd.DataFrame) -> pd.DataFrame:
    from .grammar import feature_frame

    h = hashlib.sha1("".join(nat["word"] + "\t" + nat["sentence"] + "\n").encode()).hexdigest()[:16]
    path = CACHE / f"grammar_{h}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0).set_index(nat.index)
    print("[grammar] parsing sentences with spaCy")
    G = feature_frame(nat)
    G.to_csv(path)
    return G


def pair_frame() -> pd.DataFrame:
    p = write_pairs()
    rows = []
    for i, r in p.iterrows():
        h = half_of(r.word)
        rows.append({"sid": f"p{i:03d}O", "word": r.word, "sentence": r.sentence_O, "half": h,
                     "a_sublabel": r.a_sublabel, "side": "O"})
        rows.append({"sid": f"p{i:03d}A", "word": r.word, "sentence": r.sentence_A, "half": h,
                     "a_sublabel": r.a_sublabel, "side": "A"})
    return pd.DataFrame(rows)


def quick_subset(nat: pd.DataFrame, per_word: int) -> pd.DataFrame:
    """per_word sentences per word (half per genre); gold rows are kept first."""
    keep = []
    for _, sub in nat.groupby(["word", "genre"]):
        sub = sub.sample(frac=1, random_state=config()["seed"])
        sub = pd.concat([sub[sub.gold != ""], sub[sub.gold == ""]])
        keep.append(sub.head(per_word // 2))
    return pd.concat(keep).sort_index().reset_index(drop=True)


def save_results(ctx, res: dict, meta: dict) -> None:
    """Everything summary.write needs, so wording changes do not need a rerun."""
    import pickle

    with open(rdir() / "run_results.pkl", "wb") as f:
        pickle.dump({"models": ctx.models, "sel": ctx.sel, "res": res, "meta": meta}, f)


def rewrite_summary() -> int:
    import pickle
    from types import SimpleNamespace

    p = rdir() / "run_results.pkl"
    if not p.exists():
        print(f"[summary] {p} not found; it is written at the end of a full run", file=sys.stderr)
        return 1
    with open(p, "rb") as f:
        d = pickle.load(f)
    S.write(SimpleNamespace(models=d["models"], sel=d["sel"]), d["res"], d["meta"])
    print(f"[summary] rewrote {rdir() / 'summary.md'} from {p}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mode_probe.run_all")
    ap.add_argument("--labels-ready", action="store_true")
    ap.add_argument("--label", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--romanian", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--decoder", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--summary-only", action="store_true",
                    help="rewrite summary.md, sections.json and verdicts.json from the last run's saved results")
    args = ap.parse_args(argv)

    ensure_dirs()
    set_seed()
    if args.summary_only:
        return rewrite_summary()
    if args.decoder:
        config()["models"]["decoder"]["name"] = args.decoder

    # ------------------------------------------------ stage 1: data
    if not (DATA / "natural.csv").exists():
        harvest()
    natural = load_natural()
    write_pairs()
    LB.make_gold(natural)

    if args.label:
        LB.model_label(natural)
        return 0

    if not (args.labels_ready or args.smoke):
        print(LB.gold_ready_message())
        return 0

    # ------------------------------------------------ stage 2: labels
    tag, tag_note = None, ""
    gold, missing = LB.gold_status()
    if args.smoke:
        tag = "smoke"
        tag_note = ("Labels are random. These numbers test the code only and say nothing about "
                    "the hypothesis.")
        set_output("smoke")
        rng = np.random.default_rng(config()["seed"])
        labeled = natural.copy()
        labeled["label"] = rng.choice(LABELS, size=len(natural), p=[.42, .42, .06, .06, .04])
        gold = gold.copy()
        gold["label"] = rng.choice(LABELS, size=len(gold), p=[.42, .42, .06, .06, .04])
        kappa_line = "κ not computed (smoke run)."
    else:
        if missing:
            print(f"[labels] {missing} gold rows have no valid label.")
            print(LB.gold_ready_message())
            return 1
        if config()["labeler"]["backend"] == "anthropic_batch":
            if not LB.LABELED.exists():
                print("[labels] no Claude labels yet: python -m mode_probe.label_claude --submit, then --collect")
                return 1
            labeled = pd.read_csv(LB.LABELED)
        else:
            labeled = LB.model_label(natural)
        if len(labeled) < len(natural) or labeled["label"].isna().any():
            print("[labels] model labeling incomplete")
            return 1
        k = LB.kappa_report(gold, labeled)
        print(k["table"].to_string(index=False))
        kappa_line = f"Cohen's κ (model vs gold, n={len(gold)}) = {k['overall']:.3f}"
        if LB.QWEN_V2.exists():
            q = pd.read_csv(LB.QWEN_V2)
            if len(q) == len(natural):
                kq = LB.kappa_report(gold, q, name="kappa_qwen_crosscheck")
                kappa_line += f"; cross-family check, {config()['labeler']['model']} vs gold: κ = {kq['overall']:.3f}"
                from sklearn.metrics import cohen_kappa_score

                both = labeled[["sid", "label"]].merge(q[["sid", "label"]], on="sid")
                kappa_line += (f". The two labelers agree on {(both.label_x == both.label_y).mean():.0%} "
                               f"of all {len(both):,} sentences (κ = "
                               f"{cohen_kappa_score(both.label_x, both.label_y):.3f}).")
        if k["overall"] < config()["labeler"]["kappa_min"]:
            msg = (f"# Stopped at the labeling gate\n\n{kappa_line}, below the required "
                   f"{config()['labeler']['kappa_min']}. No experiment was run. Per-label κ and the "
                   "confusion matrix are in `results/kappa.csv` and `results/kappa_confusion.csv`.\n")
            (rdir() / "summary.md").write_text(msg)
            print(msg)
            return 2
        if args.quick:
            tag, tag_note = "quick", "BERT and mpnet only, 50 sentences per word."

    nat = labeled.merge(gold[["sid", "label"]].rename(columns={"label": "gold"}), on="sid", how="left")
    nat["gold"] = nat["gold"].fillna("").astype(str)
    if args.quick or args.smoke:
        nat = quick_subset(nat, config()["harvest"]["quick_per_word"])
    nat = nat.reset_index(drop=True)

    # ------------------------------------------------ stage 3: embeddings
    device = args.device or detect_device()
    models = list(config()["quick_models"] if (args.quick or args.smoke) else config()["main_models"])
    write_env(device, models + (["multilingual"] if args.romanian else []))
    emb = {m: embed(m, nat, "natural", device) for m in models}
    pairs = pair_frame()
    pair_emb = {m: embed(m, pairs, "pairs", device) for m in models}
    if "decoder" in models:
        models.append("decoder_final")
        pair_emb["decoder_final"] = pair_emb["decoder"]

    from .concreteness import load_norms, mean_concreteness

    norms = load_norms()
    ctx = E.Ctx(nat=nat, emb=emb, models=models, pairs=pairs, pair_emb=pair_emb,
                grammar=grammar_features(nat),
                conc=mean_concreteness(nat, norms) if norms else None)

    # ------------------------------------------------ stage 4: experiments
    E.select_all(ctx)
    res = {}
    print("[E1]"); res["e1"] = E.e1_prominence(ctx)
    print("[E2]"); res["e2"] = E.e2_transfer(ctx)
    if not (args.quick or args.smoke):
        from . import robustness as R

        if R.available(len(ctx.nat)):
            print("[E2] label robustness"); res["robust"] = R.from_ctx(ctx)
    print("[E3]"); res["e3"] = E.e3_genre(ctx, res["e2"])
    print("[E4]"); res["e4"] = E.e4_grammar(ctx)
    print("[E5]"); res["e5"] = E.e5_concreteness(ctx)
    print("[E6]"); res["e6"] = E.e6_predictions(ctx)
    print("[E7]"); res["e7"] = E.e7_agreement(ctx)
    ro_path = DATA / "romanian_labeled.csv"
    if args.romanian and ro_path.exists():
        print("[E8]")
        ro = pd.read_csv(ro_path)
        ro = ro.assign(word=ro["word_en"], sid=[f"r{i:04d}" for i in range(len(ro))])
        ro_emb = embed("multilingual", ro, "romanian", device)
        en_emb = embed("multilingual", nat, "natural", device)
        res["e8"] = E.e8_romanian(ctx, ro, ro_emb, en_emb)
    elif args.romanian:
        print(f"[E8] skipped: {ro_path} not found")

    counts = nat["label"].value_counts().to_dict()
    meta = {"tag": tag, "tag_note": tag_note,
            "data_line": (f"{len(nat)} natural sentences, {nat.word.nunique()} words, "
                          f"{(nat.genre == 'wikipedia').sum()} Wikipedia and "
                          f"{(nat.genre == 'fiction').sum()} fiction; 200 minimal pairs."),
            "label_line": (f"Model labels ({LB.labeler_name()}): " +
                           ", ".join(f"{k} {counts.get(k, 0)}" for k in LABELS) +
                           f". Gold: {(nat.gold != '').sum()} sentences. {kappa_line}")}
    S.write(ctx, res, meta)
    save_results(ctx, res, meta)
    sc = nat[["sid", "word", "genre", "half", "source_id", "label", "gold"]].copy()
    for m in ctx.models:
        sc[f"{m}_swap"] = ctx.scores[m]["swap"]
        sc[f"{m}_lowo"] = ctx.scores[m]["lowo"]
    sc.to_csv(rdir() / "scores.csv", index=False)
    print(f"[done] {rdir() / 'summary.md'}  ({PERMITTED_PHRASE})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
