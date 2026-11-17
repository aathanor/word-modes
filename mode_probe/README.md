# mode_probe

## What is tested

Whether the embedding of a sentence containing a concrete noun carries a mark of the
mode of use: object-mode (`O`, one instance on one occasion) versus abstract-mode
(`A-gen` general predication, `A-idiom`, `A-def`), and whether that mark is one
direction shared across words, genres and models. The labeling criterion is
`LABELING.md`. Twenty nouns are split into Half A (all choices) and Half B (tested once).

| rung | experiment | settles |
|---|---|---|
| 1 | E1 prominence | is the mode a main axis of variation without labels? |
| 2 | E2 cross-word transfer | does a probe trained on some words work on unseen words? |
| 3 | E3 cross-genre transfer | is the mark genre (Wikipedia vs fiction)? |
| 4 | E4 grammar | is the mark more than determiner, number and tense? minimal pairs |
| 5 | E5 concreteness | is the mark context-word concreteness? |
| 6 | E6 predictions | do definitions sit at the abstract end; where do idioms sit? |
| 7 | E7 agreement | do BERT, the decoder and mpnet order sentences alike? |
| 8 | E8 Romanian (optional) | does an English-trained probe work on Romanian? |

Models: `bert-base-uncased` (target token, mean of wordpieces), `Qwen/Qwen3-1.7B`
(target token, last piece; final-token state reported as a supplement),
`all-mpnet-base-v2` (sentence), `paraphrase-multilingual-mpnet-base-v2` (E8 only).

## How to run

Environment: `conda activate mode-probe` (Python 3.11; torch for ROCm/CUDA/CPU,
transformers, sentence-transformers, scikit-learn, statsmodels, spaCy with
`en_core_web_sm`, pyarrow). Model labels come from Claude Opus 5.5 through the Message Batches API (`label_claude.py`,
`ANTHROPIC_API_KEY`); a local Ollama model relabels everything as a cross-check (`config.yaml`).
Run from the repository root.

```
python -m mode_probe.run_all                  # harvest, minimal pairs, gold set; stops
# label data/gold_to_label.csv (column `label`); record who labeled it in config.yaml `gold:`
# (current gold labels: Claude Opus 5.5, blind to the model labels; see data/gold_claude/)
python -m mode_probe.run_all --label          # model labels for all sentences (resumable)
python -m mode_probe.run_all --labels-ready   # kappa gate, embeddings, E1–E7, figures, summary
python -m mode_probe.run_all --labels-ready --romanian   # adds E8 if data/romanian_labeled.csv exists
python -m mode_probe.run_all --labels-ready --quick      # BERT + mpnet, 50 sentences per word
python -m mode_probe.run_all --smoke          # code check on random labels (results/smoke/)
python -m pytest mode_probe/tests
```

Human labels for the gold sample go in `data/gold_human_to_label.xlsx`: the label column
has a drop-down of the five labels, the target noun is in bold, and the second sheet
summarises the criterion. The sheet shows no earlier labels. When some rows are filled:

```
python -m mode_probe.human --report   # κ against Claude, Qwen and the first gold labels
python -m mode_probe.human --sheet    # rebuild the sheet (refuses if it already has labels)
```

`--decoder NAME` replaces the decoder; `--device cpu` forces CPU.

## Where the numbers are

- `results/summary.md`: the ladder, verdicts, and where the argument stops.
- `results/e1_prominence.csv` … `results/e8_romanian.csv`: one table per experiment;
  `e2_transfer_by_word.csv`, `e4_grammar_by_word.csv` and `e4_minimal_pairs.csv` hold
  the per-word and per-pair values.
- `results/kappa.csv`, `results/kappa_confusion.csv`: model labels against the gold set.
- `results/selected.csv`, `results/hyperparameters_halfA.csv`: layer and L2 choices (Half A).
- `results/harvest_counts.csv`, `results/legacy_cache_check.csv`, `results/env.txt`.
- `figures/fig1_e1_prominence` … `figures/fig8_e8_romanian` (PNG and PDF).
- Data: `data/natural.csv`, `data/natural_labeled.csv`, `data/gold_to_label.csv`,
  `data/control_pairs.csv`.
