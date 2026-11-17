# Object-words and abstract-words leave different marks in language models

Take the word "cat". You can use it of the cat on your lap, or of cats in general:
"Cats hunt at night." This repository holds an experiment asking whether that
difference leaves a trace in language models, which have only ever seen text. It
does: a straight cut through the models' representations, learned from sentences
about ten nouns, sorts sentences about ten other nouns by how the noun is used. The trace is not genre, grammar or the concreteness of the surrounding words, and it is thin: present, but not one of the main things that vary when a word changes sentence.

Results, argument and every sentence: **https://wordmodes.antimaterie.ro**

| folder | what it holds |
|---|---|
| [`mode_probe/`](mode_probe/) | the experiment: harvesting, labeling, embeddings, the eight tests, figures, summary. How to run it: [`mode_probe/README.md`](mode_probe/README.md). |
| [`mode_probe/results/`](mode_probe/results/) | every table the site draws, and [`summary.md`](mode_probe/results/summary.md) |
| [`mode_probe/data/`](mode_probe/data/) | the 5,900 sentences, their labels, the 300-sentence check sample, the minimal pairs |
| [`site/`](site/) | the Quarto source of the website and its deploy scripts |

Downloads (Wikipedia and Project Gutenberg) and model embeddings are not stored
here; `python -m mode_probe.run_all` fetches and recomputes them. The concreteness
ratings of Brysbaert, Warriner and Kuperman (2014) are downloaded the same way.

## Licences

| part | licence |
|---|---|
| Code: `mode_probe/**/*.py`, `site/*.py`, `site/*.sh`, `site/assets/*.js`, `site/styles.scss`, `site/deploy/` | MIT ([`LICENSE`](LICENSE)) |
| Data and results: everything under `mode_probe/data/`, `mode_probe/results/`, `mode_probe/figures/` | CC BY-SA 4.0 ([`LICENSES/CC-BY-SA-4.0.txt`](LICENSES/CC-BY-SA-4.0.txt)) |
| Text: the prose of the site pages (`site/*.qmd`, `site/rungs/*.qmd`) and `mode_probe/LABELING.md` | CC BY 4.0 ([`LICENSES/CC-BY-4.0.txt`](LICENSES/CC-BY-4.0.txt)) |
| Bundled libraries in `site/assets/vendor/` | their own ([`NOTICE.md`](site/assets/vendor/NOTICE.md)) |

The data carry share-alike because many of the sentences are quoted from Wikipedia,
whose text is CC BY-SA 4.0; each such sentence names its article in the site's
sentence explorer. Sentences from Project Gutenberg books are in the public domain
in the United States.

## Citing

Florin Cojocariu (2026). *Object-words and abstract-words leave different marks in
language models.* https://wordmodes.antimaterie.ro, code and data at
https://github.com/aathanor/word-modes.
