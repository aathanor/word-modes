# Further tests

These tests were added after the first results, to separate this experiment's distinction from the linguistic one between episodic and generic sentences, and to test the explanation of the mark offered on the home page. Their criterion, measures and predictions were committed to the repository before their data existed ([PREREGISTRATION_v3.md](https://github.com/aathanor/word-modes/blob/main/mode_probe/PREREGISTRATION_v3.md)), except the UDS-G transfer, which is reported without verdicts.

**Labels (version 3).** Claude Opus 5.5 labeled every sentence under [LABELING_v3.md](https://github.com/aathanor/word-modes/blob/main/mode_probe/LABELING_v3.md); the 300-sentence sample was relabeled in separate requests. Gate: κ = 0.96 (n = 300, eight labels; required 0.7) — passed. Both labelings come from the same model with the same instruction, so this measures how consistently it applies the criterion, not agreement between independent judges. Cross-check, the local Qwen3.8 27B against the sample: κ = 0.56; against Claude on all 9,488 sentences: κ = 0.59. Where both put a sentence in P-ep or K-gen, they agree on which in 94% of cases; most of the disagreement is over habits of particular things, which Qwen tends to call episodes (see E9 below).

## Reference or occasion? (E9)

The object/general difference could follow **reference** (a particular thing or the kind; the PCF reading) or **occasion** (one episode or what holds in general; the reading from the linguistics of genericity). The two agree on most sentences and disagree on two kinds: habits and standing properties of a particular thing ("My cat always sleeps by the fire", P-gen) and episodes in the history of a kind ("Cats reached Australia in the seventeenth century", K-ep). A probe trained only on the cells both readings agree on (P-ep against K-gen) places the two disputed cells on a scale from 0 (with particular episodes) to 1 (with general statements about the kind), on words it never saw.

| model | P-gen position [95% CI] | K-ep position [95% CI] | n P-gen / K-ep | verdict |
|---|---|---|---|---|
| BERT | 0.41 [0.37, 0.46] | 0.76 [0.69, 0.83] | 752 / 224 | **no verdict (pattern outside the pre-registered cases)** |
| Decoder | 0.54 [0.51, 0.58] | 0.57 [0.51, 0.63] | 752 / 224 | **both** |
| mpnet | 0.62 [0.59, 0.67] | 0.62 [0.56, 0.70] | 752 / 224 | **both** |

Reference predicts P-gen near 0 and K-ep near 1; occasion predicts the reverse. BERT and the decoder decide; mpnet is reported.

**What this shows.** Neither reading fits alone. For both primary readouts the disputed cells sit between the poles, so the direction responds to reference and to occasion. BERT leans toward reference: the episodes about kinds sit near the general statements (0.76), and the habits of particular things sit nearer the particular episodes (0.41), though not as near as the reference reading predicted; that combination was not one of the pre-registered cases, so it gets no verdict. The decoder weighs the two about equally (0.54 and 0.57). The claim that the mark is PCF's distinction rather than genericity is not supported as stated; the mark carries both.

**Minimal quadruples.** Five sets of four written sentences per word, differing only in reference and occasion. Within each set, the reference effect is how far the kind sentences sit from the particular ones; the occasion effect, how far the general sentences sit from the episodes.

| model | sets where reference > occasion | mean reference effect | mean occasion effect | verdict |
|---|---|---|---|---|
| BERT | 94 / 100 | 5.50 | 2.09 | **reference larger** |
| Decoder | 47 / 100 | 4.63 | 4.79 | **neither** |
| mpnet | 40 / 100 | 1.40 | 1.64 | **neither** |

The labeler put 96% of the written sentences in the cell they were written for. The quadruples were written by the same model that wrote the code.

**Does it depend on the labeler?** The local model (Qwen3.8 27B) agrees with Claude on almost all the particular episodes and on most general statements, but moves most of Claude's habits of particular things (P-gen) into the particular episodes. Since E9 rests on exactly that cell, it was repeated with Qwen's labels and with only the sentences both labelers put in the same cell. Added after the results; no verdicts.

| model | labels | P-gen position [95% CI] | K-ep position [95% CI] | n P-gen / K-ep |
|---|---|---|---|---|
| BERT | Claude | 0.41 [0.37, 0.46] | 0.76 [0.69, 0.83] | 752 / 224 |
| BERT | Qwen | 0.31 [0.26, 0.35] | 0.72 [0.67, 0.78] | 303 / 344 |
| BERT | both agree | 0.37 [0.34, 0.43] | 0.79 [0.73, 0.85] | 226 / 142 |
| Decoder | Claude | 0.54 [0.51, 0.58] | 0.57 [0.51, 0.63] | 752 / 224 |
| Decoder | Qwen | 0.48 [0.43, 0.53] | 0.56 [0.51, 0.61] | 303 / 344 |
| Decoder | both agree | 0.58 [0.54, 0.61] | 0.58 [0.51, 0.65] | 226 / 142 |
| mpnet | Claude | 0.62 [0.59, 0.67] | 0.62 [0.57, 0.70] | 752 / 224 |
| mpnet | Qwen | 0.54 [0.49, 0.59] | 0.67 [0.63, 0.73] | 303 / 344 |
| mpnet | both agree | 0.63 [0.58, 0.67] | 0.69 [0.60, 0.75] | 226 / 142 |

The pattern holds across labelers: the decoder places both disputed cells near the middle, and BERT leans toward reference. With Qwen's labels, BERT's numbers would meet the pre-registered reference prediction (P-gen at 0.31, K-ep at 0.72), because the sentences Qwen still calls P-gen are the clearer cases. With the sentences both labelers agree on, they would not (P-gen at 0.37). How strongly BERT leans toward reference depends on where the line between a habit and an episode is drawn.

**Which way does the original direction point?** Cosine between the E2 direction (object against general, version-2 labels) and a direction trained on reference alone or on occasion alone:

| model | with reference | with occasion |
|---|---|---|
| BERT | 0.74 | 0.61 |
| Decoder | 0.59 | 0.52 |
| mpnet | 0.89 | -0.51 |

## Writing from encounter (E10)

The home page explains the mark as the way people write when they write from an encounter. If so, particular sentences in diaries, letters and travel accounts should carry it more strongly than particular sentences in novels or in an encyclopedia. The measure is how well the probe (trained on Wikipedia and fiction) separates particular episodes from general statements within each source, on unseen words (AUC).

| model | Wikipedia | fiction | encounter writing | encounter − best other [95% CI] | verdict |
|---|---|---|---|---|---|
| BERT | 0.961 | 0.963 | 0.927 | -0.036 [-0.054, -0.021] | **inconclusive** |
| Decoder | 0.971 | 0.942 | 0.939 | -0.032 [-0.048, -0.015] | **inconclusive** |
| mpnet | 0.953 | 0.927 | 0.928 | -0.025 [-0.043, -0.007] | **inconclusive** |

**What this shows.** In all three models the mark is *less* sharp in writing from encounter than in the novels and the encyclopedia, by 0.03–0.04, and the intervals exclude zero. That is the opposite of what the encounter explanation predicts. The pre-registered rule planned for "stands out" and for "the same", not for "weaker", so its verdict is "inconclusive"; the data do not support the encounter explanation. One reason may be that diaries and letters mix reports of encounters with reflection, and that the probe was trained on the other two genres.

## Definitions, with definitions collected on purpose (E6 rerun)

Definitions were too few to test (11). Wikipedia sentences of the form "A spoon is a …" were collected on purpose and labeled with the rest; the original rule is unchanged: definitions should sit beyond the general statements on a direction that never saw them.

Collecting on purpose did not help much: Wikipedia sentences that state what a concrete noun is are rare outside each article's first line, and only 8 of the candidates were definitions.

| model | definitions | beyond the general-statement median | Mann–Whitney p | verdict |
|---|---|---|---|---|
| BERT | 8 | 0.88 | 0.0002 | **supported (underpowered)** |
| Decoder | 8 | 0.75 | 0.0712 | **not supported (underpowered)** |
| mpnet | 8 | 0.12 | 0.9806 | **not supported (underpowered)** |

## Rod and cap (E11)

The 2025 study read object uses as a tight "rod" and general uses as a wide "cap". Here the spread of each class is measured after correcting for the embeddings' uneven use of directions, and within groups of sentences on the same topic (found with the noun masked), because general sentences come from more varied topics, which alone would spread them. A ratio above 1 means general uses spread wider.

| model | words where general uses spread wider | median ratio, raw | median ratio, same topic | verdict |
|---|---|---|---|---|
| BERT | 4 / 18 | 0.948 | 0.970 | **no difference** |
| Decoder, at the word | 6 / 18 | 0.967 | 0.981 | **no difference** |
| Decoder | 7 / 18 | 0.975 | 0.996 | **no difference (not a primary readout)** |
| mpnet | 6 / 18 | 0.965 | 0.990 | **no difference (not a primary readout)** |

## Against an independent annotation of genericity (UDS-G)

Govindarajan, Van Durme and White (2019) had people rate every noun in an English treebank for how particular and how kind-like it is, and every verb for how episodic. Their annotators knew nothing of this experiment. Probes trained on one data set are tested on the other, reading the noun itself. Added after the first results; no verdicts.

| model | their particular/kind probe on our object/general sentences | our probe on their particular vs kind nouns | our probe on their episodic vs general verbs | cosine of the two directions |
|---|---|---|---|---|
| BERT | 0.794 | 0.609 | 0.574 | 0.16 |
| Decoder | 0.777 | 0.645 | 0.584 | 0.09 |
