# Predictions for the version-3 experiments, written before the data

Written 17 November 2026, before any sentence was labeled under `LABELING_v3.md`, before the
encounter sentences were collected, and before the minimal quadruples were written. The git
commit that adds this file is the record of the date. Nothing below may be changed after the
labels exist; any later change is listed under "Amendments" with its date and reason.

Models: BERT and the main decoder (readout and layer chosen on Half A, as for E1–E7) are the
primary readouts; mpnet is reported but does not decide. The probe, its regularisation and the
Half A / Half B split are as in E1–E7. All choices are made on Half A.

## Labeling gate (as before)

The full set is labeled under version 3 by Claude Opus 5.5 (Message Batches API, the criterion
as the whole instruction, output restricted to the eight labels). The 300-sentence sample is
relabeled under version 3 by Claude Opus 5.5 in a separate request set that does not see the
full-set labels. The gate is Cohen's κ ≥ 0.7 between the two, on the eight labels. The local
model (Qwen3.8 27B) relabels everything as a cross-check and is reported, not gated.
If the gate fails, no version-3 experiment is run.

## E9: reference or occasion?

**Question.** The object/general difference found in E2 could follow reference (particular vs
kind, the PCF reading) or occasion (one episode vs in general, the genericity reading). The
two agree on P-ep and K-gen and disagree on P-gen and K-ep.

**Primary measure.** A probe is trained on P-ep against K-gen only (the cells both readings
agree on), Half A → Half B and Half B → Half A, exactly as in E2. Each held-out P-gen and K-ep
sentence gets a score. For a cell c, its position is

  t(c) = (median score of c − median of P-ep) / (median of K-gen − median of P-ep),

computed on the held-out half and averaged over the two directions; 0 is the particular-episode
end and 1 the kind-general end. 95% intervals by bootstrap over sentences within cells (1,000
resamples).

**Prediction from reference (PCF).** t(P-gen) ≤ 0.33 and t(K-ep) ≥ 0.67.
**Prediction from occasion (genericity).** t(P-gen) ≥ 0.67 and t(K-ep) ≤ 0.33.

**Verdicts, per model.**
- *follows reference*: both reference predictions hold, with the 95% interval of t(P-gen)
  below 0.5 and that of t(K-ep) above 0.5;
- *follows occasion*: the mirror image;
- *both*: t(P-gen) and t(K-ep) both lie between 0.33 and 0.67, or the intervals straddle 0.5;
- *underpowered*: fewer than 20 held-out sentences in P-gen or in K-ep, pooled over both
  directions.

**Secondary measure.** Two further probes, each trained on all four cells: a reference probe
(P-* against K-*) and an occasion probe (*-ep against *-gen). Report the cosine of each with the
E2 probe direction (O against A-gen under version 2), and the held-out AUC of each. Reported,
not used for a verdict.

**Minimal quadruples.** For every word, five sets of four written sentences that differ only in
reference and occasion (for example "My cat slept by the fire last night." / "My cat sleeps by
the fire." / "Cats first reached the island in 1650." / "Cats sleep by fires."). Each sentence
is scored by the P-ep/K-gen probe trained on the other half of the words. Within a quadruple,
the reference effect is mean(K) − mean(P) and the occasion effect is mean(gen) − mean(ep).
*Prediction from reference:* the reference effect exceeds the occasion effect in more than half
the quadruples (one-sided binomial test, α = 0.01). *From occasion:* the mirror image. The
quadruples are written by the same model that wrote the code; this is reported with the result.

## E10: is the mark a mark of encounter, or of particular reference?

**Question.** The home page explains the mark as "the way people write when they write from an
encounter". If so, particular sentences written from encounter should carry it more strongly
than particular sentences in fiction or in an encyclopedia.

**Data.** A third source: diaries, letters and travel accounts from Project Gutenberg, in
English, by authors born 1760–1930, at most three books per author, at most six sentences per
book, 150 sentences per word where available, same sentence filters as the novels. Labeled under
version 3 like the rest.

**Primary measure.** The P-ep/K-gen probe from E9, trained on Wikipedia and fiction only (Half
A → Half B and back), scores the held-out words in all three sources. In each source s, the
mark's strength is the AUC of P-ep against K-gen within s, pooled over held-out words.

**Predictions.**
- *Encounter stands out*: AUC(encounter) − max(AUC(fiction), AUC(Wikipedia)) ≥ 0.03, with the
  95% bootstrap interval of the difference above 0.
- *Mark of particular reference*: every pairwise difference between the three sources has its
  95% interval within ±0.03.
- *Inconclusive*: anything else.
- *Underpowered*: fewer than 20 held-out P-ep or K-gen sentences in the encounter source.

Verdict per primary model. If "mark of particular reference" holds for both primary models, the
home page drops the encounter explanation and says the mark is one of particular reference.

## Definitions (E6, rerun)

Definitions collected on purpose (Wikipedia sentences of the form "A/An/The X is a(n) …" and
"X(s) are …" stating what X is) are added to the set and labeled with the rest. E6 is rerun with
its original, unchanged rule. If fewer than 20 A-def sentences result, E6 stays underpowered.

## Amendments

**17 November 2026, before any dispersion was computed: the rod/cap test (E11).** Uses the
version-2 labels of the 5,900 sentences, which already exist; no dispersion, distance or
clustering had been computed on them.

*Question.* The 2025 study read object-mode uses as a tight "rod" and abstract uses as a spread
"cap". Is that so once anisotropy and topic are controlled?

*Measure.* Per word and model, embeddings centred on the word's mean and divided by the pooled
standard deviation of each dimension (anisotropy correction). Dispersion of a class = mean pairwise
cosine distance within it. Topic: each sentence is embedded by mpnet with the target noun replaced
by "something", and each word's sentences are split into 8 topic clusters (k-means). The
topic-controlled dispersion of a class is the mean pairwise distance within (cluster, class),
averaged over clusters with at least 5 sentences of each class, weighted by the number of pairs.
The ratio r = dispersion(A-gen) / dispersion(O) is computed raw and topic-controlled.

*Prediction (rod/cap).* Topic-controlled r > 1 in more than half the twenty words, one-sided
binomial test α = 0.01, for each primary model (BERT; decoder at the word). Verdicts per model:
*rod and cap* if it holds; *no rod* if r < 1 in more than half the words at α = 0.01; otherwise
*no difference*. The raw ratio is reported alongside to show how much topic accounts for.
