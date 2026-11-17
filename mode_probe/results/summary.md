# Mode-of-use probe: summary

Generated 2026-11-17 15:34. The claim tested is that the embedding carries a mark of the mode of use: the text sediment carries a mark of the mode of use, and models trained on it pick that mark up. Nothing below says that a model has the distinction or understands anything.

**Data.** 5900 natural sentences, 20 words, 3000 Wikipedia and 2900 fiction; 200 minimal pairs.

**Labels.** Model labels (claude-opus-5-5): O 3016, A-gen 968, A-idiom 521, A-def 12, X 1383. Gold: 300 sentences. Cohen's κ (model vs gold, n=300) = 0.862; cross-family check, qwen3.8:27b vs gold: κ = 0.710. The two labelers agree on 80% of all 5,900 sentences (κ = 0.674).

**Choices made on Half A only** (layer and L2 strength by leave-one-word-out AUC inside Half A; Half B was not read):

| model | layer | C |
|---|---|---|
| BERT | L9 | 0.010 |
| Decoder | after_L21 | 0.001 |
| mpnet | S | 0.010 |
| Decoder, at the word (supplementary) | L21 | 0.001 |

Every number is given twice: *full* = model labels on all sentences; *gold* = the same probe scores evaluated only on the gold sentences with the gold labels. Probes are always trained on model labels. Gold labeler: Claude Opus 5.5 (claude-opus-5-5), blind to the model labels; no human labels.

## 1. Prominence (E1)

*Question.* Without labels, is the mode one of the main directions along which a word's contextual embeddings vary?

Rank = first principal component whose score separates O from A-gen with AUC ≥ 0.8 (sign-free). Pooled = all words, each centred on its own mean.

| model | pooled rank | its variance share | best pooled PC AUC | words with rank ≤ 3 | words with none in top 10 | gold pooled rank |
|---|---|---|---|---|---|---|
| BERT | n/a | n/a | 0.710 | 8/20 | 11/20 | n/a |
| Decoder | n/a | n/a | 0.764 | 12/20 | 7/20 | n/a |
| mpnet | n/a | n/a | 0.773 | 11/20 | 9/20 | n/a |
| Decoder, at the word (supplementary) | n/a | n/a | 0.725 | 3/20 | 16/20 | n/a |

*Verdict* (prominent: pooled rank ≤ 3; partly prominent: within the first 10; not prominent: none):

- BERT: **not prominent**
- Decoder: **not prominent**
- mpnet: **not prominent**
- Decoder, at the word (supplementary): **not prominent**

*What it does not show.* A mode-aligned component can be prominent because mode co-varies with something larger (genre, sentence type); rungs 3–5 test that. A non-prominent mode can still be linearly readable (rung 2): 'not prominent' says how large the mark is next to everything else that varies, not whether it is there.

**Two variants added after the first results.** Same bar (AUC ≥ 0.8), no verdicts. *E1b, known variation removed:* each word centred on its own mean, then genre, sentence length and the surface-grammar features regressed out, then PCA pooled over words. *E1c, directions shared across words:* each word centred and scaled to equal total variance, so that the leading directions are those large in all words at once.

| model | E1b rank | E1b best AUC | E1c rank | E1c best direction |
|---|---|---|---|---|
| BERT | none | 0.616 (PC5) | none | 0.727 (C2, in the top ten of 20/20 words) |
| Decoder | none | 0.616 (PC3) | none | 0.766 (C2, in the top ten of 20/20 words) |
| mpnet | none | 0.591 (PC2) | none | 0.773 (C1, in the top ten of 20/20 words) |
| Decoder, at the word (supplementary) | none | 0.632 (PC9) | none | 0.751 (C3, in the top ten of 20/20 words) |

**Model size (supplementary).** One family, Qwen3, at four sizes. For each, the readout and layer are chosen on Half A as for the main decoder. If training on more text, or compressing it with more capacity, made the mark more prominent, the best unlabeled direction would climb with size.

| model | readout chosen on Half A | E2 A↔B AUC | E1 best AUC | E1b best AUC | E1c best AUC |
|---|---|---|---|---|---|
| Qwen3-0.6B | final_L21 | 0.932 | 0.743 | 0.681 | 0.746 |
| Qwen3-1.7B | after_L21 | 0.948 | 0.764 | 0.616 | 0.766 |
| Qwen3-4B | after_L18 | 0.951 | 0.773 | 0.684 | 0.773 |
| Qwen3-8B | after_L18 | 0.956 | 0.778 | 0.630 | 0.778 |

## 2. Shared direction (E2)

*Question.* With labels, does a linear probe trained on some words transfer to words it never saw?

| model | A→B AUC [95% CI] | null 95th | B→A AUC [95% CI] | null 95th | LOWO median word AUC | words above own null | gold A→B / B→A | gold LOWO pooled |
|---|---|---|---|---|---|---|---|---|
| BERT | 0.910 [0.90, 0.92] | 0.658 | 0.931 [0.92, 0.94] | 0.658 | 0.934 | 20/20 | 0.931 / 0.947 (n=99/115) | 0.949 (n=214) |
| Decoder | 0.945 [0.93, 0.96] | 0.698 | 0.950 [0.94, 0.96] | 0.712 | 0.944 | 20/20 | 0.965 / 0.963 (n=99/115) | 0.966 (n=214) |
| mpnet | 0.915 [0.90, 0.93] | 0.609 | 0.927 [0.91, 0.94] | 0.626 | 0.911 | 20/20 | 0.924 / 0.943 (n=99/115) | 0.925 (n=214) |
| Decoder, at the word (supplementary) | 0.920 [0.91, 0.93] | 0.678 | 0.918 [0.90, 0.93] | 0.691 | 0.915 | 20/20 | 0.965 / 0.933 (n=99/115) | 0.952 (n=214) |

Null: training labels shuffled within word, 200 permutations. Per-word AUCs: `results/e2_transfer_by_word.csv`.

**Without a fitted probe.** The difference between the mean A-gen and the mean O embedding of the training words (standardised), used directly as the direction. It has no regularisation or fitting to tune, so it shows the result does not hang on the probe.

| model | A→B AUC | B→A AUC | cosine with the probe's direction |
|---|---|---|---|
| BERT | 0.860 | 0.839 | 0.51 |
| Decoder | 0.889 | 0.881 | 0.75 |
| mpnet | 0.893 | 0.888 | 0.57 |
| Decoder, at the word (supplementary) | 0.846 | 0.833 | 0.60 |

*Verdict* (supported: both directions above the null 95th percentile and median held-out-word AUC ≥ 0.7; mixed: above the null but weaker):

- BERT: **supported**
- Decoder: **supported**
- mpnet: **supported**
- Decoder, at the word (supplementary): **supported**

**Does it depend on the labeler?** The same probes (same layer and L2 strength) were refitted and retested on the cross-family labels, and on only the sentences both labelers put in the same class. The last column scores the main probes, trained on Claude's labels, against each label set.

| model | labels | n O | n A-gen | A→B AUC | B→A AUC | LOWO median word AUC | main probe vs these labels |
|---|---|---|---|---|---|---|---|
| BERT | Claude (main) | 3016 | 968 | 0.910 | 0.931 | 0.934 | 0.922 |
| BERT | Qwen | 3616 | 496 | 0.959 | 0.955 | 0.975 | 0.945 |
| BERT | both agree | 2903 | 405 | 0.977 | 0.970 | 0.995 | 0.970 |
| Decoder | Claude (main) | 3016 | 968 | 0.945 | 0.950 | 0.944 | 0.947 |
| Decoder | Qwen | 3616 | 496 | 0.960 | 0.959 | 0.967 | 0.957 |
| Decoder | both agree | 2903 | 405 | 0.980 | 0.974 | 0.985 | 0.983 |
| mpnet | Claude (main) | 3016 | 968 | 0.915 | 0.927 | 0.911 | 0.920 |
| mpnet | Qwen | 3616 | 496 | 0.939 | 0.946 | 0.948 | 0.925 |
| mpnet | both agree | 2903 | 405 | 0.969 | 0.968 | 0.983 | 0.963 |
| Decoder, at the word (supplementary) | Claude (main) | 3016 | 968 | 0.920 | 0.918 | 0.915 | 0.917 |
| Decoder, at the word (supplementary) | Qwen | 3616 | 496 | 0.960 | 0.946 | 0.960 | 0.949 |
| Decoder, at the word (supplementary) | both agree | 2903 | 405 | 0.977 | 0.960 | 0.978 | 0.973 |

The local model calls fewer sentences A-gen, keeping the clearer cases, and the agreed subset drops the cases the two labelers read differently, so higher AUCs there are expected. What the table shows is that the transfer does not hinge on one labeler's reading of the borderline sentences. Both labelers are language models; this does not replace human labels.

*What it does not show.* A shared direction may be shared because a shared confound (genre, grammar, concreteness) carries it. Rungs 3–5 test each.

## 3. Not genre (E3)

*Question.* Does the probe transfer from encyclopedic text to fiction and back?

Class balance (model labels): fiction: A-gen 260, O 1938; wikipedia: A-gen 708, O 1078.

| model | Wiki→Fiction | Fiction→Wiki | within Wiki (CV by source) | within Fiction (CV by source) | within Wiki, A↔B words | within Fiction, A↔B words | E2 A↔B (both genres) | gold W→F / F→W |
|---|---|---|---|---|---|---|---|---|
| BERT | 0.894 | 0.885 | 0.948 | 0.930 | 0.913 | 0.907 | 0.920 | 0.931 / 0.916 |
| Decoder | 0.897 | 0.902 | 0.952 | 0.931 | 0.944 | 0.926 | 0.948 | 0.933 / 0.912 |
| mpnet | 0.879 | 0.861 | 0.928 | 0.915 | 0.896 | 0.904 | 0.921 | 0.954 / 0.845 |
| Decoder, at the word (supplementary) | 0.861 | 0.849 | 0.915 | 0.902 | 0.899 | 0.894 | 0.919 | 0.895 / 0.929 |

*Verdict* (supported: both cross-genre AUCs ≥ 0.7 and each at most 0.1 below the within-genre baseline):

- BERT: **supported**
- Decoder: **supported**
- mpnet: **supported**
- Decoder, at the word (supplementary): **supported**

*What it does not show.* Cross-genre training uses all twenty words, so it is not also a cross-word test. Wikipedia and fiction differ in their O/A-gen balance; AUC is insensitive to that, the fitted probe is not. The within-genre baselines keep each article or book inside one fold.

## 4. Not reducible to surface grammar (E4)

*Question.* Does the probe beat a surface-grammar classifier, and does it survive minimal pairs with identical surface form?

(a) Held-out-word (LOWO) scores. LRT: does the embedding score add to the grammar score in a logistic model of the label (df = 1)?

| model | grammar AUC | embedding AUC | words emb > gram | Wilcoxon p | LRT χ² (p) | embedding AUC on residual | gold gram / emb |
|---|---|---|---|---|---|---|---|
| BERT | 0.801 | 0.927 | 20/20 | <1e-4 | 1338.1 (p <1e-4) | 0.829 | 0.877 / 0.949 |
| Decoder | 0.801 | 0.949 | 20/20 | <1e-4 | 1633.3 (p <1e-4) | 0.855 | 0.877 / 0.966 |
| mpnet | 0.801 | 0.927 | 20/20 | <1e-4 | 1319.8 (p <1e-4) | 0.823 | 0.877 / 0.925 |
| Decoder, at the word (supplementary) | 0.801 | 0.922 | 20/20 | <1e-4 | 1214.4 (p <1e-4) | 0.814 | 0.877 / 0.952 |

(b) Minimal pairs (Half-A probe on Half-B pairs and back; score = probe(A) − probe(O); ties count as failures):

| model | positive | fraction | ties | binomial p (>0.5) | fraction, A-gen pairs | fraction, A-def pairs |
|---|---|---|---|---|---|---|
| BERT | 192/200 | 0.960 | 0 | <1e-4 | 0.956 | 0.975 |
| Decoder | 200/200 | 1.000 | 0 | <1e-4 | 1.000 | 1.000 |
| mpnet | 189/200 | 0.945 | 0 | <1e-4 | 0.969 | 0.850 |
| Decoder, at the word (supplementary) | 0/200 | 0.000 | 200 | 1 | 0.000 | 0.000 |

*Verdict* (supported: LRT p < 0.01, embedding AUC above grammar AUC, and minimal pairs binomial p < 0.01; mixed: one of the two):

- BERT: **supported**
- Decoder: **supported**
- mpnet: **supported**
- Decoder, at the word (supplementary): **supported (minimal pairs not applicable)**

*What it does not show.* The grammar classifier uses only the listed surface features; a richer syntax model could absorb more. The minimal pairs are written, not natural. For the decoder's target-token readout both members of a pair share the prefix up to the target, so its states are identical and every pair is a tie: (b) cannot test that readout, and the supplementary final-token readout is reported for that reason.

## 5. Not concreteness (E5)

*Question.* Does the probe survive regressing out word concreteness?

Mean Brysbaert concreteness of the content words, target excluded. LOWO scores.

| model | Spearman ρ (score, concreteness) | concreteness alone AUC | AUC before | after (score resid.) | after (feature resid.) | drop | gold before → after |
|---|---|---|---|---|---|---|---|
| BERT | -0.115 | 0.555 | 0.927 | 0.925 | 0.923 | 0.004 | 0.949 → 0.949 |
| Decoder | -0.127 | 0.555 | 0.949 | 0.946 | 0.947 | 0.002 | 0.966 → 0.965 |
| mpnet | -0.116 | 0.555 | 0.927 | 0.925 | 0.925 | 0.003 | 0.925 → 0.925 |
| Decoder, at the word (supplementary) | -0.091 | 0.555 | 0.922 | 0.920 | 0.920 | 0.001 | 0.952 → 0.953 |

*Verdict* (supported: feature-residualized AUC drops by ≤ 0.05; mixed: ≤ 0.1):

- BERT: **supported**
- Decoder: **supported**
- mpnet: **supported**
- Decoder, at the word (supplementary): **supported**

*What it does not show.* Only a linear effect of mean context concreteness is removed; concreteness of particular words, or non-linear effects, could remain.

## 6. Predictions (E6)

*Question.* Do definitions land at the far abstract end, and where do idioms land?

Probe trained on O vs A-gen only (LOWO scores). Definitions and idioms are reported separately and never pooled.

| model | labels | n A-def | A-def beyond A-gen median | MW A-def > A-gen p | n A-idiom | idiom beyond A-gen median | MW idiom vs A-gen p | MW idiom vs O p | idiom placement |
|---|---|---|---|---|---|---|---|---|---|
| BERT | model labels | 12 | 0.583 | 0.29 | 521 | 0.125 | <1e-4 | <1e-4 | between O and A-gen |
| BERT | gold labels | 0 | n/a | n/a | 33 | 0.121 | <1e-4 | <1e-4 | between O and A-gen |
| Decoder | model labels | 12 | 0.667 | 0.14 | 521 | 0.081 | <1e-4 | <1e-4 | between O and A-gen |
| Decoder | gold labels | 0 | n/a | n/a | 33 | 0.091 | <1e-4 | <1e-4 | between O and A-gen |
| mpnet | model labels | 12 | 0.250 | 0.87 | 521 | 0.150 | <1e-4 | <1e-4 | between O and A-gen |
| mpnet | gold labels | 0 | n/a | n/a | 33 | 0.152 | <1e-4 | 0.0023 | between O and A-gen |
| Decoder, at the word (supplementary) | model labels | 12 | 0.500 | 0.86 | 521 | 0.142 | <1e-4 | <1e-4 | between O and A-gen |
| Decoder, at the word (supplementary) | gold labels | 0 | n/a | n/a | 33 | 0.091 | <1e-4 | <1e-4 | between O and A-gen |

*Verdict* (definitions supported: Mann–Whitney p < 0.05 and more than half beyond the A-gen median):

- BERT: definitions **not supported (underpowered)**; idioms **between O and A-gen**
- Decoder: definitions **not supported (underpowered)**; idioms **between O and A-gen**
- mpnet: definitions **not supported (underpowered)**; idioms **between O and A-gen**
- Decoder, at the word (supplementary): definitions **not supported (underpowered)**; idioms **between O and A-gen**

*What it does not show.* Natural definitions are mostly Wikipedia lead sentences, so their position also reflects that register. Idioms are rare in this sample.

## 7. Same direction across models (E7)

*Question.* Do an encoder, a sentence model and a decoder order the same sentences the same way?

Half-B sentences, scores from the Half-A probe of each model.

| pair | Spearman ρ | within O | within A-gen | gold ρ |
|---|---|---|---|---|
| bert vs decoder | 0.778 | 0.624 | 0.598 | 0.788 (n=94) |
| bert vs mpnet | 0.740 | 0.576 | 0.530 | 0.789 (n=94) |
| decoder vs mpnet | 0.791 | 0.621 | 0.652 | 0.779 (n=94) |

*Verdict* (supported: every pair ρ ≥ 0.5; mixed: ≥ 0.3): **supported**

*What it does not show.* Agreement on the overall ordering is driven largely by the O/A-gen split itself; the within-label correlations show whether the models also agree inside a class.

## 8. Not English syntax (E8, optional)

Not run (no `data/romanian_labeled.csv`, or `--romanian` not set).

## Where the argument stops

- No human gold set: the gold labels come from a second model (Claude Opus 5.5 (claude-opus-5-5), blind to the model labels; no human labels), so κ measures agreement between two models, not between a model and a person.
- E1 (BERT): not prominent — the mode is not among the top three unlabeled directions of variation pooled over words; see E2 for whether it is there.
- E1 (Decoder): not prominent — the mode is not among the top three unlabeled directions of variation pooled over words; see E2 for whether it is there.
- E1 (mpnet): not prominent — the mode is not among the top three unlabeled directions of variation pooled over words; see E2 for whether it is there.
- E1 (Decoder, at the word (supplementary)): not prominent — the mode is not among the top three unlabeled directions of variation pooled over words; see E2 for whether it is there.
- E6 definitions (BERT): not supported (underpowered).
- E6 definitions (Decoder): not supported (underpowered).
- E6 definitions (mpnet): not supported (underpowered).
- E6 definitions (Decoder, at the word (supplementary)): not supported (underpowered).
