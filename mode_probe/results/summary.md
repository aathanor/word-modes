# Mode-of-use probe: summary

Generated 2026-11-17 08:54. The claim tested is that the embedding carries a mark of the mode of use: the text sediment carries a mark of the mode of use, and models trained on it pick that mark up. Nothing below says that a model has the distinction or understands anything.

**Data.** 5900 natural sentences, 20 words, 3000 Wikipedia and 2900 fiction; 200 minimal pairs.

**Labels.** Model labels (claude-opus-5-5): O 3023, A-gen 956, A-idiom 519, A-def 11, X 1391. Gold: 300 sentences. Cohen's κ (model vs gold, n=300) = 0.862; cross-family check, qwen3.8:27b vs gold: κ = 0.710. The two labelers agree on 80% of all 5,900 sentences (κ = 0.674).

**Choices made on Half A only** (layer and L2 strength by leave-one-word-out AUC inside Half A; Half B was not read):

| model | layer | C |
|---|---|---|
| BERT | L9 | 0.010 |
| Decoder | L21 | 0.001 |
| mpnet | S | 0.001 |
| Decoder, final-token readout (supplementary) | final_L21 | 0.001 |

Every number is given twice: *full* = model labels on all sentences; *gold* = the same probe scores evaluated only on the gold sentences with the gold labels. Probes are always trained on model labels. Gold labeler: Claude Opus 5.5 (claude-opus-5-5), blind to the model labels; no human labels.

## 1. Prominence (E1)

*Question.* Without labels, is the mode one of the main directions along which a word's contextual embeddings vary?

Rank = first principal component whose score separates O from A-gen with AUC ≥ 0.8 (sign-free). Pooled = all words, each centred on its own mean.

| model | pooled rank | its variance share | best pooled PC AUC | words with rank ≤ 3 | words with none in top 10 | gold pooled rank |
|---|---|---|---|---|---|---|
| BERT | n/a | n/a | 0.710 | 7/20 | 12/20 | n/a |
| Decoder | n/a | n/a | 0.723 | 3/20 | 16/20 | 3 |
| mpnet | n/a | n/a | 0.773 | 11/20 | 9/20 | n/a |
| Decoder, final-token readout (supplementary) | n/a | n/a | 0.744 | 11/20 | 9/20 | n/a |

*Verdict* (prominent: pooled rank ≤ 3; partly prominent: within the first 10; not prominent: none):

- BERT: **not prominent**
- Decoder: **not prominent**
- mpnet: **not prominent**
- Decoder, final-token readout (supplementary): **not prominent**

*What it does not show.* A mode-aligned component can be prominent because mode co-varies with something larger (genre, sentence type); rungs 3–5 test that. A non-prominent mode can still be linearly readable (rung 2): 'not prominent' says how large the mark is next to everything else that varies, not whether it is there.

## 2. Shared direction (E2)

*Question.* With labels, does a linear probe trained on some words transfer to words it never saw?

| model | A→B AUC [95% CI] | null 95th | B→A AUC [95% CI] | null 95th | LOWO median word AUC | words above own null | gold A→B / B→A | gold LOWO pooled |
|---|---|---|---|---|---|---|---|---|
| BERT | 0.912 [0.90, 0.92] | 0.649 | 0.929 [0.92, 0.94] | 0.642 | 0.932 | 20/20 | 0.933 / 0.947 (n=99/115) | 0.947 (n=214) |
| Decoder | 0.920 [0.91, 0.93] | 0.677 | 0.916 [0.90, 0.93] | 0.684 | 0.916 | 20/20 | 0.967 / 0.933 (n=99/115) | 0.953 (n=214) |
| mpnet | 0.924 [0.91, 0.94] | 0.675 | 0.926 [0.91, 0.94] | 0.696 | 0.915 | 20/20 | 0.948 / 0.945 (n=99/115) | 0.936 (n=214) |
| Decoder, final-token readout (supplementary) | 0.936 [0.92, 0.95] | 0.711 | 0.945 [0.93, 0.95] | 0.730 | 0.948 | 20/20 | 0.962 / 0.978 (n=99/115) | 0.965 (n=214) |

Null: training labels shuffled within word, 200 permutations. Per-word AUCs: `results/e2_transfer_by_word.csv`.

*Verdict* (supported: both directions above the null 95th percentile and median held-out-word AUC ≥ 0.7; mixed: above the null but weaker):

- BERT: **supported**
- Decoder: **supported**
- mpnet: **supported**
- Decoder, final-token readout (supplementary): **supported**

**Does it depend on the labeler?** The same probes (same layer and L2 strength) were refitted and retested on the cross-family labels, and on only the sentences both labelers put in the same class. The last column scores the main probes, trained on Claude's labels, against each label set.

| model | labels | n O | n A-gen | A→B AUC | B→A AUC | LOWO median word AUC | main probe vs these labels |
|---|---|---|---|---|---|---|---|
| BERT | Claude (main) | 3023 | 956 | 0.912 | 0.929 | 0.932 | 0.922 |
| BERT | Qwen | 3624 | 493 | 0.957 | 0.951 | 0.971 | 0.944 |
| BERT | both agree | 2911 | 398 | 0.976 | 0.967 | 0.995 | 0.970 |
| Decoder | Claude (main) | 3023 | 956 | 0.920 | 0.916 | 0.916 | 0.916 |
| Decoder | Qwen | 3624 | 493 | 0.958 | 0.944 | 0.958 | 0.949 |
| Decoder | both agree | 2911 | 398 | 0.976 | 0.957 | 0.979 | 0.972 |
| mpnet | Claude (main) | 3023 | 956 | 0.924 | 0.926 | 0.915 | 0.922 |
| mpnet | Qwen | 3624 | 493 | 0.935 | 0.948 | 0.952 | 0.936 |
| mpnet | both agree | 2911 | 398 | 0.966 | 0.963 | 0.983 | 0.967 |
| Decoder, final-token readout (supplementary) | Claude (main) | 3023 | 956 | 0.936 | 0.945 | 0.948 | 0.939 |
| Decoder, final-token readout (supplementary) | Qwen | 3624 | 493 | 0.944 | 0.958 | 0.962 | 0.946 |
| Decoder, final-token readout (supplementary) | both agree | 2911 | 398 | 0.973 | 0.977 | 0.989 | 0.977 |

The local model calls fewer sentences A-gen, keeping the clearer cases, and the agreed subset drops the cases the two labelers read differently, so higher AUCs there are expected. What the table shows is that the transfer does not hinge on one labeler's reading of the borderline sentences. Both labelers are language models; this does not replace human labels.

*What it does not show.* A shared direction may be shared because a shared confound (genre, grammar, concreteness) carries it. Rungs 3–5 test each.

## 3. Not genre (E3)

*Question.* Does the probe transfer from encyclopedic text to fiction and back?

Class balance (model labels): fiction: A-gen 260, O 1938; wikipedia: A-gen 696, O 1085.

| model | Wiki→Fiction | Fiction→Wiki | within Wiki (CV by source) | within Fiction (CV by source) | within Wiki, A↔B words | within Fiction, A↔B words | E2 A↔B (both genres) | gold W→F / F→W |
|---|---|---|---|---|---|---|---|---|
| BERT | 0.889 | 0.884 | 0.954 | 0.930 | 0.913 | 0.907 | 0.920 | 0.925 / 0.916 |
| Decoder | 0.861 | 0.848 | 0.919 | 0.902 | 0.897 | 0.894 | 0.918 | 0.898 / 0.929 |
| mpnet | 0.872 | 0.871 | 0.924 | 0.917 | 0.908 | 0.912 | 0.925 | 0.943 / 0.855 |
| Decoder, final-token readout (supplementary) | 0.870 | 0.891 | 0.951 | 0.922 | 0.941 | 0.914 | 0.940 | 0.935 / 0.873 |

*Verdict* (supported: both cross-genre AUCs ≥ 0.7 and each at most 0.1 below the within-genre baseline):

- BERT: **supported**
- Decoder: **supported**
- mpnet: **supported**
- Decoder, final-token readout (supplementary): **supported**

*What it does not show.* Cross-genre training uses all twenty words, so it is not also a cross-word test. Wikipedia and fiction differ in their O/A-gen balance; AUC is insensitive to that, the fitted probe is not. Legacy Wikipedia sentences have no article id, so the within-Wikipedia baseline may share articles across folds.

## 4. Not grammar (E4)

*Question.* Does the probe beat a surface-grammar classifier, and does it survive minimal pairs with identical surface form?

(a) Held-out-word (LOWO) scores. LRT: does the embedding score add to the grammar score in a logistic model of the label (df = 1)?

| model | grammar AUC | embedding AUC | words emb > gram | Wilcoxon p | LRT χ² (p) | embedding AUC on residual | gold gram / emb |
|---|---|---|---|---|---|---|---|
| BERT | 0.799 | 0.925 | 20/20 | <1e-4 | 1325.2 (p <1e-4) | 0.828 | 0.878 / 0.947 |
| Decoder | 0.799 | 0.920 | 20/20 | <1e-4 | 1200.7 (p <1e-4) | 0.813 | 0.878 / 0.953 |
| mpnet | 0.799 | 0.929 | 20/20 | <1e-4 | 1317.6 (p <1e-4) | 0.823 | 0.878 / 0.936 |
| Decoder, final-token readout (supplementary) | 0.799 | 0.943 | 20/20 | <1e-4 | 1579.3 (p <1e-4) | 0.848 | 0.878 / 0.965 |

(b) Minimal pairs (Half-A probe on Half-B pairs and back; score = probe(A) − probe(O); ties count as failures):

| model | positive | fraction | ties | binomial p (>0.5) | fraction, A-gen pairs | fraction, A-def pairs |
|---|---|---|---|---|---|---|
| BERT | 192/200 | 0.960 | 0 | <1e-4 | 0.956 | 0.975 |
| Decoder | 0/200 | 0.000 | 200 | 1 | 0.000 | 0.000 |
| mpnet | 194/200 | 0.970 | 0 | <1e-4 | 0.988 | 0.900 |
| Decoder, final-token readout (supplementary) | 200/200 | 1.000 | 0 | <1e-4 | 1.000 | 1.000 |

*Verdict* (supported: LRT p < 0.01, embedding AUC above grammar AUC, and minimal pairs binomial p < 0.01; mixed: one of the two):

- BERT: **supported**
- Decoder: **mixed**
- mpnet: **supported**
- Decoder, final-token readout (supplementary): **supported**

*What it does not show.* The grammar classifier uses only the listed surface features; a richer syntax model could absorb more. The minimal pairs are written, not natural. For the decoder's target-token readout both members of a pair share the prefix up to the target, so its states are identical and every pair is a tie: (b) cannot test that readout, and the supplementary final-token readout is reported for that reason.

## 5. Not concreteness (E5)

*Question.* Does the probe survive regressing out word concreteness?

Mean Brysbaert concreteness of the content words, target excluded. LOWO scores.

| model | Spearman ρ (score, concreteness) | concreteness alone AUC | AUC before | after (score resid.) | after (feature resid.) | drop | gold before → after |
|---|---|---|---|---|---|---|---|
| BERT | -0.120 | 0.559 | 0.925 | 0.922 | 0.920 | 0.005 | 0.947 → 0.946 |
| Decoder | -0.093 | 0.559 | 0.919 | 0.917 | 0.918 | 0.001 | 0.953 → 0.955 |
| mpnet | -0.104 | 0.559 | 0.929 | 0.927 | 0.926 | 0.003 | 0.936 → 0.936 |
| Decoder, final-token readout (supplementary) | -0.163 | 0.559 | 0.943 | 0.939 | 0.939 | 0.004 | 0.965 → 0.965 |

*Verdict* (supported: feature-residualized AUC drops by ≤ 0.05; mixed: ≤ 0.1):

- BERT: **supported**
- Decoder: **supported**
- mpnet: **supported**
- Decoder, final-token readout (supplementary): **supported**

*What it does not show.* Only a linear effect of mean context concreteness is removed; concreteness of particular words, or non-linear effects, could remain.

## 6. Predictions (E6)

*Question.* Do definitions land at the far abstract end, and where do idioms land?

Probe trained on O vs A-gen only (LOWO scores). Definitions and idioms are reported separately and never pooled.

| model | labels | n A-def | A-def beyond A-gen median | MW A-def > A-gen p | n A-idiom | idiom beyond A-gen median | MW idiom vs A-gen p | MW idiom vs O p | idiom placement |
|---|---|---|---|---|---|---|---|---|---|
| BERT | model labels | 11 | 0.545 | 0.29 | 519 | 0.135 | <1e-4 | <1e-4 | between O and A-gen |
| BERT | gold labels | 0 | n/a | n/a | 33 | 0.121 | <1e-4 | <1e-4 | between O and A-gen |
| Decoder | model labels | 11 | 0.455 | 0.86 | 519 | 0.148 | <1e-4 | <1e-4 | between O and A-gen |
| Decoder | gold labels | 0 | n/a | n/a | 33 | 0.121 | <1e-4 | <1e-4 | between O and A-gen |
| mpnet | model labels | 11 | 0.364 | 0.76 | 519 | 0.118 | <1e-4 | <1e-4 | between O and A-gen |
| mpnet | gold labels | 0 | n/a | n/a | 33 | 0.091 | <1e-4 | 0.0014 | between O and A-gen |
| Decoder, final-token readout (supplementary) | model labels | 11 | 0.909 | 0.0028 | 519 | 0.077 | <1e-4 | <1e-4 | between O and A-gen |
| Decoder, final-token readout (supplementary) | gold labels | 0 | n/a | n/a | 33 | 0.061 | <1e-4 | 0.00014 | between O and A-gen |

*Verdict* (definitions supported: Mann–Whitney p < 0.05 and more than half beyond the A-gen median):

- BERT: definitions **not supported (underpowered)**; idioms **between O and A-gen**
- Decoder: definitions **not supported (underpowered)**; idioms **between O and A-gen**
- mpnet: definitions **not supported (underpowered)**; idioms **between O and A-gen**
- Decoder, final-token readout (supplementary): definitions **supported (underpowered)**; idioms **between O and A-gen**

*What it does not show.* Natural definitions are mostly Wikipedia lead sentences, so their position also reflects that register. Idioms are rare in this sample.

## 7. Same direction across models (E7)

*Question.* Do an encoder, a sentence model and a decoder order the same sentences the same way?

Half-B sentences, scores from the Half-A probe of each model.

| pair | Spearman ρ | within O | within A-gen | gold ρ |
|---|---|---|---|---|
| bert vs decoder | 0.773 | 0.630 | 0.608 | 0.828 (n=94) |
| bert vs mpnet | 0.767 | 0.620 | 0.558 | 0.799 (n=94) |
| decoder vs mpnet | 0.748 | 0.586 | 0.487 | 0.783 (n=94) |

*Verdict* (supported: every pair ρ ≥ 0.5; mixed: ≥ 0.3): **supported**

*What it does not show.* Agreement on the overall ordering is driven largely by the O/A-gen split itself; the within-label correlations show whether the models also agree inside a class.

## 8. Not English syntax (E8, optional)

Not run (no `data/romanian_labeled.csv`, or `--romanian` not set).

## Where the argument stops

- No human gold set: the gold labels come from a second model (Claude Opus 5.5 (claude-opus-5-5), blind to the model labels; no human labels), so κ measures agreement between two models, not between a model and a person.
- E1 (BERT): not prominent — the mode is not among the top three unlabeled directions of variation pooled over words; see E2 for whether it is there.
- E1 (Decoder): not prominent — the mode is not among the top three unlabeled directions of variation pooled over words; see E2 for whether it is there.
- E1 (mpnet): not prominent — the mode is not among the top three unlabeled directions of variation pooled over words; see E2 for whether it is there.
- E1 (Decoder, final-token readout (supplementary)): not prominent — the mode is not among the top three unlabeled directions of variation pooled over words; see E2 for whether it is there.
- E4 (Decoder): mixed.
- E6 definitions (BERT): not supported (underpowered).
- E6 definitions (Decoder): not supported (underpowered).
- E6 definitions (mpnet): not supported (underpowered).
