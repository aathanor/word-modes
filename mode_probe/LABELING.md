# Labeling criterion

This file is the full instruction for every labeler, human or model. It was fixed
before any embedding was computed.

Label `O` if the sentence reports something about one specific instance of the target noun on an occasion: a definite or indefinite particular, an event or state located in time. Label `A` otherwise, with sub-label `A-gen` (literal general predication about the kind), `A-idiom` (fixed expression or figure whose meaning does not involve the thing), `A-def` (states what the thing is: "a/the X is (a) Y").

Decision tests, applied in order: (1) Could the sentence be true of exactly one X on one day, and is that its point? → `O`. (2) Is X inside a fixed expression or figure? → `A-idiom`. (3) Is it of the form "a/the X is (a) Y" stating what X is? → `A-def`. (4) Otherwise → `A-gen`. Label `X` (exclude) for sentences where the target word is a proper name part, a different sense (the Unix `cat`), or undecidable.

Allowed labels: `O`, `A-gen`, `A-idiom`, `A-def`, `X`.

## Clarifications (version 2, 16 November 2026)

Added after the first agreement check between the two labelers (κ = 0.55) showed
cases the criterion above does not settle. The criterion itself is unchanged.

- **Modifier or material.** If the target noun only modifies another noun ("stone statues", "the cow shed", "apple pie", "tomato sauce", "mountain ranges") or names a substance ("built of stone and brick"), so that no individual X is referred to, label `X`. If an individual X is still referred to ("four cat actors", "the cows' milk", "fossil cat bones of a jaguarundi"), apply the decision tests as usual.
- **Habits and procedures.** A sentence about what happens repeatedly or as a rule, even inside one place or story ("rescues were necessary every year at the river crossing", "the men climb up the trees and shake the branches"), fails test 1 (it is not about one X on one day) and is `A-gen` unless test 2 or 3 applies.
- **Similes and comparisons.** If X appears only as the image in a comparison ("as a cat watches for a mouse", "fell like a stone", "shaped like a knife blade"), it is inside a figure: `A-idiom`.
- **Other senses and parts of speech.** Label `X` when the word is a verb ("to hammer things through", "the plant bottles water", "the road forks"), a person ("the chair of the commission"), a company or title ("Apple", the song "Spoon"), or another sense: a fork in a road or river, the front fork of a bicycle or motorcycle, the hammer of a gun, the hammer throw, bridge the card game, the bridge of a song, a railroad called "the road", cavalry called "light horse", a wooden gymnastics horse.
- **Plural particulars.** Several particular Xs on an occasion ("the cats scratched me as I put them out") are `O`.
