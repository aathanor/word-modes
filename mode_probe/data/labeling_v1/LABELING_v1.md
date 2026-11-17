# Labeling criterion

This file is the full instruction for every labeler, human or model. It was fixed
before any embedding was computed.

Label `O` if the sentence reports something about one specific instance of the target noun on an occasion: a definite or indefinite particular, an event or state located in time. Label `A` otherwise, with sub-label `A-gen` (literal general predication about the kind), `A-idiom` (fixed expression or figure whose meaning does not involve the thing), `A-def` (states what the thing is: "a/the X is (a) Y").

Decision tests, applied in order: (1) Could the sentence be true of exactly one X on one day, and is that its point? → `O`. (2) Is X inside a fixed expression or figure? → `A-idiom`. (3) Is it of the form "a/the X is (a) Y" stating what X is? → `A-def`. (4) Otherwise → `A-gen`. Label `X` (exclude) for sentences where the target word is a proper name part, a different sense (the Unix `cat`), or undecidable.

Allowed labels: `O`, `A-gen`, `A-idiom`, `A-def`, `X`.
