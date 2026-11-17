# Labeling criterion, version 3

This file is the full instruction for every labeler, human or model, in the version-3
experiments (E9, E10). It was fixed before any sentence was labeled under it. Version 2
(`LABELING.md`) stays the criterion for E1–E7.

Version 3 splits the old general class in two directions, so that two accounts of the
difference can be told apart. One account (the Pattern Constellations Framework) says what
matters is **reference**: is the word used of a particular thing, or of the kind? The other
(the linguistics of genericity) says what matters is **occasion**: does the sentence report
one episode, or what happens repeatedly, as a rule, or always? Most sentences answer both
questions the same way. The labels below record each answer separately.

## Step 1: exclusions and figures

Answer these first, in order. If one applies, that is the label.

- **X (exclude).** The target noun only modifies another noun or names a substance, so that
  no individual and no kind of X is referred to ("stone wall", "the cow shed", "apple pie",
  "built of stone and brick"). Or the word is a verb ("to hammer", "the road forks"), a
  person ("the chair of the commission"), a name or title ("Apple", the song "Spoon"), or
  another sense (a fork in a road or river, the front fork of a bicycle, the hammer of a gun,
  the hammer throw, bridge the card game, the bridge of a song, a railroad called "the road",
  a gymnastics horse, a binary tree). Or the sentence cannot be decided.
- **A-idiom.** X sits inside a fixed expression whose meaning does not involve the thing
  ("let the cat out of the bag", "a dog's life", "hit the road", "the apple of his eye").
- **A-simile.** X appears only as the image in a comparison: "like", "as … as", "as if"
  ("fell like a stone", "as a cat watches for a mouse", "shaped like a knife blade"). The
  sentence is not about any X; it borrows X to describe something else.
- **A-def.** The sentence states what X is, in the form "a/an/the X is (a) Y" or "X(s) are Y"
  given as a definition ("A spoon is a utensil consisting of a small shallow bowl …").

## Step 2: reference

Is the target noun phrase about **particular** things, or about the **kind**?

- **Particular (P).** One or more specific individuals: "the cat on the mat", "my cat", "a cat
  jumped onto the table", "the three cows in the field", "spoons were found in the tomb"
  (some specific spoons), "the bridge at Arnhem". An individual identified by name or
  description is particular ("the bridge over the Tagus", "the mountain we climbed").
- **Kind (K).** The kind as such, or any or a typical member: "cats", "the cat" meaning the
  species, "a cat is …" said of any cat, "every horse", "the domestic cow", "tomatoes" in
  "tomatoes were introduced to Europe".

Test: could you point to the X (or Xs) the sentence is about, at least in principle? If yes,
**P**. If the sentence would stay true whichever members of the kind existed, **K**.

## Step 3: occasion

Does the sentence report **one occasion**, or what holds **in general**?

- **Episodic (ep).** One event, or a state at one time: "the cat jumped", "the bridge
  collapsed in 1940", "the lamp was lit" (in a narrated scene), "tomatoes reached Italy in
  the 1540s", "the horse was domesticated around 3500 BC" (one historical episode, however
  long).
- **General (gen).** What happens repeatedly, habitually or as a rule, or a standing
  property stated as characteristic: "my cat always sleeps by the fire", "every spring the
  river floods the meadow", "the bridge carries six lanes of traffic", "cats hunt at night",
  "horses are herbivores".

Test: does the sentence locate its point at a time (an episode, even a long one), or does it
say how things go or are, at any time? Descriptions inside a narrated scene ("the room was
cold and the lamp smoked") are episodic: they belong to that scene's time.

## The labels

| label | reference | occasion | example |
|---|---|---|---|
| **P-ep** | particular | one occasion | The cat jumped off the table and knocked the cup over. |
| **P-gen** | particular | in general | My cat always sleeps by the fire. |
| **K-ep** | kind | one occasion | Cats reached Australia in the seventeenth century. |
| **K-gen** | kind | in general | Cats hunt at night. |
| **A-idiom** | | | He let the cat out of the bag. |
| **A-simile** | | | She moved as quietly as a cat. |
| **A-def** | | | A cat is a small domesticated carnivorous mammal. |
| **X** | | | the cat flap; Cat Stevens; a binary tree |

Allowed labels: `P-ep`, `P-gen`, `K-ep`, `K-gen`, `A-idiom`, `A-simile`, `A-def`, `X`.

## Correspondence with version 2

`O` in version 2 is mostly `P-ep`. `A-gen` in version 2 is mostly `K-gen`, and also holds
the habits and standing properties of particulars (`P-gen`) and episodes about kinds
(`K-ep`). Version 2's `A-idiom` is split into `A-idiom` and `A-simile`.
