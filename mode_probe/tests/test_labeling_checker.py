"""The checker flags labels that violate the decision tests of LABELING.md."""
import pytest

from mode_probe.grammar import check_label

# word, sentence, label, expected flag (None = the label passes the checker)
CASES = [
    ("cat", "The cat jumped off the table and knocked the cup over.", "O", None),
    ("horse", "A horse stood by the gate in the rain.", "O", None),
    ("dog", "The dogs barked all night outside my window.", "O", None),
    ("knife", "She picked up the knife and cut the bread.", "O", None),
    ("river", "We crossed the river at dawn on the third day.", "O", None),
    ("cat", "Cats hunt at night.", "A-gen", None),
    ("tree", "Trees lose their leaves in autumn.", "A-gen", None),
    ("cat", "Curiosity killed the cat.", "A-idiom", None),
    ("horse", "Don't look a gift horse in the mouth.", "A-idiom", None),
    ("cat", "A cat is a small domesticated carnivore.", "A-def", None),
    ("hammer", "The hammer is a tool for driving nails.", "A-def", None),
    ("apple", "An apple is the fruit of the apple tree.", "A-def", None),
    # violations
    ("cat", "Cats hunt mice at night.", "O", "O_bare_plural_generic_present"),
    ("dog", "Dogs bark at strangers.", "O", "O_bare_plural_generic_present"),
    ("bottle", "Bottles break when they fall.", "O", "O_bare_plural_generic_present"),
    ("horse", "Horses need several acres of pasture.", "O", "O_bare_plural_generic_present"),
    ("cat", "Cats were domesticated in the Near East.", "A-def", "A-def_not_X_is_Y"),
    ("tree", "Trees lose their leaves in autumn.", "A-def", "A-def_not_X_is_Y"),
    ("river", "People have always settled near rivers.", "A-def", "A-def_not_X_is_Y"),
    ("chair", "A chair is a seat with a back for one person.", "A-gen", "A-gen_has_X_is_a_Y_form"),
]


def test_twenty_cases():
    assert len(CASES) == 20


@pytest.mark.parametrize("word,sentence,label,flag", CASES)
def test_checker(word, sentence, label, flag):
    flags = check_label(word, sentence, label)
    if flag is None:
        assert flags == [], flags
    else:
        assert flag in flags, flags
