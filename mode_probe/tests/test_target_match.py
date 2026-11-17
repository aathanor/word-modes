from mode_probe.common import target_spans
from mode_probe.harvest import keep_sentence


def test_forms():
    assert len(target_spans("Two knives lay on the table.", "knife")) == 1
    assert len(target_spans("The tomatoes were ripe.", "tomato")) == 1
    assert target_spans("A cat-like grace.", "cat") == []
    assert target_spans("The catalogue was long.", "cat") == []


def test_keep_one_occurrence():
    assert keep_sentence("The cat sat quietly on the warm mat.", "cat")
    assert not keep_sentence("The cat chased another cat around the yard.", "cat")
    assert not keep_sentence("Cat", "cat")


def test_drop_mid_sentence_capitals():
    assert not keep_sentence("They sailed up the Hudson River in the spring of that year.", "river")
    assert keep_sentence("River levels rose sharply after the long storm.", "river")
    assert keep_sentence("\"Bridges were down all along the valley,\" he said.", "bridge")
