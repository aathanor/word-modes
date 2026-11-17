"""Each minimal pair differs only in the predicate: same subject NP, determiner, number, tense."""
import pytest

from mode_probe.common import all_words, target_span
from mode_probe.grammar import features, nlp
from mode_probe.pairs import pairs_frame

PAIRS = pairs_frame()


def test_ten_pairs_per_word():
    counts = PAIRS.groupby("word").size()
    assert set(counts.index) == set(all_words())
    assert (counts == 10).all()


@pytest.mark.parametrize("i", range(len(PAIRS)))
def test_pair_surface(i):
    r = PAIRS.iloc[i]
    o, a = r.sentence_O, r.sentence_A
    # identical up to and including the target and the following verb token
    end_o, end_a = target_span(o, r.word)[1], target_span(a, r.word)[1]
    assert o[:end_o] == a[:end_a]
    to = [t.text for t in nlp()(o)]
    ta = [t.text for t in nlp()(a)]
    k = len(nlp()(o[:end_o]))
    assert to[:k + 1] == ta[:k + 1], (to, ta)
    fo, fa = features(r.word, o), features(r.word, a)
    for key in ("det", "plural", "tense"):
        assert fo[key] == fa[key], (key, fo, fa)
    assert to != ta
