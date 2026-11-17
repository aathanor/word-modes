from mode_probe.common import all_words
from mode_probe.harvest import keep_sentence
from mode_probe.quads import CELLS, quad_frame


def test_quads_complete_and_clean():
    df = quad_frame()
    assert set(df["word"]) == set(all_words())
    assert (df.groupby("word")["qid"].nunique() == 5).all()
    assert (df.groupby("qid")["cell"].apply(lambda c: tuple(c) == CELLS)).all()
    assert not df["sentence"].duplicated().any()
    assert all(keep_sentence(s, w) for s, w in zip(df["sentence"], df["word"]))
