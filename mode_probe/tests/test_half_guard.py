import numpy as np
import pandas as pd
import pytest

from mode_probe import probe as P
from mode_probe.common import HalfBTuningError, assert_no_half_b, select_rows


def _df():
    rows = []
    for i in range(6):
        for j in range(30):
            rows.append({"word": f"w{i}", "half": "A" if i < 3 else "B",
                         "label": "O" if j % 2 else "A-gen"})
    return pd.DataFrame(rows)


def test_guard_raises_on_half_b_when_tuning():
    df = _df()
    with pytest.raises(HalfBTuningError):
        select_rows(df, halves=("A", "B"), tuning=True)
    with pytest.raises(HalfBTuningError):
        select_rows(df, halves=("B",), tuning=True)
    with pytest.raises(HalfBTuningError):
        assert_no_half_b(df)
    assert len(select_rows(df, halves=("A", "B"), tuning=False)) == len(df)


def test_selection_never_reads_half_b():
    """Half-B embeddings are NaN: if selection touched them, fitting would fail."""
    df = _df()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(len(df), 8))
    X[(df.half == "B").to_numpy()] = np.nan
    res = P.select_hyperparams(df, {"L1": X, "L2": X * 2}, ["L1", "L2"])
    assert res["layer"] in ("L1", "L2")
    assert np.isfinite(res["grid"]["auc"]).all()
