import numpy as np

from mode_probe import probe as P
from mode_probe.embed import load_cache, save_cache


def test_cache_round_trip(tmp_path):
    sids = [f"s{i}" for i in range(7)]
    arrays = {"L6": np.random.default_rng(0).normal(size=(7, 5)).astype(np.float32),
              "final_L6": np.ones((7, 5), np.float32)}
    path = tmp_path / "x.npz"
    save_cache(path, sids, arrays)
    s2, a2 = load_cache(path)
    assert s2 == sids
    assert set(a2) == set(arrays)
    for k in arrays:
        np.testing.assert_array_equal(a2[k], arrays[k])


def test_probe_reproducible():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 30))
    y = (X[:, 0] + rng.normal(scale=0.5, size=120) > 0).astype(int)
    a = P.fit(X, y, 0.1).decision_function(X)
    b = P.fit(X, y, 0.1).decision_function(X)
    np.testing.assert_array_equal(a, b)
