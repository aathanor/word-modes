import numpy as np
import pandas as pd

from mode_probe import probe as P


def test_null_mean_near_half():
    rng = np.random.default_rng(3)
    words = [f"w{i}" for i in range(10)]
    rows = []
    for i, w in enumerate(words):
        for j in range(40):
            rows.append({"word": w, "half": "A" if i < 5 else "B",
                         "label": "O" if j % 2 else "A-gen"})
    df = pd.DataFrame(rows)
    X = rng.normal(size=(len(df), 20))
    X[:, 0] += np.where(df["label"] == "A-gen", 1.0, 0.0)      # real signal; null must ignore it
    null, per_word = P.permutation_null(df, X, (df.half == "A").to_numpy(),
                                        (df.half == "B").to_numpy(), C=0.1, n=200)
    assert abs(np.mean(null) - 0.5) <= 0.03
    assert set(per_word) == set(words[5:])
