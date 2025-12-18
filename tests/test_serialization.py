import joblib
import numpy as np
from pathlib import Path
from src.transformer import FeatureTransformer


def test_joblib_roundtrip(make_df, tmp_path: Path):
    X, y = make_df(seed=6)

    ft = FeatureTransformer()
    ft.fit(X, y)

    Z_before = ft.transform(X)

    p = tmp_path / "ft.joblib"
    joblib.dump(ft, p)
    ft_loaded = joblib.load(p)

    Z_after = ft_loaded.transform(X)
    np.testing.assert_allclose(Z_before, Z_after)