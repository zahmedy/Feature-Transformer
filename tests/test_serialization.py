import joblib
import numpy as np
from pathlib import Path
from src.transformer import FeatureTransformer


def _to_dense(Z):
    return Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)

def test_joblib_roundtrip(make_df, tmp_path: Path):
    X, y = make_df(seed=6)

    ft = FeatureTransformer()
    ft.fit(X, y)

    Z_before = _to_dense(ft.transform(X))

    p = tmp_path / "ft.joblib"
    joblib.dump(ft, p)
    ft2 = joblib.load(p)

    Z_after = _to_dense(ft2.transform(X))
    np.testing.assert_allclose(Z_before, Z_after)
