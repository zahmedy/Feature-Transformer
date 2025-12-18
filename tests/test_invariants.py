import numpy as np
import pandas as pd
from src.transformer import FeatureTransformer

def _to_dense(Z):
    return Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)

def test_output_shape_is_stable(make_df):
    X, y = make_df(seed=1)

    ft = FeatureTransformer()
    Z1 = ft.fit_transform(X, y)

    # shuffle rows
    X2 = X.sample(frac=1.0, random_state=42).reset_index(drop=True)
    Z2 = ft.transform(X2)

    assert Z1.shape[1] == Z2.shape[1]

def test_deterministic_transform(make_df):
    X, y = make_df(seed=2)

    ft = FeatureTransformer()
    ft.fit(X, y)

    Z1 = _to_dense(ft.transform(X))
    Z2 = _to_dense(ft.transform(X))

    np.testing.assert_allclose(Z1, Z2, atol=0, rtol=0)

