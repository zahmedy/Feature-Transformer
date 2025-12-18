import numpy as np
import pandas as pd
from src.transformer import FeatureTransformer


def test_no_missing_values_after_transform(make_df):
    X, y = make_df()

    ft = FeatureTransformer()
    ft.fit(X, y)
    Z = ft.transform(X)

    assert Z.shape[0] == X.shape[0]
    assert not np.isnan(Z).any()


def test_column_order_invariance(make_df):
    X, y = make_df()

    ft = FeatureTransformer()
    ft.fit(X, y)

    Z1 = ft.transform(X)

    X_shuffled = X[X.columns[::-1]]

    Z2 = ft.transform(X_shuffled)

    np.testing.assert_allclose(Z1, Z2)