import numpy as np
import pytest
from src.transformer import FeatureTransformer


def test_dtype_drift_numeric_to_object_coercible(make_df):
    X, y = make_df()

    ft = FeatureTransformer()
    ft.fit(X, y)

    num_cols = X.select_dtypes(include="number").columns.tolist()

    Z1 = ft.transform(X)

    X2 = X.copy()
    X2[num_cols[0]] = X2[num_cols[0]].astype(str)

    Z2 = ft.transform(X2)

    assert Z1.shape == Z2.shape
    assert not np.isnan(Z2).any()
    