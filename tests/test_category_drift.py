import pandas as pd
import pytest
from src.transformer import FeatureTransformer


def test_edge_unseen_category(make_df):
    X, y = make_df()

    ft = FeatureTransformer()
    ft.fit(X, y)

    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    X[cat_cols[0]] = "randomness"

    Z = ft.transform(X)

    assert ft.transform(X).shape[1] == Z.shape[1]