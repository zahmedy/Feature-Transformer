import pytest
from src.transformer import FeatureTransformer

def test_missing_required_column_behavior(make_df):
    X, y = make_df(seed=4)

    ft = FeatureTransformer()
    ft.fit(X, y)

    X_bad = X.drop(columns=["income"])

    # Choose one behavior and enforce it:
    # 1) strict: raise a clear error
    with pytest.raises(Exception):
        ft.transform(X_bad)

def test_extra_columns_ignored(make_df):
    X, y = make_df(seed=5)

    ft = FeatureTransformer()
    ft.fit(X, y)

    X_extra = X.copy()
    X_extra["some_new_col"] = 123

    Z = ft.transform(X_extra)
    assert Z.shape[0] == X.shape[0]
