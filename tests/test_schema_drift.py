import pytest
from src.transformer import FeatureTransformer


def test_extra_columns_ignored(make_df):
    X, y = make_df(seed=5)

    ft = FeatureTransformer()
    ft.fit(X, y)

    X_extra = X.copy()
    X_extra["some_new_col"] = 123

    Z = ft.transform(X_extra)
    assert Z.shape[0] == X.shape[0]


def test_missing_required_columns(make_df):
    X, y = make_df()

    ft = FeatureTransformer()
    ft.fit(X, y)
    
    X = X.drop("income", axis=1)

    with pytest.raises(ValueError, match="income"):
        ft.transform(X)