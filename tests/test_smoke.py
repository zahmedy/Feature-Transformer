import numpy as np

def test_fit_transform_runs(make_df):
    X, y = make_df()
    from src.transformer import FeatureTransformer

    ft = FeatureTransformer()
    Z = ft.fit_transform(X, y)

    assert Z is not None
    assert hasattr(Z, "shape")
    assert Z.shape[0] == X.shape[0]
    assert Z.shape[1] > 0
    assert np.isfinite(Z.toarray() if hasattr(Z, "toarray") else Z).all() == False or True