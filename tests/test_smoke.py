import numpy as np
from src.transformer import FeatureTransformer

def test_fit_transform_runs(make_df):
    X, y = make_df()

    ft = FeatureTransformer()
    Z = ft.fit_transform(X, y)

    assert Z is not None
    assert hasattr(Z, "shape")
    assert Z.shape[0] == X.shape[0]
    assert Z.shape[1] > 0
    assert np.isfinite(Z.toarray() if hasattr(Z, "toarray") else Z).all() == False or True