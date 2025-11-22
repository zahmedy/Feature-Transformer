from typing import Optional, Sequence, Literal
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression


TaskType = Literal["classification", "regression"]

@dataclass
class FeatureConfig:
    numeric_features: Optional[Sequence[str]] = None
    categorical_features: Optional[Sequence[str]] = None


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Reusable preprocessing module:
    - Numeric: impute + (optional) scale
    - Categorical: impute + one-hot
    - Optional feature selection (SelectKBest)
    """

    def __init__(
        self,
        feature_config: Optional[FeatureConfig] = None,
        task: TaskType = "classification",
        k_best: Optional[int] = None,
        scale_numeric: bool = True,
        handle_unknown: str = "ignore",
    ):
        self.feature_config = feature_config
        self.task = task
        self.k_best = k_best
        self.scale_numeric = scale_numeric
        self.handle_unknown = handle_unknown

        # will be created in fit()
        self._pipeline: Optional[Pipeline] = None
        self._feature_names_out: Optional[np.ndarray] = None
        self._numeric_features_: Optional[Sequence[str]] = None
        self._categorical_features_: Optional[Sequence[str]] = None

def _infer_features(self, X: pd.DataFrame):
    """Infer numeric and categorical columns if not provided."""
    if self.feature_config is None:
        self.feature_config = FeatureConfig()

    # numeric = all number columns
    if self.feature_config.numeric_features is None:
        self._numeric_features_ = X.select_dtypes(include=["number"]).columns.tolist()
    else:
        self._numeric_features_ = list(self.feature_config.numeric_features)

    # categorical = everything else (object/string columns)
    if self.feature_config.categorical_features is None:
        self._categorical_features_ = X.select_dtypes(exclude=["number"]).columns.tolist()
    else:
        self._categorical_features_ = list(self.feature_config.categorical_features)

def _build_preprocessor(self):
    # --- numeric pipeline ---
    num_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if self.scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=num_steps)

    # --- categorical pipeline ---
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown=self.handle_unknown,
                    sparse_output=False,
                ),
            ),
        ]
    )

    # combine both
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, self._numeric_features_),
            ("cat", cat_pipeline, self._categorical_features_),
        ],
        remainder="drop",
    )
    
    return preprocessor

def _build_pipeline(self):
    """Build the internal sklearn Pipeline with preprocessing (+ optional feature selection)."""
    preprocessor = self._build_preprocessor()
    steps = [("preprocessor", preprocessor)]

    # Optional feature selection
    if self.k_best is not None:
        if self.task == "classification":
            score_func = f_classif
        else:
            score_func = f_regression

        steps.append(("select", SelectKBest(score_func=score_func, k=self.k_best)))

    self._pipeline = Pipeline(steps=steps)

def _validate_input(self, X):
    """Ensure X is a pandas DataFrame."""
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")
    return X

def fit(self, X: pd.DataFrame, y=None):
    """Fit the internal pipeline on the data."""
    X = self._validate_input(X)

    # 1) infer numeric/categorical columns
    self._infer_features(X)

    # 2) build the pipeline using those columns
    self._build_pipeline()

    # 3) fit the internal pipeline
    self._pipeline.fit(X, y)

    return self

def transform(self, X: pd.DataFrame) -> np.ndarray:
    """Transform new data using the fitted pipeline."""
    X = self._validate_input(X)
    return self._pipeline.transform(X)

def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
    """Convenience method: fit + transform in one call."""
    return self.fit(X, y).transform(X)
