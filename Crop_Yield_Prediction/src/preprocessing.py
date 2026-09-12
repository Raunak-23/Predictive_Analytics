"""Preprocessing pipeline construction for Experiment 08.

Enforces:
1. Whitelist of predictors only (no production, no area).
2. All learned scalers/encoders strictly inside ColumnTransformer and Pipeline objects.
3. Categorical handling via OneHotEncoder(handle_unknown='ignore').
4. Numeric scaling via SimpleImputer(strategy='median') + StandardScaler().
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from src.config import (
        CLS_FEATURES,
        REG_CATEGORICAL_FEATURES,
        REG_NUMERIC_FEATURES,
    )
except ImportError:
    from config import (
        CLS_FEATURES,
        REG_CATEGORICAL_FEATURES,
        REG_NUMERIC_FEATURES,
    )


def build_regression_preprocessor() -> ColumnTransformer:
    """Builds scikit-learn ColumnTransformer for regression pipeline.

    Categorical: state, district, season -> OneHotEncoder(handle_unknown='ignore')
    Numeric: year -> SimpleImputer(strategy='median') -> StandardScaler()
    """
    cat_pipeline = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, REG_CATEGORICAL_FEATURES),
            ("num", num_pipeline, REG_NUMERIC_FEATURES),
        ],
        remainder="drop",  # Safeguard: discard any unexpected column
    )
    return preprocessor


def build_classification_preprocessor() -> Pipeline:
    """Builds scikit-learn Pipeline for classification feature preprocessing.

    Numeric features: N, P, K, temp, humidity, pH, rainfall
    -> SimpleImputer(strategy='median') -> StandardScaler()
    """
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return pipeline
