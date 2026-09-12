"""Candidate model specifications for Experiment 08.

Implements:
1. Core Regression models (Section 18):
   - Median Baseline (DummyRegressor)
   - Ridge Trend Model
   - Regression Decision Tree (depth=6, leaf=10)
   - Random Forest Regressor (trees=60, depth=12, leaf=5)
2. Classification models (Section 31):
   - Majority Baseline (DummyClassifier)
   - Multinomial-capable Logistic Regression (max_iter=2000)
   - Random Forest Classifier (trees=60, leaf=2)
"""

from typing import Dict
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from src.config import RANDOM_SEED
    from src.preprocessing import (
        build_classification_preprocessor,
        build_regression_preprocessor,
    )
except ImportError:
    from config import RANDOM_SEED
    from preprocessing import (
        build_classification_preprocessor,
        build_regression_preprocessor,
    )


def get_regression_models(include_depth_ablation: bool = False) -> Dict[str, Pipeline]:
    """Returns initialized regression candidate pipelines."""
    preprocessor = build_regression_preprocessor()

    models = {
        "median": Pipeline(
            steps=[
                ("pre", preprocessor),
                ("model", DummyRegressor(strategy="median")),
            ]
        ),
        "ridge_trend": Pipeline(
            steps=[
                ("pre", preprocessor),
                ("model", Ridge(alpha=1.0, solver="lsqr", random_state=RANDOM_SEED)),
            ]
        ),
        "tree": Pipeline(
            steps=[
                ("pre", preprocessor),
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=6,
                        min_samples_leaf=10,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "forest": Pipeline(
            steps=[
                ("pre", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=60,
                        max_depth=12,
                        min_samples_leaf=5,
                        n_jobs=2,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }

    if include_depth_ablation:
        models["forest_depth6"] = Pipeline(
            steps=[
                ("pre", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=60,
                        max_depth=6,
                        min_samples_leaf=5,
                        n_jobs=2,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )

    return models


def get_classification_models() -> Dict[str, Pipeline]:
    """Returns initialized classification candidate pipelines."""
    preprocessor = build_classification_preprocessor()

    models = {
        "majority": Pipeline(
            steps=[
                ("pre", preprocessor),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "logistic": Pipeline(
            steps=[
                ("pre", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "forest": Pipeline(
            steps=[
                ("pre", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=60,
                        min_samples_leaf=2,
                        n_jobs=2,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }
    return models
