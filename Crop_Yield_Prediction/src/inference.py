"""Inference and Model Bundle Management for Experiment 08.

Implements:
1. Self-contained model bundles with full metadata and validation envelopes (Section 34).
2. Strict runtime input validation gates:
   - Unknown category rejection (unseen state/district/season)
   - Year validation (integer, within evaluated range)
   - Physical bounds validation for classification (pH, humidity, nutrients)
   - Non-zero exit code on invalid input.
"""

from pathlib import Path
from typing import Any, Dict, List, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

try:
    from src.config import (
        CLS_FEATURES,
        CLS_TARGET,
        RANDOM_SEED,
        REG_CATEGORICAL_FEATURES,
        REG_FEATURES,
        REG_TARGET,
    )
    from src.utils import setup_logger
except ImportError:
    from config import (
        CLS_FEATURES,
        CLS_TARGET,
        RANDOM_SEED,
        REG_CATEGORICAL_FEATURES,
        REG_FEATURES,
        REG_TARGET,
    )
    from utils import setup_logger

logger = setup_logger("inference")


def create_model_bundle(
    pipeline: Pipeline,
    task: str,
    training_df: pd.DataFrame,
    data_sha256: str,
    config_sha256: str,
    model_name: str,
    save_path: Path,
) -> Dict[str, Any]:
    """Serializes a fitted pipeline into a validated, self-contained inference bundle."""
    import platform
    import sklearn

    model_step = pipeline.named_steps.get("model")
    raw_params = model_step.get_params() if model_step is not None else {}
    # Sanitize parameters for portable metadata serialization
    sanitized_params = {
        k: (str(v) if not isinstance(v, (int, float, bool, str, type(None), list, dict)) else v)
        for k, v in raw_params.items()
    }

    bundle: Dict[str, Any] = {
        "pipeline": pipeline,
        "task": task,
        "model_name": model_name,
        "model_family": model_name.replace("_tuned", ""),
        "parameters": sanitized_params,
        "features": list(REG_FEATURES if task == "regression" else CLS_FEATURES),
        "target": REG_TARGET if task == "regression" else CLS_TARGET,
        "training_years": (
            [int(training_df["year"].min()), int(training_df["year"].max())]
            if "year" in training_df.columns
            else []
        ),
        "data_sha256": data_sha256,
        "config_sha256": config_sha256,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "random_seed": RANDOM_SEED,
        "version": "1.0.0",
    }

    bundle["metadata"] = {
        "task": task,
        "model_name": model_name,
        "data_sha256": data_sha256,
        "config_sha256": config_sha256,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "random_seed": RANDOM_SEED,
        "version": "1.0.0",
    }

    if task == "regression":
        bundle["year_range"] = [int(training_df["year"].min()), int(training_df["year"].max())]
        bundle["categories"] = {
            cat_col: sorted(training_df[cat_col].astype(str).unique().tolist())
            for cat_col in REG_CATEGORICAL_FEATURES
        }
    else:
        bundle["classes"] = sorted(training_df[CLS_TARGET].unique().tolist())

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, save_path)
    logger.info("Saved model bundle to %s", save_path)

    # Immediate reload verification
    reloaded = joblib.load(save_path)
    sample_df = training_df.iloc[:2]
    feats = bundle["features"]
    orig_pred = pipeline.predict(sample_df[feats])
    reloaded_pred = reloaded["pipeline"].predict(sample_df[feats])
    np.testing.assert_array_equal(
        orig_pred,
        reloaded_pred,
        err_msg="Reloaded bundle predictions differ from original fitted pipeline!",
    )
    logger.info("Model reload verification passed for %s.", save_path.name)

    return bundle


def predict_validated(
    bundle: Dict[str, Any], records: Union[List[Dict[str, Any]], Dict[str, Any]]
) -> np.ndarray:
    """Performs strict validation-gated inference against the provided model bundle."""
    if isinstance(records, dict):
        records = [records]

    df = pd.DataFrame(records)
    features = bundle["features"]
    task = bundle["task"]

    # 1. Feature presence check
    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required input features: {missing}")

    if len(df) == 0:
        raise ValueError("Input records cannot be empty!")

    # 2. Task-specific validation
    if task == "regression":
        # Check numeric year
        for idx, val in df["year"].items():
            try:
                num_yr = float(val)
            except (ValueError, TypeError):
                raise ValueError(f"Non-numeric year provided at record {idx}: {val}")
            if num_yr % 1 != 0:
                raise ValueError(f"Non-integer harvest year rejected: {val}")

        df["year"] = df["year"].astype(int)
        min_yr, max_yr = bundle["year_range"]
        out_of_range = df[~df["year"].between(min_yr, max_yr)]
        if not out_of_range.empty:
            bad_years = out_of_range["year"].tolist()
            raise ValueError(
                f"Year(s) {bad_years} outside evaluated range [{min_yr}, {max_yr}]. "
                "Retrospective model cannot extrapolate reliably outside learned range."
            )

        # Check categorical validity (reject unknown categories)
        for cat_col, valid_cats in bundle["categories"].items():
            unknown = df[~df[cat_col].astype(str).isin(valid_cats)]
            if not unknown.empty:
                bad_vals = unknown[cat_col].unique().tolist()
                raise ValueError(
                    f"Unknown or unvalidated {cat_col}(s) rejected: {bad_vals}. "
                    "Inference refused to prevent uncalibrated predictions."
                )

    elif task == "classification":
        # Validate numeric features
        for c in CLS_FEATURES:
            df[c] = pd.to_numeric(df[c], errors="raise")
            if not np.isfinite(df[c]).all():
                raise ValueError(f"Non-finite input encountered in feature '{c}'")

        if not df["ph"].between(0, 14).all():
            raise ValueError("Soil pH must be between 0 and 14")
        if not df["humidity"].between(0, 100).all():
            raise ValueError("Relative humidity must be between 0 and 100%")
        if (df[["N", "P", "K", "rainfall"]] < 0).any().any():
            raise ValueError("Nutrient (N, P, K) and rainfall inputs must be non-negative")

    predictions = bundle["pipeline"].predict(df[features])
    return predictions
