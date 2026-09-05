"""
Artifact management and reproducibility verification module for Experiment 07 Recommendation System.
Saves model checkpoints, schemas, manifests, and verifies model reload equivalence.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd


def save_run_artifacts(
    model: Any,
    feature_cols: List[str],
    split_dates: Dict[str, str],
    candidate_policy: Dict[str, Any],
    out_dir: Union[str, Path],
    models_dir: Union[str, Path],
    model_filename: str = "random_forest.joblib",
) -> Dict[str, Path]:
    """
    Save all required reproducibility artifacts to their standard directory locations.

    Parameters:
        model: Trained scikit-learn model.
        feature_cols: List of engineered feature column names.
        split_dates: Dictionary containing train_end, val_end, and window counts.
        candidate_policy: Dictionary defining candidate generation and negative sampling policies.
        out_dir: Directory for JSON artifact outputs (e.g. artifacts/d1_online_retail/).
        models_dir: Directory for model binary checkpoints (e.g. models/d1_online_retail/).
        model_filename: Name of serialized model file.

    Returns:
        saved_paths: Dictionary mapping artifact names to their saved file paths.
    """
    artifacts_path = Path(out_dir)
    models_path = Path(models_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    # 1. Save feature schema
    schema_file = artifacts_path / "feature_schema.json"
    with open(schema_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_count": len(feature_cols),
                "feature_columns": feature_cols,
            },
            f,
            indent=2,
        )

    # 2. Save split manifest
    manifest_file = artifacts_path / "split_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(split_dates, f, indent=2)

    # 3. Save candidate policy
    policy_file = artifacts_path / "candidate_policy.json"
    with open(policy_file, "w", encoding="utf-8") as f:
        json.dump(candidate_policy, f, indent=2)

    # 4. Save model checkpoint
    model_file = models_path / model_filename
    joblib.dump(model, model_file)

    return {
        "feature_schema": schema_file,
        "split_manifest": manifest_file,
        "candidate_policy": policy_file,
        "model_file": model_file,
    }


def reload_and_verify(
    model_path: Union[str, Path],
    in_memory_model: Any,
    X_sample: pd.DataFrame,
    feature_cols: List[str],
    tolerance: float = 1e-6,
) -> bool:
    """
    Verify reproducibility by reloading serialized model and confirming identical inference predictions.

    Parameters:
        model_path: Path to serialized joblib model.
        in_memory_model: Active model instance in memory.
        X_sample: Sample feature DataFrame for test inference.
        feature_cols: List of numerical feature columns.
        tolerance: Numerical difference tolerance.

    Returns:
        matches: True if predictions are numerically identical within tolerance.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found at: {model_path}")

    reloaded_model = joblib.load(model_path)

    sample_features = X_sample[feature_cols]
    preds_in_memory = in_memory_model.predict_proba(sample_features)
    preds_reloaded = reloaded_model.predict_proba(sample_features)

    diff = np.max(np.abs(preds_in_memory - preds_reloaded))
    if diff > tolerance:
        raise AssertionError(
            f"Appendix C violation: reloaded model predictions differ by {diff} (tolerance {tolerance})"
        )

    return True
