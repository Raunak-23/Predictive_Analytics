"""Leakage-safe hyperparameter tuning for Experiment 08.

Enforces:
1. Tuning runs ONLY on the previously selected model family.
2. Tuning is performed strictly on TRAINING DATA ONLY.
3. Time-aware cross-validation for regression ensures no training fold contains future years.
4. External validation and test sets remain strictly untouched during hyperparameter search.
5. Evaluates tuned model once on external validation set to compare with default model.
"""

import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

try:
    from src.config import (
        CLS_FEATURES,
        CLS_TARGET,
        CLS_TUNING_GRIDS,
        MODELS_CLS_DIR,
        MODELS_REG_DIR,
        RANDOM_SEED,
        REG_CV_FOLDS_FILE,
        REG_FEATURES,
        REG_TARGET,
        REG_TUNING_GRIDS,
        RESULTS_CLS_DIR,
        RESULTS_REG_DIR,
        TEST_LOCK_FILE,
        TUNING_DIR,
        get_current_config_sha256,
    )
    from src.evaluation import evaluate_classification, evaluate_regression
    from src.utils import compute_sha256, save_json, setup_logger
except ImportError:
    from config import (
        CLS_FEATURES,
        CLS_TARGET,
        CLS_TUNING_GRIDS,
        MODELS_CLS_DIR,
        MODELS_REG_DIR,
        RANDOM_SEED,
        REG_CV_FOLDS_FILE,
        REG_FEATURES,
        REG_TARGET,
        REG_TUNING_GRIDS,
        RESULTS_CLS_DIR,
        RESULTS_REG_DIR,
        TEST_LOCK_FILE,
        TUNING_DIR,
        get_current_config_sha256,
    )
    from evaluation import evaluate_classification, evaluate_regression
    from utils import compute_sha256, save_json, setup_logger

logger = setup_logger("tuning")


def forward_year_block_cv(
    df_train: pd.DataFrame,
    val_years: Optional[List[int]] = None,
    save_folds_path: Optional[Path] = REG_CV_FOLDS_FILE,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Generates strictly forward year-block cross-validation folds on training data.

    For each validation year in val_years (all <= 2011):
      - training fold: all observations with year < validation_year
      - validation fold: observations with year == validation_year

    Guarantees:
      - max(train_year) < min(validation_year)
      - No future observations appear in the training fold.
      - Never receives 2012 or later (strictly training-only).
    """
    records = []
    folds = []

    if val_years is None:
        avail_years = sorted(df_train["year"].unique())
        valid_train_years = [int(y) for y in avail_years if y < 2012]
        if len(valid_train_years) <= 2:
            raise ValueError(f"Insufficient training years ({len(valid_train_years)}) for CV!")
        n_val = min(5, len(valid_train_years) - 2)
        val_years = valid_train_years[-n_val:]
    else:
        val_years = sorted(val_years)

    for fold_idx, val_yr in enumerate(val_years, start=1):
        if val_yr >= 2012:
            raise ValueError(f"Tuning CV must not include heldout year {val_yr} (>= 2012)!")

        train_mask = df_train["year"] < val_yr
        val_mask = df_train["year"] == val_yr

        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]

        if len(train_indices) == 0 or len(val_indices) == 0:
            raise ValueError(f"Fold {fold_idx} for validation year {val_yr} has empty split!")

        train_min_yr = int(df_train.iloc[train_indices]["year"].min())
        train_max_yr = int(df_train.iloc[train_indices]["year"].max())
        val_yr_int = int(df_train.iloc[val_indices]["year"].iloc[0])

        assert train_max_yr < val_yr_int, (
            f"Temporal leakage in Fold {fold_idx}: train_max_yr ({train_max_yr}) >= val_yr ({val_yr_int})"
        )

        records.append(
            {
                "fold": fold_idx,
                "train_min_year": train_min_yr,
                "train_max_year": train_max_yr,
                "validation_year": val_yr_int,
                "n_train": len(train_indices),
                "n_validation": len(val_indices),
            }
        )
        folds.append((train_indices, val_indices))

    if save_folds_path is not None:
        save_folds_path.parent.mkdir(parents=True, exist_ok=True)
        folds_df = pd.DataFrame(records)
        folds_df.to_csv(save_folds_path, index=False)
        logger.info("Saved forward CV partition manifest to %s", save_folds_path)

    for fold in folds:
        yield fold


def temporal_year_cv_split(
    df_train: pd.DataFrame, n_splits: int = 5
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Compatibility wrapper delegating to forward_year_block_cv."""
    avail_years = sorted(df_train["year"].unique())
    valid_train_years = [int(y) for y in avail_years if y < 2012]
    n_val = min(n_splits, max(1, len(valid_train_years) - 2))
    val_years = valid_train_years[-n_val:]
    yield from forward_year_block_cv(df_train, val_years=val_years, save_folds_path=None)


def tune_regression_model(
    selected_model_name: str,
    base_pipeline: Pipeline,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    data_sha256: str,
    config_sha256: Optional[str] = None,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Tunes the selected regression model family using training-only temporal CV."""
    if TEST_LOCK_FILE.exists():
        raise RuntimeError(
            f"TEST_LOCK exists at {TEST_LOCK_FILE}! "
            "Hyperparameter tuning cannot be run after test evaluation has been locked."
        )

    # Use authoritative current configuration hash
    config_sha256 = get_current_config_sha256()

    if selected_model_name not in REG_TUNING_GRIDS:
        logger.info(
            "Model '%s' has no parameter tuning grid (e.g. median baseline). Skipping tuning.",
            selected_model_name,
        )
        return base_pipeline, {"tuned": False, "reason": "No grid configured"}

    param_grid = REG_TUNING_GRIDS[selected_model_name]
    logger.info("Initiating temporal CV tuning for '%s'...", selected_model_name)
    logger.info("Parameter grid: %s", param_grid)

    cv_generator = list(
        forward_year_block_cv(
            df_train,
            val_years=[2007, 2008, 2009, 2010, 2011],
            save_folds_path=REG_CV_FOLDS_FILE,
        )
    )

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=cv_generator,
        n_jobs=1,
        refit=True,  # Refits on complete training set
    )

    t0 = time.perf_counter()
    grid_search.fit(df_train[REG_FEATURES], df_train[REG_TARGET])
    tune_time = time.perf_counter() - t0

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = float(-grid_search.best_score_)

    logger.info(
        "Tuning complete in %.2fs. Best CV MAE: %.4f t/ha with params: %s",
        tune_time,
        best_cv_score,
        best_params,
    )

    # Save detailed CV results table
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    # Extract clean summary columns for analysis
    if "param_model__alpha" in cv_results_df.columns:
        cv_results_df["candidate_alpha"] = cv_results_df["param_model__alpha"].astype(float)
        cv_results_df["mean_cv_mae"] = -cv_results_df["mean_test_score"]
        cv_results_df["std_cv_mae"] = cv_results_df["std_test_score"]
        cv_results_df["rank"] = cv_results_df["rank_test_score"]
        # Order by rank
        cv_results_df = cv_results_df.sort_values("rank").reset_index(drop=True)

        # Check flatness of tuning surface
        best_mae = cv_results_df.iloc[0]["mean_cv_mae"]
        alpha_1_rows = cv_results_df[cv_results_df["candidate_alpha"] == 1.0]
        if not alpha_1_rows.empty:
            alpha_1_mae = float(alpha_1_rows.iloc[0]["mean_cv_mae"])
            mae_diff = abs(alpha_1_mae - best_mae)
            if mae_diff < 0.001:
                logger.info(
                    "Tuning surface around optimum is flat: best CV MAE = %.4f vs alpha=1.0 CV MAE = %.4f (diff = %.4f < 0.001).",
                    best_mae,
                    alpha_1_mae,
                    mae_diff,
                )

    RESULTS_REG_DIR.mkdir(parents=True, exist_ok=True)
    cv_results_path = RESULTS_REG_DIR / "tuning_results.csv"
    cv_results_df.to_csv(cv_results_path, index=False)
    logger.info("Saved tuning CV results to %s", cv_results_path)

    # Evaluate tuned model on untouched validation set
    val_pred = best_pipeline.predict(df_val[REG_FEATURES])
    val_metrics = evaluate_regression(df_val[REG_TARGET], val_pred)

    # Save tuned model artifact immediately
    MODELS_REG_DIR.mkdir(parents=True, exist_ok=True)
    tuned_artifact_path = MODELS_REG_DIR / "tuned_selected_model.joblib"
    joblib.dump(best_pipeline, tuned_artifact_path)

    tuning_metadata = {
        "model_family": selected_model_name,
        "search_method": "GridSearchCV",
        "cv_splitter": "forward_year_block_temporal_cv",
        "n_folds": 5,
        "validation_years": [2007, 2008, 2009, 2010, 2011],
        "parameter_grid": param_grid,
        "best_params": best_params,
        "best_cv_mae_t_ha": round(best_cv_score, 4),
        "validation_metrics": val_metrics,
        "tune_seconds": round(tune_time, 2),
        "data_sha256": data_sha256,
        "config_sha256": config_sha256,
        "random_seed": RANDOM_SEED,
        "artifact_path": str(tuned_artifact_path.name),
    }

    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    save_json(tuning_metadata, TUNING_DIR / "best_params.json")
    logger.info("Tuning metadata saved to %s", TUNING_DIR / "best_params.json")

    return best_pipeline, tuning_metadata


def tune_classification_model(
    selected_model_name: str,
    base_pipeline: Pipeline,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    data_sha256: str,
    config_sha256: Optional[str] = None,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Tunes selected classification model using stratified CV on training data only."""
    config_sha256 = get_current_config_sha256()
    if selected_model_name not in CLS_TUNING_GRIDS:
        logger.info("Model '%s' has no grid configured. Skipping tuning.", selected_model_name)
        return base_pipeline, {"tuned": False, "reason": "No grid configured"}

    param_grid = CLS_TUNING_GRIDS[selected_model_name]
    logger.info("Initiating Stratified CV tuning for '%s'...", selected_model_name)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=skf,
        n_jobs=1,
        refit=True,
    )

    t0 = time.perf_counter()
    grid_search.fit(df_train[CLS_FEATURES], df_train[CLS_TARGET])
    tune_time = time.perf_counter() - t0

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = float(grid_search.best_score_)

    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    RESULTS_CLS_DIR.mkdir(parents=True, exist_ok=True)
    cv_results_df.to_csv(RESULTS_CLS_DIR / "tuning_results.csv", index=False)

    val_pred = best_pipeline.predict(df_val[CLS_FEATURES])
    val_metrics = evaluate_classification(df_val[CLS_TARGET], val_pred)

    MODELS_CLS_DIR.mkdir(parents=True, exist_ok=True)
    tuned_artifact_path = MODELS_CLS_DIR / "tuned_selected_model.joblib"
    joblib.dump(best_pipeline, tuned_artifact_path)

    tuning_metadata = {
        "model_family": selected_model_name,
        "search_method": "GridSearchCV",
        "cv_splitter": "StratifiedKFold(n_splits=3)",
        "parameter_grid": param_grid,
        "best_params": best_params,
        "best_cv_macro_f1": round(best_cv_score, 4),
        "validation_metrics": val_metrics,
        "tune_seconds": round(tune_time, 2),
        "data_sha256": data_sha256,
        "config_sha256": config_sha256,
        "random_seed": RANDOM_SEED,
    }

    save_json(tuning_metadata, TUNING_DIR / "classification_best_params.json")
    return best_pipeline, tuning_metadata
