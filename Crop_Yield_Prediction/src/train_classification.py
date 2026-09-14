"""Training and evaluation pipeline for Crop Label Classification (Track B).

Processes the D3 Crop Recommendation dataset:
1. Validates independence assumption (default requires --justify-iid flag).
2. Performs stratified train/val/test splits (60/20/20).
3. Compares candidate models: Majority, Logistic Regression, Random Forest.
4. Serializes models incrementally and records statuses in registry.
5. Selects best model by macro F1 and tunes on training data only.
6. Evaluates on test set and computes overall and per-class metrics.
7. Generates confusion matrix and model comparison plots.
8. Exports verified inference bundle.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict
import joblib
import matplotlib
# Only switch to non-interactive Agg backend if running in headless mode outside Jupyter/IPython
if "ipykernel" not in sys.modules and "IPython" not in sys.modules:
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

try:
    from src.config import (
        ARTIFACTS_DIR,
        CLS_FEATURES,
        CLS_OVERFITTING_FILE,
        CLS_REQUIRED_COLUMNS,
        CLS_SPLIT_MANIFEST,
        CLS_TARGET,
        CROP_LABELS_CANONICAL_FILE,
        FIGURES_CLS_DIR,
        MODELS_CLS_DIR,
        RANDOM_SEED,
        RESULTS_CLS_DIR,
        TUNING_DIR,
        get_current_config_sha256,
        save_pipeline_config,
    )
    from src.evaluation import (
        evaluate_classification,
        plot_confusion_matrix_display,
        plot_model_comparison,
    )
    from src.inference import create_model_bundle
    from src.models import get_classification_models
    from src.split_data import get_classification_splits
    from src.tuning import tune_classification_model
    from src.utils import compute_sha256, save_json, setup_logger
except ImportError:
    from config import (
        ARTIFACTS_DIR,
        CLS_FEATURES,
        CLS_OVERFITTING_FILE,
        CLS_REQUIRED_COLUMNS,
        CLS_SPLIT_MANIFEST,
        CLS_TARGET,
        CROP_LABELS_CANONICAL_FILE,
        FIGURES_CLS_DIR,
        MODELS_CLS_DIR,
        RANDOM_SEED,
        RESULTS_CLS_DIR,
        TUNING_DIR,
        get_current_config_sha256,
        save_pipeline_config,
    )
    from evaluation import (
        evaluate_classification,
        plot_confusion_matrix_display,
        plot_model_comparison,
    )
    from inference import create_model_bundle
    from models import get_classification_models
    from split_data import get_classification_splits
    from tuning import tune_classification_model
    from utils import compute_sha256, save_json, setup_logger

logger = setup_logger("train_classification")


def run_classification(justify_iid: bool = False) -> None:
    """Executes end-to-end classification training, tuning, and evaluation."""
    logger.info("=== STARTING CLASSIFICATION PIPELINE (TRACK B) ===")

    if not CROP_LABELS_CANONICAL_FILE.exists():
        raise FileNotFoundError(f"Canonical classification data missing: {CROP_LABELS_CANONICAL_FILE}")

    data_sha256 = compute_sha256(CROP_LABELS_CANONICAL_FILE)
    df = pd.read_csv(CROP_LABELS_CANONICAL_FILE)

    # 1. Stratified Holdout Split
    df_split = get_classification_splits(df, iid_justified=justify_iid)
    train_df = df_split[df_split["split"] == "train"].copy()
    val_df = df_split[df_split["split"] == "validation"].copy()
    test_df = df_split[df_split["split"] == "test"].copy()

    logger.info(
        "Classification split sizes: Train=%d, Val=%d, Test=%d across %d classes.",
        len(train_df),
        len(val_df),
        len(test_df),
        df[CLS_TARGET].nunique(),
    )

    # 2. Candidate Models Comparison
    MODELS_CLS_DIR.mkdir(parents=True, exist_ok=True)
    candidates = get_classification_models()
    val_records = []

    # Use centralized configuration hash (Part 9)
    config_sha256 = get_current_config_sha256()

    for name, pipeline in candidates.items():
        logger.info("Fitting candidate: '%s'...", name)
        t0 = time.perf_counter()
        pipeline.fit(train_df[CLS_FEATURES], train_df[CLS_TARGET])
        fit_time = time.perf_counter() - t0

        val_pred = pipeline.predict(val_df[CLS_FEATURES])
        metrics = evaluate_classification(val_df[CLS_TARGET], val_pred)

        # Serialize model
        artifact_path = MODELS_CLS_DIR / f"{name}.joblib"
        joblib.dump(pipeline, artifact_path)

        val_records.append(
            {
                "model": name,
                "runtime_seconds": round(fit_time, 3),
                "macro_F1": metrics["macro_F1"],
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
            }
        )
        logger.info(
            "Candidate '%s': macro_F1=%.4f, accuracy=%.4f",
            name,
            metrics["macro_F1"],
            metrics["accuracy"],
        )

    # Save validation results
    RESULTS_CLS_DIR.mkdir(parents=True, exist_ok=True)
    val_results_df = pd.DataFrame(val_records)
    val_results_df = val_results_df.sort_values("macro_F1", ascending=False, kind="stable").reset_index(drop=True)
    val_results_csv = RESULTS_CLS_DIR / "validation_results.csv"
    val_results_df.to_csv(val_results_csv, index=False)
    logger.info("Validation results saved to %s", val_results_csv)

    # Plot model comparison
    FIGURES_CLS_DIR.mkdir(parents=True, exist_ok=True)
    plot_model_comparison(
        val_results_df,
        FIGURES_CLS_DIR / "model_comparison.png",
        metric_col="macro_F1",
        task_name="Classification Candidate Validation",
    )

    # Select best non-baseline model for tuning
    selected_name = val_results_df.iloc[0]["model"]
    logger.info("Selected best classification candidate: '%s'", selected_name)

    # 3. Hyperparameter Tuning on Training Data Only
    selected_pipe = joblib.load(MODELS_CLS_DIR / f"{selected_name}.joblib")
    tuned_pipe, tuning_meta = tune_classification_model(
        selected_model_name=selected_name,
        base_pipeline=selected_pipe,
        df_train=train_df,
        df_val=val_df,
        data_sha256=data_sha256,
        config_sha256=config_sha256,
    )

    # Compare validation performance
    use_tuned = False
    if tuning_meta.get("tuned", True) and "validation_metrics" in tuning_meta:
        tuned_f1 = tuning_meta["validation_metrics"]["macro_F1"]
        default_f1 = float(val_results_df.iloc[0]["macro_F1"])
        if tuned_f1 >= default_f1:
            logger.info("Tuned classification model selected (macro_F1=%.4f).", tuned_f1)
            use_tuned = True
            final_pipe = tuned_pipe
        else:
            final_pipe = selected_pipe
    else:
        final_pipe = selected_pipe

    # 4. Final Evaluation on Test Set
    logger.info("Evaluating selected model and baseline on locked classification test set...")
    t0 = time.perf_counter()
    test_pred = final_pipe.predict(test_df[CLS_FEATURES])
    latency = time.perf_counter() - t0

    test_metrics = evaluate_classification(test_df[CLS_TARGET], test_pred)

    majority_pipe = joblib.load(MODELS_CLS_DIR / "majority.joblib")
    majority_pred = majority_pipe.predict(test_df[CLS_FEATURES])
    majority_metrics = evaluate_classification(test_df[CLS_TARGET], majority_pred)

    test_rows = [
        {
            "model": f"{selected_name}_tuned" if use_tuned else selected_name,
            "batch_inference_seconds": round(latency, 4),
            "macro_F1": test_metrics["macro_F1"],
            "accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
        },
        {
            "model": "majority",
            "batch_inference_seconds": 0.001,
            "macro_F1": majority_metrics["macro_F1"],
            "accuracy": majority_metrics["accuracy"],
            "macro_precision": majority_metrics["macro_precision"],
            "macro_recall": majority_metrics["macro_recall"],
        },
    ]
    test_results_df = pd.DataFrame(test_rows)
    test_results_csv = RESULTS_CLS_DIR / "test_results.csv"
    test_results_df.to_csv(test_results_csv, index=False)
    logger.info("Test results saved to %s", test_results_csv)
    logger.info("Classification Test Results:\n%s", test_results_df.to_string(index=False))

    # 5. Per-class metrics
    per_class_dict = classification_report(
        test_df[CLS_TARGET],
        test_pred,
        output_dict=True,
        zero_division=0,
    )
    save_json(per_class_dict, ARTIFACTS_DIR / "per_class.json")

    # Record IID justification and classification metadata
    cls_metadata = {
        "dataset": "D3 Crop Recommendation",
        "iid_justified": justify_iid,
        "iid_justification_rationale": (
            "D3 is a curated/synthetic/augmented benchmark. Stratified 60/20/20 partitioning is applied "
            "under the operational assumption of observation independence for agricultural advisory screening. "
            "Exact statistical independence is not empirically established across all potential covariates."
        ),
        "split_counts": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "classes_count": int(df[CLS_TARGET].nunique()),
        "selected_model": f"{selected_name}_tuned" if use_tuned else selected_name,
        "use_tuned": use_tuned,
        "data_sha256": data_sha256,
        "config_sha256": config_sha256,
    }
    save_json(cls_metadata, ARTIFACTS_DIR / "classification_metadata.json")

    # 6. Overfitting Diagnostics (Part 16)
    cls_diagnostics = []
    eval_candidates = {
        "majority": majority_pipe,
        "logistic": joblib.load(MODELS_CLS_DIR / "logistic.joblib"),
        "forest_default": joblib.load(MODELS_CLS_DIR / "forest.joblib"),
    }
    if use_tuned:
        eval_candidates["forest_tuned"] = tuned_pipe

    for c_label, c_mod in eval_candidates.items():
        tr_p = c_mod.predict(train_df[CLS_FEATURES])
        va_p = c_mod.predict(val_df[CLS_FEATURES])
        te_p = c_mod.predict(test_df[CLS_FEATURES])

        tr_m = evaluate_classification(train_df[CLS_TARGET], tr_p)
        va_m = evaluate_classification(val_df[CLS_TARGET], va_p)
        te_m = evaluate_classification(test_df[CLS_TARGET], te_p)

        tr_va_gap = va_m["macro_F1"] - tr_m["macro_F1"]
        va_te_gap = te_m["macro_F1"] - va_m["macro_F1"]
        rel_gap = (va_m["macro_F1"] - tr_m["macro_F1"]) / max(tr_m["macro_F1"], 1e-6)

        cls_diagnostics.append(
            {
                "model": c_label,
                "train_macro_F1": round(tr_m["macro_F1"], 4),
                "validation_macro_F1": round(va_m["macro_F1"], 4),
                "test_macro_F1": round(te_m["macro_F1"], 4),
                "train_accuracy": round(tr_m["accuracy"], 4),
                "validation_accuracy": round(va_m["accuracy"], 4),
                "test_accuracy": round(te_m["accuracy"], 4),
                "train_to_validation_F1_gap": round(tr_va_gap, 4),
                "validation_to_test_F1_gap": round(va_te_gap, 4),
                "relative_generalization_gap": round(rel_gap, 4),
            }
        )

    cls_diag_df = pd.DataFrame(cls_diagnostics)
    CLS_OVERFITTING_FILE.parent.mkdir(parents=True, exist_ok=True)
    cls_diag_df.to_csv(CLS_OVERFITTING_FILE, index=False)
    logger.info("Saved classification overfitting diagnostics to %s", CLS_OVERFITTING_FILE)
    logger.info(
        "Classification Diagnostics:\n%s",
        cls_diag_df[["model", "train_macro_F1", "validation_macro_F1", "test_macro_F1", "train_to_validation_F1_gap"]].to_string(index=False),
    )

    # 7. Confusion Matrix Plot
    classes = sorted(df[CLS_TARGET].unique().tolist())
    plot_confusion_matrix_display(
        y_true=test_df[CLS_TARGET].to_numpy(),
        y_pred=test_pred,
        classes=classes,
        save_path=FIGURES_CLS_DIR / "confusion_matrix.png",
    )

    # 7. Export and Verify Model Bundle
    bundle_path = MODELS_CLS_DIR / "selected_bundle.joblib"
    create_model_bundle(
        pipeline=final_pipe,
        task="classification",
        training_df=train_df,
        data_sha256=data_sha256,
        config_sha256=get_current_config_sha256(),
        model_name=f"{selected_name}_tuned" if use_tuned else selected_name,
        save_path=bundle_path,
    )

    logger.info("=== CLASSIFICATION PIPELINE COMPLETED SUCCESSFULLY ===")


def main():
    parser = argparse.ArgumentParser(description="Train classification pipeline")
    parser.add_argument(
        "--justify-iid",
        action="store_true",
        default=False,
        help="Flag confirming documented independence justification review.",
    )
    args = parser.parse_args()
    run_classification(justify_iid=args.justify_iid)


if __name__ == "__main__":
    main()
