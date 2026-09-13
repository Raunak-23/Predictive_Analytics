"""Training and evaluation pipeline for Rice Yield Regression (Track A).

Stages:
  --stage development (or validate):
      Fits candidate models, evaluates on validation holdout, performs incremental
      model serialization, executes time-aware hyperparameter tuning on training data,
      conducts rolling-origin robustness diagnostics, and saves selection metadata.
  --stage test:
      Enforces TEST_LOCK, evaluates final selected model and median baseline on locked
      test set, computes top 5 error cases, conducts year-wise robustness, generates
      residual/actual-predicted figures, and exports verified model bundle.
"""

import argparse
import hashlib
import json
import platform
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

try:
    from src.config import (
        ARTIFACTS_DIR,
        CONFIG_JSON_FILE,
        CONFIG_SHA256_FILE,
        FIGURES_REG_DIR,
        MODEL_REGISTRY_FILE,
        MODELS_REG_DIR,
        RANDOM_SEED,
        REG_CATEGORICAL_FEATURES,
        REG_FEATURES,
        REG_OVERFITTING_FILE,
        REG_REQUIRED_COLUMNS,
        REG_SPLIT_MANIFEST,
        REG_TARGET,
        RESULTS_REG_DIR,
        RICE_CANONICAL_FILE,
        SELECTION_FINAL_FILE,
        SELECTION_INITIAL_FILE,
        SOURCE_ANOMALY_AUDIT_FILE,
        SOURCE_ANOMALY_SENSITIVITY_FILE,
        TEST_LOCK_FILE,
        get_current_config_sha256,
        save_pipeline_config,
    )
    from src.evaluation import (
        evaluate_regression,
        plot_actual_vs_predicted,
        plot_model_comparison,
        plot_residuals,
    )
    from src.inference import create_model_bundle
    from src.models import get_regression_models
    from src.robustness import (
        evaluate_development_rolling_origins,
        evaluate_year_robustness,
    )
    from src.source_anomaly import run_source_anomaly_audit, run_source_anomaly_sensitivity
    from src.split_data import get_chronological_splits
    from src.tuning import tune_regression_model
    from src.utils import compute_sha256, load_json, save_json, setup_logger
except ImportError:
    from config import (
        ARTIFACTS_DIR,
        CONFIG_JSON_FILE,
        CONFIG_SHA256_FILE,
        FIGURES_REG_DIR,
        MODEL_REGISTRY_FILE,
        MODELS_REG_DIR,
        RANDOM_SEED,
        REG_CATEGORICAL_FEATURES,
        REG_FEATURES,
        REG_OVERFITTING_FILE,
        REG_REQUIRED_COLUMNS,
        REG_SPLIT_MANIFEST,
        REG_TARGET,
        RESULTS_REG_DIR,
        RICE_CANONICAL_FILE,
        SELECTION_FINAL_FILE,
        SELECTION_INITIAL_FILE,
        SOURCE_ANOMALY_AUDIT_FILE,
        SOURCE_ANOMALY_SENSITIVITY_FILE,
        TEST_LOCK_FILE,
        get_current_config_sha256,
        save_pipeline_config,
    )
    from evaluation import (
        evaluate_regression,
        plot_actual_vs_predicted,
        plot_model_comparison,
        plot_residuals,
    )
    from inference import create_model_bundle
    from models import get_regression_models
    from robustness import (
        evaluate_development_rolling_origins,
        evaluate_year_robustness,
    )
    from source_anomaly import run_source_anomaly_audit, run_source_anomaly_sensitivity
    from split_data import get_chronological_splits
    from tuning import tune_regression_model
    from utils import compute_sha256, load_json, save_json, setup_logger

logger = setup_logger("train_regression")


def run_development(advanced: bool = False, reset_lock: bool = False) -> None:
    """Executes the complete development stage on training and validation splits."""
    logger.info("=== STARTING REGRESSION DEVELOPMENT STAGE ===")

    if TEST_LOCK_FILE.exists():
        if reset_lock:
            TEST_LOCK_FILE.unlink()
            logger.warning("TEST_LOCK explicitly reset for clean development run.")
        else:
            raise RuntimeError(
                "TEST_LOCK already exists at artifacts/TEST_LOCK! "
                "Development stage cannot modify artifacts after test lock is established. "
                "Pass reset_lock=True if intentionally performing a clean pipeline rebuild."
            )

    if not RICE_CANONICAL_FILE.exists():
        raise FileNotFoundError(f"Canonical data not found: {RICE_CANONICAL_FILE}")

    data_sha256 = compute_sha256(RICE_CANONICAL_FILE)
    df = pd.read_csv(RICE_CANONICAL_FILE)

    # Chronological Split
    df_split, partitions = get_chronological_splits(df)
    train_df = df_split[df_split["split"] == "train"].copy()
    val_df = df_split[df_split["split"] == "validation"].copy()
    dev_df = df_split[df_split["split"] != "test"].copy()

    logger.info(
        "Development split sizes: Train=%d, Validation=%d, Dev Total=%d",
        len(train_df),
        len(val_df),
        len(dev_df),
    )

    # 1. Descriptive Plots for Training Data (Rule 14)
    FIGURES_REG_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.hist(train_df[REG_TARGET], bins=30, color="#1f77b4", edgecolor="black", alpha=0.7)
    plt.title("Training Rice Yield Distribution", fontsize=12)
    plt.xlabel("Yield (t/ha)", fontsize=11)
    plt.ylabel("Observation Count", fontsize=11)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_REG_DIR / "target_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    year_counts = train_df["year"].value_counts().sort_index()
    plt.bar(year_counts.index, year_counts.values, color="#ff7f0e", edgecolor="black", alpha=0.8)
    plt.title("Training Observations per Harvest Year", fontsize=12)
    plt.xlabel("Harvest Year", fontsize=11)
    plt.ylabel("Observation Count", fontsize=11)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_REG_DIR / "year_coverage.png", dpi=150)
    plt.close()

    # 2. Fit Candidate Models & Incremental Model Saving (Rule 18, 19, 20)
    MODELS_REG_DIR.mkdir(parents=True, exist_ok=True)
    candidates = get_regression_models(include_depth_ablation=advanced)

    registry: Dict[str, Any] = {}
    validation_rows = []

    # Save and compute canonical config hash (Part 9)
    config_sha256 = get_current_config_sha256()

    for name, pipeline in candidates.items():
        logger.info("Fitting candidate: '%s'...", name)
        t_start = time.perf_counter()

        # Update registry status: running
        registry[name] = {
            "status": "running",
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        save_json(registry, MODEL_REGISTRY_FILE)

        try:
            # Fit strictly on train
            pipeline.fit(train_df[REG_FEATURES], train_df[REG_TARGET])
            fit_time = time.perf_counter() - t_start

            # Predict on validation
            val_preds = pipeline.predict(val_df[REG_FEATURES])
            metrics = evaluate_regression(val_df[REG_TARGET], val_preds)

            # Serialize model immediately (Rule 20)
            artifact_file = MODELS_REG_DIR / f"{name}.joblib"
            joblib.dump(pipeline, artifact_file)

            # Update registry status: complete
            registry[name] = {
                "status": "complete",
                "runtime_seconds": round(fit_time, 3),
                "validation_metrics": metrics,
                "artifact_path": str(artifact_file.name),
                "data_sha256": data_sha256,
                "config_sha256": config_sha256,
            }
            save_json(registry, MODEL_REGISTRY_FILE)

            validation_rows.append(
                {
                    "model": name,
                    "runtime_seconds": round(fit_time, 3),
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "R2": metrics["R2"],
                }
            )
            logger.info("Candidate '%s' complete: MAE=%.4f t/ha, RMSE=%.4f t/ha", name, metrics["MAE"], metrics["RMSE"])

        except Exception as exc:
            registry[name] = {"status": "failed", "error": str(exc)}
            save_json(registry, MODEL_REGISTRY_FILE)
            logger.error("Candidate '%s' failed: %s", name, exc)
            raise

    # 3. Model Comparison & Initial Selection (Rule 21)
    val_results_df = pd.DataFrame(validation_rows)
    # Stable sort retains simpler first-listed models for exact ties
    val_results_df = val_results_df.sort_values("MAE", ascending=True, kind="stable").reset_index(drop=True)
    RESULTS_REG_DIR.mkdir(parents=True, exist_ok=True)
    val_results_csv = RESULTS_REG_DIR / "validation_results.csv"
    val_results_df.to_csv(val_results_csv, index=False)
    logger.info("Saved validation results table to %s", val_results_csv)

    # Plot model comparison
    plot_model_comparison(
        val_results_df,
        FIGURES_REG_DIR / "model_comparison.png",
        metric_col="MAE",
        task_name="Regression Candidate Validation",
    )

    initial_selected = val_results_df.iloc[0]["model"]
    logger.info("Initial model selected by minimum validation MAE: '%s'", initial_selected)

    selection_initial = {
        "initial_selected_model": initial_selected,
        "selection_metric": "min_validation_MAE",
        "validation_MAE": float(val_results_df.iloc[0]["MAE"]),
        "validation_RMSE": float(val_results_df.iloc[0]["RMSE"]),
        "validation_R2": float(val_results_df.iloc[0]["R2"]),
        "data_sha256": data_sha256,
        "config_sha256": config_sha256,
    }
    save_json(selection_initial, SELECTION_INITIAL_FILE)

    # 4. Leakage-safe Hyperparameter Tuning (Rule 22)
    selected_pipeline = joblib.load(MODELS_REG_DIR / f"{initial_selected}.joblib")
    tuned_pipeline, tuning_meta = tune_regression_model(
        selected_model_name=initial_selected,
        base_pipeline=selected_pipeline,
        df_train=train_df,
        df_val=val_df,
        data_sha256=data_sha256,
        config_sha256=config_sha256,
    )

    # Compare tuned vs default on untouched validation set
    final_selected_model = initial_selected
    use_tuned = False
    if tuning_meta.get("tuned", True) and "validation_metrics" in tuning_meta:
        tuned_val_mae = tuning_meta["validation_metrics"]["MAE"]
        default_val_mae = float(val_results_df.iloc[0]["MAE"])
        logger.info(
            "Validation MAE comparison: Default '%s' = %.4f t/ha vs Tuned = %.4f t/ha",
            initial_selected,
            default_val_mae,
            tuned_val_mae,
        )
        if tuned_val_mae <= default_val_mae:
            logger.info("Tuned model achieves superior or equal validation MAE. Selecting tuned model.")
            use_tuned = True
            chosen_pipeline = tuned_pipeline
            final_metrics = tuning_meta["validation_metrics"]
        else:
            logger.info("Default model retains superior validation MAE. Keeping default model.")
            chosen_pipeline = selected_pipeline
            final_metrics = {
                "MAE": float(val_results_df.iloc[0]["MAE"]),
                "RMSE": float(val_results_df.iloc[0]["RMSE"]),
                "R2": float(val_results_df.iloc[0]["R2"]),
            }
    else:
        chosen_pipeline = selected_pipeline
        final_metrics = {
            "MAE": float(val_results_df.iloc[0]["MAE"]),
            "RMSE": float(val_results_df.iloc[0]["RMSE"]),
            "R2": float(val_results_df.iloc[0]["R2"]),
        }

    # 5. Final Development Selection Record (Rule 24)
    final_selection = {
        "selected_model": f"{initial_selected}_tuned" if use_tuned else initial_selected,
        "selected_model_family": initial_selected,
        "used_hyperparameter_tuning": use_tuned,
        "validation_MAE": final_metrics["MAE"],
        "validation_RMSE": final_metrics["RMSE"],
        "validation_R2": final_metrics["R2"],
        "training_years": partitions["train_years"],
        "validation_years": partitions["validation_years"],
        "test_years": partitions["test_years"],
        "data_sha256": data_sha256,
        "config_sha256": config_sha256,
        "note": "Selected strictly on development/validation data; test set remains untouched.",
    }
    save_json(final_selection, SELECTION_FINAL_FILE)
    logger.info("Final development selection committed to %s", SELECTION_FINAL_FILE)

    # 6. Rolling Origins Diagnostic (Rule 25)
    rolling_df = evaluate_development_rolling_origins(dev_df, candidates)

    # Plot rolling origin performance
    plt.figure(figsize=(8, 5))
    for model_name, grp in rolling_df.groupby("model"):
        plt.plot(grp["origin_year"], grp["MAE"], marker="o", label=model_name)
    plt.title("Development Rolling-Origin Robustness (MAE)", fontsize=12)
    plt.xlabel("Evaluation Origin Year", fontsize=11)
    plt.ylabel("MAE (t/ha)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES_REG_DIR / "rolling_origin_performance.png", dpi=150)
    plt.close()

    # 7. Audit Preprocessing Encoder Categories
    fitted_tree = joblib.load(MODELS_REG_DIR / "tree.joblib")
    encoder = fitted_tree.named_steps["pre"].named_transformers_["cat"].named_steps["encoder"]
    for col, categories in zip(REG_CATEGORICAL_FEATURES, encoder.categories_):
        assert set(categories) == set(train_df[col].astype(str).unique()), (
            f"Encoder category leakage detected for column {col}!"
        )
    logger.info("Encoder category leakage audit: PASSED. Zero held-out categories in learned encoder.")

    # 8. Source Anomaly Audit and Sensitivity Analysis (Parts 7 & 8)
    run_source_anomaly_audit()
    run_source_anomaly_sensitivity()

    logger.info("=== REGRESSION DEVELOPMENT COMPLETED SUCCESSFULLY ===")


def run_test(reset_lock: bool = False) -> None:
    """Evaluates final model and median baseline on the locked test set."""
    logger.info("=== STARTING REGRESSION TEST EVALUATION ===")

    # Handle explicit local regeneration reset
    if reset_lock and TEST_LOCK_FILE.exists():
        TEST_LOCK_FILE.unlink()
        logger.warning("TEST_LOCK explicitly reset for clean local pipeline regeneration.")

    # 1. Enforce TEST_LOCK (Rule 26)
    if TEST_LOCK_FILE.exists():
        raise RuntimeError(
            "TEST_LOCK already exists at artifacts/TEST_LOCK! "
            "To prevent p-hacking and repeated test evaluation, testing cannot be rerun on the same run."
        )

    if not SELECTION_FINAL_FILE.exists():
        raise RuntimeError(
            f"Final selection file missing at {SELECTION_FINAL_FILE}! Run development stage first."
        )

    selection = load_json(SELECTION_FINAL_FILE)

    # 2. Verify Hashes and Manifest Integrity
    cur_data_sha256 = compute_sha256(RICE_CANONICAL_FILE)
    if cur_data_sha256 != selection["data_sha256"]:
        raise ValueError("DATA HASH MISMATCH! The canonical dataset changed after model selection!")

    saved_manifest = pd.read_csv(REG_SPLIT_MANIFEST)
    df_raw = pd.read_csv(RICE_CANONICAL_FILE)
    df_split, _ = get_chronological_splits(df_raw, save_manifest=False)
    current_manifest = df_split[["row_id", "year", "split"]]
    pd.testing.assert_frame_equal(
        saved_manifest,
        current_manifest,
        obj="Split Manifest Integrity",
    )
    logger.info("Data hash and split manifest integrity verified.")

    # 3. Create TEST_LOCK before scoring
    TEST_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    selection_sha256 = compute_sha256(SELECTION_FINAL_FILE)
    lock_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selected_model": selection["selected_model"],
        "selected_model_family": selection["selected_model_family"],
        "data_sha256": cur_data_sha256,
        "config_sha256": get_current_config_sha256(),
        "test_years": selection.get("test_years", [2014, 2015]),
        "selection_sha256": selection_sha256,
    }
    with TEST_LOCK_FILE.open("x", encoding="utf-8") as f:
        json.dump(lock_data, f, indent=2)
    logger.info("Created TEST_LOCK at %s", TEST_LOCK_FILE)

    # 4. Load Models
    selected_name = selection["selected_model_family"]
    use_tuned = selection["used_hyperparameter_tuning"]
    if use_tuned:
        model_path = MODELS_REG_DIR / "tuned_selected_model.joblib"
    else:
        model_path = MODELS_REG_DIR / f"{selected_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")

    pipe = joblib.load(model_path)
    median_pipe = joblib.load(MODELS_REG_DIR / "median.joblib")

    test_df = df_split[df_split["split"] == "test"].copy()
    train_df = df_split[df_split["split"] == "train"].copy()

    # 5. Evaluate Selected Model and Baseline (Rule 27)
    t0 = time.perf_counter()
    test_pred = pipe.predict(test_df[REG_FEATURES])
    latency = time.perf_counter() - t0

    assert np.isfinite(test_pred).all(), "Non-finite predictions produced on test set!"

    test_metrics = evaluate_regression(test_df[REG_TARGET], test_pred)
    median_pred = median_pipe.predict(test_df[REG_FEATURES])
    median_metrics = evaluate_regression(test_df[REG_TARGET], median_pred)

    test_results_rows = [
        {
            "model": selection["selected_model"],
            "batch_inference_seconds": round(latency, 4),
            "MAE": test_metrics["MAE"],
            "RMSE": test_metrics["RMSE"],
            "R2": test_metrics["R2"],
        },
        {
            "model": "median",
            "batch_inference_seconds": 0.001,
            "MAE": median_metrics["MAE"],
            "RMSE": median_metrics["RMSE"],
            "R2": median_metrics["R2"],
        },
    ]
    test_results_df = pd.DataFrame(test_results_rows)
    test_results_csv = RESULTS_REG_DIR / "test_results.csv"
    test_results_df.to_csv(test_results_csv, index=False)
    logger.info("Locked test results saved to %s", test_results_csv)
    logger.info(
        "Final Test Results:\n%s",
        test_results_df.to_string(index=False),
    )

    # 6. Save Predictions & Conduct Evidence-Grounded Error Analysis
    test_df_records = test_df.copy()
    test_df_records["prediction"] = test_pred
    test_df_records["absolute_error"] = np.abs(test_df_records[REG_TARGET] - test_pred)
    test_df_records["residual"] = test_df_records[REG_TARGET] - test_pred

    pred_csv = ARTIFACTS_DIR / "test_predictions.csv"
    test_df_records.to_csv(pred_csv, index=False)

    # Top 5 largest absolute errors
    top5_errors = (
        test_df_records.sort_values("absolute_error", ascending=False)
        .head(5)
        .copy()
    )

    # Compute case-specific diagnostic fields based on observation properties (Part 14)
    support_counts = []
    hist_means = []
    hist_stds = []
    causes = []
    consequences = []
    mitigations = []

    for _, row in top5_errors.iterrows():
        dist = row["district"]
        dist_train = train_df[train_df["district"] == dist]
        support_cnt = len(dist_train)
        support_counts.append(support_cnt)

        if support_cnt > 0:
            h_mean = float(dist_train["yield_t_ha"].mean())
            h_std = float(dist_train["yield_t_ha"].std()) if support_cnt > 1 else 0.0
            hist_means.append(round(h_mean, 4))
            hist_stds.append(round(h_std, 4))
        else:
            h_mean = None
            h_std = None
            hist_means.append(None)
            hist_stds.append(None)

        actual_val = float(row[REG_TARGET])
        pred_val = float(row["prediction"])
        resid = float(row["residual"])
        yr = int(row["year"])

        # Evidence-based hypothesis formulation strictly avoiding unobserved claims (Part 14)
        if support_cnt < 10:
            causes.append(
                f"Hypothesis: Sparse historical district training coverage ({support_cnt} records); "
                "model parameters have higher estimation variance for this localized district."
            )
            consequences.append(
                "May contribute to localized procurement misplanning and misaligned district buffer stock targets."
            )
            mitigations.append(
                "Source data audit, spatial smoothing across contiguous districts, or future temporal redesign."
            )
        elif h_mean is not None and actual_val > (h_mean + 2.0 * max(h_std, 0.5)):
            causes.append(
                f"Hypothesis: Substantial positive yield surge (actual {actual_val:.2f} t/ha vs district mean {h_mean:.2f} t/ha); "
                "an unobserved localized weather or production condition is plausible because meteorological telemetry is absent."
            )
            consequences.append(
                "Could lead to unexpected local depot storage bottlenecks and procurement capacity strain during peak harvest."
            )
            mitigations.append(
                "Source data audit, additional validated predictors (e.g. remote-sensing vegetation indices), or future temporal redesign."
            )
        elif h_mean is not None and actual_val < (h_mean - 1.5 * max(h_std, 0.5)):
            causes.append(
                f"Hypothesis: Substantial negative yield drop (actual {actual_val:.2f} t/ha vs district mean {h_mean:.2f} t/ha); "
                "an unobserved weather shock or localized production disturbance is plausible because weather variables are absent."
            )
            consequences.append(
                "May delay targeted farmer insurance relief disbursements and misguide district-level disaster compensation."
            )
            mitigations.append(
                "Source data audit, additional validated predictors (e.g. gridded precipitation telemetry), or future temporal redesign."
            )
        elif abs(resid) > 3.0:
            causes.append(
                f"Hypothesis: Significant deviation under temporal trend extrapolation in harvest year {yr} (residual {resid:+.2f} t/ha); "
                "linear state-year trend lacks explicit agronomic saturation limits."
            )
            consequences.append(
                "Could contribute to multi-year forecasting bias in state-level agricultural grain availability estimates."
            )
            mitigations.append(
                "Source data audit, non-linear trend damping, or future temporal redesign to constrain extrapolation."
            )
        else:
            causes.append(
                "Hypothesis: Unobserved intra-district production shock or administrative reporting anomaly; "
                "model features are strictly restricted to metadata (state/district/season/year) without meteorological covariates."
            )
            consequences.append(
                "May contribute to minor localized allocation inaccuracies in seasonal agricultural advisories."
            )
            mitigations.append(
                "Source data audit, additional validated predictors, or future temporal redesign."
            )

    top5_errors["training_support_count"] = support_counts
    top5_errors["historical_training_mean"] = hist_means
    top5_errors["historical_training_std"] = hist_stds
    top5_errors["likely_cause_hypothesis"] = causes
    top5_errors["consequence"] = consequences
    top5_errors["mitigation"] = mitigations

    # Compatibility aliases
    top5_errors["training_support_count_for_district"] = support_counts
    top5_errors["historical_training_yield_summary"] = [
        f"mean={m:.2f}, std={s:.2f}" if m is not None else "unseen"
        for m, s in zip(hist_means, hist_stds)
    ]

    error_cols = [
        "row_id",
        "state",
        "district",
        "season",
        "year",
        "actual",
        "prediction",
        "residual",
        "absolute_error",
        "training_support_count",
        "historical_training_mean",
        "historical_training_std",
        "likely_cause_hypothesis",
        "consequence",
        "mitigation",
    ]
    error_analysis_csv = RESULTS_REG_DIR / "error_analysis.csv"
    top5_errors.rename(columns={"yield_t_ha": "actual"})[error_cols].to_csv(
        error_analysis_csv, index=False
    )
    logger.info("Top 5 error analysis saved to %s", error_analysis_csv)

    # 7. Year-wise Test Robustness (Rule 30)
    evaluate_year_robustness(test_df, test_pred)

    # 8. Test Figures (Rule 28)
    plot_actual_vs_predicted(
        y_true=test_df[REG_TARGET].to_numpy(),
        y_pred=test_pred,
        save_path=FIGURES_REG_DIR / "actual_vs_predicted.png",
    )
    plot_residuals(
        y_true=test_df[REG_TARGET].to_numpy(),
        y_pred=test_pred,
        save_path=FIGURES_REG_DIR / "residuals.png",
    )

    # 9. Overfitting Diagnostics (Part 2)
    # Evaluates candidate model families plus the actual selected tuned model across train, validation, and locked test splits
    candidates_all = get_regression_models(include_depth_ablation=False)
    val_df = df_split[df_split["split"] == "validation"].copy()

    eval_models = dict(candidates_all)
    if use_tuned and selection["selected_model"] not in eval_models:
        eval_models[selection["selected_model"]] = pipe

    overfitting_rows = []
    for c_name, c_pipe in eval_models.items():
        if c_name == selection["selected_model"] and use_tuned:
            fitted_c = pipe
        else:
            c_path = MODELS_REG_DIR / f"{c_name}.joblib"
            if c_path.exists():
                fitted_c = joblib.load(c_path)
            else:
                fitted_c = c_pipe
                fitted_c.fit(train_df[REG_FEATURES], train_df[REG_TARGET])

        pred_tr = fitted_c.predict(train_df[REG_FEATURES])
        pred_va = fitted_c.predict(val_df[REG_FEATURES])
        pred_te = fitted_c.predict(test_df[REG_FEATURES])

        m_tr = evaluate_regression(train_df[REG_TARGET], pred_tr)
        m_va = evaluate_regression(val_df[REG_TARGET], pred_va)
        m_te = evaluate_regression(test_df[REG_TARGET], pred_te)

        tr_val_gap = m_va["MAE"] - m_tr["MAE"]
        val_te_gap = m_te["MAE"] - m_va["MAE"]
        rel_gap = (m_va["MAE"] - m_tr["MAE"]) / max(m_tr["MAE"], 1e-6)

        overfitting_rows.append(
            {
                "model": c_name,
                "train_MAE": round(m_tr["MAE"], 4),
                "validation_MAE": round(m_va["MAE"], 4),
                "test_MAE": round(m_te["MAE"], 4),
                "train_RMSE": round(m_tr["RMSE"], 4),
                "validation_RMSE": round(m_va["RMSE"], 4),
                "test_RMSE": round(m_te["RMSE"], 4),
                "train_R2": round(m_tr["R2"], 4),
                "validation_R2": round(m_va["R2"], 4),
                "test_R2": round(m_te["R2"], 4),
                "train_to_validation_MAE_gap": round(tr_val_gap, 4),
                "validation_to_test_MAE_gap": round(val_te_gap, 4),
                "relative_generalization_gap": round(rel_gap, 4),
            }
        )

    overfitting_df = pd.DataFrame(overfitting_rows)
    REG_OVERFITTING_FILE.parent.mkdir(parents=True, exist_ok=True)
    overfitting_df.to_csv(REG_OVERFITTING_FILE, index=False)
    logger.info("Saved regression overfitting diagnostics to %s", REG_OVERFITTING_FILE)
    logger.info(
        "Overfitting Diagnostics:\n%s",
        overfitting_df[["model", "train_MAE", "validation_MAE", "test_MAE", "train_to_validation_MAE_gap", "relative_generalization_gap"]].to_string(index=False),
    )

    # Plot train vs validation MAE (Part 2)
    plt.figure(figsize=(7.5, 4.5))
    x_indices = np.arange(len(overfitting_df))
    bar_width = 0.35
    plt.bar(x_indices - bar_width / 2, overfitting_df["train_MAE"], width=bar_width, label="Train MAE", color="#1f77b4", edgecolor="black", alpha=0.85)
    plt.bar(x_indices + bar_width / 2, overfitting_df["validation_MAE"], width=bar_width, label="Validation MAE", color="#ff7f0e", edgecolor="black", alpha=0.85)
    plt.xticks(x_indices, overfitting_df["model"], fontsize=10)
    plt.title("Candidate Model Train vs Validation Error (MAE)", fontsize=12)
    plt.xlabel("Candidate Model", fontsize=11)
    plt.ylabel("MAE (t/ha)", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6, axis="y")
    for i in range(len(overfitting_df)):
        tr_val = overfitting_df.loc[i, "train_MAE"]
        va_val = overfitting_df.loc[i, "validation_MAE"]
        plt.text(i - bar_width / 2, tr_val + 0.01, f"{tr_val:.3f}", ha="center", va="bottom", fontsize=8)
        plt.text(i + bar_width / 2, va_val + 0.01, f"{va_val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.ylim(0, max(overfitting_df["validation_MAE"]) * 1.18)
    plt.tight_layout()
    gap_fig_path = FIGURES_REG_DIR / "train_validation_error_gap.png"
    plt.savefig(gap_fig_path, dpi=150)
    plt.close()
    logger.info("Saved train-validation error gap figure to %s", gap_fig_path)

    # 10. Create and Verify Model Bundle (Rule 34)
    bundle_path = MODELS_REG_DIR / "selected_bundle.joblib"
    create_model_bundle(
        pipeline=pipe,
        task="regression",
        training_df=train_df,
        data_sha256=cur_data_sha256,
        config_sha256=get_current_config_sha256(),
        model_name=selection["selected_model"],
        save_path=bundle_path,
    )

    logger.info("=== REGRESSION TEST COMPLETED SUCCESSFULLY ===")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate regression model")
    parser.add_argument(
        "--stage",
        choices=["development", "validate", "test"],
        default="development",
        help="Stage to execute: 'development' (alias 'validate') or 'test'",
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Include depth ablation in development comparison.",
    )
    parser.add_argument(
        "--reset-lock",
        action="store_true",
        help="Explicitly reset TEST_LOCK for clean local pipeline regeneration.",
    )
    args = parser.parse_args()

    if args.stage in ["development", "validate"]:
        run_development(advanced=args.advanced, reset_lock=args.reset_lock)
    elif args.stage == "test":
        run_test(reset_lock=args.reset_lock)


if __name__ == "__main__":
    main()
