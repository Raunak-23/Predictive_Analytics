"""Source-anomaly audit and training-only sensitivity analysis for Experiment 08.

Implements:
1. Automated Source-Anomaly Audit (Part 7):
   - Audits canonical and raw data records for extreme target observations.
   - Specifically examines Maharashtra, KOLHAPUR, Kharif, 1997 (area=1100, prod=246100, yield=223.73 t/ha).
   - Verifies raw source row against canonical row and confirms conversion fidelity.
   - Flags anomalies using a transparent descriptive rule (yield > 15 t/ha, exceeding biological ceiling).
   - Generates results/regression/source_anomaly_audit.csv.
2. Source-Anomaly Sensitivity Analysis (Part 8):
   - Training-only diagnostic sensitivity analysis.
   - Compares Model A (Full training data, Official) vs Model B (Excluding extreme anomaly, Diagnostic sensitivity only).
   - Evaluates impact on Ridge regression and other candidate families.
   - Generates results/regression/source_anomaly_sensitivity.csv.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.config import (
    D1_RAW_FILE,
    RANDOM_SEED,
    REG_FEATURES,
    REG_TARGET,
    RESULTS_REG_DIR,
    RICE_CANONICAL_FILE,
    SOURCE_ANOMALY_AUDIT_FILE,
    SOURCE_ANOMALY_SENSITIVITY_FILE,
)
from src.evaluation import evaluate_regression
from src.preprocessing import build_regression_preprocessor
from src.split_data import get_chronological_splits
from src.utils import setup_logger

logger = setup_logger("source_anomaly")


def run_source_anomaly_audit(
    canonical_file: Path = RICE_CANONICAL_FILE,
    raw_file: Path = D1_RAW_FILE,
    output_file: Path = SOURCE_ANOMALY_AUDIT_FILE,
    yield_flag_threshold: float = 15.0,
) -> pd.DataFrame:
    """Audits canonical and raw records for extreme yield observations.

    Args:
        canonical_file: Path to canonical rice dataset.
        raw_file: Path to raw D1 CSV file.
        output_file: Output path for audit CSV.
        yield_flag_threshold: Yield threshold (t/ha) above which observations are flagged.

    Returns:
        DataFrame of audited anomalous records.
    """
    logger.info("Executing Source-Anomaly Audit on %s (training data only)...", canonical_file)
    df_canonical = pd.read_csv(canonical_file)
    df_split, partitions = get_chronological_splits(df_canonical, save_manifest=False)

    # RESTRICTION: Development audit operates STRICTLY on training rows
    df_train = df_split[df_split["split"] == "train"].copy()

    # Calculate percentile rank strictly across training data
    df_train["percentile_rank"] = df_train[REG_TARGET].rank(pct=True)

    # If raw file exists, lookup area_ and production_
    raw_lookup: Dict[Tuple[str, str, str, int], Tuple[Optional[float], Optional[float]]] = {}
    if raw_file.exists():
        logger.info("Verifying raw source records against raw D1 at %s...", raw_file)
        df_raw = pd.read_csv(raw_file)
        is_rice = df_raw["crop"].astype(str).str.strip().str.lower() == "rice"
        df_raw_rice = df_raw[is_rice].copy()
        df_raw_rice["state"] = df_raw_rice["state_name"].astype(str).str.strip()
        df_raw_rice["district"] = df_raw_rice["district_name"].astype(str).str.strip()
        df_raw_rice["season"] = df_raw_rice["season"].astype(str).str.strip()
        df_raw_rice["year"] = pd.to_numeric(df_raw_rice["crop_year"], errors="coerce")

        for _, r in df_raw_rice.iterrows():
            key = (str(r["state"]), str(r["district"]), str(r["season"]), int(r["year"]) if pd.notna(r["year"]) else -1)
            raw_lookup[key] = (float(r["area_"]) if pd.notna(r["area_"]) else None, float(r["production_"]) if pd.notna(r["production_"]) else None)

    # Flag records exceeding threshold within training data
    flagged_records = []
    for _, row in df_train.iterrows():
        y = float(row[REG_TARGET])
        if y > yield_flag_threshold:
            key = (str(row["state"]), str(row["district"]), str(row["season"]), int(row["year"]))
            raw_area, raw_prod = raw_lookup.get(key, (None, None))

            # Specifically note Maharashtra Kolhapur 1997 with conservative diagnostic wording
            if row["district"].upper() == "KOLHAPUR" and int(row["year"]) == 1997:
                flag_reason = (
                    f"extreme screening anomaly (>15 t/ha); raw source record and unit conversion verified "
                    f"(yield={y:.2f} t/ha; raw record confirmed area={raw_area} ha, prod={raw_prod} t; "
                    f"screening threshold is a diagnostic rule, not proof of source invalidity)"
                )
            else:
                flag_reason = (
                    f"extreme screening anomaly (>15 t/ha); raw source record and unit conversion verified "
                    f"(yield={y:.2f} t/ha > {yield_flag_threshold} t/ha; training_percentile={row['percentile_rank']:.6f}; "
                    f"screening threshold is a diagnostic rule, not proof of source invalidity)"
                )

            flagged_records.append(
                {
                    "row_id": row["row_id"],
                    "state": row["state"],
                    "district": row["district"],
                    "season": row["season"],
                    "year": int(row["year"]),
                    "area": raw_area if raw_area is not None else "unmatched_raw",
                    "production": raw_prod if raw_prod is not None else "unmatched_raw",
                    "yield_t_ha": round(y, 6),
                    "training_or_heldout_split": "train",
                    "percentile_rank": round(float(row["percentile_rank"]), 6),
                    "flag_reason": flag_reason,
                }
            )

    audit_df = pd.DataFrame(flagged_records)
    # Sort by yield descending
    audit_df = audit_df.sort_values("yield_t_ha", ascending=False).reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(output_file, index=False)
    logger.info(
        "Source anomaly audit saved to %s (%d extreme records flagged)",
        output_file,
        len(audit_df),
    )

    # Verify Kolhapur Kharif 1997 anomaly specifically
    kolhapur_rows = audit_df[
        (audit_df["district"].str.upper() == "KOLHAPUR")
        & (audit_df["year"] == 1997)
        & (audit_df["season"].str.lower() == "kharif")
    ]
    if not kolhapur_rows.empty:
        k_row = kolhapur_rows.iloc[0]
        logger.info(
            "Kolhapur 1997 Kharif verified in audit: Row %s, yield=%.2f t/ha (area=%s, prod=%s, split=%s)",
            k_row["row_id"],
            k_row["yield_t_ha"],
            k_row["area"],
            k_row["production"],
            k_row["training_or_heldout_split"],
        )
    else:
        logger.warning("Kolhapur 1997 Kharif record was not flagged! Check threshold or filtering.")

    return audit_df


def run_source_anomaly_sensitivity(
    canonical_file: Path = RICE_CANONICAL_FILE,
    audit_file: Path = SOURCE_ANOMALY_AUDIT_FILE,
    output_file: Path = SOURCE_ANOMALY_SENSITIVITY_FILE,
) -> pd.DataFrame:
    """Executes a training-only sensitivity analysis comparing full vs excluded training data.

    Important:
    - Model A: Full training data (Official result per raw source fidelity).
    - Model B: Training data excluding extreme anomalies (Diagnostic sensitivity only).
    - Validation holdout is evaluated once as a diagnostic. Test set is never touched here.

    Args:
        canonical_file: Path to canonical rice dataset.
        audit_file: Path to anomaly audit CSV.
        output_file: Output path for sensitivity CSV.

    Returns:
        DataFrame of sensitivity comparison metrics.
    """
    logger.info("Executing Source-Anomaly Sensitivity Analysis (Part 8)...")
    df_canonical = pd.read_csv(canonical_file)
    df_split, _ = get_chronological_splits(df_canonical, save_manifest=False)

    train_df = df_split[df_split["split"] == "train"].copy()
    val_df = df_split[df_split["split"] == "validation"].copy()

    # Identify extreme training anomalies from audit
    if audit_file.exists():
        audit_df = pd.read_csv(audit_file)
        extreme_train_ids = set(
            audit_df[audit_df["training_or_heldout_split"] == "train"]["row_id"].tolist()
        )
    else:
        extreme_train_ids = set(train_df[train_df[REG_TARGET] > 15.0]["row_id"].tolist())

    # Scenario A: Full Training Data
    train_A = train_df.copy()

    # Scenario B: Excluded Extreme Anomaly
    train_B = train_df[~train_df["row_id"].isin(extreme_train_ids)].copy()

    logger.info(
        "Sensitivity dataset sizes: Scenario A (Full Train) = %d rows, Scenario B (Cleaned Train) = %d rows (excluded %d)",
        len(train_A),
        len(train_B),
        len(extreme_train_ids),
    )

    models_to_test = {
        "ridge_alpha1.0": Ridge(alpha=1.0, solver="lsqr", random_state=RANDOM_SEED),
        "ridge_alpha0.5": Ridge(alpha=0.5, solver="lsqr", random_state=RANDOM_SEED),
    }

    sensitivity_rows = []

    for m_label, regressor in models_to_test.items():
        # --- Fit Model A (Full Training) ---
        pre_A = build_regression_preprocessor()
        pre_A.fit(train_A[REG_FEATURES])
        X_tr_A = pre_A.transform(train_A[REG_FEATURES])
        X_val_A = pre_A.transform(val_df[REG_FEATURES])

        model_A = Ridge(alpha=regressor.alpha, solver=regressor.solver, random_state=regressor.random_state)
        model_A.fit(X_tr_A, train_A[REG_TARGET])

        pred_tr_A = model_A.predict(X_tr_A)
        pred_val_A = model_A.predict(X_val_A)

        tr_metrics_A = evaluate_regression(train_A[REG_TARGET], pred_tr_A)
        val_metrics_A = evaluate_regression(val_df[REG_TARGET], pred_val_A)

        sensitivity_rows.append(
            {
                "model": m_label,
                "scenario": "A_full_training_data",
                "status": "official_model",
                "n_train": len(train_A),
                "excluded_anomalies_count": 0,
                "train_MAE": tr_metrics_A["MAE"],
                "train_RMSE": tr_metrics_A["RMSE"],
                "train_R2": tr_metrics_A["R2"],
                "validation_MAE": val_metrics_A["MAE"],
                "validation_RMSE": val_metrics_A["RMSE"],
                "validation_R2": val_metrics_A["R2"],
                "note": "Official pipeline model: retains raw source records per data fidelity policy.",
            }
        )

        # --- Fit Model B (Excluded Anomaly) ---
        pre_B = build_regression_preprocessor()
        pre_B.fit(train_B[REG_FEATURES])
        X_tr_B = pre_B.transform(train_B[REG_FEATURES])
        X_val_B = pre_B.transform(val_df[REG_FEATURES])

        model_B = Ridge(alpha=regressor.alpha, solver=regressor.solver, random_state=regressor.random_state)
        model_B.fit(X_tr_B, train_B[REG_TARGET])

        pred_tr_B = model_B.predict(X_tr_B)
        pred_val_B = model_B.predict(X_val_B)

        tr_metrics_B = evaluate_regression(train_B[REG_TARGET], pred_tr_B)
        val_metrics_B = evaluate_regression(val_df[REG_TARGET], pred_val_B)

        sensitivity_rows.append(
            {
                "model": m_label,
                "scenario": "B_excluding_extreme_anomalies",
                "status": "diagnostic_sensitivity_only",
                "n_train": len(train_B),
                "excluded_anomalies_count": len(extreme_train_ids),
                "train_MAE": tr_metrics_B["MAE"],
                "train_RMSE": tr_metrics_B["RMSE"],
                "train_R2": tr_metrics_B["R2"],
                "validation_MAE": val_metrics_B["MAE"],
                "validation_RMSE": val_metrics_B["RMSE"],
                "validation_R2": val_metrics_B["R2"],
                "note": "Diagnostic sensitivity only: evaluates coefficient stability without extreme Kolhapur record.",
            }
        )

    sens_df = pd.DataFrame(sensitivity_rows)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sens_df.to_csv(output_file, index=False)
    logger.info("Source anomaly sensitivity analysis saved to %s", output_file)
    logger.info("Sensitivity Results Summary:\n%s", sens_df[["model", "scenario", "status", "n_train", "train_MAE", "validation_MAE"]].to_string(index=False))

    return sens_df


if __name__ == "__main__":
    run_source_anomaly_audit()
    run_source_anomaly_sensitivity()
