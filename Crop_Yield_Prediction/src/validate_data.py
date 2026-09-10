"""Data validation module for Experiment 08.

Enforces strict integrity, schema, range, uniqueness, and leakage constraints
for both Regression (Track A) and Classification (Track B) canonical datasets.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd

try:
    from src.config import (
        CLS_FEATURES,
        CLS_REQUIRED_COLUMNS,
        CLS_TARGET,
        CROP_LABELS_CANONICAL_FILE,
        DATA_VAL_DIR,
        MIN_REQUIRED_YEARS,
        REG_FEATURES,
        REG_REQUIRED_COLUMNS,
        REG_TARGET,
        RICE_CANONICAL_FILE,
    )
    from src.utils import compute_sha256, save_json, setup_logger
except ImportError:
    from config import (
        CLS_FEATURES,
        CLS_REQUIRED_COLUMNS,
        CLS_TARGET,
        CROP_LABELS_CANONICAL_FILE,
        DATA_VAL_DIR,
        MIN_REQUIRED_YEARS,
        REG_FEATURES,
        REG_REQUIRED_COLUMNS,
        REG_TARGET,
        RICE_CANONICAL_FILE,
    )
    from utils import compute_sha256, save_json, setup_logger

logger = setup_logger("validate_data")


def validate_regression_data(file_path: Path = RICE_CANONICAL_FILE) -> Dict[str, Any]:
    """Validates canonical rice yield regression dataset against all strict constraints."""
    if not file_path.exists():
        raise FileNotFoundError(f"Canonical rice file not found: {file_path}")

    logger.info("Validating regression dataset: %s", file_path)
    df = pd.read_csv(file_path)

    # 1. Required columns check
    missing_cols = set(REG_REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing canonical regression columns: {missing_cols}")

    # 2. Leakage safeguard: ensure no post-harvest or direct yield reconstructing columns
    forbidden_cols = ["production", "area", "production_", "area_", "production_t", "area_ha"]
    found_forbidden = [c for c in forbidden_cols if c in df.columns]
    if found_forbidden:
        raise ValueError(f"LEAKAGE VIOLATION: Forbidden columns found in canonical data: {found_forbidden}")

    # 3. row_id uniqueness and completeness
    if df["row_id"].isna().any():
        raise ValueError("row_id contains missing values!")
    if df["row_id"].duplicated().any():
        raise ValueError("row_id contains duplicate entries!")

    # 4. Target validity
    if df[REG_TARGET].isna().any():
        raise ValueError(f"Target '{REG_TARGET}' contains missing values!")
    if not np.isfinite(df[REG_TARGET]).all():
        raise ValueError(f"Target '{REG_TARGET}' contains non-finite values!")
    if (df[REG_TARGET] < 0).any():
        raise ValueError(f"Target '{REG_TARGET}' contains negative values!")

    # 5. Crop scope check
    if "crop" not in df.columns or set(df["crop"].unique()) != {"Rice"}:
        raise ValueError("Regression scope must be exclusively 'Rice'!")

    # 6. Categorical features check
    for col in ["state", "district", "season"]:
        if df[col].isna().any():
            raise ValueError(f"Grouping identifier '{col}' contains null values!")
        if (df[col].astype(str).str.strip() == "").any():
            raise ValueError(f"Grouping identifier '{col}' contains blank strings!")

    # 7. Year checks
    if df["year"].isna().any() or (df["year"] % 1 != 0).any():
        raise ValueError("year must be non-missing integer index!")
    unique_years = sorted(df["year"].unique())
    if len(unique_years) < MIN_REQUIRED_YEARS:
        raise ValueError(
            f"Need at least {MIN_REQUIRED_YEARS} unique years; found {len(unique_years)}: {unique_years}"
        )

    # 8. Uniqueness of agricultural key: (state, district, season, year)
    key_cols = ["state", "district", "season", "year"]
    if df.duplicated(subset=key_cols).any():
        raise ValueError("Repeated (state, district, season, year) keys detected in canonical dataset!")

    audit_report = {
        "status": "PASSED",
        "file": str(file_path.name),
        "sha256": compute_sha256(file_path),
        "total_rows": len(df),
        "unique_years_count": len(unique_years),
        "year_range": [int(unique_years[0]), int(unique_years[-1])],
        "unique_states_count": int(df["state"].nunique()),
        "unique_districts_count": int(df["district"].nunique()),
        "unique_seasons_count": int(df["season"].nunique()),
        "yield_summary_t_ha": {
            "min": float(df[REG_TARGET].min()),
            "mean": float(df[REG_TARGET].mean()),
            "median": float(df[REG_TARGET].median()),
            "max": float(df[REG_TARGET].max()),
            "std": float(df[REG_TARGET].std()),
        },
        "duplicate_agricultural_keys": 0,
        "leakage_check_passed": True,
    }

    out_file = DATA_VAL_DIR / "regression_validation.json"
    save_json(audit_report, out_file)
    logger.info("Regression validation PASSED. Report saved to %s", out_file)
    return audit_report


def validate_classification_data(file_path: Path = CROP_LABELS_CANONICAL_FILE) -> Dict[str, Any]:
    """Validates canonical crop recommendation classification dataset."""
    if not file_path.exists():
        raise FileNotFoundError(f"Canonical classification file not found: {file_path}")

    logger.info("Validating classification dataset: %s", file_path)
    df = pd.read_csv(file_path)

    missing_cols = set(CLS_REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing canonical classification columns: {missing_cols}")

    if df["row_id"].isna().any() or df["row_id"].duplicated().any():
        raise ValueError("Classification row_id must be non-missing and unique!")

    # Check numeric types
    for col in CLS_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Non-finite values found in feature '{col}'")

    # Physical checks
    if not df["ph"].between(0, 14).all():
        raise ValueError("pH outside physically valid [0, 14] bounds!")
    if not df["humidity"].between(0, 100).all():
        raise ValueError("Relative humidity outside physically valid [0, 100] bounds!")
    if (df[["N", "P", "K", "rainfall"]] < 0).any().any():
        raise ValueError("Nutrient or rainfall values cannot be negative!")

    # Check exact duplicate feature vectors
    if df.duplicated(subset=CLS_FEATURES).any():
        raise ValueError("Duplicate feature vectors detected in classification data!")

    # Class balance check
    num_classes = df[CLS_TARGET].nunique()
    min_class_count = df[CLS_TARGET].value_counts().min()
    if num_classes < 2:
        raise ValueError(f"Need >= 2 classes; found {num_classes}")
    if min_class_count < 10:
        raise ValueError(f"Each class must have >= 10 records; found minimum {min_class_count}")

    audit_report = {
        "status": "PASSED",
        "file": str(file_path.name),
        "sha256": compute_sha256(file_path),
        "total_rows": len(df),
        "num_classes": int(num_classes),
        "min_class_count": int(min_class_count),
        "features": CLS_FEATURES,
        "target": CLS_TARGET,
        "duplicate_feature_vectors": 0,
    }

    out_file = DATA_VAL_DIR / "classification_validation.json"
    save_json(audit_report, out_file)
    logger.info("Classification validation PASSED. Report saved to %s", out_file)
    return audit_report


def main():
    parser = argparse.ArgumentParser(description="Validate canonical datasets for Experiment 08")
    parser.parse_args()

    try:
        validate_regression_data()
        validate_classification_data()
        logger.info("ALL data validation checks PASSED successfully.")
    except Exception as e:
        logger.error("Data validation FAILED: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
