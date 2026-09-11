"""Dataset splitting module for Experiment 08.

Implements:
1. Strict chronological holdout splitting for Rice yield regression (Section 15).
2. Governed independence-gated stratified splitting for Crop classification (Section 16).
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src.config import (
        CLS_SPLIT_MANIFEST,
        CLS_TARGET,
        CROP_LABELS_CANONICAL_FILE,
        IID_JUSTIFIED_DEFAULT,
        IID_JUSTIFICATION_NOTE,
        MANIFESTS_DIR,
        MIN_REQUIRED_YEARS,
        RANDOM_SEED,
        REG_SPLIT_MANIFEST,
        RICE_CANONICAL_FILE,
    )
    from src.utils import setup_logger
except ImportError:
    from config import (
        CLS_SPLIT_MANIFEST,
        CLS_TARGET,
        CROP_LABELS_CANONICAL_FILE,
        IID_JUSTIFIED_DEFAULT,
        IID_JUSTIFICATION_NOTE,
        MANIFESTS_DIR,
        MIN_REQUIRED_YEARS,
        RANDOM_SEED,
        REG_SPLIT_MANIFEST,
        RICE_CANONICAL_FILE,
    )
    from utils import setup_logger

logger = setup_logger("split_data")


def get_chronological_splits(
    df: pd.DataFrame, save_manifest: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """Applies strict chronological splitting to the regression dataset.

    Splits:
        - Test: Latest 2 observed years
        - Validation: 2 years immediately preceding test
        - Train: All earlier observed years

    Args:
        df: Input DataFrame containing 'year' and 'row_id'.
        save_manifest: If True, writes REG_SPLIT_MANIFEST to disk. If False, operates in memory.

    Returns:
        Tuple of (DataFrame with an added 'split' column, dictionary of year partitions).
    """
    years = sorted(df["year"].unique())
    if len(years) < MIN_REQUIRED_YEARS:
        raise ValueError(f"Need >= {MIN_REQUIRED_YEARS} unique years for chronological split. Found: {years}")

    test_years = years[-2:]
    val_years = years[-4:-2]
    train_years = years[:-4]

    split_col = np.where(
        df["year"].isin(test_years),
        "test",
        np.where(df["year"].isin(val_years), "validation", "train"),
    )
    df_split = df.copy()
    df_split["split"] = split_col

    # Strict temporal verification
    train_max_year = df_split.loc[df_split["split"] == "train", "year"].max()
    val_min_year = df_split.loc[df_split["split"] == "validation", "year"].min()
    val_max_year = df_split.loc[df_split["split"] == "validation", "year"].max()
    test_min_year = df_split.loc[df_split["split"] == "test", "year"].min()

    assert train_max_year < val_min_year, (
        f"TEMPORAL LEAKAGE: Train max year ({train_max_year}) >= Val min year ({val_min_year})"
    )
    assert val_max_year < test_min_year, (
        f"TEMPORAL LEAKAGE: Val max year ({val_max_year}) >= Test min year ({test_min_year})"
    )

    for s in ["train", "validation", "test"]:
        count = (df_split["split"] == s).sum()
        if count < 2:
            raise ValueError(f"Split '{s}' has insufficient observations ({count})")

    year_partitions = {
        "train_years": [int(y) for y in train_years],
        "validation_years": [int(y) for y in val_years],
        "test_years": [int(y) for y in test_years],
        "counts": {
            "train": int((df_split["split"] == "train").sum()),
            "validation": int((df_split["split"] == "validation").sum()),
            "test": int((df_split["split"] == "test").sum()),
        },
    }

    # Save manifest only when explicitly requested (avoids test side effects)
    if save_manifest:
        MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
        manifest = df_split[["row_id", "year", "split"]].copy()
        manifest.to_csv(REG_SPLIT_MANIFEST, index=False)
        logger.info("Saved regression split manifest to %s", REG_SPLIT_MANIFEST)
        logger.info(
            "Chronological splits created: Train=%d years (%d rows), Val=%d years (%d rows), Test=%d years (%d rows)",
            len(train_years),
            year_partitions["counts"]["train"],
            len(val_years),
            year_partitions["counts"]["validation"],
            len(test_years),
            year_partitions["counts"]["test"],
        )
    else:
        logger.debug("Chronological splits created in memory (save_manifest=False).")

    return df_split, year_partitions


def get_classification_splits(
    df: pd.DataFrame,
    iid_justified: bool = IID_JUSTIFIED_DEFAULT,
    save_manifest: bool = True,
) -> pd.DataFrame:
    """Applies independence-reviewed stratified holdout splitting for classification.

    Args:
        df: Canonical classification DataFrame.
        iid_justified: Boolean flag indicating explicit independence review (default: False).
        save_manifest: If True, writes CLS_SPLIT_MANIFEST to disk. If False, operates in memory.

    Raises:
        ValueError: If iid_justified is False.
    """
    if not iid_justified:
        raise ValueError(
            "Classification splitting halted: iid_justified is False.\n"
            "An explicit independence review and documented rationale is required before random splitting.\n"
            f"Review guidance: {IID_JUSTIFICATION_NOTE}"
        )

    logger.info("IID assumption explicitly justified. Proceeding with stratified 60/20/20 split.")

    indices = np.arange(len(df))
    # 60% train, 40% holdout
    idx_train, idx_rest = train_test_split(
        indices,
        test_size=0.40,
        stratify=df[CLS_TARGET],
        random_state=RANDOM_SEED,
    )
    # 20% validation, 20% test (from 40% holdout)
    idx_val, idx_test = train_test_split(
        idx_rest,
        test_size=0.50,
        stratify=df.iloc[idx_rest][CLS_TARGET],
        random_state=RANDOM_SEED,
    )

    split_arr = np.full(len(df), "train", dtype=object)
    split_arr[idx_val] = "validation"
    split_arr[idx_test] = "test"

    df_split = df.copy()
    df_split["split"] = split_arr

    # Save manifest only when explicitly requested
    if save_manifest:
        MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
        manifest = df_split[["row_id", CLS_TARGET, "split"]].copy()
        manifest.to_csv(CLS_SPLIT_MANIFEST, index=False)
        logger.info("Saved classification split manifest to %s", CLS_SPLIT_MANIFEST)
    else:
        logger.debug("Classification splits created in memory (save_manifest=False).")
    return df_split


def main():
    parser = argparse.ArgumentParser(description="Create dataset split manifests")
    parser.add_argument(
        "--justify-iid",
        action="store_true",
        help="Provide explicit justification for D3 classification IID splitting.",
    )
    args = parser.parse_args()

    try:
        df_rice = pd.read_csv(RICE_CANONICAL_FILE)
        get_chronological_splits(df_rice)

        df_cls = pd.read_csv(CROP_LABELS_CANONICAL_FILE)
        get_classification_splits(df_cls, iid_justified=args.justify_iid)
        logger.info("Splits generated successfully.")
    except Exception as e:
        logger.error("Splitting failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
