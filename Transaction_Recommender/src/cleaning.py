"""
Transaction cleaning and data integrity module for Experiment 07 Recommendation System.
Enforces explicit removal rules for missing IDs, cancellations, and duplicates with complete audit logs.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def clean_transactions(
    df: pd.DataFrame,
    schema: Dict[str, Any],
    dataset_name: str = "d1_online_retail",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean transaction logs according to strict integrity policies.

    Parameters:
        df: Input DataFrame with canonical columns (user_id, item_id, order_id, ts, amount).
        schema: Configuration schema dictionary for the dataset.
        dataset_name: Name of the dataset being processed.

    Returns:
        cleaned_df: Cleaned transaction DataFrame.
        cleaning_log: Detailed dictionary recording row counts removed per rule.
    """
    initial_rows = len(df)
    log: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "initial_rows": int(initial_rows),
    }

    # Step 1: Remove rows with unusable/missing user_id
    valid_user_mask = (
        df["user_id"].notna()
        & (df["user_id"].astype(str).str.strip() != "")
        & (df["user_id"].astype(str).str.strip().str.lower() != "nan")
    )
    df_step1 = df[valid_user_mask].copy()
    rows_missing_user = initial_rows - len(df_step1)
    log["removed_missing_user_id"] = int(rows_missing_user)

    # Step 2: Handle cancellations/returns explicitly
    cancellation_prefix = schema.get("cancellation_prefix")
    if cancellation_prefix:
        is_cancel = (
            df_step1["order_id"].astype(str).str.startswith(cancellation_prefix)
            | (df_step1["quantity"] < 0)
        )
    else:
        # Fallback cancellation check: negative quantity
        is_cancel = df_step1["quantity"] < 0

    df_step2 = df_step1[~is_cancel].copy()
    rows_cancellations = len(df_step1) - len(df_step2)
    log["removed_cancellations_and_returns"] = int(rows_cancellations)

    # Step 3: Filter non-positive quantities or prices
    valid_values = (df_step2["quantity"] > 0) & (df_step2["unit_price"] >= 0)
    df_step3 = df_step2[valid_values].copy()
    rows_invalid_values = len(df_step2) - len(df_step3)
    log["removed_nonpositive_values"] = int(rows_invalid_values)

    # Step 4: Deduplicate exact duplicate transaction records
    subset_cols = [c for c in ["user_id", "item_id", "order_id", "ts"] if c in df_step3.columns]
    df_step4 = df_step3.drop_duplicates(subset=subset_cols).copy()
    rows_duplicates = len(df_step3) - len(df_step4)
    log["removed_duplicate_rows"] = int(rows_duplicates)

    # Step 5: Ensure amount is accurate and positive
    if "amount" not in df_step4.columns or df_step4["amount"].isna().any():
        df_step4["amount"] = df_step4["quantity"] * df_step4["unit_price"]

    # Ensure uniform key types and valid datetime timestamps
    df_step4["user_id"] = df_step4["user_id"].astype(str).str.strip()
    df_step4["item_id"] = df_step4["item_id"].astype(str).str.strip()
    df_step4["ts"] = pd.to_datetime(df_step4["ts"])

    # Filter out missing timestamps
    df_clean = df_step4[df_step4["ts"].notna()].sort_values("ts").reset_index(drop=True)
    rows_missing_ts = len(df_step4) - len(df_clean)
    log["removed_missing_timestamp"] = int(rows_missing_ts)

    final_rows = len(df_clean)
    log["final_clean_rows"] = int(final_rows)
    log["total_removed"] = int(initial_rows - final_rows)
    log["retention_rate_pct"] = round((final_rows / max(initial_rows, 1)) * 100.0, 2)
    log["unique_users_clean"] = int(df_clean["user_id"].nunique())
    log["unique_items_clean"] = int(df_clean["item_id"].nunique())
    log["date_range_start"] = str(df_clean["ts"].min())
    log["date_range_end"] = str(df_clean["ts"].max())

    return df_clean, log
