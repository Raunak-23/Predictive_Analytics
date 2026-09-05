"""
IO and dataset loading utilities for Experiment 07 Recommendation System.
Provides schema-driven transaction loading, canonical column mapping, and audit logging.
"""

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd


def load_transactions(
    path: Union[str, Path],
    schema: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load raw transaction data, map columns to canonical schema, and compute audit metrics.

    Parameters:
        path: Path to raw transaction CSV file.
        schema: Dictionary defining mapping for user, item, order, ts, amount columns.

    Returns:
        df: DataFrame with canonical column names [user_id, item_id, order_id, ts, amount].
        audit: Dictionary of summary audit statistics (row count, unique counts, date range, missing %).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Transaction file not found at: {path}")

    try:
        raw_df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        raw_df = pd.read_csv(path, encoding="ISO-8859-1")

    # Required raw columns verification
    user_col = schema.get("user_col", "CustomerID")
    item_col = schema.get("item_col", "StockCode")
    order_col = schema.get("order_col", "InvoiceNo")
    ts_col = schema.get("ts_col", "InvoiceDate")

    required_raw = {user_col, item_col, order_col, ts_col}
    missing_raw = required_raw - set(raw_df.columns)
    if missing_raw:
        raise ValueError(
            f"Missing required raw columns for schema: {missing_raw}. "
            f"Available columns: {list(raw_df.columns)}"
        )

    # Compute audit metrics before dropping/cleaning
    total_raw_rows = len(raw_df)
    missing_users = raw_df[user_col].isna().sum()
    pct_missing_user = float((missing_users / max(total_raw_rows, 1)) * 100.0)
    missing_items = raw_df[item_col].isna().sum()
    pct_missing_item = float((missing_items / max(total_raw_rows, 1)) * 100.0)

    # Create canonical DataFrame
    df = pd.DataFrame()
    df["user_id"] = raw_df[user_col].astype(str).str.strip()
    # If original was NaN, user_id string might be 'nan'
    df.loc[raw_df[user_col].isna(), "user_id"] = np.nan

    df["item_id"] = raw_df[item_col].astype(str).str.strip()
    df.loc[raw_df[item_col].isna(), "item_id"] = np.nan

    df["order_id"] = raw_df[order_col].astype(str).str.strip()

    # Parse timestamps
    ts_format = schema.get("ts_format", "%Y-%m-%d %H:%M:%S")
    if ts_format == "ordinal_sequence":
        base_date = pd.Timestamp(schema.get("base_date", "2020-01-01"))
        # In Instacart, order_sequence represents the sequential order for each customer
        # We synthesize chronological days: base_date + order_sequence days
        seq_numeric = pd.to_numeric(raw_df[ts_col], errors="coerce").fillna(1)
        df["ts"] = base_date + pd.to_timedelta(seq_numeric * 7, unit="D")
    else:
        df["ts"] = pd.to_datetime(raw_df[ts_col], errors="coerce")

    # Handle amount / quantity / unit_price
    qty_col = schema.get("quantity_col")
    price_col = schema.get("price_col")
    amount_col = schema.get("amount_col")

    if amount_col and amount_col in raw_df.columns:
        df["amount"] = pd.to_numeric(raw_df[amount_col], errors="coerce").fillna(0.0)
    elif qty_col and price_col and qty_col in raw_df.columns and price_col in raw_df.columns:
        qty = pd.to_numeric(raw_df[qty_col], errors="coerce").fillna(0.0)
        price = pd.to_numeric(raw_df[price_col], errors="coerce").fillna(0.0)
        df["amount"] = qty * price
    elif qty_col and qty_col in raw_df.columns:
        df["amount"] = pd.to_numeric(raw_df[qty_col], errors="coerce").fillna(1.0)
    else:
        df["amount"] = 1.0

    if qty_col and qty_col in raw_df.columns:
        df["quantity"] = pd.to_numeric(raw_df[qty_col], errors="coerce").fillna(1.0)
    else:
        df["quantity"] = 1.0

    if price_col and price_col in raw_df.columns:
        df["unit_price"] = pd.to_numeric(raw_df[price_col], errors="coerce").fillna(0.0)
    else:
        df["unit_price"] = df["amount"] / np.maximum(df["quantity"], 1.0)

    # Copy optional metadata columns if present
    desc_col = schema.get("description_col")
    if desc_col and desc_col in raw_df.columns:
        df["description"] = raw_df[desc_col].fillna("").astype(str).str.strip()

    country_col = schema.get("country_col")
    if country_col and country_col in raw_df.columns:
        df["country"] = raw_df[country_col].fillna("Unknown").astype(str).str.strip()

    cat_col = schema.get("category_col")
    if cat_col and cat_col in raw_df.columns:
        df["category"] = raw_df[cat_col].fillna("general").astype(str).str.strip()

    # Calculate audit statistics
    valid_users = df["user_id"].dropna().nunique()
    valid_items = df["item_id"].dropna().nunique()
    min_ts = str(df["ts"].min()) if not df["ts"].isna().all() else "None"
    max_ts = str(df["ts"].max()) if not df["ts"].isna().all() else "None"

    audit = {
        "raw_row_count": int(total_raw_rows),
        "unique_users_raw": int(valid_users),
        "unique_items_raw": int(valid_items),
        "min_timestamp": min_ts,
        "max_timestamp": max_ts,
        "missing_user_count": int(missing_users),
        "pct_missing_user_id": round(pct_missing_user, 2),
        "missing_item_count": int(missing_items),
        "pct_missing_item_id": round(pct_missing_item, 2),
    }

    return df, audit
