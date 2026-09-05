"""
Data loading, cleaning, feature engineering, and modeling utilities for Experiment 07:
Constructing a Recommendation System from Customer Transaction Data using Random Forest.

Implements guidelines from MDI3003 Lab Manual (QP 7), explicitly preserving and modeling:
1. Cancellations & Returns (negative propensity signals, return rate features, net quantities)
2. Missing CustomerIDs / Guest checkouts (market-level popularity, cold-start priors)
3. Chronological time-split validation & feature calculation without future leakage
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def compute_file_hash(filepath: Union[str, Path], algorithm: str = "sha256") -> str:
    """Compute hash of a file for reproducibility manifests (Section 6.2 / 24)."""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return f"{algorithm}:{h.hexdigest()}"


def load_online_retail_raw(
    csv_path: Union[str, Path],
    encoding: str = "ISO-8859-1"
) -> pd.DataFrame:
    """
    Load raw UCI Online Retail dataset (D1) per Section 10.2.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {path}")

    logger.info(f"Loading raw Online Retail data from {path}...")
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding=encoding)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    required = {"InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in Online Retail dataset: {missing}")

    logger.info(f"Loaded raw dataset with {len(df):,} rows and {len(df.columns)} columns.")
    return df


def enrich_online_retail(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Union[int, float, str, dict]]]:
    """
    Enrich Online Retail transactions by preserving and explicitly modeling:
    1. Guest/Unregistered checkouts (missing CustomerID represents anonymous guests, valuable for cold-start/market popularity)
    2. Cancellations & Returns (InvoiceNo starting with 'C' or Quantity < 0 represents returns, providing negative feedback signals)
    3. Price adjustments & non-product administrative codes (e.g. POST, D, BANK CHARGES)
    4. Net quantities and spend per transaction line

    Returns:
        df_enriched: DataFrame containing all 541,909 transactions with explicit semantic flags.
        audit_stats: Dictionary containing comprehensive audit metrics.
    """
    raw_rows = len(df)
    df_out = df.copy()

    # Standardize string fields
    df_out["InvoiceNo"] = df_out["InvoiceNo"].astype(str).str.strip()
    df_out["StockCode"] = df_out["StockCode"].astype(str).str.strip()
    df_out["Description"] = df_out["Description"].fillna("").astype(str).str.strip()
    df_out["Country"] = df_out["Country"].fillna("Unknown").astype(str).str.strip()
    df_out["InvoiceDate"] = pd.to_datetime(df_out["InvoiceDate"])

    # 1. Guest checkout / missing CustomerID modeling
    df_out["is_guest"] = df_out["CustomerID"].isna()
    df_out["is_registered"] = ~df_out["is_guest"]
    df_out["raw_customer_id"] = df_out["CustomerID"]
    # Unified CustomerID: registered numeric string or guest token
    df_out["CustomerID"] = np.where(
        df_out["is_registered"],
        df_out["raw_customer_id"].fillna(0).astype(np.int64).astype(str),
        "GUEST_" + df_out["InvoiceNo"]
    )

    # 2. Cancellations & Returns modeling
    # Invoices starting with 'C' or negative quantities are cancellations/returns
    df_out["is_cancellation"] = df_out["InvoiceNo"].str.startswith("C") | (df_out["Quantity"] < 0)

    # 3. Transaction classification
    # Classify non-standard stock codes (e.g. POST, D, DOT, BANK CHARGES, CR)
    service_codes = {"POST", "D", "DOT", "M", "BANK CHARGES", "CR", "PADS", "AMAZONFEE", "S"}
    df_out["is_service_code"] = df_out["StockCode"].isin(service_codes)

    conditions = [
        df_out["is_cancellation"],
        df_out["is_service_code"],
        (df_out["Quantity"] > 0) & (df_out["UnitPrice"] > 0)
    ]
    choices = [
        "CANCELLATION_RETURN",
        "SERVICE_ADJUSTMENT",
        "STANDARD_PURCHASE"
    ]
    df_out["transaction_type"] = np.select(conditions, choices, default="ZERO_PRICE_OR_OTHER")

    # 4. Quantity and Financials
    # Signed quantity and signed amount (returns reflect negative cash/inventory flow)
    df_out["SignedQuantity"] = df_out["Quantity"]
    df_out["SignedAmount"] = (df_out["Quantity"] * df_out["UnitPrice"]).round(4)
    # Absolute amount for volume modeling
    df_out["Amount"] = (df_out["Quantity"].abs() * df_out["UnitPrice"]).round(4)

    # 5. Recommendation positive outcome flag
    # A true positive future purchase MUST be a registered, non-cancelled purchase with Quantity > 0 and UnitPrice > 0
    df_out["is_positive_purchase"] = (
        df_out["is_registered"]
        & (~df_out["is_cancellation"])
        & (~df_out["is_service_code"])
        & (df_out["Quantity"] > 0)
        & (df_out["UnitPrice"] > 0)
    )

    # Sort chronologically for temporal integrity
    df_out = df_out.sort_values("InvoiceDate").reset_index(drop=True)

    guest_count = int(df_out["is_guest"].sum())
    cancel_count = int(df_out["is_cancellation"].sum())
    pos_purchase_count = int(df_out["is_positive_purchase"].sum())

    audit_stats = {
        "raw_rows": raw_rows,
        "total_enriched_rows": len(df_out),
        "guest_transactions_count": guest_count,
        "guest_transactions_pct": round(guest_count / raw_rows * 100, 2),
        "registered_transactions_count": int(df_out["is_registered"].sum()),
        "cancellations_returns_count": cancel_count,
        "cancellations_returns_pct": round(cancel_count / raw_rows * 100, 2),
        "service_adjustments_count": int(df_out["is_service_code"].sum()),
        "positive_purchases_count": pos_purchase_count,
        "unique_registered_customers": int(df_out[df_out["is_registered"]]["CustomerID"].nunique()),
        "unique_items_catalog": int(df_out["StockCode"].nunique()),
        "min_date": str(df_out["InvoiceDate"].min()),
        "max_date": str(df_out["InvoiceDate"].max()),
        "gross_positive_revenue": round(float(df_out[df_out["is_positive_purchase"]]["Amount"].sum()), 2),
        "net_revenue": round(float(df_out["SignedAmount"].sum()), 2),
    }

    logger.info(
        f"Enrichment summary: {raw_rows:,} raw records enriched.\n"
        f"  - Registered Customers: {audit_stats['unique_registered_customers']:,} (Positive Purchases: {pos_purchase_count:,})\n"
        f"  - Guest Checkouts: {guest_count:,} ({audit_stats['guest_transactions_pct']}%)\n"
        f"  - Returns & Cancellations: {cancel_count:,} ({audit_stats['cancellations_returns_pct']}%)\n"
        f"  - Net Revenue: GBP {audit_stats['net_revenue']:,.2f}"
    )

    return df_out, audit_stats


def clean_online_retail(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Union[int, float, str]]]:
    """
    Standard registered transaction view (Section 10.3 scaffold):
    Filters to registered positive purchases for direct compatibility with baseline ranker.
    """
    df_enriched, stats = enrich_online_retail(df)
    df_clean = df_enriched[df_enriched["is_positive_purchase"]].copy().reset_index(drop=True)
    return df_clean, stats


def load_online_retail_cleaned(
    cleaned_path: Union[str, Path]
) -> pd.DataFrame:
    """Load preprocessed Online Retail transaction dataset."""
    path = Path(cleaned_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    return df


def compute_dataset_card(
    df_raw: pd.DataFrame,
    df_enriched: pd.DataFrame,
    audit_stats: dict,
    source_name: str = "UCI Online Retail (D1)",
    source_url: str = "https://archive.ics.uci.edu/dataset/352/online+retail",
    version_hash: str = ""
) -> Dict:
    """
    Construct standardized Dataset Card per Section 6.2 of QP 7 with explicit return/guest policy.
    """
    return {
        "dataset_name": source_name,
        "source_url": source_url,
        "version_hash": version_hash,
        "raw_rows": int(len(df_raw)),
        "enriched_total_rows": int(len(df_enriched)),
        "unique_registered_customers": audit_stats["unique_registered_customers"],
        "unique_items_catalog": audit_stats["unique_items_catalog"],
        "date_range": {
            "start": str(df_enriched["InvoiceDate"].min()),
            "end": str(df_enriched["InvoiceDate"].max())
        },
        "return_cancellation_policy": (
            "Preserved and explicitly modeled. Invoices with prefix 'C' or negative Quantity "
            "are tagged with `is_cancellation=True` and `transaction_type='CANCELLATION_RETURN'`. "
            "They provide return-rate features (customer return propensity and item return risk) "
            "and are excluded from positive future purchase target definitions to avoid recommending returned products."
        ),
        "missing_id_policy": (
            "Preserved and explicitly modeled. Transactions without customer IDs (~24.9% of data) "
            "represent guest/anonymous checkouts (`is_guest=True`). They are used for market-level item "
            "popularity and cold-start baseline priors, while personalized user ranking models train "
            "on registered customer histories (`is_registered=True`)."
        ),
        "privacy_usage_note": (
            "Transactional data contains anonymized numeric CustomerIDs and item codes. "
            "No PII (names, emails, payment details) is stored or exposed."
        ),
        "audit_stats": audit_stats
    }


def create_chronological_splits(
    df: pd.DataFrame,
    q_train: float = 0.70,
    q_val: float = 0.85,
    fixed_train_end: Optional[pd.Timestamp] = None,
    fixed_val_end: Optional[pd.Timestamp] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Split dataset chronologically per Section 11.1 to prevent data leakage:
    - train_hist: transactions strictly before train_end (features & training pairs)
    - val_future: transactions in [train_end, val_end) (validation target window)
    - test_future: transactions >= val_end (locked evaluation window)
    """
    if fixed_train_end is not None and fixed_val_end is not None:
        t1 = pd.Timestamp(fixed_train_end)
        t2 = pd.Timestamp(fixed_val_end)
    else:
        t1 = df["InvoiceDate"].quantile(q_train)
        t2 = df["InvoiceDate"].quantile(q_val)

    train_hist = df[df["InvoiceDate"] < t1].copy()
    val_future = df[(df["InvoiceDate"] >= t1) & (df["InvoiceDate"] < t2)].copy()
    test_future = df[df["InvoiceDate"] >= t2].copy()

    # Acceptance assertions per Appendix C
    assert train_hist["InvoiceDate"].max() < t1, "Train history contains future timestamps!"
    assert val_future["InvoiceDate"].min() >= t1, "Val future contains historical timestamps!"
    assert test_future["InvoiceDate"].min() >= t2, "Test future contains historical timestamps!"

    split_manifest = {
        "train_end": str(t1),
        "val_end": str(t2),
        "train_rows": len(train_hist),
        "val_rows": len(val_future),
        "test_rows": len(test_future),
        "train_registered_customers": int(train_hist[train_hist.get("is_registered", True)]["CustomerID"].nunique()),
        "val_registered_customers": int(val_future[val_future.get("is_registered", True)]["CustomerID"].nunique()),
        "test_registered_customers": int(test_future[test_future.get("is_registered", True)]["CustomerID"].nunique()),
    }

    logger.info(
        f"Chronological split completed:\n"
        f"  Train history: {len(train_hist):,} rows (< {t1})\n"
        f"  Val target:    {len(val_future):,} rows ([{t1} to {t2}))\n"
        f"  Test target:   {len(test_future):,} rows (>= {t2})"
    )

    return train_hist, val_future, test_future, split_manifest


# =====================================================================
# LEAKAGE-SAFE FEATURE ENGINEERING UTILIZING RETURNS & BEHAVIOR (Sec 12)
# =====================================================================

def compute_customer_features(hist: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    Compute customer features including return propensity, net monetary value, and frequency.
    Only strictly historical transactions (InvoiceDate < cutoff) are used (no future leakage).
    """
    h_reg = hist[hist["is_registered"]].copy() if "is_registered" in hist.columns else hist.copy()
    h_reg["pos_qty"] = np.maximum(h_reg["Quantity"], 0)
    h_reg["is_cancel_int"] = h_reg["is_cancellation"].astype(int) if "is_cancellation" in h_reg.columns else 0
    signed_amt_col = "SignedAmount" if "SignedAmount" in h_reg.columns else "Amount"
    
    out = h_reg.groupby("CustomerID").agg(
        cust_txns=("InvoiceNo", "nunique"),
        cust_items=("StockCode", "nunique"),
        cust_qty_gross=("pos_qty", "sum"),
        cust_spend_gross=("Amount", "sum"),
        cust_net_spend=(signed_amt_col, "sum"),
        cust_return_txns=("is_cancel_int", "sum"),
        cust_last=("InvoiceDate", "max"),
        cust_first=("InvoiceDate", "min"),
    ).reset_index()
    
    out["cust_recency_days"] = (cutoff - out["cust_last"]).dt.total_seconds() / (24 * 3600)
    out["cust_tenure_days"] = (cutoff - out["cust_first"]).dt.total_seconds() / (24 * 3600)
    out["cust_avg_basket_items"] = (out["cust_items"] / np.maximum(out["cust_txns"], 1)).round(2)
    out["cust_return_rate"] = (out["cust_return_txns"] / np.maximum(out["cust_txns"], 1)).round(4)
    return out.drop(columns=["cust_last", "cust_first"])


def compute_item_features(hist: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    Compute item features including market-wide popularity (including guests),
    buyer breadth, return risk rate, and average price.
    """
    h = hist.copy()
    h["pos_qty"] = np.maximum(h["Quantity"], 0)
    h["is_cancel_int"] = h["is_cancellation"].astype(int) if "is_cancellation" in h.columns else 0
    
    out = h.groupby("StockCode").agg(
        item_txns=("InvoiceNo", "nunique"),
        item_buyers=("CustomerID", "nunique"),
        item_qty_gross=("pos_qty", "sum"),
        item_avg_price=("UnitPrice", "mean"),
        item_return_count=("is_cancel_int", "sum"),
        item_last=("InvoiceDate", "max"),
    ).reset_index()
    
    out["item_recency_days"] = (cutoff - out["item_last"]).dt.total_seconds() / (24 * 3600)
    out["item_return_rate"] = (out["item_return_count"] / np.maximum(out["item_txns"], 1)).round(4)
    return out.drop(columns=["item_last"])


def compute_pair_features(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Compute customer-item interaction features:
    - Prior purchase count
    - Prior return count (explicit negative feedback)
    - Repeat purchase indicator
    """
    h_reg = hist[hist["is_registered"]].copy() if "is_registered" in hist.columns else hist.copy()
    h_reg["pos_qty"] = np.maximum(h_reg["Quantity"], 0)
    h_reg["is_cancel_int"] = h_reg["is_cancellation"].astype(int) if "is_cancellation" in h_reg.columns else 0
    
    out = h_reg.groupby(["CustomerID", "StockCode"]).agg(
        pair_purchases=("InvoiceNo", "nunique"),
        pair_qty=("pos_qty", "sum"),
        pair_spend=("Amount", "sum"),
        pair_returns=("is_cancel_int", "sum"),
    ).reset_index()
    
    out["pair_is_repeat"] = (out["pair_purchases"] > 1).astype(int)
    out["pair_has_returned"] = (out["pair_returns"] > 0).astype(int)
    return out


def load_instacart_data(
    instacart_dir: Union[str, Path],
    load_prior: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Load extracted Instacart Market Basket dataset files (D3).
    """
    idir = Path(instacart_dir)
    if not idir.exists():
        raise FileNotFoundError(f"Instacart directory not found: {idir}")

    data = {
        "orders": pd.read_csv(idir / "orders.csv"),
        "products": pd.read_csv(idir / "products.csv"),
        "aisles": pd.read_csv(idir / "aisles.csv"),
        "departments": pd.read_csv(idir / "departments.csv"),
        "order_products_train": pd.read_csv(idir / "order_products__train.csv"),
    }
    if load_prior and (idir / "order_products__prior.csv").exists():
        data["order_products_prior"] = pd.read_csv(idir / "order_products__prior.csv")

    return data
