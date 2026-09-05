"""
Dataset Conversion and Preprocessing Pipeline (Enriched & Leakage-Safe)
Experiment 07: Constructing a Recommendation System from Customer Transaction Data using Random Forest
Course: MDI3003 - Advanced Predictive Analytics (SCOPE, VIT Vellore)

Key Preprocessing Highlights:
1. Dataset D1 (UCI Online Retail):
   - Preserves all 541,909 raw transactions without arbitrary record deletion.
   - Explicitly models Guest Checkouts (missing CustomerIDs) for market popularity and cold-start priors.
   - Explicitly models Cancellations & Returns (InvoiceNo 'C' / negative Quantity) as negative signals and return-risk features.
   - Exports both complete enriched dataset (`data/processed/online_retail_enriched.csv`) and confirmed positive purchase subset (`data/processed/online_retail_cleaned.csv`).
2. Dataset D3 (Instacart Market Basket):
   - Extracts relational tables to `data/raw/instacart/`.
   - Models first-order structural nulls and reorder history into harmonized transaction format (`data/processed/instacart_transactions_train.csv`).
3. Generates comprehensive Dataset Cards, split manifests, and integrity audits in `artifacts/`.

Usage:
    python src/convert_datasets.py
    python src/convert_datasets.py --d1-only
    python src/convert_datasets.py --d3-only
"""

import argparse
import io
import json
import logging
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Local imports
try:
    from src.data_loader import (
        clean_online_retail,
        enrich_online_retail,
        compute_dataset_card,
        compute_file_hash,
        create_chronological_splits,
    )
except ImportError:
    # Direct script execution fallback
    from data_loader import (
        clean_online_retail,
        enrich_online_retail,
        compute_dataset_card,
        compute_file_hash,
        create_chronological_splits,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("DataConverter")


def process_d1_online_retail(
    raw_dir: Path,
    processed_dir: Path,
    artifacts_dir: Path,
    save_parquet: bool = True
) -> dict:
    """
    Extract, convert, enrich, and validate Dataset D1 (UCI Online Retail).
    Preserves all 541,909 rows with semantic flags for cancellations, returns, and guests.
    """
    logger.info("=" * 65)
    logger.info("STARTING DATASET D1: UCI ONLINE RETAIL ENRICHED CONVERSION")
    logger.info("=" * 65)
    start_time = time.time()

    zip_path = raw_dir / "online+retail.zip"
    raw_csv_path = raw_dir / "online_retail.csv"
    enriched_csv_path = processed_dir / "online_retail_enriched.csv"
    enriched_parquet_path = processed_dir / "online_retail_enriched.parquet"
    cleaned_csv_path = processed_dir / "online_retail_cleaned.csv"
    cleaned_parquet_path = processed_dir / "online_retail_cleaned.parquet"

    # Step 1: Load raw data from CSV or Excel zip
    if raw_csv_path.exists():
        logger.info(f"Loading raw CSV from {raw_csv_path}...")
        try:
            df_raw = pd.read_csv(raw_csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            df_raw = pd.read_csv(raw_csv_path, encoding="ISO-8859-1")
    elif zip_path.exists():
        logger.info(f"Extracting and reading 'Online Retail.xlsx' from {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            excel_files = [f for f in z.namelist() if f.endswith(".xlsx") or f.endswith(".xls")]
            if not excel_files:
                raise ValueError(f"No Excel file found inside {zip_path}")
            excel_filename = excel_files[0]
            logger.info(f"Found Excel file: '{excel_filename}'. Loading into pandas...")
            with z.open(excel_filename) as f:
                df_raw = pd.read_excel(f, engine="openpyxl")

        logger.info(f"Saving raw CSV to: {raw_csv_path}...")
        df_raw.to_csv(raw_csv_path, index=False, encoding="utf-8")
    else:
        raise FileNotFoundError(f"Neither {zip_path} nor {raw_csv_path} found in {raw_dir}")

    # Explicit string typing for identifiers & text to prevent mixed types
    df_raw["InvoiceNo"] = df_raw["InvoiceNo"].astype(str).str.strip()
    df_raw["StockCode"] = df_raw["StockCode"].astype(str).str.strip()
    if "Description" in df_raw.columns:
        df_raw["Description"] = df_raw["Description"].fillna("").astype(str).str.strip()
    if "Country" in df_raw.columns:
        df_raw["Country"] = df_raw["Country"].fillna("Unknown").astype(str).str.strip()
    df_raw["InvoiceDate"] = pd.to_datetime(df_raw["InvoiceDate"])

    raw_hash = compute_file_hash(raw_csv_path if raw_csv_path.exists() else zip_path)
    logger.info(f"Raw D1 record count: {len(df_raw):,} rows, {len(df_raw.columns)} columns.")

    # Step 2: Semantic enrichment (Preserve guest checkouts + Cancellations/Returns)
    logger.info("Enriching transactions: modeling guest checkouts, cancellations/returns, net revenue, and positive targets...")
    df_enriched, audit_stats = enrich_online_retail(df_raw)

    # Step 3: Save enriched dataset (ALL 541,909 records)
    logger.info(f"Saving full enriched dataset ({len(df_enriched):,} rows) to: {enriched_csv_path}...")
    df_enriched.to_csv(enriched_csv_path, index=False, encoding="utf-8")
    if save_parquet:
        logger.info(f"Saving enriched dataset as Parquet to: {enriched_parquet_path}...")
        df_enriched.to_parquet(enriched_parquet_path, index=False, engine="pyarrow")

    # Step 4: Save confirmed positive purchases subset (397,924 records) for ranker target
    df_cleaned = df_enriched[df_enriched["is_positive_purchase"]].copy().reset_index(drop=True)
    logger.info(f"Saving positive purchase subset ({len(df_cleaned):,} rows) to: {cleaned_csv_path}...")
    df_cleaned.to_csv(cleaned_csv_path, index=False, encoding="utf-8")
    if save_parquet:
        df_cleaned.to_parquet(cleaned_parquet_path, index=False, engine="pyarrow")

    # Step 5: Chronological splits per Section 11.1
    logger.info("Computing chronological train / validation / test splits on enriched data...")
    train_hist, val_future, test_future, split_manifest = create_chronological_splits(
        df_enriched, q_train=0.70, q_val=0.85
    )

    # Step 6: Comprehensive Dataset Card per Section 6.2
    dataset_card = compute_dataset_card(
        df_raw=df_raw,
        df_enriched=df_enriched,
        audit_stats=audit_stats,
        source_name="UCI Online Retail (D1)",
        source_url="https://archive.ics.uci.edu/dataset/352/online+retail",
        version_hash=raw_hash
    )

    # Save manifests and dataset cards to artifacts/ and data/processed/
    for folder in [processed_dir, artifacts_dir]:
        with open(folder / "dataset_card_d1.json", "w", encoding="utf-8") as f:
            json.dump(dataset_card, f, indent=2)
        with open(folder / "split_manifest.json", "w", encoding="utf-8") as f:
            json.dump(split_manifest, f, indent=2)

    elapsed = time.time() - start_time
    logger.info(f"Dataset D1 processing completed in {elapsed:.2f} seconds.")
    return {
        "dataset": "D1_Online_Retail",
        "total_enriched_rows": len(df_enriched),
        "positive_purchases_rows": len(df_cleaned),
        "guest_transactions": audit_stats["guest_transactions_count"],
        "cancellations_returns": audit_stats["cancellations_returns_count"],
        "unique_registered_customers": audit_stats["unique_registered_customers"],
        "unique_items_catalog": audit_stats["unique_items_catalog"],
        "elapsed_seconds": round(elapsed, 2)
    }


def process_d3_instacart(
    raw_dir: Path,
    processed_dir: Path,
    artifacts_dir: Path,
    sample_users: int = 25000,
    save_parquet: bool = True
) -> dict:
    """
    Extract, convert, and harmonize Dataset D3 (Instacart Market Basket Analysis).
    """
    logger.info("=" * 65)
    logger.info("STARTING DATASET D3: INSTACART MARKET BASKET ANALYSIS CONVERSION")
    logger.info("=" * 65)
    start_time = time.time()

    zip_path = raw_dir / "archive.zip"
    instacart_raw_dir = raw_dir / "instacart"
    instacart_raw_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        raise FileNotFoundError(f"Instacart archive {zip_path} was not found.")

    # Step 1: Extract CSV files from zip
    logger.info(f"Extracting CSV files from {zip_path} to {instacart_raw_dir}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            target = instacart_raw_dir / member.filename
            if not target.exists() or target.stat().st_size != member.file_size:
                logger.info(f"  Extracting: {member.filename} ({member.file_size / (1024*1024):.2f} MB)...")
                z.extract(member, instacart_raw_dir)
            else:
                logger.info(f"  Already extracted: {member.filename}")

    # Step 2: Load metadata tables (products, aisles, departments)
    logger.info("Loading metadata tables (products, aisles, departments)...")
    df_products = pd.read_csv(instacart_raw_dir / "products.csv")
    df_aisles = pd.read_csv(instacart_raw_dir / "aisles.csv")
    df_depts = pd.read_csv(instacart_raw_dir / "departments.csv")

    df_item_meta = df_products.merge(df_aisles, on="aisle_id", how="left").merge(
        df_depts, on="department_id", how="left"
    )

    # Step 3: Build harmonized train transaction table
    logger.info("Building harmonized transaction format for Instacart train set (Experiment E9)...")
    df_orders = pd.read_csv(instacart_raw_dir / "orders.csv")
    df_train_items = pd.read_csv(instacart_raw_dir / "order_products__train.csv")

    # Join orders and items to get full transaction log
    df_train_orders = df_orders[df_orders["eval_set"] == "train"].copy()
    df_instacart_train = df_train_items.merge(df_train_orders, on="order_id", how="inner")
    df_instacart_train = df_instacart_train.merge(df_item_meta, on="product_id", how="left")

    # Rename / align columns with standard recommender taxonomy (Section 6)
    harmonized_columns = {
        "order_id": "InvoiceNo",
        "user_id": "CustomerID",
        "product_id": "StockCode",
        "product_name": "Description",
        "department": "Category",
        "aisle": "SubCategory",
        "order_number": "OrderSequence",
        "order_dow": "DayOfWeek",
        "order_hour_of_day": "HourOfDay",
        "days_since_prior_order": "DaysSincePriorOrder",
        "add_to_cart_order": "CartOrder",
        "reordered": "IsReordered",
    }
    df_instacart_train = df_instacart_train.rename(columns=harmonized_columns)
    df_instacart_train["Quantity"] = 1  # Standard basket unit count
    
    # Model structural missing values: days_since_prior_order is NaN for first orders
    df_instacart_train["is_first_order"] = (df_instacart_train["OrderSequence"] == 1).astype(int)
    df_instacart_train["DaysSincePriorOrder"] = df_instacart_train["DaysSincePriorOrder"].fillna(0)

    # Explicit types for clean serialization and joins
    df_instacart_train["StockCode"] = df_instacart_train["StockCode"].astype(str)
    df_instacart_train["CustomerID"] = df_instacart_train["CustomerID"].astype(str)
    df_instacart_train["InvoiceNo"] = df_instacart_train["InvoiceNo"].astype(str)
    df_instacart_train["Description"] = df_instacart_train["Description"].fillna("").astype(str)
    df_instacart_train["Category"] = df_instacart_train["Category"].fillna("Unknown").astype(str)
    df_instacart_train["SubCategory"] = df_instacart_train["SubCategory"].fillna("Unknown").astype(str)

    # Save harmonized train dataset
    train_out_csv = processed_dir / "instacart_transactions_train.csv"
    train_out_parquet = processed_dir / "instacart_transactions_train.parquet"

    logger.info(f"Saving harmonized Instacart train dataset ({len(df_instacart_train):,} rows) to {train_out_csv}...")
    df_instacart_train.to_csv(train_out_csv, index=False)
    if save_parquet:
        df_instacart_train.to_parquet(train_out_parquet, index=False, engine="pyarrow")

    # Step 4: Build a representative sample for rapid benchmarking
    if sample_users > 0 and df_instacart_train["CustomerID"].nunique() > sample_users:
        logger.info(f"Creating representative subset of {sample_users:,} users for rapid testing...")
        sample_user_ids = np.random.default_rng(42).choice(
            df_instacart_train["CustomerID"].unique(), size=sample_users, replace=False
        )
        df_sample = df_instacart_train[df_instacart_train["CustomerID"].isin(sample_user_ids)].copy()
        sample_out_csv = processed_dir / "instacart_transactions_sample.csv"
        df_sample.to_csv(sample_out_csv, index=False)
        logger.info(f"Saved Instacart sample dataset ({len(df_sample):,} rows) to {sample_out_csv}")

    # Step 5: Dataset Card for D3 (Section 6.2)
    instacart_hash = compute_file_hash(zip_path)
    dataset_card_d3 = {
        "dataset_name": "Instacart Market Basket Analysis (D3)",
        "source_url": "https://www.kaggle.com/competitions/instacart-market-basket-analysis/data",
        "version_hash": instacart_hash,
        "total_orders_in_orders_csv": int(len(df_orders)),
        "train_orders": int(len(df_train_orders)),
        "train_transactions_rows": int(len(df_instacart_train)),
        "unique_users_in_train": int(df_instacart_train["CustomerID"].nunique()),
        "unique_products_in_train": int(df_instacart_train["StockCode"].nunique()),
        "total_catalog_products": int(len(df_products)),
        "total_aisles": int(len(df_aisles)),
        "total_departments": int(len(df_depts)),
        "structural_missing_policy": "DaysSincePriorOrder is NaN on first orders (order_number=1). Modeled with `is_first_order=1` and `DaysSincePriorOrder=0`.",
        "recommended_use": "Advanced next-basket / repeat-purchase recommendation and cross-dataset replication (E9).",
        "privacy_usage_note": "Anonymized user identifiers with grocery baskets. No direct PII."
    }

    for folder in [processed_dir, artifacts_dir]:
        with open(folder / "dataset_card_d3.json", "w", encoding="utf-8") as f:
            json.dump(dataset_card_d3, f, indent=2)

    elapsed = time.time() - start_time
    logger.info(f"Dataset D3 processing completed in {elapsed:.2f} seconds.")
    return {
        "dataset": "D3_Instacart",
        "train_rows": len(df_instacart_train),
        "unique_users": int(df_instacart_train["CustomerID"].nunique()),
        "unique_products": int(df_instacart_train["StockCode"].nunique()),
        "elapsed_seconds": round(elapsed, 2)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert and prepare D1 (UCI Online Retail) and D3 (Instacart) datasets."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Path to raw data directory (default: Transaction_Recommender/data/raw)"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Path to processed data directory (default: Transaction_Recommender/data/processed)"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="Path to artifacts directory (default: Transaction_Recommender/artifacts)"
    )
    parser.add_argument("--d1-only", action="store_true", help="Only process Dataset D1 (UCI Online Retail)")
    parser.add_argument("--d3-only", action="store_true", help="Only process Dataset D3 (Instacart)")
    parser.add_argument("--no-parquet", action="store_true", help="Do not save parquet copies")
    parser.add_argument("--sample-users", type=int, default=25000, help="Instacart user sample size")

    args = parser.parse_args()

    # Determine paths relative to project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    raw_dir = Path(args.raw_dir) if args.raw_dir else project_root / "data" / "raw"
    processed_dir = Path(args.processed_dir) if args.processed_dir else project_root / "data" / "processed"
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else project_root / "artifacts"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Project Root:   {project_root}")
    logger.info(f"Raw Directory:  {raw_dir}")
    logger.info(f"Processed Dir:  {processed_dir}")
    logger.info(f"Artifacts Dir:  {artifacts_dir}")

    summary = {}
    save_parquet = not args.no_parquet

    if not args.d3_only:
        summary["D1"] = process_d1_online_retail(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            artifacts_dir=artifacts_dir,
            save_parquet=save_parquet
        )

    if not args.d1_only:
        summary["D3"] = process_d3_instacart(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            artifacts_dir=artifacts_dir,
            sample_users=args.sample_users,
            save_parquet=save_parquet
        )

    # Save overall conversion summary
    summary_path = processed_dir / "conversion_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 65)
    logger.info("ALL DATASET CONVERSIONS AND VALIDATIONS SUCCEEDED!")
    logger.info(f"Summary written to: {summary_path}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
