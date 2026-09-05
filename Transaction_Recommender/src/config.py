"""
Global configuration module for Experiment 07 Recommendation System.
Provides seed pinning, directory paths, schema definitions, and default model hyperparameters.
"""

from pathlib import Path
from typing import Any, Dict

# 1. Deterministic Seed
SEED: int = 42

# 2. Base Directories (relative to Transaction_Recommender root)
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
MODELS_DIR: Path = BASE_DIR / "models"
RESULTS_DIR: Path = BASE_DIR / "results"
FIGURES_DIR: Path = BASE_DIR / "figures"
NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"
SCRIPTS_DIR: Path = BASE_DIR / "scripts"

# Ensure all primary runtime directories exist
for _d in [
    RAW_DIR,
    PROCESSED_DIR,
    ARTIFACTS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    NOTEBOOKS_DIR,
    SCRIPTS_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)

# 3. Dataset Schemas: maps raw column names to canonical schema
# Canonical names: user_id, item_id, order_id, ts, amount, quantity, unit_price
DATASET_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "d1_online_retail": {
        "raw_path": RAW_DIR / "d1_online_retail" / "online_retail.csv",
        "processed_dir": PROCESSED_DIR / "d1_online_retail",
        "artifacts_dir": ARTIFACTS_DIR / "d1_online_retail",
        "models_dir": MODELS_DIR / "d1_online_retail",
        "results_dir": RESULTS_DIR / "d1_online_retail",
        "figures_dir": FIGURES_DIR / "d1_online_retail",
        "user_col": "CustomerID",
        "item_col": "StockCode",
        "order_col": "InvoiceNo",
        "ts_col": "InvoiceDate",
        "quantity_col": "Quantity",
        "price_col": "UnitPrice",
        "amount_col": "Amount",
        "description_col": "Description",
        "country_col": "Country",
        "ts_format": "%Y-%m-%d %H:%M:%S",
        "cancellation_prefix": "C",
        "train_end": "2011-09-01",
        "val_end": "2011-10-15",
        "min_candidates": 500,
        "max_candidates": 2000,
        "top_n_candidates": 1000,
        "n_negatives": 50,
        "k_values": [5, 10, 20],
    },
    "d2_advanced": {
        "raw_path": RAW_DIR / "d2_advanced" / "instacart_transactions.csv",
        "processed_dir": PROCESSED_DIR / "d2_advanced",
        "artifacts_dir": ARTIFACTS_DIR / "d2_advanced",
        "models_dir": MODELS_DIR / "d2_advanced",
        "results_dir": RESULTS_DIR / "d2_advanced",
        "figures_dir": FIGURES_DIR / "d2_advanced",
        "user_col": "CustomerID",
        "item_col": "StockCode",
        "order_col": "InvoiceNo",
        "ts_col": "OrderSequence",
        "quantity_col": "Quantity",
        "price_col": None,
        "amount_col": "Quantity",
        "description_col": "Description",
        "category_col": "Category",
        "subcategory_col": "SubCategory",
        "ts_format": "ordinal_sequence",
        "base_date": "2020-01-01",
        "cancellation_prefix": None,
        "train_end": "2020-05-15",
        "val_end": "2020-08-15",
        "min_candidates": 500,
        "max_candidates": 2000,
        "top_n_candidates": 1000,
        "n_negatives": 50,
        "k_values": [5, 10, 20],
    },
}

# 4. Model Training & Tuning Parameters (per Manual Sec. 13.2, 13.3)
DEFAULT_RF_PARAMS: Dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "random_state": SEED,
    "n_jobs": -1,
}

TUNING_PARAM_GRID: Dict[str, list] = {
    "n_estimators": [200, 300],
    "max_depth": [None, 12],
    "min_samples_leaf": [2, 5],
    "max_features": ["sqrt"],
    "class_weight": ["balanced_subsample"],
}

# Advanced evaluation seeds for uncertainty estimation (Manual Sec. 18.1)
ADVANCED_SEEDS: list[int] = [42, 101, 2024]
