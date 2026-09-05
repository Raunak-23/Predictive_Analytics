# Transaction Recommender System (Experiment 07)
## MDI3003 - Advanced Predictive Analytics | SCOPE, VIT Vellore

### Overview
This repository implements a **Supervised Recommender System** built on customer transaction logs using **Random Forest** candidate scoring and ranking, alongside popularity baselines, candidate-generation recall audits, and collaborative filtering / matrix factorization extensions.

---

### Dataset Architecture & Preprocessing

The raw and processed datasets correspond to **Dataset D1 (UCI Online Retail)** and **Dataset D3 (Instacart Market Basket Analysis)** specified in the laboratory manual.

#### 1. Dataset D1: UCI Online Retail (`data/raw/online_retail.csv`)
- **Source**: UCI Machine Learning Repository (01-Dec-2010 to 09-Dec-2011)
- **Raw Records**: 541,909 transactions across 8 columns.
- **Cleaning & Integrity Protocol (Section 6.1 & 10.3)**:
  - Dropped 135,080 rows lacking customer IDs for personalized recommendation modeling.
  - Removed 8,905 return/cancellation invoices (prefixed with `'C'`) and non-positive quantity/price entries to prevent counting returns as positive recommendation outcomes.
  - Standardized customer identifiers as clean string keys.
  - Formatted `InvoiceDate` into ISO-8601 datetimes.
  - Computed transaction value: `Amount = Quantity * UnitPrice`.
  - Sorted chronologically for strict temporal integrity.
- **Cleaned Dataset**: 397,924 valid transactions across 4,339 unique customers and 3,665 unique items.
- **Outputs**:
  - `data/raw/online_retail.csv` (Standard raw CSV for Section 10.2 scaffold)
  - `data/processed/online_retail_cleaned.csv` & `.parquet` (Cleaned transaction log)
  - `artifacts/dataset_card_d1.json` (Dataset Card per Section 6.2)
  - `artifacts/split_manifest.json` (Chronological 70% / 85% train-val-test split cutoffs per Section 11.1)

#### 2. Dataset D3: Instacart Market Basket Analysis (`data/raw/instacart/`)
- **Source**: Instacart Public Grocery Benchmark
- **Raw Tables Extracted**: `orders.csv`, `products.csv`, `aisles.csv`, `departments.csv`, `order_products__train.csv`, `order_products__prior.csv`
- **Harmonized Recommender Schema**:
  - `InvoiceNo` (`order_id`), `CustomerID` (`user_id`), `StockCode` (`product_id`), `Description` (`product_name`), `Category` (`department`), `SubCategory` (`aisle`), `OrderSequence`, `DayOfWeek`, `HourOfDay`, `DaysSincePriorOrder`, `IsReordered`.
- **Outputs**:
  - `data/processed/instacart_transactions_train.csv` (1,384,617 rows)
  - `data/processed/instacart_transactions_sample.csv` (264,159 rows for rapid E9 benchmarking)
  - `artifacts/dataset_card_d3.json` (Dataset Card per Section 6.2)

---

### Quickstart

To run the data conversion and preparation pipeline:
```bash
# Process both D1 (Online Retail) and D3 (Instacart)
python src/convert_datasets.py

# Process only UCI Online Retail (D1)
python src/convert_datasets.py --d1-only

# Process only Instacart (D3)
python src/convert_datasets.py --d3-only
```

To load and use the datasets in Python/Jupyter:
```python
from src.data_loader import load_online_retail_cleaned, create_chronological_splits

# Load cleaned transaction log
df = load_online_retail_cleaned("data/processed/online_retail_cleaned.csv")

# Create chronological train, validation, and test splits
train_hist, val_future, test_future, split_info = create_chronological_splits(df)
```
