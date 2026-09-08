# Transaction Recommender System (Experiment 07)
## MDI3003 - Advanced Predictive Analytics | SCOPE, VIT Vellore

### Overview
This repository implements an industry-grade **Supervised Transaction Recommender System** built on customer purchasing logs. The primary architecture employs **Random Forest** candidate scoring and ranking, benchmarked against **Global Popularity** baselines and **Matrix Factorization (Truncated SVD)** collaborative filtering models.

All pipeline components, feature extractors, training routines, and CLI inference utilities are housed strictly in `src/` for clean modularity, zero code duplication, and strict reproducibility.

---

### Repository Structure

```
Transaction_Recommender/
├── data/
│   ├── raw/                             # Raw transaction archives and tables
│   │   ├── online_retail.csv            # UCI Online Retail raw dataset (D1)
│   │   ├── d1_online_retail/            # D1 raw reference copy
│   │   ├── d2_advanced/                 # Instacart benchmark merged transactions
│   │   └── instacart/                   # Raw relational Instacart tables
│   └── processed/                       # Cleaned, leakage-safe transaction logs
│       ├── online_retail_cleaned.csv    # D1 confirmed positive purchase subset (397k rows)
│       ├── online_retail_enriched.csv   # D1 enriched log with guest & cancellation tags
│       └── instacart_transactions_train.csv # D2 harmonized multi-basket log (1.38M rows)
├── src/                                 # Centralized application logic & CLI tools
│   ├── __init__.py                      # Package initialization
│   ├── config.py                        # Paths, seeds, and hyperparameter grids
│   ├── io_utils.py                      # Robust, leakage-safe data loading & validation
│   ├── cleaning.py                      # Transaction cleaning & sanitization protocols
│   ├── eda.py                           # Exploratory data analysis & RFM distributions
│   ├── splitting.py                     # Strict chronological train-val-test splitting
│   ├── candidates.py                    # Candidate universe generation & recall audit
│   ├── features.py                      # Customer, item, and pair feature engineering
│   ├── models.py                        # Random Forest, Matrix Factorization, Popularity
│   ├── evaluation.py                    # P@K, R@K, HR@K, NDCG@K, catalog coverage
│   ├── visualization.py                 # Publication-ready diagnostic plotting
│   ├── artifacts.py                     # Artifact persistence & reload verification
│   ├── convert_datasets.py              # Raw-to-processed dataset converter
│   ├── run_d1_online_retail.py          # End-to-end execution pipeline for Dataset D1
│   ├── run_d2_advanced.py               # End-to-end execution pipeline for Dataset D2
│   ├── prepare_instacart_data.py        # Instacart raw tables merge utility
│   ├── generate_report.py               # Automated publication-quality Word report generator
│   └── infer.py                         # CLI inference tool for custom data & single queries
├── artifacts/                           # Machine-readable schemas, policies & manifests
│   ├── d1_online_retail/                # D1 feature schema, split manifest, candidate policy
│   └── d2_advanced/                     # D2 feature schema, split manifest, candidate policy
├── models/                              # Serialized model binary checkpoints (.joblib)
│   ├── d1_online_retail/                # Trained Random Forest classifier
│   └── d2_advanced/                     # Random Forest, Matrix Factorization & Popularity
├── results/                             # Evaluation metrics, error audits & final reports
│   ├── Experiment_07_Report.docx        # Generated comprehensive 100-mark lab report
│   ├── d1_online_retail/                # Ranking metrics, candidate recall, recommendations
│   └── d2_advanced/                     # Ranking metrics, uncertainty analysis, efficiency
├── figures/                             # Generated diagnostic plots (EDA, ROC, metrics)
└── notebooks/                           # Step-by-step interactive Jupyter notebooks
    ├── 01_d1_online_retail.ipynb
    └── 02_d2_advanced.ipynb
```

---

### Dataset Architecture & Preprocessing

The repository implements strict data hygiene and temporal integrity across two major benchmark datasets:

#### 1. Dataset D1: UCI Online Retail (`data/raw/online_retail.csv`)
- **Source**: UCI Machine Learning Repository (01-Dec-2010 to 09-Dec-2011).
- **Raw Records**: 541,909 transactions across 8 columns.
- **Cleaning Protocol**:
  - Dropped 135,080 rows lacking customer IDs for personalized recommendation modeling.
  - Excluded 8,905 return/cancellation invoices (`'C'` prefix) and non-positive quantity/price entries to ensure only valid purchase events represent positive recommendation targets.
  - Formatted `InvoiceDate` to ISO-8601 datetimes and sorted strictly chronologically.
  - Cleaned dataset yields 397,924 valid transactions across 4,339 customers and 3,665 unique items.

#### 2. Dataset D2: Instacart Market Basket Benchmark (`data/raw/instacart/`)
- **Source**: Instacart Public Grocery Benchmark.
- **Harmonized Schema**: `InvoiceNo`, `CustomerID`, `StockCode`, `Description`, `Category`, `SubCategory`, `OrderSequence`, `Quantity`, `DaysSincePriorOrder`.

---

### Running Pipelines and Workflows

All executable scripts are run directly from `src/`:

#### 1. Dataset Conversion & Preparation
Converts raw datasets into processed, leakage-safe transaction files and outputs Dataset Cards:
```bash
# Process both D1 and D2 datasets
python src/convert_datasets.py

# Process only UCI Online Retail (D1)
python src/convert_datasets.py --d1-only

# Process only Instacart (D2)
python src/convert_datasets.py --d3-only
```

#### 2. Execute End-to-End Recommender Pipelines
Executes Stages 1 through 13 (Loading, Cleaning, EDA, Splitting, Candidate Generation, Feature Engineering, RF Validation Tuning, Test Evaluation, 5-Case Audit, Visualizations, and Artifact Persistence):
```bash
# Run complete D1 pipeline (UCI Online Retail)
python src/run_d1_online_retail.py

# Run complete D2 pipeline (Instacart Benchmark)
python src/run_d2_advanced.py
```

#### 3. Generate Comprehensive Lab Report (.docx)
Compiles all computed metrics, tables, diagrams, viva voce items, and ethics disclosures into an industry-grade Word report:
```bash
python src/generate_report.py
```
Output is saved to `results/Experiment_07_Report.docx`.

---

### CLI Model Inference & Custom Data Engine (`src/infer.py`)

The CLI inference tool `src/infer.py` allows users to load pre-trained pipelines and models to generate Top-K personalized recommendations on **their own custom data**, specific customer IDs, or ad-hoc shopping carts.

#### 1. Quick Customer Query (Existing Models)
Query personalized recommendations for a specific customer ID from the trained dataset:
```bash
# Recommend Top-10 items for Customer 12349 using D1 Random Forest
python src/infer.py --dataset d1 --user 12349 --top-k 10

# Recommend Top-5 items for Customer 169 using D2 Matrix Factorization
python src/infer.py --dataset d2 --model-type mf --user 169 --top-k 5

# Recommend Top-5 items using Global Popularity baseline
python src/infer.py --dataset d1 --model-type pop --user 12349 --top-k 5
```

Sample CLI Output:
```text
================================================================================
  Top Recommendations for Customer: 12349
================================================================================
  Rank  | Item ID    | Propensity / Score | Description
  ------+------------+--------------------+------------------------------------
  1     | 22423      | 1.0000 (100.0%)    | REGENCY CAKESTAND 3 TIER
  2     | 22720      | 1.0000 (100.0%)    | SET OF 3 CAKE TINS PANTRY DESIGN
  3     | 22960      | 1.0000 (100.0%)    | JAM MAKING SET WITH JARS
  4     | 22666      | 1.0000 (100.0%)    | RECIPE BOX PANTRY YELLOW DESIGN
  5     | 20914      | 1.0000 (100.0%)    | SET/5 RED RETROSPOT LID GLASS BOWLS
```

#### 2. Inference on Your Own Custom Data File
Users can pass their own transaction dataset (`.csv`, `.json`, or `.parquet`). The engine automatically maps custom column names, calculates customer/item/pair interaction features, scores candidates, and generates recommendations:
```bash
# Infer on custom transactions CSV and save recommendations to CSV
python src/infer.py --data path/to/my_transactions.csv --output output_recs.csv --top-k 10

# Infer on custom data for a specific customer only
python src/infer.py --data path/to/my_transactions.csv --user 99001 --top-k 5
```

**Supported Column Header Variations**:
- **User Identifier**: `CustomerID`, `customer_id`, `user_id`, `user`, `shopper_id`, `ShopperID`
- **Item Identifier**: `StockCode`, `product_id`, `item_id`, `item`, `ProductCode`, `sku`
- **Timestamp**: `InvoiceDate`, `ts`, `date`, `timestamp`, `OrderSequence`
- **Quantity**: `Quantity`, `quantity`, `qty`, `count`
- **Price / Spend**: `UnitPrice`, `price`, `Amount`, `amount`, `spend`
- **Description**: `Description`, `description`, `product_name`, `item_name`, `name`

#### 3. Ad-hoc Basket / Cold-Start Session Inference
Test recommendations for an ad-hoc customer who purchased a specific set of items:
```bash
# Condition recommendations on seed items
python src/infer.py --user test_shopper --items "85123A,22423,47566" --top-k 5

# Discover only novel items (exclude previously purchased items)
python src/infer.py --user test_shopper --items "85123A,22423,47566" --no-allow-repeats --top-k 5
```

#### 4. Pre-computed Feature Matrix Direct Scoring
If you have already engineered the 20 canonical features, score them directly:
```bash
python src/infer.py --features path/to/precomputed_features.csv --output scored_recs.csv
```

#### 5. Output Formatting & Quiet Mode
Export recommendations formatted for consumption by downstream systems or APIs:
```bash
# Output clean JSON to stdout
python src/infer.py --dataset d1 --user 12349 --top-k 5 --format json --quiet

# Export to CSV
python src/infer.py --dataset d1 --user 12349 --top-k 10 --output recommendations.csv
```

#### 6. Interactive Terminal Wizard
Launch a step-by-step interactive CLI guide:
```bash
python src/infer.py --interactive
```

---

### Command-Line Arguments Reference (`src/infer.py`)

| Flag | Description | Default |
|------|-------------|---------|
| `-d`, `--dataset` | Base dataset domain: `d1` (Online Retail) or `d2` (Instacart) | `d1_online_retail` |
| `-m`, `--model-type` | Architecture: `rf` (Random Forest), `mf` (Matrix Factorization), `pop` (Popularity) | `rf` |
| `--model-path` | Custom file path to `.joblib` model binary | Auto-resolved |
| `-i`, `--data` | Path to custom transaction file (`.csv`, `.json`, `.parquet`) | Built-in dataset |
| `--features` | Path to precomputed feature matrix CSV | None |
| `-u`, `--user` | Target Customer / User ID to score | Top active users |
| `--items` | Comma-separated seed items for ad-hoc basket inference | None |
| `-k`, `--top-k` | Number of recommendations to generate per user | `10` |
| `--allow-repeats` | Toggle repeat-purchase recommendations (`--allow-repeats` / `--no-allow-repeats`) | `True` |
| `--max-users` | Maximum users to score when running batch inference on an entire dataset | `50` |
| `-o`, `--output` | Output file path (`.csv` or `.json`) to save recommendations | None (Console only) |
| `--format` | Output display format: `table`, `csv`, or `json` | `table` |
| `-q`, `--quiet` | Suppress info logs and output only recommendation results | `False` |
| `--interactive` | Launch terminal prompt wizard | `False` |

---

### Programmatic Python Usage

Load and use the components inside Python scripts or Jupyter notebooks:
```python
from src.infer import load_model, run_rf_inference, load_candidate_items, load_feature_columns
from src.data_loader import load_online_retail_cleaned

# 1. Load trained model & metadata
rf_model = load_model("d1_online_retail", model_type="rf")
candidates = load_candidate_items("d1_online_retail")
feature_cols = load_feature_columns("d1_online_retail")

# 2. Load historical transactions
df = load_online_retail_cleaned("data/processed/online_retail_cleaned.csv")

# 3. Generate Top-10 recommendations for customer 12349
top_recs = run_rf_inference(
    model=rf_model,
    history_df=df,
    target_users=["12349"],
    candidates=candidates,
    feature_cols=feature_cols,
    k=10,
    allow_repeats=True,
)
print(top_recs[["rank", "item_id", "score"]])
```
