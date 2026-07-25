# House Price Prediction & Valuation — A Comparative Machine Learning Approach

Lab assignment for the Predictive Analytics lab. This project implements a complete,
**reproducible** machine-learning workflow on two real-estate datasets — the
**UCI Real Estate Valuation** and the **Ames Housing** datasets — and ships a
command-line tool so the best models can be used for sample inference **without
retraining**, or retrained on new data.

The workflow covers: data preprocessing, exploratory data analysis (performed
**only on the training split** to prevent leakage), 5-fold cross-validation,
hyperparameter tuning, residual diagnostics, feature-importance analysis, and
artifact serialization with `joblib`.

> **Reproducibility**: every numeric result below was produced by executing the
> two notebooks end-to-end (`jupyter nbconvert --execute --inplace`) with a fixed
> seed (`SEED = 42`). The CLI's `list` and `info` commands read the same
> serialized artifacts, so the numbers here, in the notebooks' outputs, and at
> the command line all agree.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Quickstart](#quickstart)
3. [Datasets Overview](#datasets-overview)
4. [Notebook Workflows](#notebook-workflows)
5. [Model Results](#model-results)
   - [UCI Real Estate Valuation](#uci-real-estate-valuation)
   - [Ames Housing](#ames-housing)
6. [Reproducing the Experiments](#reproducing-the-experiments)
7. [Serialized Artifacts](#serialized-artifacts)
8. [CLI Reference](#cli-reference)
   - [`info`](#info)
   - [`list`](#list)
   - [`predict`](#predict)
   - [`evaluate`](#evaluate)
   - [`retrain`](#retrain)
   - [`interactive`](#interactive)
9. [Environment & Dependencies](#environment--dependencies)

---

## Project Structure

```
House_price_pred/
├── README.md                          # This file
├── requirements.txt                   # Pinned dependencies (verified on Python 3.10.0)
├── .gitignore
│
├── data/
│   └── AmesHousing.csv                # Ames Housing dataset (2,930 rows, 82 cols,
│                                      # incl. target 'SalePrice')
│
├── notebooks/
│   ├── uci_house.ipynb                # UCI Real Estate Valuation — full workflow
│   └── ames_housing.ipynb             # Ames Housing — full workflow
│
├── src/
│   └── house_price_cli.py             # CLI: inference, evaluation, retraining, registry
│
├── artifacts/                         # Serialized models, preprocessor, results & metadata
│   ├── uci_best_model.joblib          # Best UCI model (Tuned Random Forest)
│   ├── uci_Tuned_Random_Forest.joblib # ... and the other tuned/base models
│   ├── uci_preprocessing_pipeline.joblib  # Fitted ColumnTransformer (impute+scale+onehot)
│   ├── uci_all_model_results.csv      # Test-set metrics for all UCI models
│   ├── uci_cv_results.csv             # 5-fold CV metrics for UCI models
│   ├── uci_evaluation_results.csv     # Default-model test metrics
│   ├── uci_model_registry.json        # UCI tuned hyperparameters + best-model info
│   ├── uci_training_metadata.json     # UCI run metadata (split, env, best model)
│   ├── ames_best_model.joblib         # Best Ames model (Tuned XGBoost)
│   ├── ames_preprocessor.joblib       # Ames fitted ColumnTransformer
│   ├── ames_all_results.csv           # Test-set metrics for all Ames models
│   ├── ames_cv_results.csv            # 5-fold CV metrics for Ames models
│   └── ames_metadata.json             # Ames run metadata (best model, split)
│
└── examples/                          # Ready-to-run sample inputs for the CLI
    ├── uci_sample.json                # 8 labeled UCI records (for predict/evaluate)
    ├── uci_retrain_example.csv         # 80 labeled UCI rows (for a quick retrain)
    ├── ames_eval_sample.csv            # 10 labeled Ames rows (for predict/evaluate)
    └── ames_retrain_example.csv       # 40 labeled Ames rows (for a quick retrain)
```

Each notebook's first code cell bootstraps the working directory so the relative
paths (`data/AmesHousing.csv`, the `artifacts/` outputs) resolve no matter where
the notebook is launched from.

---

## Quickstart

```bash
# From the repo root (the folder above House_price_pred/), one-time setup:
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell   (or: source .venv/bin/activate)
cd House_price_pred
pip install -r requirements.txt

# Sanity check: are the libraries and shipped artifacts present?
python src/house_price_cli.py info

# Sample inference with the shipped best models — no retraining needed:
python src/house_price_cli.py predict  uci  examples/uci_sample.json
python src/house_price_cli.py evaluate ames examples/ames_eval_sample.csv

# See the full model registry + metrics:
python src/house_price_cli.py list uci
python src/house_price_cli.py list ames
```

---

## Datasets Overview

### 1. UCI Real Estate Valuation Dataset
* **Origin**: Sindian District, New Taipei City, Taiwan (compiled by I-Cheng Yeh, [UCI ID 477](https://archive.ics.uci.edu/dataset/477)).
* **Scale**: 414 instances, 6 features, 1 target.
* **Target**: `Y house price of unit area` (10,000 New Taiwan Dollar per Ping; 1 Ping = 3.3 m²).
* **Features**: `X1 transaction date`, `X2 house age`, `X3 distance to the nearest MRT station`,
  `X4 number of convenience stores`, `X5 latitude`, `X6 longitude`.

### 2. Ames Housing Dataset
* **Origin**: Compiled by Dean De Cock; house sales in Ames, Iowa (2006–2010).
* **Scale**: 2,930 instances, 79 explanatory features, 1 target.
* **Target**: `SalePrice` (USD).
* **Features**: 79 nominal, ordinal, discrete, and continuous variables (quality, zoning,
  location, dimensions, basement, garage, utilities, …).

---

## Notebook Workflows

[`notebooks/uci_house.ipynb`](notebooks/uci_house.ipynb) and
[`notebooks/ames_housing.ipynb`](notebooks/ames_housing.ipynb) follow the same
scikit-learn pipeline:

1. **Bootstrap & imports** — resolve the project root, fix `SEED = 42`, print library versions.
2. **Data audit** — shape, dtypes, missing values, duplicates, target distribution.
3. **Leakage-safe split** — 80/20 `train_test_split` **before** any EDA or fitting.
4. **EDA** (training split only) — target histogram/boxplot, scatter plots vs target, correlation heatmap, pairplots.
5. **Preprocessing pipeline** — `ColumnTransformer` with median-imputation + `StandardScaler` for
   numeric columns and most-frequent imputation + `OneHotEncoder(handle_unknown='ignore')` for
   categoricals.
6. **Naive baseline** — `DummyRegressor(strategy='mean')`.
7. **Simple linear regression** — one high-correlation feature (`X3…MRT` for UCI, `Gr Liv Area` for Ames).
8. **Linear models** — Multiple Linear, Ridge, Lasso, Elastic Net.
9. **Tree & ensemble models** — Decision Tree, Random Forest, Gradient Boosting, XGBoost.
10. **Cross-validation** — 5-fold CV (RMSE, MAE, R²) on the training fold.
11. **Hyperparameter tuning** — `GridSearchCV` on the strongest base learners.
12. **Final evaluation** — MAE, RMSE, R², MAPE on the held-out test split.
13. **Residual diagnostics** — residuals-vs-fitted, Q-Q plot, scale-location, normality tests.
14. **Feature importance** — linear coefficients + tree `feature_importances_`.
15. **Serialization** — `joblib` the best model, the preprocessor, every individual model, the
    results CSVs, and a JSON metadata/registry — all written to `artifacts/`.

---

## Model Results

All numbers below are loaded from the serialized CSVs/JSON in `artifacts/` (re-read live by
`python src/house_price_cli.py list {uci,ames}`).

### UCI Real Estate Valuation

#### 5-fold cross-validation (training set), sorted by CV RMSE

| | Model | CV MAE (mean) | CV RMSE (mean) | CV R² (mean) |
|:---:|:---|:---:|:---:|:---:|
| 1 | **XGBoost** | 5.1093 | 7.7122 | 0.6758 |
| 2 | **Random Forest** | 5.2242 | 7.7845 | 0.6670 |
| 3 | **Gradient Boosting** | 5.4448 | 8.3502 | 0.6195 |
| 4 | **Ridge** | 6.5455 | 9.1083 | 0.5481 |
| 5 | **Elastic Net** | 6.5455 | 9.1105 | 0.5478 |
| 6 | **Lasso** | 6.5455 | 9.1108 | 0.5478 |
| 7 | **Linear Regression** | 6.5453 | 9.1109 | 0.5478 |
| 8 | **Decision Tree** | 6.1188 | 9.5809 | 0.4958 |

#### Final test set, sorted by Test RMSE (naive Mean RMSE = 13.1137)

| Model | Test MAE | Test RMSE | Test R² | MAPE (%) |
|:---|:---:|:---:|:---:|:---:|
| **Tuned Random Forest** | **3.7342** | **5.5174** | **0.8185** | **11.61** |
| XGBoost | 3.7615 | 5.5327 | 0.8175 | 12.31 |
| Tuned XGBoost | 3.8926 | 5.6600 | 0.8090 | 12.47 |
| Gradient Boosting | 3.9036 | 5.8443 | 0.7964 | 11.96 |
| Random Forest | 4.2806 | 5.9220 | 0.7910 | 13.64 |
| Decision Tree | 4.7395 | 6.2470 | 0.7674 | 14.68 |
| Polynomial Regression | 4.9593 | 6.8856 | 0.7174 | 16.49 |
| Tuned Ridge | 5.2969 | 7.2835 | 0.6838 | 16.92 |
| Ridge Regression | 5.3022 | 7.3108 | 0.6814 | 17.15 |
| Elastic Net | 5.3048 | 7.3139 | 0.6811 | 17.18 |
| Lasso Regression | 5.3052 | 7.3144 | 0.6811 | 17.18 |
| Multiple Linear Regression | 5.3054 | 7.3148 | 0.6811 | 17.18 |
| Simple Linear Regression | 6.9210 | 8.7945 | 0.5390 | 23.90 |

> **Best UCI model — Tuned Random Forest** (`n_estimators=200`, `max_depth=10`,
> `min_samples_leaf=2`): Test RMSE **5.5174**, R² **0.8185** — a **57.9%** error
> reduction versus the naive-mean baseline.

---

### Ames Housing

#### 5-fold cross-validation (training set), sorted by CV RMSE

| | Model | CV MAE (mean) | CV RMSE (mean) | CV R² (mean) |
|:---:|:---|:---:|:---:|:---:|
| 1 | **XGBoost** | 14,915.10 | 24,277.95 | 0.8959 |
| 2 | **Gradient Boosting** | 15,095.39 | 24,717.69 | 0.8905 |
| 3 | **Random Forest** | 16,080.99 | 26,596.70 | 0.8754 |
| 4 | **Lasso** | 16,351.75 | 27,951.15 | 0.8548 |
| 5 | **Linear Regression** | 16,425.69 | 28,193.65 | 0.8516 |
| 6 | **Elastic Net** | 16,571.96 | 29,034.87 | 0.8450 |
| 7 | **Ridge** | 16,578.45 | 29,053.63 | 0.8448 |
| 8 | **Decision Tree** | 24,353.90 | 37,968.25 | 0.7514 |

#### Final test set, sorted by Test RMSE (naive Mean RMSE = 90,222.37)

| Model | Test MAE | Test RMSE | Test R² | MAPE (%) |
|:---|:---:|:---:|:---:|:---:|
| **Tuned XGBoost** | **14,693.53** | **24,676.31** | **0.9241** | **7.75** |
| XGBoost | 15,045.41 | 25,100.92 | 0.9161 | 8.04 |
| Gradient Boosting | 15,165.20 | 25,928.42 | 0.9102 | 8.06 |
| Random Forest | 15,729.69 | 26,827.57 | 0.9096 | 8.53 |
| Ridge | 16,186.78 | 28,896.36 | 0.8959 | 8.77 |
| Elastic Net | 16,243.90 | 28,923.28 | 0.8957 | 8.78 |
| Lasso | 15,868.93 | 29,253.40 | 0.8933 | 9.18 |
| Linear Regression | 16,007.38 | 29,615.38 | 0.8906 | 9.28 |
| Decision Tree | 22,329.61 | 32,284.94 | 0.8700 | 12.59 |

> **Best Ames model — Tuned XGBoost** (`learning_rate=0.03`, `max_depth=4`,
> `n_estimators=300`, `subsample=0.8`): Test RMSE **24,676.31**, R² **0.9241** — a
> **72.6%** error reduction versus the naive-mean baseline. Note the plain
> **XGBoost** (default params) wins on cross-validation (CV RMSE 24,277.95) and is
> a very close second on test (RMSE 25,100.92); the tuned model edges it on this
> particular split.

---

## Reproducing the Experiments

### Prerequisites
* Python 3.10.x (verified on 3.10.0)
* A virtual environment (`venv` or `conda`)

### Steps

```bash
# 1. Clone & enter the project
git clone <repository_url>
cd House_price_pred

# 2. Create & activate a venv
python -m venv venv
.\venv\Scripts\Activate.ps1            # Windows PowerShell
# source venv/bin/activate             # macOS / Linux

# 3. Install pinned dependencies (also installs jupyter + nbconvert)
pip install --upgrade pip
pip install -r requirements.txt

# 4a. Reproduce the experiments in Jupyter (interactive)
jupyter notebook                       # open & run all cells in notebooks/

# 4b. Or reproduce end-to-end, non-interactively (regenerates artifacts/)
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=900 notebooks/uci_house.ipynb
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=900 notebooks/ames_housing.ipynb
```

Running either notebook overwrites the corresponding `artifacts/{uci,ames}_*`
files (models, preprocessor, results CSVs, metadata). Because `SEED = 42` is
fixed throughout, the metrics are deterministic.

---

## Serialized Artifacts

Trained pipelines (`preprocessing + estimator`) are saved to `artifacts/` as
`.joblib` files, so the CLI can make predictions on **raw** data without any
re-fitting — the `ColumnTransformer` applies imputation/scaling/one-hot encoding
automatically.

### Load a model directly (programmatically)

```python
import joblib, pandas as pd

pipe = joblib.load('artifacts/uci_best_model.joblib')   # full Pipeline
new = pd.DataFrame([{
    'X1 transaction date': 2013.500,
    'X2 house age': 13.3,
    'X3 distance to the nearest MRT station': 561.9845,
    'X4 number of convenience stores': 5,
    'X5 latitude': 24.98746,
    'X6 longitude': 121.54391}])
print(pipe.predict(new)[0])   # preprocessing is applied inside the pipeline
```

Prefer the CLI for everyday use — see the next section.

---

## CLI Reference

`src/house_price_cli.py` is the single entry point for inference, evaluation,
retraining and registry inspection. It resolves paths from its own location
(`src/`), so it works from any working directory — no `cd` required.

```text
python src/house_price_cli.py info
python src/house_price_cli.py list    {uci,ames}
python src/house_price_cli.py predict {uci,ames} <file.csv|file.json> [-o out] [--model NAME]
python src/house_price_cli.py evaluate {uci,ames} <file.csv|file.json> [-o out] [--model NAME] [--head N]
python src/house_price_cli.py retrain {uci,ames} <file.csv|file.json> [--tag NAME]
python src/house_price_cli.py interactive {uci,ames}
```

| Command | What it does |
|---|---|
| `info`      | Prints Python/library versions and an inventory of `artifacts/` (sanity check). |
| `list`      | Shows every model's test metrics (sorted by RMSE), the best-model info, tuned configs (UCI), artifact file list, and the feature list. |
| `predict`   | Loads the best (or a `--model`-named) pipeline and runs inference on a CSV/JSON file. Prints predictions; optionally saves with `-o`. |
| `evaluate`  | Like `predict`, but the input **must** include the target column; computes MAE/RMSE/R²/MAPE vs ground truth and prints actual-vs-predicted. |
| `retrain`   | Refits the preprocessing pipeline and all 8 models on new data (leakage-safe split, `SEED=42`), refits the best model on all new data, and **overwrites** the artifacts. |
| `interactive` | Prompts for feature values one at a time (UCI) or loads a CSV / dataset sample (Ames) and predicts. |

Input formats: `.csv` (any number of rows; for Ames include all 79 feature columns
— `Order`, `PID`, `SalePrice` are ignored automatically), or `.json` (an array of
objects, or an object with a `records`/`features`/`data`/`instances` list).

---

### `info`

```bash
python src/house_price_cli.py info
```

```
============================================================
      HOUSE PREDICTION - ENVIRONMENT & ARTIFACTS
============================================================

[1/3] Python environment
  [OK] python         3.10.0
  [OK] numpy          2.2.6
  [OK] pandas         2.3.2
  [OK] scikit-learn   1.7.2
  [OK] xgboost        3.2.0
  [OK] joblib         1.5.2
  [OK] scipy          1.15.3

[3/3] Artifact inventory
  --- UCI ---
  [OK] uci_best_model.joblib               2,347.0 KB
  [OK] uci_preprocessing_pipeline.joblib       2.9 KB
  [OK] uci_all_model_results.csv               1.9 KB
  [OK] uci_training_metadata.json             1.1 KB
  best model : Tuned Random Forest
  --- AMES ---
  ...
  Ready: both best models + preprocessors present.
```

---

### `list`

```bash
python src/house_price_cli.py list uci
```

Lists every model's test metrics (sorted by RMSE), the best-model info, the tuned
hyperparameter configs (UCI only — from the registry), the artifact file list with
sizes, and the feature list for the dataset.

```
============================================================
                    MODEL REGISTRY: UCI
============================================================
  All models (from uci_all_model_results.csv, sorted by Test_RMSE ->)
  ...
       Tuned Random Forest     5.5174   0.8185    3.7342 ...
                   XGBoost     5.5327   0.8175 ...
  * Best model
  ------------------------------
    Name : Tuned Random Forest
    RMSE : 5.5174
    R2   : 0.8185
    MAE  : 3.7342
    MAPE : 11.6080
  Tuned model configs (from registry)
  ------------------------------
    Random_Forest:
      model__max_depth: 10
      model__min_samples_leaf: 2
      model__n_estimators: 200
    ...
```

---

### `predict`

Inference on a CSV or JSON file with the best shipped model — **no retraining**.

```bash
python src/house_price_cli.py predict uci examples/uci_sample.json
python src/house_price_cli.py predict ames examples/ames_eval_sample.csv \
    -o predictions.csv --no-features
# Use a specific saved model instead of the best:
python src/house_price_cli.py predict uci examples/uci_sample.json --model XGBoost
```

Sample output (UCI):
```
============================================================
                    PREDICT - UCI
============================================================
  File  : examples/uci_sample.json
  Model : best (default)

[1/3] Load model
  [OK] loaded: Tuned Random Forest
[2/3] Load data
  [OK] 8 record(s), 6 feature(s)
[3/3] Inference

  -------------------------------------------------------
  PREDICTIONS (10,000 NTD/Ping)
  -------------------------------------------------------
  Record 1:
    X1 transaction date: 2012.917
    X2 house age: 32.0
    ...
    ->> Predicted Y house price of unit area: 46.7300 10,000 NTD/Ping
  ...
```

`examples/uci_sample.json`:
```json
[
  {"X1 transaction date": 2012.917, "X2 house age": 32.0,
   "X3 distance to the nearest MRT station": 84.87882,
   "X4 number of convenience stores": 10,
   "X5 latitude": 24.98298, "X6 longitude": 121.54024}
  // ...more records
]
```

---

### `evaluate`

Same as `predict`, but the input must contain the target column too; it then scores
the predictions against ground truth (MAE, RMSE, R², MAPE) and prints an
actual-vs-predicted table.

```bash
python src/house_price_cli.py evaluate uci  examples/uci_sample.json
python src/house_price_cli.py evaluate ames examples/ames_eval_sample.csv --head 5
python src/house_price_cli.py evaluate ames examples/ames_eval_sample.csv -o scored.csv
```

Sample output (Ames):
```
  MODEL: Tuned XGBoost
  -------------------------------------------------------
    MAE  : 12,203.2773  USD
    RMSE : 13,568.4511  USD
    R2   : 0.8643
    MAPE : 6.72 %
  -------------------------------------------------------
   Record          Actual       Predicted           Error
        1      215,000.00      207,346.36       -7,653.64
        2      105,000.00      122,105.49       17,105.49
        ...
```

> With `-o scored.csv`, every record is saved with its predicted value, signed
> error, and absolute percentage error.

---

### `retrain`

Refits the pipeline on **new data** and overwrites the artifacts in `artifacts/`.
This is the path to updating the shipped models.

```bash
python src/house_price_cli.py retrain uci  examples/uci_retrain_example.csv
python src/house_price_cli.py retrain ames examples/ames_retrain_example.csv
# Save under a custom tag (--tag) so the standard best-model files are untouched:
python src/house_price_cli.py retrain uci new_data.csv --tag uci_v2
```

**What retraining does (all leak-safe, `SEED = 42`):**
1. Loads the new CSV/JSON (must include the target column: `Y house price of unit area`
   for UCI, `SalePrice` for Ames).
2. Splits 80/20 into train/test.
3. Rebuilds & fits the preprocessing `ColumnTransformer` on the train split (median
   impute + scale for numeric; most-frequent + one-hot for categorical).
4. Trains all 8 base models, evaluates each on the test split, and picks the best by RMSE.
5. Refits the best model on **all** the new data and saves it as `{tag}_best_model.joblib`.
6. Saves the preprocessor, every individual model pipeline, a results CSV, and a
   metadata JSON.

```
[4/6] Train 8 models
  [OK] Linear Regression      RMSE=8.5328  R2=0.5133
  ...
  * Best on new test set: XGBoost  RMSE=6.4822  R2=0.7191
[5/6] Refit best model on ALL new data, save artifacts
  [OK] uci_best_model.joblib
  [OK] uci_preprocessing_pipeline.joblib
  [OK] 8 individual model pipelines
[6/6] Write metadata
  [OK] uci_training_metadata.json
  RETRAINING COMPLETE - artifacts overwritten. Use `list uci` to review.
```

Format requirements for the training file:
* **UCI**: the 6 feature columns + `Y house price of unit area`.
* **Ames**: the 79 feature columns + `SalePrice` (the `Order` and `PID` id columns
  may be present — they are dropped automatically).

---

### `interactive`

```bash
python src/house_price_cli.py interactive uci
```

Prompts for each UCI feature value and predicts; type `quit` to exit. For Ames
(79 features), it asks for a CSV path or uses a real record from `data/AmesHousing.csv`.

```
============================================================
  INTERACTIVE MODE: UCI
============================================================
  Loaded model: Tuned Random Forest
  Target: Y house price of unit area (10,000 NTD/Ping)

  Enter the 6 feature values. Type 'quit' to exit.
  ---------------------------------------------
  New prediction
  ---------------------------------------------
    X1 transaction date: 2013.5
    X2 house age: 13.3
    X3 distance to the nearest MRT station: 561.98
    X4 number of convenience stores: 5
    X5 latitude: 24.987
    X6 longitude: 121.544
    ->> Predicted Y house price of unit area: 46.7200 10,000 NTD/Ping
```

### Windows PowerShell notes
* Use relative paths or wrap absolute paths in quotes.
* Avoid inline JSON on the PowerShell command line (quoting); always pass a file.

---

## Environment & Dependencies

Trained and verified (notebook execution + CLI) under:

| Package | Version |
|---|---|
| Python | 3.10.0 |
| NumPy | 2.2.6 |
| Pandas | 2.3.2 |
| SciPy | 1.15.3 |
| scikit-learn | 1.7.2 |
| XGBoost | 3.2.0 |
| Joblib | 1.5.2 |
| Matplotlib | 3.10.6 |
| Seaborn | 0.13.2 |
| ucimlrepo | 0.0.7 |
| jupyter / nbconvert | installed via `requirements.txt` |

Install everything with:

```bash
pip install -r requirements.txt
```
