# Agricultural Predictive Analytics (MDI3003 — Experiment 08)

An end-to-end, reproducible machine learning system evaluating district-level rice yield prediction on official Government of India agricultural statistics (Track A) and multi-class crop-label prediction (Track B).

---

## Methodology

### Track A: Core Regression (Rice Yield)
```
D1 Government API 
  → Canonicalization (t/ha derivation, unit validation, duplicate check)
  → Training EDA (restricted strictly to training rows; holdouts uninspected)
  → Chronological Split (Train: 1997–2011, Val: 2012–2013, Test: 2014–2015)
  → Preprocessing (OHE for categories, Median Impute + StandardScale for year)
  → Model Comparison (Median, Ridge Trend, Decision Tree, Random Forest)
  → Time-Aware Tuning (Expanding-window CV on training years only)
  → Locked Test Evaluation (Enforced one-time TEST_LOCK)
  → Robustness & Error Diagnostics (Rolling origins, year-wise metrics, top-5 errors)
  → Validated Inference (Range & category guards, self-contained bundles)
```

### Track B: Classification Extension (D3 Dataset)
Independent supervised benchmark predicting 22 crop classes from soil/climate features using an independence-gated stratified split, standard scaling, and random forest tuning.

---

## Results

### Track A: Locked Test Set (Harvest Years 2014–2015, $N=846$)
| Model Family | Status | Test MAE ($t/ha$) | Test RMSE ($t/ha$) | Test $R^2$ |
| :--- | :--- | :---: | :---: | :---: |
| **Ridge Trend (Tuned)** | **Selected** | **0.4883** | **0.7377** | **0.2643** |
| Median Baseline | Baseline | 0.7637 | 0.9674 | -0.2649 |

### Track B: Locked Test Set ($N=440$)
| Model Family | Status | Macro $F_1$ | Accuracy | Macro Precision | Macro Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **Selected** | **0.9909** | **0.9909** | **0.9919** | **0.9909** |
| Majority Baseline | Baseline | 0.0040 | 0.0455 | 0.0021 | 0.0455 |

---

## How to Run

### 1. Environment Setup
```bash
# Set your API key in .env (GOV_API_KEY canonical, DATA_API also supported; never commit .env to Git)
echo "GOV_API_KEY=your_data_gov_in_api_key_here" > .env

# Install dependencies
pip install -r requirements.txt
```

### 2. Execution Pipeline
```bash
# Acquire official D1 statistics from data.gov.in API
python -m src.acquire_d1

# Prepare canonical datasets
python -m src.prepare_data

# Run strict schema and leakage validation
python -m src.validate_data

# Create split manifests
python -m src.split_data --justify-iid

# Development stage: candidate fitting, validation comparison, temporal tuning
python -m src.train_regression --stage development

# Locked test stage: creates TEST_LOCK and evaluates final holdout
python -m src.train_regression --stage test

# Run classification extension (requires explicit IID review justification)
python -m src.train_classification --justify-iid

# Verify reproducibility and acceptance
python -m src.verify_reproducibility

# Run automated unit test suite
python -m unittest discover -s tests

```

### 3. Offline Validated Inference
```bash
python -m src.predict --model models/regression/selected_bundle.joblib --input artifacts/sample_regression_input.json
```

---

## Repository Structure

```
├── src/            # Modular Python implementations (acquisition, prep, models, eval, inference)
├── tests/          # Automated integrity, leakage, and validation unit tests
├── data/           # Raw (gitignored) and processed canonical datasets
├── results/        # Generated validation, test, error analysis, and robustness CSVs
├── artifacts/      # Manifests, hashes, TEST_LOCK, model bundles, and acceptance report
├── figures/        # Actual vs predicted, residuals, and comparison plots
├── models/         # Serialized candidate pipelines and inference bundles
├── notebooks/      # Executed Jupyter notebooks with rendered outputs and interpretations
├── README.md       # Crisp project overview
└── requirements.txt# Minimal pinned dependencies
```

---

## Limitations

- **Retrospective Scope:** Evaluated on historical series (1997–2015); does not extrapolate reliably to unobserved districts or future decades without updated calibration.
- **Omitted Agronomic Features:** Lacks in-season weather events, high-resolution remote sensing, pest incidence, and seed variety adoption data.
- **Classification Nature:** Track B predicts dataset labels from synthetic profiles, not agronomic crop-choice prescriptions.
