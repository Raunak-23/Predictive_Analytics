# Medical Diagnosis Support — Disease Classification Using Decision Trees

<!--
  Language: Python 3.10+
  Libraries: numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, joblib,
             ucimlrepo, jupyter, streamlit
  Author: Raunak Pal | Reg: 23MID0045 | MDI3003 Lab 02
  Repository: https://github.com/Raunak-23/Predictive_Analytics/tree/master/Medical_diagnosis_support
-->

An educational, leakage-safe prototype for **MDI3003 Lab 02** — study of
Decision Tree classification diagnostics. The two pipelines automatically fit
and select a model, lock an operating threshold via out-of-fold sensitivity
profiling, evaluate once on a holdout test set, and persist every artifact so
the **CLI and Streamlit GUI** can immediately perform zero-retraining inference
against the same saved pipeline — including the underlying Decision Tree
and its entire decision-path trace.

> **Boundary**: these are research/teaching prototypes. They must **not** be
> used for patient care, treatment, triage, screening, or medical advice.

---

## Key Results

| Metric                        | Breast Cancer Wisconsin      | Early Stage Diabetes Risk        |
|-------------------------------|-----------------------------|----------------------------------|
| Best selected model           | Logistic Regression            | Random Forest                   |
| Operating threshold           | 0.620                         | 0.615                           |
| Rows / predictors             | 569 / 30                    | 520 / 16                       |
| Positive-class prevalence     | 37.4 %                       | 61.5 %                          |
| **CV 5-fold (mean +/- std)**    |                             |                                  |
| ROC-AUC                       | 0.9954 +/- 0.0048               | 0.9978 +/- 0.0014                 |
| F1                            | 0.9616 +/- 0.0241               | 0.9746 +/- 0.0161               |
| Balanced accuracy             | 0.9689 +/- 0.0195               | 0.9664 +/- 0.0196               |
| **Locked test set**           |                             |                                  |
| Accuracy                      | 0.9716                       | 0.9631                         |
| ROC-AUC                       | 0.9986                       | 0.9997                       |
| Sensitivity / Specificity     | 0.9737 / 0.9690                 | 0.9315 / 0.9899                 |
| **Confusion matrix**          | TN 71, FP 1, FN 3, TP 39  | TN 40, FP 0, FN 4, TP 60 |

## Project structure

```
Medical_diagnosis_support/
├── notebooks/               # Deliverable Jupyter notebooks
│   ├── breast_cancer.ipynb
│   └── diabetes.ipynb
├── src/                     # Shared engine + CLI + Streamlit GUI
│   ├── meddiag_common.py    # Dataset metadata, pipeline builder, CV, threshold,
│   │                        #   persist/load, plotting helpers
│   ├── meddiag_cli.py       # CLI (info, samples, list, predict, evaluate,
│   │                        #   retrain, interactive, explain, gui)
│   ├── meddiag_gui.py       # Legacy Tkinter GUI (retained for reference)
│   └── meddiag_streamlit.py # Streamlit GUI (primary – `streamlit run`)
├── tools/                   # Build + train utilities
│   ├── build_notebooks.py   # Generate reproducible notebooks from template cells
│   ├── train_all.py         # 1-command retrain for all configured datasets
│   └── make_examples.py     # Generate example fixtures
├── data/                    # Static CSV inputs
├── examples/                # Small JSON/CSV fixtures for quick inference demos
├── artifacts/               # Saved pipelines, threshold, metadata, CSV reports,
│   └── figures/             #   300 DPI PNGs (confusion matrix, ROC, PR, etc.)
└── requirements.txt
```

## Getting started

### 1. Create & activate a virtual environment (recommended)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # macOS / Linux
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Drop the offline CSV files into `data/`

```
data/
  breast_cancer_wisconsin.csv
  diabetes_risk_prediction.csv
```

The notebooks also know how to pull them from the UCI ML Repository via
`ucimlrepo`.

### 4. Train (or re-train) pipelines

```powershell
# Both datasets at once
python tools/train_all.py --datasets breast diabetes

# Or a single dataset via the CLI
python src/meddiag_cli.py retrain breast data/breast_cancer_wisconsin.csv
```

Training protocol (identical in notebooks, CLI and `train_all.py`):
1. Audit schema, missingness, duplicates, constant/leakage columns
2. **Stratified 80/20 split** and lock the test set
3. Four-model CV (dummy, raw CART, cost-complexity pruned CART, Random Forest)
4. **Out-of-fold threshold sweep** to reach the declared sensitivity target
5. Freeze the best model, evaluate **once** on the locked test set
6. Persist pipelines, metadata, CSV reports and figures to `artifacts/`

### 5. Notebooks

Open and `Restart & Run All`:

```
notebooks/breast_cancer.ipynb
notebooks/diabetes.ipynb
```

The final cell of each notebook calls `meddiag_common.save_artifacts` to keep
everything synced with the CLI and GUI.

## CLI

```text
usage: meddiag_cli.py [-h] [--version]
                      {info,gui,samples,list,predict,evaluate,retrain,
                       interactive,explain} ...

Commands
--------
  info         Show environment versions & artifact inventory (sanity check).
  samples      Isolated inference on bundled `examples/<ds>_sample.json` fixtures
               -- the quickest sample-inference fetch from saved artefacts
               (no retraining).
  list         Print the model registry, CV + test metrics and artifact files.
  predict      Run inference on a CSV/JSON file -> probabilities + class calls.
  evaluate     Inference + scoring against ground-truth labels (full metric block).
  retrain      Refit the whole training workflow on new data, overwrite artifacts.
  interactive  Prompt for feature values one at a time and predict.
  explain      Trace the decision path of a single record through the tree.
  gui          Launch the Streamlit desktop GUI.
```

### Quick-run examples (from `Medical_diagnosis_support/`)

```powershell
# Show environment + artifact manifest
python src/meddiag_cli.py info

# Fetch sample inference from the bundled fixtures (no file path required)
python src/meddiag_cli.py samples breast
python src/meddiag_cli.py samples diabetes

# List the full model registry with CV & test numbers
python src/meddiag_cli.py list breast

# Isolated inference on your own data
python src/meddiag_cli.py predict breast my_patients.csv
python src/meddiag_cli.py predict diabetes my_patients.csv --model 'Tuned and pruned CART'

# Evaluate -- needs ground-truth labels
python src/meddiag_cli.py evaluate diabetes examples/diabetes_sample.csv

# Interactively enter features one-by-one
python src/meddiag_cli.py interactive diabetes

# Trace the decision path of the first record
python src/meddiag_cli.py explain breast examples/breast_sample.json --index 0

# Launch the Streamlit GUI
streamlit run src/meddiag_streamlit.py
```

## Streamlit GUI

```powershell
streamlit run src/meddiag_streamlit.py     # opens in browser
#  or:  python src/meddiag_cli.py gui        # same
```

The GUI is designed for **non-technical users**: pick a dataset, click
**"Load model"**, press **"Draw random sample"** to pull a fresh row from the
cached dataset, then click **"Predict"** to see a live probability and class
call. The threshold **slider** updates instantly -- drag it to feel the
sensitivity-specificity tradeoff. Feature values are editable via spinboxes
(numeric) and dropdowns (categorical).

Expandable sections:

- **Model info & CV summary** -- metadata, dataset specs, and the 5-fold
  cross-validation table for every model in the registry
- **Evaluation charts** -- confusion-matrix, ROC, precision-recall,
  threshold sweep, calibration plot, decision tree (all loaded from
  `artifacts/figures/`)

No retraining is ever triggered. Everything reads from saved artefacts.

## Reproducible protocol

Both notebooks follow the lab manual and use `SEED = 42`:
1. Define the population, target, positive class, prediction time, intended use
   and prohibited use.
2. Audit schema, missingness, duplicates, identifiers, class balance, and
   leakage risks.
3. Lock a stratified 80/20 split before tuning.
4. Fit mean-imputation in a `ColumnTransformer` inside a `Pipeline`, so
   validation folds cannot influence feature transforms.
5. Compare **Dummy (prior)**, **basic CART**, **cost-complexity-pruned CART**,
   and **Random Forest** with *fully training* stratified 5-fold CV.
6. Select the operating threshold from out-of-fold predictions via swept
   Youden J with a high-sensitivity target (breast >= 0.95; diabetes
   FN = 5x FP cost).
7. Once a model+threshold are chosen, evaluate **exactly once** on the locked
   test set -- no iterative test-set tuning.
8. Report: sensitivity, specificity, precision, F1, balanced accuracy,
   ROC-AUC, PR-AUC, Brier score, and the full confusion table.
9. Persist everything (pipeline `.joblib`, metadata `.json`, CV and test
   `.csv`, threshold and pruning-scan CSV, 300 DPI figures) via
   `meddiag_common.save_artifacts`.

## Technical stack

| Component       | Technology / library               |
|----------------|-----------------------------------|
| Language       | Python 3.10                       |
| Core ML        | scikit-learn 1.7.2                |
| Models         | Decision Tree (default + CART pruned), Random Forest, Logistic Regression |
| Data           | pandas 2.3.2, ucimlrepo 0.0.7    |
| Scientific     | numpy 2.2.6, scipy 1.15.3         |
| Visualization  | matplotlib 3.10.6, seaborn 0.13.2 |
| Persistence    | joblib 1.5.2 + JSON / CSV          |
| CLI            | argparse (stdlib)                  |
| GUI            | Streamlit 1.32+                    |
| Notebooks      | Jupyter / nbconvert               |

## Data sets

| Dataset                               | UCI ID | DOI            | Rows | Cols | Balance      |
|---------------------------------------|--------|----------------|------|------|--------------|
| Breast Cancer Wisconsin (Diagnostic)  | 17     | 10.24432/C5DW2B | 569  | 30   | 37.3% M      |
| Early Stage Diabetes Risk Prediction  | 529    | 10.24432/C5VG8H | 520  | 16   | 61.5% positive |

## Important Remarks

- This is an **educational** project; performance figures derive from a
  single split and must not be interpreted as clinically validated.
- For the diabetes dataset, `Yes` / `No` are treated as nominal categories and
  one-hot-encoded -- no ordinal scale is assumed.
- The "**false-negative cost is higher than false-positive**" assumption is a
  teaching experiment; it does not constitute a clinical standard.
- Subgroup / fairness audits are limited: the BC dataset lacks demographic
  attributes; the diabetes dataset includes age and gender only.
- All inference commands use **saved artifacts only**, ensuring the pipeline
  never accidentally differs from what the notebooks produce.

## Repo maintenance

- **Artifacts**: `tools/train_all.py` re-builds the canonical `artifacts/`
  directory.
- **GitHub**: the parent repository's `.gitignore` excludes `*.pdf`, `*.docx`,
  and rendered `*.html`; only notebook source and shared code are tracked.
- **Legacy**: the old Tkinter GUI (`src/meddiag_gui.py`) is retained for reference
  but the primary UI is now Streamlit.

## References

- Laboratory manual:
  `../Medical-Diagnosis-Support_-Disease-Classification-Using-Decision-Trees.pdf`
- Breiman et al., *Classification and Regression Trees*, 1984
- scikit-learn Decision Tree and cost-complexity pruning documentation
- Early Stage Diabetes Risk Prediction, UCI ML Repository, DOI 10.24432/C5VG8H
- Breast Cancer Wisconsin (Diagnostic), UCI ML Repository, DOI 10.24432/C5DW2B