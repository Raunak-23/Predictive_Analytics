# Medical Diagnosis Support — Disease Classification Using Decision Trees

**MDI3003 · Advanced Predictive Analytics · Lab 02**

A leakage-safe, reproducible disease-status classification study built around
**CART Decision Trees**, with a required progression from basic CART → tuned &
pruned CART → Random Forest, plus an optional Logistic Regression benchmark.

The project ships three trained pipelines, three executable notebooks, full
visualisation artefacts, machine-readable results, model cards, an industry-style
clinical prediction report, and a CLI / GUI that can run sample inference
**without retraining**.

> **Educational-use boundary.** These models are research / teaching prototypes.
> They are **not** clinically validated diagnostic systems, must **not** be used
> for patient care, and must **not** be presented as medical advice.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Quickstart](#quickstart)
3. [Datasets](#datasets)
4. [Notebook Workflows](#notebook-workflows)
5. [Models & Protocol](#models--protocol)
6. [Serialized Artifacts](#serialized-artifacts)
7. [CLI Reference](#cli-reference)
8. [GUI](#gui)
9. [Deliverables](#deliverables)
10. [Environment](#environment)
11. [Responsible Use](#responsible-use)

---

## Project Structure

```
Medical_diagnosis_support/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/                              # Cached UCI CSVs (offline-safe)
│   ├── breast_cancer_wisconsin.csv    # Core lab (569 × 30 numeric)
│   ├── diabetes_risk_prediction.csv   # Extension (520 × mixed types)
│   └── heart_disease.csv              # Extension (303 × missing ca/thal)
│
├── notebooks/
│   ├── breast_cancer.ipynb            # Core 20-step lab notebook
│   ├── diabetes_risk.ipynb            # Mixed-type + gender subgroup
│   └── heart_disease.ipynb            # Missing values + sex subgroup
│
├── src/
│   ├── meddiag_common.py              # Shared engine (data, models, metrics)
│   ├── meddiag_cli.py                 # CLI: info/list/predict/evaluate/retrain/...
│   └── meddiag_gui.py                 # Tkinter desktop GUI
│
├── artifacts/                         # Pipelines, thresholds, CSVs, metadata
│   ├── figures/                       # PNG visualisations
│   ├── {ds}_best_model.joblib
│   ├── {ds}_threshold.joblib
│   ├── {ds}_cv_results.csv
│   ├── {ds}_test_results.csv
│   ├── {ds}_final_test_metrics.csv
│   ├── {ds}_metadata.json
│   └── ... per-model joblibs
│
├── examples/                          # Ready-to-run CLI inputs
│   ├── breast_sample.json / .csv
│   ├── diabetes_sample.json / .csv
│   └── heart_sample.json / .csv
│
├── reports/
│   ├── Lab02_Clinical_Prediction_Report.md
│   ├── model_card_breast.md
│   ├── model_card_diabetes.md
│   └── model_card_heart.md
│
└── tools/
    ├── build_notebooks.py             # (Re)generate the three notebooks
    ├── make_examples.py               # Build examples/ from data/
    └── train_all.py                   # Train all pipelines + figures
```

---

## Quickstart

```bash
# From the repo root (folder above Medical_diagnosis_support/)
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
cd Medical_diagnosis_support
pip install -r requirements.txt

# Sanity check: libraries + shipped artifacts
python src/meddiag_cli.py info

# Sample inference — no retraining needed
python src/meddiag_cli.py predict  breast  examples/breast_sample.json
python src/meddiag_cli.py evaluate heart   examples/heart_sample.csv
python src/meddiag_cli.py list     diabetes

# Trace a single record through the tree
python src/meddiag_cli.py explain breast examples/breast_sample.json --index 0

# Optional desktop GUI
python src/meddiag_cli.py gui
```

### Re-train everything from scratch

```bash
python tools/train_all.py
# or a single dataset:
python src/meddiag_cli.py retrain breast data/breast_cancer_wisconsin.csv
```

### Rebuild notebooks (template → .ipynb)

```bash
python tools/build_notebooks.py
```

---

## Datasets

| Key | Dataset | Role | Scale | Notes |
|---|---|---|---|---|
| `breast` | Breast Cancer Wisconsin (Diagnostic) · UCI 17 · DOI `10.24432/C5DW2B` | **Core lab** | 569 × 30 numeric | No missing values; FNA image-derived measurements; malignant vs benign |
| `diabetes` | Early Stage Diabetes Risk Prediction · UCI 529 · DOI `10.24432/C5VG8H` | Extended | 520 × 16 mixed | Categorical symptoms + age/gender; questionnaire setting |
| `heart` | Heart Disease (Cleveland) · UCI 45 · DOI `10.24432/C52P4X` | Extended | 303 × 13 | Missing `ca`/`thal`; target `num` binarised to presence vs absence |

**Positive class is always disease-present = 1.** Label mappings are fixed in
`meddiag_common.DATASETS` and written into every metadata JSON.

---

## Notebook Workflows

Each notebook follows the lab manual’s 20-step protocol:

1. Project charter & educational-use boundary  
2. Bootstrap, seed, software versions  
3. Load data & define the positive class  
4. Data & leakage audit  
5. Lock stratified 80/20 train–test split  
6. Focused EDA on the **training** split only  
7. Dummy baseline  
8. Basic (unconstrained) CART  
9. Tuned & cost-complexity-pruned CART  
10. Random Forest (+ Logistic Regression benchmark)  
11. Training-only 5-fold stratified CV comparison  
12. Out-of-fold threshold selection (declared sensitivity target)  
13. Single locked test evaluation  
14. Confusion matrix, ROC, PR, calibration  
15. Compact tree + decision-path audit  
16. Impurity vs permutation importance  
17. Stability (breast) or subgroup (diabetes/heart) experiment  
18. Persist artifacts + model card  
19. Evidence-based conclusion  
20. Responsible-use statement  

Heavy lifting is performed by `meddiag_common.train_dataset`, so notebook,
CLI and GUI results always agree under `SEED = 42`.

---

## Models & Protocol

| Level | Model | Focus |
|---|---|---|
| Baseline | `DummyClassifier(strategy="prior")` | Majority / prior floor |
| Basic | CART (`criterion="gini"`, unconstrained) | Overfitting demonstration |
| Advanced 1 | Tuned + cost-complexity-pruned CART | Bias–variance control, readable rules |
| Advanced 2 | Tuned Random Forest (≥300 trees) | Stability / ensemble |
| Optional | Logistic Regression (scaled) | Non-tree comparator |

**Non-negotiable experimental protocol**

1. Lock the test set **before** any tuning or threshold work.  
2. Fit imputers / encoders only inside training folds (sklearn `Pipeline`).  
3. Stratified 5-fold CV for model selection.  
4. Select model + threshold on training-only evidence.  
5. Evaluate the locked model on the test set **once**.  
6. Report sensitivity, specificity, precision, F1, balanced accuracy, ROC-AUC,
   PR-AUC, Brier, and confusion counts.  
7. Persist code, seeds, versions, results and metadata.

Default instructional costs: \(C_{FN}=5\), \(C_{FP}=1\). Default sensitivity
target: 0.95 (relaxed automatically if infeasible).

---

## Serialized Artifacts

For each dataset key `{ds}` ∈ {`breast`, `diabetes`, `heart`}:

| File | Contents |
|---|---|
| `{ds}_best_model.joblib` | Full fitted pipeline (preprocessor + estimator) |
| `{ds}_threshold.joblib` | Operating threshold (float) |
| `{ds}_*.joblib` | Individual model pipelines |
| `{ds}_cv_results.csv` | Training-only CV metrics (mean ± used in tables) |
| `{ds}_test_results.csv` | Locked test metrics for every model |
| `{ds}_final_test_metrics.csv` | Best-model test metrics row |
| `{ds}_threshold_curve.csv` | Sensitivity / specificity vs threshold |
| `{ds}_pruning_cv.csv` | Cost-complexity pruning CV grid |
| `{ds}_metadata.json` | Provenance, params, software, intended use |
| `figures/{ds}_*.png` | Class dist, CM, ROC, PR, calibration, tree, … |

---

## CLI Reference

The CLI mirrors the house-price lab tool: one entry point, absolute path
resolution from the script location, works from any working directory.

```text
python src/meddiag_cli.py <command> [dataset] [args]
```

| Command | Purpose |
|---|---|
| `info` | Python / library versions + artifact inventory |
| `list {ds}` | Model registry, CV + test tables, features |
| `predict {ds} <file>` | Inference on CSV/JSON → probabilities + class calls |
| `evaluate {ds} <file>` | Inference + full metric block vs ground-truth labels |
| `retrain {ds} <file>` | Refit the whole workflow; overwrite artifacts + figures |
| `interactive {ds}` | Prompt for feature values one at a time |
| `explain {ds} <file>` | Trace one record through the decision tree |
| `gui` | Launch the Tkinter desktop GUI |

### Examples

```bash
python src/meddiag_cli.py info
python src/meddiag_cli.py list breast
python src/meddiag_cli.py predict breast examples/breast_sample.json -o preds.csv
python src/meddiag_cli.py evaluate diabetes examples/diabetes_sample.csv
python src/meddiag_cli.py explain heart examples/heart_sample.json --index 0
python src/meddiag_cli.py retrain breast data/breast_cancer_wisconsin.csv
python src/meddiag_cli.py interactive breast
```

Common flags: `--model NAME`, `--threshold 0.4`, `--tag TAG`, `-o/--output`,
`--head N`, `--index N`.

---

## GUI

```bash
python src/meddiag_cli.py gui
# or
python src/meddiag_gui.py
```

Pick a trained dataset, edit feature controls (spinboxes / dropdowns), slide the
operating threshold, and inspect the predicted probability, class call, and
decision-path text. Same engine as the notebooks and CLI.

---

## Deliverables

| Deliverable | Location |
|---|---|
| Executable notebooks | `notebooks/*.ipynb` |
| Industry-style report | `reports/Lab02_Clinical_Prediction_Report.md` |
| Results CSVs | `artifacts/*_cv_results.csv`, `*_test_results.csv`, `*_final_test_metrics.csv` |
| Saved model artefacts | `artifacts/*_best_model.joblib` + threshold + metadata |
| Model cards | `reports/model_card_*.md` |
| Visualisations | `artifacts/figures/*.png` |
| CLI + GUI | `src/meddiag_cli.py`, `src/meddiag_gui.py` |

---

## Environment

Pinned in `requirements.txt` (verified on Python 3.10+):

```
numpy, pandas, scipy, scikit-learn, joblib,
matplotlib, seaborn, ucimlrepo, jupyter, nbconvert
```

```bash
pip install -r requirements.txt
```

---

## Responsible Use

* Public benchmark data only; no confidential patient records.  
* Do not claim clinical safety, diagnostic accuracy, or regulatory approval.  
* Report sensitivity / specificity / class counts — never accuracy alone.  
* Interpret tree rules and feature importance as **predictive**, not causal.  
* Subgroup attributes (gender / sex) are audit lenses, never eligibility criteria.  
* External multi-site validation, clinical review, and governance would be
  required before any real-world evaluation.

---

## References (selected)

1. Breiman et al., *Classification and Regression Trees*, 1984.  
2. Hastie, Tibshirani & Friedman, *Elements of Statistical Learning*, 2009.  
3. scikit-learn Decision Trees documentation.  
4. Wolberg et al., Breast Cancer Wisconsin (Diagnostic), DOI 10.24432/C5DW2B.  
5. Janosi et al., Heart Disease, DOI 10.24432/C52P4X.  
6. Early Stage Diabetes Risk Prediction, DOI 10.24432/C5VG8H.  
7. WHO, *Ethics and Governance of AI for Health*, 2021.
