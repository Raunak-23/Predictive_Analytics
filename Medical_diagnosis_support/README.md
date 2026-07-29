# Medical Diagnosis Support — Disease Classification Using Decision Trees

<!--
  Language: Python 3.10+
  Libraries: numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, joblib,
             ucimlrepo, jupyter, tkinter
  Author: Raunak Pal | Reg: 23MID0045 | MDI3003 Lab 02
  Repository: https://github.com/Raunak-23/Predictive_Analytics/tree/master/Medical_diagnosis_support
-->

An educational, leakage-safe prototype for **MDI3003 Lab 02** for study of
Decision Tree classification diagnostics. The two pipelines automatically fit
and select a model, lock an operating threshold via out-of-fold sensitivity
profiling, evaluate once on a holdout test set, and persist every artifact so
the **CLI and Tkinter GUI** can immediately perform zero-retraining inference
against the same saved pipeline &mdash; including the underlying Decision Tree
and its entire decision‑path trace.

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
| **CV 5‑fold (mean ± std)**     |                             |                                  |
| ROC‑AUC                       | 0.9954 ± 0.0048               | 0.9978 ± 0.0014                 |
| F1                            | 0.9616 ± 0.0241               | 0.9746 ± 0.0161               |
| Balanced accuracy             | 0.9689 ± 0.0195               | 0.9664 ± 0.0196               |
| **Locked test set**           |                             |                                  |
| Accuracy                      | 0.9716                       | 0.9631                         |
| ROC‑AUC                       | 0.9986                       | 0.9997                       |
| Sensitivity / Specificity     | 0.9737 / 0.9690                 | 0.9315 / 0.9899                 |
| **Confusion matrix**          | TN 71, FP 1, FN 3, TP 39  | TN 40, FP 0, FN 4, TP 60 |

## Project structure

```
Medical_diagnosis_support/
├── notebooks/               # The two primary deliverable Jupyter notebooks
│   ├── breast_cancer.ipynb
│   └── diabetes.ipynb
├── src/                     # Shared engine + CLI + GUI
│   ├── meddiag_common.py    # Dataset metadata, pipeline builder, CV, threshold,
│   │                        #   persist‑load, plotting helpers
│   ├── meddiag_cli.py       # Full interactive/headless CLI (info, samples, list,
│   │                        #   predict, evaluate, retrain, interactive, explain,
│   │                        #   gui)
│   └── meddiag_gui.py       # Tkinter GUI: dataset selection, feature spinboxes,
│                            #   live threshold slider, predict, decision‑path
│                            #   explain, model‑info & CV popups
├── tools/                   # Reproducible notebooks generator & notebook
│   │                        #   comment-refinement helpers
│   ├── _refine_helpers.py
│   ├── refine_breast.py / refine_diabetes.py
│   ├── build_notebooks.py
│   ├── train_all.py         # 1‑command retrain for all configured datasets
│   └── make_examples.py
├── data/                    # Static CSV inputs
├── examples/                # tiny (>‑100 row) JSON/CSV fixtures for quick
│                            #   inference demonstrations
├── artifacts/               # Canonical save directory (created by train_all or
│                            #   the notebooks themselves after refinement)
└── requirements.txt
```

## Getting started

### 1. Create and activate a virtual environment (recommended)

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
  diabetes_data_upload.csv
```

The notebooks also know how to pull them from the UCI ML Repository via
`ucimlrepo`.

### 4. Train (or re‑train) the pipelines

If the `artifacts/` directory is empty, run one of:

```powershell
# BOTH datasets at once
python tools/train_all.py --datasets breast diabetes

# Or a single dataset via the CLI
python src/meddiag_cli.py retrain breast data/breast_cancer_wisconsin.csv
```

The training protocol (identical in notebooks and CLI):

1. Audit schema, missingness, duplicates, constant / leakage columns
2. **Stratified 80/20 split** and lock the test set
3. Four‑model CV (dummy, raw CART, cost‑complexity pruned CART, Random Forest)
4. **Out‑of‑fold threshold sweep** to reach the declared sensitivity target
5. Freeze the best model and evaluate **once** on the locked test set
6. Persist pipelines, metadata and every CSV report to `artifacts/`

### 5. Notebooks

Open and `Restart & Run All`:

```
notebooks/breast_cancer.ipynb
notebooks/diabetes.ipynb
```

The notebooks write artifacts into `artifacts/`, keeping them synced with the
CLI/GUI. When run fresh, **execute the retrain cell** (cell 106 / 75) after the
training loop to populate the canonical artifact set; the old notebooks used
ad‑hoc bundles, but the refined notebooks call `meddiag_common.save_artifacts`.

## CLI

```text
usage: meddiag_cli.py [-h] [--version]
                      {info,gui,samples,list,predict,evaluate,retrain,
                       interactive,explain} ...

Commands
────────
  info         Show environment versions & artifact inventory (sanity check).
  samples      Isolated inference on bundled `examples/<ds>_sample.json` fixtures
               — the quickest sample‑inference fetch from saved artefacts
               (no retraining).
  list         Print the model registry, CV + test metrics and artifact files.
  predict      Run inference on a CSV/JSON file -> probabilities + class calls.
  evaluate     Inference + scoring against ground‑truth labels (full metric block).
  retrain      Refit the whole training workflow on new data, overwrite artifacts.
  interactive  Prompt for feature values one at a time and predict.
  explain      Trace the decision path of a single record through the tree.
  gui          Launch the Tkinter desktop GUI.
```

### Quick-run examples (from the `Medical_diagnosis_support/` directory)

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

# Evaluate — needs ground‑truth labels
python src/meddiag_cli.py evaluate diabetes examples/diabetes_sample.csv

# Interactively enter features one‑by‑one
python src/meddiag_cli.py interactive diabetes

# Trace the decision path of the first record in the breast fixture
python src/meddiag_cli.py explain breast examples/breast_sample.json --index 0

# Launch the Tkinter desktop GUI
python src/meddiag_cli.py gui
```

## Tkinter GUI

```powershell
python src/meddiag_cli.py gui    #  or:  python src/meddiag_gui.py
```

The GUI is designed for **non‑technical users**: pick a dataset, click
"Load model", press "Use sample row" to populate features, then click
**Predict** to see a live probability and class call. Dragging the **threshold
slider** re‑predicts instantly so a learner can feel the sensitivity‑specificity
tradeoff. Click **Explain path** to see the Decision Tree split‑by‑split.

Buttons in the toolbar:

- **Info 📋** — metadata report (dataset source, population, model, threshold)
- **CV summary** — 5‑fold cross‑validation table for every model in the registry
- **Conf matrix** — contingency table preview for the selected threshold

All three buttons get their data from `artifacts/<ds>_metadata.json` only — no
retraining is ever triggered from the GUI.

## Reproducible protocol

Both notebooks follow the lab manual and use `SEED = 42`. The protocol is:

1. Define the population, target, positive class, prediction time, intended use,
   and prohibited use.
2. Audit schema, missingness, duplicates, identifiers, class balance, and
   leakage risks.
3. Lock a stratified 80/20 split before tuning.
4. Fit mean‑imputation in a `ColumnTransformer` inside a `Pipeline`, so
   validation folds cannot influence feature transforms.
5. Compare **Dummy (prior)**, **basic CART**, **cost‑complexity‑pruned CART**,
   and **Random Forest** with *training‑only* stratified 5‑fold CV.
6. Select the operating threshold from out‑of‑fold predictions via swept
   Youden J with a high‑sensitivity target (breast ≥ 0.95; diabetes
   FN=5 × FP cost).
7. Once a model+threshold are chosen, evaluate **exactly once** on the locked
   test set – no iterative test‑set tuning.
8. Report: sensitivity and specificity, precision, F1, balanced accuracy,
   ROC‑AUC, PR‑AUC, Brier score, and the full confusion table.
9. Persist everything (estimator `.jobib`, metadata `.json`, CV and test
   `.csv`, threshold and pruning‑scan CSV) via
   `meddiag_common.save_artifacts`.

## Technical stack

| Component       | Technology / library               |
|----------------|-----------------------------------|
| Language       | Python 3.10                       |
| Core ML        | scikit‑learn 1.7.2                |
| Models         | Decision Tree (default + CART pruned), Random Forest, Logistic Regression |
| Data           | pandas 2.3.2, ucimlrepo 0.0.7   |
| Scientific     | numpy 2.2.6, scipy 1.15.3        |
| Visualization  | matplotlib 3.10.6, seaborn 0.13.2|
| Persistence    | joblib 1.5.2 + JSON / CSV          |
| CLI            | argparse (stdlib)                  |
| GUI (TK)       | tkinter (CPython stdlib)           |
| Notebooks      | Jupyter / nbconvert              |

## Datasets

| Dataset                                | UCI ID | DOI               | Rows | Cols | Balance      |
|----------------------------------------|--------|--------------------|------|------|-------------|
| Breast Cancer Wisconsin (Diagnostic)    | 17     | 10.24432/C5DW2B    | 569  | 30   | 37.3% M      |
| Early Stage Diabetes Risk Prediction    | 529    | 10.24432/C5VG8H    | 520  | 16   | 61.5% pos    |

## Important Remarks

- This is an **educational** project; the performance figures are derived from a
  single split and should never be interpreted as clinically validated.
- For the diabetes dataset, `Yes` / `No` are treated as nominal categories and
  one‑hot‑encoded — no ordinal scale is assumed.
- The "**false‑negative cost is higher than false‑positive**" assumption is a
  teaching experiment; it does not constitute a clinical standard.
- Subgroup / fairness audits are limited: the BC dataset lacks demographic
  attributes; the diabetes dataset includes age and gender only.
- All inference commands use **saved artifacts only**, ensuring the pipeline
  cannot accidentally differ from what the notebooks produce.

## Repo maintenance

- **Notebook refinements**: if a cell structure changes, re‑run the tools in
  `tools/refine_*.py` to re‑normalize markdown sections and comment templates.
- **Artifacts**: `tools/train_all.py` re‑builds the canonical `artifacts/`
  directory. Stale ad‑hoc bundles in `notebooks/artifacts*/` have been purged.
- **GitHub**: the parent repository’s `.gitignore` excludes `*.docx`, `*.pdf`,
  and `*.html`; only the notebook source and the shared code are tracked in
  `Medical_diagnosis_support/`.

## References

- Laboratory manual:
  `../Medical-Diagnosis-Support_-Disease-Classification-Using-Decision-Trees.pdf`
- Breiman et al., *Classification and Regression Trees*, 1984
- scikit‑learn Decision Tree and cost‑complexity pruning documentation
- Early Stage Diabetes Risk Prediction, UCI ML Repository, DOI 10.24432/C5VG8H
- Breast Cancer Wisconsin (Diagnostic), UCI ML Repository, DOI 10.24432/C5DW2B