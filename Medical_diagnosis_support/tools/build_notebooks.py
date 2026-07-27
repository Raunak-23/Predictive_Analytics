#!/usr/bin/env python3
"""
Build the three Lab-02 Jupyter notebooks from template cells.

Run once to (re)generate:
    notebooks/breast_cancer.ipynb      (core lab - Breast Cancer Wisconsin)
    notebooks/diabetes_risk.ipynb     (extension - mixed / categorical)
    notebooks/heart_disease.ipynb     (extension - missing values + subgroup)

Every notebook imports the shared engine ``src/meddiag_common`` and follows the
same 20-step protocol so the notebook, CLI and GUI cannot disagree. Markdown
narrative + code + inline figures render directly; executing end-to-end writes
the artifacts under ``artifacts/``.

Usage:
    python tools/build_notebooks.py
"""
from __future__ import annotations

import os
import sys

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NB_DIR = os.path.join(ROOT, "notebooks")
SRC_DIR = os.path.join(ROOT, "src")
os.makedirs(NB_DIR, exist_ok=True)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import meddiag_common as M  # noqa: E402

DISCLAIMER_NOTE = (
    "EDUCATIONAL-USE BOUNDARY — the artefacts are a research/teaching "
    "prototype and are NOT a clinically validated diagnostic system; "
    "never use for patient care."
)


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


# ---------------------------------------------------------------------------
# Shared bootstrap cell
# ---------------------------------------------------------------------------
BOOTSTRAP = r'''
import sys, os, platform, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Resolve the project root from this notebook's location so the shared engine
# (src/meddiag_common.py) and the cached datasets resolve no matter where the
# notebook is launched from.
cwd = Path.cwd().resolve()
if cwd.name == "notebooks":
    PROJECT_ROOT = cwd.parent
elif (cwd / "src" / "meddiag_common.py").exists():
    PROJECT_ROOT = cwd
elif (cwd / "Medical_diagnosis_support" / "src" / "meddiag_common.py").exists():
    PROJECT_ROOT = cwd / "Medical_diagnosis_support"
else:
    # walk up a few levels looking for the engine
    PROJECT_ROOT = cwd
    for p in [cwd, *cwd.parents]:
        if (p / "src" / "meddiag_common.py").exists():
            PROJECT_ROOT = p
            break

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import meddiag_common as M

plt = M.init_plot_style()
sns.set_style("whitegrid", {
    "grid.linestyle": "-",
    "grid.color": M.PALETTE["gridline"],
})

print("Python :", platform.python_version())
print("NumPy  :", np.__version__, "| pandas:", pd.__version__,
      "| scikit-learn:", M._SK_VERSION)
print("Project:", PROJECT_ROOT)
print("Datasets available:", list(M.CHOICES))

DISCLAIMER = (
    "EDUCATIONAL-USE BOUNDARY  -  This notebook builds a research / teaching "
    "prototype for MDI3003 Lab 02. The models are NOT clinically validated "
    "diagnostic systems and must NOT be used for patient care, treatment or "
    "triage. Every dataset-derived threshold is a model property, not a "
    "clinical rule.")
print(DISCLAIMER)

RANDOM_STATE = M.SEED
TARGET_SENSITIVITY = M.TARGET_SENSITIVITY
COST_FN, COST_FP = M.COST_FN, M.COST_FP
print(f"\nRANDOM_STATE={RANDOM_STATE} | CV folds={M.CV_FOLDS} | "
      f"test_size={M.TEST_SIZE} | target sensitivity={TARGET_SENSITIVITY} "
      f"| cost FN={COST_FN}, FP={COST_FP}")

fig_count = 0

def fig_num():
    global fig_count
    fig_count += 1
    return fig_count

def save_show(name, tight=True):
    """Save under artifacts/figures/ and display inline."""
    if tight:
        plt.tight_layout()
    path = M.save_fig(plt, name)
    print(f"[Fig {fig_num()}] saved -> {path}")
    plt.show()
'''


def header_cells(ds_key: str):
    s = M.spec(ds_key)
    title_map = {
        "breast": (
            "Medical Diagnosis Support — Disease Classification using Decision Trees",
            "Lab 02 — CORE laboratory  ·  Breast Cancer Wisconsin (Diagnostic)",
        ),
        "diabetes": (
            "Medical Diagnosis Support — Disease Classification using Decision Trees",
            "Lab 02 — EXTENDED study  ·  Early Stage Diabetes Risk Prediction "
            "(mixed / categorical)",
        ),
        "heart": (
            "Medical Diagnosis Support — Disease Classification using Decision Trees",
            "Lab 02 — EXTENDED study  ·  Heart Disease (Cleveland; missing values "
            "+ subgroup)",
        ),
    }
    main, sub = title_map[ds_key]

    if ds_key == "breast":
        charter_extra = (
            "- **Positive class** = `M` (Malignant) remapped to `1`. Disease present "
            "is the costly error.\n"
            "- **Error priority**: a false negative (missed malignant case) is far "
            "more costly than a false positive, so we select an operating "
            "*sensitivity* target rather than tune for accuracy alone.\n"
            "- **Success criteria**: report sensitivity, specificity, precision, F1, "
            "balanced accuracy, ROC-AUC, PR-AUC, Brier and the confusion counts; "
            "show the tree; analyse two decision paths.\n"
        )
        selected_feats = '["radius1", "texture1", "concavity1", "radius3"]'
        subgroup_note = (
            "Demographic attributes are absent in this dataset, so a fairness/"
            "subgroup audit **cannot** be performed. We instead run a "
            "**split-stability** experiment (multiple random seeds)."
        )
        subgroup_code = "STABILITY"
    elif ds_key == "diabetes":
        charter_extra = (
            "- **Positive class** = `Positive` (diabetes) remapped to `1`.\n"
            "- **Error priority**: missing a real case (FN) is costlier than a "
            "false alarm in a screening-like setting; we apply a declared "
            "sensitivity target.\n"
            "- **Subgroup lens**: gender is a sensitive attribute — reported only "
            "as a fairness-style audit, never as an eligibility criterion.\n"
            "- **Success criteria**: as above, plus the subgroup analysis by gender.\n"
        )
        selected_feats = '["age", "polyuria", "polydipsia", "sudden_weight_loss"]'
        subgroup_note = (
            "We evaluate sensitivity and specificity by **gender** (sample sizes "
            "and uncertainty reported; do not over-interpret tiny groups)."
        )
        subgroup_code = "GENDER"
    else:
        charter_extra = (
            "- **Positive class** = heart disease present (`num > 0`) remapped to `1`.\n"
            "- **Missingness**: `ca` (4) and `thal` (2) are imputed *inside training "
            "folds only* and treated as categorical codes.\n"
            "- **Subgroup lens**: sex (1=Male / 0=Female) is a sensitive attribute "
            "used only for a fairness-style audit.\n"
            "- **Success criteria**: as above plus a missingness plot and the "
            "subgroup analysis by sex.\n"
        )
        selected_feats = '["age", "thalach", "oldpeak", "chol"]'
        subgroup_note = (
            "We evaluate sensitivity and specificity by **sex** (sample sizes and "
            "uncertainty reported)."
        )
        subgroup_code = "SEX"

    cells = [
        md(f"""# {main}

## {sub}

**MDI3003 — Advanced Predictive Analytics** · Dr. Durgesh Kumar, SCOPE, VIT Vellore

> {DISCLAIMER_NOTE}

This notebook implements the lab's leakage-safe, reproducible protocol on the
**{s.display_name}** dataset (UCI id {s.ucid}, DOI `{s.doi}`). It follows the
20-step workflow defined in the laboratory manual:

1. Problem framing & project charter
2. Library bootstrap & reproducibility stamp
3. Load data & define the positive class
4. Data & leakage audit
5. Lock the stratified train / test split
6. Focused exploratory analysis (training only)
7. Dummy baseline
8. Basic (unconstrained) CART
9. Tuned & cost-complexity-pruned CART
10. Random Forest (+ optional Logistic Regression)
11. Training-only CV comparison of the required trio
12. Out-of-fold threshold selection
13. Single locked test evaluation
14. Confusion matrix, ROC, PR, calibration
15. Tree interpretation & decision paths
16. Feature importance (impurity + permutation)
17. Robustness / subgroup experiment
18. Persist artifacts, results CSVs, model card
19. Evidence-based conclusion
20. Responsible-use statement

`RANDOM_STATE = 42` throughout; all numeric results are deterministic and match
those produced by `python src/meddiag_cli.py list {ds_key}`.

---

## Step 1 — Project charter & prediction question

| Item | Statement |
|---|---|
| **Population** | {s.observation_unit} |
| **Outcome / label** | {s.population_note} |
| **Positive class** | `1` = **{s.positive_label}** (raw values {list(s.positive_raw_values)}) |
| **Negative class** | `0` = **{s.negative_label}** |
| **Prediction time** | The listed predictors must all be available *before* the outcome / decision. |
| **Intended use** | Educational predictive-analytics research prototype only. |
| **Out-of-scope use** | Clinical diagnosis, treatment, triage, screening or any patient management. |
| **Error priority** | False negatives (missed disease) cost more than false positives in this lab. |
| **Success criteria** | Sensitivity, specificity, precision, F1, balanced accuracy, ROC-AUC, PR-AUC, Brier, confusion counts; compact tree; two decision paths. |

{charter_extra}

### Why accuracy alone is inadequate

For disease classification the disease-positive class is often the minority and
carries asymmetric consequences. A majority-class predictor (always predict
"no disease") scores high accuracy while missing nearly every true case.
Accuracy therefore hides minority-class failure, ignores prevalence, and says
nothing about false-negative cost or probability quality. We report class
counts, sensitivity, specificity, PR-AUC, balanced accuracy and the confusion
matrix alongside any headline number.
"""),
        code(BOOTSTRAP),
        md(f"""## Step 2–3 — Load dataset, define positive class, provenance

**Source**: {s.source_note}

**DOI**: `{s.doi}` · **UCI id**: {s.ucid} · **CSV cache**: `data/{s.csv_name}`

Observation unit: *{s.observation_unit}*
"""),
        code(f'''
DS = "{ds_key}"
s = M.spec(DS)
print(s.display_name)
print("DOI:", s.doi, "| UCI id:", s.ucid)
print("Target column:", s.target_col)
print("Positive raw values:", s.positive_raw_values, "->", s.positive_label)
print("Features:", len(s.feature_cols),
      f"({len(s.numeric_cols)} numeric, {len(s.categorical_cols)} categorical)")

raw = M.load_dataset(DS)
print("\\nRaw shape:", raw.shape)
print(raw.head(3))

X, y = M.make_xy(DS, raw)
print("\\nFeature matrix:", X.shape)
print("Class counts (0=neg, 1=pos):")
print(y.value_counts().sort_index())
print(f"Positive class: 1 = {s.positive_label!r}  |  prevalence = {{y.mean():.3f}}")
print("Feature columns:", list(X.columns))
'''),
        md("""## Step 4 — Data & leakage audit

Required observations before any modelling:

* Describe the observation unit and whether multiple rows may belong to one patient.
* List missing-value patterns (absent / unknown / not measured).
* Identify identifiers, duplicates, implausible values, and predictors that might
  reveal the outcome.
* State whether all predictors would be available at the declared prediction time.
"""),
        code(f'''
audit = pd.DataFrame({{
    "dtype": X.dtypes.astype(str),
    "missing": X.isna().sum(),
    "missing_pct": (X.isna().mean() * 100).round(2),
    "unique": X.nunique(),
}})
# numeric range where available
num_cols, cat_cols = M.numeric_categorical(DS, X)
if num_cols:
    audit.loc[num_cols, "min"] = X[num_cols].min()
    audit.loc[num_cols, "max"] = X[num_cols].max()
    audit.loc[num_cols, "median"] = X[num_cols].median()

print(audit.to_string())
print("\\nDuplicate predictor rows:", int(X.duplicated().sum()))
print("Constant columns:", X.columns[X.nunique() <= 1].tolist())
print("Total missing cells:", int(X.isna().sum().sum()))

# Leakage assertions — target / labels must never enter X
assert s.target_col not in X.columns, "target leaked into X"
assert y.name not in X.columns
for forbidden in ("id", "ID", "patient_id", "PatientId"):
    assert forbidden not in X.columns, f"identifier column present: {{forbidden}}"

print("\\n[OK] Leakage assertions passed (no target / id columns in X).")
print("Observation unit:", s.observation_unit)
print("Population note :", s.population_note)
'''),
        md("""## Step 5 — Create and lock the final test set

**Do not touch the test set.** All model comparison, pruning, hyperparameter
tuning, feature selection, class weighting, and threshold selection must be
completed using the training set and its cross-validation folds. The final test
set is evaluated **once** after the analysis plan is locked.
"""),
        code('''
from sklearn.model_selection import train_test_split, StratifiedKFold

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=M.TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
)
cv = StratifiedKFold(n_splits=M.CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

print("Training shape:", X_train.shape)
print("Test shape    :", X_test.shape)
print(f"Training prevalence: {y_train.mean():.3f}")
print(f"Test prevalence    : {y_test.mean():.3f}")
print(f"CV folds: {M.CV_FOLDS}-fold stratified, seed={RANDOM_STATE}")
'''),
        md("""## Step 6 — Focused exploratory analysis (training only)

EDA stays on the **training** split. Every plot answers a modelling or evaluation
question; we do not generate dozens of unfocused figures.
"""),
        code(f'''
P = M.PALETTE
class_names = list(s.class_names)

# --- Class distribution ---
fig, ax = plt.subplots(figsize=(6, 4))
counts = y_train.value_counts().sort_index()
bars = ax.bar([class_names[0], class_names[1]], counts.values,
              color=[P["neg"], P["pos"]], edgecolor=P["baseline"], width=0.55)
for b, c in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
            f"{{c}} ({{100*c/len(y_train):.1f}}%)",
            ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Count")
ax.set_title(f"Training class distribution  (n={{len(y_train)}})")
ax.set_ylim(0, counts.max() * 1.15)
save_show(f"{{DS}}_class_distribution")

# --- Selected feature distributions by class ---
SELECTED = {selected_feats}
eda = X_train.copy()
eda["__y__"] = y_train.values

present = [c for c in SELECTED if c in eda.columns]
num_present = [c for c in present if c in num_cols]
cat_present = [c for c in present if c in cat_cols]

if num_present:
    n = len(num_present)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
    for ax, feat in zip(axes[0], num_present):
        for cls, color, lab in ((0, P["neg"], class_names[0]),
                                (1, P["pos"], class_names[1])):
            sns.kdeplot(data=eda[eda["__y__"] == cls], x=feat, ax=ax,
                        color=color, fill=True, alpha=0.35, label=lab, warn_singular=False)
        ax.set_title(feat)
        ax.legend(fontsize=8)
    fig.suptitle("Class-conditional feature distributions (training)", y=1.02)
    save_show(f"{{DS}}_feature_distributions")

if cat_present:
    n = len(cat_present)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
    for ax, feat in zip(axes[0], cat_present):
        ct = pd.crosstab(eda[feat], eda["__y__"], normalize="index")
        ct.columns = [class_names[c] if c in (0, 1) else c for c in ct.columns]
        ct.plot(kind="bar", ax=ax, color=[P["neg"], P["pos"]],
                edgecolor=P["baseline"])
        ax.set_title(feat)
        ax.set_ylabel("Row-normalised share")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Categorical feature vs class (training)", y=1.02)
    save_show(f"{{DS}}_categorical_vs_class")

# --- Correlation among numeric predictors ---
if len(num_cols) >= 2:
    corr_cols = num_cols[:min(12, len(num_cols))]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(X_train[corr_cols].corr(), annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, ax=ax, square=True,
                cbar_kws={{"shrink": 0.8}})
    ax.set_title("Selected feature correlations (training)")
    save_show(f"{{DS}}_correlation")

# --- Missingness (if any) ---
miss = X_train.isna().mean().sort_values(ascending=False)
miss = miss[miss > 0]
if len(miss):
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(miss))))
    ax.barh(miss.index, miss.values * 100, color=P["warning"],
            edgecolor=P["baseline"])
    ax.set_xlabel("Missing %")
    ax.set_title("Training missingness by feature")
    ax.invert_yaxis()
    save_show(f"{{DS}}_missingness")
else:
    print("No missing values in the training predictors.")

print("""
EDA observations (write these into the report):
  1. Class balance / prevalence informs the metric choice (PR-AUC, sensitivity).
  2. Class-conditional separation on top features motivates tree splits.
  3. Correlation among related measurements (e.g. radius/perimeter/area) warns
     that impurity importance can be shared across collinear predictors.
  4. Missingness, if present, is handled only inside the training-fitted pipeline.
""")
'''),
        md("""## Steps 7–12 — Full leakage-safe training workflow

The shared engine `M.train_dataset` executes the lab's non-negotiable protocol:

1. Stratified 80/20 split (already locked above — the engine re-splits with the
   same seed so results match).
2. Two-stage **pre-prune then cost-complexity prune** CART (training-only CV).
3. Compact Random Forest grid (training-only CV).
4. Dummy baseline + Logistic Regression optional benchmark.
5. Identical stratified 5-fold CV comparison table.
6. Out-of-fold threshold selection against a *declared* sensitivity target.
7. Fit on all training data; **one** final test evaluation.

This guarantees that the notebook, `meddiag_cli.py`, and the GUI never disagree.
"""),
        code('''
print("=" * 60)
print("Running full training workflow (this may take several minutes)...")
print("=" * 60)
state = M.train_dataset(DS, df=raw, verbose=True)

print("\\n" + "=" * 60)
print(f"Best model by CV ROC-AUC : {state['best_name']}")
print(f"Selected threshold       : {state['threshold']:.3f} "
      f"(satisfied sens >= {state['threshold_target_satisfied']:.2f})")
print(f"Test sensitivity         : {state['test_metrics']['sensitivity']:.4f}")
print(f"Test specificity         : {state['test_metrics']['specificity']:.4f}")
print(f"Test ROC-AUC             : {state['test_metrics']['roc_auc']:.4f}")
print(f"Test FN / FP             : {state['test_metrics']['FN']} / "
      f"{state['test_metrics']['FP']}")
print("=" * 60)
'''),
        md("""## Step 7 — Dummy baseline & Step 8 — Basic CART complexity

A large train–validation gap on the unconstrained tree is evidence of overfitting.
"""),
        code('''
cv_table = state["cv_table"]
print("Cross-validation comparison (training-only, sorted by ROC-AUC):")
display_cols = [c for c in cv_table.columns if c == "Model" or c.endswith("_mean")
                or c.endswith("_std")]
print(cv_table[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\\nTree complexity (fitted on training data):")
for label, info in state["complexity"].items():
    print(f"  {label:28} depth={info['depth']:2d}  leaves={info['leaves']:3d}  "
          f"nodes={info['nodes']:3d}  criterion={info['criterion']}  "
          f"ccp_alpha={info['ccp_alpha']:.5f}")
'''),
        md("""## Steps 9–11 — Pruning path, model comparison chart
"""),
        code('''
P = M.PALETTE

# CV comparison bar chart
fig, ax = plt.subplots(figsize=(9, 5))
cvt = cv_table.copy()
ypos = np.arange(len(cvt))
ax.barh(ypos, cvt["roc_auc_mean"], xerr=cvt["roc_auc_std"],
        color=[P["cat"][i % 8] for i in range(len(cvt))],
        edgecolor=P["baseline"], height=0.6, capsize=3)
ax.set_yticks(ypos)
ax.set_yticklabels(cvt["Model"])
ax.set_xlabel("CV ROC-AUC (mean ± std)")
ax.set_title("Model comparison — 5-fold stratified CV (training only)")
ax.invert_yaxis()
save_show(f"{DS}_cv_comparison")

# Cost-complexity pruning curve (mean CV score vs alpha)
try:
    gs2 = state["pruned_gs2"]
    res = pd.DataFrame(gs2.cv_results_)
    # average over class_weight for each alpha
    if "param_model__ccp_alpha" in res.columns:
        grp = res.groupby("param_model__ccp_alpha").agg(
            mean_auc=("mean_test_roc_auc", "mean"),
            std_auc=("std_test_roc_auc", "mean"),
        ).reset_index()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.errorbar(grp["param_model__ccp_alpha"], grp["mean_auc"],
                    yerr=grp["std_auc"], fmt="-o", color=P["pos"],
                    ecolor=P["ink_muted"], capsize=2, ms=4)
        chosen = state["pruned_params"].get("model__ccp_alpha", 0)
        ax.axvline(float(chosen), ls="--", color=P["ink_muted"],
                   label=f"selected α={float(chosen):.5f}")
        ax.set_xlabel("ccp_alpha")
        ax.set_ylabel("CV ROC-AUC")
        ax.set_title("Cost-complexity pruning path (training-only CV)")
        ax.legend()
        if grp["param_model__ccp_alpha"].max() > 0:
            ax.set_xscale("symlog", linthresh=1e-4)
        save_show(f"{DS}_pruning_path")
except Exception as e:
    print("Pruning path plot skipped:", e)

print("Selected pruned CART params:")
for k, v in state["pruned_params"].items():
    print(f"  {k}: {v}")
print("\\nSelected Random Forest params:")
for k, v in state.get("rf_params", {}).items():
    print(f"  {k}: {v}")
'''),
        md("""## Step 12–13 — Threshold selection (out-of-fold) & locked test evaluation

The threshold is chosen on **out-of-fold training probabilities** to meet a
declared sensitivity target while maximising specificity. The test set is scored
**once** with the locked model and threshold.
"""),
        code('''
tdf = state["threshold_df"]
threshold = state["threshold"]
P = M.PALETTE

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(tdf["threshold"], tdf["sensitivity"], color=P["pos"], lw=2, label="Sensitivity")
ax.plot(tdf["threshold"], tdf["specificity"], color=P["neg"], lw=2, label="Specificity")
ax.plot(tdf["threshold"], tdf["precision"], color=P["cat"][2], lw=1.5,
        alpha=0.85, label="Precision")
ax.axvline(threshold, ls="--", color=P["ink_muted"],
           label=f"selected thr={threshold:.3f}")
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Rate")
ax.set_title("Threshold sweep (out-of-fold training predictions)")
ax.legend(loc="best")
ax.set_xlim(0.05, 0.95)
save_show(f"{DS}_threshold_sweep")

print("Final test metrics (selected threshold):")
for k, v in state["test_metrics"].items():
    if isinstance(v, float):
        print(f"  {k:20}: {v:.4f}")
    else:
        print(f"  {k:20}: {v}")

print("\\nAblation at default threshold 0.50:")
for k in ("sensitivity", "specificity", "FN", "FP", "precision_ppv", "f1"):
    v = state["default_threshold_metrics"][k]
    print(f"  {k:20}: {v:.4f}" if isinstance(v, float) else f"  {k:20}: {v}")

print("\\nAll models — locked test set (same threshold):")
test_df = pd.DataFrame(state["all_test_rows"])
cols = ["Model", "roc_auc", "pr_auc", "sensitivity", "specificity",
        "f1", "FN", "FP", "accuracy", "brier"]
cols = [c for c in cols if c in test_df.columns]
print(test_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
'''),
        md("""## Step 14 — Confusion matrix, ROC, PR, calibration
"""),
        code('''
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score,
                             brier_score_loss)
from sklearn.calibration import calibration_curve

y_te = state["y_test"].values
prob = state["test_prob"]
pred = state["test_pred"]
threshold = state["threshold"]
class_names = list(state["class_names"])
P = M.PALETTE

# Confusion matrix
cm = confusion_matrix(y_te, pred, labels=[0, 1])
fig, ax = plt.subplots(figsize=(4.8, 4.2))
im = ax.imshow(cm, cmap="medposneg", aspect="auto")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Test confusion matrix (thr={threshold:.2f})")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color=P["ink_primary"], fontsize=16, fontweight="bold")
save_show(f"{DS}_confusion_matrix")

# ROC
fpr, tpr, _ = roc_curve(y_te, prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color=P["pos"], lw=2, label=f"AUC = {auc(fpr, tpr):.3f}")
ax.plot([0, 1], [0, 1], "--", color=P["ink_muted"], label="Chance")
ax.set_xlabel("False positive rate (1 − specificity)")
ax.set_ylabel("True positive rate (sensitivity)")
ax.set_title("ROC curve — final test set")
ax.legend(loc="lower right")
save_show(f"{DS}_roc")

# PR
prec, rec, _ = precision_recall_curve(y_te, prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec, prec, color=P["cat"][2], lw=2,
        label=f"PR-AUC = {average_precision_score(y_te, prob):.3f}")
ax.axhline(y_te.mean(), ls="--", color=P["ink_muted"],
           label=f"Prevalence {y_te.mean():.2f}")
ax.set_xlabel("Recall (sensitivity)"); ax.set_ylabel("Precision (PPV)")
ax.set_title("Precision–recall curve — final test set")
ax.legend(loc="upper right")
save_show(f"{DS}_pr")

# Calibration
fig, ax = plt.subplots(figsize=(6, 5))
try:
    frac_pos, mean_pred = calibration_curve(y_te, prob, n_bins=8, strategy="quantile")
    ax.plot(mean_pred, frac_pos, "o-", color=P["pos"],
            label=f"Brier = {brier_score_loss(y_te, prob):.3f}")
except Exception:
    ax.plot([0, 1], [0, 1], color=P["pos"], label="(insufficient bins)")
ax.plot([0, 1], [0, 1], "--", color=P["ink_muted"], label="Perfect calibration")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration plot — final test set")
ax.legend(loc="upper left")
save_show(f"{DS}_calibration")

print(f"""
Error-cost interpretation (instructional costs CFN={M.COST_FN}, CFP={M.COST_FP}):
  FN = {state['test_metrics']['FN']}  (missed disease cases)
  FP = {state['test_metrics']['FP']}  (false alarms)
  Expected cost = {state['test_metrics']['expected_cost']:.0f}
A false negative is treated as {M.COST_FN/M.COST_FP:.0f}× costlier than a false positive
in this educational design — not a clinical standard.
""")
'''),
        md("""## Step 15–16 — Tree visualisation, decision paths, rules

Show only the upper levels of the tree. Dataset-derived thresholds are **not**
clinically validated rules.
"""),
        code('''
from sklearn.tree import plot_tree, export_text

best_pipe = state["best_pipe"]
inner = best_pipe.named_steps.get("model") if hasattr(best_pipe, "named_steps") else best_pipe
is_tree = "sklearn.tree" in type(inner).__module__

# Prefer the tuned & pruned CART for interpretability even if RF won CV
tree_pipe = state["models"].get("Tuned and pruned CART") or state["models"].get("Basic CART")
tree_inner = None
if tree_pipe is not None:
    tree_inner = tree_pipe.named_steps.get("model")

def _feature_names(pipe):
    try:
        pre = pipe.named_steps["pre"]
        names = []
        for _, sub, cols, *_ in pre.transformers_:
            if hasattr(sub, "get_feature_names_out"):
                names.extend([str(x) for x in sub.get_feature_names_out()])
            else:
                names.extend(list(cols) if isinstance(cols, (list, tuple)) else [cols])
        return names
    except Exception:
        return list(state["feature_cols"])

if tree_inner is not None and hasattr(tree_inner, "tree_"):
    fnames = _feature_names(tree_pipe)
    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(tree_inner, feature_names=fnames, class_names=list(state["class_names"]),
              filled=True, rounded=True, proportion=True, precision=2,
              max_depth=4, ax=ax, fontsize=8)
    ax.set_title(f"Tuned & pruned CART — top 4 levels ({DS})\\n"
                 "(dataset-derived thresholds — NOT clinically validated)")
    save_show(f"{DS}_tree")

    print("Extracted rules (truncated):")
    rules = export_text(tree_inner, feature_names=fnames, max_depth=6)
    print(rules[:4000])
    if len(rules) > 4000:
        print("... [truncated]")
else:
    print("No single-tree model available for visualisation.")

# Trace 2 correct + 2 incorrect test predictions through the pruned tree
if tree_inner is not None and tree_pipe is not None:
    print("\\n--- Decision-path audit on test cases ---")
    Xte = state["X_test"]
    yte = state["y_test"].values
    p_tree = tree_pipe.predict_proba(Xte)[:, 1]
    pred_tree = (p_tree >= threshold).astype(int)
    correct = np.where(pred_tree == yte)[0]
    wrong = np.where(pred_tree != yte)[0]
    sample_idx = list(correct[:2]) + list(wrong[:2])
    Xt_all = tree_pipe.named_steps["pre"].transform(Xte)
    fnames = _feature_names(tree_pipe)
    for idx in sample_idx:
        leaf = int(tree_inner.apply(Xt_all[idx:idx+1])[0])
        leaf_p = tree_inner.predict_proba(Xt_all[idx:idx+1])[0, 1]
        tag = "CORRECT" if pred_tree[idx] == yte[idx] else "ERROR"
        print(f"\\n  [{tag}] test row {idx}: actual={yte[idx]}  "
              f"P={p_tree[idx]:.3f}  pred={pred_tree[idx]}  leaf={leaf}  "
              f"leaf_P={leaf_p:.3f}")
        print("  (Path is descriptive of the model, not a causal/clinical explanation.)")
'''),
        md("""## Step 17 — Feature importance (impurity vs permutation)

Impurity importance can favour predictors with many split candidates and can
share importance across correlated features. Permutation importance measures
predictive dependence under a chosen metric — **not** causal effect.
"""),
        code('''
from sklearn.inspection import permutation_importance

pipe_for_imp = state["best_pipe"]
inner_imp = pipe_for_imp.named_steps.get("model")
fnames = _feature_names(pipe_for_imp)

# Impurity importance (if available)
if hasattr(inner_imp, "feature_importances_"):
    fi = pd.Series(inner_imp.feature_importances_, index=fnames).sort_values(ascending=False)
    print("Top impurity importances:")
    print(fi.head(15).to_string())
    top = fi.head(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index[::-1], top.values[::-1], color=P["cat"][0],
            edgecolor=P["baseline"])
    ax.set_xlabel("Impurity importance")
    ax.set_title("Feature importance (impurity) — non-causal")
    save_show(f"{DS}_importance_impurity")
else:
    fi = None
    print("Best model has no impurity importances; skipping.")

# Permutation importance on the test set (scoring = ROC-AUC)
print("\\nComputing permutation importance (n_repeats=20)...")
perm = permutation_importance(
    pipe_for_imp, state["X_test"], state["y_test"],
    scoring="roc_auc", n_repeats=20, random_state=RANDOM_STATE, n_jobs=2,
)
perm_df = pd.DataFrame({
    "feature": list(state["X_test"].columns),
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)
print(perm_df.head(15).to_string(index=False))

top_p = perm_df.head(12)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(top_p["feature"][::-1], top_p["importance_mean"][::-1],
        xerr=top_p["importance_std"][::-1], color=P["cat"][2],
        edgecolor=P["baseline"], capsize=2)
ax.set_xlabel("Permutation importance (ROC-AUC drop)")
ax.set_title("Feature importance (permutation) — non-causal")
save_show(f"{DS}_importance_permutation")

print("""
Interpretation caution:
  * These scores measure predictive dependence under the chosen metric.
  * They do NOT establish that a feature causes the disease.
  * Correlated measurements (e.g. radius / perimeter / area) share importance.
""")
'''),
        md(f"""## Step 18 — Robustness / subgroup experiment

{subgroup_note}
"""),
        code(f'''
# Subgroup / stability experiment
subgroup_mode = "{subgroup_code}"  # STABILITY | GENDER | SEX

if subgroup_mode == "STABILITY":
    print("Split-stability experiment: retrain selection under multiple seeds.")
    print("(Using a lighter protocol: Basic CART CV ROC-AUC across seeds.)")
    from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
    rows = []
    for seed in [0, 1, 2, 7, 42]:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=M.TEST_SIZE, stratify=y, random_state=seed)
        pipe = M.attach(M.build_preprocessor(DS, Xtr, scale=False), M.basic_cart(seed))
        cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        scores = cross_val_score(pipe, Xtr, ytr, cv=cv_s, scoring="roc_auc", n_jobs=2)
        pipe.fit(Xtr, ytr)
        inner = pipe.named_steps["model"]
        te_prob = pipe.predict_proba(Xte)[:, 1]
        from sklearn.metrics import roc_auc_score
        rows.append({{
            "seed": seed,
            "cv_roc_auc_mean": scores.mean(),
            "cv_roc_auc_std": scores.std(),
            "test_roc_auc": roc_auc_score(yte, te_prob),
            "depth": inner.get_depth(),
            "leaves": inner.get_n_leaves(),
        }})
    stab = pd.DataFrame(rows)
    print(stab.to_string(index=False, float_format=lambda x: f"{{x:.4f}}"))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(stab["seed"].astype(str), stab["cv_roc_auc_mean"],
                yerr=stab["cv_roc_auc_std"], fmt="o-", color=P["pos"], capsize=3,
                label="CV ROC-AUC")
    ax.plot(stab["seed"].astype(str), stab["test_roc_auc"], "s--",
            color=P["neg"], label="Test ROC-AUC")
    ax.set_xlabel("Random seed"); ax.set_ylabel("ROC-AUC")
    ax.set_title("Split stability — Basic CART across seeds")
    ax.legend()
    save_show(f"{{DS}}_stability")
    print("\\nObservation: metrics and tree size vary with the split; "
          "a single hold-out is not a clinical validation.")

elif subgroup_mode in ("GENDER", "SEX"):
    key = "gender" if subgroup_mode == "GENDER" else "sex"
    print(f"Subgroup performance by '{key}' on the locked test set.")
    Xte = state["X_test"].reset_index(drop=True)
    yte = np.asarray(state["y_test"]).astype(int)
    prob = np.asarray(state["test_prob"], dtype=float)
    thr = state["threshold"]
    if key not in Xte.columns:
        print(f"Column {key} not in features; skipping.")
    else:
        rows = []
        for gval, subdf in Xte.groupby(key):
            idx = subdf.index.to_numpy()
            if len(idx) < 5:
                continue
            m = M.evaluate_binary(yte[idx], prob[idx], thr)
            rows.append({{
                "group": str(gval),
                "n": int(len(idx)),
                "positive_pct": float(yte[idx].mean()),
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "precision": m["precision_ppv"],
                "roc_auc": m["roc_auc"],
                "FN": m["FN"], "FP": m["FP"],
            }})
        sub = pd.DataFrame(rows)
        print(sub.to_string(index=False, float_format=lambda x: f"{{x:.4f}}"))
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(sub))
        w = 0.35
        ax.bar(x - w/2, sub["sensitivity"], w, label="Sensitivity", color=P["pos"])
        ax.bar(x + w/2, sub["specificity"], w, label="Specificity", color=P["neg"])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{{g}}\\n(n={{n}})" for g, n in zip(sub["group"], sub["n"])])
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Subgroup metrics by {key} (test set, thr={{thr:.2f}})")
        ax.legend()
        save_show(f"{{DS}}_subgroup")
        print("\\nCaveat: small n -> wide uncertainty; do not treat group differences "
              "as definitive disparity evidence. Sensitive attributes are audit lenses, "
              "not eligibility criteria.")
'''),
        md("""## Step 19 — Persist artifacts, results CSVs, model card metadata
"""),
        code('''
paths = M.save_artifacts(DS, state, tag=DS)
print("Artifacts written under:", M.ART_DIR)
print("Tag:", paths.get("tag"))
print("Metadata:", paths.get("metadata"))

# Also emit a concise model-card markdown snippet for this run
meta = M.load_metadata(DS)
card_path = os.path.join(M.PROJECT_ROOT, "reports", f"model_card_{DS}.md")
os.makedirs(os.path.dirname(card_path), exist_ok=True)
tm = meta["test_metrics"]
card = f"""# Model Card — {meta['dataset_display_name']}

> EDUCATIONAL-USE BOUNDARY: research/teaching prototype only. NOT a clinical
> diagnostic system. Do not use for patient care.

## Intended use
- **Intended**: educational predictive-analytics laboratory (MDI3003 Lab 02).
- **Out of scope**: diagnosis, treatment, triage, screening, patient management.

## Population & data
- **Source**: {meta['source_note']}
- **DOI**: {meta['doi']}
- **Observation unit**: {meta['observation_unit']}
- **Rows**: {meta['n_rows']} (train {meta['n_train']} / test {meta['n_test']})
- **Positive class**: 1 = {meta['positive_class']}
- **Prevalence**: train {meta['train_prevalence']:.3f}, test {meta['test_prevalence']:.3f}

## Model
- **Selected model**: {meta['best_model']}
- **Threshold**: {meta['threshold']:.3f} (target sens {meta['threshold_target']:.2f},
  satisfied {meta['threshold_target_satisfied']:.2f})
- **Split**: {meta['split']}
- **CV**: {meta['cv_folds']}-fold stratified, seed {meta['seed']}

## Test-set performance (single locked evaluation)
| Metric | Value |
|---|---|
| Sensitivity | {tm['sensitivity']:.4f} |
| Specificity | {tm['specificity']:.4f} |
| Precision (PPV) | {tm['precision_ppv']:.4f} |
| F1 | {tm['f1']:.4f} |
| ROC-AUC | {tm['roc_auc']:.4f} |
| PR-AUC | {tm['pr_auc']:.4f} |
| Brier | {tm['brier']:.4f} |
| TN / FP / FN / TP | {tm['TN']} / {tm['FP']} / {tm['FN']} / {tm['TP']} |

## Caveats
- {meta['population_note']}
- Dataset-derived thresholds are not clinical rules.
- No external validation; single public benchmark split.
- Subgroup fairness may be unassessable when demographics are absent.

## Software
{meta['software']}

## Monitoring (if ever evaluated outside the lab)
- Track prevalence shift, calibration drift, FN/FP rates by site.
- Human oversight required; rollback plan mandatory.
- Never deploy without clinical validation and governance review.
"""
with open(card_path, "w", encoding="utf-8") as f:
    f.write(card)
print(f"Model card -> {card_path}")
print(f"Trained at: {meta.get('trained_at')}")
'''),
        md("""## Step 20 — Evidence-based conclusion

Answer the five required questions:

1. **How did basic CART, tuned & pruned CART, and Random Forest compare with the
   dummy baseline?** See the CV table and test-results CSV.
2. **How much did pruning change generalization, tree size, and interpretability?**
   Compare complexity dict (depth / leaves / nodes) and CV scores.
3. **What were the FN and FP counts and consequences?** See the confusion matrix
   and expected-cost line.
4. **Which limitations prevent clinical deployment?** Single-institution / survey
   data, no external validation, possible label noise, absent demographics
   (breast), small n (heart), questionnaire setting (diabetes).
5. **What would be required before any real-world use?** Prospective multi-site
   validation, clinical review, fairness audit, calibration monitoring, human
   oversight, regulatory pathway, and a clear intended-use specification.

### Responsible-use statement

This notebook produces an **educational research prototype**. It must not be
presented as medical advice, a diagnostic device, or evidence of clinical safety
or effectiveness. All thresholds and rules are dataset-derived under a fixed
seed and a single public benchmark.

### Assistance disclosure

Automated coding assistance may have been used to scaffold notebooks and
engineering glue; every metric, split, and figure was verified by executing the
shared engine (`meddiag_common`) end-to-end with `SEED=42`.
"""),
        code('''
print("=" * 60)
print("CONCLUSION SNAPSHOT")
print("=" * 60)
print(f"Dataset     : {s.display_name}")
print(f"Best model  : {state['best_name']}")
print(f"Threshold   : {state['threshold']:.3f}")
print(f"Test sens   : {state['test_metrics']['sensitivity']:.4f}")
print(f"Test spec   : {state['test_metrics']['specificity']:.4f}")
print(f"Test AUC    : {state['test_metrics']['roc_auc']:.4f}")
print(f"FN / FP     : {state['test_metrics']['FN']} / {state['test_metrics']['FP']}")
print(f"Complexity  : {state['complexity']}")
print()
print(DISCLAIMER)
print()
print("Reproduce via CLI:")
print(f"  python src/meddiag_cli.py list {DS}")
print(f"  python src/meddiag_cli.py predict {DS} examples/{DS}_sample.json")
print(f"  python src/meddiag_cli.py evaluate {DS} examples/{DS}_sample.csv")
'''),
    ]
    return cells


def build_notebook(ds_key: str) -> str:
    cells = header_cells(ds_key)
    nb = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    name_map = {
        "breast": "breast_cancer.ipynb",
        "diabetes": "diabetes_risk.ipynb",
        "heart": "heart_disease.ipynb",
    }
    path = os.path.join(NB_DIR, name_map[ds_key])
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    return path


def main():
    print("Building Lab 02 notebooks...")
    for ds in ("breast", "diabetes", "heart"):
        path = build_notebook(ds)
        print(f"  [OK] {path}")
    print("Done.")


if __name__ == "__main__":
    main()
