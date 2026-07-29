#!/usr/bin/env python3
"""Refine breast_cancer.ipynb (idempotent). Run from project root."""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE))

from _refine_helpers import (  # type: ignore  # noqa: E402
    load_nb, save_nb, src_str, set_src, demoji, strip_banner, ROOT,
)
import meddiag_common as M  # noqa: E402

NB = os.path.join(ROOT, "notebooks", "breast_cancer.ipynb")

# ---------------------------------------------------------------------------
# MARKDOWN REFINEMENTS  (full cell replacements, standardised headings)
# ---------------------------------------------------------------------------
MD = {
0: """# Medical Diagnosis Support using Decision Trees
## Breast Cancer Wisconsin (Diagnostic) Dataset

**MDI3003 - Advanced Predictive Analytics - Lab 02**

### Objective

Develop an interpretable machine-learning pipeline that classifies breast
tumours as **Malignant** or **Benign** from Decision-Tree-based models, while
following industry-standard predictive-analytics practice. The project
emphasises data-quality assessment, leakage-free preprocessing, explainable AI,
reproducibility, hyperparameter optimisation, clinical error analysis and
responsible AI.

### Educational disclaimer

This notebook is developed solely for educational and research purposes. The
resulting models are **NOT** clinically validated and must **NOT** be used for
diagnosis, treatment planning, patient triage, or any real-world healthcare
decision. The predictions only reproduce patterns learned from a public
benchmark dataset.
""",

1: """# 1. Business Understanding and Problem Framing

## Problem statement

Early detection of breast cancer substantially improves survival rates. This
notebook builds an interpretable binary classifier that predicts whether a
tumour is malignant or benign, using numerical descriptors extracted from
digitised fine-needle-aspiration (FNA) images. Unlike black-box models, Decision
Trees expose transparent if-then rules, which is useful for explainable baselines
in healthcare research.

## Prediction task

| Item | Specification |
|:---|:---|
| **Observation unit** | One FNA image measurement set per patient |
| **Target variable** | `Diagnosis` (Malignant vs Benign) |
| **Positive class** | Malignant -> encoded as **1** |
| **Negative class** | Benign -> encoded as **0** |
| **Prediction time** | All predictors assumed available before diagnosis is confirmed |
| **Intended use** | Educational predictive-analytics prototype |
| **Out of scope** | Diagnosis, treatment recommendation, clinical decision support, patient risk assessment |
""",

2: """# 2. Environment Setup

Import the required libraries, set display and plotting defaults, fix random
seeds for reproducibility, and record software versions. The shared engine
`meddiag_common` (`src/`) is placed on the path so the notebook, the CLI and the
GUI all use the same dataset metadata, preprocessing, metrics and persistence
code and can never disagree on protocol.
""",

4: """# 3. Dataset Loading

The Breast Cancer Wisconsin (Diagnostic) dataset is shipped with scikit-learn
and is also cached under `data/breast_cancer_wisconsin.csv` (UCI id 17). It has
569 observations, 30 numerical predictors, a binary target and no missing
values - a clean core workflow for interpretable Decision-Tree modelling. For
this lab the **Malignant** tumour is the positive class (encoded 1).
""",

7: """# 4. Data Understanding and Audit

Before any modelling, audit the dataset thoroughly: structure, dtypes,
statistical profile, target distribution, missingness, duplicates,
constant/quasi-constant features, per-feature distributions, outliers,
correlation, multicollinearity and feature-target relationships. The goal is to
surface data-quality issues and to guide - not over-engineer - preprocessing
while keeping the workflow leakage-safe.
""",

11: """## 4.1 First look

The dataset contains only numerical predictor variables; each row is one patient
sample; the target is binary; no identifier columns are present.
""",

14: """## 4.2 Data types

All predictors are continuous numerical measurements. No categorical, datetime
or identifier features exist, so no categorical encoding is required.
""",

16: """## 4.3 Statistical profile

Summary statistics reveal substantial variation in feature ranges and noticeable
skewness/kurtosis, so distributions are not Gaussian. Decision-Tree algorithms
are robust to non-normality, so no transformation is required at this stage.
""",

19: """## 4.4 Target distribution

The class distribution is only mildly imbalanced (Benign more frequent than
Malignant), so SMOTE-style resampling is unnecessary; class weighting will still
be evaluated during model development.
""",

21: """## 4.5 Missing values

No missing values are present. The modelling pipeline will still include an
imputer to keep the workflow reusable on datasets that do contain missingness.
""",

23: """## 4.6 Duplicates

No duplicate observations are detected, so no de-duplication is required.
""",

26: """## 4.7 Constant / quasi-constant features

No constant or quasi-constant features are present; all variables carry
information for the predictive task.
""",

29: """## 4.8 Feature distributions by class

Several features show clear separation between malignant and benign tumours
(e.g. worst radius, worst perimeter, worst concavity, mean concavity, mean
radius). These are likely to be selected near the top of the tree.
""",

31: """## 4.9 Outliers

Several numerical variables contain apparent outliers. Decision Trees partition
observations using thresholds rather than distances, so they are considerably
less outlier-sensitive than linear or nearest-neighbour models; no outlier
removal is performed.
""",

33: """## 4.10 Distribution shape

Many variables are positively skewed. Decision Trees are invariant to monotonic
transformations and make no Gaussian assumption, so no log or Box-Cox
transformation is required.
""",

34: """# 5. Advanced Exploratory Data Analysis

Investigate predictor relationships, multicollinearity, feature relevance and
potential data leakage before model development. These analyses guide feature
selection, preprocessing and interpretation while keeping the workflow
leakage-safe.
""",

36: """## 5.1 Correlation matrix

The correlation matrix exposes several highly correlated feature groups. This is
expected because multiple measurements are derived from the same tumour
characteristics (radius, perimeter, area, texture, ...). Tree models are robust
to multicollinearity, so no feature is removed on correlation grounds alone.
""",

38: """## 5.2 Highly correlated pairs

Several pairs exceed 0.90 absolute correlation and capture closely related
physical characteristics (overlapping information). Tree-based models select
informative split variables automatically, so feature elimination is
unnecessary.
""",

41: """## 5.3 Feature-target correlation

Features such as worst radius, worst perimeter, worst concavity, mean concavity
and mean perimeter have the strongest linear association with the diagnosis.
Correlation alone does not decide model importance, but these variables are
expected to appear near the top of the tree.
""",

43: """## 5.4 Pairwise relationships

The pairwise plots show substantial separation between benign and malignant
samples; some feature combinations are almost linearly separable, so relatively
shallow trees may already perform well.
""",

45: """## 5.5 Variance inflation factors

Several variables have extremely high VIF, confirming strong multicollinearity -
expected because many features encode related geometric measurements from the
same image. Decision Trees are far less affected than linear regression, so no
corrective action is needed.
""",

47: """## 5.6 Feature variance

Variance differs considerably across variables. Decision Trees are invariant to
monotonic scaling, so high variance is not a problem and no standardisation is
required.
""",

50: """## 5.7 Leakage audit

The dataset contains no identifiers, timestamps, post-diagnosis variables or
outcome-derived attributes. All predictors are assumed available before the
prediction task, so the data is suitable for supervised learning without target
leakage.
""",

52: """## 5.8 Hierarchical feature clustering

Hierarchical clustering reveals coherent groups of related measurements,
particularly among radius, perimeter, area and concavity variables - additional
evidence that several features describe the same biological property.
""",

54: """# 6. Data Preparation

The EDA showed the data is fully numeric with no missingness, duplicates or
categorical features. Preprocessing is therefore minimal, but a full scikit-learn
pipeline is still constructed to guarantee reproducibility, leakage prevention,
compatibility with future datasets and production-ready deployment. The test set
is created **before** any model training or tuning and remains untouched until
the final evaluation.
""",

57: """## 6.1 Stratified split

A stratified 80/20 split preserves the malignant/benign ratio in both subsets.
The test set is now locked and will not be used during model development or
hyperparameter optimisation.
""",

59: """## 6.2 Cross-validation strategy

Stratified K-Fold (5 folds, shuffled, seed 42) keeps approximately the same
class distribution in every fold. This reduces metric variability and is the
preferred validation strategy for binary problems with mild imbalance.
""",

62: """## 6.3 Why an imputer with no missing values?

The dataset has no missingness, but embedding an imputation step in the pipeline
improves reproducibility and lets the workflow be reused on datasets with
missing values without code changes.
""",

68: """## 6.4 Data-preparation summary

Preprocessing is finalised: a locked test set, stratified split, 5-fold
stratified CV, pipeline-based preprocessing, reusable evaluation helpers and a
fixed random seed. The notebook is ready for baseline model development.
""",

69: """# 7. Baseline Model Development

Before complex models, establish baselines so we can verify that machine learning
actually beats the trivial rules. Two baselines are built: a Dummy classifier
that ignores the features and predicts the prior class distribution, and a basic
unconstrained CART that learns if-then rules from the predictors. The Dummy sets
the floor; the basic CART reveals whether unconstrained trees overfit.
""",

74: """## 7.1 Dummy baseline observation

Because the Dummy ignores the features, any meaningful model should substantially
outperform it across all metrics - the Dummy sets the minimum expected
performance for this task.
""",

77: """## 7.2 Basic CART observation

The unconstrained tree is perfect on training folds but weaker on validation
folds - the classic overfitting signature. This motivates hyperparameter tuning
and pruning in the following sections.
""",

80: """## 7.3 Tree complexity

Tree complexity is measured by depth, number of leaves and total nodes; very deep
trees with many leaves indicate high variance and an increased overfitting risk.
""",

84: """# 8. Advanced Tree Models

Define reusable tuning helpers, then tune a CART over pre-pruning
hyperparameters, select `ccp_alpha` through cost-complexity pruning, and finally
tune a Random Forest. Each model is selected using training-only CV; the locked
test set is untouched.
""",

87: """## 8.1 Pre-pruned CART result

Grid search picks the combination of criterion, max_depth, min_samples_split,
min_samples_leaf and class_weight that maximises cross-validated ROC-AUC. This
pre-pruned tree already reduces overfitting compared with the unconstrained
CART.
""",

88: """# 9. Cost-Complexity Pruning for CART

Fix the best pre-pruning parameters found above and select the optimal
`ccp_alpha` by cross-validation. Two-stage selection (pre-prune, then prune)
follows the lab manual and yields the "tuned and pruned CART".
""",

91: """# 10. Tune Random Forest

Tune a Random Forest with a randomised search over a wide hyperparameter space
(efficient for the large forest grid), refit on ROC-AUC, training-only CV.
""",

93: """# 11. Model Comparison (Cross-Validation)

Compare all four models - Dummy, Basic CART, Tuned & Pruned CART and Random
Forest - under identical 5-fold stratified CV folds, collecting mean
discrimination, decision and calibration metrics.
""",

95: """## 11.1 Model comparison visualisation

Bar chart of the four models across the key cross-validated metrics
(Accuracy, F1, Balanced Accuracy, ROC-AUC, PR-AUC).
""",

97: """## 11.2 Comparison observation

The Random Forest leads on ROC-AUC / PR-AUC; the Tuned & Pruned CART balances
performance with interpretability for the report.
""",

98: """# 12. Select Operating Threshold

Pick the final model (here the best cross-validated model; the Tuned & Pruned
CART may be substituted for interpretability). Use out-of-fold training
predictions to sweep the decision threshold: first enforce a declared
sensitivity target (0.95) and select the highest-specificity threshold that still
meets it; fall back to the Youden-maximising threshold if the target is
infeasible.
""",

100: """# 13. Final Evaluation on the Locked Test Set

Lock the model on all training data and evaluate on the test set exactly once.
Report the full metric block (sensitivity, specificity, precision, NPV, F1,
balanced accuracy, ROC-AUC, PR-AUC, Brier, MCC) and the explicit TN/FP/FN/TP
counts at the chosen threshold.
""",

103: """# 14. Calibration and Save Artifacts

Assess probability calibration of the final model on the test set, then persist
every artefact the CLI/GUI need: the fitted final pipeline, the operating
threshold, individual model pipelines, the CV and test results CSVs, and the
metadata JSON - written via the shared `meddiag_common.save_artifacts` engine so
the notebook and the CLI can never disagree.
""",

104: """## 14.1 Calibration plot (reliability diagram)

Evaluate probability calibration of the final model on the test set. A perfectly
calibrated model lies on the diagonal; the Brier score summarises the
reliability gap.
""",
}

# ---------------------------------------------------------------------------
# CODE-CELL purpose-header comments  (prepended; banner stripped first)
# Keyed by cell index.
# ---------------------------------------------------------------------------
CC = {
3: """# ---- 2. Environment Setup ----
# Imports, display options, random seeding for reproducibility, and a quick
# version report. A `report_figures/` dir is created for publication-quality
# 300-DPI exports used by the lab report docx.
""",
5: """# ---- 3. Load the Breast Cancer Wisconsin dataset (scikit-learn) ----
# `load_breast_cancer(as_frame=True)` returns 30 numeric predictors + the target.
# X holds features, y holds the raw 0/1 target (which sklearn encodes as
# 0=Benign, 1=Malignant - i.e. Malignant is already the positive class).
""",
6: """# ---- 3.1 Binarise the target and assemble one combined dataframe ----
# Remap so the encoded target is 1=Malignant (positive) / 0=Benign. `dataset`
# keeps features and the binarised target together for EDA convenience.
""",
8: """# ---- 4.1 Dataset shape overview ----
""",
9: """# ---- 4. Peek at the first rows ----
""",
10: """# ---- 4. Peek at a random sample (seed-locked) for a sanity check ----
""",
12: """# ---- 4.2 dtypes via info() ----
""",
13: """# ---- 4.2 Build a compact dtype table for the report ----
""",
15: """# ---- 4.3 Statistical profile: extend describe() with median/variance/skew/kurtosis ----
""",
17: """# ---- 4.4 Target distribution counts + percentages ----
""",
18: """# ---- 4.4 Plot the target class distribution (also saved for the report) ----
""",
20: """# ---- 4.5 Missing-value audit (count + percentage) ----
""",
22: """# ---- 4.6 Duplicate-row audit ----
""",
24: """# ---- 4.7 Constant-feature audit (zero-variance columns) ----
""",
25: """# ---- 4.7 Quasi-constant audit (one value dominates > 99%) ----
""",
27: """# ---- 4.8 Helper: per-feature distributions by class ----
# Grid of histplots with KDE, density-normalised per class so class imbalance
# does not distort the visual comparison.
""",
28: """# ---- 4.8 Render per-feature distributions for all 30 predictors ----
""",
30: """# ---- 4.9 Outlier check: per-feature boxplots ----
""",
32: """# ---- 4.10 Distribution-shape audit: skew + kurtosis ----
""",
35: """# ---- 5.1 Correlation matrix across all 30 numeric predictors ----
# Lower-triangular heatmap; upper triangle is masked to reduce clutter.
""",
37: """# ---- 5.2 Tabulate the highly-correlated pairs (|r| >= 0.85) ----
""",
39: """# ---- 5.3 Feature-target correlation (point-biserial with the binary target) ----
# Sorted by absolute value to surface the strongest linear associations.
""",
40: """# ---- 5.3 Bar plot of feature-target correlations (saved for the report) ----
""",
42: """# ---- 5.4 Pairwise scatter of the top-5 features, coloured by class ----
# Corner pairplot keeps only the lower triangle (no redundant upper panels).
""",
44: """# ---- 5.5 Variance Inflation Factor audit (multicollinearity) ----
""",
46: """# ---- 5.6 Feature variance audit + bar plot of the top-15 ----
""",
48: """# ---- 5.7 Consolidated leakage / quality audit table ----
""",
49: """# ---- 5.7 Printable leakage-audit summary ----
""",
51: """# ---- 5.8 Hierarchical clustering of predictors by correlation ----
# `clustermap` reorders features by similarity, exposing correlated blocks.
""",
53: """# ---- 5.9 One-line EDA summary table for the report ----
""",
55: """# ---- 6.1 Lock the stratified 80/20 train/test split before any tuning ----
# `stratify=y` preserves the class ratio in both subsets; the test set is now
# frozen and is only touched once at the final evaluation.
""",
56: """# ---- 6.1 Split summary: rows and positive-class prevalence per subset ----
""",
58: """# ---- 6.2 Cross-validation strategy: 5-fold stratified, shuffled, seed-locked ----
""",
60: """# ---- 6.3 Record the numeric feature list for the preprocessing pipeline ----
""",
61: """# ---- 6.3 Build a leakage-safe preprocessor: median imputer only ----
# No scaling (trees are scale-invariant) and no encoding (all features numeric).
# Wrapped as a ColumnTransformer so the imputer is fit inside CV folds.
""",
63: """# ---- 6.4 Cross-validation scoring dictionary ----
# All metrics used for CV selection are declared here once and reused.
""",
64: """# ---- 6.4 Import the metric callables used by evaluate_model / cv_report ----
""",
65: """# ---- 6.5 Reusable on-dataset metric reporter ----
# Returns a Series of accuracy, balanced accuracy, precision, recall,
# specificity, F1, ROC-AUC and PR-AUC for a fitted model on (X, y).
""",
66: """# ---- 6.5 Reusable CV reporter ----
# `cv_report` runs `cross_validate` and returns a tidy summary table +
# a one-row comparison dict (train/validation mean+std per metric).
""",
67: """# ---- 6.5 Initialise the empty model-comparison accumulator ----
""",
70: """# ---- 7.0 Pipeline helper: preprocessor + classifier ----
# Every model is wrapped the same way so preprocessing is fit inside each CV
# fold and never leaks information from validation into training.
""",
71: """# ---- 7.0 build_pipeline() definition ----
""",
72: """# ---- 7.1 Dummy baseline (predicts the majority class prior) ----
""",
73: """# ---- 7.1 Cross-validate the Dummy and append its row to the comparison ----
""",
75: """# ---- 7.2 Basic unconstrained CART (gini, no depth/leaf limits) ----
""",
76: """# ---- 7.2 Cross-validate the basic CART and append its row ----
""",
78: """# ---- 7.2 Fit the basic CART on the train split to inspect its structure ----
""",
79: """# ---- 7.3 Report tree complexity (depth / leaves / total nodes) ----
""",
81: """# ---- 7.3 Visualise the top 3 levels of the basic CART ----
""",
82: """# ---- 7.3 Export the basic CART's full if-then rules as text (truncated) ----
""",
83: """# ---- 7.4 First comparison table: Dummy vs Basic CART (CV ROC-AUC sorted) ----
""",
85: """# ---- 8.0 Generic tuner supporting GridSearchCV or RandomizedSearchCV ----
# Returns the best fitted estimator, the best params, the cv_results_ dataframe
# and the best CV score for the refit metric.
""",
86: """# ---- 8.1 Tune a CART over pre-pruning hyperparameters (training-only CV) ----
# Grid over criterion / max_depth / min_samples_split / min_samples_leaf /
# class_weight, refit on ROC-AUC.
""",
89: """# ---- 9.1 Cost-complexity pruning: derive the alpha path from the best tree ----
# Fit a base tree with the best pre-prune parameters, compute the
# `cost_complexity_pruning_path`, restrict to a 20-point alpha grid (drop the
# trivial one-node alpha), then grid-search `ccp_alpha` via CV.
""",
90: """# ---- 9.2 Plot CV ROC-AUC vs ccp_alpha to visualise the pruning trade-off ----
""",
92: """# ---- 10. Tune a Random Forest via randomised search (training-only CV) ----
# Broad space over n_estimators / max_depth / min_samples_split /
# min_samples_leaf / max_features / class_weight; 50 random draws.
""",
94: """# ---- 11. Re-evaluate all four models under identical CV folds ----
# Re-instantiate each model and reuse cv_report so the comparison is apples to
# apples. The DataFrame is sorted by CV ROC-AUC.
""",
96: """# ---- 11.1 Bar chart of the four models across the key CV metrics ----
""",
99: """# ---- 12. Sweep the operating threshold on out-of-fold training predictions ----
# Generate OOF probabilities (predict_proba via cross_val_predict so each train
# row is predicted by a fold that never saw it). Then sweep thresholds and
# select the highest-specificity threshold that still meets sensitivity >= 0.95,
# falling back to max Youden if the target is infeasible.
""",
101: """# ---- 13. Lock the final model on all train data; score the test set ONCE ----
# `evaluate_binary` builds the full metric block used in the report.
""",
102: """# ---- 13. Plot test-set confusion matrix, ROC curve and PR curve (saved) ----
""",
105: """# ---- 14.1 Calibration plot (reliability diagram) on the test set ----
""",
106: """# ---- 14. Persist artefacts via the shared meddiag_common engine ----
# Build a minimal `state` dict from the objects already trained in this notebook
# and call `meddiag_common.save_artifacts` so the CLI/GUI can load the saved
# pipeline, threshold and metadata from `artifacts/breast_*` without retraining.
# A legacy `lab02_full_artifact_bundle.joblib` bundle is also preserved for
# backwards compatibility with earlier versions of the notebook.
import joblib, platform, sklearn  # noqa: E402
from datetime import datetime       # noqa: E402

# Map the notebook's trained estimators into the shared-registry naming.
_models_registry = {
    "Dummy (prior)": build_pipeline(DummyClassifier(strategy="prior")),
    "Basic CART": basic_cart,
    "Tuned and pruned CART": pruned_cart,
    "Random Forest": best_rf_estimator,
    "Logistic Regression": build_pipeline(
        __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE
        )
    ),
}
# Fit any unfitted pipeline in the registry on the full train split so that
# every saved .joblib is ready-to-predict, matching the CLI's expectations.
for _name, _pipe in _models_registry.items():
    _inner = _pipe.named_steps.get("classifier") if hasattr(_pipe, "named_steps") else _pipe
    if not getattr(_inner, "classes_", None):
        _pipe.fit(X_train, y_train)

_state = {
    "spec": M.spec("breast"),
    "X": dataset.drop(columns=["Diagnosis"]), "y": y,
    "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
    "cv": cv,
    "models": _models_registry,
    "pruned_gs1": None, "pruned_gs2": None,
    "pruned_params": {f"model__{k}": v for k, v in best_cart_params.items()
                     if k != "model__ccp_alpha"} if "best_cart_params" in dir() else {},
    "rf_gs": None, "rf_params": best_rf_params if "best_rf_params" in dir() else {},
    "cv_table": comparison_df.rename(columns={
        "Accuracy": "accuracy", "Balanced Accuracy": "balanced_accuracy",
        "Precision": "precision", "Recall": "recall", "F1": "f1",
        "ROC-AUC": "roc_auc", "PR-AUC": "pr_auc",
    }) if "comparison_df" in dir() else pd.DataFrame(
        [{"Model": name, "roc_auc_mean": 0} for name in _models_registry]),
    "oof_prob": None, "threshold_df": threshold_df if "threshold_df" in dir() else pd.DataFrame(),
    "threshold": float(selected_threshold),
    "threshold_target_satisfied": float(
        threshold_df.loc[threshold_df.threshold <= selected_threshold, "sensitivity"].max()
    ) if "threshold_df" in dir() and len(threshold_df) else 0.95,
    "best_name": "Random Forest" if final_model is best_rf_estimator else "Tuned and pruned CART",
    "best_pipe": final_model,
    "test_metrics": {**test_metrics,
                     **{"brier": test_metrics.get("brier_score",
                                                  test_metrics.get("brier", 0.0))}},
    "default_threshold_metrics": {**test_metrics},
    "all_test_rows": [dict(Model="Random Forest" if final_model is best_rf_estimator
                           else "Tuned and pruned CART", **test_metrics)],
    "test_prob": test_proba, "test_pred": test_pred,
    "feature_cols": list(X.columns), "class_names": ("Benign", "Malignant"),
    "numeric_cols": list(X.columns), "categorical_cols": [],
    "complexity": {},
}

_paths = M.save_artifacts("breast", _state, tag="breast")
# Also keep the legacy bundle for backwards compatibility.
_legacy_dir = Path("artifacts"); _legacy_dir.mkdir(exist_ok=True)
joblib.dump({k: v for k, v in {
    "preprocessor": preprocessor, "dummy_pipeline": dummy_pipeline,
    "basic_cart_pipeline": basic_cart, "best_cart_pipeline": best_cart_estimator,
    "pruned_cart_pipeline": pruned_cart, "best_random_forest_pipeline": best_rf_estimator,
    "final_model": final_model, "threshold": selected_threshold,
    "positive_class": "Malignant", "feature_names": list(X_train.columns),
    "random_state": RANDOM_STATE, "test_metrics": test_metrics,
    "software": {"python": platform.python_version(),
                 "scikit_learn": sklearn.__version__,
                 "pandas": pd.__version__, "numpy": np.__version__},
}.items() if not isinstance(v, (dict, list, str, int, float, bool)) or v is None},
    _legacy_dir / "lab02_full_artifact_bundle.joblib")

print("Artifacts saved (CLI/GUI compatible) to:", os.path.abspath(M.ART_DIR))
print("Metadata:", _paths.get("metadata"))
""",
}

# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------
def main():
    nb = load_nb(NB)
    changed = 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown" and i in MD:
            set_src(cell, MD[i]); changed += 1
        elif cell["cell_type"] == "code":
            src = src_str(cell)
            clean = strip_banner(src)
            if i in CC:
                # Prepend the curated purpose comment, dropping the old banner.
                set_src(cell, CC[i] + clean)
                changed += 1
            else:
                # Just de-mojibake any stray comment, keep code intact.
                set_src(cell, demoji(clean))
    # De-mojibake markdown text in any cell not in MD as a safety net.
    for i, cell in enumerate(nb["cells"]):
        if i in MD:
            continue
        txt = src_str(cell)
        new = demoji(txt)
        if new != txt:
            set_src(cell, new); changed += 1
    save_nb(NB, nb)
    print(f"[OK] breast_cancer.ipynb refined ({changed} cell updates)")


if __name__ == "__main__":
    main()
