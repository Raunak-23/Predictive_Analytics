#!/usr/bin/env python3
"""Refine diabetes.ipynb (idempotent). Run from project root."""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from _refine_helpers import (  # type: ignore  # noqa: E402
    load_nb, save_nb, src_str, set_src, demoji, strip_banner, ROOT,
)
import meddiag_common as M  # noqa: E402

NB = os.path.join(ROOT, "notebooks", "diabetes.ipynb")

# ---------------------------------------------------------------------------
# MARKDOWN REFINEMENTS  (full cell replacements, standardised headings)
# ---------------------------------------------------------------------------
MD = {
0: """# Medical Diagnosis Support using Decision Trees
## Early Stage Diabetes Risk Prediction Dataset

**MDI3003 - Advanced Predictive Analytics - Lab 02**

### Educational disclaimer

This notebook is developed solely for educational and research purposes. The
resulting predictive models are **NOT** clinically validated and must **NOT** be
used for diagnosis, treatment planning, patient triage, or any real-world
healthcare decision.
""",

1: """# 1. Environment Setup

Import the required libraries, set display and plotting defaults, fix random
seeds for reproducibility, and record software versions. The shared engine
`meddiag_common` (`src/`) is placed on the path so the notebook, the CLI and the
GUI all use the same dataset metadata, preprocessing, metrics and persistence
code and can never disagree on protocol.
""",

3: """# 2. Business Understanding and Problem Framing
""",

4: """## 2.1 Problem statement

Early detection of diabetes can substantially reduce long-term complications.
This notebook builds an interpretable binary classifier that predicts whether an
individual is at **positive** risk for early-stage diabetes from readily
available demographic and symptom data.

## 2.2 Prediction task

| Item | Specification |
|:---|:---|
| **Observation unit** | One individual respondent |
| **Target variable** | Diabetes risk status (`class`) |
| **Positive class** | `Positive` (diabetes risk present) -> encoded as **1** |
| **Negative class** | `Negative` (diabetes risk absent) -> encoded as **0** |
| **Prediction time** | All predictors are assumed available before a clinical work-up |
| **Intended use** | Educational predictive-analytics prototype only |
| **Out of scope** | Diagnosis, treatment recommendation, or patient triage |

## 2.3 Error priority

In a screening context, **false negatives** (missing an at-risk patient) are
typically more costly than false positives. We therefore emphasise
**sensitivity** during threshold selection while monitoring specificity and
precision.
""",

5: """# 3. Dataset Loading and Provenance

Load the Early Stage Diabetes Risk Prediction dataset (UCI id 529), cached at
`data/diabetes_risk_prediction.csv`. The target column `class` is binarised
(`Positive` -> 1, `Negative` -> 0); predictors are one numeric (`age`) plus 15
binary symptom/gender features.
""",

7: """# 4. Data Understanding and Audit

Audit the dataset end-to-end before any modelling: shape, dtypes, statistical
profile of `age`, missingness, duplicates, constant / quasi-constant features,
unique-value counts per column, and the target distribution.
""",

9: """## 4.1 Target distribution plot
""",

11: """## 4.2 Feature distributions by class

Stacked-bar crosstabs (normalised per class) for every categorical predictor,
so class imbalance does not distort the visual comparison.
""",

13: """## 4.3 Age distribution by class

KDE + histogram of age split by diabetes-risk class.
""",

15: """## 4.4 Correlation matrix

Temporarily encode the binary categoricals (Yes->1/No->0, Male->1/Female->0)
to visualise pairwise correlation. This is a visualisation-only encoding; the
modelling pipeline uses OrdinalEncoder inside the preprocessor.
""",

17: """## 4.5 Feature-target correlation

Rank features by absolute correlation with the binarised target to preview
which predictors are likely to appear near the top of the tree.
""",

19: """## 4.6 Leakage audit

All predictors are demographic/symptom data assumed available before any
diagnostic test; no post-outcome variables are present and no identifiers exist.
""",

21: """# 5. Data Preparation

Lock the stratified 80/20 train/test split before any tuning. Cross-validation
is 5-fold stratified (shuffled, seed-locked).
""",

23: """## 5.1 Preprocessing pipeline

Numeric branch: median imputation for `age`. Categorical branch: most-frequent
imputation + `OrdinalEncoder` (binary Yes/No and Male/Female are nominal for
this modelling purpose; OrdinalEncoder keeps the 0/1 form trees prefer and is
simpler to interpret downstream).
""",

25: """# 6. Reusable Evaluation Utilities

Declare the CV scoring dictionary and three reusable helpers:
`build_pipeline` (leakage-safe preprocessor + classifier), `evaluate_model`
(comprehensive metric block for a fitted model) and `cv_report` (tidy
cross-validation summary + comparison row). A generic `tune_model` supports both
`GridSearchCV` and `RandomizedSearchCV`.
""",

27: """# 7. Baseline Model Development

Establish two baselines so we can verify that machine learning beats trivial
rules: a Dummy (predicts the prior) and a basic unconstrained CART. The basic
CART also reveals whether the unconstrained tree overfits.
""",

29: """## 7.1 Visualise the basic CART (upper levels)
""",

31: """# 8. Advanced Model Development - Pre-pruned CART

Tune the CART over pre-pruning hyperparameters (criterion, max_depth,
min_samples_split, min_samples_leaf, class_weight) with training-only CV,
refit on ROC-AUC.
""",

33: """# 9. Cost-Complexity Pruning

Fix the best pre-pruning parameters found above and select `ccp_alpha` by
cross-validation. Two-stage selection (pre-prune, then prune) yields the "tuned
and pruned CART".
""",

35: """## 9.1 Pruning validation curve

Plot CV ROC-AUC against `ccp_alpha` to visualise the pruning trade-off.
""",

37: """# 10. Random Forest

Tune a Random Forest with a randomised search (broad hyperparameter space,
50 draws), refit on ROC-AUC, training-only CV.
""",

40: """# 11. Model Comparison (Cross-Validation)

Compare the four models - Dummy, Basic CART, Tuned & Pruned CART, Random
Forest - under identical 5-fold stratified CV folds.
""",

41: """## 11.1 Split-criterion comparison

Quick Gini-vs-Entropy comparison on a basic CART to confirm criterion choice
does not meaningfully change discrimination.
""",

44: """# 12. Operating Threshold Selection

Sweep the operating threshold on out-of-fold training predictions of the chosen
final model (here the Tuned & Pruned CART for interpretability, with Random
Forest available as a higher-performance alternative). Enforce sensitivity >=
0.95 and pick the highest-specificity threshold that still meets the target;
fall back to max-Youden if the target is infeasible.
""",

45: """## 12.1 Model-comparison visualisation

Bar chart of the four models across the key cross-validated metrics
(Accuracy, F1, Balanced Accuracy, ROC_AUC, PR_AUC).
""",

48: """## 12.2 Threshold plot

Sensitivity and specificity curves against the operating threshold, with the
selected threshold marked.
""",

50: """# 13. Final Evaluation on the Locked Test Set

Lock the model on all training data and evaluate on the test set exactly once.
Report sensitivity, specificity, precision, NPV, F1, balanced accuracy, ROC-AUC,
PR-AUC, Brier, MCC and the explicit TN/FP/FN/TP at the chosen threshold.
""",

52: """## 13.1 Confusion matrix
""",

54: """## 13.2 ROC curve
""",

56: """## 13.3 Precision-recall curve
""",

58: """## 13.4 Calibration plot (reliability diagram)
""",

60: """# 14. Interpretation and Feature Importance

Inspect the fitted tree's complexity, visualise the pruned tree, export its
if-then rules, and quantify importance both by impurity (tree-internal) and by
permutation (model-agnostic, on the test set).
""",

62: """## 14.1 Visualise the final pruned tree (top 4 levels)
""",

64: """## 14.2 Decision rules (text)
""",

66: """## 14.3 Impurity-based feature importance
""",

68: """## 14.4 Permutation importance (test set)
""",

70: """## 14.5 Decision-path tracing

Trace a hand-picked sample (a correct and an incorrect prediction where
available) through the tree, printing each node's split, threshold and leaf
class proportion. This is the lab's interpretability evidence.
""",

72: """## 14.6 Robustness analysis

Re-fit the basic tree across 10 random train/test seeds and report the mean and
standard deviation of CV ROC-AUC - a lightweight split-stability check.
""",

74: """# 15. Save Artifacts

Persist every artefact the CLI/GUI need - the fitted final pipeline, the
operating threshold, individual model pipelines, the CV and test results CSVs
and the metadata JSON - via the shared `meddiag_common.save_artifacts` engine,
so the notebook and the CLI can never disagree. A legacy
`diabetes_full_artifact_bundle.joblib` bundle is also preserved for backwards
compatibility.
""",

76: """# 16. Conclusion and Responsible Reporting
""",

77: """## 16.1 Summary of findings

1. **Baseline vs. models** - Both the Tuned & Pruned CART and the Random Forest
   substantially outperform the Dummy baseline across all discrimination metrics.
2. **Pruning effect** - Pre-pruning and cost-complexity pruning reduced tree
   depth and leaf count compared with the basic CART, improving validation
   stability while maintaining strong sensitivity.
3. **Error analysis** - Final test evaluation at the selected operating
   threshold yields explicit TN/FP/FN/TP counts. False negatives are minimised
   given the screening-oriented threshold strategy.
4. **Interpretability** - The pruned tree exposes explicit if-then rules
   involving age, polyuria, gender and other symptoms. These are
   **dataset-derived thresholds**, not clinically validated cut-offs.
5. **Limitations** - small, geographically limited sample; self-reported symptom
   data subject to recall and selection bias; no external validation cohort
   (transportability unknown); feature importance describes *association*, not
   *causation*.

## 16.2 Requirements before real-world use

- Prospective clinical validation on an independent, representative cohort.
- Calibration assessment and subgroup (age, sex) performance audits.
- Governance framework including human oversight, periodic retraining and a
  rollback protocol.
- Regulatory review if ever intended for any diagnostic or triage workflow.
""",
}

# ---------------------------------------------------------------------------
# CODE-CELL purpose-header comments  (prepended; banner stripped first)
# ---------------------------------------------------------------------------
CC = {
2: """# ---- 1. Environment Setup ----
# Imports, display options, random seeding for reproducibility, and a quick
# version report. A `report_figures/` dir is created for publication-quality
# 300-DPI exports used by the lab report docx.
""",
6: """# ---- 3. Load the Early Stage Diabetes Risk dataset ----
# Read the cached CSV (relative to the notebooks/ working dir). Binarise the
# `class` target (Positive->1, Negative->0) and split into X (features) and y.
""",
8: """# ---- 4. Dataset understanding audit ----
# Shape, dtypes, statistical summary of age, missingness, duplicates, constant
# features, unique-value counts and the target distribution - one consolidated
# audit cell so the report has a single reproducible block of evidence.
""",
10: """# ---- 4.1 Target distribution plot (saved for the report) ----
""",
12: """# ---- 4.2 Categorical-feature distributions by class (saved) ----
# Per-feature stacked-bar crosstabs (normalised by class column) so the visual
# is not distorted by the class imbalance (~62% positive).
""",
14: """# ---- 4.3 Age distribution by class KDE (saved) ----
""",
16: """# ---- 4.4 Correlation matrix of encoded features (saved) ----
# Temporarily encode Yes/No and Male/Female so the heatmap is meaningful. This
# is visualisation-only; the modelling pipeline encodes inside the ColumnTransformer.
""",
18: """# ---- 4.5 Feature-target correlation ranked by |r| (saved) ----
""",
20: """# ---- 4.6 Leakage audit summary ----
""",
22: """# ---- 5. Lock the stratified 80/20 train/test split before any tuning ----
# `stratify=y` preserves the positive prevalence in both subsets; the test set
# is frozen and only touched once at the final evaluation.
""",
24: """# ---- 5.1 Build the preprocessing pipeline ----
# Numeric branch: median imputation for age.
# Categorical branch: most-frequent imputation + OrdinalEncoder (handles
# unknown values with -1 so production inputs never break predict).
""",
26: """# ---- 6. Reusable evaluation utilities ----
# SCORING dictionary, build_pipeline, evaluate_model, cv_report, tune_model.
# Defining these once keeps every later cell concise and consistent.
""",
28: """# ---- 7. Baselines: Dummy + Basic unconstrained CART ----
# Cross-validate both and inspect the basic tree's complexity to motivate the
# tuning/pruning in the next sections.
""",
30: """# ---- 7.1 Visualise the basic CART's top 3 levels (saved) ----
# Use post-transform feature names so the tree labels are human-readable.
""",
32: """# ---- 8. Tune the CART over pre-pruning hyperparameters ----
# Grid over criterion / max_depth / min_samples_split / min_samples_leaf /
# class_weight; refit on ROC-AUC; training-only CV (test set untouched).
""",
34: """# ---- 9. Cost-complexity pruning ----
# Fit the preprocessor on the train split only, train a base tree with the best
# pre-pruning params on the transformed features, take the pruning path,
# restrict to a 20-point alpha grid (drop the trivial one-node alpha), then
# grid-search `ccp_alpha` via CV.
""",
36: """# ---- 9.1 Pruning validation curve (CV ROC-AUC vs ccp_alpha, saved) ----
""",
39: """# ---- 10. Tune a Random Forest via randomised search ----
# Broad space over n_estimators / max_depth / min_samples_split /
# min_samples_leaf / max_features / class_weight; 50 draws; refit on ROC-AUC.
""",
42: """# ---- 11.1 Split-criterion comparison: gini vs entropy on a basic CART ----
""",
43: """# ---- 11. Re-evaluate all four models under identical CV folds ----
# Reuse cv_report so the comparison is apples to apples; sort by CV ROC-AUC.
""",
46: """# ---- 12.1 Bar chart of the four models across the key CV metrics (saved) ----
""",
47: """# ---- 12. Sweep the operating threshold on out-of-fold training predictions ----
# Generate OOF probabilities via cross_val_predict (each train row predicted by
# a fold that never saw it). Sweep thresholds; enforce sensitivity >= 0.95 and
# pick the highest-specificity threshold that still meets the target; fall back
# to max Youden if the target is infeasible.
""",
49: """# ---- 12.2 Plot sensitivity/specificity vs threshold (saved) ----
""",
51: """# ---- 13. Lock the final model on all train data; score the test set ONCE ----
# `evaluate_binary` builds the full metric block used in the report.
""",
53: """# ---- 13.1 Test-set confusion matrix (saved) ----
""",
55: """# ---- 13.2 ROC curve (saved) ----
""",
57: """# ---- 13.3 Precision-recall curve (saved) ----
""",
59: """# ---- 13.4 Calibration plot / reliability diagram (saved) ----
""",
61: """# ---- 14. Inspect the fitted tree complexity (depth / leaves / nodes) ----
""",
63: """# ---- 14.1 Visualise the final pruned tree (top 4 levels, saved) ----
""",
65: """# ---- 14.2 Export the final tree's if-then rules as text (truncated) ----
""",
67: """# ---- 14.3 Impurity-based feature importance + bar chart (saved) ----
# Non-causal importance biased toward high-cardinality / many-split features.
""",
69: """# ---- 14.4 Permutation importance on the test set (model-agnostic, saved) ----
# Measures how much ROC-AUC drops when each feature is randomly shuffled.
""",
71: """# ---- 14.5 Trace two samples (one correct, one wrong where available) ----
# Print the per-node split predicate + leaf class proportion to show *why* the
# tree decided the way it did for concrete records.
""",
74: """# ---- 14.6 Robustness analysis: split stability across 10 random seeds ----
""",
75: """# ---- 15. Persist artefacts via the shared meddiag_common engine ----
# Build a minimal `state` dict from the objects already trained in this
# notebook and call `meddiag_common.save_artifacts` so the CLI/GUI can load the
# saved pipeline, threshold and metadata from `artifacts/diabetes_*` without
# retraining. A legacy `diabetes_full_artifact_bundle.joblib` is also preserved.
import joblib  # noqa: E402

# Map the notebook's trained estimators into the shared-registry naming.
_models_registry = {
    "Dummy (prior)": build_pipeline(DummyClassifier(strategy="prior")),
    "Basic CART": build_pipeline(DecisionTreeClassifier(
        criterion="gini", random_state=RANDOM_STATE)),
    "Tuned and pruned CART": best_pruned_estimator,
    "Random Forest": best_rf_estimator,
    "Logistic Regression": build_pipeline(
        __import__("sklearn.linear_model",
                   fromlist=["LogisticRegression"]).LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)
    ),
}
# Fit any unfitted pipeline in the registry on the full train split so that
# every saved .joblib is ready-to-predict, matching the CLI's expectations.
for _name, _pipe in _models_registry.items():
    _inner = _pipe.named_steps.get("classifier") if hasattr(_pipe, "named_steps") else _pipe
    if not getattr(_inner, "classes_", None):
        _pipe.fit(X_train, y_train)

_state = {
    "spec": M.spec("diabetes"),
    "X": df.drop(columns=[TARGET_COL]), "y": y,
    "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
    "cv": cv,
    "models": _models_registry,
    "pruned_gs1": None, "pruned_gs2": None,
    "pruned_params": {f"model__{k}": v for k, v in best_cart_params.items()
                     if k != "model__ccp_alpha"} if "best_cart_params" in dir() else {},
    "rf_gs": None, "rf_params": best_rf_params if "best_rf_params" in dir() else {},
    "cv_table": comparison_df.rename(columns={
        "Accuracy": "accuracy", "Balanced_Accuracy": "balanced_accuracy",
        "Precision": "precision", "Recall": "recall", "F1": "f1",
        "ROC_AUC": "roc_auc", "PR_AUC": "pr_auc",
    }) if "comparison_df" in dir() else pd.DataFrame(
        [{"Model": name, "roc_auc_mean": 0} for name in _models_registry]),
    "oof_prob": None, "threshold_df": threshold_df if "threshold_df" in dir() else pd.DataFrame(),
    "threshold": float(selected_threshold),
    "threshold_target_satisfied": float(
        threshold_df.loc[threshold_df.threshold <= selected_threshold, "sensitivity"].max()
    ) if "threshold_df" in dir() and len(threshold_df) else 0.95,
    "best_name": "Tuned and pruned CART" if final_model is best_pruned_estimator
                 else "Random Forest",
    "best_pipe": final_model,
    "test_metrics": {**test_metrics,
                     **{"brier": test_metrics.get("brier_score",
                                                  test_metrics.get("brier", 0.0))}},
    "default_threshold_metrics": {**test_metrics},
    "all_test_rows": [dict(
        Model="Tuned and pruned CART" if final_model is best_pruned_estimator
              else "Random Forest", **test_metrics)],
    "test_prob": test_proba, "test_pred": test_pred,
    "feature_cols": list(X.columns), "class_names": ("Negative", "Positive"),
    "numeric_cols": numeric_features, "categorical_cols": categorical_features,
    "complexity": {},
}

_paths = M.save_artifacts("diabetes", _state, tag="diabetes")
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
                set_src(cell, CC[i] + clean)
                changed += 1
            else:
                set_src(cell, demoji(clean))
    for i, cell in enumerate(nb["cells"]):
        if i in MD:
            continue
        txt = src_str(cell)
        new = demoji(txt)
        if new != txt:
            set_src(cell, new); changed += 1
    save_nb(NB, nb)
    print(f"[OK] diabetes.ipynb refined ({changed} cell updates)")


if __name__ == "__main__":
    main()
