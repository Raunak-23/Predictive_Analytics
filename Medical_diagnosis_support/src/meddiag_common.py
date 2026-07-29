#!/usr/bin/env python3
"""
meddiag_common - shared engine for the Medical Diagnosis Support lab (Lab 02,
Disease Classification using Decision Trees).

This module is the single source of truth used by
  * the three notebooks in ``notebooks/``,
  * ``src/meddiag_cli.py`` (inference / evaluation / retraining / registry),
  * ``src/meddiag_gui.py`` (Tkinter desktop GUI).

It understands two UCI datasets in a dataset-aware way::

    breast     Breast Cancer Wisconsin (Diagnostic)  - core, 30 numeric, no missing
    diabetes   Early Stage Diabetes Risk Prediction  - 15 binary symptoms + gender + age
    heart      Heart Disease (Cleveland, UCI id 45)   - numeric codes, missing ca/thal

Each dataset is reduced to a binary classification problem with an explicitly
*defined* positive class (the disease state). Everything that touches the data -
splitting, imputation, one-hot encoding, scaling, cross-validation, tuning,
pruning, threshold selection and the single final test evaluation - is encoded
here once and reused, so the notebook / CLI / GUI can never disagree on protocol.

The protocol follows the lab manual: a stratified 80/20 split locked before any
model work, training-only 5-fold stratified CV for selection, cost-complexity
pruning chosen by CV, an out-of-fold threshold selected against a *declared*
sensitivity target, and a single locked test evaluation. Tree ensembling and the
optional Logistic Regression benchmark round out the required progression.

Educational-use boundary
-------------------------
The artefacts produced here are a research / teaching prototype. They are NOT a
clinically validated diagnostic system and must never be used for patient care.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# VERSIONS / PATHS / SEED  (resolved against THIS file so the CLI works
# from any working directory, exactly like the reference house-price CLI)
# ============================================================
import platform  # noqa: E402

try:
    import sklearn
    _SK_VERSION = sklearn.__version__
except Exception:
    _SK_VERSION = "?"

try:
    import xgboost  # noqa: F401  (optional bench; presence is informational)
    _XGB_VERSION = xgboost.__version__
except Exception:
    _XGB_VERSION = None

try:
    import joblib
    _JB_VERSION = joblib.__version__
except Exception:
    _JB_VERSION = "?"

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
# dist layout: <project>/src/meddiag_common.py  OR  <project>/meddiag_common.py
PARENT = os.path.dirname(MODULE_DIR)
SRC_DIR = MODULE_DIR if os.path.basename(MODULE_DIR) == "src" else PARENT
PROJECT_ROOT = os.path.dirname(SRC_DIR) if os.path.basename(SRC_DIR) == "src" else PARENT
ART_DIR = os.path.join(PROJECT_ROOT, "artifacts")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIG_DIR = os.path.join(ART_DIR, "figures")
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "examples")

SEED = 42
CV_FOLDS = 5
TEST_SIZE = 0.20
TARGET_SENSITIVITY = 0.95   # instructional default; relaxed if infeasible
COST_FN = 5.0              # instructional false-negative cost (screening-like)
COST_FP = 1.0

# ---- dataviz skill: validated, design-system-agnostic palette (light surface) --
# diverging poles for the binary disease state (red = disease present), and an
# 8-slot categorical order for any multi-series plot, with recessive grid/ink.
PALETTE = {
    "surface": "#fcfcfb",
    "ink_primary": "#0b0b0b",
    "ink_secondary": "#52514e",
    "ink_muted": "#898781",
    "gridline": "#e1e0d9",
    "baseline": "#c3c2b7",
    "neg": "#2a78d6",        # disease-absent  (blue)
    "pos": "#e34948",        # disease-present (red)
    "good": "#0ca30c",
    "warning": "#fab219",
    "cat": [                # 8-slot fixed categorical order
        "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
        "#e87ba4", "#008300", "#4a3aa7", "#e34948",
    ],
}


# ============================================================
# DATASET REGISTRY
# ============================================================
@dataclass
class DatasetSpec:
    key: str
    ucid: int
    display_name: str
    csv_name: str
    doi: str
    source_note: str
    target_col: str
    # raw_label -> 0/1 : which raw values count as the positive (disease) class
    positive_raw_values: tuple
    positive_label: str          # short name of the disease state, e.g. "Malignant"
    negative_label: str
    class_names: tuple            # (neg, pos) display strings for confusion matrices / trees
    numeric_cols: tuple
    categorical_cols: tuple
    drop_cols: tuple = ()         # id cols etc.
    observation_unit: str = ""
    population_note: str = ""
    # value codes documented so the GUI / inputs are self-describing
    category_codes: dict = field(default_factory=dict)

    @property
    def feature_cols(self):
        return list(self.numeric_cols) + list(self.categorical_cols)

    @property
    def positive_pct_note(self):
        return ""


DATASETS = {
    "breast": DatasetSpec(
        key="breast",
        ucid=17,
        display_name="Breast Cancer Wisconsin (Diagnostic)",
        csv_name="breast_cancer_wisconsin.csv",
        doi="10.24432/C5DW2B",
        source_note=("Wolberg, Mangasarian, Street & Street. UCI ML Repository. "
                     "569 FNA-image-derived measurements; target Diagnosis M/B."),
        target_col="Diagnosis",
        positive_raw_values=("M",),
        positive_label="Malignant",
        negative_label="Benign",
        class_names=("Benign", "Malignant"),
        numeric_cols=tuple(
            [f"{m}{i}" for i in (1, 2, 3) for m in
             ("radius", "texture", "perimeter", "area", "smoothness",
              "compactness", "concavity", "concave_points", "symmetry",
              "fractal_dimension")]
        ),
        categorical_cols=(),
        observation_unit="One fine-needle-aspirate image measurement set per patient.",
        population_note=("Convenience sample from a single institution's image "
                         "extraction process; measurements derive from digitised FNA "
                         "images, not raw medical images. Demographic attributes are "
                         "absent, so a subgroup / fairness audit cannot be performed."),
    ),
    "diabetes": DatasetSpec(
        key="diabetes",
        ucid=529,
        display_name="Early Stage Diabetes Risk Prediction",
        csv_name="diabetes_risk_prediction.csv",
        doi="10.24432/C5VG8H",
        source_note=("Islam & Rahman. UCI ML Repository. 520 questionnaire rows; "
                     "16 symptom/demographic predictors; target 'class'."),
        target_col="class",
        positive_raw_values=("Positive",),
        positive_label="Positive (diabetes)",
        negative_label="Negative",
        class_names=("Negative", "Positive"),
        numeric_cols=("age",),
        categorical_cols=(
            "gender", "polyuria", "polydipsia", "sudden_weight_loss", "weakness",
            "polyphagia", "genital_thrush", "visual_blurring", "itching",
            "irritability", "delayed_healing", "partial_paresis",
            "muscle_stiffness", "alopecia", "obesity",
        ),
        observation_unit="One questionnaire respondent (early-presentation visit).",
        population_note=("Single-source questionnaire from one clinical setting; "
                         "age is self-reported and gender is a sensitive attribute "
                         "intended only as a modelled predictor and *subgroup lens*, "
                         "never as a triage/eligibility criterion."),
        category_codes={
            "gender": ["Female", "Male"],
            "polyuria": ["No", "Yes"], "polydipsia": ["No", "Yes"],
            "sudden_weight_loss": ["No", "Yes"], "weakness": ["No", "Yes"],
            "polyphagia": ["No", "Yes"], "genital_thrush": ["No", "Yes"],
            "visual_blurring": ["No", "Yes"], "itching": ["No", "Yes"],
            "irritability": ["No", "Yes"], "delayed_healing": ["No", "Yes"],
            "partial_paresis": ["No", "Yes"], "muscle_stiffness": ["No", "Yes"],
            "alopecia": ["No", "Yes"], "obesity": ["No", "Yes"],
        },
    ),
}

CHOICES = tuple(DATASETS.keys())


def spec(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {CHOICES}.")
    return DATASETS[name]


# ============================================================
# DATA LOADING (cache-first; re-fetch from UCIML Repo on demand)
# ============================================================
def load_dataset(name: str, from_cache: bool = True) -> pd.DataFrame:
    """Return the raw dataset as a DataFrame (features + target, unmodified)."""
    s = spec(name)
    path = os.path.join(DATA_DIR, s.csv_name)
    if from_cache and os.path.exists(path):
        return pd.read_csv(path)
    # Fall back to fetching directly from the UCI repository.
    try:
        from ucimlrepo import fetch_ucirepo
        d = fetch_ucirepo(id=s.ucid)
        df = pd.concat([d.data.features.copy(), d.data.targets.copy()], axis=1)
        names = list(df.columns)
        # normalise the target column name produced by ucimlrepo to the lab's name
        if s.target_col not in df.columns and len(d.data.targets.columns) == 1:
            df = df.rename(columns={d.data.targets.columns[0]: s.target_col})
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(path, index=False)
        return df
    except Exception as e:  # pragma: no cover
        raise FileNotFoundError(
            f"Could not find or fetch '{name}' dataset.\n"
            f"Expected cached CSV at {path}.\nOriginal error: {e}"
        ) from e


def make_xy(name: str, df: pd.DataFrame | None = None):
    """Return (X, y) with the target binarised to the explicitly-defined positive
    class (disease state = 1). Drops id / leakage columns and the target itself."""
    s = spec(name)
    df = df if df is not None else load_dataset(name)
    cols = [c for c in df.columns
            if c in set(s.feature_cols) and c not in s.drop_cols]
    X = df[cols].copy()
    raw_t = df[s.target_col].to_numpy()
    pos_set = set(s.positive_raw_values)
    # handle string vs numeric raw values robustly
    def _is_pos(v):
        if pd.isna(v):
            return False
        if isinstance(v, str):
            return v.strip() in pos_set
        try:
            return float(v) in pos_set or int(round(v)) in {
                int(p) for p in pos_set if not isinstance(p, str)
            } or float(v) in {float(p) for p in pos_set}
        except (TypeError, ValueError):
            return v in pos_set
    y = pd.Series(np.array([1 if _is_pos(v) else 0 for v in raw_t], dtype=int),
                  name=name + "_positive")
    # coerce numeric cols to float; keep categorical cols as object strings
    for c in s.numeric_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in s.categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype("object").astype(str)
            X.loc[X[c].isin(("nan", "None", "")), c] = np.nan
    return X, y


# ============================================================
# PREPROCESSING (training-fitted pipeline; trees need impute+OHE, not scale)
# ============================================================
def numeric_categorical(name: str, X: pd.DataFrame | None = None):
    """Return (numeric_cols_present, categorical_cols_present) for a dataset,
    restricted to the columns actually present in X (defensive)."""
    s = spec(name)
    if X is None:
        num = [c for c in s.numeric_cols]
        cat = [c for c in s.categorical_cols]
        return num, cat
    num = [c for c in s.numeric_cols if c in X.columns]
    cat = [c for c in s.categorical_cols if c in X.columns]
    # also auto-detect any numeric/object columns the spec didn't pre-list
    extra_num = [c for c in X.select_dtypes(include=[np.number]).columns
                 if c not in cat and c not in num]
    extra_cat = [c for c in X.select_dtypes(exclude=[np.number]).columns
                 if c not in num and c not in cat]
    return num + extra_num, cat + extra_cat


def build_preprocessor(name: str, X: pd.DataFrame | None = None,
                       scale: bool = False, refit_obj=None):
    """Build (and optionally fit) a ColumnTransformer that imputes then encodes.

    * numeric: median imputation (+ StandardScaler only when scale=True).
      Trees do not need scaling; the Logistic Regression benchmark does.
    * categorical: most-frequent imputation + one-hot (handle_unknown='ignore').
      ca/thal-style codes with NaN are handled correctly.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    num, cat = numeric_categorical(name, X)
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers = []
    if num:
        transformers.append(("num", num_pipe, num))
    if cat:
        transformers.append(("cat", cat_pipe, cat))
    pre = ColumnTransformer(transformers, remainder="drop")
    if refit_obj is not None:
        pre.fit(refit_obj[0], refit_obj[1])
    return pre


def attach(preprocessor, estimator):
    """Wrap a fitted-unfitted ColumnTransformer + an estimator into a Pipeline."""
    from sklearn.pipeline import Pipeline
    return Pipeline([("pre", preprocessor), ("model", estimator)])


def expected_columns(preprocessor) -> list:
    """Recover the ordered feature columns a ColumnTransformer was fit on."""
    cols = []
    try:
        for _, _sub, col_list, *_rest in preprocessor.transformers_:
            cols.extend(col_list)
    except AttributeError:
        for name, _sub, col_list in preprocessor.transformers:
            if name == "remainder":
                continue
            cols.extend(col_list)
    return cols


# ============================================================
# MODELS  (Blueprints for the required progression + optional benchmarks)
# ============================================================
def basic_cart(seed: int = SEED):
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(criterion="gini", random_state=seed)  # unconstrained


def random_forest(seed: int = SEED):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=500, max_features="sqrt",
        class_weight="balanced", random_state=seed, n_jobs=4,
    )


def tune_random_forest(name: str, X_train, y_train, cv, n_jobs: int = 4):
    """Compact Random-Forest grid search (training-only CV, refit on ROC-AUC).

    Lab requirement 12.3: tune n_estimators, max_depth, min_samples_leaf,
    max_features and class_weight without touching the locked test set.
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier

    pre = build_preprocessor(name, X_train, scale=False)
    grid = {
        "model__n_estimators": [300, 500],
        "model__max_depth": [None, 8, 12],
        "model__min_samples_leaf": [1, 2, 5],
        "model__max_features": ["sqrt", "log2"],
        "model__class_weight": [None, "balanced"],
    }
    gs = GridSearchCV(
        attach(pre, RandomForestClassifier(random_state=SEED, n_jobs=1)),
        param_grid=grid, scoring=SCORING, refit="roc_auc",
        cv=cv, n_jobs=n_jobs, return_train_score=True, error_score="raise",
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs, gs.best_params_


def logistic_regression(seed: int = SEED):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=5000, class_weight="balanced",
                              random_state=seed)


def dummy_prior():
    from sklearn.dummy import DummyClassifier
    return DummyClassifier(strategy="prior")


def base_blueprints() -> dict:
    """Estimators that need *no* tuning - used directly in the comparison table."""
    return {
        "Dummy (prior)": dummy_prior(),
        "Basic CART": basic_cart(),
        "Random Forest": random_forest(),
    }


def logistic_blueprint():
    """Logistic Regression pipeline blueprint (needs scaling alongside the OHE)."""
    return logistic_regression()


# ---- SCORING / METRICS -----------------------------------------------------
def _make_specificity_scorer():
    from sklearn.metrics import make_scorer, recall_score
    # specificity = recall of the negative class
    return make_scorer(recall_score, pos_label=0, zero_division=0)


SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "recall": "recall",                 # sensitivity for the disease class
    "specificity": _make_specificity_scorer(),
    "precision": "precision",
    "f1": "f1",
    "mcc": "matthews_corrcoef",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "brier": "neg_brier_score",
}


def evaluate_binary(y_true, probability, threshold: float = 0.5) -> dict:
    """Full binary test-set metrics block, exactly as the lab manual requires
    (Step 14): confusion counts + discrimination + decision + cost + uncertainty."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                  precision_score, recall_score, f1_score,
                                  roc_auc_score, average_precision_score,
                                  matthews_corrcoef, brier_score_loss,
                                  confusion_matrix)
    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()

    def _safe(num, den):
        return float(num / den) if den else float("nan")

    spec_ = _safe(tn, tn + fp)
    npv = _safe(tn, tn + fn)
    return {
        "threshold": float(threshold),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "N": int(tn + fp + fn + tp),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(spec_),
        "precision_ppv": float(precision_score(y_true, prediction, zero_division=0)),
        "npv": float(npv),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "expected_cost": float(COST_FN * fn + COST_FP * fp),
    }


def expected_cost(fn: int, fp: int, cfn: float = COST_FN, cfp: float = COST_FP) -> float:
    return float(cfn * fn + cfp * fp)


# ---- THRESHOLD SELECTION ---------------------------------------------------
def threshold_curve(oof_prob, y_true):
    """Sensitivity & specificity across thresholds 0.05..0.95."""
    from sklearn.metrics import confusion_matrix
    y_true = np.asarray(y_true).astype(int)
    oof_prob = np.asarray(oof_prob, dtype=float)
    rows = []
    for t in np.linspace(0.05, 0.95, 181):
        pred = (oof_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        rows.append((float(t), float(sens), float(spec), float(prec)))
    return pd.DataFrame(rows, columns=["threshold", "sensitivity",
                                       "specificity", "precision"])


def select_threshold(threshold_df: pd.DataFrame, target_sens: float = TARGET_SENSITIVITY):
    """Highest-specificity threshold still meeting a declared sensitivity target.
    Relaxes the target (0.95 -> 0.90 -> 0.85 -> 0.80 -> 0.50) if infeasible, and
    records which target the chosen threshold actually satisfied."""
    chosen = None
    satisfied = None
    for target in (target_sens, 0.90, 0.85, 0.80, 0.50):
        feasible = threshold_df[threshold_df["sensitivity"] >= target]
        if not feasible.empty:
            best = feasible.sort_values("specificity", ascending=False).iloc[0]
            chosen, satisfied = float(best["threshold"]), float(target)
            break
    if chosen is None:
        # degenerate: take the threshold with maximum sensitivity
        chosen = float(threshold_df.sort_values("sensitivity", ascending=False)
                       .iloc[0]["threshold"])
        satisfied = float(threshold_df.sort_values("sensitivity", ascending=False)
                          .iloc[0]["sensitivity"])
    return chosen, satisfied


# ---- COST-COMPLEXITY PRUNING (the "tuned & pruned CART") -------------------
def tune_pruned_cart(name: str, X_train, y_train, cv, n_jobs: int = 4):
    """Two-stage selection faithful to the lab manual:
      (1) pre-pruning grid over criterion / max_depth / min_samples_split /
          min_samples_leaf / class_weight, refit on ROC-AUC, training-only;
      (2) fix the best pre-prune parameters and run cost-complexity pruning over
          ccp_alpha (alphas from the pruning path, minus the trivial one-node
          tree), refit on ROC-AUC.
    Returns (best_pipe, cv_grid_stage1, cv_grid_stage2, selected_params).
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.tree import DecisionTreeClassifier

    pre = build_preprocessor(name, X_train, scale=False)
    # ---- Stage 1: pre-pruning grid ------------------------------------------
    grid1 = {
        "model__criterion": ["gini", "entropy", "log_loss"],
        "model__max_depth": [2, 3, 4, 5, 6, 8, None],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 5, 10, 20],
        "model__class_weight": [None, "balanced"],
    }
    gs1 = GridSearchCV(attach(pre, DecisionTreeClassifier(random_state=SEED)),
                       param_grid=grid1, scoring=SCORING, refit="roc_auc",
                       cv=cv, n_jobs=n_jobs, return_train_score=True, error_score="raise")
    gs1.fit(X_train, y_train)
    best1 = gs1.best_params_

    # ---- Stage 2: cost-complexity pruning on top of the best pre-prune --------
    # recover candidate alphas from a tree fit with the chosen pre-prune params
    base_params = {
        "criterion": best1["model__criterion"],
        "max_depth": best1["model__max_depth"],
        "min_samples_split": best1["model__min_samples_split"],
        "min_samples_leaf": best1["model__min_samples_leaf"],
        "random_state": SEED,
    }
    pre2 = build_preprocessor(name, X_train, scale=False)
    Xt_pre = pre2.fit_transform(X_train, y_train)
    base_tree = DecisionTreeClassifier(**base_params)
    base_tree.fit(Xt_pre, y_train)
    path = base_tree.cost_complexity_pruning_path(Xt_pre, y_train)
    ccp_alphas = np.unique(path.ccp_alphas[:-1])   # drop the one-node trivial tree
    ccp_alphas = ccp_alphas[ccp_alphas < 0.05] if len(ccp_alphas) > 30 else ccp_alphas
    if len(ccp_alphas) == 0:
        ccp_alphas = np.array([0.0])

    grid2 = {
        "model__ccp_alpha": ccp_alphas.tolist(),
        "model__class_weight": [None, "balanced"],
    }
    pre3 = build_preprocessor(name, X_train, scale=False)
    gs2 = GridSearchCV(
        attach(pre3, DecisionTreeClassifier(
            criterion=base_params["criterion"],
            max_depth=base_params["max_depth"],
            min_samples_split=base_params["min_samples_split"],
            min_samples_leaf=base_params["min_samples_leaf"],
            random_state=SEED)),
        param_grid=grid2, scoring=SCORING, refit="roc_auc",
        cv=cv, n_jobs=n_jobs, return_train_score=True, error_score="raise")
    gs2.fit(X_train, y_train)

    final_params = {**best1, **gs2.best_params_}
    return gs2.best_estimator_, gs1, gs2, final_params


# ---- MODEL COMPARISON ------------------------------------------------------
def cv_summary_table(name: str, models: dict, X_train, y_train, cv,
                     n_jobs: int = 4) -> pd.DataFrame:
    """Run all models under identical stratified CV folds; report mean +/- std of
    ROC-AUC, PR-AUC, sensitivity, specificity, balanced accuracy, F1, MCC."""
    from sklearn.model_selection import cross_validate
    rows = []
    for model_name, pipe in models.items():
        sc = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING,
                            n_jobs=n_jobs, return_train_score=False)
        row = {"Model": model_name}
        for metric, col in (("roc_auc", "test_roc_auc"), ("pr_auc", "test_pr_auc"),
                            ("sensitivity", "test_recall"),
                            ("specificity", "test_specificity"),
                            ("balanced_accuracy", "test_balanced_accuracy"),
                            ("f1", "test_f1"), ("mcc", "test_mcc"),
                            ("brier", "test_brier")):
            row[f"{metric}_mean"] = float(np.mean(sc[col]))
            row[f"{metric}_std"] = float(np.std(sc[col]))
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    return df


# ---- THE FULL TRAINING ORCHESTRATION (used by notebooks + CLI retrain) -----
def train_dataset(name: str, df: pd.DataFrame | None = None,
                  verbose: bool = True, return_state: bool = True) -> dict:
    """Run the complete leakage-safe workflow for one dataset and return a
    dictionary holding every artefact (models, threshold, test metrics, CV
    tables, fitted trees, prose provenance). Used by both the notebooks (which
    also surface each intermediate for narrative) and `meddiag_cli.py retrain`."""
    from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                          cross_val_predict, cross_validate)

    s = spec(name)
    df = df if df is not None else load_dataset(name)
    X, y = make_xy(name, df)

    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    log(f"\n[1/7] Problem framing - {s.display_name}")
    log(f"      positive class = 1 = '{s.positive_label}' "
        f"(raw {[str(v) for v in s.positive_raw_values]})")

    log(f"\n[2/7] Leakage-safe split (test_size={TEST_SIZE}, stratified, seed={SEED})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    log(f"      train={len(X_train)}  test={len(X_test)}  "
        f"train prevalence={y_train.mean():.3f}  test prevalence={y_test.mean():.3f}")

    log("\n[3/7] Tune & prune the advanced CART (training-only CV)")
    pruned_pipe, gs1, gs2, pruned_params = tune_pruned_cart(name, X_train, y_train, cv)
    log(f"      pre-prune best: { {k:v for k,v in pruned_params.items() if k!='model__ccp_alpha'} }")
    log(f"      ccp_alpha: {pruned_params.get('model__ccp_alpha')}")
    log(f"      CV ROC-AUC (pruned): {gs2.best_score_:.4f}")

    log("\n[4/8] Tune Random Forest (compact training-only grid)")
    rf_pipe, rf_gs, rf_params = tune_random_forest(name, X_train, y_train, cv)
    log(f"      RF best params: {rf_params}")
    log(f"      CV ROC-AUC (RF): {rf_gs.best_score_:.4f}")

    log("\n[5/8] Build the required model trio + optional Logistic Regression")
    pre_log = build_preprocessor(name, X_train, scale=True)  # LR needs scaling
    models = {
        "Dummy (prior)": attach(build_preprocessor(name, X_train, scale=False),
                                dummy_prior()),
        "Basic CART": attach(build_preprocessor(name, X_train, scale=False),
                             basic_cart()),
        "Tuned and pruned CART": pruned_pipe,
        "Random Forest": rf_pipe,
        "Logistic Regression": attach(pre_log, logistic_regression()),
    }
    cv_table = cv_summary_table(name, models, X_train, y_train, cv)
    log(cv_table.to_string(index=False,
        float_format=lambda x: f"{x:.4f}"))
    best_name = cv_table.iloc[0]["Model"]
    best_pipe = models[best_name]
    log(f"\n      Best by CV ROC-AUC: {best_name}")

    log("\n[6/8] Out-of-fold threshold selection (declared sensitivity target)")
    oof = cross_val_predict(best_pipe, X_train, y_train, cv=cv,
                            method="predict_proba", n_jobs=4)[:, 1]
    tdf = threshold_curve(oof, y_train)
    threshold, satisfied = select_threshold(tdf, TARGET_SENSITIVITY)
    log(f"      chosen threshold = {threshold:.3f} "
        f"(satisfied sensitivity >= {satisfied:.2f})")

    log("\n[7/8] Lock the model on ALL training data; evaluate the test set ONCE")
    final_pipe = best_pipe.fit(X_train, y_train)
    test_prob = final_pipe.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_binary(y_test, test_prob, threshold)
    # also score at the default 0.50 for the required threshold ablation
    default_metrics = evaluate_binary(y_test, test_prob, 0.50)
    log(f"      threshold={threshold:.3f}  "
        f"sens={test_metrics['sensitivity']:.3f}  "
        f"spec={test_metrics['specificity']:.3f}  "
        f"roc_auc={test_metrics['roc_auc']:.3f}  "
        f"pr_auc={test_metrics['pr_auc']:.3f}  "
        f"F1={test_metrics['f1']:.3f}  "
        f"FN={test_metrics['FN']}  FP={test_metrics['FP']}")
    log(f"      (ablation thr=0.50) sens={default_metrics['sensitivity']:.3f}  "
        f"spec={default_metrics['specificity']:.3f}  "
        f"FN={default_metrics['FN']}  FP={default_metrics['FP']}")

    log("\n[8/8] Splice individual model test metrics for the registry")
    all_rows = []
    for mname, pipe in models.items():
        # cross_validate clones estimators, so the originals here are unfitted -
        # fit each one fresh on the train split before scoring on the test set,
        # and store the FITTED pipeline back so the notebooks / CLI / GUI can
        # reuse it (and save_artifacts serialises a ready-to-predict pipeline).
        fitter = pipe.fit(X_train, y_train)
        models[mname] = fitter
        p = fitter.predict_proba(X_test)[:, 1] if hasattr(fitter, "predict_proba") \
            else np.where(fitter.predict(X_test) == 1, 1.0, 0.0)
        all_rows.append({"Model": mname, **evaluate_binary(y_test, p, threshold)})

    # tree complexity for basic vs pruned CART (interpretability evidence)
    complexity = {}
    for label, key in (("Basic CART", "Basic CART"),
                       ("Tuned and pruned CART", "Tuned and pruned CART")):
        pipe = models.get(key)
        if pipe is None:
            continue
        inner = pipe.named_steps.get("model") if hasattr(pipe, "named_steps") else pipe
        if hasattr(inner, "get_depth"):
            complexity[label] = {
                "depth": int(inner.get_depth()),
                "leaves": int(inner.get_n_leaves()),
                "nodes": int(inner.tree_.node_count),
                "criterion": getattr(inner, "criterion", "?"),
                "ccp_alpha": float(getattr(inner, "ccp_alpha", 0.0)),
            }

    state = {
        "spec": s,
        "X": X, "y": y,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "cv": cv,
        "models": models,
        "pruned_gs1": gs1, "pruned_gs2": gs2, "pruned_params": pruned_params,
        "rf_gs": rf_gs, "rf_params": rf_params,
        "cv_table": cv_table,
        "oof_prob": oof, "threshold_df": tdf,
        "threshold": threshold, "threshold_target_satisfied": satisfied,
        "best_name": best_name, "best_pipe": final_pipe,
        "test_metrics": test_metrics, "default_threshold_metrics": default_metrics,
        "all_test_rows": all_rows,
        "test_prob": test_prob, "test_pred": (test_prob >= threshold).astype(int),
        "feature_cols": s.feature_cols, "class_names": s.class_names,
        "numeric_cols": numeric_categorical(name, X)[0],
        "categorical_cols": numeric_categorical(name, X)[1],
        "complexity": complexity,
    }
    return state


# ---- PERSISTENCE -----------------------------------------------------------
def _dataclass_to_dict(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dataclass_to_dict(getattr(obj, k))
                for k in obj.__dataclass_fields__}
    if isinstance(obj, (tuple, list)):
        return [_dataclass_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def save_artifacts(name: str, state: dict, tag: str | None = None) -> dict:
    """Serialise everything the lab requires: the final pipeline (imputer/encoder/
    estimator) so raw inputs predict without re-fitting, the threshold, every
    individual model, the CV + test results CSVs, and a metadata JSON."""
    os.makedirs(ART_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    s = state["spec"]
    tag = tag or name
    paths = {}

    def _p(fn):
        paths.setdefault("artifacts", ART_DIR)
        return os.path.join(ART_DIR, fn)

    best_pipe = state["best_pipe"]
    joblib.dump(best_pipe, _p(f"{tag}_best_model.joblib"))
    joblib.dump(state["threshold"], _p(f"{tag}_threshold.joblib"))

    # the fitted final estimator alone (for tree visualisation / path tracing)
    try:
        inner = best_pipe.named_steps.get("model") if hasattr(best_pipe, "named_steps") else None
        if inner is not None:
            joblib.dump(inner, _p(f"{tag}_best_inner_estimator.joblib"))
    except Exception:
        pass

    # individual model pipelines (so `list` / `--model NAME` can load them).
    # NB: state["models"] holds FITTED pipelines (set in train_dataset step 7),
    # so every saved file is ready to predict without re-fitting.
    for mname, pipe in state["models"].items():
        slug = mname.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(pipe, _p(f"{tag}_{slug}.joblib"))

    # results CSVs
    state["cv_table"].to_csv(_p(f"{tag}_cv_results.csv"), index=False)
    pd.DataFrame(state["all_test_rows"]).to_csv(
        _p(f"{tag}_test_results.csv"), index=False)
    pd.DataFrame([state["test_metrics"]]).to_csv(
        _p(f"{tag}_final_test_metrics.csv"), index=False)

    # threshold curve + pruned-tuning CV tables
    state["threshold_df"].to_csv(_p(f"{tag}_threshold_curve.csv"), index=False)
    try:
        pd.DataFrame(state["pruned_gs2"].cv_results_).to_csv(
            _p(f"{tag}_pruning_cv.csv"), index=False)
    except Exception:
        pass

    # metadata JSON (everything needed to predict or reproduce)
    meta = {
        "dataset_key": name,
        "dataset_display_name": s.display_name,
        "uci_id": s.ucid,
        "doi": s.doi,
        "source_note": s.source_note,
        "target_col": s.target_col,
        "positive_class": s.positive_label,
        "negative_class": s.negative_label,
        "class_names": list(s.class_names),
        "positive_raw_values": [str(v) for v in s.positive_raw_values],
        "feature_cols": state["feature_cols"],
        "numeric_cols": state["numeric_cols"],
        "categorical_cols": state["categorical_cols"],
        "category_codes": s.category_codes,
        "category_codes_serialisable": {
            k: ({kk: str(vv) for kk, vv in v.items()}
                if isinstance(v, dict) else list(v))
            for k, v in s.category_codes.items()
        },
        "observation_unit": s.observation_unit,
        "population_note": s.population_note,
        "n_rows": int(len(state["X"])),
        "n_train": int(len(state["X_train"])),
        "n_test": int(len(state["X_test"])),
        "train_prevalence": float(state["y_train"].mean()),
        "test_prevalence": float(state["y_test"].mean()),
        "seed": SEED,
        "cv_folds": CV_FOLDS,
        "test_size": TEST_SIZE,
        "split": "Stratified train_test_split (test_size=0.20, seed=42)",
        "best_model": state["best_name"],
        "threshold": float(state["threshold"]),
        "threshold_target": float(TARGET_SENSITIVITY),
        "threshold_target_satisfied": float(state["threshold_target_satisfied"]),
        "test_metrics": {k: (float(v) if isinstance(v, (np.floating, np.integer))
                             else (int(v) if isinstance(v, (int,)) else v))
                         for k, v in state["test_metrics"].items()},
        "pruned_params": {k: (str(v) if v is None else v)
                          for k, v in state["pruned_params"].items()},
        "rf_params": {k: (str(v) if v is None else v)
                      for k, v in state.get("rf_params", {}).items()},
        "complexity": state.get("complexity", {}),
        "default_threshold_metrics": {
            k: (float(v) if isinstance(v, (np.floating, np.integer, float, int)) else v)
            for k, v in state.get("default_threshold_metrics", {}).items()
        },
        "cost_fn": COST_FN, "cost_fp": COST_FP,
        "software": {
            "python": platform.python_version(),
            "scikit_learn": _SK_VERSION,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": _JB_VERSION,
            "xgboost": _XGB_VERSION,
        },
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "intended_use": ("Educational predictive-analytics laboratory prototype "
                         "for MDI3003 Lab 02 only."),
        "prohibited_use": ("Clinical diagnosis, treatment, triage, screening, or "
                           "any patient-management decision."),
    }
    with open(_p(f"{tag}_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    paths.update({"tag": tag, "metadata": _p(f"{tag}_metadata.json")})
    return paths


# ---- LOADING (for CLI / GUI predict) ---------------------------------------
def load_best_pipeline(name: str, tag: str | None = None) -> "object":
    return joblib.load(os.path.join(ART_DIR, f"{tag or name}_best_model.joblib"))


def load_threshold(name: str, tag: str | None = None) -> float:
    p = os.path.join(ART_DIR, f"{tag or name}_threshold.joblib")
    if os.path.exists(p):
        return float(joblib.load(p))
    meta = load_metadata(name, tag)
    return float(meta.get("threshold", 0.5))


def load_metadata(name: str, tag: str | None = None) -> dict:
    p = os.path.join(ART_DIR, f"{tag or name}_metadata.json")
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_model(name: str, model_name: str | None = None, tag: str | None = None):
    """Return (pipeline or estimator-pipeline, display_name, threshold).

    Best model by default; if model_name is given, load the matching individual
    pipeline (matches the naming convention in save_artifacts)."""
    if not model_name:
        return load_best_pipeline(name, tag), (model_name or "best"), load_threshold(name, tag)
    # named model load path
    slug = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    cand = os.path.join(ART_DIR, f"{tag or name}_{slug}.joblib")
    if not os.path.exists(cand):
        raise FileNotFoundError(
            f"No model named '{model_name}' for dataset '{name}'.\n"
            f"Looked for: {cand}\nList available models with `meddiag_cli list {name}`.")
    return joblib.load(cand), model_name, load_threshold(name, tag)


# placeholder closure fixups (keep module importable even if parts moved)
_ = load_best_pipeline


# ---- MATPLOTLIB STYLE HELPER (shared by notebooks + figure generator) ------
def init_plot_style():
    """Apply the dataviz-skill validated style: recessive grid/axes, the
    blue/red diverging poles for the disease state, fixed-order categorical
    palette for any multi-series chart, default figure size + DPI."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    base = {
        "figure.figsize": (9, 5.2),
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.edgecolor": PALETTE["baseline"],
        "axes.labelcolor": PALETTE["ink_primary"],
        "axes.titlecolor": PALETTE["ink_primary"],
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": PALETTE["gridline"],
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": PALETTE["ink_secondary"],
        "ytick.color": PALETTE["ink_secondary"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": PALETTE["surface"],
        "axes.facecolor": PALETTE["surface"],
        "figure.facecolor": PALETTE["surface"],
        "legend.frameon": False,
        "legend.fontsize": 9,
    }
    mpl.rcParams.update(base)
    # Register seamap-free diverging pole mapping for confusion matrices etc.
    if "medposneg" not in mpl.colormaps:
        from matplotlib.colors import LinearSegmentedColormap
        mpl.colormaps.register(
            LinearSegmentedColormap.from_list(
                "medposneg", [PALETTE["neg"], "#ffffff", PALETTE["pos"]]),
            force=True)
    return plt


def save_fig(plt, name: str, force_show: bool = False):
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.savefig(path)
    return path
