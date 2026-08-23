"""
Customer Segmentation & Bank Marketing Prediction Engine
=========================================================
Script for loading trained pipelines/models, performing isolated inference on user
profiles (with confidence scoring & review triggers), and generating evaluation metrics
on locked test datasets.

Supported Datasets:
  1. 'janata' (JanataHack Multiclass Customer Segmentation: A, B, C, D)
  2. 'bank_marketing' / 'uci' (UCI Bank Marketing Binary Campaign Subscription: yes / no)

Usage Examples:
  # 1. Run evaluation metrics on test dataset:
  python test_script.py --dataset janata --mode evaluate
  python test_script.py --dataset bank_marketing --model fitted_CategoricalNB --mode evaluate

  # 2. Run isolated inference using built-in synthetic sample:
  python test_script.py --dataset janata --mode predict
  python test_script.py --dataset bank_marketing --mode predict

  # 3. Run isolated inference on custom JSON payload:
  python test_script.py --dataset janata --mode predict --input-json '{"Age": 45, "Gender": "Female", "Ever_Married": "Yes", "Graduated": "Yes", "Profession": "Artist", "Spending_Score": "Average", "Family_Size": 3, "Var_1": "Cat_6", "Work_Experience": 5}'

  # 4. Run both evaluation and sample inference:
  python test_script.py --dataset janata --mode both
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import OrdinalEncoder

# ==============================================================================
# CUSTOM TRANSFORMER DEFINITION & PICKLE REGISTRATION
# ==============================================================================

class SafeOrdinalToNonNegative(BaseEstimator, TransformerMixin):
    """
    Safely encode categorical columns to non-negative integers with reserved code for unknown values.
    Required for unpickling and running saved CategoricalNB pipelines from both datasets.
    """
    def __init__(self, unknown_code: int = 0):
        self.unknown_code = unknown_code
        self.enc = None
        self.encoders = {}

    def fit(self, X, y=None):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_df = pd.DataFrame(X)
            self.encoders = {}
            for col in X_df.columns:
                clean_vals = sorted(X_df[col].dropna().unique())
                self.encoders[col] = {val: i + 1 for i, val in enumerate(clean_vals)}
        self.enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.enc.fit(X)
        return self

    def transform(self, X):
        if hasattr(self, "encoders") and self.encoders and isinstance(X, (pd.DataFrame, pd.Series)):
            X_df = pd.DataFrame(X).copy()
            for col in X_df.columns:
                if col in self.encoders:
                    mapping = self.encoders[col]
                    X_df[col] = X_df[col].map(mapping).fillna(self.unknown_code).astype(int)
            return X_df.values

        if self.enc is not None:
            encoded = self.enc.transform(X).astype(int)
            return encoded + 1

        return X


# Register into sys.modules['__main__'] to guarantee unpickling success
if "__main__" in sys.modules:
    setattr(sys.modules["__main__"], "SafeOrdinalToNonNegative", SafeOrdinalToNonNegative)


# ==============================================================================
# CONFIGURATION & PATH SETUP
# ==============================================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset Configurations & Defaults
DATASET_CONFIGS = {
    "janata": {
        "name": "JanataHack Customer Segmentation (Multiclass: A, B, C, D)",
        "default_model": "selected_pipeline",
        "available_models": [
            "selected_pipeline",
            "preprocessor",
        ],
        "target_col": "Segmentation",
        "classes": ["A", "B", "C", "D"],
        "train_features": PROCESSED_DIR / "janata" / "X_train.csv",
        "train_target": PROCESSED_DIR / "janata" / "y_train.csv",
        "test_features": PROCESSED_DIR / "janata" / "X_test.csv",
        "test_target": PROCESSED_DIR / "janata" / "y_test.csv",
        "numeric_cols": ["Age", "Family_Size", "Work_Experience_Cleaned"],
        "categorical_cols": [
            "Gender",
            "Ever_Married",
            "Graduated",
            "Profession",
            "Spending_Score",
            "Var_1",
        ],
        "default_sample": {
            "Gender": "Male",
            "Ever_Married": "Yes",
            "Age": 38,
            "Graduated": "Yes",
            "Profession": "Executive",
            "Work_Experience": 4.0,
            "Spending_Score": "High",
            "Family_Size": 3.0,
            "Var_1": "Cat_6",
        },
        "review_threshold": 0.75,
    },
    "bank_marketing": {
        "name": "UCI Bank Marketing Campaign Response (Binary: no, yes)",
        "default_model": "fitted_CategoricalNB",
        "available_models": [
            "fitted_CategoricalNB",
            "fitted_LogisticRegression",
            "pipeline_categorical",
            "pipeline_bernoulli",
            "pipeline_gaussian",
            "pipeline_dummy",
        ],
        "target_col": "y",
        "classes": ["no", "yes"],
        "train_features": PROCESSED_DIR / "X_train.csv",
        "train_target": PROCESSED_DIR / "y_train.csv",
        "test_features": PROCESSED_DIR / "X_test.csv",
        "test_target": PROCESSED_DIR / "y_test.csv",
        "numeric_cols": ["age", "balance", "day", "campaign", "pdays", "previous"],
        "categorical_cols": [
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "poutcome",
        ],
        "excluded_cols": ["duration", "row_index", "y"],
        "default_sample": {
            "age": 42,
            "job": "technician",
            "marital": "married",
            "education": "secondary",
            "default": "no",
            "balance": 1500,
            "housing": "no",
            "loan": "no",
            "contact": "cellular",
            "day": 15,
            "month": "aug",
            "campaign": 2,
            "pdays": -1,
            "previous": 0,
            "poutcome": "nonexistent",
        },
        "review_threshold": 0.75,
    },
}

# Alias for bank marketing
DATASET_CONFIGS["uci"] = DATASET_CONFIGS["bank_marketing"]
DATASET_CONFIGS["bank"] = DATASET_CONFIGS["bank_marketing"]


# ==============================================================================
# MODEL & DATASET LOADER UTILITIES
# ==============================================================================

def normalize_dataset_key(dataset_key: str) -> str:
    """Normalize input dataset name to canonical key."""
    key = dataset_key.strip().lower()
    if key in ["janata", "janatahacks", "customer_segmentation", "segmentation"]:
        return "janata"
    elif key in ["bank", "uci", "bank_marketing", "marketing", "uci_bank"]:
        return "bank_marketing"
    else:
        raise ValueError(
            f"Unknown dataset '{dataset_key}'. Supported datasets: 'janata', 'bank_marketing' (or 'uci')."
        )


def list_available_models(dataset_key: str) -> List[str]:
    """List available model artifacts for a given dataset."""
    canonical_key = normalize_dataset_key(dataset_key)
    return DATASET_CONFIGS[canonical_key]["available_models"]


def load_model_pipeline(dataset_key: str, model_name: Optional[str] = None) -> Any:
    """
    Fetch and deserialize the requested model pipeline from the models directory.
    """
    canonical_key = normalize_dataset_key(dataset_key)
    cfg = DATASET_CONFIGS[canonical_key]

    if not model_name:
        model_name = cfg["default_model"]

    # Normalize file extension
    model_filename = model_name if model_name.endswith(".joblib") else f"{model_name}.joblib"
    model_path = MODELS_DIR / model_filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact '{model_filename}' not found in {MODELS_DIR}.\n"
            f"Available models for '{canonical_key}': {cfg['available_models']}"
        )

    print(f"\n[+] Loading model pipeline: {model_path.name}")
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except Exception as exc:
        raise RuntimeError(f"Failed to deserialize model pipeline from {model_path}: {exc}") from exc


def load_dataset_splits(dataset_key: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load train and test feature matrices and target labels for a dataset.
    """
    canonical_key = normalize_dataset_key(dataset_key)
    cfg = DATASET_CONFIGS[canonical_key]

    if not cfg["test_features"].exists() or not cfg["test_target"].exists():
        raise FileNotFoundError(
            f"Processed data files for '{canonical_key}' not found under {PROCESSED_DIR}.\n"
            f"Expected: {cfg['test_features']} and {cfg['test_target']}"
        )

    X_train = pd.read_csv(cfg["train_features"])
    y_train = pd.read_csv(cfg["train_target"]).squeeze("columns")

    X_test = pd.read_csv(cfg["test_features"])
    y_test = pd.read_csv(cfg["test_target"]).squeeze("columns")

    return X_train, y_train, X_test, y_test


# ==============================================================================
# ISOLATED INFERENCE & PREDICTION ENGINE
# ==============================================================================

def validate_and_format_janata_input(
    raw_data: Union[Dict[str, Any], pd.DataFrame, pd.Series],
    X_train_ref: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Validate and format an isolated customer profile for the JanataHack model.
    Applies column alignment, Work_Experience_Cleaned derivation, and default imputation.
    """
    if isinstance(raw_data, dict):
        df = pd.DataFrame([raw_data])
    elif isinstance(raw_data, pd.Series):
        df = pd.DataFrame([raw_data.to_dict()])
    elif isinstance(raw_data, pd.DataFrame):
        df = raw_data.copy()
    else:
        raise TypeError("raw_data must be a dict, pandas Series, or pandas DataFrame.")

    model_expected_cols = [
        "Gender",
        "Ever_Married",
        "Age",
        "Graduated",
        "Profession",
        "Spending_Score",
        "Family_Size",
        "Var_1",
        "Work_Experience_Cleaned",
    ]

    # Handle Work_Experience -> Work_Experience_Cleaned
    if "Work_Experience_Cleaned" not in df.columns and "Work_Experience" in df.columns:
        df["Work_Experience_Cleaned"] = df["Work_Experience"]
        if "Age" in df.columns:
            age_val = pd.to_numeric(df["Age"], errors="coerce")
            exp_val = pd.to_numeric(df["Work_Experience_Cleaned"], errors="coerce")
            mask = (exp_val == 0) & (age_val > 22)
            df.loc[mask, "Work_Experience_Cleaned"] = np.nan

    # Fill missing expected columns with reference training modes/medians if available
    for col in model_expected_cols:
        if col not in df.columns:
            if X_train_ref is not None and col in X_train_ref.columns:
                if pd.api.types.is_numeric_dtype(X_train_ref[col]):
                    df[col] = X_train_ref[col].median()
                else:
                    df[col] = X_train_ref[col].mode().iloc[0]
            else:
                df[col] = np.nan

    # Cast numeric columns
    for num_col in ["Age", "Family_Size", "Work_Experience_Cleaned"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    # Cast categorical columns as string (handling NaN)
    cat_cols = ["Gender", "Ever_Married", "Graduated", "Profession", "Spending_Score", "Var_1"]
    for cat_col in cat_cols:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str).replace("nan", np.nan).replace("None", np.nan)

    return df[model_expected_cols]


def validate_and_format_bank_input(
    raw_data: Union[Dict[str, Any], pd.DataFrame, pd.Series],
    X_train_ref: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Validate and format an isolated customer profile for UCI Bank Marketing model.
    Excludes leakage columns ('duration') and standardizes feature schema.
    """
    if isinstance(raw_data, dict):
        df = pd.DataFrame([raw_data])
    elif isinstance(raw_data, pd.Series):
        df = pd.DataFrame([raw_data.to_dict()])
    elif isinstance(raw_data, pd.DataFrame):
        df = raw_data.copy()
    else:
        raise TypeError("raw_data must be a dict, pandas Series, or pandas DataFrame.")

    # Drop leakage / target columns if accidentally passed in
    drop_cols = [c for c in ["duration", "row_index", "y"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    expected_cols = [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
    ]

    for col in expected_cols:
        if col not in df.columns:
            if X_train_ref is not None and col in X_train_ref.columns:
                if pd.api.types.is_numeric_dtype(X_train_ref[col]):
                    df[col] = X_train_ref[col].median()
                else:
                    df[col] = X_train_ref[col].mode().iloc[0]
            else:
                df[col] = np.nan

    num_cols = ["age", "balance", "day", "campaign", "pdays", "previous"]
    for num_col in num_cols:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    cat_cols = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
    ]
    for cat_col in cat_cols:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str).replace("nan", np.nan).replace("None", np.nan)

    return df[expected_cols]


def predict_single_profile(
    dataset_key: str,
    pipeline: Any,
    input_data: Union[Dict[str, Any], pd.DataFrame, pd.Series],
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Perform isolated inference on a single user profile.
    Returns predicted label, posterior probability distribution, confidence tier,
    and governance review recommendation.
    """
    canonical_key = normalize_dataset_key(dataset_key)
    cfg = DATASET_CONFIGS[canonical_key]
    thresh = threshold if threshold is not None else cfg["review_threshold"]

    # 1. Format input DataFrame based on dataset rules
    if canonical_key == "janata":
        formatted_df = validate_and_format_janata_input(input_data)
    else:
        formatted_df = validate_and_format_bank_input(input_data)

    # 2. Predict class label
    raw_pred = pipeline.predict(formatted_df)[0]
    predicted_label = str(raw_pred)

    # 3. Compute Posterior Probabilities & Confidence
    probabilities = {}
    max_confidence = 1.0
    has_proba = hasattr(pipeline, "predict_proba")

    if has_proba:
        try:
            proba_arr = pipeline.predict_proba(formatted_df)[0]
            classes = pipeline.classes_
            probabilities = {str(cls_name): float(prob) for cls_name, prob in zip(classes, proba_arr)}
            max_confidence = float(np.max(proba_arr))
        except Exception:
            probabilities = {predicted_label: 1.0}
            max_confidence = 1.0

    # 4. Derive Confidence Tier & Actionable Recommendation
    if max_confidence >= thresh:
        confidence_tier = "HIGH"
        review_recommendation = "Automated processing approved (Prediction meets confidence criteria)."
    elif max_confidence >= 0.50:
        confidence_tier = "MODERATE"
        review_recommendation = (
            f"Advisory check recommended: Confidence ({max_confidence:.1%}) is below threshold ({thresh:.1%})."
        )
    else:
        confidence_tier = "LOW"
        review_recommendation = (
            f"MANUAL REVIEW MANDATORY: High uncertainty detected ({max_confidence:.1%}). Fallback to specialist."
        )

    return {
        "dataset": canonical_key,
        "input_features": formatted_df.iloc[0].to_dict(),
        "predicted_label": predicted_label,
        "confidence": max_confidence,
        "confidence_tier": confidence_tier,
        "posterior_distribution": probabilities,
        "threshold": thresh,
        "review_recommendation": review_recommendation,
    }


# ==============================================================================
# EVALUATION ENGINE (METRICS REPORTING)
# ==============================================================================

def evaluate_pipeline_on_test_set(
    dataset_key: str,
    pipeline: Any,
    model_name: str = "Pipeline",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a fitted pipeline against the locked test split.
    Calculates Accuracy, Macro F1, Weighted F1, Per-Class Precision/Recall,
    and Confusion Matrix.
    """
    canonical_key = normalize_dataset_key(dataset_key)
    cfg = DATASET_CONFIGS[canonical_key]

    _, _, X_test, y_test = load_dataset_splits(canonical_key)

    # Align columns
    if canonical_key == "janata":
        X_test_eval = validate_and_format_janata_input(X_test)
    else:
        X_test_eval = validate_and_format_bank_input(X_test)

    # Predict
    y_pred = pipeline.predict(X_test_eval)

    # Normalize target type comparison if necessary
    y_test_str = y_test.astype(str)
    y_pred_str = pd.Series(y_pred).astype(str)

    # Compute Core Metrics
    acc = accuracy_score(y_test_str, y_pred_str)
    macro_f1 = f1_score(y_test_str, y_pred_str, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test_str, y_pred_str, average="weighted", zero_division=0)
    macro_prec = precision_score(y_test_str, y_pred_str, average="macro", zero_division=0)
    macro_rec = recall_score(y_test_str, y_pred_str, average="macro", zero_division=0)

    # Classes in pipeline
    classes_str = [str(c) for c in getattr(pipeline, "classes_", sorted(y_test_str.unique()))]

    # Per-class report
    cls_report = classification_report(y_test_str, y_pred_str, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test_str, y_pred_str, labels=classes_str)

    results = {
        "dataset": canonical_key,
        "model_name": model_name,
        "sample_count": len(y_test),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "classification_report": cls_report,
        "confusion_matrix": cm.tolist(),
        "classes": classes_str,
    }

    if verbose:
        print_evaluation_dashboard(results)

    return results


def print_evaluation_dashboard(results: Dict[str, Any]) -> None:
    """Print a clean, visually structured ASCII metric dashboard to stdout."""
    title = f"MODEL EVALUATION REPORT: {results['model_name']} ({results['dataset'].upper()})"
    banner = "=" * max(65, len(title))
    print(f"\n{banner}\n{title}\n{banner}")
    print(f"Total Test Samples : {results['sample_count']}")
    print(f"Overall Accuracy   : {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)")
    print(f"Macro F1 Score     : {results['macro_f1']:.4f}")
    print(f"Weighted F1 Score  : {results['weighted_f1']:.4f}")
    print(f"Macro Precision    : {results['macro_precision']:.4f}")
    print(f"Macro Recall       : {results['macro_recall']:.4f}")
    print("-" * len(banner))

    print("\nPER-CLASS CLASSIFICATION BREAKDOWN:")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 58)

    for cls_name in results["classes"]:
        if cls_name in results["classification_report"]:
            m = results["classification_report"][cls_name]
            print(
                f"{cls_name:<12} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1-score']:<12.4f} {int(m['support']):<10}"
            )
    print("-" * 58)

    print("\nCONFUSION MATRIX:")
    classes = results["classes"]
    header = "True \\ Pred".ljust(14) + "".join([f"{c:>10}" for c in classes])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(results["confusion_matrix"]):
        row_str = f"{classes[i]:<14}" + "".join([f"{val:>10}" for val in row])
        print(row_str)
    print(f"{banner}\n")


def print_prediction_dashboard(pred_result: Dict[str, Any]) -> None:
    """Print an isolated inference prediction result with confidence distribution."""
    title = f"ISOLATED INFERENCE RESULT ({pred_result['dataset'].upper()})"
    banner = "=" * max(65, len(title))
    print(f"\n{banner}\n{title}\n{banner}")
    print(f"Predicted Target / Segment : >>> {pred_result['predicted_label']} <<<")
    print(f"Confidence Level           : {pred_result['confidence']:.2%} [{pred_result['confidence_tier']}]")
    print(f"Decision Threshold         : {pred_result['threshold']:.2%}")
    print(f"Governance Recommendation  : {pred_result['review_recommendation']}")
    print("-" * len(banner))

    print("\nPOSTERIOR CLASS PROBABILITIES:")
    for cls_name, prob in pred_result["posterior_distribution"].items():
        bar_len = int(prob * 30)
        bar = "#" * bar_len + "-" * (30 - bar_len)
        print(f"  Class {cls_name:<10}: {prob:>6.2%} |{bar}|")
    print("-" * len(banner))

    print("\nINPUT FEATURE PROFILE:")
    for feature, val in pred_result["input_features"].items():
        print(f"  {feature:<26}: {val}")
    print(f"{banner}\n")


# ==============================================================================
# CLI DISPATCHER & MAIN ENTRY POINT
# ==============================================================================

def build_cli_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Customer Segmentation & Bank Marketing Prediction Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_script.py --dataset janata --mode evaluate
  python test_script.py --dataset bank_marketing --model fitted_CategoricalNB --mode evaluate
  python test_script.py --dataset janata --mode predict
  python test_script.py --dataset janata --mode predict --input-json '{"Age": 32, "Gender": "Female"}'
  python test_script.py --dataset janata --mode both
        """,
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="janata",
        choices=["janata", "bank_marketing", "uci", "bank"],
        help="Dataset selection: 'janata' (multiclass segmentation) or 'bank_marketing' (binary response).",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Specific model artifact to load (e.g., 'selected_pipeline', 'fitted_CategoricalNB', 'fitted_LogisticRegression').",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["evaluate", "predict", "both"],
        help="Execution mode: 'evaluate' (test dataset metrics), 'predict' (isolated inference), or 'both'.",
    )

    parser.add_argument(
        "--input-json",
        type=str,
        default=None,
        help="JSON string representing a single customer profile for inference.",
    )

    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to CSV/JSON file containing customer profile(s) for batch/single inference.",
    )

    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=None,
        help="Confidence threshold for automated decision vs manual review trigger (default: 0.75).",
    )

    parser.add_argument(
        "--save-output",
        type=str,
        default=None,
        help="Optional file path to save output JSON results.",
    )

    return parser


def parse_json_or_dict_string(input_str: str) -> Dict[str, Any]:
    """Parse JSON string, with fallback to ast.literal_eval for Windows/PowerShell compatibility."""
    import ast

    try:
        return json.loads(input_str)
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(input_str)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    try:
        return json.loads(input_str.replace("'", '"'))
    except Exception as exc:
        raise ValueError(
            f"Could not parse input string '{input_str}'. Expected JSON or Python dict format."
        ) from exc


def main() -> None:
    """Main execution dispatcher."""
    parser = build_cli_parser()
    args = parser.parse_args()

    canonical_dataset = normalize_dataset_key(args.dataset)
    cfg = DATASET_CONFIGS[canonical_dataset]

    model_to_load = args.model if args.model else cfg["default_model"]

    print("\n" + "=" * 65)
    print("CUSTOMER SEGMENTATION & PREDICTIVE ANALYTICS ENGINE")
    print(f"Dataset Selected : {cfg['name']}")
    print(f"Model Pipeline   : {model_to_load}")
    print(f"Execution Mode   : {args.mode.upper()}")
    print("=" * 65)

    # 1. Load pipeline
    pipeline = load_model_pipeline(canonical_dataset, model_to_load)

    output_payload: Dict[str, Any] = {
        "dataset": canonical_dataset,
        "model": model_to_load,
    }

    # 2. Evaluation Mode
    if args.mode in ["evaluate", "both"]:
        metrics = evaluate_pipeline_on_test_set(
            dataset_key=canonical_dataset,
            pipeline=pipeline,
            model_name=model_to_load,
            verbose=True,
        )
        output_payload["evaluation_metrics"] = metrics

    # 3. Prediction Mode (Isolated Inference)
    if args.mode in ["predict", "both"]:
        if args.input_json:
            try:
                user_profile = parse_json_or_dict_string(args.input_json)
            except Exception as err:
                print(f"[!] Error parsing --input-json: {err}")
                sys.exit(1)
        elif args.input_file:
            input_file_path = Path(args.input_file)
            if not input_file_path.exists():
                print(f"[!] Input file '{args.input_file}' not found.")
                sys.exit(1)
            if input_file_path.suffix.lower() == ".json":
                with open(input_file_path, "r", encoding="utf-8") as f:
                    user_profile = json.load(f)
            else:
                user_df = pd.read_csv(input_file_path)
                user_profile = user_df.iloc[0].to_dict()
        else:
            user_profile = cfg["default_sample"]
            print("[*] Using standard synthetic profile for isolated inference demonstration.")

        prediction = predict_single_profile(
            dataset_key=canonical_dataset,
            pipeline=pipeline,
            input_data=user_profile,
            threshold=args.threshold,
        )
        print_prediction_dashboard(prediction)
        output_payload["isolated_inference"] = prediction

    # 4. Optional Output File Saving
    if args.save_output:
        save_path = Path(args.save_output)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)
        print(f"[+] Complete output results saved to: {save_path.resolve()}")


if __name__ == "__main__":
    main()
