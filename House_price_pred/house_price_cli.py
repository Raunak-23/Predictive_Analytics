#!/usr/bin/env python3
"""
House Price Prediction & Valuation - CLI Tool

A command-line interface for loading trained models, making predictions on
new data, retraining models on separate data, and managing model artifacts
for both the UCI Real Estate Valuation and Ames Housing datasets.

Usage:
  python house_price_cli.py predict --dataset uci --input sample.json
  python house_price_cli.py predict --dataset ames --input sample.csv
  python house_price_cli.py retrain --dataset uci --data new_data.csv
  python house_price_cli.py list-models --dataset uci
  python house_price_cli.py interactive --dataset ames
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")

# UCI dataset constants
UCI_FEATURES = [
    "X1 transaction date",
    "X2 house age",
    "X3 distance to the nearest MRT station",
    "X4 number of convenience stores",
    "X5 latitude",
    "X6 longitude",
]
UCI_TARGET = "Y house price of unit area"
UCI_TARGET_UNIT = "10,000 NTD/Ping"

# Ames dataset constants
AMES_TARGET = "SalePrice"
AMES_TARGET_UNIT = "USD"

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def _resolve_artifact_path(filename: str) -> str:
    """Return the absolute path of an artifact file."""
    return os.path.join(ARTIFACTS_DIR, filename)


def _load_model(dataset: str) -> tuple:
    """
    Load the best model pipeline and metadata for a dataset.

    Parameters
    ----------
    dataset : str
        'uci' or 'ames'

    Returns
    -------
    model : Pipeline  (fitted sklearn pipeline with preprocess + estimator)
    metadata : dict   (dataset-specific metadata)
    """
    if dataset == "uci":
        model_path = _resolve_artifact_path("uci_best_model.joblib")
        meta_path = _resolve_artifact_path("uci_training_metadata.json")
    elif dataset == "ames":
        model_path = _resolve_artifact_path("ames_best_model.joblib")
        meta_path = _resolve_artifact_path("ames_metadata.json")
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'uci' or 'ames'.")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Best model not found at '{model_path}'. "
            f"Please run the notebook first to generate artifacts."
        )

    import joblib

    model = joblib.load(model_path)
    metadata = None
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
    return model, metadata


def _load_preprocessor(dataset: str):
    """
    Load the saved preprocessing pipeline (ColumnTransformer) for a dataset.

    Parameters
    ----------
    dataset : str
        'uci' or 'ames'

    Returns
    -------
    preprocessor : ColumnTransformer
    """
    if dataset == "uci":
        path = _resolve_artifact_path("uci_preprocessing_pipeline.joblib")
    elif dataset == "ames":
        path = _resolve_artifact_path("ames_preprocessor.joblib")
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'uci' or 'ames'.")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Preprocessor not found at '{path}'. "
            f"Please run the notebook first to generate artifacts."
        )

    import joblib

    return joblib.load(path)


def _load_model_registry(dataset: str) -> dict:
    """Load the model registry JSON for UCI dataset."""
    if dataset != "uci":
        return {}
    path = _resolve_artifact_path("uci_model_registry.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _get_feature_info(dataset: str) -> dict:
    """
    Return information about the dataset features.
    """
    if dataset == "uci":
        return {
            "features": UCI_FEATURES,
            "target": UCI_TARGET,
            "target_unit": UCI_TARGET_UNIT,
            "feature_descriptions": {
                "X1 transaction date": "Transaction date (e.g., 2013.250 = 2013 March)",
                "X2 house age": "Age of the house in years",
                "X3 distance to the nearest MRT station": "Distance to nearest MRT station (meters)",
                "X4 number of convenience stores": "Number of convenience stores in walking distance",
                "X5 latitude": "Latitude coordinate (degrees)",
                "X6 longitude": "Longitude coordinate (degrees)",
            },
            "feature_types": {f: "float" for f in UCI_FEATURES},
        }
    elif dataset == "ames":
        return {
            "features": "79 features (see AmesHousing.csv for full schema)",
            "target": AMES_TARGET,
            "target_unit": AMES_TARGET_UNIT,
            "feature_descriptions": "Refer to the Ames Housing dataset documentation.",
            "feature_types": "Mixed numeric/categorical (79 columns)",
        }
    else:
        raise ValueError(f"Unknown dataset '{dataset}'.")


def _load_data(data_source: str, dataset: str) -> pd.DataFrame:
    """
    Load new data from a JSON file, CSV file, or a raw JSON string.

    Parameters
    ----------
    data_source : str
        Path to a .json / .csv file, or a JSON string.
    dataset : str
        'uci' or 'ames'

    Returns
    -------
    pd.DataFrame
    """
    # If it's a file path
    if os.path.isfile(data_source):
        ext = os.path.splitext(data_source)[1].lower()
        if ext == ".json":
            with open(data_source, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # Could be { "features": [ {...}, ... ] } or single record
                if "features" in data:
                    return pd.DataFrame(data["features"])
                return pd.DataFrame([data])
        elif ext == ".csv":
            return pd.read_csv(data_source)
        else:
            raise ValueError(
                f"Unsupported file format '{ext}'. Use .json or .csv."
            )
    else:
        # Treat as raw JSON string
        data = json.loads(data_source)
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # Check if it contains multiple records under a key
            for key in ("features", "data", "records", "instances"):
                if key in data and isinstance(data[key], list):
                    return pd.DataFrame(data[key])
            return pd.DataFrame([data])
        else:
            raise ValueError("Invalid JSON format. Provide an object or array.")


def _validate_uci_features(df: pd.DataFrame):
    """Check that the UCI DataFrame has the required columns."""
    missing = [col for col in UCI_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required features for UCI dataset: {missing}\n"
            f"Required features: {UCI_FEATURES}"
        )


def _validate_ames_features(df: pd.DataFrame):
    """Check that the Ames DataFrame has the required columns (basic check)."""
    # Just check that it's not empty and has at least some numeric columns
    if df.empty:
        raise ValueError("Empty DataFrame provided for Ames dataset.")
    # The Ames model has 79 features - we just require that SalePrice is absent
    # (since inference data shouldn't have the target)
    if AMES_TARGET in df.columns:
        print(
            f"  [INFO] Input data contains '{AMES_TARGET}' column. "
            f"This column will be ignored for prediction."
        )


# ---------------------------------------------------------------------------
# PREDICT command
# ---------------------------------------------------------------------------
def cmd_predict(args):
    """
    Load the best model and predict on new data.
    """
    dataset = args.dataset
    data_source = args.input

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset.upper()}")
    print(f"  Input:   {data_source}")
    print(f"{'='*60}\n")

    # Load model
    print("[1/3] Loading best model pipeline...")
    model, metadata = _load_model(dataset)
    print(f"  Model loaded successfully.", end="")

    if metadata:
        if dataset == "uci":
            print(f"  Best: {metadata.get('best_model', {}).get('name', 'N/A')}")
        else:
            print(f"  Best: {metadata.get('best_model', 'N/A')}")
    else:
        print()

    # Load input data
    print("[2/3] Loading input data...")
    df = _load_data(data_source, dataset)

    if dataset == "uci":
        _validate_uci_features(df)
    else:
        _validate_ames_features(df)

    # Remove target column if present in inference data
    target_col = UCI_TARGET if dataset == "uci" else AMES_TARGET
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    print(f"  Loaded {len(df)} record(s) with {len(df.columns)} feature(s).")

    # Predict
    print("[3/3] Running inference...")
    predictions = model.predict(df)

    # Build output
    results = df.copy()
    pred_col = f"Predicted_{target_col.replace(' ', '_')}"
    results[pred_col] = predictions

    unit = UCI_TARGET_UNIT if dataset == "uci" else AMES_TARGET_UNIT
    print(f"\n  {'='*50}")
    print(f"  PREDICTION RESULTS ({unit})")
    print(f"  {'='*50}")
    for i, row in results.iterrows():
        print(f"\n  Record {i+1}:")
        if dataset == "uci":
            for feat in UCI_FEATURES:
                val = row.get(feat, "N/A")
                print(f"    {feat}: {val}")
        print(f"    >>> Predicted {target_col}: {row[pred_col]:.4f} {unit}")
        print(f"    {'-'*40}")

    # Save output if requested
    if args.output:
        output_path = args.output
        if output_path.endswith(".csv"):
            results.to_csv(output_path, index=False)
        else:
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "dataset": dataset,
                        "predictions": results.to_dict(orient="records"),
                        "unit": unit,
                    },
                    f,
                    indent=2,
                )
        print(f"\n  Results saved to '{output_path}'")

    return results


# ---------------------------------------------------------------------------
# RETRAIN command
# ---------------------------------------------------------------------------
def cmd_retrain(args):
    """
    Load the saved preprocessing pipeline and retrain all models on new data.
    Saves the best model and updated metadata.
    """
    dataset = args.dataset
    data_path = args.data

    print(f"\n{'='*60}")
    print(f"  RETRAINING: {dataset.upper()}")
    print(f"  Training data: {data_path}")
    print(f"{'='*60}\n")

    # Load the saved preprocessor
    print("[1/5] Loading saved preprocessing pipeline...")
    preprocessor = _load_preprocessor(dataset)
    print("  Preprocessor loaded.")

    # Load new training data
    print("[2/5] Loading and preparing training data...")
    df = _load_data(data_path, dataset)

    target_col = UCI_TARGET if dataset == "uci" else AMES_TARGET
    if target_col not in df.columns:
        raise ValueError(
            f"Training data must contain the target column '{target_col}'."
        )

    # Separate features and target
    if dataset == "uci":
        _validate_uci_features(df)
        X_new = df[UCI_FEATURES]
    else:
        X_new = df.drop(columns=[target_col])
    y_new = df[target_col]

    print(f"  Loaded {len(df)} training samples.")
    print(f"  Features: {X_new.shape[1]}, Target: {target_col}")

    # Import required models
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from xgboost import XGBRegressor

    SEED = 42

    # Build model pipelines using the loaded preprocessor
    print("[3/5] Building and training models...")
    model_configs = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.001, max_iter=20000),
        "Elastic Net": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=20000),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=SEED),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=SEED, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    trained_models = {}
    results_list = []

    for name, estimator in model_configs.items():
        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        print(f"  Training {name}...", end=" ")
        pipeline.fit(X_new, y_new)
        trained_models[name] = pipeline

        # Evaluate
        preds = pipeline.predict(X_new)
        rmse = np.sqrt(mean_squared_error(y_new, preds))
        mae = mean_absolute_error(y_new, preds)
        r2 = r2_score(y_new, preds)
        print(f"RMSE={rmse:.4f}, R²={r2:.4f}")

        results_list.append(
            {
                "Model": name,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
            }
        )

    results_df = pd.DataFrame(results_list).sort_values("RMSE")
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    print(f"\n  Best Model: {best_model_name}")
    print(f"    RMSE: {results_df.iloc[0]['RMSE']:.4f}")
    print(f"    R²:   {results_df.iloc[0]['R2']:.4f}")
    print(f"    MAE:  {results_df.iloc[0]['MAE']:.4f}")

    # Save artifacts
    print("\n[4/5] Saving retrained artifacts...")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Save best model
    best_model_path = _resolve_artifact_path(
        f"{dataset}_best_model.joblib"
    )
    import joblib

    joblib.dump(best_model, best_model_path)
    print(f"  Best model saved: '{best_model_path}'")

    # Save all individual models
    model_save_paths = {}
    for name, pipeline in trained_models.items():
        filename = (
            f"{dataset}_"
            f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
            f".joblib"
        )
        path = _resolve_artifact_path(filename)
        joblib.dump(pipeline, path)
        model_save_paths[name] = path

    # Save results CSV
    results_csv = _resolve_artifact_path(f"{dataset}_all_model_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"  Results saved: '{results_csv}'")

    # Save updated metadata
    print("[5/5] Saving updated metadata...")
    metadata = {
        "dataset": "UCI Real Estate Valuation" if dataset == "uci" else "Ames Housing",
        "target": target_col,
        "retrained_on": data_path,
        "retrain_date": datetime.now().isoformat(),
        "best_model": best_model_name,
        "best_metrics": {
            "RMSE": float(results_df.iloc[0]["RMSE"]),
            "R2": float(results_df.iloc[0]["R2"]),
            "MAE": float(results_df.iloc[0]["MAE"]),
        },
        "all_models": results_list,
    }
    meta_path = _resolve_artifact_path(
        f"{dataset}_training_metadata.json"
        if dataset == "uci"
        else f"{dataset}_metadata.json"
    )
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: '{meta_path}'")
    print(f"\n  {'='*50}")
    print(f"  RETRAINING COMPLETE")
    print(f"  {'='*50}")

    return trained_models, results_df


# ---------------------------------------------------------------------------
# LIST-MODELS command
# ---------------------------------------------------------------------------
def cmd_list_models(args):
    """
    List all available models and their performance metrics.
    """
    dataset = args.dataset

    print(f"\n{'='*60}")
    print(f"  MODEL REGISTRY: {dataset.upper()}")
    print(f"{'='*60}\n")

    # Load metadata
    if dataset == "uci":
        registry = _load_model_registry(dataset)
        meta_path = _resolve_artifact_path("uci_training_metadata.json")
    else:
        registry = {}
        meta_path = _resolve_artifact_path("ames_metadata.json")

    # Load results CSV if available
    results_csv_path = _resolve_artifact_path(
        f"{dataset}_all_model_results.csv"
    )
    if os.path.exists(results_csv_path):
        results_df = pd.read_csv(results_csv_path)
        print("  All Models Performance (sorted by RMSE):")
        print(f"  {'-'*60}")
        print(
            results_df.to_string(
                index=False, float_format=lambda x: f"{x:.4f}"
            )
        )
        print()
    else:
        print("  (No detailed results CSV found.)")

    # Show best model info
    print(f"\n  Best Model Info:")
    print(f"  {'-'*40}")

    if dataset == "uci":
        if registry and "best_model" in registry:
            bm = registry["best_model"]
            print(f"    Name:    {bm.get('name', 'N/A')}")
            print(f"    File:    {bm.get('file', 'N/A')}")
            metrics = bm.get("metrics", {})
            for k, v in metrics.items():
                print(f"    {k}: {v:.4f}")
    else:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        print(f"    Name:    {meta.get('best_model', 'N/A')}")
        bm = meta.get("best_metrics", {})
        for k, v in bm.items():
            print(f"    {k}: {v:.4f}")

    # Show tuned models (UCI only)
    if dataset == "uci" and registry and "tuned_models" in registry:
        print(f"\n  Tuned Model Configurations:")
        print(f"  {'-'*40}")
        for name, config in registry["tuned_models"].items():
            print(f"    {name}:")
            params = config.get("params", config.get("alpha", {}))
            if isinstance(params, dict):
                for pk, pv in params.items():
                    print(f"      {pk}: {pv}")
            else:
                print(f"      alpha/value: {params}")

    # Show available artifact files
    print(f"\n  Available Artifact Files:")
    print(f"  {'-'*40}")
    artifact_files = [
        f for f in os.listdir(ARTIFACTS_DIR) if f.startswith(dataset)
    ]
    for fname in sorted(artifact_files):
        fpath = os.path.join(ARTIFACTS_DIR, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {fname} ({size_kb:.1f} KB)")

    feat_info = _get_feature_info(dataset)
    print(f"\n  Feature Info:")
    print(f"  {'-'*40}")
    print(f"    Target: {feat_info['target']} ({feat_info['target_unit']})")
    if isinstance(feat_info["features"], list):
        print(f"    Features ({len(feat_info['features'])}):")
        for f in feat_info["features"]:
            desc = feat_info.get("feature_descriptions", {}).get(f, "")
            print(f"      - {f}")
            if desc:
                print(f"          {desc}")
    else:
        print(f"    Features: {feat_info['features']}")


# ---------------------------------------------------------------------------
# INTERACTIVE command
# ---------------------------------------------------------------------------
def cmd_interactive(args):
    """
    Interactive prompt to enter feature values and get predictions.
    """
    dataset = args.dataset

    print(f"\n{'='*60}")
    print(f"  INTERACTIVE MODE: {dataset.upper()}")
    print(f"{'='*60}\n")

    # Load model
    model, metadata = _load_model(dataset)
    feat_info = _get_feature_info(dataset)

    print(f"  Enter feature values. Type 'quit' to exit.\n")

    if dataset == "uci":
        # UCI has fixed features - ask for each one
        while True:
            print(f"  {'-'*50}")
            print(f"  New Prediction")
            print(f"  {'-'*50}")
            record = {}
            for feat in UCI_FEATURES:
                desc = feat_info.get("feature_descriptions", {}).get(feat, "")
                prompt = f"    {feat}"
                if desc:
                    prompt += f" ({desc})"
                prompt += ": "
                val = input(prompt).strip()
                if val.lower() in ("quit", "exit", "q"):
                    print("  Exiting interactive mode.")
                    return
                try:
                    record[feat] = float(val) if "." in val else int(val)
                except ValueError:
                    record[feat] = val

            df = pd.DataFrame([record])
            pred = model.predict(df)[0]
            print(
                f"\n    >>> Predicted {UCI_TARGET}: "
                f"{pred:.4f} {UCI_TARGET_UNIT}\n"
            )
    else:
        # Ames has 79 features - load from CSV file or provide default sample
        print(
            "  Ames has 79 features. You can either:\n"
            "    1. Provide a CSV file with all feature columns\n"
            "    2. Enter individual values for a subset (not recommended)\n"
        )
        choice = input("  Enter CSV file path (or press Enter for sample data): ").strip()

        if choice and os.path.isfile(choice):
            df = pd.read_csv(choice)
            # Drop SalePrice if present
            if AMES_TARGET in df.columns:
                df = df.drop(columns=[AMES_TARGET])
        else:
            # Load a sample from the AmesHousing.csv
            sample_path = os.path.join(SCRIPT_DIR, "AmesHousing.csv")
            if os.path.exists(sample_path):
                sample_df = pd.read_csv(sample_path)
                # Drop target and identifiers
                drop_cols = ["SalePrice", "Order", "PID"]
                drop_cols = [c for c in drop_cols if c in sample_df.columns]
                sample_df = sample_df.drop(columns=drop_cols)
                df = sample_df.iloc[[0]].reset_index(drop=True)
                print(
                    "  Using first record from AmesHousing.csv as sample input.\n"
                )
            else:
                print("  No sample data available. Exiting.")
                return

        pred = model.predict(df)[0]
        print(f"\n  >>> Predicted {AMES_TARGET}: ${pred:,.2f} {AMES_TARGET_UNIT}\n")

        # Also show the feature values used
        print(f"  Features used ({len(df.columns)}):")
        print(f"  {'-'*40}")
        row = df.iloc[0]
        # Show first 10 features only for brevity
        shown = 0
        for col in df.columns:
            if shown >= 10:
                print(f"    ... and {len(df.columns) - 10} more features")
                break
            print(f"    {col}: {row[col]}")
            shown += 1


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="House Price Prediction CLI - Load models, predict, "
        "retrain, and manage artifacts for UCI and Ames datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python house_price_cli.py predict --dataset uci --input sample.json
  python house_price_cli.py predict --dataset ames --input new_houses.csv
  python house_price_cli.py predict --dataset uci --input '{"X1 transaction date":2013.5,"X2 house age":13.3,"X3 distance to the nearest MRT station":561.98,"X4 number of convenience stores":5,"X5 latitude":24.987,"X6 longitude":121.544}'
  python house_price_cli.py retrain --dataset uci --data new_training_data.csv
  python house_price_cli.py list-models --dataset uci
  python house_price_cli.py interactive --dataset ames
        """,
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    # predict
    p_predict = subparsers.add_parser(
        "predict", help="Predict house prices using the best model"
    )
    p_predict.add_argument(
        "--dataset",
        "-d",
        required=True,
        choices=["uci", "ames"],
        help="Dataset to use (uci or ames)",
    )
    p_predict.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to JSON/CSV file, or a JSON string with feature values",
    )
    p_predict.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional path to save predictions (CSV or JSON)",
    )
    p_predict.set_defaults(func=cmd_predict)

    # retrain
    p_retrain = subparsers.add_parser(
        "retrain",
        help="Retrain all models on new data using the saved preprocessing pipeline",
    )
    p_retrain.add_argument(
        "--dataset",
        "-d",
        required=True,
        choices=["uci", "ames"],
        help="Dataset to retrain (uci or ames)",
    )
    p_retrain.add_argument(
        "--data",
        required=True,
        help="Path to CSV/JSON file with training data (must include target column)",
    )
    p_retrain.set_defaults(func=cmd_retrain)

    # list-models
    p_list = subparsers.add_parser(
        "list-models",
        help="List all available models and their performance metrics",
    )
    p_list.add_argument(
        "--dataset",
        "-d",
        required=True,
        choices=["uci", "ames"],
        help="Dataset to list models for",
    )
    p_list.set_defaults(func=cmd_list_models)

    # interactive
    p_interactive = subparsers.add_parser(
        "interactive",
        help="Interactive mode - enter feature values and get predictions",
    )
    p_interactive.add_argument(
        "--dataset",
        "-d",
        required=True,
        choices=["uci", "ames"],
        help="Dataset to use (uci or ames)",
    )
    p_interactive.set_defaults(func=cmd_interactive)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()

