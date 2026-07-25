#!/usr/bin/env python3
"""
House Price Prediction CLI - Inference, Evaluation, Retraining & Management.

A single entry point for the lab's two trained pipelines (UCI Real Estate
Valuation and Ames Housing). It loads serialized artifacts (models + the
fitted preprocessing ColumnTransformer) so the professor can run sample
inference **without retraining**, and also supports retraining on new data.

Commands
--------
  info         Show environment versions + artifact inventory (sanity check).
  list         Print the model registry, metrics and artifact files.
  predict      Run inference on a CSV/JSON file with the best (or named) model.
  evaluate     Run inference and score it against ground-truth prices.
  retrain      Refit the pipeline on new data and overwrite the artifacts.
  interactive  Prompt for feature values and predict one record at a time.

Layout (expected relative to the project root)
----------------------------------------------
    House_price_pred/
      src/house_price_cli.py        <- this file
      artifacts/                    <- models, preprocessor, results, metadata
      data/AmesHousing.csv
      examples/                     <- sample files for predict/evaluate

Because SCRIPT_DIR (the src/ folder) is resolved absolutely, the CLI works
from any working directory - no `cd` required.

Examples
--------
  python src/house_price_cli.py info
  python src/house_price_cli.py list uci
  python src/house_price_cli.py predict uci  examples/uci_sample.json
  python src/house_price_cli.py predict ames examples/ames_eval_sample.csv -o predictions.csv
  python src/house_price_cli.py evaluate uci examples/uci_sample.json
  python src/house_price_cli.py retrain uci  examples/uci_retrain_example.csv
  python src/house_price_cli.py interactive uci
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

# ============================================================
# CONFIG - file locations (resolved from this script's path)
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)              # src/ -> project root
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "examples")

SEED = 42

UCI_FEATURES = [
    "X1 transaction date",
    "X2 house age",
    "X3 distance to the nearest MRT station",
    "X4 number of convenience stores",
    "X5 latitude",
    "X6 longitude",
]
UCI_TARGET = "Y house price of unit area"
UCI_UNIT = "10,000 NTD/Ping"

AMES_TARGET = "SalePrice"
AMES_UNIT = "USD"
AMES_ID_COLS = ["Order", "PID"]

# Per-dataset config: artifact filenames + estimator factory tie this together
# so every command resolves files the same way.
DATASET_CFG = {
    "uci": {
        "best_model": "uci_best_model.joblib",
        "preprocessor": "uci_preprocessing_pipeline.joblib",
        "results_csv": "uci_all_model_results.csv",
        "metadata": "uci_training_metadata.json",
        "registry": "uci_model_registry.json",
        "features": UCI_FEATURES,
        "target": UCI_TARGET,
        "unit": UCI_UNIT,
        "id_cols": [],
    },
    "ames": {
        "best_model": "ames_best_model.joblib",
        "preprocessor": "ames_preprocessor.joblib",
        "results_csv": "ames_all_results.csv",
        "metadata": "ames_metadata.json",
        "registry": None,
        "features": None,                 # read from the saved preprocessor
        "target": AMES_TARGET,
        "unit": AMES_UNIT,
        "id_cols": AMES_ID_COLS,
    },
}

# ASCII markers (kept ASCII so the source file is plain UTF-8 on every OS)
OK = "[OK]"
BEST = "*"
ARROW = "->"


# ============================================================
# PATH / IO HELPERS
# ============================================================
def _artifact(path):
    """Resolve an artifact filename under artifacts/, or an absolute path."""
    return path if os.path.isabs(path) else os.path.join(ARTIFACTS_DIR, path)


def _data_path(path):
    """Resolve a user data path, also allowing files under data/ & examples/."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    for base in (DATA_DIR, EXAMPLES_DIR, PROJECT_ROOT):
        cand = os.path.join(base, path)
        if os.path.exists(cand):
            return cand
    return path  # let a later os.path.exists raise the friendly error


def _require_artifact(path):
    full = _artifact(path)
    if not os.path.exists(full):
        raise FileNotFoundError(
            f"Artifact not found: {full}\n"
            "Run the notebook(s) in notebooks/ first to generate artifacts, "
            "or run `python src/house_price_cli.py retrain {uci,ames}`."
        )
    return full


def _load_joblib(path):
    import joblib
    return joblib.load(_require_artifact(path))


def _load_json(path, default=None):
    full = _artifact(path)
    if not os.path.exists(full):
        return default
    with open(full, "r", encoding="utf-8") as f:
        return json.load(f)


def _cfg(dataset):
    return DATASET_CFG[dataset]


def _metadata(dataset):
    cfg = _cfg(dataset)
    return _load_json(cfg["metadata"], default={})


def _registry(dataset):
    cfg = _cfg(dataset)
    if not cfg["registry"]:
        return {}
    return _load_json(cfg["registry"], default={})


def load_best_model(dataset):
    cfg = _cfg(dataset)
    return _load_joblib(cfg["best_model"])


def load_preprocessor(dataset):
    cfg = _cfg(dataset)
    return _load_joblib(cfg["preprocessor"])


def _ames_expected_columns(preprocessor):
    """Recover the exact feature columns the Ames ColumnTransformer was fit on."""
    cols = []
    try:
        for _, sub_pipe, col_list, *_ in preprocessor.transformers_:
            cols.extend(col_list)
    except AttributeError:
        for name, sub_pipe, col_list in preprocessor.transformers:
            if name == "remainder":
                continue
            cols.extend(col_list)
    return cols


def _feature_columns(dataset, df, preprocessor=None):
    """Return the ordered X columns to feed the pipeline for a given dataset."""
    cfg = _cfg(dataset)
    if dataset == "uci":
        return df[cfg["features"]]
    # Ames: align to the columns the preprocessor expects (order-insensitive)
    expected = _ames_expected_columns(preprocessor) if preprocessor is not None else None
    drop = [c for c in (cfg["id_cols"] + [cfg["target"]]) if c in df.columns]
    X = df.drop(columns=drop, errors="ignore")
    if expected is None:
        return X
    missing = [c for c in expected if c not in X.columns]
    for c in missing:
        X[c] = np.nan          # imputer fills these; keeps the transform call valid
    return X[expected]


def load_data(filepath):
    fp = _data_path(filepath)
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Input file not found: {fp}")
    ext = os.path.splitext(fp)[1].lower()
    if ext == ".csv":
        return pd.read_csv(fp)
    if ext == ".json":
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            for key in ("records", "features", "data", "instances"):
                if key in data and isinstance(data[key], list):
                    return pd.DataFrame(data[key])
            return pd.DataFrame([data])
        raise ValueError("JSON must be an object or an array of objects.")
    raise ValueError(f"Unsupported format '{ext}'. Use .csv or .json.")


def validate(df, dataset):
    cfg = _cfg(dataset)
    if df.empty:
        raise ValueError("Input data is empty.")
    if dataset == "uci":
        missing = [c for c in cfg["features"] if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}\n"
                f"Expected UCI features: {cfg['features']}"
            )


def _pick_model(dataset, model_name=None, preprocessor=None):
    """Return (pipeline, display_name). Best model by default, or a named one."""
    if not model_name:
        return load_best_model(dataset), _read_best_name(dataset)

    # Try a few filename conventions the notebooks use.
    slug = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    candidates = [
        f"{dataset}_{slug}.joblib",
        os.path.join(ARTIFACTS_DIR, f"{dataset}_{slug}.joblib"),
    ]
    for cand in candidates:
        full = cand if os.path.isabs(cand) else _artifact(cand)
        if os.path.exists(full):
            import joblib
            return joblib.load(full), model_name
    # Named-but-missing: clear error
    _require_artifact(candidates[0])  # raises FileNotFoundError with the right message


def _read_best_name(dataset):
    meta = _metadata(dataset)
    if dataset == "uci":
        # UCI metadata nests best_model as {name, ...}
        bm = meta.get("best_model", {})
        if isinstance(bm, dict):
            return bm.get("name", "best model")
        if isinstance(bm, str):
            return bm
    return meta.get("best_model", "best model")


def _metric_cols(df):
    """Pick the RMSE/r2 column names actually present in a results CSV."""
    rmse = next((c for c in df.columns if "rmse" in c.lower()), None)
    r2 = next(
        (c for c in df.columns
         if c.lower() in ("r2", "test_r2", "cv_r2_mean", "r2_score") or c.lower().endswith("_r2")),
        None,
    )
    return rmse or df.columns[0], r2


def _hr(n=60, ch="-"):
    return ch * n


def _banner(title, width=60):
    rule = "=" * width
    return f"\n{rule}\n{title.center(width)}\n{rule}"


# ============================================================
# COMMAND: info
# ============================================================
def cmd_info(args):
    import platform
    import sklearn
    import joblib

    print(_banner("HOUSE PRICE PREDICTION - ENVIRONMENT & ARTIFACTS"))

    print("\n[1/3] Python environment")
    print(f"  {OK} python         {platform.python_version()}")
    pkgs = [
        ("numpy", "numpy"), ("pandas", "pandas"), ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"), ("joblib", "joblib"), ("scipy", "scipy"),
    ]
    for display, import_name in pkgs:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "?")
            print(f"  {OK} {display:14} {ver}")
        except Exception as e:
            print(f"  [MISSING] {display:14} {e}")

    print(f"\n[2/3] Project paths")
    print(f"  project root : {PROJECT_ROOT}")
    print(f"  artifacts    : {ARTIFACTS_DIR}  ({'exists' if os.path.isdir(ARTIFACTS_DIR) else 'MISSING'})")
    print(f"  data         : {DATA_DIR}  ({'exists' if os.path.isdir(DATA_DIR) else 'MISSING'})")
    print(f"  examples     : {EXAMPLES_DIR}  ({'exists' if os.path.isdir(EXAMPLES_DIR) else 'MISSING'})")

    print(f"\n[3/3] Artifact inventory")
    if not os.path.isdir(ARTIFACTS_DIR):
        print("  [!] artifacts/ not found - run the notebooks to generate it.")
        return

    all_ok = True
    for ds in ("uci", "ames"):
        cfg = _cfg(ds)
        print(f"\n  --- {ds.upper()} ---")
        for key, fname in (("best_model", cfg["best_model"]),
                           ("preprocessor", cfg["preprocessor"]),
                           ("results_csv", cfg["results_csv"]),
                           ("metadata", cfg["metadata"])):
            full = _artifact(fname)
            if os.path.exists(full):
                sz = os.path.getsize(full) / 1024
                print(f"  {OK} {fname:42} {sz:9.1f} KB")
            else:
                print(f"  [MISSING] {fname}")
                if key in ("best_model", "preprocessor"):
                    all_ok = False
        meta = _metadata(ds)
        if meta:
            print(f"  best model : {_read_best_name(ds)}")
            bm = meta.get("best_model", meta.get("best_metrics", {}))
    print()
    if all_ok:
        print("  Ready: both best models + preprocessors present.")
    else:
        print("  [!] At least one best model/preprocessor is missing - train via the notebooks or `retrain`.")


# ============================================================
# COMMAND: list
# ============================================================
def cmd_list(args):
    dataset = args.dataset
    cfg = _cfg(dataset)
    print(_banner(f"MODEL REGISTRY: {dataset.upper()}"))

    # Results table
    csv_path = _artifact(cfg["results_csv"])
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        rmse_col, r2_col = _metric_cols(df)
        df = df.sort_values(rmse_col)
        print(f"\n  All models (from {os.path.basename(csv_path)}, sorted by {rmse_col} {ARROW})")
        print(f"  {_hr(55)}")
        view = df[[c for c in [rmse_col, r2_col] if c]] if len(df) else df
        # keep model name first if present
        cols = [c for c in df.columns if c.lower() in ("model", "name")]
        cols += [c for c in [rmse_col, r2_col] if c]
        cols += [c for c in df.columns if c not in cols]
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    else:
        print("\n  (No results CSV found - run the notebook or `retrain`.)")

    # Best model
    print(f"\n  {BEST} Best model")
    print(f"  {_hr(30)}")
    meta = _metadata(dataset)
    bm_name = _read_best_name(dataset)
    print(f"    Name : {bm_name}")
    metrics = meta.get("best_metrics") or meta.get("best_model", {}).get("performance") or \
        (meta.get("best_model", {}).get("metrics") if isinstance(meta.get("best_model"), dict) else {})
    for k, v in (metrics.items() if isinstance(metrics, dict) else []):
        try:
            print(f"    {k:6}: {float(v):,.4f}")
        except (TypeError, ValueError):
            print(f"    {k:6}: {v}")

    # UCI tuned configs from the registry
    if dataset == "uci":
        reg = _registry(dataset)
        tuned = reg.get("tuned_models") or {}
        if tuned:
            print(f"\n  Tuned model configs (from registry)")
            print(f"  {_hr(30)}")
            for name, cfgx in tuned.items():
                print(f"    {name}:")
                params = cfgx.get("params", cfgx)
                if isinstance(params, dict):
                    for pk, pv in params.items():
                        print(f"      {pk}: {pv}")
                else:
                    print(f"      {params}")

    # Artifact files
    print(f"\n  Artifact files")
    print(f"  {_hr(30)}")
    if not os.path.isdir(ARTIFACTS_DIR):
        print("    (artifacts/ not found)")
    else:
        for fname in sorted(f for f in os.listdir(ARTIFACTS_DIR) if f.startswith(dataset)):
            sz = os.path.getsize(os.path.join(ARTIFACTS_DIR, fname)) / 1024
            print(f"    {fname}  ({sz:,.1f} KB)")

    # Features
    print(f"\n  Features")
    print(f"  {_hr(30)}")
    print(f"    Target    : {cfg['target']}  ({cfg['unit']})")
    if dataset == "uci":
        print(f"    {len(cfg['features'])} numeric features:")
        for f in cfg["features"]:
            print(f"      - {f}")
    else:
        try:
            pre = load_preprocessor(dataset)
            cols = _ames_expected_columns(pre)
            print(f"    {len(cols)} features (numeric + nominal, fit by preprocessor)")
            print(f"      first 5: {cols[:5]}")
        except Exception:
            print("    (load preprocessor to see the feature list)")
    print()


# ============================================================
# COMMAND: predict
# ============================================================
def cmd_predict(args):
    dataset = args.dataset
    cfg = _cfg(dataset)
    print(_banner(f"PREDICT - {dataset.upper()}"))
    print(f"  File  : {args.filepath}")
    print(f"  Model : {'best (default)' if not args.model else args.model}")

    print(f"\n[1/3] Load model")
    pre = load_preprocessor(dataset)
    model, name = _pick_model(dataset, args.model, preprocessor=pre)
    print(f"  {OK} loaded: {name}")

    print(f"\n[2/3] Load data")
    df = load_data(args.filepath)
    validate(df, dataset)
    X = _feature_columns(dataset, df, preprocessor=pre)
    print(f"  {OK} {len(X)} record(s), {X.shape[1]} feature(s)")

    print(f"\n[3/3] Inference")
    preds = model.predict(X)

    print(f"\n  {_hr(55)}")
    print(f"  PREDICTIONS ({cfg['unit']})")
    print(f"  {_hr(55)}")
    for i in range(len(preds)):
        print(f"\n  Record {i + 1}:")
        if dataset == "uci" and args.show_features:
            for f in cfg["features"]:
                print(f"    {f}: {X.iloc[i][f]}")
        print(f"    {ARROW}> Predicted {cfg['target']}: {preds[i]:,.4f} {cfg['unit']}")
    print()

    if args.output:
        out_df = X.copy()
        out_df[f"Predicted_{cfg['target'].replace(' ', '_')}"] = preds
        if args.output.lower().endswith(".csv"):
            out_df.to_csv(args.output, index=False)
        else:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(out_df.to_dict(orient="records"), f, indent=2)
        print(f"  {OK} predictions saved -> {args.output}\n")


# ============================================================
# COMMAND: evaluate
# ============================================================
def _metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) \
        if nonzero.any() else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_%": mape}


def cmd_evaluate(args):
    dataset = args.dataset
    cfg = _cfg(dataset)
    print(_banner(f"EVALUATE - {dataset.upper()}"))
    print(f"  File  : {args.filepath}")
    print(f"  Model : {'best (default)' if not args.model else args.model}")

    print(f"\n[1/4] Load model")
    pre = load_preprocessor(dataset)
    model, name = _pick_model(dataset, args.model, preprocessor=pre)
    print(f"  {OK} loaded: {name}")

    print(f"\n[2/4] Load data")
    df = load_data(args.filepath)
    validate(df, dataset)
    if cfg["target"] not in df.columns:
        raise ValueError(
            f"'{args.filepath}' has no '{cfg['target']}' column.\n"
            f"`evaluate` needs ground-truth prices. Use `predict` for unlabeled data."
        )
    X = _feature_columns(dataset, df, preprocessor=pre)
    y = df[cfg["target"]].astype(float).values
    print(f"  {OK} {len(X)} record(s), {X.shape[1]} feature(s), target present")

    print(f"\n[3/4] Inference")
    preds = model.predict(X)

    print(f"\n[4/4] Metrics versus ground truth")
    m = _metrics(y, preds)
    print(f"  {_hr(55)}")
    print(f"  MODEL: {name}")
    print(f"  {_hr(55)}")
    print(f"    MAE  : {m['MAE']:,.4f}  {cfg['unit']}")
    print(f"    RMSE : {m['RMSE']:,.4f}  {cfg['unit']}")
    print(f"    R2   : {m['R2']:.4f}")
    print(f"    MAPE : {m['MAPE_%']:.2f} %")
    print(f"  {_hr(55)}")
    print(f"  {'Record':>7}  {'Actual':>14}  {'Predicted':>14}  {'Error':>14}")
    for i in range(min(len(y), args.head if args.head else len(y))):
        print(f"  {i + 1:>7}  {y[i]:>14,.2f}  {preds[i]:>14,.2f}  {preds[i] - y[i]:>14,.2f}")
    if args.head and len(y) > args.head:
        print(f"  ... {len(y) - args.head} more row(s) (set --head 0 / larger to show all)")
    print()

    if args.output:
        out = X.copy()
        out[cfg["target"]] = y
        out[f"Predicted_{cfg['target'].replace(' ', '_')}"] = preds
        out["Error"] = preds - y
        out["Abs_Error_%"] = np.abs(preds - y) / np.where(y == 0, np.nan, y) * 100
        out.to_csv(args.output, index=False)
        print(f"  {OK} scored predictions saved -> {args.output}\n")


# ============================================================
# COMMAND: retrain
# ============================================================
def _estimators(dataset):
    """Return the 8 base regressors, faithful to each notebook's defaults."""
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor

    if dataset == "uci":
        return {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.001, max_iter=20000),
            "Elastic Net": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=20000),
            "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=SEED),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, max_depth=4, min_samples_leaf=2,
                random_state=SEED, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
            "XGBoost": XGBRegressor(
                n_estimators=100, learning_rate=0.03, max_depth=4,
                subsample=0.8, colsample_bytree=0.8,
                objective="reg:squarederror", random_state=SEED, n_jobs=-1),
        }
    # Ames
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.001, max_iter=20000),
        "Elastic Net": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=20000),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=SEED),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2,
            random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
        "XGBoost": XGBRegressor(
            n_estimators=200, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", random_state=SEED, n_jobs=-1),
    }


def cmd_retrain(args):
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import joblib

    dataset = args.dataset
    cfg = _cfg(dataset)
    tag = args.tag or dataset
    print(_banner(f"RETRAIN - {dataset.upper()}"))
    print(f"  File : {args.filepath}")
    print(f"  seed : {SEED}   test split : 0.20   tag : {tag}")

    print(f"\n[1/6] Load new training data")
    df = load_data(args.filepath)
    if cfg["target"] not in df.columns:
        raise ValueError(f"Training data must contain the target column '{cfg['target']}'.")
    drop = [c for c in (cfg["id_cols"] + [cfg["target"]]) if c in df.columns]
    X = df.drop(columns=drop, errors="ignore")
    y = df[cfg["target"]].astype(float)
    print(f"  {OK} {len(df)} samples, {X.shape[1]} feature columns")

    print(f"\n[2/6] Leakage-safe train/test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED)
    print(f"  {OK} train={len(X_train)}  test={len(X_test)}")

    print(f"\n[3/6] Build & fit preprocessing pipeline (on train split)")
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ], remainder="drop")
    preprocess.fit(X_train)
    print(f"  {OK} numeric={len(numeric_features)}  categorical={len(categorical_features)}")

    print(f"\n[4/6] Train {8} models")
    estimators = _estimators(dataset)
    rows = []
    trained = {}
    for name, est in estimators.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", est)])
        pipe.fit(X_train, y_train)
        trained[name] = pipe
        p = pipe.predict(X_test)
        m = _metrics(y_test.values, p)
        rows.append({
            "Model": name, "Test_MAE": m["MAE"], "Test_RMSE": m["RMSE"],
            "Test_R2": m["R2"], "MAPE_%": m["MAPE_%"],
        })
        print(f"  {OK} {name:22} RMSE={m['RMSE']:,.4f}  R2={m['R2']:.4f}")
    res_df = pd.DataFrame(rows).sort_values("Test_RMSE").reset_index(drop=True)
    best_name = res_df.iloc[0]["Model"]
    best_row = res_df.iloc[0]
    print(f"\n  {BEST} Best on new test set: {best_name}  "
          f"RMSE={best_row['Test_RMSE']:,.4f}  R2={best_row['Test_R2']:.4f}")

    print(f"\n[5/6] Refit best model on ALL new data, save artifacts")

    final_pre = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ], remainder="drop")
    final_pipe = Pipeline([("preprocess", final_pre), ("model", _estimators(dataset)[best_name])])
    final_pipe.fit(X, y)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(final_pipe, _artifact(f"{tag}_best_model.joblib"))
    print(f"  {OK} {tag}_best_model.joblib")
    joblib.dump(final_pre, _artifact(cfg["preprocessor"]))
    print(f"  {OK} {cfg['preprocessor']}")
    for name, pipe in trained.items():
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(pipe, _artifact(f"{tag}_{slug}.joblib"))
    print(f"  {OK} {len(trained)} individual model pipelines")
    res_df.to_csv(_artifact(f"{tag}_all_model_results.csv"), index=False)

    print(f"\n[6/6] Write metadata")
    meta = {
        "dataset": "UCI Real Estate Valuation" if dataset == "uci" else "Ames Housing",
        "target": cfg["target"],
        "unit": cfg["unit"],
        "training_file": os.path.abspath(args.filepath),
        "retrain_date": datetime.now().isoformat(timespec="seconds"),
        "random_seed": SEED,
        "test_split": 0.20,
        "n_samples": int(len(df)),
        "best_model": best_name,
        "best_metrics": {
            "MAE": float(best_row["Test_MAE"]),
            "RMSE": float(best_row["Test_RMSE"]),
            "R2": float(best_row["Test_R2"]),
            "MAPE": float(best_row["MAPE_%"]),
        },
        "all_models": rows,
        "features": X.columns.tolist(),
    }
    with open(_artifact(cfg["metadata"]), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  {OK} {cfg['metadata']}")

    print(f"\n  {'=' * 50}")
    print(f"  RETRAINING COMPLETE - artifacts overwritten. "
          f"Use `list {dataset}` to review.")
    print(f"  {'=' * 50}\n")


# ============================================================
# COMMAND: interactive
# ============================================================
def cmd_interactive(args):
    dataset = args.dataset
    cfg = _cfg(dataset)
    print(_banner(f"INTERACTIVE MODE: {dataset.upper()}"))
    pre = load_preprocessor(dataset)
    model, name = _pick_model(dataset, preprocessor=pre)
    print(f"  Loaded model: {name}")
    print(f"  Target: {cfg['target']} ({cfg['unit']})")

    if dataset == "uci":
        print(f"\n  Enter the {len(cfg['features'])} feature values. Type 'quit' to exit.\n")
        while True:
            print(f"  {_hr(45)}")
            print(f"  New prediction")
            print(f"  {_hr(45)}")
            record = {}
            for f in cfg["features"]:
                val = input(f"    {f}: ").strip()
                if val.lower() in ("quit", "exit", "q"):
                    print("  Goodbye!")
                    return
                try:
                    record[f] = float(val)
                except ValueError:
                    record[f] = val
            df = pd.DataFrame([record])
            pred = model.predict(df[cfg["features"]])[0]
            print(f"\n    {ARROW}> Predicted {cfg['target']}: {pred:,.4f} {cfg['unit']}\n")
        return

    # Ames - too many features to type; pick a CSV or use a dataset sample
    print("\n  Ames has ~79 features. Options:")
    print("    1. provide a CSV with the feature columns")
    print("    2. use a real record from data/AmesHousing.csv\n")
    choice = input("  CSV path (Enter = use a dataset sample): ").strip()
    if choice:
        df = load_data(choice)
    else:
        sample_path = os.path.join(DATA_DIR, "AmesHousing.csv")
        if not os.path.exists(sample_path):
            print("  data/AmesHousing.csv not found. Exiting.")
            return
        full = pd.read_csv(sample_path)
        drop = [c for c in (cfg["id_cols"] + [cfg["target"]]) if c in full.columns]
        df = full.drop(columns=drop).iloc[[0]]
        print("  Using the first record of data/AmesHousing.csv\n")
    X = _feature_columns(dataset, df, preprocessor=pre)
    pred = model.predict(X)[0]
    print(f"  {ARROW}> Predicted {cfg['target']}: ${pred:,.2f}")
    print(f"  ({len(X.columns)} features used, first 10 shown)")
    for i, col in enumerate(X.columns[:10]):
        print(f"    {col}: {X.iloc[0][col]}")
    print()


# ============================================================
# ARG PARSING
# ============================================================
def build_parser():
    parser = argparse.ArgumentParser(
        prog="house_price_cli.py",
        description="House Price Prediction CLI - inference, evaluation, "
                    "retraining & model management for the UCI and Ames pipelines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/house_price_cli.py info\n"
            "  python src/house_price_cli.py list uci\n"
            "  python src/house_price_cli.py predict uci  examples/uci_sample.json\n"
            "  python src/house_price_cli.py predict ames examples/ames_eval_sample.csv -o out.csv\n"
            "  python src/house_price_cli.py evaluate ames examples/ames_eval_sample.csv\n"
            "  python src/house_price_cli.py retrain uci  examples/uci_retrain_example.csv\n"
            "  python src/house_price_cli.py interactive uci\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Show Python/library versions + artifact inventory")

    p = sub.add_parser("list", help="Show the model registry, metrics & artifacts")
    p.add_argument("dataset", choices=["uci", "ames"])

    p = sub.add_parser("predict", help="Run inference on a CSV/JSON file")
    p.add_argument("dataset", choices=["uci", "ames"])
    p.add_argument("filepath", help="Path to a .csv or .json input file")
    p.add_argument("-o", "--output", help="Save predictions to a .csv or .json file")
    p.add_argument("--model", help="Override best model with a named saved model (e.g. 'XGBoost')")
    p.add_argument("--no-features", dest="show_features", action="store_false",
                   default=True, help="Do not print per-record feature values")

    p = sub.add_parser("evaluate", help="Inference + scoring against ground-truth prices")
    p.add_argument("dataset", choices=["uci", "ames"])
    p.add_argument("filepath", help="Path to a .csv or .json file containing the target column")
    p.add_argument("-o", "--output", help="Save scored predictions (with errors) to a .csv file")
    p.add_argument("--model", help="Override best model with a named saved model")
    p.add_argument("--head", type=int, default=10,
                   help="Max rows of actual-vs-predicted to print (default 10)")

    p = sub.add_parser("retrain", help="Refit the pipeline on new data and overwrite artifacts")
    p.add_argument("dataset", choices=["uci", "ames"])
    p.add_argument("filepath", help="Path to a .csv/.json file with features + target column")
    p.add_argument("--tag", help="Filename tag for saved artifacts (default: the dataset name)")

    p = sub.add_parser("interactive", help="Prompt for feature values and predict")
    p.add_argument("dataset", choices=["uci", "ames"])

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cmds = {
        "info": cmd_info,
        "list": cmd_list,
        "predict": cmd_predict,
        "evaluate": cmd_evaluate,
        "retrain": cmd_retrain,
        "interactive": cmd_interactive,
    }
    try:
        cmds[args.command](args)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}\n", file=sys.stderr)
        sys.exit(2)
    except ValueError as e:
        print(f"\n[ERROR] {e}\n", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nInterrupted.\n", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
