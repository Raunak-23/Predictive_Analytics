#!/usr/bin/env python3
"""
meddiag_cli - Medical Diagnosis Support CLI
(Lab 02 - Disease Classification using Decision Trees).

Single entry point for the three trained pipelines (Breast Cancer Wisconsin
Diagnostic, Early Stage Diabetes Risk Prediction, Heart Disease - Cleveland).
It loads serialized artifacts (final pipeline + operating threshold + metadata)
so the professor can run sample inference **without retraining**, and also
supports retraining on new data, threshold adjustment, and a Tkinter GUI.

Commands
--------
  info         Show environment versions + artifact inventory (sanity check).
  samples      Isolated inference on bundled `examples/<ds>_sample.json` fixtures - the
               quickest sample-inference fetch from saved artefacts (no retraining).
  list         Print the model registry, CV + test metrics and artifact files.
  predict      Run inference on a CSV/JSON file -> probabilities + class calls.
  evaluate     Inference + scoring against ground-truth labels (full metric block).
  retrain      Refit the whole training workflow on new data, overwrite artifacts.
  interactive  Prompt for feature values one at a time and predict.
  explain      Trace the decision path of a single record through the tree.
  gui          Launch the Tkinter desktop GUI.

All `predict` / `samples` / `evaluate` / `explain` commands load only the saved
pipelines + threshold + metadata; nothing is retrained. Use `retrain` to fit the
leakage-safe workflow on a new dataset and overwrite the saved artefacts.

Layout (expected relative to the project root)
----------------------------------------------
    Medical_diagnosis_support/
      src/meddiag_cli.py          <- this file
      src/meddiag_common.py       <- shared engine (datasets, preprocessing, models)
      artifacts/                  <- the trained pipelines, thresholds, results, metadata
      data/                       <- cached UCI CSVs
      examples/                   <- sample inputs for predict/evaluate

Because SCRIPT_DIR (the src/ folder) is resolved absolutely and added to
sys.path, the CLI works from any working directory - no `cd` required.

Educational-use boundary: the artifacts are a research/teaching prototype and
are NOT a clinically validated diagnostic system; never use for patient care.
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import meddiag_common as M  # noqa: E402

OK = "[OK]"
BEST = "*"
ARROW = "->"
ART_DIR = M.ART_DIR
DATA_DIR = M.DATA_DIR
EXAMPLES_DIR = M.EXAMPLES_DIR


# ============================================================
# INPUT LOADING + FEATURE ALIGNMENT
# ============================================================
def _data_path(path):
    if os.path.isabs(path) or os.path.exists(path):
        return path
    for base in (DATA_DIR, EXAMPLES_DIR, M.PROJECT_ROOT):
        cand = os.path.join(base, path)
        if os.path.exists(cand):
            return cand
    return path


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


def _align_features(dataset, df, meta):
    """Coerce a raw input frame to the model's expected feature columns, in order,
    coercing dtypes so the preprocessor's imputer/encoder accept it."""
    s = M.spec(dataset)
    expected = meta.get("feature_cols", s.feature_cols)
    g = df.copy()
    # keep only expected cols; fill any missing with NaN (imputed at predict)
    out = pd.DataFrame(index=g.index)
    for c in expected:
        if c in g.columns:
            out[c] = g[c]
        else:
            out[c] = np.nan
    # coerce dtypes to match training
    num_set = set(meta.get("numeric_cols", []))
    cat_set = set(meta.get("categorical_cols", []))
    for c in out.columns:
        if c in num_set:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        elif c in cat_set:
            out[c] = out[c].astype("object").astype(str)
            out.loc[out[c].isin(("nan", "None", "", "NaN")), c] = np.nan
    return out[expected]


def _proba_to_class(prob, threshold, class_names):
    return [(class_names[1] if p >= threshold else class_names[0], p) for p in prob]


def _pick_model(dataset, model_name=None, tag=None):
    return M.choose_model(dataset, model_name=model_name, tag=tag)


# ============================================================
# PRINTING HELPERS
# ============================================================
def _hr(n=60, ch="-"):
    return ch * n


def _banner(title, width=60):
    rule = "=" * width
    return f"\n{rule}\n{title.center(width)}\n{rule}"


def _best_name(dataset, tag=None):
    meta = M.load_metadata(dataset, tag)
    return meta.get("best_model", "best model")


# ============================================================
# COMMAND: info
# ============================================================
def cmd_info(args):
    import platform
    print(_banner("MEDICAL DIAGNOSIS SUPPORT - ENVIRONMENT & ARTIFACTS"))

    print("\n[1/3] Python environment")
    print(f"  {OK} python         {platform.python_version()}")
    pkgs = [("numpy", "numpy"), ("pandas", "pandas"), ("scikit-learn", "sklearn"),
            ("joblib", "joblib"), ("matplotlib", "matplotlib"),
            ("seaborn", "seaborn"), ("ucimlrepo", "ucimlrepo")]
    for disp, imp in pkgs:
        try:
            mod = __import__(imp)
            print(f"  {OK} {disp:14} {getattr(mod, '__version__', '?')}")
        except Exception as e:
            print(f"  [MISSING] {disp:14} {e}")
    try:
        import xgboost
        print(f"  {OK} xgboost        {xgboost.__version__} (optional bench)")
    except Exception:
        print(f"  [--] xgboost        not installed (optional)")

    print(f"\n[2/3] Project paths")
    print(f"  project root : {M.PROJECT_ROOT}")
    print(f"  artifacts    : {ART_DIR}  ({'exists' if os.path.isdir(ART_DIR) else 'MISSING'})")
    print(f"  data         : {DATA_DIR}  ({'exists' if os.path.isdir(DATA_DIR) else 'MISSING'})")
    print(f"  examples     : {EXAMPLES_DIR}  ({'exists' if os.path.isdir(EXAMPLES_DIR) else 'MISSING'})")

    print(f"\n[3/3] Artifact inventory")
    if not os.path.isdir(ART_DIR):
        print("  [!] artifacts/ not found - run the notebooks to generate it.")
        return
    all_ok = True
    for ds in M.CHOICES:
        meta = M.load_metadata(ds)
        print(f"\n  --- {ds.upper()} : {meta.get('dataset_display_name','(not trained)')} ---")
        need = (f"{ds}_best_model.joblib", f"{ds}_threshold.joblib",
               f"{ds}_cv_results.csv", f"{ds}_test_results.csv",
               f"{ds}_final_test_metrics.csv", f"{ds}_metadata.json")
        for fname in need:
            full = os.path.join(ART_DIR, fname)
            if os.path.exists(full):
                sz = os.path.getsize(full) / 1024
                print(f"  {OK} {fname:42} {sz:9.1f} KB")
            else:
                print(f"  [MISSING] {fname}")
                if fname.endswith("best_model.joblib") or fname.endswith("threshold.joblib"):
                    all_ok = False
        if meta:
            print(f"  best model   : {meta.get('best_model','?')}  "
                  f"(threshold {meta.get('threshold','?'):.3f})")
    print()
    print("  Ready: all best pipelines + thresholds present." if all_ok
          else "  [!] At least one pipeline/threshold is missing - train via the notebooks or `retrain`.")


# ============================================================
# COMMAND: samples  (isolated inference on bundled example fixtures)
# ============================================================
def cmd_samples(args):
    """Predict on the bundled `examples/<dataset>_sample.json` fixture without
    requiring the user to know a path - the quickest way to *fetch a sample
    inference* from the saved artefacts. Falls back to `_sample.csv` then to the
    first rows of the cached dataset so the command always produces something.
    This is fully isolated inference: it only loads saved pipelines,
    never retrains.
    """
    ds = args.dataset
    s = M.spec(ds)
    meta = M.load_metadata(ds)
    if not meta:
        raise FileNotFoundError(
            f"No trained pipeline for '{ds}'. Run `retrain {ds} "
            f"data/{s.csv_name}` first.")
    thr = float(args.threshold) if args.threshold is not None else meta["threshold"]
    # locate a fixture in preference order
    cands = [os.path.join(EXAMPLES_DIR, f"{ds}_sample.json"),
             os.path.join(EXAMPLES_DIR, f"{ds}_sample.csv"),
             os.path.join(DATA_DIR, s.csv_name)]
    filepath = next((p_ for p_ in cands if os.path.exists(p_)), None)
    if filepath is None:
        raise FileNotFoundError(
            f"No sample fixture for '{ds}'. Expected one of {cands}.")
    print(_banner(f"SAMPLES - {ds.upper()} ({meta['dataset_display_name']})"))
    print(f"  Fixture : {filepath}")
    print(f"  Model   : {args.model or 'best'}   (threshold {thr:.3f})")
    print(f"  Mode    : isolated inference from saved artefacts (no retraining)")

    pipe, name, _ = _pick_model(ds, args.model, args.tag)
    print(f"  {OK} loaded: {name}")
    df = load_data(filepath)
    # If the cached dataset was selected, drop the target & take only a few rows.
    if os.path.abspath(filepath) == os.path.abspath(os.path.join(DATA_DIR, s.csv_name)):
        df = df.drop(columns=[c for c in (s.target_col,) if c in df.columns]).head(4)
    X = _align_features(ds, df, meta)
    proba = pipe.predict_proba(X)[:, 1]
    class_names = meta["class_names"]
    print()
    print(f"  {_hr(62)}")
    print(f"  PREDICTIONS  (positive = 1 = '{meta['positive_class']}'; thr {thr:.3f})")
    print(f"  {_hr(62)}")
    for i in range(len(X)):
        cls = class_names[1] if proba[i] >= thr else class_names[0]
        flag = "POSITIVE" if proba[i] >= thr else "negative"
        print(f"  [{i+1:>2}] P={proba[i]:.4f}  -> {flag} ({cls})")
    if args.output:
        out = df.copy()
        out[f"p_{meta['positive_class']}"] = proba
        out["predicted_class"] = [class_names[1] if p >= thr else class_names[0]
                                   for p in proba]
        if args.output.lower().endswith(".csv"):
            out.to_csv(args.output, index=False)
        else:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(out.to_dict(orient="records"), f, indent=2, default=str)
        print(f"\n  {OK} predictions saved -> {args.output}")
    print("\n  Educational prototype - NOT a clinical diagnostic system.\n")


# ============================================================
# COMMAND: list
# ============================================================
def cmd_list(args):
    ds = args.dataset
    meta = M.load_metadata(ds)
    print(_banner(f"MODEL REGISTRY: {ds.upper()}"))
    if not meta:
        print("\n  (No metadata yet - run the notebook or `retrain`.)")
        return
    print(f"\n  Dataset      : {meta['dataset_display_name']}")
    print(f"  UCI id       : {meta['uci_id']}   DOI: {meta['doi']}")
    print(f"  Observation  : {meta['observation_unit']}")
    print(f"  Positive     : 1 = '{meta['positive_class']}' "
          f"(raw {meta['positive_raw_values']})")
    print(f"  Rows / split : {meta['n_rows']}  ->  train {meta['n_train']} / test {meta['n_test']}  "
          f"(seed {meta['seed']}, {meta['cv_folds']}-fold stratified, test_size={meta['test_size']})")
    print(f"  Prevalence   : train {meta['train_prevalence']:.3f}  test {meta['test_prevalence']:.3f}")

    # CV table
    cv_csv = os.path.join(ART_DIR, f"{ds}_cv_results.csv")
    if os.path.exists(cv_csv):
        cv = pd.read_csv(cv_csv)
        print(f"\n  Cross-validation (training-only, sorted by CV ROC-AUC {ARROW})")
        print(f"  {_hr(72)}")
        cols = ["Model", "roc_auc_mean", "roc_auc_std", "pr_auc_mean",
                "sensitivity_mean", "specificity_mean", "f1_mean"]
        cols = [c for c in cols if c in cv.columns]
        print(cv[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # all test rows
    test_csv = os.path.join(ART_DIR, f"{ds}_test_results.csv")
    if os.path.exists(test_csv):
        tr = pd.read_csv(test_csv)
        print(f"\n  Final test-set metrics (threshold {meta['threshold']:.3f})")
        print(f"  {_hr(72)}")
        cols = ["Model", "roc_auc", "pr_auc", "sensitivity", "specificity",
                "f1", "FN", "FP", "accuracy"]
        cols = [c for c in cols if c in tr.columns]
        print(tr[cols].to_string(index=False,
              float_format=lambda x: f"{x:.4f}"))

    # best block
    print(f"\n  {BEST} Best model")
    print(f"  {_hr(30)}")
    print(f"    Name      : {meta['best_model']}")
    print(f"    Threshold : {meta['threshold']:.3f}  "
          f"(target sens {meta['threshold_target']:.2f}, satisfied {meta['threshold_target_satisfied']:.2f})")
    tm = meta["test_metrics"]
    for k in ("TN", "FP", "FN", "TP", "sensitivity", "specificity",
              "precision_ppv", "f1", "roc_auc", "pr_auc", "brier", "accuracy"):
        if k in tm:
            v = tm[k]
            print(f"    {k:13}: {v:,.4f}" if isinstance(v, float) else f"    {k:13}: {v}")

    if meta.get("pruned_params"):
        print(f"\n  Tuned & pruned CART parameters")
        print(f"  {_hr(30)}")
        for k, v in meta["pruned_params"].items():
            print(f"    {k}: {v}")

    # artifact files
    print(f"\n  Artifact files")
    print(f"  {_hr(30)}")
    if os.path.isdir(ART_DIR):
        for fname in sorted(f for f in os.listdir(ART_DIR) if f.startswith(ds)):
            sz = os.path.getsize(os.path.join(ART_DIR, fname)) / 1024
            print(f"    {fname}  ({sz:,.1f} KB)")

    print(f"\n  Features")
    print(f"  {_hr(30)}")
    fc = meta.get("feature_cols", [])
    print(f"    {len(fc)} features  ({len(meta.get('numeric_cols',[]))} numeric, "
          f"{len(meta.get('categorical_cols',[]))} categorical)")
    for f in fc:
        kind = "num" if f in meta.get("numeric_cols", []) else "cat"
        print(f"      - {f}  ({kind})")
    print()


# ============================================================
# COMMAND: predict
# ============================================================
def cmd_predict(args):
    ds = args.dataset
    meta = M.load_metadata(ds)
    if not meta:
        raise FileNotFoundError(f"No trained pipeline for '{ds}'. Run the notebook or `retrain`.")
    threshold = float(args.threshold) if args.threshold is not None else meta["threshold"]
    print(_banner(f"PREDICT - {ds.upper()} ({meta['dataset_display_name']})"))
    print(f"  File  : {args.filepath}")
    print(f"  Model : {args.model or 'best'}  (threshold {threshold:.3f})")

    print(f"\n[1/3] Load model")
    pipe, name, _thr = _pick_model(ds, args.model, args.tag)
    print(f"  {OK} loaded: {name}")

    print(f"\n[2/3] Load data")
    df = load_data(args.filepath)
    X = _align_features(ds, df, meta)
    print(f"  {OK} {len(X)} record(s), {X.shape[1]} feature(s)")

    print(f"\n[3/3] Inference")
    proba = pipe.predict_proba(X)[:, 1]
    class_names = meta["class_names"]

    print(f"\n  {_hr(62)}")
    print(f"  PREDICTIONS  (positive = 1 = '{meta['positive_class']}'; thr {threshold:.3f})")
    print(f"  {_hr(62)}")
    for i in range(len(X)):
        cls = class_names[1] if proba[i] >= threshold else class_names[0]
        flag = "POSITIVE" if proba[i] >= threshold else "negative"
        print(f"\n  Record {i + 1}:")
        if args.show_features:
            for c in X.columns[:8]:
                print(f"    {c}: {X.iloc[i][c]}")
            if X.shape[1] > 8:
                print(f"    ... +{X.shape[1] - 8} more feature(s)")
        print(f"    {ARROW}> P({meta['positive_class']}) = {proba[i]:.4f}   "
              f"-> {flag} ({cls})")
    print()

    if args.output:
        out = df.copy()
        out[f"p_{meta['positive_class']}"] = proba
        out["predicted_class"] = [class_names[1] if p >= threshold else class_names[0]
                                   for p in proba]
        if args.output.lower().endswith(".csv"):
            out.to_csv(args.output, index=False)
        else:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(out.to_dict(orient="records"), f, indent=2, default=str)
        print(f"  {OK} predictions saved -> {args.output}\n")


# ============================================================
# COMMAND: evaluate
# ============================================================
def cmd_evaluate(args):
    ds = args.dataset
    meta = M.load_metadata(ds)
    if not meta:
        raise FileNotFoundError(f"No trained pipeline for '{ds}'.")
    threshold = float(args.threshold) if args.threshold is not None else meta["threshold"]
    print(_banner(f"EVALUATE - {ds.upper()} ({meta['dataset_display_name']})"))
    print(f"  File  : {args.filepath}")
    print(f"  Model : {args.model or 'best'}  (threshold {threshold:.3f})")

    print(f"\n[1/4] Load model")
    pipe, name, _ = _pick_model(ds, args.model, args.tag)
    print(f"  {OK} loaded: {name}")

    print(f"\n[2/4] Load data")
    df = load_data(args.filepath)
    if meta["target_col"] not in df.columns:
        raise ValueError(
            f"'{args.filepath}' has no '{meta['target_col']}' column.\n"
            f"`evaluate` needs ground-truth labels. Use `predict` for unlabeled data.")
    X = _align_features(ds, df, meta)
    # rebuild the binary y from the raw target column using the spec's mapping
    s = M.spec(ds)
    Xraw, yraw = M.make_xy(ds, df)
    y = yraw.values
    print(f"  {OK} {len(X)} record(s), {X.shape[1]} feature(s), target present")

    print(f"\n[3/4] Inference")
    proba = pipe.predict_proba(X)[:, 1]

    print(f"\n[4/4] Metrics versus ground truth")
    m = M.evaluate_binary(y, proba, threshold)
    print(f"  {_hr(62)}")
    print(f"  MODEL: {name}  (threshold {threshold:.3f})")
    print(f"  {_hr(62)}")
    print(f"    Confusion   TN={m['TN']}  FP={m['FP']}  FN={m['FN']}  TP={m['TP']}")
    print(f"    Sensitivity : {m['sensitivity']:.4f}     Specificity : {m['specificity']:.4f}")
    print(f"    Precision   : {m['precision_ppv']:.4f}        F1         : {m['f1']:.4f}")
    print(f"    ROC-AUC     : {m['roc_auc']:.4f}        PR-AUC      : {m['pr_auc']:.4f}")
    print(f"    Brier       : {m['brier']:.4f}        MCC         : {m['mcc']:.4f}")
    print(f"    Bal. acc    : {m['balanced_accuracy']:.4f}    Expected cost (FNx{meta['cost_fn']}+FPx{meta['cost_fp']}): {m['expected_cost']:.0f}")
    print(f"  {_hr(62)}")
    print(f"  {'Rec':>4} {'Actual':>8} {'P(pos)':>8} {'Pred':>10} {'Flag':>10}")
    head = args.head if args.head else len(y)
    for i in range(min(len(y), head)):
        cls = meta['class_names'][1] if proba[i] >= threshold else meta['class_names'][0]
        flag = "POSITIVE" if proba[i] >= threshold else "negative"
        print(f"  {i+1:>4} {int(y[i]):>8} {proba[i]:>8.4f} {cls:>10} {flag:>10}")
    if len(y) > head:
        print(f"  ... {len(y) - head} more row(s) (use --head 0 to show all)")
    print()

    if args.output:
        out = df.copy()
        out["__actual_positive__"] = y
        out[f"p_{meta['positive_class']}"] = proba
        out["predicted_class"] = [meta['class_names'][1] if p >= threshold else meta['class_names'][0]
                                   for p in proba]
        out["correct"] = (y == (proba >= threshold).astype(int)).astype(int)
        out.to_csv(args.output, index=False)
        print(f"  {OK} scored predictions saved -> {args.output}\n")


# ============================================================
# COMMAND: retrain
# ============================================================
def cmd_retrain(args):
    ds = args.dataset
    s = M.spec(ds)
    tag = args.tag or ds
    print(_banner(f"RETRAIN - {ds.upper()} ({s.display_name})"))
    print(f"  File : {args.filepath}")
    print(f"  seed : {M.SEED}   test split : {M.TEST_SIZE}   tag : {tag}")

    print(f"\n[1/6] Load new training data")
    df = load_data(args.filepath)
    if s.target_col not in df.columns:
        raise ValueError(f"Training data must contain the target column '{s.target_col}'.")
    print(f"  {OK} {len(df)} samples loaded")

    print(f"\n[2/6] Run leakage-safe workflow (split, tune+prune, CV, threshold, one test eval)")
    state = M.train_dataset(ds, df=df, verbose=True)

    print(f"\n[3/6] Summary of new best model")
    print(f"  {OK} best: {state['best_name']}  threshold {state['threshold']:.3f}")
    tm = state["test_metrics"]
    print(f"      sens {tm['sensitivity']:.4f}  spec {tm['specificity']:.4f}  "
          f"roc_auc {tm['roc_auc']:.4f}  FN {tm['FN']}  FP {tm['FP']}")

    print(f"\n[4/6] Save artifacts (pipelines, threshold, results, metadata)")
    paths = M.save_artifacts(ds, state, tag=tag)
    print(f"  {OK} artifacts written under tag '{tag}'")

    print(f"\n[5/6] Save retrain figures")
    try:
        generate_figures(ds, state, tag=tag)
        print(f"  {OK} figures -> artifacts/figures/")
    except Exception as e:
        print(f"  [!] figure generation skipped: {e}")

    print(f"\n[6/6] Done. Use `list {ds}` to review, `predict {ds} <file>` to infer.")
    print(f"  {'=' * 50}\n")


# ============================================================
# COMMAND: interactive
# ============================================================
def cmd_interactive(args):
    ds = args.dataset
    meta = M.load_metadata(ds)
    if not meta:
        raise FileNotFoundError(f"No trained pipeline for '{ds}'.")
    threshold = float(args.threshold) if args.threshold is not None else meta["threshold"]
    pipe, name, _ = _pick_model(ds, args.model, args.tag)
    s = M.spec(ds)
    print(_banner(f"INTERACTIVE MODE: {ds.upper()}"))
    print(f"  Loaded model : {name}   threshold {threshold:.3f}")
    print(f"  Positive     : 1 = '{meta['positive_class']}'")
    codes = meta.get("category_codes_serialisable", {})

    print(f"\n  Enter the {len(meta['feature_cols'])} feature values. Type 'quit' to exit.\n")
    while True:
        print(f"  {_hr(50)}")
        print(f"  New prediction")
        print(f"  {_hr(50)}")
        rec = {}
        for c in meta["feature_cols"]:
            hint = ""
            if c in codes:
                vals = codes[c]
                if isinstance(vals, dict):
                    hint = f"  (e.g. {list(vals.values())[:4]})"
                else:
                    hint = f"  (one of {list(vals)[:5]})"
            raw = input(f"    {c}{hint}: ").strip()
            if raw.lower() in ("quit", "exit", "q"):
                print("  Goodbye!")
                return
            if c in meta.get("numeric_cols", []):
                try:
                    rec[c] = float(raw)
                except ValueError:
                    rec[c] = np.nan
            else:
                rec[c] = raw if raw else np.nan
        df = pd.DataFrame([rec])
        X = _align_features(ds, df, meta)
        p = pipe.predict_proba(X)[0, 1]
        cls = meta['class_names'][1] if p >= threshold else meta['class_names'][0]
        flag = "POSITIVE" if p >= threshold else "negative"
        print(f"\n    {ARROW}> P({meta['positive_class']}) = {p:.4f}   -> {flag} ({cls})")
        print(f"    (threshold {threshold:.3f}; educational prototype - not clinical advice)\n")


# ============================================================
# COMMAND: explain  (decision path tracing, lab Step 16)
# ============================================================
def cmd_explain(args):
    ds = args.dataset
    meta = M.load_metadata(ds)
    if not meta:
        raise FileNotFoundError(f"No trained pipeline for '{ds}'.")
    threshold = float(args.threshold) if args.threshold is not None else meta["threshold"]
    print(_banner(f"EXPLAIN - {ds.upper()} ({meta['dataset_display_name']})"))

    pipe, name, _ = _pick_model(ds, args.model, args.tag)
    inner = pipe.named_steps.get("model") if hasattr(pipe, "named_steps") else pipe
    is_tree = "sklearn.tree" in type(inner).__module__
    if not is_tree:
        print(f"\n  Selected model '{name}' is not a single Decision Tree "
              f"(it is a {type(inner).__name__}). Showing global importance instead.\n")

    df = load_data(args.filepath)
    X = _align_features(ds, df, meta)
    proba = pipe.predict_proba(X)[:, 1]
    idx = args.index
    if idx < 0 or idx >= len(X):
        raise ValueError(f"--index must be in [0, {len(X)-1}].")
    x0 = X.iloc[[idx]]

    print(f"\n  Record index {idx}:")
    for c in x0.columns[:12]:
        print(f"    {c}: {x0.iloc[0][c]}")
    if x0.shape[1] > 12:
        print(f"    ... +{x0.shape[1]-12} more feature(s)")
    print(f"\n  Predicted P({meta['positive_class']}) = {proba[idx]:.4f}")
    print(f"  Decision at threshold {threshold:.3f}: "
          f"{'POSITIVE' if proba[idx] >= threshold else 'negative'}")

    if is_tree:
        from sklearn.tree import export_text
        try:
            print(f"\n  Decision path through the tree ({name}):")
            print(_hr(60))
            # transform via the preprocessor then trace
            Xt = pipe.named_steps["pre"].transform(x0)
            feature_names = _tree_feature_names(pipe)
            node_idx = inner.decision_path(Xt).indices
            leaf = inner.apply(Xt)[0]
            path_desc = export_text(inner, feature_names=feature_names, max_depth=10)
            print(path_desc[:3000])
            print(f"\n  Leaf node {leaf}; leaf class-probability of "
                  f"'{meta['positive_class']}' = {inner.predict_proba(Xt)[0,1]:.4f}")
        except Exception as e:
            print(f"  [!] path tracing failed: {e}")

    if hasattr(inner, "feature_importances_"):
        import numpy as np2
        fi = getattr(inner, "feature_importances_", None)
        if fi is not None:
            names = _tree_feature_names(pipe)
            order = np.argsort(fi)[::-1][:10]
            print(f"\n  Top feature importances (impurity, non-causal):")
            print(_hr(60))
            for j in order:
                if fi[j] > 0:
                    print(f"    {names[j]:30} {fi[j]:.4f}")
    print("\n  Reminder: dataset-derived thresholds - NOT clinically validated.\n")


def _tree_feature_names(pipe):
    """Best-effort recovery of feature names for the fitted tree inside a
    Pipeline (post-transform)."""
    try:
        pre = pipe.named_steps["pre"]
        out = []
        for _, sub, _cols, *_rest in pre.transformers_:
            if hasattr(sub, "get_feature_names_out"):
                out.extend(sub.get_feature_names_out())
            else:
                out.extend(_cols if isinstance(_cols, (list, tuple)) else [_cols])
        return [str(x) for x in out]
    except Exception:
        return None


# ============================================================
# FIGURES (used by `retrain` so artifacts are immediately complete)
# ============================================================
def generate_figures(dataset, state, tag=None):
    """Render the lab's required visualisations for one run into
    artifacts/figures/. Mirrors what each notebook does inline."""
    import matplotlib
    matplotlib.use("Agg")
    plt = M.init_plot_style()
    tag = tag or dataset
    s = state["spec"]
    y_test = state["y_test"].values
    test_prob = state["test_prob"]
    test_pred = state["test_pred"]
    threshold = state["threshold"]
    class_names = state["class_names"]
    P = M.PALETTE

    # 1. confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="medposneg", aspect="auto")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Test confusion matrix (threshold {threshold:.2f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=P["ink_primary"], fontsize=14, fontweight="bold")
    fig.tight_layout(); M.save_fig(plt, f"{tag}_confusion_matrix"); plt.close()

    # 2. ROC
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    fpr, tpr, _ = roc_curve(y_test, test_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color=P["pos"], lw=2, label=f"AUC = {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "--", color=P["ink_muted"], label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curve - final test set"); ax.legend(loc="lower right")
    M.save_fig(plt, f"{tag}_roc"); plt.close()

    # 3. PR
    prec, rec, _ = precision_recall_curve(y_test, test_prob)
    fig, ax = plt.subplots()
    ax.plot(rec, prec, color=P["cat"][2], lw=2,
            label=f"PR-AUC = {average_precision_score(y_test, test_prob):.3f}")
    ax.axhline(y_test.mean(), ls="--", color=P["ink_muted"],
               label=f"Prevalence {y_test.mean():.2f}")
    ax.set_xlabel("Recall (sensitivity)"); ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-recall curve"); ax.legend(loc="upper right")
    M.save_fig(plt, f"{tag}_pr"); plt.close()

    # 4. threshold sweep
    tdf = state["threshold_df"]
    fig, ax = plt.subplots()
    ax.plot(tdf["threshold"], tdf["sensitivity"], color=P["pos"], lw=2, label="Sensitivity")
    ax.plot(tdf["threshold"], tdf["specificity"], color=P["neg"], lw=2, label="Specificity")
    ax.axvline(threshold, ls="--", color=P["ink_muted"], label=f"selected {threshold:.2f}")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Rate")
    ax.set_title("Threshold sweep (out-of-fold)"); ax.legend(loc="lower right")
    M.save_fig(plt, f"{tag}_threshold_sweep"); plt.close()

    # 5. CV comparison bar chart
    cvt = state["cv_table"].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(cvt))
    ax.barh(y, cvt["roc_auc_mean"], xerr=cvt["roc_auc_std"],
            color=[P["cat"][i % 8] for i in range(len(cvt))],
            edgecolor=P["baseline"], height=0.6)
    ax.set_yticks(y); ax.set_yticklabels(cvt["Model"])
    ax.set_xlabel("CV ROC-AUC (mean +/- std)")
    ax.set_title("Model comparison - 5-fold cross-validation")
    ax.invert_yaxis()
    M.save_fig(plt, f"{tag}_cv_comparison"); plt.close()

    # 6. compact pruned tree (prefer tuned CART for readability even if RF won)
    try:
        from sklearn.tree import plot_tree
        tree_pipe = state["models"].get("Tuned and pruned CART") or state["best_pipe"]
        inner = tree_pipe.named_steps.get("model") if hasattr(tree_pipe, "named_steps") else tree_pipe
        if "sklearn.tree" in type(inner).__module__:
            # Recover post-transform feature names (OHE expands categoricals).
            try:
                pre = tree_pipe.named_steps["pre"]
                fnames = []
                for _, sub, cols, *_ in pre.transformers_:
                    if hasattr(sub, "get_feature_names_out"):
                        fnames.extend([str(x) for x in sub.get_feature_names_out()])
                    else:
                        fnames.extend(list(cols) if isinstance(cols, (list, tuple)) else [cols])
            except Exception:
                fnames = list(state["feature_cols"])
            fig, ax = plt.subplots(figsize=(16, 9))
            plot_tree(inner, feature_names=fnames,
                      class_names=list(class_names), filled=True, rounded=True,
                      proportion=True, precision=2, ax=ax, max_depth=4)
            ax.set_title(f"Tuned & pruned CART - {s.key} (top 4 levels)")
            M.save_fig(plt, f"{tag}_tree"); plt.close()
    except Exception:
        pass

    # 7. calibration
    try:
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import brier_score_loss
        fig, ax = plt.subplots()
        frac, meanp = calibration_curve(y_test, test_prob, n_bins=8, strategy="quantile")
        ax.plot(meanp, frac, "o-", color=P["pos"],
                label=f"Brier = {brier_score_loss(y_test, test_prob):.3f}")
        ax.plot([0, 1], [0, 1], "--", color=P["ink_muted"], label="Perfect")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration plot - final test set")
        ax.legend(loc="upper left")
        M.save_fig(plt, f"{tag}_calibration"); plt.close()
    except Exception:
        pass

    return [f for f in os.listdir(M.FIG_DIR) if f.startswith(tag)]


# ============================================================
# ARG PARSING
# ============================================================
def build_parser():
    parser = argparse.ArgumentParser(
        prog="meddiag_cli.py",
        description="Medical Diagnosis Support CLI - inference, evaluation, "
                    "retraining, explaining & model management for the three "
                    "disease-classification pipelines (Decision Trees).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/meddiag_cli.py info\n"
            "  python src/meddiag_cli.py samples breast          # isolated inference on bundled fixtures\n"
            "  python src/meddiag_cli.py list breast\n"
            "  python src/meddiag_cli.py predict breast examples/breast_sample.json\n"
            "  python src/meddiag_cli.py evaluate diabetes examples/diabetes_sample.csv\n"
            "  python src/meddiag_cli.py explain breast examples/breast_sample.json --index 0\n"
            "  python src/meddiag_cli.py interactive diabetes\n"
            "  python src/meddiag_cli.py retrain breast data/breast_cancer_wisconsin.csv\n"
            "  python src/meddiag_cli.py gui        # launch the Tkinter GUI\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    parser.add_argument("--version", action="version",
                        version="meddiag-cli 1.0 (MDI3003 Lab 02 prototype)")

    sub.add_parser("info", help="Show Python/library versions + artifact inventory")
    sub.add_parser("gui", help="Launch the Tkinter desktop GUI")

    p = sub.add_parser("samples", help="Fetch & predict on bundled example fixtures "
                                        "(`examples/<ds>_sample.json`)")
    p.add_argument("dataset", choices=list(M.CHOICES))
    p.add_argument("-o", "--output", help="Save predictions to .csv/.json")
    p.add_argument("--model", help="Use a named saved model (else the best)")
    p.add_argument("--tag", help="Artifact tag (defaults to dataset name)")
    p.add_argument("--threshold", type=float, help="Override the saved threshold")

    p = sub.add_parser("list", help="Show the model registry, CV + test metrics")
    p.add_argument("dataset", choices=list(M.CHOICES))

    p = sub.add_parser("predict", help="Run inference on a CSV/JSON file")
    p.add_argument("dataset", choices=list(M.CHOICES))
    p.add_argument("filepath", help="Path to a .csv or .json input file")
    p.add_argument("-o", "--output", help="Save predictions to .csv/.json")
    p.add_argument("--model", help="Use a named saved model (else the best)")
    p.add_argument("--tag", help="Artifact tag (defaults to dataset name)")
    p.add_argument("--threshold", type=float, help="Override the saved threshold")
    p.add_argument("--no-features", dest="show_features", action="store_false",
                   default=True, help="Do not print per-record feature values")

    p = sub.add_parser("evaluate", help="Inference + scoring vs ground-truth labels")
    p.add_argument("dataset", choices=list(M.CHOICES))
    p.add_argument("filepath", help="CSV/JSON containing the target column too")
    p.add_argument("-o", "--output", help="Save scored predictions")
    p.add_argument("--model", help="Use a named saved model")
    p.add_argument("--tag", help="Artifact tag")
    p.add_argument("--threshold", type=float, help="Override threshold")
    p.add_argument("--head", type=int, default=10, help="Rows to show (0 = all)")

    p = sub.add_parser("retrain", help="Refit the whole workflow on new data")
    p.add_argument("dataset", choices=list(M.CHOICES))
    p.add_argument("filepath", help="CSV/JSON with features + target column")
    p.add_argument("--tag", help="Filename tag for saved artifacts")

    p = sub.add_parser("interactive", help="Prompt for features and predict")
    p.add_argument("dataset", choices=list(M.CHOICES))
    p.add_argument("--model", help="Use a named saved model")
    p.add_argument("--tag", help="Artifact tag")
    p.add_argument("--threshold", type=float, help="Override threshold")

    p = sub.add_parser("explain", help="Trace one record through the decision tree")
    p.add_argument("dataset", choices=list(M.CHOICES))
    p.add_argument("filepath", help="CSV/JSON with at least one feature row")
    p.add_argument("--index", type=int, default=0, help="Row index to explain")
    p.add_argument("--model", help="Use a named saved model (a tree)")
    p.add_argument("--tag", help="Artifact tag")
    p.add_argument("--threshold", type=float, help="Override threshold")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "gui":
        # lazy import keeps `info`/CLI lightweight on headless machines
        try:
            from meddiag_gui import launch
            launch()
        except Exception as e:
            print(f"\n[ERROR] could not launch GUI: {e}\n", file=sys.stderr)
            sys.exit(2)
        return
    cmds = {
        "info": cmd_info, "samples": cmd_samples, "list": cmd_list, "predict": cmd_predict,
        "evaluate": cmd_evaluate, "retrain": cmd_retrain,
        "interactive": cmd_interactive, "explain": cmd_explain,
    }
    try:
        cmds[args.command](args)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}\n", file=sys.stderr); sys.exit(2)
    except ValueError as e:
        print(f"\n[ERROR] {e}\n", file=sys.stderr); sys.exit(2)
    except KeyboardInterrupt:
        print("\nInterrupted.\n", file=sys.stderr); sys.exit(130)


if __name__ == "__main__":
    main()
