#!/usr/bin/env python3
"""
meddiag_streamlit — Streamlit GUI for the Medical Diagnosis Support project
(Lab 02 — Disease Classification using Decision Trees).

A clean, non-technical interface that loads the saved artefacts and lets a user:
  * Select a dataset (breast / diabetes)
  * Draw a random sample from the cached dataset (fetched once from `data/`)
  * Adjust feature values via spinboxes (numeric) and dropdowns (categorical)
  * Tune the operating threshold with a live slider
  * Run inference and see predicted probability, class call and the decision
    path traced through the fitted Decision Tree
  * View model metadata, CV comparison table, confusion matrix, ROC curve,
    premium-recall curve and calibration chart — all rendered from the saved
    artefacts in `artifacts/`

No retraining is ever triggered. Everything loads from the joblib/JSON/CSV
files persisted by `meddiag_common.save_artifacts`.

Usage
-----
  streamlit run src/meddiag_streamlit.py
"""

from __future__ import annotations

import json
import os
import sys
import random

import numpy as np
import pandas as pd

# -- paths --------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import meddiag_common as M  # noqa: E402

# -- Streamlit ----------------------------------------------------------
import streamlit as st

st.set_page_config(
    page_title="Medical Decision Support UI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── session helpers ──────────────────────────────────────────────────
if "ds" not in st.session_state:
    st.session_state.ds = None
if "loaded" not in st.session_state:
    st.session_state.loaded = False
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.5
if "features" not in st.session_state:
    st.session_state.features = {}
if "sample_prob" not in st.session_state:
    st.session_state.sample_prob = None
if "predicted" not in st.session_state:
    st.session_state.predicted = False

# =========================================================================
# dataset-level initialisation
# =========================================================================
def load_artefacts(dataset_key: str) -> dict:
    """Load everything needed from saved artefacts; raise if missing."""
    meta = M.load_metadata(dataset_key)
    if not meta:
        st.error(f"No trained artefacts for **{dataset_key}** — run `tools/train_all.py --datasets {dataset_key}` first.")
        st.stop()
    pipe, _name, _ = M.choose_model(dataset_key)
    return {"meta": meta, "pipe": pipe}


def draw_random_sample(dataset_key: str, meta: dict) -> dict:
    """Pull one random row from the cached CSV, dropping the target column.

    Returns a flat dict {feature: raw_value} that mirrors the dataset's
    raw feature values.
    """
    s = M.spec(dataset_key)
    df = M.load_dataset(dataset_key)
    # keep only the feature cols we know about
    cols = [c for c in meta.get("feature_cols", []) if c in df.columns]
    row = df[cols].sample(1, random_state=random.randint(0, 10_000)).iloc[0]
    return {c: row[c] for c in cols}


def align_one_record(features_dict: dict, meta: dict) -> pd.DataFrame:
    """Turn the per-feature dict into a DataFrame the pipeline accepts."""
    ds = meta["dataset_key"]
    df = pd.DataFrame([features_dict])
    # Try CLI helper first; fall back to a self-contained aligner
    try:
        from meddiag_cli import _align_features
        return _align_features(ds, df, meta)
    except Exception:
        return _align_features_inline(ds, df, meta)


def _align_features_inline(dataset_key: str, df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Inline aligner: reorders/casts columns to match the saved pipeline.

    Uses the metadata's ``feature_cols`` and ``categorical_cols`` to build a
    DataFrame whose columns exactly match what the fitted Pipeline expects.
    """
    feature_cols = list(meta.get("feature_cols", []))
    cat_cols = [c for c in meta.get("categorical_cols", []) if c in feature_cols]

    # Reindex to the training column order; add missing cols as NaN, drop extras
    aligned = df.reindex(columns=feature_cols)

    # Cast categoricals to string so the saved encoder handles them
    for c in cat_cols:
        if c in aligned.columns:
            aligned[c] = aligned[c].astype(str)

    return aligned


def decide(proba: float, threshold: float, meta: dict) -> str:
    """Return the text class label."""
    class_names = meta.get("class_names", ["Negative", "Positive"])
    return class_names[1] if proba >= threshold else class_names[0]

# =========================================================================
# TITLE + BAR
# =========================================================================
st.title("🩺 Medical Decision-Support Prototype")
st.caption(
    "**Educational prototype for MDI3003 Lab 02** — Decision Tree classification | "
    "No re-training, everything loads from saved artefacts | "
    "**Not a clinical diagnostic system**"
)

col_ds, col_btn, col_sp = st.columns([3, 1, 4])
with col_ds:
    ds = st.selectbox("Dataset", list(M.CHOICES),
                       format_func=lambda k: M.DATASETS[k].display_name)
with col_btn:
    st.write("");  # spacer
    st.write("")
    if st.button("🔄 Load model", width="stretch"):
        data = load_artefacts(ds)
        st.session_state.ds = ds
        st.session_state.meta = data["meta"]
        st.session_state.pipe = data["pipe"]
        st.session_state.threshold = float(data["meta"]["threshold"])
        st.session_state.loaded = True
        st.session_state.features = {}
        st.session_state.sample_prob = None
        st.session_state.predicted = False
        st.rerun()
with col_sp:
    if st.session_state.loaded:
        m = st.session_state.meta
        st.success(f"**{st.session_state.ds.upper()}** loaded — {m['best_model']} threshold={m['threshold']:.3f}")

if not st.session_state.loaded:
    st.info("👆 Click **Load model** to activate the UI.")
    st.stop()

# =========================================================================
# MAIN LAYOUT: two columns (inputs left, outputs right)
# =========================================================================
meta = st.session_state["meta"]
pipe = st.session_state["pipe"]
ds = st.session_state["ds"]
s = M.spec(ds)
thr = st.session_state.threshold

left, right = st.columns(2, gap="medium")

# -------------------------------------------------------------------------
# LEFT COLUMN : features + threshold
# -------------------------------------------------------------------------
with left:
    st.subheader("🧪 Feature Values")
    st.caption("Edit values below — then click **Predict** to re-run inference.")

    # -- Threshold slider (live) ------------------------------------------
    st.slider(
        "Operating threshold",
        min_value=0.0, max_value=1.0,
        value=st.session_state["threshold"],
        step=0.005,
        format="%.3f",
        key="thr_slider",
        help="Adjust the probability cutoff used to call Positive vs Negative.",
    )
    st.session_state.threshold = st.session_state["thr_slider"]

    # -- Buttons row -------------------------------------------------------
    b1, b2 = st.columns(2)
    with b1:
        if st.button("🎲 Draw random sample", width="stretch"):
            features = draw_random_sample(meta["dataset_key"], meta)
            st.session_state.features = features
            st.session_state.sample_prob = None
            st.session_state.predicted = False
    with b2:
        if st.button("▶️ Predict", type="primary", width="stretch"):
            if not st.session_state.features:
                st.warning("Pull a random sample first.")
            else:
                x_df = align_one_record(st.session_state.features, meta)
                v_proba = pipe.predict_proba(x_df)[0, 1]
                st.session_state.sample_prob = v_proba
                st.session_state.predicted = True

    # -- Feature inputs ---------------------------------------------------------
    features = st.session_state.features
    if not features:
        st.info("Click **Draw random sample** → a fresh row from the dataset will appear here.")
        st.stop()

    ff = {}
    raw_cols = [c for c in meta.get("feature_cols", []) if c not in meta.get("categorical_cols", [])]
    cat_cols = [c for c in meta.get("categorical_cols", []) if c in meta.get("feature_cols", [])]

    st.write("#### Numeric")
    for c in raw_cols:
        default = features.get(c, 0.0)
        ff[c] = st.number_input(
            f"**{c}**", value=float(default) if pd.notnull(default) else 0.0,
            key=f"n_{c}", step=0.1, format="%.4f",
        )

    codes = meta.get("category_codes_serialisable", {})
    st.write("#### Categorical")
    for c in cat_cols:
        opts = codes.get(c, [])
        if isinstance(opts, dict):
            opts = list(opts.values())
        curr = features.get(c, opts[0] if opts else "")
        # ensure initial value is in opts
        if curr not in opts and opts:
            curr = opts[0]
        ff[c] = st.selectbox(
            f"**{c}**", options=opts, index=opts.index(curr) if curr in opts else 0,
            key=f"c_{c}",
        )
    st.session_state.features = ff

# -------------------------------------------------------------------------
# RIGHT COLUMN : prediction result + charts
# -------------------------------------------------------------------------
with right:
    st.subheader("🩻 Prediction")
    proba = st.session_state.sample_prob
    thr = st.session_state.threshold

    if proba is not None and st.session_state.predicted:
        thr = st.session_state.threshold
        class_names = meta["class_names"]
        cls = class_names[1] if proba >= thr else class_names[0]
        is_pos = proba >= thr
        st.metric(
            label=f"P({meta['positive_class']})",
            value=f"{proba:.4f}",
            delta=f"{'🔴 Positive' if is_pos else '🟢 Negative'} ({cls})",
        )
    else:
        st.info("Click **Predict** to run inference on the selected feature values.")

    # -------- Model info --------------------------------------------------
    with st.expander("📋  Model info &amp; CV summary", expanded=False):
        st.markdown(f"**Dataset:** {meta['dataset_display_name']}")
        st.markdown(f"**Best model:** {meta['best_model']} | **Threshold:** {meta['threshold']:.3f}")
        st.markdown(f"**Positive class:** 1 = {meta['positive_class']} | Raw: {meta['positive_raw_values']}")
        st.markdown(f"**Rows / train / test:** {meta['n_rows']} / {meta['n_train']} / {meta['n_test']}")
        st.markdown(f"**Prevalence:** train {meta['train_prevalence']:.3f} | test {meta['test_prevalence']:.3f}")

        cv_csv = os.path.join(M.ART_DIR, f"{ds}_cv_results.csv")
        if os.path.exists(cv_csv):
            cv_df = pd.read_csv(cv_csv)
            st.markdown("#### CV table (5-fold cross-validation)")
            st.dataframe(cv_df, width="stretch")

    # -------- charts --------------------------------------------------------
    with st.expander("📊   Evaluation charts", expanded=False):
        fig_dir = M.FIG_DIR
        figs_available = sorted([
            fn for fn in os.listdir(fig_dir) if fn.startswith(ds) and fn.endswith(".png")
        ])
        if not figs_available:
            st.info("No figures found in `artifacts/figures/` — run `tools/train_all.py` to create them.")
        else:
            cols = st.columns(2)
            for i, fn in enumerate(figs_available):
                label = fn[len(ds) + 1:].replace(".png", "").replace("_", " ").title()
                with cols[i % 2]:
                    st.image(
                        os.path.join(fig_dir, fn),
                        caption=label,
                        width="stretch",
                    )

# =========================================================================
# FOOTER
# =========================================================================
st.divider()
st.caption(
    "**Educational use only.** Not a clinical diagnostic system.\n"
    f"Saved aARCH facts at `artifacts/` — pipeline+fitted model+threshold+metadata+figures.\n"
    f"Author: Raunak Pal | Reg: 23MID0045 | MDI3003 Lab 02"
)