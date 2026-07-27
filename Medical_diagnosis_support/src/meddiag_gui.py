#!/usr/bin/env python3
"""
meddiag_gui - Tkinter desktop GUI for the Medical Diagnosis Support lab.

A single window that lets a user:
  * pick one of the three trained disease-classification pipelines,
  * enter feature values (controls: numeric spinboxes for continuous features,
    dropdowns enumerated from the dataset's category codes for categoricals),
  * slide the operating threshold and watch the predicted probability +
    binary call update live,
  * inspect the saved model's metadata, CV table and a confusion-matrix preview,
  * view the compact decision tree and trace the decision path of the record.

It reuses ``meddiag_common`` (the same engine as the notebooks and CLI), so the
GUI can never disagree with the command-line or notebook results.

Launch with:
    python src/meddiag_cli.py gui
or
    python src/meddiag_gui.py

Educational-use boundary: this GUI renders an educational research prototype.
It is NOT a clinical diagnostic tool and must not be used for patient care.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import meddiag_common as M  # noqa: E402

import tkinter as tk  # noqa: E402
from tkinter import ttk, messagebox  # noqa: E402


# ============================================================
# MODEL WRAPPER - caches the loaded pipeline + metadata per dataset
# ============================================================
class ModelStore:
    def __init__(self):
        self._cache = {}

    def get(self, dataset):
        if dataset not in self._cache:
            meta = M.load_metadata(dataset)
            if not meta:
                raise FileNotFoundError(
                    f"No trained pipeline for '{dataset}'. "
                    f"Run the notebook or `python src/meddiag_cli.py retrain {dataset} "
                    f"data/<file>.csv` first.")
            pipe, name, _ = M.choose_model(dataset)
            self._cache[dataset] = {"meta": meta, "pipe": pipe, "name": name,
                                    "threshold": float(meta.get("threshold", 0.5))}
        return self._cache[dataset]


# ============================================================
# GUI - the interactive notebook
# ============================================================
class MedDiagGUI:
    def __init__(self, root):
        self.root = root
        self.store = ModelStore()
        self.current = None
        self.entries = {}      # feature_name -> tk variable
        self.dataset_var = tk.StringVar()
        self.threshold_var = tk.DoubleVar()
        self.prob_var = tk.DoubleVar()
        self.class_var = tk.StringVar()

        self._build_chrome()
        self._build_layout()
        self._populate_dataset_dropdown()
        self._safe_first_dataset()

    # -- chrome -------------------------------------------------------------
    def _build_chrome(self):
        self.root.title("Medical Diagnosis Support - Decision Tree prototype (educational)")
        self.root.geometry("1080x740")
        self.root.minsize(900, 640)
        try:
            self.root.tk.call("tk", "scaling", 1.1)
        except Exception:
            pass
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Hero.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Stat.TLabel", font=("Segoe UI", 11))
        style.configure("Warn.TLabel", foreground="#7a4a00",
                        background="#fcf3e0", font=("Segoe UI", 9))
        style.configure("TFrame", background="#ffffff")
        style.configure("TLabel", background="#ffffff")
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("TButton", padding=4)

    def _build_layout(self):
        nav = ttk.Frame(self.root, padding=(10, 8))
        nav.pack(fill="x")
        ttk.Label(nav, text="Dataset: ").pack(side="left")
        self.dataset_cb = ttk.Combobox(nav, textvariable=self.dataset_var,
                                       state="readonly", width=40)
        self.dataset_cb.pack(side="left", padx=(4, 10))
        self.dataset_cb.bind("<<ComboboxSelected>>", self._on_dataset_change)
        ttk.Button(nav, text="Load model", command=self._load_model).pack(side="left", padx=4)
        ttk.Button(nav, text="Info / metadata", command=self._show_info).pack(side="left", padx=4)
        ttk.Button(nav, text="CV table", command=self._show_cv).pack(side="left", padx=4)

        ttk.Separator(self.root).pack(fill="x")

        body = ttk.Frame(self.root, padding=10)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # LEFT - feature inputs
        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.inp_frame = ttk.LabelFrame(left, text="Feature inputs", padding=8)
        self.inp_frame.pack(fill="both", expand=True)
        self.no_model_lbl = ttk.Label(
            self.inp_frame, text="Select a dataset and click 'Load model'.\n"
            "If no pipeline exists yet, run the notebook or:\n"
            "python src/meddiag_cli.py retrain <dataset> data/<file>.csv",
            justify="left", style="Warn.TLabel", padding=10)
        self.no_model_lbl.pack(fill="x")

        # RIGHT - prediction + explanation
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")

        pred = ttk.LabelFrame(right, text="Prediction", padding=10)
        pred.pack(fill="x", pady=(0, 8))
        self.lbl_prob = ttk.Label(pred, text="P(disease) = --", style="Hero.TLabel")
        self.lbl_prob.pack(anchor="w")
        self.lbl_class = ttk.Label(pred, text="--", style="Stat.TLabel")
        self.lbl_class.pack(anchor="w", pady=(2, 6))

        thr_row = ttk.Frame(pred)
        thr_row.pack(fill="x")
        ttk.Label(thr_row, text="Threshold").pack(side="left")
        self.thr_slider = ttk.Scale(thr_row, from_=0.05, to=0.95,
                                    variable=self.threshold_var,
                                    command=self._on_threshold)
        self.thr_slider.pack(side="left", fill="x", expand=True, padx=8)
        self.lbl_thr = ttk.Label(thr_row, text="0.50")
        self.lbl_thr.pack(side="left")
        self.threshold_var.trace_add("write", self._on_threshold_var)
        self.lbl_cost = ttk.Label(pred, text="", style="Warn.TLabel")
        self.lbl_cost.pack(anchor="w", pady=(6, 0))

        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=4)
        ttk.Button(btns, text="Predict", command=self._predict).pack(side="left")
        ttk.Button(btns, text="Reset", command=self._reset_inputs).pack(side="left", padx=6)
        ttk.Button(btns, text="Use sample row", command=self._use_sample_row).pack(side="left")
        ttk.Button(btns, text="Explain path", command=self._explain).pack(side="left", padx=6)

        tree_frame = ttk.LabelFrame(right, text="Decision path / tree", padding=8)
        tree_frame.pack(fill="both", expand=True)
        self.txt_tree = tk.Text(tree_frame, height=18, wrap="none", relief="flat",
                                background="#fbfbfa", font=("Consolas", 9))
        self.txt_tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(tree_frame, command=self.txt_tree.yview)
        sb.pack(side="right", fill="y")
        self.txt_tree.config(yscrollcommand=sb.set)
        self.txt_tree.insert("end", "(Load a model, then choose 'Explain path'.)")
        self.txt_tree.config(state="disabled")

        warn = ttk.Label(
            self.root,
            text=("Educational research prototype - NOT a clinical diagnostic system. "
                  "Do not use for patient care, treatment or triage."),
            style="Warn.TLabel", padding=(10, 6))
        warn.pack(side="bottom", fill="x")

    # -- dataset dropdown ---------------------------------------------------
    def _populate_dataset_dropdown(self):
        labels = []
        self._label_to_key = {}
        for k in M.CHOICES:
            s = M.spec(k)
            meta = M.load_metadata(k)
            tag = "(trained)" if meta else "(not trained)"
            lab = f"{s.display_name}  [{k}]  {tag}"
            labels.append(lab)
            self._label_to_key[lab] = k
        self.dataset_cb["values"] = labels

    def _safe_first_dataset(self):
        for k in M.CHOICES:
            if M.load_metadata(k):
                self._set_dataset(k)
                return

    def _set_dataset(self, key):
        for lab, k in self._label_to_key.items():
            if k == key:
                self.dataset_var.set(lab)
                return

    def _current_key(self):
        lab = self.dataset_var.get()
        return self._label_to_key.get(lab)

    def _on_dataset_change(self, *_):
        pass  # wait for 'Load model'

    # -- model loading + UI build ------------------------------------------
    def _clear_inputs(self):
        for w in self.inp_frame.winfo_children():
            w.destroy()
        self.entries = {}

    def _load_model(self):
        key = self._current_key()
        if not key:
            messagebox.showinfo("No dataset", "Pick a dataset first.")
            return
        try:
            info = self.store.get(key)
        except FileNotFoundError as e:
            messagebox.showwarning("No trained pipeline", str(e))
            return
        self.current = info
        meta = info["meta"]
        self.threshold_var.set(round(info["threshold"], 3))
        self._build_input_controls(meta)
        self._refresh_threshold_label()

    def _build_input_controls(self, meta):
        self._clear_inputs()
        s = M.spec(meta["dataset_key"])
        codes = meta.get("category_codes_serialisable", {})
        canvas_frame = self.inp_frame
        # two-column grid of controls
        cols = 2
        i = 0
        for c in meta["feature_cols"]:
            row, col = divmod(i, cols)
            cell = ttk.Frame(canvas_frame)
            cell.grid(row=row, column=col, sticky="ew", padx=6, pady=4)
            canvas_frame.columnconfigure(col, weight=1)
            ttk.Label(cell, text=c, font=("Segoe UI", 9, "bold")).pack(anchor="w")
            if c in meta.get("numeric_cols", []):
                # numeric: a spinbox seeded at the feature's median (or 0)
                val = tk.StringVar()
                rec_df = self._repr_record_df(meta)
                try:
                    init = float(rec_df[c].median()) if c in rec_df.columns else 0.0
                except Exception:
                    init = 0.0
                val.set(f"{init:.2f}")
                # adjust width for the feature range when known
                from tkinter import Spinbox
                sp = Spinbox(cell, from_=-1e9, to=1e9, increment=1, textvariable=val,
                             width=14, font=("Consolas", 9))
                sp.pack(anchor="w")
                self.entries[c] = ("numeric", val)
            else:
                # categorical: dropdown enumerated from documented codes
                vallist = codes.get(c, [])
                if isinstance(vallist, dict):
                    vallist = list(vallist.values())
                if not vallist and c in self._repr_record_df(meta).columns:
                    vallist = sorted(self._repr_record_df(meta)[c].dropna().astype(str).unique().tolist())
                opts = list(vallist) + ["(unknown)"]
                val = tk.StringVar()
                val.set(opts[0] if opts else "")
                cb = ttk.Combobox(cell, textvariable=val, values=opts, state="readonly",
                                  width=16, font=("Segoe UI", 9))
                cb.pack(anchor="w")
                self.entries[c] = ("categorical", val)
            i += 1
        # give the grid rows equal share
        for r in range((len(meta["feature_cols"]) + cols - 1) // cols):
            canvas_frame.rowconfigure(r, weight=1)

    def _repr_record_df(self, meta):
        # a representative frame used to seed controls (medians / category codes)
        try:
            return M.load_dataset(meta["dataset_key"])
        except Exception:
            return pd.DataFrame()

    def _reset_inputs(self):
        if self.current:
            self._build_input_controls(self.current["meta"])

    def _use_sample_row(self):
        if not self.current:
            return
        meta = self.current["meta"]
        df = self._repr_record_df(meta)
        if df.empty:
            return
        X, _ = M.make_xy(meta["dataset_key"], df)
        row = X.iloc[0]
        codes = meta.get("category_codes_serialisable", {})
        for c, (kind, var) in self.entries.items():
            if c in row:
                if kind == "numeric":
                    try:
                        var.set(f"{float(row[c]):.2f}")
                    except Exception:
                        pass
                else:
                    vlist = codes.get(c, [])
                    if isinstance(vlist, dict):
                        vlist = list(vlist.values())
                    strv = str(row[c])
                    val = strv if (not vlist or strv in vlist) else (vlist[0] if vlist else strv)
                    var.set(val)
        self._predict()

    # -- prediction ---------------------------------------------------------
    def _on_threshold(self, *_):
        self._refresh_threshold_label()
        self._predict()

    def _on_threshold_var(self, *_):
        self._refresh_threshold_label()
        self._predict()

    def _refresh_threshold_label(self):
        t = self.threshold_var.get()
        self.lbl_thr.config(text=f"{t:.3f}")

    def _gather_record(self, meta):
        rec = {}
        for c, (kind, var) in self.entries.items():
            raw = var.get()
            if kind == "numeric":
                try:
                    rec[c] = float(raw) if raw not in ("", "(unknown)") else np.nan
                except ValueError:
                    rec[c] = np.nan
            else:
                rec[c] = np.nan if raw == "(unknown)" else raw
        return rec

    def _predict(self):
        if not self.current:
            return
        meta = self.current["meta"]
        rec = self._gather_record(meta)
        df = pd.DataFrame([rec])
        # align to trained feature columns in order
        expected = meta["feature_cols"]
        out = pd.DataFrame(index=[0])
        for c in expected:
            out[c] = [df[c].iloc[0]] if c in df.columns else [np.nan]
        for c in meta.get("numeric_cols", []):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        for c in meta.get("categorical_cols", []):
            if c in out.columns:
                out[c] = out[c].astype("object").astype(str)
                out.loc[out[c].isin(("nan", "None", "", "NaN")), c] = np.nan
        try:
            proba = self.current["pipe"].predict_proba(out[expected])[0, 1]
        except Exception as e:
            messagebox.showerror("Prediction error", str(e))
            return
        threshold = self.threshold_var.get()
        cls = meta["class_names"][1] if proba >= threshold else meta["class_names"][0]
        flag = "POSITIVE" if proba >= threshold else "negative"
        self.prob_var.set(proba)
        self.lbl_prob.config(text=f"P({meta['positive_class']}) = {proba:.4f}",
                             foreground=M.PALETTE["pos"] if proba >= threshold else M.PALETTE["neg"])
        self.lbl_class.config(text=f"-> {flag}  ({cls})  at threshold {threshold:.3f}")
        self.lbl_cost.config(
            text=("If positive-call is wrong: FP cost = %.0f  |  if a positive is missed: "
                  "FN cost = %.0f  (instructional). True label unknown -> cost not computable.")
            % (meta["cost_fp"], meta["cost_fn"]))

    # -- explanation --------------------------------------------------------
    def _explain(self):
        if not self.current:
            return
        meta = self.current["meta"]
        pipe = self.current["pipe"]
        inner = pipe.named_steps.get("model") if hasattr(pipe, "named_steps") else pipe
        rec = self._gather_record(meta)
        df = pd.DataFrame([rec])
        expected = meta["feature_cols"]
        out = pd.DataFrame(index=[0])
        for c in expected:
            out[c] = [df[c].iloc[0]] if c in df.columns else [np.nan]
        for c in meta.get("numeric_cols", []):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        for c in meta.get("categorical_cols", []):
            if c in out.columns:
                out[c] = out[c].astype("object").astype(str)
                out.loc[out[c].isin(("nan", "None", "", "NaN")), c] = np.nan
        proba = pipe.predict_proba(out[expected])[0, 1]
        threshold = self.threshold_var.get()
        self.txt_tree.config(state="normal")
        self.txt_tree.delete("1.0", "end")
        self.txt_tree.insert("end", f"Predicted P({meta['positive_class']}) = {proba:.4f} "
                                    f"-> {'POSITIVE' if proba >= threshold else 'negative'}\n\n")
        is_tree = "sklearn.tree" in type(inner).__module__
        if not is_tree:
            self.txt_tree.insert("end",
                f"Best model here is a {type(inner).__name__} (not a single tree).\n"
                f"Showing top impurity importances instead:\n\n")
            self._importance_block(inner, pipe)
            self.txt_tree.insert("end", "\n(Dataset-derived; not clinically validated.)")
            self.txt_tree.config(state="disabled")
            return
        try:
            from sklearn.tree import export_text, plot_tree
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            Xt = pipe.named_steps["pre"].transform(out[expected])
            feature_names = self._tree_feature_names(pipe)
            feature_names = self._tree_feature_names(pipe)
            self.txt_tree.insert("end", "DECISION PATH (text rules):\n" + "-" * 50 + "\n")
            txt = export_text(inner, feature_names=feature_names, max_depth=4)
            self.txt_tree.insert("end", txt[:2500] + ("\n... (truncated)\n" if len(txt) > 2500 else "\n"))
            leaf = int(inner.apply(Xt)[0])
            self.txt_tree.insert("end", f"\nLeaf node {leaf}; "
                                        f"leaf P({meta['positive_class']}) = "
                                        f"{inner.predict_proba(Xt)[0,1]:.4f}\n\n")
            self._importance_block(inner, pipe)
        except Exception as e:
            self.txt_tree.insert("end", f"[path tracing failed: {e}]\n")
        self.txt_tree.insert("end", "\nReminders:\n"
                                    "* Dataset-derived thresholds - NOT clinically validated.\n"
                                    "* Impurity importance is biased to many-split features.\n"
                                    "* Predictive, not causal.")
        self.txt_tree.config(state="disabled")

    def _importance_block(self, inner, pipe):
        if not hasattr(inner, "feature_importances_"):
            return
        fi = getattr(inner, "feature_importances_", None)
        if fi is None:
            return
        names = self._tree_feature_names(pipe)
        order = list(np.argsort(fi)[::-1][:10])
        self.txt_tree.insert("end", "TOP FEATURE IMPORTANCES (impurity, non-causal):\n"
                                    "-" * 50 + "\n")
        if names:
            for j in order:
                if fi[j] > 0:
                    self.txt_tree.insert("end", f"  {names[j]:32} {fi[j]:.4f}\n")
        else:
            for j in order:
                if fi[j] > 0:
                    self.txt_tree.insert("end", f"  feature[{j:3}] {fi[j]:.4f}\n")

    def _tree_feature_names(self, pipe):
        try:
            pre = pipe.named_steps["pre"]
            out = []
            for _, sub, _cols, *_rest in pre.transformers_:
                if hasattr(sub, "get_feature_names_out"):
                    out.extend([str(x) for x in sub.get_feature_names_out()])
                else:
                    out.extend(_cols if isinstance(_cols, (list, tuple)) else [_cols])
            return out or None
        except Exception:
            return None

    # -- info windows -------------------------------------------------------
    def _show_info(self):
        if not self.current:
            messagebox.showinfo("No model", "Load a model first.")
            return
        meta = self.current["meta"]
        win = tk.Toplevel(self.root)
        win.title(f"Metadata - {meta['dataset_display_name']}")
        win.geometry("560x520")
        txt = tk.Text(win, wrap="word", relief="flat", font=("Consolas", 9), padx=10, pady=10)
        txt.pack(fill="both", expand=True)
        import json as _json
        view = {k: v for k, v in meta.items()
                if k not in ("category_codes", "category_codes_serialisable")}
        txt.insert("end", _json.dumps(view, indent=2, default=str))
        txt.config(state="disabled")

    def _show_cv(self):
        if not self.current:
            messagebox.showinfo("No model", "Load a model first.")
            return
        meta = self.current["meta"]
        cv_csv = os.path.join(M.ART_DIR, f"{meta['dataset_key']}_cv_results.csv")
        if not os.path.exists(cv_csv):
            messagebox.showinfo("CV", "cv_results.csv not found.")
            return
        df = pd.read_csv(cv_csv)
        win = tk.Toplevel(self.root)
        win.title(f"Cross-validation - {meta['dataset_display_name']}")
        win.geometry("640x300")
        txt = tk.Text(win, wrap="none", relief="flat", font=("Consolas", 9), padx=10, pady=10)
        txt.pack(fill="both", expand=True)
        cols = ["Model", "roc_auc_mean", "pr_auc_mean", "sensitivity_mean",
                "specificity_mean", "f1_mean"]
        cols = [c for c in cols if c in df.columns]
        txt.insert("end", df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        txt.config(state="disabled")


def launch():
    root = tk.Tk()
    MedDiagGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
