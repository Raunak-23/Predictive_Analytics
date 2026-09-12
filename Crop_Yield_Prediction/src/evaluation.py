"""Evaluation metrics and visualization utilities for Experiment 08."""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Only switch to non-interactive Agg backend if running in headless mode outside Jupyter/IPython
if "ipykernel" not in sys.modules and "IPython" not in sys.modules:
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend for headless CLI runs
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes standard regression metrics (MAE in t/ha, RMSE in t/ha, R2).

    Note: MAPE is omitted as zero/small yields cause unstable ratios.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))

    # Compute R2 safely: R2 is undefined for zero-variance targets
    if len(y_true_arr) > 1 and np.ptp(y_true_arr) > 0:
        r2 = float(r2_score(y_true_arr, y_pred_arr))
    else:
        r2 = float("nan")

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4) if not np.isnan(r2) else float("nan"),
    }


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes standard classification metrics (Macro F1, Accuracy, Macro Precision, Macro Recall)."""
    acc = float(accuracy_score(y_true, y_pred))
    pr, re, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(pr), 4),
        "macro_recall": round(float(re), 4),
        "macro_F1": round(float(f1), 4),
    }


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    title: str = "Locked-Test Rice Yield: Actual vs Predicted",
) -> None:
    """Generates and saves Actual vs Predicted scatter plot with 45-degree identity line."""
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=16, edgecolors="none", color="#1f77b4")

    # 45-degree reference line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5, label="Perfect Prediction (y=x)")

    plt.title(title, fontsize=12, pad=10)
    plt.xlabel("Actual Yield (t/ha)", fontsize=11)
    plt.ylabel("Predicted Yield (t/ha)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    title: str = "Locked-Test Residuals vs Predicted",
) -> None:
    """Generates and saves Residuals (Actual - Predicted) vs Predicted plot.

    Positive residual indicates underprediction; negative indicates overprediction.
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(7, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, s=16, edgecolors="none", color="#d62728")
    plt.axhline(0, color="black", linestyle="--", lw=1.5, label="Zero Error")

    plt.title(title, fontsize=12, pad=10)
    plt.xlabel("Predicted Yield (t/ha)", fontsize=11)
    plt.ylabel("Residual [Actual - Predicted] (t/ha)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_model_comparison(
    results_df: pd.DataFrame,
    save_path: Path,
    metric_col: str = "MAE",
    task_name: str = "Regression Model Comparison",
) -> None:
    """Generates bar chart comparing candidate models by primary metric."""
    plt.figure(figsize=(8, 5))
    bars = plt.bar(results_df["model"], results_df[metric_col], color="#2ca02c", width=0.5)
    plt.title(f"{task_name} ({metric_col})", fontsize=12, pad=10)
    plt.xlabel("Model Family", fontsize=11)
    plt.ylabel(f"{metric_col} ({'t/ha' if metric_col in ['MAE', 'RMSE'] else 'Score'})", fontsize=11)
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    # Add numeric labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix_display(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    save_path: Path,
) -> None:
    """Generates and saves multiclass confusion matrix."""
    fig, ax = plt.subplots(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=90, colorbar=True)
    plt.title("Classification Locked-Test Confusion Matrix", fontsize=13, pad=12)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
