"""Robustness evaluation module for Experiment 08.

Implements:
1. Three development-only rolling-origin evaluations (Section 25).
2. Year-wise test robustness diagnostics (Section 30).
"""

from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

try:
    from src.config import REG_FEATURES, REG_TARGET, RESULTS_REG_DIR
    from src.evaluation import evaluate_regression
    from src.utils import setup_logger
except ImportError:
    from config import REG_FEATURES, REG_TARGET, RESULTS_REG_DIR
    from evaluation import evaluate_regression
    from utils import setup_logger

logger = setup_logger("robustness")


def evaluate_development_rolling_origins(
    df_dev: pd.DataFrame,
    candidate_models: Dict[str, Pipeline],
    output_csv: Path = RESULTS_REG_DIR / "rolling_origins.csv",
) -> pd.DataFrame:
    """Performs three rolling-origin backtests strictly within development years.

    For each origin year T in the last 3 development years:
        - Train = all development observations strictly before T (year < T)
        - Evaluation = observations at year T (year == T)
        - Fresh pipeline fitted from scratch.
    """
    dev_years = sorted(df_dev["year"].unique())
    if len(dev_years) < 4:
        raise ValueError(f"Need at least 4 development years for rolling origins; found {len(dev_years)}")

    origins = dev_years[-3:]
    logger.info("Evaluating rolling origins on development years: %s", origins)

    records: List[dict] = []

    for origin in origins:
        past = df_dev[df_dev["year"] < origin]
        future = df_dev[df_dev["year"] == origin]
        train_max_year = int(past["year"].max())

        logger.info(
            "Rolling Origin %d: Training on <= %d (%d rows), Evaluating on %d (%d rows)",
            origin,
            train_max_year,
            len(past),
            origin,
            len(future),
        )

        for name, pipe in candidate_models.items():
            model = clone(pipe)
            model.fit(past[REG_FEATURES], past[REG_TARGET])
            pred = model.predict(future[REG_FEATURES])
            metrics = evaluate_regression(future[REG_TARGET], pred)

            records.append(
                {
                    "origin_year": int(origin),
                    "model": name,
                    "training_max_year": train_max_year,
                    "n_eval": len(future),
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "R2": metrics["R2"],
                }
            )

    results_df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    logger.info("Rolling-origin diagnostics saved to %s", output_csv)
    return results_df


def evaluate_year_robustness(
    df_test: pd.DataFrame,
    predictions: np.ndarray,
    output_csv: Path = RESULTS_REG_DIR / "year_robustness.csv",
) -> pd.DataFrame:
    """Computes test metrics separately for each test year to evaluate temporal robustness."""
    df_eval = df_test.copy()
    df_eval["prediction"] = predictions

    records: List[dict] = []
    for yr, group in df_eval.groupby("year"):
        metrics = evaluate_regression(group[REG_TARGET], group["prediction"])
        records.append(
            {
                "year": int(yr),
                "n_samples": len(group),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            }
        )

    results_df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    logger.info("Year-wise test robustness saved to %s", output_csv)
    return results_df
