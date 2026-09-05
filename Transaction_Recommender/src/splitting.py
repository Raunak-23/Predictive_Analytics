"""
Chronological splitting and temporal validation module for Experiment 07 Recommendation System.
Enforces strict chronological order, prevents future leakage, and validates Appendix C assertions.
"""

from typing import Tuple, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    ts_col: str,
    train_end: Union[str, pd.Timestamp],
    val_end: Union[str, pd.Timestamp],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split transaction log chronologically into train_hist, val_future, and test_future windows.

    Parameters:
        df: Input DataFrame containing timestamp column.
        ts_col: Name of datetime timestamp column.
        train_end: Cutoff date separating training history from validation window.
        val_end: Cutoff date separating validation window from test window.

    Returns:
        train_hist: Historical transactions strictly before train_end.
        val_future: Validation transactions in [train_end, val_end).
        test_future: Test transactions on or after val_end.
    """
    t_train_end = pd.to_datetime(train_end)
    t_val_end = pd.to_datetime(val_end)

    if t_train_end >= t_val_end:
        raise ValueError(
            f"train_end ({t_train_end}) must be strictly before val_end ({t_val_end})"
        )

    # Ensure timestamp column is proper datetime
    df_sorted = df.copy()
    if not np.issubdtype(df_sorted[ts_col].dtype, np.datetime64):
        df_sorted[ts_col] = pd.to_datetime(df_sorted[ts_col])

    train_hist = df_sorted[df_sorted[ts_col] < t_train_end].copy()
    val_future = df_sorted[
        (df_sorted[ts_col] >= t_train_end) & (df_sorted[ts_col] < t_val_end)
    ].copy()
    test_future = df_sorted[df_sorted[ts_col] >= t_val_end].copy()

    # Appendix C Acceptance Assertions
    if len(train_hist) == 0:
        raise AssertionError("train_hist window is empty. Verify train_end cutoff.")
    if len(val_future) == 0:
        raise AssertionError("val_future window is empty. Verify train_end / val_end cutoffs.")
    if len(test_future) == 0:
        raise AssertionError("test_future window is empty. Verify val_end cutoff.")

    max_train_ts = train_hist[ts_col].max()
    min_val_ts = val_future[ts_col].min()
    min_test_ts = test_future[ts_col].min()

    if max_train_ts >= t_train_end:
        raise AssertionError(
            f"Appendix C violation: train_hist max ts ({max_train_ts}) >= train_end ({t_train_end})"
        )
    if min_val_ts < t_train_end:
        raise AssertionError(
            f"Appendix C violation: val_future min ts ({min_val_ts}) < train_end ({t_train_end})"
        )
    if min_test_ts < t_val_end:
        raise AssertionError(
            f"Appendix C violation: test_future min ts ({min_test_ts}) < val_end ({t_val_end})"
        )

    # Disjoint index test
    disjoint_check = set(test_future.index).isdisjoint(set(train_hist.index))
    if not disjoint_check:
        raise AssertionError(
            "Appendix C violation: test_future and train_hist have overlapping row indices."
        )

    return train_hist, val_future, test_future


def split_timeline_plot(
    train_hist: pd.DataFrame,
    val_future: pd.DataFrame,
    test_future: pd.DataFrame,
    ts_col: str = "ts",
) -> plt.Figure:
    """
    Generate horizontal timeline diagram illustrating the three non-overlapping chronological windows.

    Parameters:
        train_hist: Training history DataFrame.
        val_future: Validation window DataFrame.
        test_future: Test window DataFrame.
        ts_col: Timestamp column name.

    Returns:
        fig: Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=150)

    # Window bounds
    t0 = train_hist[ts_col].min()
    t1 = train_hist[ts_col].max()
    t2 = val_future[ts_col].min()
    t3 = val_future[ts_col].max()
    t4 = test_future[ts_col].min()
    t5 = test_future[ts_col].max()

    total_rows = len(train_hist) + len(val_future) + len(test_future)
    pct_train = (len(train_hist) / total_rows) * 100.0
    pct_val = (len(val_future) / total_rows) * 100.0
    pct_test = (len(test_future) / total_rows) * 100.0

    # Horizontal bar segments
    y_pos = 0.5
    bar_height = 0.4

    # Convert to matplotlib dates
    d0, d1 = mdates.date2num(t0), mdates.date2num(t1)
    d2, d3 = mdates.date2num(t2), mdates.date2num(t3)
    d4, d5 = mdates.date2num(t4), mdates.date2num(t5)

    ax.barh(y_pos, d1 - d0, left=d0, height=bar_height, color="#1f77b4", edgecolor="#114b75", label=f"Train History ({pct_train:.1f}%)")
    ax.barh(y_pos, d3 - d2, left=d2, height=bar_height, color="#ff7f0e", edgecolor="#b35500", label=f"Validation Window ({pct_val:.1f}%)")
    ax.barh(y_pos, d5 - d4, left=d4, height=bar_height, color="#2ca02c", edgecolor="#1b6e1b", label=f"Locked Test Window ({pct_test:.1f}%)")

    # Annotate row counts inside or above bars
    ax.text((d0 + d1) / 2, y_pos, f"Train: {len(train_hist):,} txns", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    ax.text((d2 + d3) / 2, y_pos, f"Val: {len(val_future):,} txns", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    ax.text((d4 + d5) / 2, y_pos, f"Test: {len(test_future):,} txns", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_title("Chronological Train / Validation / Test Windows (Leakage Prevention)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.25), ncol=3, frameon=True)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()

    return fig
