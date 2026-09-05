"""
Universal Exploratory Data Analysis (EDA) module for Experiment 07 Recommendation System.
Generates publication-quality figures and returns exact underlying statistical summaries.
"""

from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_plotting_style() -> None:
    """Set clean, publication-ready styling for matplotlib."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["figure.titlesize"] = 13


def txn_volume_over_time(
    df: pd.DataFrame,
    ts_col: str = "ts",
    freq: str = "W",
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Compute and plot transaction volume over time.

    Parameters:
        df: Input DataFrame containing timestamp column.
        ts_col: Name of datetime timestamp column.
        freq: Resampling frequency (e.g. 'D', 'W', 'M').

    Returns:
        fig: Matplotlib Figure.
        summary: Dictionary of statistical summary metrics.
    """
    set_plotting_style()
    df_ts = df.dropna(subset=[ts_col]).copy()
    df_ts = df_ts.set_index(ts_col).sort_index()

    # Resample transaction count
    volume_series = df_ts.resample(freq).size()
    total_volume = int(volume_series.sum())
    peak_date = str(volume_series.idxmax().date()) if not volume_series.empty else "N/A"
    peak_vol = int(volume_series.max()) if not volume_series.empty else 0
    mean_vol = float(volume_series.mean()) if not volume_series.empty else 0.0

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    ax.plot(
        volume_series.index,
        volume_series.values,
        color="#1f77b4",
        linewidth=2,
        marker="o",
        markersize=3,
        alpha=0.9,
        label=f"Weekly Volume ({freq})",
    )
    ax.fill_between(volume_series.index, volume_series.values, color="#1f77b4", alpha=0.15)
    ax.set_title("Transaction Volume Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Transactions")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    summary = {
        "resample_frequency": freq,
        "total_transactions": total_volume,
        "peak_period": peak_date,
        "peak_volume": peak_vol,
        "mean_volume_per_period": round(mean_vol, 2),
        "periods_analyzed": len(volume_series),
    }
    return fig, summary


def top_n_items(
    df: pd.DataFrame,
    item_col: str = "item_id",
    order_col: str = "order_id",
    n: int = 15,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Compute and plot the top N most frequently ordered items.

    Parameters:
        df: Input DataFrame.
        item_col: Item identifier column.
        order_col: Order/Invoice identifier column.
        n: Number of top items to visualize.

    Returns:
        fig: Matplotlib Figure.
        top_df: DataFrame of top N items with order counts and share percentages.
    """
    set_plotting_style()
    item_counts = (
        df.groupby(item_col)[order_col]
        .nunique()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    item_counts.columns = [item_col, "order_count"]
    total_orders = df[order_col].nunique()
    item_counts["order_share_pct"] = round(
        (item_counts["order_count"] / max(total_orders, 1)) * 100.0, 2
    )

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    bars = ax.barh(
        np.arange(len(item_counts)),
        item_counts["order_count"],
        color="#2ca02c",
        edgecolor="#1b6e1b",
        alpha=0.85,
    )
    ax.set_yticks(np.arange(len(item_counts)))
    ax.set_yticklabels(item_counts[item_col].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel("Number of Distinct Orders")
    ax.set_title(f"Top {n} Items by Distinct Order Volume")
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    # Annotate bars with counts
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + (max(item_counts["order_count"]) * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{int(w):,}",
            va="center",
            ha="left",
            fontsize=8,
            color="#222222",
        )

    plt.tight_layout()
    return fig, item_counts


def purchase_frequency_distribution(
    df: pd.DataFrame,
    user_col: str = "user_id",
    order_col: str = "order_id",
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Compute and plot customer purchase frequency (orders per user).

    Parameters:
        df: Input DataFrame.
        user_col: User identifier column.
        order_col: Order identifier column.

    Returns:
        fig: Matplotlib Figure.
        stats: Dictionary of distribution summary statistics.
    """
    set_plotting_style()
    freq = df.groupby(user_col)[order_col].nunique()

    stats = {
        "user_count": int(len(freq)),
        "mean_orders": round(float(freq.mean()), 2),
        "median_orders": round(float(freq.median()), 2),
        "std_orders": round(float(freq.std()), 2),
        "p25_orders": round(float(freq.quantile(0.25)), 2),
        "p75_orders": round(float(freq.quantile(0.75)), 2),
        "p90_orders": round(float(freq.quantile(0.90)), 2),
        "max_orders": int(freq.max()),
        "pct_single_order_users": round(float((freq == 1).mean() * 100.0), 2),
    }

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    bins = np.logspace(0, np.log10(max(freq.max(), 2)), 30)
    ax.hist(
        freq,
        bins=bins,
        color="#ff7f0e",
        edgecolor="#b35500",
        alpha=0.85,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Orders per Customer (Log Scale)")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Purchase Frequency Distribution")
    ax.axvline(
        freq.median(),
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"Median: {stats['median_orders']:.1f}",
    )
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    return fig, stats


def rfm_distributions(
    df: pd.DataFrame,
    user_col: str = "user_id",
    ts_col: str = "ts",
    amount_col: str = "amount",
    order_col: str = "order_id",
    cutoff: Optional[pd.Timestamp] = None,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Compute customer Recency, Frequency, and Monetary (RFM) distributions.

    Parameters:
        df: Input DataFrame.
        user_col: User identifier column.
        ts_col: Timestamp column.
        amount_col: Transaction amount column.
        order_col: Order identifier column.
        cutoff: Reference timestamp for recency calculation.

    Returns:
        fig: Matplotlib Figure with 3 subplots.
        rfm_df: Customer-level RFM metrics DataFrame.
    """
    set_plotting_style()
    ref_time = cutoff if cutoff is not None else df[ts_col].max()

    rfm = (
        df.groupby(user_col)
        .agg(
            last_purchase=(ts_col, "max"),
            frequency=(order_col, "nunique"),
            monetary=(amount_col, "sum"),
        )
        .reset_index()
    )
    rfm["recency"] = (ref_time - rfm["last_purchase"]).dt.total_seconds() / 86400.0
    rfm["recency"] = np.maximum(rfm["recency"], 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=150)

    # 1. Recency
    axes[0].hist(rfm["recency"], bins=25, color="#1f77b4", edgecolor="#114b75", alpha=0.8)
    axes[0].set_title("Recency (Days since Last Order)")
    axes[0].set_xlabel("Days")
    axes[0].set_ylabel("Customers")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Frequency
    axes[1].hist(
        rfm["frequency"],
        bins=np.logspace(0, np.log10(max(rfm["frequency"].max(), 2)), 25),
        color="#2ca02c",
        edgecolor="#1b6e1b",
        alpha=0.8,
    )
    axes[1].set_xscale("log")
    axes[1].set_title("Frequency (Order Count, Log)")
    axes[1].set_xlabel("Number of Orders")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # 3. Monetary
    pos_monetary = rfm.loc[rfm["monetary"] > 0, "monetary"]
    min_m = max(pos_monetary.min(), 1.0) if not pos_monetary.empty else 1.0
    max_m = max(pos_monetary.max(), 10.0) if not pos_monetary.empty else 10.0
    axes[2].hist(
        pos_monetary,
        bins=np.logspace(np.log10(min_m), np.log10(max_m), 25),
        color="#d62728",
        edgecolor="#851414",
        alpha=0.8,
    )
    axes[2].set_xscale("log")
    axes[2].set_title("Monetary Value (Total Spend, Log)")
    axes[2].set_xlabel("Total Spend")
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    return fig, rfm


def sparsity(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> float:
    """
    Compute user-item interaction matrix sparsity: 1 - (|E| / (|U| * |I|)).

    Parameters:
        df: Input DataFrame.
        user_col: User identifier column.
        item_col: Item identifier column.

    Returns:
        sparsity_value: Float in range [0.0, 1.0].
    """
    n_users = df[user_col].nunique()
    n_items = df[item_col].nunique()
    n_interactions = df.drop_duplicates(subset=[user_col, item_col]).shape[0]

    total_possible = float(n_users) * float(n_items)
    if total_possible == 0:
        return 0.0
    return float(1.0 - (n_interactions / total_possible))
