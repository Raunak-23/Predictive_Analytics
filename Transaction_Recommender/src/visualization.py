"""
Diagnostic and ranking visualization module for Experiment 07 Recommendation System.
Generates publication-ready figures for metric curves, feature importances, score distributions, and class balance,
returning exact underlying numerical data alongside every Matplotlib figure.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_plotting_style() -> None:
    """Set publication-grade aesthetics."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.labelweight"] = "bold"


def class_balance_plot(y: Union[pd.Series, np.ndarray]) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Plot and summarize class balance after negative sampling (Section 17 item 5).
    """
    set_plotting_style()
    y_arr = np.asarray(y)
    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())
    total = n_pos + n_neg
    ratio = round(n_neg / max(n_pos, 1), 2)

    stats = {
        "positive_pairs": n_pos,
        "negative_pairs": n_neg,
        "total_pairs": total,
        "negative_to_positive_ratio": f"{ratio}:1",
        "positive_prevalence_pct": round((n_pos / max(total, 1)) * 100.0, 2),
    }

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    bars = ax.bar(
        ["Positive (y=1)", "Sampled Negative (y=0)"],
        [n_pos, n_neg],
        color=["#2ca02c", "#d62728"],
        edgecolor=["#1b6e1b", "#851414"],
        alpha=0.85,
        width=0.45,
    )
    ax.set_ylabel("Pair Count")
    ax.set_title("Class Balance After Bounded Negative Sampling")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + (total * 0.015),
            f"{int(h):,} ({h/total*100:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    ax.set_ylim(0, max(n_neg, n_pos) * 1.15)
    plt.tight_layout()
    return fig, stats


def feature_importance_plot(
    model: Any,
    feature_cols: List[str],
    top_n: int = 15,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Plot Gini feature importances for Random Forest with non-causal interpretation guidelines (Section 17 item 6).
    """
    set_plotting_style()
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    imp_df["relative_pct"] = round((imp_df["importance"] / imp_df["importance"].sum()) * 100.0, 2)
    top_imp = imp_df.head(top_n).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    bars = ax.barh(
        top_imp["feature"],
        top_imp["importance"],
        color="#1f77b4",
        edgecolor="#114b75",
        alpha=0.85,
    )
    ax.set_xlabel("Mean Decrease in Impurity (Gini Importance)")
    ax.set_title(f"Top {top_n} Random Forest Feature Importances (Non-Causal Diagnostic)")
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + (max(top_imp["importance"]) * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{w:.4f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    plt.tight_layout()
    return fig, imp_df


def metric_vs_k_plot(
    metrics_df: pd.DataFrame,
    k_values: Sequence[int] = (5, 10, 20),
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Plot Precision@K and Recall@K versus recommendation list size K (Section 17 item 7).
    """
    set_plotting_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    curve_rows = []
    models = metrics_df["Model"].unique()
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]

    for idx, model_name in enumerate(models):
        m_row = metrics_df[metrics_df["Model"] == model_name].iloc[0]
        color = colors[idx % len(colors)]

        p_vals = [m_row[f"P@{k}"] for k in k_values]
        r_vals = [m_row[f"R@{k}"] for k in k_values]

        for k, p, r in zip(k_values, p_vals, r_vals):
            curve_rows.append({"Model": model_name, "K": k, "Precision": p, "Recall": r})

        # Precision@K plot
        axes[0].plot(k_values, p_vals, marker="o", linewidth=2, label=model_name, color=color)
        # Recall@K plot
        axes[1].plot(k_values, r_vals, marker="s", linewidth=2, label=model_name, color=color)

    axes[0].set_title("Precision@K vs List Size K")
    axes[0].set_xlabel("Cutoff Rank (K)")
    axes[0].set_ylabel("Precision@K")
    axes[0].set_xticks(list(k_values))
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].set_title("Recall@K vs List Size K")
    axes[1].set_xlabel("Cutoff Rank (K)")
    axes[1].set_ylabel("Recall@K")
    axes[1].set_xticks(list(k_values))
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    return fig, pd.DataFrame(curve_rows)


def model_comparison_bar(
    metrics_df: pd.DataFrame,
    k: int = 10,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Grouped bar chart comparing Popularity vs Random Forest vs Advanced models (Section 17 item 8).
    """
    set_plotting_style()
    metric_cols = [f"P@{k}", f"R@{k}", f"HR@{k}", f"NDCG@{k}"]
    available_cols = [c for c in metric_cols if c in metrics_df.columns]

    models = metrics_df["Model"].tolist()
    n_models = len(models)
    n_metrics = len(available_cols)

    x = np.arange(n_metrics)
    width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e"]

    comp_data = []
    for i, model_name in enumerate(models):
        m_row = metrics_df[metrics_df["Model"] == model_name].iloc[0]
        vals = [float(m_row[col]) for col in available_cols]
        comp_data.append({"Model": model_name, **{col: val for col, val in zip(available_cols, vals)}})

        offset = (i - (n_models - 1) / 2) * width
        rects = ax.bar(
            x + offset,
            vals,
            width,
            label=model_name,
            color=colors[i % len(colors)],
            edgecolor="#222222",
            alpha=0.85,
        )

        for rect in rects:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h + 0.005,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(available_cols)
    ax.set_ylabel(f"Score (at K={k})")
    ax.set_title(f"Headline Comparison: Popularity Baseline vs Recommender Models (K={k})")
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    return fig, pd.DataFrame(comp_data)


def score_distribution_plot(
    scored_df: pd.DataFrame,
    label_col: str = "label",
    score_col: str = "score",
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Plot predicted recommendation-score distribution for positives vs negatives (Section 17 item 9).
    """
    set_plotting_style()
    pos_scores = scored_df.loc[scored_df[label_col] == 1, score_col].dropna().values
    neg_scores = scored_df.loc[scored_df[label_col] == 0, score_col].dropna().values

    pos_median = float(np.median(pos_scores)) if len(pos_scores) > 0 else 0.0
    neg_median = float(np.median(neg_scores)) if len(neg_scores) > 0 else 0.0
    gap = round(pos_median - neg_median, 4)

    stats = {
        "positive_score_mean": round(float(np.mean(pos_scores)), 4) if len(pos_scores) > 0 else 0.0,
        "positive_score_median": round(pos_median, 4),
        "negative_score_mean": round(float(np.mean(neg_scores)), 4) if len(neg_scores) > 0 else 0.0,
        "negative_score_median": round(neg_median, 4),
        "median_separability_gap": gap,
    }

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    bins = np.linspace(0.0, 1.0, 35)

    ax.hist(
        neg_scores,
        bins=bins,
        density=True,
        alpha=0.6,
        color="#d62728",
        edgecolor="#851414",
        label=f"Negatives (y=0, Median: {neg_median:.3f})",
    )
    ax.hist(
        pos_scores,
        bins=bins,
        density=True,
        alpha=0.6,
        color="#2ca02c",
        edgecolor="#1b6e1b",
        label=f"Positives (y=1, Median: {pos_median:.3f})",
    )

    ax.axvline(neg_median, color="#851414", linestyle="--", linewidth=1.8)
    ax.axvline(pos_median, color="#1b6e1b", linestyle="--", linewidth=1.8)

    ax.set_title("Propensity Score Distribution: Positives vs Sampled Negatives")
    ax.set_xlabel("Predicted Purchase Propensity P(y=1)")
    ax.set_ylabel("Probability Density")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    return fig, stats


def catalog_coverage_plot(
    recs_by_user: Dict[str, Sequence[str]],
    catalog_items: Sequence[str],
    k: int = 10,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Plot recommendation frequency concentration and catalog coverage curve (Section 17 item 10).
    """
    set_plotting_style()
    catalog_set = set(str(c) for c in catalog_items)
    all_recs = []
    for u, rec_list in recs_by_user.items():
        all_recs.extend([str(i) for i in rec_list[:k]])

    rec_series = pd.Series(all_recs)
    item_freq = rec_series.value_counts()

    recommended_unique = len(item_freq)
    total_catalog = len(catalog_set)
    catalog_coverage_pct = round((recommended_unique / max(total_catalog, 1)) * 100.0, 2)

    # Lorenz curve for recommendation concentration
    sorted_freq = np.sort(item_freq.values)
    cum_freq = np.cumsum(sorted_freq) / max(cum_freq_sum := np.sum(sorted_freq), 1.0)
    x_lorenz = np.linspace(0, 1, len(cum_freq))

    # Gini coefficient
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    gini = float(1.0 - 2.0 * _trapz(cum_freq, x_lorenz)) if len(cum_freq) > 1 else 0.0

    stats = {
        "total_catalog_size": total_catalog,
        "recommended_unique_items": recommended_unique,
        "catalog_coverage_pct": catalog_coverage_pct,
        "recommendation_gini_index": round(gini, 3),
        "top_10_items_share_pct": round(float(item_freq.head(10).sum() / max(cum_freq_sum, 1.0) * 100.0), 2),
    }

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(x_lorenz, cum_freq, color="#9467bd", linewidth=2.2, label=f"Recommendations (Gini: {gini:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7f7f7f", label="Perfect Equality")

    ax.set_title(f"Catalog Coverage & Recommendation Inequality (Coverage: {catalog_coverage_pct}%)")
    ax.set_xlabel("Cumulative Share of Recommended Products")
    ax.set_ylabel("Cumulative Share of Total Recommendations Delivered")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    return fig, stats
