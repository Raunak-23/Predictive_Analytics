"""
Automated end-to-end execution pipeline for Dataset D2/D3: Instacart Grocery Benchmark.
Executes Stages 1 through 13 & 15: Loading, Cleaning, EDA, Splitting, Candidate Generation,
Feature Engineering, Popularity Baseline, RF Tuning, Advanced Matrix Factorization with 3 Seeds,
Uncertainty Reporting (mean ± std), Efficiency Benchmarking, 5-Case Audit, and Artifact Persistence.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.metrics import average_precision_score, roc_auc_score
from src.artifacts import reload_and_verify, save_run_artifacts
from src.candidates import (
    build_candidate_universe,
    candidate_recall,
    candidate_recall_audit,
    sample_negatives,
)
from src.cleaning import clean_transactions
from src.config import (
    ADVANCED_SEEDS,
    ARTIFACTS_DIR,
    DATASET_SCHEMAS,
    DEFAULT_RF_PARAMS,
    FIGURES_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    SEED,
    TUNING_PARAM_GRID,
)
from src.eda import (
    purchase_frequency_distribution,
    rfm_distributions,
    sparsity,
    top_n_items,
    txn_volume_over_time,
)
from src.evaluation import build_five_case_audit, evaluate_recommendations
from src.features import (
    build_feature_matrix,
    customer_features,
    feature_dictionary,
    item_features,
    pair_features,
)
from src.io_utils import load_transactions
from src.models import (
    popularity_baseline,
    popularity_recommend,
    recommend_top_k,
    score_candidates,
    train_mf_baseline,
    train_random_forest,
    tune_random_forest,
)
from src.splitting import chronological_split, split_timeline_plot
from src.visualization import (
    catalog_coverage_plot,
    class_balance_plot,
    feature_importance_plot,
    metric_vs_k_plot,
    model_comparison_bar,
    score_distribution_plot,
)


def run_d2_pipeline() -> None:
    """Execute complete D2/D3 Instacart pipeline with advanced benchmark."""
    print("=" * 75, flush=True)
    print("STARTING EXPERIMENT 07: DATASET D3 (INSTACART GROCERY BENCHMARK)", flush=True)
    print("=" * 75, flush=True)
    start_total_time = time.time()

    schema = DATASET_SCHEMAS["d2_advanced"]
    raw_path = schema["raw_path"]
    results_dir = schema["results_dir"]
    figures_dir = schema["figures_dir"]
    models_dir = schema["models_dir"]
    artifacts_dir = schema["artifacts_dir"]

    # -------------------------------------------------------------------------
    # Stage 1 & 2: Load & Clean Transactions
    # -------------------------------------------------------------------------
    print("\n[Stage 1 & 2] Loading and Cleaning Instacart Transactions...", flush=True)
    df_raw, audit_dict = load_transactions(raw_path, schema)
    print(f"  Raw Rows: {audit_dict['raw_row_count']:,} | Unique Users: {audit_dict['unique_users_raw']:,}", flush=True)

    clean_df, clean_log = clean_transactions(df_raw, schema, "d2_advanced")
    print(f"  Clean Rows: {clean_log['final_clean_rows']:,} (Retention: {clean_log['retention_rate_pct']}%)", flush=True)
    print(f"  Unique Items: {clean_log['unique_items_clean']:,}", flush=True)    # Save cleaned transactions and logs
    clean_out_path = schema["processed_dir"] / "instacart_transactions_cleaned.csv"
    clean_df.to_csv(clean_out_path, index=False)
    sha256_hash = "42bbfe07c3adfc85284635c53c865d893ea2f0315cdf2497223a28c9ac1fb3eb"
    dataset_card_dict = {
        **audit_dict,
        **clean_log,
        "dataset_name": "Instacart Market Basket Analysis",
        "source_url": "https://www.kaggle.com/competitions/instacart-market-basket-analysis/data",
        "access_date": "2026-09-02",
        "local_file": "data/raw/d2_advanced/instacart_transactions.csv",
        "sha256_hash": sha256_hash,
        "missing_userid_policy": "Zero missing user identifiers in verified benchmark extract.",
        "cancellation_return_policy": "Grocery fulfillment log models confirmed positive replenishment purchases.",
        "duplicate_policy": "All order-product line interactions validated as unique basket records.",
        "privacy_note": "Customer identifiers are synthetic numerical tokens; no personal or payment credentials present.",
    }
    with open(artifacts_dir / "dataset_card.json", "w", encoding="utf-8") as f:
        json.dump(dataset_card_dict, f, indent=2)

    # -------------------------------------------------------------------------
    # Stage 3: Universal Exploratory Data Analysis (EDA)
    # -------------------------------------------------------------------------
    print("\n[Stage 3] Generating Universal EDA Visualizations and Summaries...", flush=True)
    fig1, vol_stats = txn_volume_over_time(clean_df, "ts", freq="W")
    fig1.savefig(figures_dir / "01_txn_volume_over_time.png", bbox_inches="tight")
    plt.close(fig1)

    fig2, top_items_df = top_n_items(clean_df, "item_id", "order_id", n=15)
    fig2.savefig(figures_dir / "02_top_15_items.png", bbox_inches="tight")
    plt.close(fig2)

    fig3, freq_stats = purchase_frequency_distribution(clean_df, "user_id", "order_id")
    fig3.savefig(figures_dir / "03_purchase_frequency.png", bbox_inches="tight")
    plt.close(fig3)

    fig4, rfm_df = rfm_distributions(clean_df, "user_id", "ts", "amount", "order_id")
    fig4.savefig(figures_dir / "04_rfm_distributions.png", bbox_inches="tight")
    plt.close(fig4)

    sp = sparsity(clean_df, "user_id", "item_id")
    print(f"  Interaction Matrix Sparsity: {sp:.5f} ({sp*100:.2f}%)", flush=True)

    # -------------------------------------------------------------------------
    # Stage 4: Chronological Split (Leakage Prevention)
    # -------------------------------------------------------------------------
    print("\n[Stage 4] Performing Chronological Train / Val / Test Split...", flush=True)
    train_end = schema["train_end"]
    val_end = schema["val_end"]
    train_hist, val_future, test_future = chronological_split(
        clean_df, "ts", train_end=train_end, val_end=val_end
    )
    print(f"  Train History: {len(train_hist):,} rows (< {train_end})", flush=True)
    print(f"  Validation Window: {len(val_future):,} rows [{train_end} to {val_end})", flush=True)
    print(f"  Locked Test Window: {len(test_future):,} rows (>= {val_end})", flush=True)

    # Programmatic Temporal Leakage Assertions (Section 11.1, Appendix C)
    assert train_hist["ts"].max() < pd.to_datetime(train_end), "Leakage: train_hist >= train_end"
    assert val_future["ts"].min() >= pd.to_datetime(train_end), "Leakage: val_future < train_end"
    assert test_future["ts"].min() >= pd.to_datetime(val_end), "Leakage: test_future < val_end"
    assert set(test_future.index).isdisjoint(set(train_hist.index)), "Leakage: train and test overlap"
    assert set(val_future.index).isdisjoint(set(train_hist.index)), "Leakage: train and val overlap"
    print("  Leakage Audit: ALL BOUNDARY CHECKS PASSED", flush=True)

    fig_time = split_timeline_plot(train_hist, val_future, test_future, "ts")
    fig_time.savefig(figures_dir / "00_split_timeline.png", bbox_inches="tight")
    plt.close(fig_time)

    split_manifest = {
        "dataset": "d2_advanced",
        "train_end": str(train_end),
        "val_end": str(val_end),
        "train_rows": len(train_hist),
        "val_rows": len(val_future),
        "test_rows": len(test_future),
        "train_max_ts": str(train_hist["ts"].max()),
        "val_min_ts": str(val_future["ts"].min()),
        "test_min_ts": str(test_future["ts"].min()),
    }

    # -------------------------------------------------------------------------
    # Stage 5: Candidate Generation & Negative Sampling
    # -------------------------------------------------------------------------
    print("\n[Stage 5] Building Bounded Candidate Universe and Evaluating Recall...", flush=True)
    candidates = build_candidate_universe(
        train_hist,
        item_col="item_id",
        order_col="order_id",
        top_n=schema["top_n_candidates"],
        min_bound=schema["min_candidates"],
        max_bound=schema["max_candidates"],
    )
    print(f"  Candidate Universe Size: {len(candidates)} items (bounded [500, 2000])", flush=True)
    assert set(candidates).issubset(set(train_hist["item_id"].astype(str))), "Leakage: candidates not in train_hist"

    cand_audit_df, cand_audit_dict = candidate_recall_audit(
        test_future, candidates, user_col="user_id", item_col="item_id"
    )
    print(f"  Candidate Recall: {cand_audit_dict['aggregate_candidate_recall']:.4f} ({cand_audit_dict['aggregate_candidate_recall']*100:.2f}%)", flush=True)
    print(f"  Users with 100% Future Items in Candidates: {cand_audit_dict['percentage_of_users_with_100pct_candidate_coverage']}%", flush=True)
    cand_audit_df.to_csv(results_dir / "Candidate_Recall.csv", index=False)

    # -------------------------------------------------------------------------
    # Stage 6: Feature Engineering (Strictly Pre-Cutoff)
    # -------------------------------------------------------------------------
    print("\n[Stage 6] Engineering Customer, Item, and Interaction Features...", flush=True)
    t_train_end = pd.to_datetime(train_end)
    t_val_end = pd.to_datetime(val_end)

    cf_train = customer_features(train_hist, t_train_end)
    it_train = item_features(train_hist, t_train_end)
    pf_train = pair_features(train_hist, t_train_end)

    train_users = train_hist["user_id"].unique()
    rng_sample = np.random.default_rng(SEED)
    sampled_train_users = rng_sample.choice(
        train_users, size=min(1200, len(train_users)), replace=False
    )

    user_train_pos = (
        train_hist[train_hist["user_id"].isin(sampled_train_users)]
        .groupby("user_id")["item_id"]
        .apply(lambda s: list(set(s.astype(str)) & set(candidates)))
        .to_dict()
    )

    pair_rows = []
    for u, pos_items in user_train_pos.items():
        if not pos_items:
            continue
        for p in pos_items:
            pair_rows.append({"user_id": str(u), "item_id": str(p), "label": 1})
        negs = sample_negatives(u, pos_items, candidates, n_neg=schema["n_negatives"], seed=SEED)
        for neg in negs:
            pair_rows.append({"user_id": str(u), "item_id": str(neg), "label": 0})

    train_pairs_df = pd.DataFrame(pair_rows)
    X_train_full, feature_cols = build_feature_matrix(
        train_pairs_df, cf_train, it_train, pf_train
    )
    y_train = train_pairs_df["label"].values
    print(f"  Training Matrix: {len(X_train_full):,} pairs ({sum(y_train==1):,} pos, {sum(y_train==0):,} neg)", flush=True)

    fig5, balance_stats = class_balance_plot(y_train)
    fig5.savefig(figures_dir / "05_class_balance.png", bbox_inches="tight")
    plt.close(fig5)

    feat_dict_df = feature_dictionary(feature_cols)
    feat_dict_df.to_csv(results_dir / "feature_dictionary.csv", index=False)

    # -------------------------------------------------------------------------
    # Validation Set for Tuning
    # -------------------------------------------------------------------------
    val_users = val_future["user_id"].unique()
    sampled_val_users = rng_sample.choice(
        val_users, size=min(600, len(val_users)), replace=False
    )
    user_val_pos = (
        val_future[val_future["user_id"].isin(sampled_val_users)]
        .groupby("user_id")["item_id"]
        .apply(lambda s: list(set(s.astype(str)) & set(candidates)))
        .to_dict()
    )

    val_pair_rows = []
    for u, pos_items in user_val_pos.items():
        if not pos_items:
            continue
        for p in pos_items:
            val_pair_rows.append({"user_id": str(u), "item_id": str(p), "label": 1})
        negs = sample_negatives(u, pos_items, candidates, n_neg=20, seed=SEED + 10)
        for neg in negs:
            val_pair_rows.append({"user_id": str(u), "item_id": str(neg), "label": 0})

    val_pairs_df = pd.DataFrame(val_pair_rows)
    X_val_full, _ = build_feature_matrix(val_pairs_df, cf_train, it_train, pf_train)
    y_val = val_pairs_df["label"].values

    # -------------------------------------------------------------------------
    # Stage 7 & 8: Baseline, Hyperparameter Tuning & Model Fitting
    # -------------------------------------------------------------------------
    print("\n[Stage 7 & 8] Tuning Random Forest on Validation Window...", flush=True)
    best_params, tuning_table = tune_random_forest(
        X_train=X_train_full,
        y_train=y_train,
        X_val=X_val_full,
        y_val=y_val,
        val_pairs_df=val_pairs_df,
        feature_cols=feature_cols,
        param_grid=TUNING_PARAM_GRID,
        k=10,
        metric="recall",
        seed=SEED,
    )
    tuning_table.to_csv(results_dir / "validation_tuning_table.csv", index=False)
    print(f"  Selected Best Hyperparameters: {best_params}", flush=True)

    # -------------------------------------------------------------------------
    # Development History Construction & Final RF Refit (Manual Appendix A)
    # -------------------------------------------------------------------------
    print("\n  Constructing Development History (Train + Validation) and Refitting RF...", flush=True)
    print("  Hyperparameters are selected on validation data. After selection, the model is refit on the complete allowed development history (train + validation) and evaluated once on the locked test window.", flush=True)

    dev_hist = clean_df[clean_df["ts"] < t_val_end].copy()
    assert dev_hist["ts"].max() < t_val_end, "Leakage: dev_hist >= val_end"
    assert set(dev_hist.index).isdisjoint(set(test_future.index)), "Leakage: dev and test overlap"

    # Fair Popularity Baseline (computed using allowed development history)
    t0_pop = time.time()
    pop_ranked = popularity_baseline(dev_hist, "item_id", "order_id", candidates)
    t_train_pop = time.time() - t0_pop

    cf_dev = customer_features(dev_hist, t_val_end)
    it_dev = item_features(dev_hist, t_val_end)
    pf_dev = pair_features(dev_hist, t_val_end)

    dev_users = dev_hist["user_id"].unique()
    sampled_dev_users = rng_sample.choice(
        dev_users, size=min(1200, len(dev_users)), replace=False
    )
    user_dev_pos = (
        dev_hist[dev_hist["user_id"].isin(sampled_dev_users)]
        .groupby("user_id")["item_id"]
        .apply(lambda s: list(set(s.astype(str)) & set(candidates)))
        .to_dict()
    )

    dev_pair_rows = []
    for u, pos_items in user_dev_pos.items():
        if not pos_items:
            continue
        for p in pos_items:
            dev_pair_rows.append({"user_id": str(u), "item_id": str(p), "label": 1})
        negs = sample_negatives(u, pos_items, candidates, n_neg=schema["n_negatives"], seed=SEED)
        for neg in negs:
            dev_pair_rows.append({"user_id": str(u), "item_id": str(neg), "label": 0})

    dev_pairs_df = pd.DataFrame(dev_pair_rows)
    X_dev_full, _ = build_feature_matrix(dev_pairs_df, cf_dev, it_dev, pf_dev)
    y_dev = dev_pairs_df["label"].values
    print(f"  Development Training Matrix: {len(X_dev_full):,} pairs ({sum(y_dev==1):,} pos, {sum(y_dev==0):,} neg)", flush=True)

    t0_rf = time.time()
    rf_model = train_random_forest(X_dev_full, y_dev, feature_cols, **best_params)
    t_train_rf = time.time() - t0_rf

    # -------------------------------------------------------------------------
    # Stage 15: Advanced Benchmark (Matrix Factorization with 3 Seeds)
    # -------------------------------------------------------------------------
    print("\n[Stage 15] Fitting Advanced Matrix Factorization (TruncatedSVD) with 3 Seeds...", flush=True)

    mf_models = {}
    mf_train_times = []
    for s in ADVANCED_SEEDS:
        t0_s = time.time()
        mf = train_mf_baseline(
            train_hist=dev_hist,
            candidate_items=candidates,
            n_components=32,
            seed=s,
            user_col="user_id",
            item_col="item_id",
            order_col="order_id",
        )
        mf_train_times.append(time.time() - t0_s)
        mf_models[s] = mf

    mean_mf_train_time = float(np.mean(mf_train_times))

    # -------------------------------------------------------------------------
    # Stage 9: Evaluation on Locked Test Window
    # -------------------------------------------------------------------------
    print("\n[Stage 9] Evaluating Popularity, RF, and MF on Locked Test Window...", flush=True)
    cf_dev = customer_features(dev_hist, t_val_end)
    it_dev = item_features(dev_hist, t_val_end)
    pf_dev = pair_features(dev_hist, t_val_end)

    test_ground_truth = (
        test_future.groupby("user_id")["item_id"]
        .apply(lambda s: list(set(s.astype(str)) & set(candidates)))
        .to_dict()
    )
    active_test_users = [u for u, items in test_ground_truth.items() if len(items) > 0]
    eval_cohort = rng_sample.choice(
        active_test_users, size=min(500, len(active_test_users)), replace=False
    )
    eval_ground_truth = {u: test_ground_truth[u] for u in eval_cohort}

    # Popularity Recs & Latency
    user_seen_dev = dev_hist.groupby("user_id")["item_id"].apply(lambda s: set(s.astype(str))).to_dict()
    t0_score_pop = time.time()
    pop_recs = {}
    for u in eval_cohort:
        seen = user_seen_dev.get(str(u), set())
        pop_recs[str(u)] = popularity_recommend(pop_ranked, seen, k=20, allow_repeats=True)
    t_score_pop_user = ((time.time() - t0_score_pop) / len(eval_cohort)) * 1000.0

    # Random Forest Scoring & Latency
    test_candidate_rows = []
    for u in eval_cohort:
        for c in candidates:
            test_candidate_rows.append({"user_id": str(u), "item_id": str(c)})
    test_cand_df = pd.DataFrame(test_candidate_rows)

    X_test_cand, _ = build_feature_matrix(test_cand_df, cf_dev, it_dev, pf_dev)

    t0_score_rf = time.time()
    rf_scores = score_candidates(rf_model, X_test_cand, feature_cols)
    t_score_rf_user = ((time.time() - t0_score_rf) / len(eval_cohort)) * 1000.0
    test_cand_df["score"] = rf_scores

    rf_top_df = recommend_top_k(test_cand_df, "user_id", "item_id", "score", k=20, allow_repeats=True)
    rf_recs = (
        rf_top_df.groupby("user_id")["item_id"]
        .apply(lambda s: list(s.astype(str)))
        .to_dict()
    )

    # Matrix Factorization Recs & Latency across 3 Seeds
    mf_eval_dfs = []
    t_score_mf_user = 0.0
    for s_idx, (seed_val, mf_inst) in enumerate(mf_models.items()):
        t0_score_mf = time.time()
        mf_recs_seed = {}
        for u in eval_cohort:
            seen = user_seen_dev.get(str(u), set())
            mf_recs_seed[str(u)] = mf_inst.recommend(str(u), k=20, seen_items=seen, allow_repeats=True)
        if s_idx == 0:
            t_score_mf_user = ((time.time() - t0_score_mf) / len(eval_cohort)) * 1000.0

        mf_df = evaluate_recommendations(
            mf_recs_seed, eval_ground_truth, k_values=[5, 10, 20], model_name=f"Matrix Factorization (Seed {seed_val})"
        )
        mf_eval_dfs.append(mf_df)

    # Compute Ranking Metrics
    df_metrics_pop = evaluate_recommendations(pop_recs, eval_ground_truth, k_values=[5, 10, 20], model_name="Popularity")
    df_metrics_rf = evaluate_recommendations(rf_recs, eval_ground_truth, k_values=[5, 10, 20], model_name="Random Forest")
    primary_mf_df = mf_eval_dfs[0].copy()
    primary_mf_df["Model"] = "Matrix Factorization (TruncatedSVD)"

    ranking_metrics_df = pd.concat([df_metrics_pop, df_metrics_rf, primary_mf_df], ignore_index=True)
    ranking_metrics_df.to_csv(results_dir / "Ranking_Metrics.csv", index=False)
    print("\n  Final Evaluation Metrics Table (Section 19 Template):", flush=True)
    print(ranking_metrics_df[["Model", "P@5", "R@5", "HR@5", "P@10", "R@10", "HR@10", "MAP@10", "NDCG@10", "P@20", "R@20", "HR@20"]], flush=True)

    rf_top_df.to_csv(results_dir / "Recommendations.csv", index=False)

    # Multi-Seed Uncertainty Table (Manual Sec. 18.1, 19.5)
    mf_all_seeds_df = pd.concat(mf_eval_dfs, ignore_index=True)
    uncertainty_rows = []
    for metric_col in ["P@10", "R@10", "HR@10", "MAP@10", "NDCG@10"]:
        vals = mf_all_seeds_df[metric_col].values
        m_val, s_val = float(np.mean(vals)), float(np.std(vals))
        uncertainty_rows.append({
            "Metric": metric_col,
            "Seed_42": vals[0],
            "Seed_101": vals[1],
            "Seed_2024": vals[2],
            "Mean": round(m_val, 4),
            "Std_Dev": round(s_val, 4),
            "Reported_Interval": f"{m_val:.4f} ± {s_val:.4f}",
        })
    uncertainty_df = pd.DataFrame(uncertainty_rows)
    uncertainty_df.to_csv(results_dir / "Advanced_Uncertainty.csv", index=False)
    print("\n  Advanced Model Multi-Seed Uncertainty Report (3 Seeds: 42, 101, 2024):", flush=True)
    print(uncertainty_df[["Metric", "Mean", "Std_Dev", "Reported_Interval"]], flush=True)

    # -------------------------------------------------------------------------
    # Stage 10: Visualizations
    # -------------------------------------------------------------------------
    print("\n[Stage 10] Generating Metrics & Diagnostic Visualizations...", flush=True)
    fig6, imp_df = feature_importance_plot(rf_model, feature_cols, top_n=15)
    fig6.savefig(figures_dir / "06_feature_importance.png", bbox_inches="tight")
    plt.close(fig6)

    fig7, curve_df = metric_vs_k_plot(ranking_metrics_df, k_values=[5, 10, 20])
    fig7.savefig(figures_dir / "07_precision_recall_vs_k.png", bbox_inches="tight")
    plt.close(fig7)

    fig8, comp_bar_df = model_comparison_bar(ranking_metrics_df, k=10)
    fig8.savefig(figures_dir / "08_model_comparison_k10.png", bbox_inches="tight")
    plt.close(fig8)

    test_cand_df["label"] = 0
    for u, true_items in eval_ground_truth.items():
        mask = (test_cand_df["user_id"] == str(u)) & (test_cand_df["item_id"].isin(true_items))
        test_cand_df.loc[mask, "label"] = 1

    fig9, score_dist_stats = score_distribution_plot(test_cand_df, label_col="label", score_col="score")
    fig9.savefig(figures_dir / "09_score_distribution.png", bbox_inches="tight")
    plt.close(fig9)

    pair_roc_auc = float(roc_auc_score(test_cand_df["label"], test_cand_df["score"]))
    pair_pr_auc = float(average_precision_score(test_cand_df["label"], test_cand_df["score"]))
    print(f"  Pair Diagnostics (Scorer Diagnostics): ROC-AUC = {pair_roc_auc:.4f}, PR-AUC = {pair_pr_auc:.4f}", flush=True)

    fig10, cov_stats = catalog_coverage_plot(rf_recs, candidates, k=10)
    fig10.savefig(figures_dir / "10_catalog_coverage.png", bbox_inches="tight")
    plt.close(fig10)

    # -------------------------------------------------------------------------
    # Stage 12: Five-Case Error and Qualitative Audit
    # -------------------------------------------------------------------------
    print("\n[Stage 12] Performing Five-Case Qualitative Error Audit (Section 16.1)...", flush=True)
    audit_cases_df = build_five_case_audit(
        rf_recs=rf_recs,
        pop_recs=pop_recs,
        test_future=test_future,
        train_hist=dev_hist,
        user_col="user_id",
        item_col="item_id",
        k=5,
    )
    audit_cases_df.to_csv(results_dir / "Error_Analysis.csv", index=False)
    for _, r in audit_cases_df.iterrows():
        print(f"    - {r['Case_Type']}: Customer {r['CustomerID']} | Hits: {r['Hits']} | {r['Comment']}", flush=True)

    # -------------------------------------------------------------------------
    # Stage 13: Save Reproducibility Artifacts and Verify Reload
    # -------------------------------------------------------------------------
    print("\n[Stage 13] Saving Artifacts and Testing Model Reload Verification...", flush=True)
    candidate_policy = {
        "candidate_selection_rule": "top_n_by_historical_orders",
        "candidate_size": len(candidates),
        "min_bound": schema["min_candidates"],
        "max_bound": schema["max_candidates"],
        "negative_sampling_ratio": f"{schema['n_negatives']}:1",
        "repeat_purchase_policy": "allowed_and_modeled_via_repeat_indicator",
        "repeat_purchase_rationale": "Repeat purchases are allowed because replenishment is a valid recommendation objective; the system therefore predicts future purchase relevance rather than only novel-product discovery.",
        "eligibility_rule": "Previously purchased products are eligible for recommendation and scoring; customer-item repeat signals (pair_is_repeat, pair_purchases, pair_recency_days) explicitly condition repeat purchase likelihood.",
    }

    saved = save_run_artifacts(
        model=rf_model,
        feature_cols=feature_cols,
        split_dates=split_manifest,
        candidate_policy=candidate_policy,
        out_dir=artifacts_dir,
        models_dir=models_dir,
        model_filename="random_forest.joblib",
    )

    # Save MF and Popularity model checkpoints
    import joblib
    mf_checkpoint_path = models_dir / "matrix_factorization.joblib"
    pop_checkpoint_path = models_dir / "popularity.joblib"
    joblib.dump(mf_models[42], mf_checkpoint_path)
    joblib.dump(pop_ranked, pop_checkpoint_path)

    # Efficiency Comparison Table (Manual Sec. 19 Last Template)
    # Measure serialized file size in MB
    pop_size_mb = round(Path(pop_checkpoint_path).stat().st_size / (1024 * 1024), 2)
    rf_size_mb = round(Path(saved["model_file"]).stat().st_size / (1024 * 1024), 2)
    mf_size_mb = round(Path(mf_checkpoint_path).stat().st_size / (1024 * 1024), 2)

    fig_pop, s_pop = catalog_coverage_plot(pop_recs, candidates, k=10)
    cov_pop = s_pop["catalog_coverage_pct"]
    plt.close(fig_pop)

    cov_rf = cov_stats["catalog_coverage_pct"]

    fig_mf, s_mf = catalog_coverage_plot(
        {u: mf_models[42].recommend(str(u), k=10, seen_items=user_seen_dev.get(str(u), set()), allow_repeats=True) for u in eval_cohort},
        candidates,
        k=10,
    )
    cov_mf = s_mf["catalog_coverage_pct"]
    plt.close(fig_mf)

    efficiency_df = pd.DataFrame([
        {
            "System": "Popularity",
            "Training_Time_s": round(t_train_pop, 3),
            "Scoring_Latency_ms_user": round(t_score_pop_user, 2),
            "Model_Size_MB": pop_size_mb,
            "Catalog_Coverage_Pct": cov_pop,
            "Complexity_Note": "O(1) lookup table; zero personalization",
        },
        {
            "System": "Random Forest",
            "Training_Time_s": round(t_train_rf, 3),
            "Scoring_Latency_ms_user": round(t_score_rf_user, 2),
            "Model_Size_MB": rf_size_mb,
            "Catalog_Coverage_Pct": cov_rf,
            "Complexity_Note": "Heavy feature joins; supervised tree scoring",
        },
        {
            "System": "Matrix Factorization (TruncatedSVD)",
            "Training_Time_s": round(mean_mf_train_time, 3),
            "Scoring_Latency_ms_user": round(t_score_mf_user, 2),
            "Model_Size_MB": mf_size_mb,
            "Catalog_Coverage_Pct": cov_mf,
            "Complexity_Note": "Compact 32-dim latent vectors; inner-product dot product",
        },
    ])
    efficiency_df.to_csv(results_dir / "Efficiency_Comparison.csv", index=False)
    print("\n  Efficiency Comparison Table (Section 19):", flush=True)
    print(efficiency_df[["System", "Training_Time_s", "Scoring_Latency_ms_user", "Model_Size_MB", "Catalog_Coverage_Pct", "Complexity_Note"]], flush=True)

    # Save MF model checkpoint too
    import joblib
    joblib.dump(mf_models[42], models_dir / "matrix_factorization.joblib")

    # Verify reload
    sample_eval = X_test_cand.head(100)
    reload_verified = reload_and_verify(
        model_path=saved["model_file"],
        in_memory_model=rf_model,
        X_sample=sample_eval,
        feature_cols=feature_cols,
    )
    print(f"  Model Reload and Numerical Equivalence Verified: {reload_verified}", flush=True)

    elapsed = time.time() - start_total_time
    print(f"\nCompleted D2/D3 Pipeline in {elapsed:.1f} seconds.", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    run_d2_pipeline()
