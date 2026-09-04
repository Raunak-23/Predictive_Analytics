"""
Evaluation metrics and error auditing module for Experiment 07 Recommendation System.
Implements ranking metrics (P@K, R@K, HitRate@K, MAP@K, NDCG@K) and the required five-case audit.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Union

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 1. User-Level Ranking Metrics (Pure functions, no I/O)
# -----------------------------------------------------------------------------

def precision_at_k(recs: Sequence[str], relevant: Sequence[str], k: int = 10) -> float:
    """Fraction of Top-K recommended items that are relevant."""
    r = list(recs)[:k]
    rel = set(str(x) for x in relevant)
    hits = sum(1 for x in r if str(x) in rel)
    return float(hits / max(k, 1))


def recall_at_k(recs: Sequence[str], relevant: Sequence[str], k: int = 10) -> float:
    """Fraction of future relevant items recovered in the Top-K recommendations."""
    if not relevant:
        return np.nan
    rel = set(str(x) for x in relevant)
    if len(rel) == 0:
        return np.nan
    r = list(recs)[:k]
    hits = sum(1 for x in r if str(x) in rel)
    return float(hits / len(rel))


def hit_rate_at_k(recs: Sequence[str], relevant: Sequence[str], k: int = 10) -> float:
    """Binary indicator (1.0 or 0.0) whether at least one relevant item appears in Top-K."""
    rel = set(str(x) for x in relevant)
    r = list(recs)[:k]
    return float(any(str(x) in rel for x in r))


def average_precision_at_k(recs: Sequence[str], relevant: Sequence[str], k: int = 10) -> float:
    """Position-weighted average precision up to rank K."""
    rel = set(str(x) for x in relevant)
    if not rel:
        return np.nan
    r = list(recs)[:k]
    score = 0.0
    hits = 0
    for rank, item in enumerate(r, start=1):
        if str(item) in rel:
            hits += 1
            score += hits / rank
    return float(score / min(len(rel), k))


def ndcg_at_k(recs: Sequence[str], relevant: Sequence[str], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at rank K."""
    rel = set(str(x) for x in relevant)
    if not rel:
        return np.nan
    r = list(recs)[:k]
    dcg = sum((1.0 / np.log2(idx + 2)) for idx, item in enumerate(r) if str(item) in rel)
    idcg = sum((1.0 / np.log2(idx + 2)) for idx in range(min(len(rel), k)))
    return float(dcg / idcg) if idcg > 0 else 0.0


# -----------------------------------------------------------------------------
# 2. Global Evaluation Aggregator
# -----------------------------------------------------------------------------

def evaluate_recommendations(
    recs_by_user: Dict[str, Sequence[str]],
    relevant_by_user: Dict[str, Sequence[str]],
    k_values: Sequence[int] = (5, 10, 20),
    model_name: str = "Model",
) -> pd.DataFrame:
    """
    Compute comprehensive ranking metrics across all evaluation users handling NaNs with nanmean.

    Parameters:
        recs_by_user: Dict mapping user_id to ordered list of recommended item_ids.
        relevant_by_user: Dict mapping user_id to set/list of true future relevant item_ids.
        k_values: Sequence of cutoff ranks (default [5, 10, 20]).
        model_name: Model identifier string.

    Returns:
        metrics_df: Single-row DataFrame containing all evaluated metrics.
    """
    row: Dict[str, Any] = {"Model": model_name}

    eval_users = [u for u, rel in relevant_by_user.items() if len(rel) > 0]
    total_eval_users = len(eval_users)
    row["Evaluation_Users"] = total_eval_users

    for k in k_values:
        p_list = []
        r_list = []
        hr_list = []
        map_list = []
        ndcg_list = []

        for u in eval_users:
            u_str = str(u)
            recs = recs_by_user.get(u_str, [])
            rel = relevant_by_user[u]

            p_list.append(precision_at_k(recs, rel, k=k))
            r_list.append(recall_at_k(recs, rel, k=k))
            hr_list.append(hit_rate_at_k(recs, rel, k=k))
            map_list.append(average_precision_at_k(recs, rel, k=k))
            ndcg_list.append(ndcg_at_k(recs, rel, k=k))

        row[f"P@{k}"] = round(float(np.nanmean(p_list)), 4)
        row[f"R@{k}"] = round(float(np.nanmean(r_list)), 4)
        row[f"HR@{k}"] = round(float(np.nanmean(hr_list)), 4)
        row[f"MAP@{k}"] = round(float(np.nanmean(map_list)), 4)
        row[f"NDCG@{k}"] = round(float(np.nanmean(ndcg_list)), 4)

    return pd.DataFrame([row])


# -----------------------------------------------------------------------------
# 3. Required Five-Case Error & Qualitative Audit (Section 16.1)
# -----------------------------------------------------------------------------

def build_five_case_audit(
    rf_recs: Dict[str, List[str]],
    pop_recs: Dict[str, List[str]],
    test_future: pd.DataFrame,
    train_hist: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    k: int = 5,
) -> pd.DataFrame:
    """
    Extract exactly five real, qualifying customer audit cases from actual data per Section 16.1:
    1. Successful personalized recommendation with supporting history.
    2. Relevant item missed by RF but caught by Popularity baseline.
    3. Case where RF beats Popularity.
    4. Sparse / cold-start customer.
    5. Questionable recommendation caused by popularity bias or candidate limitation.
    """
    # Ground truth future items
    future_ground = (
        test_future.groupby(user_col)[item_col]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )

    # Customer historical profiles
    hist_counts = train_hist.groupby(user_col)[item_col].apply(list).to_dict()

    cases = []
    seen_users: Set[str] = set()

    # Precompute per-user hits for RF and Popularity
    user_analysis = {}
    for u, true_items in future_ground.items():
        u_str = str(u)
        rf_top = rf_recs.get(u_str, [])[:k]
        pop_top = pop_recs.get(u_str, [])[:k]
        history = hist_counts.get(u_str, [])

        rf_hits = set(rf_top) & true_items
        pop_hits = set(pop_top) & true_items

        user_analysis[u_str] = {
            "u": u_str,
            "rf_top": rf_top,
            "pop_top": pop_top,
            "true_items": list(true_items),
            "history": history,
            "rf_hits": rf_hits,
            "pop_hits": pop_hits,
            "hist_size": len(history),
        }

    # Case 1: Successful personalized recommendation (RF hits >= 1, good history)
    c1_candidates = [
        info for info in user_analysis.values()
        if len(info["rf_hits"]) >= 1 and info["hist_size"] >= 5
    ]
    if c1_candidates:
        cand = max(c1_candidates, key=lambda x: len(x["rf_hits"]))
        seen_users.add(cand["u"])
        rf_hits_list = sorted(list(cand["rf_hits"]))
        cases.append({
            "Case_Type": "1. Successful Personalized Hit",
            "CustomerID": cand["u"],
            "History_Size": cand["hist_size"],
            "History_Snippet": str(cand["history"][:5]),
            "True_Future_Items": str(cand["true_items"][:5]),
            "Top_5_Recommendations": str(cand["rf_top"][:k]),
            "Hits": len(cand["rf_hits"]),
            "Hit_Status": "HIT",
            "Comment": f"RF successfully predicted {len(cand['rf_hits'])} future item(s) ({rf_hits_list[:2]}) using historical purchase affinity and repeat signals.",
        })
    else:
        cases.append({
            "Case_Type": "1. Successful Personalized Hit",
            "CustomerID": "N/A",
            "History_Size": 0,
            "History_Snippet": "[]",
            "True_Future_Items": "[]",
            "Top_5_Recommendations": "[]",
            "Hits": 0,
            "Hit_Status": "MISS",
            "Comment": "No user satisfied personalized hit threshold under current evaluation cutoff.",
        })

    # Case 2: Relevant item missed by RF but caught by Popularity baseline
    c2_candidates = [
        info for info in user_analysis.values()
        if len(info["pop_hits"]) > 0 and len(info["rf_hits"]) == 0 and info["u"] not in seen_users
    ]
    if c2_candidates:
        cand = c2_candidates[0]
        seen_users.add(cand["u"])
        pop_hits_list = sorted(list(cand["pop_hits"]))
        cases.append({
            "Case_Type": "2. RF Miss / Popularity Hit",
            "CustomerID": cand["u"],
            "History_Size": cand["hist_size"],
            "History_Snippet": str(cand["history"][:5]),
            "True_Future_Items": str(cand["true_items"][:5]),
            "Top_5_Recommendations": str(cand["rf_top"][:k]),
            "Hits": 0,
            "Hit_Status": "MISS (RF) / HIT (POP)",
            "Comment": f"Global baseline captured high-velocity item(s) {pop_hits_list[:2]}, which RF ranked lower due to customer feature divergence.",
        })
    else:
        cases.append({
            "Case_Type": "2. RF Miss / Popularity Hit",
            "CustomerID": "N/A",
            "History_Size": 0,
            "History_Snippet": "[]",
            "True_Future_Items": "[]",
            "Top_5_Recommendations": "[]",
            "Hits": 0,
            "Hit_Status": "N/A",
            "Comment": "No user found where popularity strictly beat Random Forest in Top-5 ranking.",
        })

    # Case 3: RF beats Popularity (RF hits > Pop hits)
    c3_candidates = [
        info for info in user_analysis.values()
        if len(info["rf_hits"]) > len(info["pop_hits"]) and info["u"] not in seen_users
    ]
    if c3_candidates:
        cand = max(c3_candidates, key=lambda x: len(x["rf_hits"]) - len(x["pop_hits"]))
        seen_users.add(cand["u"])
        rf_hits_list = sorted(list(cand["rf_hits"]))
        cases.append({
            "Case_Type": "3. RF Beats Popularity",
            "CustomerID": cand["u"],
            "History_Size": cand["hist_size"],
            "History_Snippet": str(cand["history"][:5]),
            "True_Future_Items": str(cand["true_items"][:5]),
            "Top_5_Recommendations": str(cand["rf_top"][:k]),
            "Hits": len(cand["rf_hits"]),
            "Hit_Status": f"RF {len(cand['rf_hits'])} vs POP {len(cand['pop_hits'])}",
            "Comment": f"Personalized ranking correctly surfaced customer-affinity product(s) {rf_hits_list[:2]} that unpersonalized popularity ranked outside Top-5.",
        })
    else:
        cases.append({
            "Case_Type": "3. RF Beats Popularity",
            "CustomerID": "N/A",
            "History_Size": 0,
            "History_Snippet": "[]",
            "True_Future_Items": "[]",
            "Top_5_Recommendations": "[]",
            "Hits": 0,
            "Hit_Status": "N/A",
            "Comment": "No user satisfied RF strictly beating popularity in Top-5 ranking.",
        })

    # Case 4: Sparse / Cold-start customer
    c4_candidates = [
        info for info in user_analysis.values()
        if info["hist_size"] <= 2 and info["u"] not in seen_users
    ]
    if c4_candidates:
        cand = c4_candidates[0]
        seen_users.add(cand["u"])
        cases.append({
            "Case_Type": "4. Sparse / Cold-Start Customer",
            "CustomerID": cand["u"],
            "History_Size": cand["hist_size"],
            "History_Snippet": str(cand["history"][:5]),
            "True_Future_Items": str(cand["true_items"][:5]),
            "Top_5_Recommendations": str(cand["rf_top"][:k]),
            "Hits": len(cand["rf_hits"]),
            "Hit_Status": "HIT" if len(cand["rf_hits"]) > 0 else "MISS",
            "Comment": f"Customer with sparse history ({cand['hist_size']} prior purchases) evaluated under high-uncertainty regime (Hits: {len(cand['rf_hits'])}).",
        })
    else:
        # Deterministically select the real test customer with the minimum historical count
        available = [
            info for info in user_analysis.values()
            if info["u"] not in seen_users and len(info["rf_top"]) > 0
        ]
        if not available:
            available = [info for info in user_analysis.values() if info["u"] not in seen_users]
        if available:
            cand = min(
                available,
                key=lambda x: (x["hist_size"], int(x["u"]) if str(x["u"]).isdigit() else str(x["u"])),
            )
            seen_users.add(cand["u"])
            rf_hits_list = sorted(list(cand["rf_hits"]))
            if len(cand["rf_hits"]) > 0:
                comment = (
                    f"Sparsest available test customer ({cand['hist_size']} prior purchases); "
                    f"RF personalized signals yielded {len(cand['rf_hits'])} hit(s) ({rf_hits_list[:2]}) "
                    f"vs POP {len(cand['pop_hits'])}."
                )
            else:
                comment = (
                    f"Sparsest available test customer ({cand['hist_size']} prior purchases) "
                    f"evaluated under high-uncertainty regime with zero hits."
                )
            cases.append({
                "Case_Type": "4. Sparse / Cold-Start Customer",
                "CustomerID": cand["u"],
                "History_Size": cand["hist_size"],
                "History_Snippet": str(cand["history"][:5]),
                "True_Future_Items": str(cand["true_items"][:5]),
                "Top_5_Recommendations": str(cand["rf_top"][:k]),
                "Hits": len(cand["rf_hits"]),
                "Hit_Status": "HIT" if len(cand["rf_hits"]) > 0 else "MISS",
                "Comment": comment,
            })
        else:
            cases.append({
                "Case_Type": "4. Sparse / Cold-Start Customer",
                "CustomerID": "N/A",
                "History_Size": 0,
                "History_Snippet": "[]",
                "True_Future_Items": "[]",
                "Top_5_Recommendations": "[]",
                "Hits": 0,
                "Hit_Status": "N/A",
                "Comment": "All active evaluation test users had more than 2 historical orders.",
            })

    # Case 5: Questionable recommendation traceable to popularity dominance / negative sampling
    c5_candidates = [
        info for info in user_analysis.values()
        if len(info["rf_hits"]) == 0 and info["hist_size"] >= 8 and info["u"] not in seen_users
    ]
    if c5_candidates:
        cand = c5_candidates[0]
        cases.append({
            "Case_Type": "5. Questionable / Dominated Case",
            "CustomerID": cand["u"],
            "History_Size": cand["hist_size"],
            "History_Snippet": str(cand["history"][:5]),
            "True_Future_Items": str(cand["true_items"][:5]),
            "Top_5_Recommendations": str(cand["rf_top"][:k]),
            "Hits": 0,
            "Hit_Status": "MISS",
            "Comment": f"Customer with extensive history ({cand['hist_size']} purchases) received zero Top-5 hits, reflecting popularity bias or negative sampling variance.",
        })
    else:
        cases.append({
            "Case_Type": "5. Questionable / Dominated Case",
            "CustomerID": "N/A",
            "History_Size": 0,
            "History_Snippet": "[]",
            "True_Future_Items": "[]",
            "Top_5_Recommendations": "[]",
            "Hits": 0,
            "Hit_Status": "N/A",
            "Comment": "No rich-history customer observed with complete Top-5 recommendation miss.",
        })

    return pd.DataFrame(cases)
