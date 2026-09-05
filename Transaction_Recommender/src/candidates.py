"""
Candidate generation and negative sampling module for Experiment 07 Recommendation System.
Enforces instructor catalog bounds [500, 2000], evaluates candidate recall, and performs reproducible negative sampling.
"""

from typing import List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd


def build_candidate_universe(
    train_hist: pd.DataFrame,
    item_col: str = "item_id",
    order_col: str = "order_id",
    user_col: str = "user_id",
    min_buyers: Optional[int] = None,
    top_n: Optional[int] = 1000,
    min_bound: int = 500,
    max_bound: int = 2000,
) -> List[str]:
    """
    Construct the bounded eligible candidate universe strictly from training history.

    Parameters:
        train_hist: Historical training transactions DataFrame.
        item_col: Item identifier column.
        order_col: Order identifier column.
        user_col: User identifier column.
        min_buyers: Minimum distinct customers required to be eligible.
        top_n: Maximum top frequent items to select.
        min_bound: Minimum catalog size required by instructor specification (default 500).
        max_bound: Maximum catalog size required by instructor specification (default 2000).

    Returns:
        candidates: List of eligible item IDs satisfying the bound constraints.
    """
    if min_buyers is not None:
        item_stats = (
            train_hist.groupby(item_col)[user_col]
            .nunique()
            .sort_values(ascending=False)
        )
        eligible = item_stats[item_stats >= min_buyers].index.tolist()
    else:
        item_stats = (
            train_hist.groupby(item_col)[order_col]
            .nunique()
            .sort_values(ascending=False)
        )
        eligible = item_stats.index.tolist()

    if top_n is not None:
        eligible = eligible[:top_n]

    # Enforce instructor bound [500, 2000]
    catalog_size = len(eligible)
    if not (min_bound <= catalog_size <= max_bound):
        raise AssertionError(
            f"Candidate catalog size {catalog_size} violates required bounds [{min_bound}, {max_bound}]. "
            f"Adjust top_n or min_buyers parameter."
        )

    return [str(i) for i in eligible]


def candidate_recall(
    test_future: pd.DataFrame,
    candidates: Sequence[str],
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> Tuple[float, float]:
    """
    Evaluate retrieval-stage candidate recall over future relevant items.

    Parameters:
        test_future: Future evaluation transactions DataFrame.
        candidates: Sequence of eligible candidate item IDs.
        user_col: User identifier column.
        item_col: Item identifier column.

    Returns:
        recall_float: Fraction of future relevant item interactions present in candidate universe.
        pct_users_fully_covered: Percentage of evaluation users for whom all future items are in candidates.
    """
    candidate_set: Set[str] = set(str(c) for c in candidates)
    future_pairs = test_future[[user_col, item_col]].drop_duplicates().copy()
    future_pairs[item_col] = future_pairs[item_col].astype(str)

    total_relevant = len(future_pairs)
    if total_relevant == 0:
        return 0.0, 0.0

    in_candidate = future_pairs[item_col].isin(candidate_set)
    captured_relevant = int(in_candidate.sum())
    recall_float = float(captured_relevant / total_relevant)

    # Per-user coverage: check if all items for each user are in candidates
    user_groups = future_pairs.groupby(user_col)[item_col].apply(set)
    users_fully_covered = sum(
        1 for items in user_groups if items.issubset(candidate_set)
    )
    total_users = len(user_groups)
    pct_users_covered = float(
        (users_fully_covered / max(total_users, 1)) * 100.0
    )

    return round(recall_float, 4), round(pct_users_covered, 2)


def sample_negatives(
    user_id: Union[str, int],
    positives: Sequence[str],
    candidate_items: Sequence[str],
    n_neg: int = 50,
    seed: int = 42,
) -> List[str]:
    """
    Draw reproducible negative samples from candidate universe excluding positive interactions.

    Parameters:
        user_id: User identifier (used with seed to ensure unique deterministic sampling per user).
        positives: Sequence of items positively interacted with by user.
        candidate_items: Full candidate universe.
        n_neg: Number of negative candidates to sample.
        seed: Base random seed.

    Returns:
        sampled_negatives: List of sampled negative item IDs.
    """
    # Deterministic user-specific seed combining base seed and user hash
    user_hash = abs(hash(str(user_id))) % 1_000_000
    rng = np.random.default_rng(seed + user_hash)

    pos_set = set(str(p) for p in positives)
    pool = np.array([c for c in candidate_items if str(c) not in pos_set])

    if len(pool) == 0:
        return []

    sample_size = min(n_neg, len(pool))
    sampled = rng.choice(pool, size=sample_size, replace=False).tolist()
    return [str(s) for s in sampled]
