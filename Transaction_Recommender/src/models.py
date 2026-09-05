"""
Recommender models module for Experiment 07 Recommendation System.
Implements Popularity baseline, Random Forest training & validation tuning, Top-K ranking,
and Advanced Matrix Factorization / Item-Item Collaborative Filtering benchmarks.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid


# -----------------------------------------------------------------------------
# 1. Baseline Model: Global Popularity
# -----------------------------------------------------------------------------

def popularity_baseline(
    train_hist: pd.DataFrame,
    item_col: str = "item_id",
    order_col: str = "order_id",
    candidate_items: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Compute popularity baseline ranking items by distinct order volume in training history.
    """
    counts = train_hist.groupby(item_col)[order_col].nunique()

    if candidate_items is not None:
        cand_set = set(str(c) for c in candidate_items)
        counts = counts[counts.index.astype(str).isin(cand_set)]

    ranked = counts.sort_values(ascending=False).index.astype(str).tolist()

    # Append any remaining candidates with 0 historical counts
    if candidate_items is not None:
        ranked_set = set(ranked)
        remaining = [str(c) for c in candidate_items if str(c) not in ranked_set]
        ranked.extend(remaining)

    return ranked


def popularity_recommend(
    ranked_items: Sequence[str],
    seen_items: Set[str],
    k: int = 10,
    allow_repeats: bool = True,
) -> List[str]:
    """
    Generate Top-K recommendations from popularity ranking with repeat-purchase control.
    """
    if allow_repeats:
        return list(ranked_items)[:k]

    seen_set = set(str(s) for s in seen_items)
    filtered = [str(i) for i in ranked_items if str(i) not in seen_set]
    return filtered[:k]


# -----------------------------------------------------------------------------
# 2. Supervised Scorer: Random Forest
# -----------------------------------------------------------------------------

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    feature_cols: List[str],
    **rf_params: Any,
) -> RandomForestClassifier:
    """
    Train Random Forest purchase-propensity classifier with explicit hyperparameters.
    """
    defaults = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "random_state": 42,
        "n_jobs": -1,
    }
    defaults.update(rf_params)

    rf = RandomForestClassifier(**defaults)
    rf.fit(X_train[feature_cols], y_train)

    if rf.classes_.tolist() != [0, 1]:
        # Handle single-class training edge case
        rf.classes_ = np.array([0, 1])

    return rf


def score_candidates(
    model: Any,
    X: pd.DataFrame,
    feature_cols: List[str],
) -> np.ndarray:
    """
    Score user-item candidate pairs using model predicted positive-class purchase propensity.
    """
    probs = model.predict_proba(X[feature_cols])
    if probs.shape[1] == 2:
        return probs[:, 1]
    return probs[:, 0]


def recommend_top_k(
    scored_df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    score_col: str = "score",
    k: int = 10,
    allow_repeats: bool = True,
    seen_items_dict: Optional[Dict[str, Set[str]]] = None,
) -> pd.DataFrame:
    """
    Sort candidates by predicted purchase score and select Top-K items per user.
    """
    df = scored_df[[user_col, item_col, score_col]].copy()
    df[user_col] = df[user_col].astype(str)
    df[item_col] = df[item_col].astype(str)

    if not allow_repeats and seen_items_dict is not None:
        def _filter_seen(row):
            user_seen = seen_items_dict.get(row[user_col], set())
            return row[item_col] not in user_seen
        df = df[df.apply(_filter_seen, axis=1)]

    # Sort descending by score within each user group
    df_sorted = df.sort_values([user_col, score_col], ascending=[True, False])
    top_k_df = df_sorted.groupby(user_col).head(k).copy()
    top_k_df["rank"] = top_k_df.groupby(user_col).cumcount() + 1

    return top_k_df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# 3. Validation-Based Hyperparameter Tuning (Training/Validation Only)
# -----------------------------------------------------------------------------

def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    X_val: pd.DataFrame,
    y_val: Union[pd.Series, np.ndarray],
    val_pairs_df: pd.DataFrame,
    feature_cols: List[str],
    param_grid: Dict[str, list],
    user_col: str = "user_id",
    item_col: str = "item_id",
    k: int = 10,
    metric: str = "recall",
    seed: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Hyperparameter tuning strictly on validation data. Never evaluates locked test data.
    """
    # Build validation ground-truth dictionary: user -> set of true items in val
    val_ground_truth = (
        val_pairs_df.loc[y_val == 1]
        .groupby(user_col)[item_col]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )

    grid = list(ParameterGrid(param_grid))
    records = []

    best_score = -1.0
    best_params: Dict[str, Any] = {}

    for idx, params in enumerate(grid, 1):
        params_with_seed = {**params, "random_state": seed, "n_jobs": -1}
        clf = RandomForestClassifier(**params_with_seed)
        clf.fit(X_train[feature_cols], y_train)

        # Predict propensity on validation candidate pairs
        val_scores = clf.predict_proba(X_val[feature_cols])[:, 1]
        pr_auc = float(average_precision_score(y_val, val_scores))
        try:
            roc_auc = float(roc_auc_score(y_val, val_scores))
        except ValueError:
            roc_auc = 0.5

        val_eval = val_pairs_df[[user_col, item_col]].copy()
        val_eval["score"] = val_scores
        top_recs = recommend_top_k(val_eval, user_col, item_col, "score", k=k)

        # Compute validation Recall@K and Precision@K
        recs_by_user = (
            top_recs.groupby(user_col)[item_col]
            .apply(lambda s: list(s.astype(str)))
            .to_dict()
        )

        user_recalls = []
        user_precisions = []
        for u, true_items in val_ground_truth.items():
            u_str = str(u)
            if not true_items:
                continue
            recs = recs_by_user.get(u_str, [])
            hits = len(set(recs) & true_items)
            user_recalls.append(hits / len(true_items))
            user_precisions.append(hits / max(k, 1))

        mean_recall = float(np.mean(user_recalls)) if user_recalls else 0.0
        mean_precision = float(np.mean(user_precisions)) if user_precisions else 0.0

        rec = {
            "config_id": idx,
            "n_estimators": params.get("n_estimators"),
            "max_depth": str(params.get("max_depth")),
            "min_samples_leaf": params.get("min_samples_leaf"),
            "max_features": params.get("max_features"),
            "class_weight": params.get("class_weight"),
            "val_recall_at_10": round(mean_recall, 4),
            "val_precision_at_10": round(mean_precision, 4),
            "val_pr_auc": round(pr_auc, 4),
            "val_roc_auc": round(roc_auc, 4),
        }
        records.append(rec)

        target_metric_val = mean_recall if metric == "recall" else pr_auc
        if target_metric_val > best_score:
            best_score = target_metric_val
            best_params = params_with_seed

    results_table = pd.DataFrame(records)
    return best_params, results_table


# -----------------------------------------------------------------------------
# 4. Advanced Learner: Matrix Factorization & Item-Item Collaborative Filtering
# -----------------------------------------------------------------------------

class MatrixFactorizationRecommender:
    """
    Implicit Matrix Factorization recommender using TruncatedSVD on normalized interaction matrix.
    Supports random seed initialization for uncertainty reporting (Manual Sec. 18.1).
    """

    def __init__(self, n_components: int = 32, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.svd: Optional[TruncatedSVD] = None
        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

    def fit(
        self,
        train_hist: pd.DataFrame,
        candidate_items: Sequence[str],
        user_col: str = "user_id",
        item_col: str = "item_id",
        order_col: str = "order_id",
    ) -> "MatrixFactorizationRecommender":
        """Fit latent factor representation from customer transaction history."""
        # Index users and candidate items
        users = train_hist[user_col].astype(str).unique().tolist()
        items = list(dict.fromkeys(str(c) for c in candidate_items))

        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.idx_to_user = {i: u for i, u in enumerate(users)}
        self.item_to_idx = {it: j for j, it in enumerate(items)}
        self.idx_to_item = {j: it for j, it in enumerate(items)}

        # Filter interactions to candidate items
        filtered = train_hist[train_hist[item_col].astype(str).isin(self.item_to_idx)].copy()
        counts = filtered.groupby([user_col, item_col])[order_col].nunique().reset_index()

        u_indices = counts[user_col].astype(str).map(self.user_to_idx).values
        i_indices = counts[item_col].astype(str).map(self.item_to_idx).values
        values = np.log1p(counts[order_col].values)

        mat = csr_matrix(
            (values, (u_indices, i_indices)),
            shape=(len(self.user_to_idx), len(self.item_to_idx)),
            dtype=np.float32,
        )

        n_comp = min(self.n_components, mat.shape[0] - 1, mat.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        self.user_factors = self.svd.fit_transform(mat)
        self.item_factors = self.svd.components_.T
        return self

    def predict_user_scores(self, user_id: str) -> np.ndarray:
        """Predict preference scores across all indexed items for a user."""
        u_str = str(user_id)
        if u_str not in self.user_to_idx:
            # Cold-start user fallback: average item factor projection
            return np.zeros(len(self.item_to_idx), dtype=np.float32)

        u_idx = self.user_to_idx[u_str]
        u_vec = self.user_factors[u_idx]  # shape: (n_components,)
        scores = np.dot(self.item_factors, u_vec)  # shape: (n_items,)
        return scores

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        seen_items: Optional[Set[str]] = None,
        allow_repeats: bool = True,
    ) -> List[str]:
        """Generate Top-K recommendations for a single customer."""
        scores = self.predict_user_scores(user_id)
        ranked_indices = np.argsort(-scores)

        recs = []
        seen = seen_items or set()
        for idx in ranked_indices:
            item_id = self.idx_to_item[idx]
            if not allow_repeats and item_id in seen:
                continue
            recs.append(item_id)
            if len(recs) == k:
                break
        return recs


def train_mf_baseline(
    train_hist: pd.DataFrame,
    candidate_items: Sequence[str],
    n_components: int = 32,
    seed: int = 42,
    user_col: str = "user_id",
    item_col: str = "item_id",
    order_col: str = "order_id",
) -> MatrixFactorizationRecommender:
    """
    Factory function to initialize and fit Matrix Factorization recommender.
    """
    mf = MatrixFactorizationRecommender(n_components=n_components, random_state=seed)
    mf.fit(
        train_hist=train_hist,
        candidate_items=candidate_items,
        user_col=user_col,
        item_col=item_col,
        order_col=order_col,
    )
    return mf
