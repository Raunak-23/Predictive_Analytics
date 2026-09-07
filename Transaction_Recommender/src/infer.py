"""
CLI Model Inference & Recommendation Engine for Experiment 07.
Allows loading trained pipelines and models (Random Forest, Matrix Factorization, Popularity)
and generating Top-K personalized recommendations on custom user data or customer queries.

Usage Examples:
    # 1. Infer Top-10 items for a customer using the default D1 (Online Retail) Random Forest model
    python src/infer.py --dataset d1 --user 12349 --top-k 10

    # 2. Infer using D2 (Instacart) Matrix Factorization model
    python src/infer.py --dataset d2 --model-type mf --user 2640 --top-k 5

    # 3. Infer using Popularity baseline
    python src/infer.py --dataset d1 --model-type pop --user 12349 --top-k 5

    # 4. Infer on custom transaction dataset (CSV/JSON)
    python src/infer.py --data path/to/my_transactions.csv --output recommendations.csv

    # 5. Ad-hoc items query (predict recommendations for a user who bought specific items)
    python src/infer.py --user 99999 --items "85123A,22423,47566" --top-k 5

    # 6. Interactive wizard mode
    python src/infer.py --interactive
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import joblib
import numpy as np
import pandas as pd

# Reconfigure stdout for safe terminal printing on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ARTIFACTS_DIR, DATASET_SCHEMAS, MODELS_DIR, PROCESSED_DIR, RAW_DIR
from src.features import (
    build_feature_matrix,
    customer_features,
    item_features,
    pair_features,
)
from src.models import (
    MatrixFactorizationRecommender,
    popularity_baseline,
    popularity_recommend,
    recommend_top_k,
    score_candidates,
)


# =============================================================================
# Helper Utilities: Schema Mapping, Model & Catalog Loaders
# =============================================================================

COLUMN_ALIASES = {
    "user_id": ["CustomerID", "customer_id", "user_id", "user", "shopper_id", "ShopperID", "client_id"],
    "item_id": ["StockCode", "product_id", "item_id", "item", "ProductCode", "sku", "item_code"],
    "order_id": ["InvoiceNo", "order_id", "order", "invoice", "transaction_id", "basket_id"],
    "ts": ["InvoiceDate", "ts", "date", "timestamp", "OrderSequence", "order_time", "time"],
    "quantity": ["Quantity", "quantity", "qty", "count", "units"],
    "amount": ["Amount", "amount", "spend", "total_price", "TotalPrice", "line_total"],
    "price": ["UnitPrice", "unit_price", "price", "item_price"],
    "description": ["Description", "description", "product_name", "item_name", "title", "name"],
}


def normalize_dataset_name(name: str) -> str:
    """Map user-provided dataset strings to canonical keys."""
    n = name.lower().strip()
    if n in ["d1", "d1_online_retail", "online_retail", "retail"]:
        return "d1_online_retail"
    if n in ["d2", "d2_advanced", "instacart", "d3"]:
        return "d2_advanced"
    return name


def detect_and_normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Detect column headers from flexible naming variations and harmonize to canonical schema:
    [user_id, item_id, order_id, ts, quantity, amount].
    """
    clean_df = df.copy()
    col_mapping = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        found = None
        for alias in aliases:
            if alias in clean_df.columns:
                found = alias
                break
        if found:
            col_mapping[canonical] = found
            if found != canonical:
                clean_df[canonical] = clean_df[found]

    # Defaults for missing non-critical columns
    if "user_id" not in clean_df.columns:
        raise ValueError("Custom data must contain a user identifier column (e.g. CustomerID, user_id).")
    if "item_id" not in clean_df.columns:
        raise ValueError("Custom data must contain an item identifier column (e.g. StockCode, product_id, item_id).")

    if "order_id" not in clean_df.columns:
        clean_df["order_id"] = clean_df.index.astype(str)
        col_mapping["order_id"] = "synthetic_order_id"

    if "quantity" not in clean_df.columns:
        clean_df["quantity"] = 1.0
        col_mapping["quantity"] = "default_1.0"
    else:
        clean_df["quantity"] = pd.to_numeric(clean_df["quantity"], errors="coerce").fillna(1.0)

    if "price" in clean_df.columns and "amount" not in clean_df.columns:
        clean_df["price"] = pd.to_numeric(clean_df["price"], errors="coerce").fillna(1.0)
        clean_df["amount"] = clean_df["quantity"] * clean_df["price"]
        col_mapping["amount"] = "derived(quantity*price)"
    elif "amount" in clean_df.columns:
        clean_df["amount"] = pd.to_numeric(clean_df["amount"], errors="coerce").fillna(clean_df["quantity"])
    else:
        clean_df["amount"] = clean_df["quantity"]
        col_mapping["amount"] = "default_quantity"

    if "ts" in clean_df.columns:
        try:
            clean_df["ts"] = pd.to_datetime(clean_df["ts"])
        except Exception:
            clean_df["ts"] = pd.to_datetime("2021-01-01")
    else:
        clean_df["ts"] = pd.to_datetime("2021-01-01")
        col_mapping["ts"] = "default_timestamp"

    clean_df["user_id"] = clean_df["user_id"].astype(str)
    clean_df["item_id"] = clean_df["item_id"].astype(str)
    clean_df["order_id"] = clean_df["order_id"].astype(str)

    return clean_df, col_mapping


def load_model(
    dataset: str,
    model_type: str = "rf",
    model_path: Optional[Union[str, Path]] = None,
) -> Any:
    """Load serialized model object (Random Forest, Matrix Factorization, or Popularity list)."""
    dataset_key = normalize_dataset_name(dataset)

    if model_path is not None:
        target_path = Path(model_path)
    else:
        m_dir = MODELS_DIR / dataset_key
        if model_type.lower() in ["rf", "random_forest"]:
            target_path = m_dir / "random_forest.joblib"
        elif model_type.lower() in ["mf", "matrix_factorization"]:
            target_path = m_dir / "matrix_factorization.joblib"
        elif model_type.lower() in ["pop", "popularity"]:
            target_path = m_dir / "popularity.joblib"
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose 'rf', 'mf', or 'pop'.")

    if not target_path.exists():
        raise FileNotFoundError(
            f"Model file not found at: {target_path}\n"
            f"Please ensure pipeline models are trained or provide a valid --model-path."
        )

    return joblib.load(target_path)


def load_candidate_items(dataset: str) -> List[str]:
    """Retrieve indexed candidate items for recommendation generation."""
    dataset_key = normalize_dataset_name(dataset)
    cands_file = ARTIFACTS_DIR / dataset_key / "candidate_items.json"
    if cands_file.exists():
        with open(cands_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback to popularity joblib if present
    pop_file = MODELS_DIR / dataset_key / "popularity.joblib"
    if pop_file.exists():
        return joblib.load(pop_file)

    # Fallback to default recommendations items
    recs_file = REPO_ROOT / "results" / dataset_key / "Recommendations.csv"
    if recs_file.exists():
        recs_df = pd.read_csv(recs_file)
        return recs_df["item_id"].astype(str).unique().tolist()

    raise FileNotFoundError(f"Could not locate candidate universe items for dataset: {dataset_key}")


def load_item_catalog(dataset: str) -> Dict[str, str]:
    """Load product ID to Description/Name mapping for human-readable output."""
    dataset_key = normalize_dataset_name(dataset)
    meta_file = ARTIFACTS_DIR / dataset_key / "item_metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Extract on-the-fly if needed
    if dataset_key == "d1_online_retail":
        d1_csv = PROCESSED_DIR / "online_retail_cleaned.csv"
        if d1_csv.exists():
            df = pd.read_csv(d1_csv, usecols=["StockCode", "Description"]).dropna().drop_duplicates("StockCode")
            return dict(zip(df["StockCode"].astype(str), df["Description"].astype(str)))
    elif dataset_key == "d2_advanced":
        d2_csv = RAW_DIR / "d2_advanced" / "instacart_transactions.csv"
        if d2_csv.exists():
            df = pd.read_csv(d2_csv, usecols=["StockCode", "Description"]).dropna().drop_duplicates("StockCode")
            return dict(zip(df["StockCode"].astype(str), df["Description"].astype(str)))

    return {}


def load_feature_columns(dataset: str) -> List[str]:
    """Load canonical feature schema columns."""
    dataset_key = normalize_dataset_name(dataset)
    schema_file = ARTIFACTS_DIR / dataset_key / "feature_schema.json"
    if schema_file.exists():
        with open(schema_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("feature_columns", [])

    # Default fallback to the 20 standard features
    return [
        "cust_txns", "cust_items", "cust_qty", "cust_spend", "cust_recency_days",
        "cust_avg_basket_spend", "cust_avg_basket_size",
        "item_txns", "item_buyers", "item_qty", "item_spend", "item_avg_price",
        "item_recency_days", "item_repeat_rate",
        "pair_purchases", "pair_qty", "pair_spend", "pair_recency_days",
        "pair_repeat_indicator", "pair_spend_share"
    ]


def load_default_transactions(dataset: str) -> pd.DataFrame:
    """Load default processed transaction data for the chosen dataset."""
    dataset_key = normalize_dataset_name(dataset)
    if dataset_key == "d1_online_retail":
        p = PROCESSED_DIR / "online_retail_cleaned.csv"
        if not p.exists():
            p = RAW_DIR / "online_retail.csv"
    else:
        p = RAW_DIR / "d2_advanced" / "instacart_transactions.csv"
        if not p.exists():
            p = PROCESSED_DIR / "instacart_transactions_sample.csv"

    if not p.exists():
        raise FileNotFoundError(f"Default transaction dataset not found at: {p}")

    df = pd.read_csv(p)
    clean_df, _ = detect_and_normalize_columns(df)
    return clean_df


# =============================================================================
# Inference Core Engine
# =============================================================================

def run_rf_inference(
    model: Any,
    history_df: pd.DataFrame,
    target_users: List[str],
    candidates: List[str],
    feature_cols: List[str],
    k: int = 10,
    allow_repeats: bool = True,
    cutoff_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Score candidate items for target users using Random Forest model."""
    if cutoff_date is None:
        cutoff_date = history_df["ts"].max() if len(history_df) > 0 else pd.to_datetime("2022-01-01")

    # Compute features strictly from provided history
    cf = customer_features(history_df, cutoff_date) if len(history_df) > 0 else pd.DataFrame(columns=["user_id"])
    it = item_features(history_df, cutoff_date) if len(history_df) > 0 else pd.DataFrame(columns=["item_id"])
    pf = pair_features(history_df, cutoff_date) if len(history_df) > 0 else pd.DataFrame(columns=["user_id", "item_id"])

    # Build scoring pairs (each target user paired with each candidate)
    pair_records = []
    for u in target_users:
        u_str = str(u)
        for c in candidates:
            pair_records.append({"user_id": u_str, "item_id": str(c)})

    pairs_df = pd.DataFrame(pair_records)
    if pairs_df.empty:
        return pd.DataFrame(columns=["user_id", "item_id", "score", "rank"])

    # Construct feature matrix (handles unobserved/cold users and items via robust fallbacks)
    X_mat, _ = build_feature_matrix(pairs_df, cf, it, pf)

    # Score candidates using positive-class propensity
    scores = score_candidates(model, X_mat, feature_cols)
    pairs_df["score"] = scores

    # Track seen items if repeat purchases are not allowed
    seen_dict = None
    if not allow_repeats and len(history_df) > 0:
        seen_dict = history_df.groupby("user_id")["item_id"].apply(lambda s: set(s.astype(str))).to_dict()

    top_recs = recommend_top_k(
        pairs_df,
        user_col="user_id",
        item_col="item_id",
        score_col="score",
        k=k,
        allow_repeats=allow_repeats,
        seen_items_dict=seen_dict,
    )
    return top_recs


def run_mf_inference(
    mf_model: MatrixFactorizationRecommender,
    history_df: Optional[pd.DataFrame],
    target_users: List[str],
    k: int = 10,
    allow_repeats: bool = True,
) -> pd.DataFrame:
    """Generate recommendations using Matrix Factorization model."""
    seen_dict = {}
    if not allow_repeats and history_df is not None and len(history_df) > 0:
        seen_dict = history_df.groupby("user_id")["item_id"].apply(lambda s: set(s.astype(str))).to_dict()

    rows = []
    for u in target_users:
        u_str = str(u)
        seen = seen_dict.get(u_str, set())
        recs = mf_model.recommend(user_id=u_str, k=k, seen_items=seen, allow_repeats=allow_repeats)
        scores = mf_model.predict_user_scores(u_str)

        for rank, item_id in enumerate(recs, 1):
            # Find item index for score
            idx = mf_model.item_to_idx.get(item_id)
            score_val = float(scores[idx]) if idx is not None else 0.0
            rows.append({
                "user_id": u_str,
                "item_id": str(item_id),
                "score": round(score_val, 4),
                "rank": rank,
            })

    return pd.DataFrame(rows)


def run_popularity_inference(
    pop_items: List[str],
    history_df: Optional[pd.DataFrame],
    target_users: List[str],
    k: int = 10,
    allow_repeats: bool = True,
) -> pd.DataFrame:
    """Generate recommendations using Popularity baseline ranking."""
    seen_dict = {}
    if not allow_repeats and history_df is not None and len(history_df) > 0:
        seen_dict = history_df.groupby("user_id")["item_id"].apply(lambda s: set(s.astype(str))).to_dict()

    rows = []
    for u in target_users:
        u_str = str(u)
        seen = seen_dict.get(u_str, set())
        recs = popularity_recommend(pop_items, seen, k=k, allow_repeats=allow_repeats)
        for rank, item_id in enumerate(recs, 1):
            rows.append({
                "user_id": u_str,
                "item_id": str(item_id),
                "score": round(1.0 / rank, 4),
                "rank": rank,
            })

    return pd.DataFrame(rows)


# =============================================================================
# CLI Formatting & Pretty Printing
# =============================================================================

def format_recommendations_table(
    recs_df: pd.DataFrame,
    catalog: Dict[str, str],
    show_score: bool = True,
) -> str:
    """Format recommendation output into an aligned, readable terminal table."""
    if recs_df.empty:
        return "No recommendations generated."

    lines = []
    user_groups = recs_df.groupby("user_id")

    for user_id, grp in user_groups:
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"  Top Recommendations for Customer: {user_id}")
        lines.append("=" * 80)

        header = f"  {'Rank':<5} | {'Item ID':<10} | {'Propensity / Score':<18} | {'Description'}"
        lines.append(header)
        lines.append(f"  {'-'*5}-+-{'-'*10}-+-{'-'*18}-+-{'-'*35}")

        for _, row in grp.iterrows():
            item_id = str(row["item_id"])
            desc = catalog.get(item_id, catalog.get(str(item_id).split(".")[0], "N/A"))
            if len(desc) > 38:
                desc = desc[:35] + "..."
            score_str = f"{row['score']:.4f} ({row['score']*100:.1f}%)" if show_score else "N/A"
            line = f"  {int(row['rank']):<5} | {item_id:<10} | {score_str:<18} | {desc}"
            lines.append(line)

    lines.append("")
    return "\n".join(lines)


# =============================================================================
# Interactive Mode
# =============================================================================

def interactive_wizard():
    """Launch terminal wizard prompting user for inference configuration."""
    print("\n" + "=" * 70)
    print("      Transaction Recommender System - Interactive Inference Wizard")
    print("=" * 70)

    # 1. Dataset
    print("\n1. Select Base Dataset / Domain:")
    print("   [1] D1: UCI Online Retail (E-commerce Giftware)")
    print("   [2] D2: Instacart Benchmark (Grocery Market Basket)")
    ds_choice = input("   Enter choice (1 or 2, default: 1): ").strip()
    dataset = "d2_advanced" if ds_choice == "2" else "d1_online_retail"

    # 2. Model Type
    print(f"\n2. Select Model Architecture for {dataset}:")
    print("   [1] Random Forest (Propensity Classifier & Candidate Ranker)")
    if dataset == "d2_advanced":
        print("   [2] Matrix Factorization (Latent SVD Collaborative Filtering)")
        print("   [3] Popularity Baseline (Order-Volume Ranker)")
    else:
        print("   [2] Popularity Baseline (Order-Volume Ranker)")
    m_choice = input("   Enter choice (default: 1): ").strip()

    if dataset == "d2_advanced":
        model_type = "mf" if m_choice == "2" else ("pop" if m_choice == "3" else "rf")
    else:
        model_type = "pop" if m_choice == "2" else "rf"

    # 3. Custom Data or Built-in Dataset
    print("\n3. Data Source:")
    custom_path = input("   Enter path to custom transaction CSV (or press Enter to use default dataset): ").strip()

    # 4. User ID
    print("\n4. Target Customer Query:")
    user_id = input("   Enter Customer ID (or press Enter to infer for sample users): ").strip()

    # 5. Top K
    print("\n5. Number of Recommendations (Top-K):")
    k_str = input("   Enter K (default: 10): ").strip()
    k = int(k_str) if k_str.isdigit() else 10

    # 6. Allow Repeats
    rep_choice = input("\n6. Allow repeat purchases (repurchase of previously bought items)? (y/n, default: y): ").strip().lower()
    allow_repeats = rep_choice != "n"

    # 7. Output File
    out_file = input("\n7. Save recommendations to file? (Enter filename like recs.csv, or press Enter for console only): ").strip()

    print("\nExecuting inference...\n")

    # Run inference with collected arguments
    args = argparse.Namespace(
        dataset=dataset,
        model_type=model_type,
        model_path=None,
        data=custom_path if custom_path else None,
        features=None,
        user=user_id if user_id else None,
        items=None,
        top_k=k,
        allow_repeats=allow_repeats,
        max_users=10,
        output=out_file if out_file else None,
        format="table",
        quiet=False,
        interactive=False,
    )
    main(args)


# =============================================================================
# Main CLI Entrypoint
# =============================================================================

def parse_cli_args() -> argparse.Namespace:
    """Parse command line flags and arguments."""
    parser = argparse.ArgumentParser(
        description="Transaction Recommender System: Model Loading & CLI Inference Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score customer 12349 on D1 Random Forest:
  python src/infer.py --dataset d1 --user 12349 --top-k 10

  # Score customer 2640 on D2 Matrix Factorization:
  python src/infer.py --dataset d2 --model-type mf --user 2640 --top-k 5

  # Run inference on custom transactions CSV and save to output:
  python src/infer.py --data my_data.csv --output recommendations.csv

  # Ad-hoc recommendation query with specific purchased items:
  python src/infer.py --user 99999 --items "85123A,22423,47566" --top-k 5

  # Launch interactive wizard:
  python src/infer.py --interactive
        """,
    )

    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="d1_online_retail",
        help="Dataset name or key: 'd1' (UCI Online Retail, default) or 'd2' (Instacart).",
    )
    parser.add_argument(
        "-m", "--model-type",
        type=str,
        default="rf",
        choices=["rf", "random_forest", "mf", "matrix_factorization", "pop", "popularity"],
        help="Model architecture: 'rf' (Random Forest, default), 'mf' (Matrix Factorization), or 'pop' (Popularity).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Custom file path to serialized .joblib model file (overrides default model lookup).",
    )
    parser.add_argument(
        "--data", "-i",
        type=str,
        default=None,
        help="Path to custom transaction dataset (CSV/JSON/Parquet) for inference.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Path to pre-computed feature matrix CSV to score directly.",
    )
    parser.add_argument(
        "--user", "-u",
        type=str,
        default=None,
        help="Target CustomerID / user_id to generate recommendations for.",
    )
    parser.add_argument(
        "--items",
        type=str,
        default=None,
        help="Comma-separated item IDs for ad-hoc basket inference (e.g. '85123A,22423,47566').",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=10,
        help="Number of recommendations to produce per user (default: 10).",
    )
    parser.add_argument(
        "--allow-repeats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to allow repeat purchase recommendations (default: --allow-repeats).",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=50,
        help="Maximum number of users to score when inferring across an entire dataset (default: 50).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Optional file path to save recommendations (CSV or JSON).",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["table", "csv", "json"],
        default="table",
        help="Output display format: 'table' (default ASCII table), 'csv', or 'json'.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode: suppress informational logs and print only recommendations.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive terminal wizard.",
    )

    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None):
    """Main execution pipeline for CLI inference."""
    if args is None:
        args = parse_cli_args()

    if args.interactive:
        interactive_wizard()
        return

    dataset = normalize_dataset_name(args.dataset)
    model_type = args.model_type.lower()
    if not args.quiet:
        print(f"[Infer] Loading model ({model_type.upper()}) for dataset: {dataset}...")

    # Load Model
    model = load_model(dataset, model_type=model_type, model_path=args.model_path)
    catalog = load_item_catalog(dataset)
    candidates = load_candidate_items(dataset)
    feature_cols = load_feature_columns(dataset)

    # -------------------------------------------------------------------------
    # Path A: Pre-computed Features Scoring
    # -------------------------------------------------------------------------
    if args.features:
        if not args.quiet:
            print(f"[Infer] Scoring pre-computed feature matrix: {args.features}")
        feat_df = pd.read_csv(args.features)
        missing = [c for c in feature_cols if c not in feat_df.columns]
        if missing:
            raise ValueError(f"Feature matrix is missing required columns: {missing}")

        scores = score_candidates(model, feat_df, feature_cols)
        feat_df["score"] = scores
        top_recs = recommend_top_k(
            feat_df,
            user_col="user_id",
            item_col="item_id",
            score_col="score",
            k=args.top_k,
            allow_repeats=args.allow_repeats,
        )

    # -------------------------------------------------------------------------
    # Path B: Ad-hoc Basket / Items Query
    # -------------------------------------------------------------------------
    elif args.items:
        user_id = args.user or "adhoc_customer"
        item_list = [i.strip() for i in args.items.split(",") if i.strip()]
        if not args.quiet:
            print(f"[Infer] Ad-hoc inference for user '{user_id}' with {len(item_list)} seed items: {item_list}")

        # Build synthetic history representing this basket
        now = pd.to_datetime("2021-01-01 12:00:00")
        synth_history = pd.DataFrame([
            {
                "user_id": user_id,
                "item_id": itm,
                "order_id": "adhoc_basket",
                "ts": now,
                "quantity": 1.0,
                "amount": 10.0,
            }
            for itm in item_list
        ])

        if model_type in ["rf", "random_forest"]:
            top_recs = run_rf_inference(
                model=model,
                history_df=synth_history,
                target_users=[user_id],
                candidates=candidates,
                feature_cols=feature_cols,
                k=args.top_k,
                allow_repeats=args.allow_repeats,
                cutoff_date=now,
            )
        elif model_type in ["mf", "matrix_factorization"]:
            top_recs = run_mf_inference(
                mf_model=model,
                history_df=synth_history,
                target_users=[user_id],
                k=args.top_k,
                allow_repeats=args.allow_repeats,
            )
        else:
            top_recs = run_popularity_inference(
                pop_items=candidates,
                history_df=synth_history,
                target_users=[user_id],
                k=args.top_k,
                allow_repeats=args.allow_repeats,
            )

    # -------------------------------------------------------------------------
    # Path C: Transaction Data Inference (Custom Data File or Default Dataset)
    # -------------------------------------------------------------------------
    else:
        if args.data:
            if not args.quiet:
                print(f"[Infer] Loading custom transaction data from: {args.data}")
            if args.data.endswith(".parquet"):
                raw_df = pd.read_parquet(args.data)
            elif args.data.endswith(".json"):
                raw_df = pd.read_json(args.data)
            else:
                raw_df = pd.read_csv(args.data)
            history_df, mapping = detect_and_normalize_columns(raw_df)
            if not args.quiet:
                print(f"        Mapped columns: {mapping}")
                print(f"        Transactions: {len(history_df):,} rows | Unique Users: {history_df['user_id'].nunique():,}")

            # Augment catalog with any descriptions found in custom data
            if "description" in history_df.columns:
                desc_map = history_df.dropna(subset=["description"]).drop_duplicates("item_id")
                for _, r in desc_map.iterrows():
                    catalog[str(r["item_id"])] = str(r["description"])
        else:
            if not args.quiet:
                print(f"[Infer] Using default processed transaction dataset for: {dataset}")
            history_df = load_default_transactions(dataset)

        # Determine target user(s)
        available_users = history_df["user_id"].unique().tolist()
        if args.user:
            target_user = str(args.user)
            target_users = [target_user]
            if target_user not in available_users and not args.quiet:
                print(f"  [Notice] Customer '{target_user}' not found in provided history. Generating cold-start recommendations.")
        else:
            # Score top active users up to max_users
            active_users = history_df["user_id"].value_counts().head(args.max_users).index.tolist()
            target_users = [str(u) for u in active_users]
            if not args.quiet:
                print(f"  [Infer] Scoring Top {len(target_users)} active customers from dataset...")

        # Execute model-specific inference
        if model_type in ["rf", "random_forest"]:
            top_recs = run_rf_inference(
                model=model,
                history_df=history_df,
                target_users=target_users,
                candidates=candidates,
                feature_cols=feature_cols,
                k=args.top_k,
                allow_repeats=args.allow_repeats,
            )
        elif model_type in ["mf", "matrix_factorization"]:
            top_recs = run_mf_inference(
                mf_model=model,
                history_df=history_df,
                target_users=target_users,
                k=args.top_k,
                allow_repeats=args.allow_repeats,
            )
        elif model_type in ["pop", "popularity"]:
            top_recs = run_popularity_inference(
                pop_items=candidates,
                history_df=history_df,
                target_users=target_users,
                k=args.top_k,
                allow_repeats=args.allow_repeats,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    # -------------------------------------------------------------------------
    # Output Results
    # -------------------------------------------------------------------------
    # Append item description to output dataframe
    top_recs["description"] = top_recs["item_id"].astype(str).map(
        lambda i: catalog.get(i, catalog.get(str(i).split(".")[0], "N/A"))
    )

    # Save to file if requested
    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if str(out_p).endswith(".json"):
            top_recs.to_json(out_p, orient="records", indent=2)
        else:
            top_recs.to_csv(out_p, index=False)
        if not args.quiet:
            print(f"[Infer] Recommendations saved to: {out_p} ({len(top_recs)} rows)")

    # Print to console based on format
    if args.format == "json":
        print(top_recs.to_json(orient="records", indent=2))
    elif args.format == "csv":
        print(top_recs.to_csv(index=False))
    else:
        # Table format
        table_output = format_recommendations_table(top_recs, catalog, show_score=True)
        print(table_output)


if __name__ == "__main__":
    main()
