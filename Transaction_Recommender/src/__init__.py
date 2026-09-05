"""
MDI3003 Experiment 07 Recommendation System Package.
"""

from .config import (
    ADVANCED_SEEDS,
    ARTIFACTS_DIR,
    BASE_DIR,
    DATA_DIR,
    DATASET_SCHEMAS,
    DEFAULT_RF_PARAMS,
    FIGURES_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    SEED,
    TUNING_PARAM_GRID,
)
from .io_utils import load_transactions
from .cleaning import clean_transactions
from .eda import (
    purchase_frequency_distribution,
    rfm_distributions,
    sparsity,
    top_n_items,
    txn_volume_over_time,
)
from .splitting import chronological_split, split_timeline_plot
from .candidates import build_candidate_universe, candidate_recall, sample_negatives
from .features import (
    build_feature_matrix,
    customer_features,
    feature_dictionary,
    item_features,
    pair_features,
)
from .models import (
    MatrixFactorizationRecommender,
    popularity_baseline,
    popularity_recommend,
    recommend_top_k,
    score_candidates,
    train_mf_baseline,
    train_random_forest,
    tune_random_forest,
)
from .evaluation import (
    average_precision_at_k,
    build_five_case_audit,
    evaluate_recommendations,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from .visualization import (
    catalog_coverage_plot,
    class_balance_plot,
    feature_importance_plot,
    metric_vs_k_plot,
    model_comparison_bar,
    score_distribution_plot,
)
from .artifacts import reload_and_verify, save_run_artifacts
