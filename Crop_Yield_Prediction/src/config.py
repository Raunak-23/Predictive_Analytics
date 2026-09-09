"""Central configuration module for Experiment 08: Agricultural Predictive Analytics."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory: resolved relative to this file to ensure portability
BASE_DIR = Path(__file__).resolve().parent.parent

# Load local environment variables (.env) if present
load_dotenv(BASE_DIR / ".env")

# --- Directory Paths ---
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_REG_DIR = RESULTS_DIR / "regression"
RESULTS_CLS_DIR = RESULTS_DIR / "classification"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
PROVENANCE_DIR = ARTIFACTS_DIR / "provenance"
DATA_VAL_DIR = ARTIFACTS_DIR / "data_validation"
MANIFESTS_DIR = ARTIFACTS_DIR / "manifests"
TUNING_DIR = ARTIFACTS_DIR / "tuning"
ACCEPTANCE_DIR = ARTIFACTS_DIR / "acceptance"

MODELS_DIR = BASE_DIR / "models"
MODELS_REG_DIR = MODELS_DIR / "regression"
MODELS_CLS_DIR = MODELS_DIR / "classification"

FIGURES_DIR = BASE_DIR / "figures"
FIGURES_REG_DIR = FIGURES_DIR / "regression"
FIGURES_CLS_DIR = FIGURES_DIR / "classification"

NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# --- Canonical File Paths ---
D1_RAW_FILE = DATA_RAW_DIR / "d1_government_crop_statistics.csv"
D3_RAW_FILE = DATA_RAW_DIR / "Crop_recommendation.csv"
RICE_CANONICAL_FILE = DATA_PROCESSED_DIR / "rice_canonical.csv"
CROP_LABELS_CANONICAL_FILE = DATA_PROCESSED_DIR / "crop_labels_canonical.csv"

# --- Manifest and Lock Artifacts ---
REG_SPLIT_MANIFEST = MANIFESTS_DIR / "regression_split_manifest.csv"
CLS_SPLIT_MANIFEST = MANIFESTS_DIR / "classification_split_manifest.csv"
TEST_LOCK_FILE = ARTIFACTS_DIR / "TEST_LOCK"
MODEL_REGISTRY_FILE = ARTIFACTS_DIR / "model_registry.json"
SELECTION_INITIAL_FILE = ARTIFACTS_DIR / "selection_initial.json"
SELECTION_FINAL_FILE = ARTIFACTS_DIR / "selection.json"
ACCEPTANCE_FILE = ACCEPTANCE_DIR / "acceptance.json"
CONFIG_JSON_FILE = ARTIFACTS_DIR / "config.json"
CONFIG_SHA256_FILE = ARTIFACTS_DIR / "config_sha256.txt"
REG_CV_FOLDS_FILE = TUNING_DIR / "regression_cv_folds.csv"
REG_OVERFITTING_FILE = RESULTS_REG_DIR / "overfitting_diagnostics.csv"
CLS_OVERFITTING_FILE = RESULTS_CLS_DIR / "overfitting_diagnostics.csv"
SOURCE_ANOMALY_AUDIT_FILE = RESULTS_REG_DIR / "source_anomaly_audit.csv"
SOURCE_ANOMALY_SENSITIVITY_FILE = RESULTS_REG_DIR / "source_anomaly_sensitivity.csv"

# --- API Configuration ---
# Canonical environment variable is GOV_API_KEY
DEFAULT_API_URL = "https://api.data.gov.in/resource/35be999b-0208-4354-b557-f6ca9a5355de"
GOV_API_URL = os.getenv("GOV_API_URL", DEFAULT_API_URL)
API_PAGE_LIMIT = 10000
API_MAX_RETRIES = 3
API_RETRY_BACKOFF_FACTOR = 1.5
API_TIMEOUT_SECONDS = 30

# --- Deterministic Random Seed ---
RANDOM_SEED = 42

# --- Feature Whitelists and Targets ---
# MANDATORY LEAKAGE SAFEGUARD: Whitelist only. Never use df.drop(columns=[target]).
# production and area are explicitly EXCLUDED to prevent target reconstruction leakage.
REG_CATEGORICAL_FEATURES = ["state", "district", "season"]
REG_NUMERIC_FEATURES = ["year"]
REG_FEATURES = REG_CATEGORICAL_FEATURES + REG_NUMERIC_FEATURES
REG_TARGET = "yield_t_ha"
REG_REQUIRED_COLUMNS = ["row_id", "crop"] + REG_FEATURES + [REG_TARGET]

# Classification Track (D3 Crop Recommendation)
CLS_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
CLS_TARGET = "label"
CLS_REQUIRED_COLUMNS = ["row_id"] + CLS_FEATURES + [CLS_TARGET]

# Minimum required unique years for chronological split (Section 15 & Manual)
MIN_REQUIRED_YEARS = 7

# Classification IID Policy (Section 16: default false, set with explicit rationale)
IID_JUSTIFIED_DEFAULT = False
IID_JUSTIFICATION_NOTE = (
    "Supplied D3 dataset consists of synthetic/augmented tabular observations (100 balanced samples "
    "per class across 22 classes) without temporal stamps or grouped farm/field IDs. When iid_justified "
    "is set to true, a reproducible stratified holdout split is executed."
)

# --- Hyperparameter Grids for Tuning ---
REG_TUNING_GRIDS = {
    "forest": {
        "model__n_estimators": [40, 60, 100],
        "model__max_depth": [8, 12, 16],
        "model__min_samples_leaf": [3, 5, 10],
    },
    "tree": {
        "model__max_depth": [4, 6, 8, 10],
        "model__min_samples_leaf": [5, 10, 20],
    },
    "ridge_trend": {
        "model__alpha": [
            0.01,
            0.03,
            0.1,
            0.2,
            0.3,
            0.5,
            0.7,
            1.0,
            1.5,
            2.0,
            3.0,
            5.0,
            10.0,
            20.0,
        ],
    },
}

CLS_TUNING_GRIDS = {
    "forest": {
        "model__n_estimators": [40, 60, 100],
        "model__max_depth": [8, 12, 16],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "logistic": {
        "model__C": [0.1, 1.0, 10.0],
    },
}


def get_pipeline_config() -> dict:
    """Returns the comprehensive, deterministic configuration dictionary for Experiment 08."""
    return {
        "random_seed": RANDOM_SEED,
        "regression_feature_whitelist": list(REG_FEATURES),
        "regression_target": REG_TARGET,
        "categorical_features": list(REG_CATEGORICAL_FEATURES),
        "numeric_features": list(REG_NUMERIC_FEATURES),
        "minimum_year_requirement": MIN_REQUIRED_YEARS,
        "split_configuration": {
            "strategy": "chronological_holdout",
            "train_years_range": [1997, 2011],
            "validation_years": [2012, 2013],
            "test_years": [2014, 2015],
            "min_required_years": MIN_REQUIRED_YEARS,
        },
        "temporal_cv_configuration": {
            "strategy": "forward_year_block_temporal_cv",
            "validation_years": [2007, 2008, 2009, 2010, 2011],
            "n_folds": 5,
            "strictly_forward": True,
            "max_train_year_strictly_less_than_min_val_year": True,
            "cv_partition_file": "artifacts/tuning/regression_cv_folds.csv",
        },
        "candidate_model_definitions": {
            "median": {
                "class": "DummyRegressor",
                "strategy": "median",
            },
            "ridge_trend": {
                "class": "Ridge",
                "alpha": 1.0,
                "solver": "lsqr",
                "random_state": RANDOM_SEED,
            },
            "tree": {
                "class": "DecisionTreeRegressor",
                "max_depth": 6,
                "min_samples_leaf": 10,
                "random_state": RANDOM_SEED,
            },
            "forest": {
                "class": "RandomForestRegressor",
                "n_estimators": 60,
                "max_depth": 12,
                "min_samples_leaf": 5,
                "n_jobs": 2,
                "random_state": RANDOM_SEED,
            },
        },
        "candidate_model_hyperparameters": {
            "ridge_alpha": 1.0,
            "tree_max_depth": 6,
            "tree_min_samples_leaf": 10,
            "forest_n_estimators": 60,
            "forest_max_depth": 12,
            "forest_min_samples_leaf": 5,
        },
        "regression_tuning_grids": REG_TUNING_GRIDS,
        "classification_configuration": {
            "features": list(CLS_FEATURES),
            "target": CLS_TARGET,
            "iid_justified_default": IID_JUSTIFIED_DEFAULT,
            "model_definitions": {
                "majority": {"class": "DummyClassifier", "strategy": "most_frequent"},
                "logistic": {"class": "LogisticRegression", "max_iter": 2000, "random_state": RANDOM_SEED},
                "forest": {"class": "RandomForestClassifier", "n_estimators": 60, "min_samples_leaf": 2, "n_jobs": 2, "random_state": RANDOM_SEED},
            },
        },
        "classification_tuning_grids": CLS_TUNING_GRIDS,
        "preprocessing_configuration": {
            "regression": {
                "categorical": {
                    "transformer": "OneHotEncoder",
                    "handle_unknown": "ignore",
                    "sparse_output": False,
                },
                "numeric": {
                    "imputer": "SimpleImputer(strategy='median')",
                    "scaler": "StandardScaler()",
                },
            },
            "classification": {
                "numeric": {
                    "imputer": "SimpleImputer(strategy='median')",
                    "scaler": "StandardScaler()",
                },
            },
        },
    }


def save_pipeline_config():
    """Serializes deterministic config to artifacts/config.json and writes artifacts/config_sha256.txt."""
    import hashlib
    import json
    config_dict = get_pipeline_config()
    CONFIG_JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(config_dict, sort_keys=True, indent=2) + "\n"
    raw_bytes = serialized.encode("utf-8")
    CONFIG_JSON_FILE.write_bytes(raw_bytes)
    config_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    CONFIG_SHA256_FILE.write_text(config_sha256 + "\n", encoding="utf-8")
    return CONFIG_JSON_FILE, CONFIG_SHA256_FILE, config_sha256


def get_current_config_sha256() -> str:
    """Returns the single source-of-truth SHA-256 hash of the pipeline configuration."""
    if CONFIG_SHA256_FILE.exists():
        h = CONFIG_SHA256_FILE.read_text(encoding="utf-8").strip()
        if h:
            return h
    _, _, h = save_pipeline_config()
    return h

