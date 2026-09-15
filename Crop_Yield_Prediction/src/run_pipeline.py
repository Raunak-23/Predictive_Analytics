"""Master pipeline runner for Experiment 08: Agricultural Predictive Analytics.

Executes all sequential pipeline stages:
1. Data Acquisition (if not already downloaded)
2. Data Preparation & Canonicalization
3. Strict Data Validation
4. Regression Development & Temporal CV Tuning
5. Regression Final Evaluation on Locked Test Set
6. Classification Pipeline Execution
7. Reproducibility & Acceptance Verification
"""

import argparse
import sys
from pathlib import Path

try:
    from src.acquire_d1 import acquire_d1
    from src.config import D1_RAW_FILE
    from src.prepare_data import prepare_classification_data, prepare_rice_data
    from src.train_classification import run_classification
    from src.train_regression import run_development, run_test
    from src.utils import setup_logger
    from src.validate_data import validate_classification_data, validate_regression_data
    from src.verify_reproducibility import verify_all
except ImportError:
    from acquire_d1 import acquire_d1
    from config import D1_RAW_FILE
    from prepare_data import prepare_classification_data, prepare_rice_data
    from train_classification import run_classification
    from train_regression import run_development, run_test
    from utils import setup_logger
    from validate_data import validate_classification_data, validate_regression_data
    from verify_reproducibility import verify_all

logger = setup_logger("run_pipeline")


def run_full_pipeline(
    reacquire: bool = False,
    advanced_depth: bool = False,
    include_classification: bool = False,
    justify_iid: bool = False,
    reset_lock: bool = False,
) -> None:
    """Executes all stages of the agricultural predictive analytics pipeline."""
    logger.info("==================================================")
    logger.info("STARTING EXPERIMENT 08 REPRODUCIBLE ML PIPELINE")
    logger.info("==================================================")

    if include_classification and not justify_iid:
        raise ValueError(
            "Classification requested via --include-classification, but --justify-iid was not provided.\n"
            "D3 classification requires an explicit --justify-iid flag with documented rationale."
        )

    # 1. Acquisition
    logger.info("\n--- STAGE 1: D1 DATA ACQUISITION ---")
    if not D1_RAW_FILE.exists() or reacquire:
        acquire_d1(force=reacquire)
    else:
        logger.info("Raw D1 file already present at %s. Skipping acquisition.", D1_RAW_FILE)

    # 2. Preparation
    logger.info("\n--- STAGE 2: DATA PREPARATION & CANONICALIZATION ---")
    prepare_rice_data()
    prepare_classification_data()

    # 3. Validation
    logger.info("\n--- STAGE 3: DATA VALIDATION & LEAKAGE CHECKS ---")
    validate_regression_data()
    validate_classification_data()

    # 4. Regression Development
    logger.info("\n--- STAGE 4: REGRESSION DEVELOPMENT & TUNING ---")
    run_development(advanced=advanced_depth, reset_lock=reset_lock)

    # 5. Regression Test Evaluation
    logger.info("\n--- STAGE 5: REGRESSION LOCKED TEST EVALUATION ---")
    run_test(reset_lock=reset_lock)

    # 6. Classification Pipeline (Optional Extension)
    if include_classification:
        logger.info("\n--- STAGE 6: CLASSIFICATION PIPELINE ---")
        run_classification(justify_iid=justify_iid)
    else:
        logger.info("\n--- STAGE 6: CLASSIFICATION PIPELINE (SKIPPED - not requested) ---")

    # 7. Acceptance Verification
    logger.info("\n--- STAGE 7: REPRODUCIBILITY & ACCEPTANCE VERIFICATION ---")
    report = verify_all()
    if report["acceptance_status"] != "PASSED":
        logger.error("Reproducibility verification failed!")
        sys.exit(1)

    logger.info("==================================================")
    logger.info("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    logger.info("==================================================")


def main():
    parser = argparse.ArgumentParser(description="Run complete Experiment 08 pipeline")
    parser.add_argument("--reacquire", action="store_true", help="Force re-download from API.")
    parser.add_argument("--advanced", action="store_true", help="Include depth ablation in regression.")
    parser.add_argument("--include-classification", action="store_true", help="Include classification track B.")
    parser.add_argument("--justify-iid", action="store_true", help="Explicitly justify IID assumption for classification.")
    parser.add_argument("--reset-lock", action="store_true", help="Reset TEST_LOCK for clean pipeline rebuild.")
    args = parser.parse_args()

    try:
        run_full_pipeline(
            reacquire=args.reacquire,
            advanced_depth=args.advanced,
            include_classification=args.include_classification,
            justify_iid=args.justify_iid,
            reset_lock=args.reset_lock,
        )
    except Exception as exc:
        logger.error("Pipeline run failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
