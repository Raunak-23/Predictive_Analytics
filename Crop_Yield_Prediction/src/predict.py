"""CLI tool for offline validated inference using trusted model bundles.

Usage:
    python -m src.predict --model models/regression/selected_bundle.joblib --input user_input.json
"""

import argparse
import json
import sys
from pathlib import Path
import joblib

try:
    from src.inference import predict_validated
    from src.utils import setup_logger
except ImportError:
    from inference import predict_validated
    from utils import setup_logger

logger = setup_logger("predict_cli")


def main():
    parser = argparse.ArgumentParser(description="Run validated inference with a model bundle.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trusted serialized model bundle (.joblib)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to JSON file containing record(s) to predict, or raw JSON string.",
    )
    args = parser.parse_args()

    # Load bundle
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error("Model bundle file not found: %s", model_path)
        sys.exit(1)

    try:
        bundle = joblib.load(model_path)
    except Exception as exc:
        logger.error("Failed to load model bundle: %s", exc)
        sys.exit(1)

    # Load input records
    input_str = args.input.strip()
    try:
        if Path(input_str).exists():
            with open(input_str, "r", encoding="utf-8") as f:
                records = json.load(f)
        else:
            records = json.loads(input_str)
    except Exception as exc:
        logger.error("Failed to parse input JSON: %s", exc)
        sys.exit(1)

    try:
        predictions = predict_validated(bundle, records)
    except Exception as exc:
        logger.error("Inference validation REJECTED input: %s", exc)
        sys.exit(2)

    # Format output
    task = bundle.get("task", "unknown")
    unit = "t/ha" if task == "regression" else "crop_label"
    results = [
        {"prediction": float(p) if task == "regression" else str(p), "unit": unit}
        for p in predictions
    ]

    print(json.dumps({"status": "SUCCESS", "task": task, "results": results}, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
