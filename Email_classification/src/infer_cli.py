"""Load a saved email-classification pipeline and infer from the command line.

Examples (run from ``Email_classification/``)::

    python -m src.infer_cli --subject "Invoice question" --body "Please send a copy."
    python -m src.infer_cli --text "subject: Password reset\nbody: Please help" --json

This script only loads the persisted pipeline and predicts. It does not train,
modify artifacts, call an external API, or generate/send a reply.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_DIR / "outputs" / "models" / "business_intent_multinomial_nb.joblib"


def _build_text(subject: str | None, body: str | None, text: str | None) -> str:
    if text is not None:
        if not text.strip():
            raise ValueError("--text must not be empty")
        return text
    subject = subject or ""
    body = body or ""
    if not subject.strip() and not body.strip():
        raise ValueError("provide --text, or at least one of --subject/--body")
    return f"subject: {subject.strip()}\nbody: {body.strip()}" if subject.strip() else body.strip()


def _confidence_metadata(pipeline: Any, text: str) -> dict[str, Any]:
    if hasattr(pipeline, "predict_proba"):
        probabilities = np.asarray(pipeline.predict_proba([text])[0], dtype=float)
        order = np.argsort(probabilities)[::-1]
        return {
            "signal_type": "probability",
            "confidence": float(probabilities[order[0]]),
            "margin": float(probabilities[order[0]] - probabilities[order[1]]) if len(order) > 1 else None,
        }
    if hasattr(pipeline, "decision_function"):
        scores = np.asarray(pipeline.decision_function([text]))
        if scores.ndim == 1:
            return {"signal_type": "decision_score", "confidence": float(abs(scores[0])),
                    "margin": float(abs(scores[0]))}
        order = np.argsort(scores[0])[::-1]
        return {"signal_type": "decision_score", "confidence": float(scores[0, order[0]]),
                "margin": float(scores[0, order[0]] - scores[0, order[1]])}
    return {"signal_type": "unavailable", "confidence": None, "margin": None}


def infer(pipeline: Any, text: str) -> dict[str, Any]:
    predicted = pipeline.predict([text])[0]
    return {"predicted_class": str(predicted), "input_text": text, **_confidence_metadata(pipeline, text)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Infer an email intent using a saved pipeline.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Already formatted email text.")
    source.add_argument("--subject", help="Email subject; combine with --body.")
    parser.add_argument("--body", help="Email body; may be used without --subject.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help=f"Saved joblib pipeline (default: {DEFAULT_MODEL}).")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="Emit machine-readable JSON instead of summary text.")
    args = parser.parse_args(argv)

    # argparse's mutually exclusive group cannot combine --text with --body,
    # which is intentional: --text is the complete input form.
    if args.text is not None and args.body is not None:
        parser.error("--body cannot be combined with --text")
    try:
        text = _build_text(args.subject, args.body, args.text)
        model_path = args.model if args.model.is_absolute() else Path.cwd() / args.model
        if not model_path.exists():
            parser.error(f"model file not found: {model_path}")
        pipeline = joblib.load(model_path)
        result = infer(pipeline, text)
    except ValueError as exc:
        parser.error(str(exc))

    if args.as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Signal: {result['signal_type']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Margin: {result['margin']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
