#!/usr/bin/env python3
"""
Train all three Lab-02 pipelines, write artifacts + figures.

Usage (from project root):
    python tools/train_all.py
    python tools/train_all.py --datasets breast heart
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import meddiag_common as M  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train all meddiag pipelines")
    parser.add_argument(
        "--datasets", nargs="+", default=list(M.CHOICES),
        choices=list(M.CHOICES),
        help="Which datasets to train (default: all)",
    )
    args = parser.parse_args()

    # import figure generator from the CLI module
    sys.path.insert(0, SRC)
    from meddiag_cli import generate_figures

    for ds in args.datasets:
        s = M.spec(ds)
        print("\n" + "=" * 70)
        print(f"TRAINING: {ds} — {s.display_name}")
        print("=" * 70)
        state = M.train_dataset(ds, verbose=True)
        paths = M.save_artifacts(ds, state, tag=ds)
        print(f"\n[OK] artifacts -> {paths.get('metadata')}")
        try:
            figs = generate_figures(ds, state, tag=ds)
            print(f"[OK] figures  -> {len(figs)} files under artifacts/figures/")
        except Exception as e:
            print(f"[!] figure generation failed: {e}")

        tm = state["test_metrics"]
        print(f"\nSUMMARY {ds}: best={state['best_name']}  thr={state['threshold']:.3f}  "
              f"sens={tm['sensitivity']:.4f}  spec={tm['specificity']:.4f}  "
              f"auc={tm['roc_auc']:.4f}  FN={tm['FN']} FP={tm['FP']}")

    print("\n" + "=" * 70)
    print("ALL DONE. Verify with:  python src/meddiag_cli.py info")
    print("=" * 70)


if __name__ == "__main__":
    main()
