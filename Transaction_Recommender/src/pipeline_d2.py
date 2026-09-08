"""
Backward-compatible wrapper for src.run_d2_advanced.
Executes the automated end-to-end recommender pipeline for Dataset D2 (Instacart Benchmark).
"""

from src.run_d2_advanced import *

if __name__ == "__main__":
    from src.run_d2_advanced import run_d2_pipeline
    run_d2_pipeline()
