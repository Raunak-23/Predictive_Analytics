"""
Backward-compatible wrapper for src.run_d1_online_retail.
Executes the automated end-to-end recommender pipeline for Dataset D1 (UCI Online Retail).
"""

from src.run_d1_online_retail import *

if __name__ == "__main__":
    from src.run_d1_online_retail import run_d1_pipeline
    run_d1_pipeline()
