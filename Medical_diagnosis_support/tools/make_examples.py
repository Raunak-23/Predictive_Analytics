#!/usr/bin/env python3
"""
Build ready-to-run sample inputs under examples/ for the CLI.

Creates, for each dataset:
  * {ds}_sample.json   - small unlabeled batch for `predict`
  * {ds}_sample.csv    - labeled batch for `evaluate`
  * {ds}_retrain.csv   - larger labeled slice for a quick `retrain` smoke test
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import meddiag_common as M  # noqa: E402

EX = M.EXAMPLES_DIR
os.makedirs(EX, exist_ok=True)


def main():
    for ds in M.CHOICES:
        s = M.spec(ds)
        df = M.load_dataset(ds)
        # stratified-ish head: take first positives and negatives
        pos_mask = df[s.target_col].isin(list(s.positive_raw_values))
        # raw values may be numeric for heart
        if s.key == "heart":
            pos_mask = df[s.target_col].astype(float) > 0
        elif s.key == "breast":
            pos_mask = df[s.target_col].astype(str).str.strip() == "M"
        elif s.key == "diabetes":
            pos_mask = df[s.target_col].astype(str).str.strip() == "Positive"

        pos = df[pos_mask].head(4)
        neg = df[~pos_mask].head(4)
        sample = pd.concat([pos, neg], axis=0).reset_index(drop=True)

        # labeled CSV for evaluate
        csv_path = os.path.join(EX, f"{ds}_sample.csv")
        sample.to_csv(csv_path, index=False)

        # unlabeled JSON for predict (drop target)
        feats = sample.drop(columns=[s.target_col], errors="ignore")
        # keep only known feature columns when present
        keep = [c for c in s.feature_cols if c in feats.columns]
        feats = feats[keep]
        json_path = os.path.join(EX, f"{ds}_sample.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(feats.to_dict(orient="records"), f, indent=2, default=str)

        # retrain slice (~60 rows mixed)
        retrain = pd.concat(
            [df[pos_mask].head(30), df[~pos_mask].head(30)], axis=0
        ).reset_index(drop=True)
        retrain_path = os.path.join(EX, f"{ds}_retrain.csv")
        retrain.to_csv(retrain_path, index=False)

        print(f"[OK] {ds}: {csv_path} ({len(sample)} rows), "
              f"{json_path}, {retrain_path} ({len(retrain)} rows)")


if __name__ == "__main__":
    main()
