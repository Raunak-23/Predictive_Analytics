"""Unit tests for leakage prevention, chronological splitting, and preprocessing."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import REG_FEATURES
from src.preprocessing import build_regression_preprocessor
from src.split_data import get_chronological_splits, get_classification_splits
from src.tuning import temporal_year_cv_split
from src.validate_data import validate_regression_data


class TestLeakageAndSplits(unittest.TestCase):
    """Tests temporal splitting, predictor whitelisting, and preprocessing fit scope."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create synthetic fixture with 8 years (2000-2007)
        records = []
        for i, yr in enumerate(range(2000, 2008)):
            for d in ["D1", "D2", "D3"]:
                records.append(
                    {
                        "row_id": f"R_{yr}_{d}",
                        "crop": "Rice",
                        "state": "State_X",
                        "district": d,
                        "season": "Kharif",
                        "year": yr,
                        "yield_t_ha": 2.0 + 0.1 * (yr - 2000),
                    }
                )
        self.df = pd.DataFrame(records)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_chronological_split_ordering(self):
        """Validates that train, val, test are strictly disjoint in time."""
        df_split, partitions = get_chronological_splits(self.df, save_manifest=False)

        train_years = partitions["train_years"]
        val_years = partitions["validation_years"]
        test_years = partitions["test_years"]

        self.assertEqual(len(test_years), 2)
        self.assertEqual(len(val_years), 2)
        self.assertTrue(max(train_years) < min(val_years))
        self.assertTrue(max(val_years) < min(test_years))

    def test_save_manifest_false_preserves_production_manifest(self):
        """Validates that running tests with save_manifest=False does not modify the production manifest."""
        from src.config import REG_SPLIT_MANIFEST
        if REG_SPLIT_MANIFEST.exists():
            original_bytes = REG_SPLIT_MANIFEST.read_bytes()
            # Run split with save_manifest=False on synthetic data
            get_chronological_splits(self.df, save_manifest=False)
            current_bytes = REG_SPLIT_MANIFEST.read_bytes()
            self.assertEqual(
                original_bytes,
                current_bytes,
                "Production manifest was modified despite save_manifest=False!",
            )

    def test_production_canonical_splits_exact_counts(self):
        """Validates that canonical Rice dataset produces exact 12592 / 1644 / 846 counts."""
        from src.config import RICE_CANONICAL_FILE
        if RICE_CANONICAL_FILE.exists():
            df_rice = pd.read_csv(RICE_CANONICAL_FILE)
            self.assertEqual(len(df_rice), 15082, "Canonical rice dataset must have 15,082 rows")
            _, partitions = get_chronological_splits(df_rice, save_manifest=False)
            counts = partitions["counts"]
            self.assertEqual(counts["train"], 12592)
            self.assertEqual(counts["validation"], 1644)
            self.assertEqual(counts["test"], 846)
            self.assertEqual(partitions["train_years"], list(range(1997, 2012)))
            self.assertEqual(partitions["validation_years"], [2012, 2013])
            self.assertEqual(partitions["test_years"], [2014, 2015])

    def test_production_and_area_rejected_by_whitelist(self):
        """Validates that production or area columns are caught as leakage violations."""
        leaky_df = self.df.copy()
        leaky_df["production"] = 5000.0
        csv_file = self.temp_path / "leaky.csv"
        leaky_df.to_csv(csv_file, index=False)
        with self.assertRaises(ValueError) as ctx:
            validate_regression_data(csv_file)
        self.assertIn("LEAKAGE VIOLATION", str(ctx.exception))

    def test_preprocessing_fit_only_on_training(self):
        """Ensures preprocessing fitted on train data does not know validation-only categories."""
        train_data = pd.DataFrame(
            {
                "state": ["S1", "S1", "S1"],
                "district": ["D1", "D2", "D1"],
                "season": ["Kharif", "Kharif", "Rabi"],
                "year": [2000, 2001, 2002],
            }
        )
        val_data = pd.DataFrame(
            {
                "state": ["S1", "S2"],  # S2 is unseen in training!
                "district": ["D1", "D_NEW"],  # D_NEW is unseen!
                "season": ["Kharif", "Summer"],
                "year": [2003, 2004],
            }
        )

        preprocessor = build_regression_preprocessor()
        preprocessor.fit(train_data[REG_FEATURES])

        # Check learned categories
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        learned_states = cat_encoder.categories_[0]
        self.assertIn("S1", learned_states)
        self.assertNotIn("S2", learned_states)

        # Transform on validation data should succeed without raising, encoding S2 as zeros
        val_transformed = preprocessor.transform(val_data[REG_FEATURES])
        self.assertFalse(np.isnan(val_transformed).any())

    def test_temporal_cv_expanding_window_integrity(self):
        """Ensures CV generator never leaks future year data into training folds."""
        cv_folds = list(temporal_year_cv_split(self.df, n_splits=3))
        self.assertEqual(len(cv_folds), 3)

        for tr_idx, va_idx in cv_folds:
            max_tr_yr = self.df.iloc[tr_idx]["year"].max()
            min_va_yr = self.df.iloc[va_idx]["year"].min()
            self.assertLess(
                max_tr_yr,
                min_va_yr,
                "Temporal leakage in CV: training fold contains years >= validation fold!",
            )

    def test_iid_gate_rejects_unjustified_random_split(self):
        """Classification split must fail when iid_justified is False."""
        dummy_cls = pd.DataFrame(
            {
                "row_id": [f"C_{i}" for i in range(20)],
                "N": range(20),
                "P": range(20),
                "K": range(20),
                "temperature": [25.0] * 20,
                "humidity": [60.0] * 20,
                "ph": [6.5] * 20,
                "rainfall": [100.0] * 20,
                "label": ["cropA", "cropB"] * 10,
            }
        )
        with self.assertRaises(ValueError) as ctx:
            get_classification_splits(dummy_cls, iid_justified=False, save_manifest=False)
        self.assertIn("iid_justified is False", str(ctx.exception))

    def test_classification_save_manifest_false_preserves_manifest(self):
        """Classification split with save_manifest=False must not alter disk manifest."""
        from src.config import CLS_SPLIT_MANIFEST
        dummy_cls = pd.DataFrame(
            {
                "row_id": [f"C_{i}" for i in range(20)],
                "N": range(20),
                "P": range(20),
                "K": range(20),
                "temperature": [25.0] * 20,
                "humidity": [60.0] * 20,
                "ph": [6.5] * 20,
                "rainfall": [100.0] * 20,
                "label": ["cropA", "cropB"] * 10,
            }
        )
        if CLS_SPLIT_MANIFEST.exists():
            orig_bytes = CLS_SPLIT_MANIFEST.read_bytes()
            get_classification_splits(dummy_cls, iid_justified=True, save_manifest=False)
            curr_bytes = CLS_SPLIT_MANIFEST.read_bytes()
            self.assertEqual(orig_bytes, curr_bytes)


if __name__ == "__main__":
    unittest.main()
