"""Unit tests for data integrity, validation rules, and schema checks."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.validate_data import validate_classification_data, validate_regression_data


class TestDataIntegrity(unittest.TestCase):
    """Tests schema validation, duplicate detection, and range boundaries using synthetic fixtures."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Valid synthetic regression fixture (>= 7 years)
        years = list(range(2000, 2010))
        records = []
        for i, yr in enumerate(years):
            records.append(
                {
                    "row_id": f"RICE_{i+1:04d}",
                    "crop": "Rice",
                    "state": "State_A",
                    "district": f"District_{i}",
                    "season": "Kharif",
                    "year": yr,
                    "yield_t_ha": 2.5 + 0.1 * i,
                }
            )
        self.valid_reg_df = pd.DataFrame(records)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_regression_fixture_passes(self):
        """A properly formatted synthetic fixture should pass validation."""
        csv_file = self.temp_path / "valid_rice.csv"
        self.valid_reg_df.to_csv(csv_file, index=False)
        report = validate_regression_data(csv_file)
        self.assertEqual(report["status"], "PASSED")

    def test_missing_required_columns_raises_error(self):
        """Missing canonical column should raise ValueError."""
        bad_df = self.valid_reg_df.drop(columns=["district"])
        csv_file = self.temp_path / "bad_cols.csv"
        bad_df.to_csv(csv_file, index=False)
        with self.assertRaises(ValueError):
            validate_regression_data(csv_file)

    def test_duplicate_row_id_raises_error(self):
        """Duplicate row_id must fail loudly."""
        bad_df = self.valid_reg_df.copy()
        bad_df.loc[1, "row_id"] = bad_df.loc[0, "row_id"]
        csv_file = self.temp_path / "dup_id.csv"
        bad_df.to_csv(csv_file, index=False)
        with self.assertRaises(ValueError):
            validate_regression_data(csv_file)

    def test_duplicate_agricultural_keys_raises_error(self):
        """Duplicate (state, district, season, year) must fail loudly."""
        bad_df = self.valid_reg_df.copy()
        bad_df.loc[1, ["state", "district", "season", "year"]] = bad_df.loc[
            0, ["state", "district", "season", "year"]
        ]
        csv_file = self.temp_path / "dup_keys.csv"
        bad_df.to_csv(csv_file, index=False)
        with self.assertRaises(ValueError):
            validate_regression_data(csv_file)

    def test_missing_or_invalid_target_raises_error(self):
        """NaN or negative yield values must fail validation."""
        # NaN target
        bad_df1 = self.valid_reg_df.copy()
        bad_df1.loc[2, "yield_t_ha"] = np.nan
        f1 = self.temp_path / "nan_yield.csv"
        bad_df1.to_csv(f1, index=False)
        with self.assertRaises(ValueError):
            validate_regression_data(f1)

        # Negative target
        bad_df2 = self.valid_reg_df.copy()
        bad_df2.loc[2, "yield_t_ha"] = -1.5
        f2 = self.temp_path / "neg_yield.csv"
        bad_df2.to_csv(f2, index=False)
        with self.assertRaises(ValueError):
            validate_regression_data(f2)

    def test_insufficient_years_raises_error(self):
        """Less than 7 harvest years must be rejected."""
        short_df = self.valid_reg_df[self.valid_reg_df["year"] < 2005].copy()
        csv_file = self.temp_path / "short_years.csv"
        short_df.to_csv(csv_file, index=False)
        with self.assertRaises(ValueError):
            validate_regression_data(csv_file)

    def test_classification_duplicate_feature_vectors_raises_error(self):
        """Duplicate feature vectors in D3 classification must fail loudly."""
        features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        cls_data = []
        for i in range(25):
            cls_data.append(
                {
                    "row_id": f"C_{i:03d}",
                    "N": 50 + i,
                    "P": 50,
                    "K": 50,
                    "temperature": 25.0,
                    "humidity": 60.0,
                    "ph": 6.5,
                    "rainfall": 100.0,
                    "label": "rice" if i % 2 == 0 else "maize",
                }
            )
        cls_df = pd.DataFrame(cls_data)
        # Introduce duplicate feature vector
        cls_df.loc[1, features] = cls_df.loc[0, features]
        csv_file = self.temp_path / "dup_cls.csv"
        cls_df.to_csv(csv_file, index=False)
        with self.assertRaises(ValueError):
            validate_classification_data(csv_file)


if __name__ == "__main__":
    unittest.main()
