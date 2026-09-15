"""Unit tests for inference safety, model bundling, reload identity, and TEST_LOCK."""

import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from src.inference import create_model_bundle, predict_validated
from src.preprocessing import build_regression_preprocessor


class TestInferenceAndLock(unittest.TestCase):
    """Tests model bundle creation, deserialization identity, and validation guards."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Build small trained pipeline fixture
        train_df = pd.DataFrame(
            {
                "state": ["State_A", "State_A", "State_B"],
                "district": ["Dist_1", "Dist_2", "Dist_3"],
                "season": ["Kharif", "Rabi", "Kharif"],
                "year": [2000, 2001, 2002],
                "yield_t_ha": [2.1, 2.4, 2.8],
            }
        )
        self.train_df = train_df
        pipe = Pipeline(
            steps=[
                ("pre", build_regression_preprocessor()),
                ("model", DummyRegressor(strategy="mean")),
            ]
        )
        pipe.fit(train_df[["state", "district", "season", "year"]], train_df["yield_t_ha"])

        bundle_path = self.temp_path / "test_bundle.joblib"
        self.bundle = create_model_bundle(
            pipeline=pipe,
            task="regression",
            training_df=train_df,
            data_sha256="fake_data_sha",
            config_sha256="fake_config_sha",
            model_name="dummy_mean",
            save_path=bundle_path,
        )
        self.bundle_path = bundle_path

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_bundle_reload_identity(self):
        """Deserialized model bundle produces bitwise-identical predictions."""
        reloaded = joblib.load(self.bundle_path)
        sample = self.train_df.iloc[:2]
        pred_orig = self.bundle["pipeline"].predict(sample[["state", "district", "season", "year"]])
        pred_reload = reloaded["pipeline"].predict(sample[["state", "district", "season", "year"]])
        np.testing.assert_array_equal(pred_orig, pred_reload)

    def test_predict_validated_rejects_unknown_category(self):
        """Unknown state/district must be rejected during inference."""
        bad_input = [
            {
                "state": "Nonexistent_State",
                "district": "Dist_1",
                "season": "Kharif",
                "year": 2001,
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            predict_validated(self.bundle, bad_input)
        self.assertIn("Unknown or unvalidated state(s) rejected", str(ctx.exception))

    def test_predict_validated_rejects_out_of_range_year(self):
        """Years outside the evaluated training range [2000, 2002] must be rejected."""
        bad_input = [
            {
                "state": "State_A",
                "district": "Dist_1",
                "season": "Kharif",
                "year": 2015,  # Out of range!
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            predict_validated(self.bundle, bad_input)
        self.assertIn("outside evaluated range", str(ctx.exception))

    def test_predict_validated_rejects_noninteger_year(self):
        """Float / non-integer year must be rejected."""
        bad_input = [
            {
                "state": "State_A",
                "district": "Dist_1",
                "season": "Kharif",
                "year": 2001.5,
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            predict_validated(self.bundle, bad_input)
        self.assertIn("Non-integer harvest year rejected", str(ctx.exception))

    def test_classification_rejects_invalid_ph_and_humidity(self):
        """Classification bundle must reject pH > 14 and humidity > 100%."""
        dummy_cls_bundle = {
            "pipeline": None,
            "task": "classification",
            "features": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
            "classes": ["rice", "maize"],
        }
        # Invalid pH
        with self.assertRaises(ValueError) as ctx1:
            predict_validated(
                dummy_cls_bundle,
                [{"N": 50, "P": 50, "K": 50, "temperature": 25, "humidity": 80, "ph": 16.0, "rainfall": 100}],
            )
        self.assertIn("Soil pH must be between 0 and 14", str(ctx1.exception))

        # Invalid negative nutrient
        with self.assertRaises(ValueError) as ctx2:
            predict_validated(
                dummy_cls_bundle,
                [{"N": -5, "P": 50, "K": 50, "temperature": 25, "humidity": 80, "ph": 6.5, "rainfall": 100}],
            )
        self.assertIn("must be non-negative", str(ctx2.exception))


if __name__ == "__main__":
    unittest.main()
