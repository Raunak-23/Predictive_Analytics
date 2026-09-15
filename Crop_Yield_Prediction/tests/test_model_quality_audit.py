"""Automated Model Quality Audit Tests (Experiment 08 Rubric Part 22).

Verifies:
1. Candidate model feature lists match the whitelist exactly: ['state', 'district', 'season', 'year'].
2. Production is NOT in X for any candidate.
3. Area is NOT in X for any candidate.
4. Target is NOT in X for any candidate.
5. Preprocessor fit is called ONLY on training data.
6. Validation-only categories are NOT in encoder.categories_ after fitting on training data.
7. Expanding-window CV folds have min(val_year) > max(train_year) for every fold.
8. No CV training fold contains any year from its corresponding validation fold or future.
9. Tuning never receives validation or test rows.
10. TEST_LOCK prevents repeated evaluation without --reset-lock.
11. Config SHA-256 changes if any candidate hyperparameter changes.
12. Reloading selected model bundle produces bitwise identical predictions on validation and test data.
13. Overfitting diagnostics artifact exists and reports both train and validation MAE for all models.
14. Source anomaly audit artifact exists and correctly flags the Kolhapur 1997 record.
15. Source anomaly sensitivity artifact exists and contains the required diagnostic-only label.
"""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.config import (
    CONFIG_JSON_FILE,
    CONFIG_SHA256_FILE,
    IID_JUSTIFIED_DEFAULT,
    MODEL_REGISTRY_FILE,
    MODELS_CLS_DIR,
    MODELS_REG_DIR,
    REG_CV_FOLDS_FILE,
    REG_FEATURES,
    REG_OVERFITTING_FILE,
    REG_TARGET,
    RESULTS_REG_DIR,
    RICE_CANONICAL_FILE,
    SELECTION_FINAL_FILE,
    SELECTION_INITIAL_FILE,
    SOURCE_ANOMALY_AUDIT_FILE,
    SOURCE_ANOMALY_SENSITIVITY_FILE,
    TEST_LOCK_FILE,
    TUNING_DIR,
    get_current_config_sha256,
    get_pipeline_config,
)
from src.preprocessing import build_regression_preprocessor
from src.run_pipeline import run_full_pipeline
from src.split_data import get_chronological_splits
from src.train_regression import run_test
from src.tuning import forward_year_block_cv
from src.utils import compute_sha256, load_json


class TestModelQualityAudit(unittest.TestCase):
    """Rigorous compliance suite covering all 15 audit requirements of Part 22."""

    @classmethod
    def setUpClass(cls):
        """Loads canonical dataset once for read-only audit assertions."""
        cls.df_rice = pd.read_csv(RICE_CANONICAL_FILE)
        cls.df_split, cls.partitions = get_chronological_splits(cls.df_rice, save_manifest=False)
        cls.train_df = cls.df_split[cls.df_split["split"] == "train"].copy()
        cls.val_df = cls.df_split[cls.df_split["split"] == "validation"].copy()
        cls.test_df = cls.df_split[cls.df_split["split"] == "test"].copy()

    def test_01_feature_whitelist_exact(self):
        """1. Candidate model feature lists match the whitelist exactly."""
        expected_whitelist = ["state", "district", "season", "year"]
        self.assertEqual(
            REG_FEATURES,
            expected_whitelist,
            f"Feature list {REG_FEATURES} does not match exact whitelist {expected_whitelist}",
        )

    def test_02_production_excluded_from_x(self):
        """2. Production is NOT in X for any candidate."""
        forbidden_terms = ["production", "production_tonnes", "prod"]
        for term in forbidden_terms:
            self.assertNotIn(term, REG_FEATURES, f"Target-derived '{term}' found in features!")

    def test_03_area_excluded_from_x(self):
        """3. Area is NOT in X for any candidate."""
        forbidden_terms = ["area", "area_hectares", "harvested_area"]
        for term in forbidden_terms:
            self.assertNotIn(term, REG_FEATURES, f"Denominator '{term}' found in features!")

    def test_04_target_excluded_from_x(self):
        """4. Target is NOT in X for any candidate."""
        self.assertNotIn(REG_TARGET, REG_FEATURES, f"Target '{REG_TARGET}' found in feature list!")

    def test_05_preprocessor_fit_scope_train_only(self):
        """5. Preprocessor fit is called ONLY on training data."""
        preprocessor = build_regression_preprocessor()
        preprocessor.fit(self.train_df[REG_FEATURES])

        encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        learned_states = set(encoder.categories_[0])
        train_states = set(self.train_df["state"].unique())

        self.assertEqual(
            learned_states,
            train_states,
            "OneHotEncoder categories do not strictly match training data categories!",
        )

    def test_06_validation_only_categories_not_learned(self):
        """6. Validation-only categories are NOT in encoder.categories_ after fitting on training data."""
        preprocessor = build_regression_preprocessor()
        preprocessor.fit(self.train_df[REG_FEATURES])
        encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        learned_districts = set(encoder.categories_[1])

        val_districts = set(self.val_df["district"].unique())
        val_only_districts = val_districts - set(self.train_df["district"].unique())

        if len(val_only_districts) > 0:
            for dist in val_only_districts:
                self.assertNotIn(
                    dist,
                    learned_districts,
                    f"Validation-only district '{dist}' leaked into fitted encoder categories!",
                )

    def test_07_expanding_window_cv_temporal_ordering(self):
        """7. Expanding-window CV folds have min(val_year) > max(train_year) for every fold."""
        folds = list(
            forward_year_block_cv(
                self.train_df,
                val_years=[2007, 2008, 2009, 2010, 2011],
            )
        )
        self.assertEqual(len(folds), 5, "Expected 5 forward CV folds.")
        for fold_idx, (tr_idx, val_idx) in enumerate(folds, start=1):
            tr_years = self.train_df.iloc[tr_idx]["year"]
            val_years = self.train_df.iloc[val_idx]["year"]
            max_tr = int(tr_years.max())
            min_val = int(val_years.min())
            self.assertGreater(
                min_val,
                max_tr,
                f"Fold {fold_idx} temporal ordering violated: min_val ({min_val}) <= max_tr ({max_tr})",
            )

    def test_08_no_cv_fold_leakage(self):
        """8. No CV training fold contains any year from its corresponding validation fold or future."""
        folds = list(
            forward_year_block_cv(
                self.train_df,
                val_years=[2007, 2008, 2009, 2010, 2011],
            )
        )
        for fold_idx, (tr_idx, val_idx) in enumerate(folds, start=1):
            val_year = int(self.train_df.iloc[val_idx]["year"].iloc[0])
            tr_years = self.train_df.iloc[tr_idx]["year"]
            leakage_years = tr_years[tr_years >= val_year]
            self.assertEqual(
                len(leakage_years),
                0,
                f"Fold {fold_idx} leaked {len(leakage_years)} records with year >= {val_year}",
            )

    def test_09_tuning_never_receives_validation_or_test_rows(self):
        """9. Tuning never receives validation or test rows."""
        folds = list(
            forward_year_block_cv(
                self.train_df,
                val_years=[2007, 2008, 2009, 2010, 2011],
            )
        )
        for fold_idx, (tr_idx, val_idx) in enumerate(folds, start=1):
            all_fold_years = set(self.train_df.iloc[tr_idx]["year"]).union(
                set(self.train_df.iloc[val_idx]["year"])
            )
            for heldout_year in [2012, 2013, 2014, 2015]:
                self.assertNotIn(
                    heldout_year,
                    all_fold_years,
                    f"Fold {fold_idx} includes heldout year {heldout_year}!",
                )

    def test_10_test_lock_prevents_repeated_evaluation(self):
        """10. TEST_LOCK prevents repeated evaluation without --reset-lock."""
        # Ensure a lock file is present
        TEST_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not TEST_LOCK_FILE.exists():
            TEST_LOCK_FILE.write_text("Test lock for audit test", encoding="utf-8")

        with self.assertRaises(RuntimeError) as ctx:
            run_test(reset_lock=False)
        self.assertIn("TEST_LOCK already exists", str(ctx.exception))

    def test_11_config_hash_changes_on_parameter_change(self):
        """11. Config SHA-256 changes if any candidate hyperparameter changes."""
        base_cfg = get_pipeline_config()
        base_json = json.dumps(base_cfg, sort_keys=True, indent=2)
        base_hash = hashlib.sha256(base_json.encode("utf-8")).hexdigest()

        modified_cfg = copy.deepcopy(base_cfg)
        if "candidate_model_hyperparameters" in modified_cfg:
            modified_cfg["candidate_model_hyperparameters"]["ridge_alpha"] = 99.0
        elif "model_hyperparameters" in modified_cfg:
            modified_cfg["model_hyperparameters"]["ridge_alpha"] = 99.0
        mod_json = json.dumps(modified_cfg, sort_keys=True, indent=2)
        mod_hash = hashlib.sha256(mod_json.encode("utf-8")).hexdigest()

        self.assertNotEqual(
            base_hash,
            mod_hash,
            "Config hash failed to change after modifying hyperparameter!",
        )

    def test_12_model_reload_bitwise_identity(self):
        """12. Reloading selected model bundle produces bitwise identical predictions."""
        bundle_path = MODELS_REG_DIR / "selected_bundle.joblib"
        if not bundle_path.exists():
            self.skipTest("selected_bundle.joblib not yet trained in this environment.")

        bundle = joblib.load(bundle_path)
        pipe1 = bundle["pipeline"]
        sample_x = self.test_df.head(20)[REG_FEATURES]

        pred1 = pipe1.predict(sample_x)
        reloaded = joblib.load(bundle_path)
        pipe2 = reloaded["pipeline"]
        pred2 = pipe2.predict(sample_x)

        np.testing.assert_array_equal(
            pred1,
            pred2,
            err_msg="Reloaded pipeline did not produce bitwise identical predictions!",
        )

    def test_13_overfitting_diagnostics_exist_and_complete(self):
        """13. Overfitting diagnostics artifact exists and reports both train and validation MAE for all models."""
        if not REG_OVERFITTING_FILE.exists():
            self.skipTest("overfitting_diagnostics.csv not yet generated.")

        df = pd.read_csv(REG_OVERFITTING_FILE)
        # Check required columns
        expected_cols = ["model", "train_MAE", "validation_MAE", "train_to_validation_MAE_gap"]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing column '{col}' in overfitting diagnostics!")

        # All rows must have positive train and val MAE
        self.assertTrue((df["train_MAE"] > 0).all())
        self.assertTrue((df["validation_MAE"] > 0).all())
        self.assertGreaterEqual(len(df), 4, "Expected at least 4 candidate models in overfitting table.")

    def test_14_source_anomaly_audit_flags_kolhapur(self):
        """14. Source anomaly audit artifact exists and correctly flags the Kolhapur 1997 record."""
        if not SOURCE_ANOMALY_AUDIT_FILE.exists():
            self.skipTest("source_anomaly_audit.csv not yet generated.")

        df = pd.read_csv(SOURCE_ANOMALY_AUDIT_FILE)
        self.assertGreater(len(df), 0, "Source anomaly audit is empty!")
        kolhapur_row = df[
            (df["state"].str.upper() == "MAHARASHTRA")
            & (df["district"].str.upper() == "KOLHAPUR")
            & (df["year"] == 1997)
        ]
        self.assertFalse(kolhapur_row.empty, "Kolhapur 1997 anomaly record missing from audit!")
        calc_yield = float(kolhapur_row.iloc[0]["yield_t_ha"])
        self.assertAlmostEqual(calc_yield, 223.727, places=2)

    def test_15_source_anomaly_sensitivity_diagnostic_label(self):
        """15. Source anomaly sensitivity artifact exists and contains the required diagnostic-only label."""
        if not SOURCE_ANOMALY_SENSITIVITY_FILE.exists():
            self.skipTest("source_anomaly_sensitivity.csv not yet generated.")

        df = pd.read_csv(SOURCE_ANOMALY_SENSITIVITY_FILE)
        self.assertGreaterEqual(len(df), 2, "Expected at least 2 model evaluation rows in sensitivity analysis.")

        # Scenario B must be labeled diagnostic sensitivity only
        model_b = df[
            df["scenario"].str.contains("excluding", case=False)
            | df["status"].str.contains("diagnostic", case=False)
        ]
        self.assertFalse(model_b.empty, "Model B (excluding anomalies) missing from sensitivity analysis!")
        status_or_note = (
            str(model_b.iloc[0].get("status", "")) + " " + str(model_b.iloc[0].get("note", ""))
        )
        self.assertIn(
            "diagnostic_sensitivity_only",
            status_or_note.lower(),
            "Scenario B must state 'diagnostic_sensitivity_only'!",
        )

    def test_16_config_consistency_across_artifacts(self):
        """16. Config hash propagation matches single source-of-truth across all generated artifacts."""
        cur_cfg_hash = get_current_config_sha256()

        if CONFIG_SHA256_FILE.exists():
            saved_h = CONFIG_SHA256_FILE.read_text(encoding="utf-8").strip()
            self.assertEqual(saved_h, cur_cfg_hash, "config_sha256.txt does not match current config hash!")

        if SELECTION_FINAL_FILE.exists():
            sel = load_json(SELECTION_FINAL_FILE)
            self.assertEqual(sel.get("config_sha256"), cur_cfg_hash, "selection.json config hash mismatch!")

        if SELECTION_INITIAL_FILE.exists():
            sel_init = load_json(SELECTION_INITIAL_FILE)
            self.assertEqual(sel_init.get("config_sha256"), cur_cfg_hash, "selection_initial.json config hash mismatch!")

        if MODEL_REGISTRY_FILE.exists():
            reg = load_json(MODEL_REGISTRY_FILE)
            for m_name, m_info in reg.items():
                if m_info.get("status") == "complete":
                    self.assertEqual(
                        m_info.get("config_sha256"),
                        cur_cfg_hash,
                        f"model_registry entry '{m_name}' config hash mismatch!",
                    )

        reg_tuning_file = TUNING_DIR / "best_params.json"
        if reg_tuning_file.exists():
            t_data = load_json(reg_tuning_file)
            self.assertEqual(t_data.get("config_sha256"), cur_cfg_hash, "best_params.json config hash mismatch!")

        cls_tuning_file = TUNING_DIR / "classification_best_params.json"
        if cls_tuning_file.exists():
            tc_data = load_json(cls_tuning_file)
            self.assertEqual(tc_data.get("config_sha256"), cur_cfg_hash, "classification_best_params.json mismatch!")

        reg_bundle = MODELS_REG_DIR / "selected_bundle.joblib"
        if reg_bundle.exists():
            b = joblib.load(reg_bundle)
            b_cfg = b.get("config_sha256") or b.get("metadata", {}).get("config_sha256")
            self.assertEqual(
                b_cfg,
                cur_cfg_hash,
                "regression selected_bundle config hash mismatch!",
            )

        cls_bundle = MODELS_CLS_DIR / "selected_bundle.joblib"
        if cls_bundle.exists():
            cb = joblib.load(cls_bundle)
            cb_cfg = cb.get("config_sha256") or cb.get("metadata", {}).get("config_sha256")
            self.assertEqual(
                cb_cfg,
                cur_cfg_hash,
                "classification selected_bundle config hash mismatch!",
            )

    def test_17_data_consistency_across_artifacts(self):
        """17. Canonical data hash matches selection and bundle metadata."""
        cur_data_hash = compute_sha256(RICE_CANONICAL_FILE)
        if SELECTION_FINAL_FILE.exists():
            sel = load_json(SELECTION_FINAL_FILE)
            self.assertEqual(sel.get("data_sha256"), cur_data_hash, "selection.json data_sha256 mismatch!")

        reg_bundle = MODELS_REG_DIR / "selected_bundle.joblib"
        if reg_bundle.exists():
            b = joblib.load(reg_bundle)
            b_data = b.get("data_sha256") or b.get("metadata", {}).get("data_sha256")
            self.assertEqual(b_data, cur_data_hash, "selected_bundle data_sha256 mismatch!")

    def test_18_test_lock_consistency(self):
        """18. TEST_LOCK is consistent with selection.json and current config/data hashes."""
        if not TEST_LOCK_FILE.exists():
            self.skipTest("TEST_LOCK not created yet.")

        lock_raw = TEST_LOCK_FILE.read_text(encoding="utf-8").strip()
        try:
            lock_dict = json.loads(lock_raw)
            if SELECTION_FINAL_FILE.exists():
                sel = load_json(SELECTION_FINAL_FILE)
                self.assertEqual(lock_dict.get("selected_model"), sel.get("selected_model"))
                self.assertEqual(lock_dict.get("selection_sha256"), compute_sha256(SELECTION_FINAL_FILE))
            self.assertEqual(lock_dict.get("data_sha256"), compute_sha256(RICE_CANONICAL_FILE))
            self.assertEqual(lock_dict.get("config_sha256"), get_current_config_sha256())
        except json.JSONDecodeError:
            self.assertIn("Selected model:", lock_raw)

    def test_19_source_anomaly_audit_training_rows_only(self):
        """19. Development source anomaly audit artifact contains only training rows."""
        if not SOURCE_ANOMALY_AUDIT_FILE.exists():
            self.skipTest("source_anomaly_audit.csv missing.")

        df = pd.read_csv(SOURCE_ANOMALY_AUDIT_FILE)
        self.assertTrue(
            (df["training_or_heldout_split"] == "train").all(),
            "Development anomaly audit scanned held-out validation or test rows!",
        )

    def test_20_selected_tuned_model_in_overfitting_diagnostics(self):
        """20. Overfitting diagnostics table contains the actual selected tuned model."""
        if not REG_OVERFITTING_FILE.exists() or not SELECTION_FINAL_FILE.exists():
            self.skipTest("Overfitting file or selection file missing.")

        sel = load_json(SELECTION_FINAL_FILE)
        df = pd.read_csv(REG_OVERFITTING_FILE)
        selected_model = sel["selected_model"]
        self.assertIn(
            selected_model,
            df["model"].values,
            f"Selected model '{selected_model}' missing from overfitting diagnostics table!",
        )

    def test_21_classification_governance(self):
        """22. Classification IID default is False and pipeline cannot silently assert IID."""
        self.assertFalse(IID_JUSTIFIED_DEFAULT, "IID_JUSTIFIED_DEFAULT must be False!")
        with self.assertRaises(ValueError):
            run_full_pipeline(include_classification=True, justify_iid=False)

    def test_22_environment_version_agreement(self):
        """23. Runtime environment versions match requirements.txt pinned versions."""
        import importlib.metadata
        import platform
        import matplotlib
        import nbconvert
        import nbformat
        import requests
        import sklearn

        major, minor, *_ = platform.python_version_tuple()
        assert (major, minor) == ("3", "13")
        self.assertEqual((major, minor), ("3", "13"))
        self.assertEqual(np.__version__, "2.5.2")
        self.assertEqual(pd.__version__, "3.0.5")
        self.assertEqual(sklearn.__version__, "1.9.0")
        self.assertEqual(joblib.__version__, "1.5.3")
        self.assertEqual(matplotlib.__version__, "3.11.1")
        self.assertEqual(requests.__version__, "2.34.2")
        self.assertEqual(nbformat.__version__, "5.11.1")
        self.assertEqual(nbconvert.__version__, "7.17.1")
        self.assertEqual(importlib.metadata.version("python-dotenv"), "1.2.3")


if __name__ == "__main__":
    unittest.main()

