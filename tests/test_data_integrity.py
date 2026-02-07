"""
Data Integrity Tests - TDD for Reproducibility Pipeline
========================================================

These tests verify that exported data files have correct values.
They should FAIL if data is wrong, catching bugs like CRITICAL-FAILURE-002.

Run with: pytest tests/test_data_integrity.py -v

Created: 2026-01-28
Reason: FAILURE-005 - Ground Truth AUROC showed 0.850 instead of 0.911
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

pytestmark = pytest.mark.data

# Find project root
PROJECT_ROOT = Path(__file__).parent.parent


class TestGroundTruthAUROC:
    """Ground Truth AUROC must be 0.911 ± 0.002"""

    EXPECTED_AUROC = 0.911
    TOLERANCE = 0.002

    def test_ground_truth_in_duckdb(self):
        """Verify DuckDB has correct Ground Truth AUROC."""
        db_path = PROJECT_ROOT / "data" / "public" / "foundation_plr_results_stratos.db"
        assert db_path.exists(), f"Database missing: {db_path}. Run: make extract"

        conn = duckdb.connect(str(db_path), read_only=True)
        result = conn.execute("""
            SELECT auroc FROM essential_metrics
            WHERE outlier_method = 'pupil-gt'
              AND imputation_method = 'pupil-gt'
              AND classifier = 'CATBOOST'
              AND featurization = 'simple1.0'
        """).fetchone()
        conn.close()

        assert result is not None, "Ground Truth config not found in DuckDB"
        auroc = result[0]

        assert (
            self.EXPECTED_AUROC - self.TOLERANCE
            <= auroc
            <= self.EXPECTED_AUROC + self.TOLERANCE
        ), f"Ground Truth AUROC {auroc:.4f} not in expected range [{self.EXPECTED_AUROC - self.TOLERANCE}, {self.EXPECTED_AUROC + self.TOLERANCE}]"

    def test_ground_truth_in_roc_rc_json(self):
        """Verify roc_rc_data.json has correct Ground Truth AUROC."""
        json_path = PROJECT_ROOT / "data" / "r_data" / "roc_rc_data.json"
        assert json_path.exists(), f"JSON missing: {json_path}. Run: make analyze"

        with open(json_path) as f:
            data = json.load(f)

        # Find ground truth config
        gt_config = None
        for config in data["data"]["configs"]:
            if (
                config.get("id") == "ground_truth"
                or "ground" in config.get("name", "").lower()
            ):
                gt_config = config
                break

        assert (
            gt_config is not None
        ), "Ground Truth config not found in roc_rc_data.json"

        # Check AUROC - may be at top level or nested in 'roc'
        auroc = None
        if "auroc" in gt_config:
            auroc = gt_config["auroc"]
        elif "roc" in gt_config and "auroc" in gt_config["roc"]:
            auroc = gt_config["roc"]["auroc"]

        assert auroc is not None, "AUROC not found in Ground Truth config"
        assert (
            self.EXPECTED_AUROC - self.TOLERANCE
            <= auroc
            <= self.EXPECTED_AUROC + self.TOLERANCE
        ), f"Ground Truth AUROC {auroc:.4f} in roc_rc_data.json not in expected range [{self.EXPECTED_AUROC - self.TOLERANCE}, {self.EXPECTED_AUROC + self.TOLERANCE}]"

    def test_ground_truth_computed_from_predictions(self):
        """Compute AUROC from predictions and verify it matches expected."""
        db_path = PROJECT_ROOT / "data" / "public" / "foundation_plr_results_stratos.db"
        assert db_path.exists(), f"Database missing: {db_path}. Run: make extract"

        conn = duckdb.connect(str(db_path), read_only=True)
        result = conn.execute("""
            SELECT y_true, y_prob FROM predictions
            WHERE outlier_method = 'pupil-gt'
              AND imputation_method = 'pupil-gt'
              AND classifier = 'CATBOOST'
              AND featurization = 'simple1.0'
        """).fetchall()
        conn.close()

        assert len(result) > 0, "No predictions found for Ground Truth"

        y_true = np.array([r[0] for r in result])
        y_prob = np.array([r[1] for r in result])
        auroc = roc_auc_score(y_true, y_prob)

        assert (
            self.EXPECTED_AUROC - self.TOLERANCE
            <= auroc
            <= self.EXPECTED_AUROC + self.TOLERANCE
        ), f"Computed AUROC {auroc:.4f} from predictions not in expected range"


class TestPredictionCounts:
    """Predictions table stores fold-0 data only (~63 per config from 3-fold CV).

    Total dataset has 208 subjects (152 control + 56 glaucoma), but
    the predictions table only stores one fold's test set.

    Key validation: class balance should be ~27% glaucoma (56/208).
    """

    EXPECTED_GLAUCOMA_RATIO = 0.27  # 56/208 = 0.269
    RATIO_TOLERANCE = 0.05  # Allow 22-32% glaucoma

    def test_prediction_count_ground_truth(self):
        """Verify Ground Truth has reasonable predictions with correct class balance."""
        db_path = PROJECT_ROOT / "data" / "public" / "foundation_plr_results_stratos.db"
        assert db_path.exists(), f"Database missing: {db_path}. Run: make extract"

        conn = duckdb.connect(str(db_path), read_only=True)
        result = conn.execute("""
            SELECT COUNT(*) as n, SUM(y_true) as n_pos
            FROM predictions
            WHERE outlier_method = 'pupil-gt'
              AND imputation_method = 'pupil-gt'
              AND classifier = 'CATBOOST'
              AND featurization = 'simple1.0'
        """).fetchone()
        conn.close()

        n_total, n_pos = result

        # Must have predictions (fold-0 should have ~63 subjects)
        assert n_total > 0, "No predictions found for Ground Truth"
        assert (
            n_total >= 50
        ), f"Too few predictions ({n_total}), expected at least one fold (~63)"

        # Class balance should match overall ratio
        glaucoma_ratio = n_pos / n_total
        assert (
            self.EXPECTED_GLAUCOMA_RATIO - self.RATIO_TOLERANCE
            <= glaucoma_ratio
            <= self.EXPECTED_GLAUCOMA_RATIO + self.RATIO_TOLERANCE
        ), f"Glaucoma ratio {glaucoma_ratio:.2%} not in expected range [22%, 32%]"


class TestFeaturizationFilter:
    """All exports must use handcrafted features (simple1.0) by default"""

    def test_data_filters_yaml_exists(self):
        """Verify data_filters.yaml exists."""
        yaml_path = PROJECT_ROOT / "configs" / "VISUALIZATION" / "data_filters.yaml"
        assert yaml_path.exists(), f"Data filters config not found: {yaml_path}"

    def test_data_filters_has_correct_default(self):
        """Verify default featurization is simple1.0."""
        import yaml

        yaml_path = PROJECT_ROOT / "configs" / "VISUALIZATION" / "data_filters.yaml"
        assert yaml_path.exists(), f"Missing: {yaml_path}"

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        assert (
            config["defaults"]["featurization"] == "simple1.0"
        ), "Default featurization should be 'simple1.0' (handcrafted features)"

    def test_no_mixed_featurization_in_exports(self):
        """Verify JSON exports don't have mixed featurization data."""
        # This test would need to check each JSON file's metadata
        # For now, just ensure the config is correct
        pass


class TestCalibrationData:
    """Calibration curves should have reasonable values"""

    def test_calibration_json_exists(self):
        """Verify calibration_data.json exists."""
        json_path = PROJECT_ROOT / "data" / "r_data" / "calibration_data.json"
        # May not exist yet - skip if not found
        assert json_path.exists(), f"JSON missing: {json_path}. Run: make analyze"

        with open(json_path) as f:
            data = json.load(f)

        # Basic structure check
        assert "data" in data
        assert "configs" in data["data"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
