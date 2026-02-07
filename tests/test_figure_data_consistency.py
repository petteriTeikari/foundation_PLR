"""
Figure Data Consistency Tests
=============================

Validates that JSON exports for figures have correct:
1. Pipeline definitions (outlier_method, imputation_method)
2. Category names matching the 5 standard categories
3. AUROC values matching database computations
4. Consistent data across all figure exports

These tests prevent issues like showing wrong pipeline data with renamed labels.

Run with: pytest tests/test_figure_data_consistency.py -v

Created: 2026-02-02
Reason: User reported mismatch between figure labels and actual data
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


# Expected pipeline definitions for 5 standard categories
EXPECTED_PIPELINES = {
    "ground_truth": {
        "category_name": "Ground Truth",
        "outlier_method": "pupil-gt",
        "imputation_method": "pupil-gt",
        "expected_auroc": 0.911,
        "auroc_tolerance": 0.005,
    },
    "best_ensemble": {
        "category_name": "Ensemble FM",
        "outlier_method": "ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune",
        "imputation_method": "CSDI",
        "expected_auroc": 0.913,
        "auroc_tolerance": 0.005,
    },
    "best_single_fm": {
        "category_name": "Single-model FM",
        "outlier_method": "MOMENT-gt-finetune",
        "imputation_method": "SAITS",
        "expected_auroc": 0.913,
        "auroc_tolerance": 0.005,
    },
    "deep_learning": {
        "category_name": "Deep Learning",
        "outlier_method": "TimesNet-orig",
        "imputation_method": "TimesNet",
        "expected_auroc": 0.898,
        "auroc_tolerance": 0.005,
    },
    "traditional": {
        "category_name": "Traditional",
        "outlier_method": "LOF",
        "imputation_method": "SAITS",
        "expected_auroc": 0.861,
        "auroc_tolerance": 0.005,
    },
}


class TestROCRCDataConsistency:
    """Test roc_rc_data.json has correct pipelines and values."""

    @pytest.fixture
    def roc_rc_data(self):
        """Load ROC/RC JSON data."""
        json_path = PROJECT_ROOT / "data" / "r_data" / "roc_rc_data.json"
        if not json_path.exists():
            pytest.skip(f"JSON not found: {json_path}. Run: make analyze")
        with open(json_path) as f:
            return json.load(f)

    def test_has_all_5_categories(self, roc_rc_data):
        """Verify all 5 standard categories are present."""
        config_ids = {cfg["id"] for cfg in roc_rc_data["data"]["configs"]}
        expected_ids = set(EXPECTED_PIPELINES.keys())

        missing = expected_ids - config_ids
        assert not missing, f"Missing categories in roc_rc_data.json: {missing}"

    @pytest.mark.parametrize("config_id", EXPECTED_PIPELINES.keys())
    def test_category_name_correct(self, roc_rc_data, config_id):
        """Verify each config has correct category name."""
        expected = EXPECTED_PIPELINES[config_id]

        config = next(
            (c for c in roc_rc_data["data"]["configs"] if c["id"] == config_id),
            None,
        )
        assert config is not None, f"Config {config_id} not found"

        assert config["name"] == expected["category_name"], (
            f"Config {config_id} has name '{config['name']}' "
            f"but expected '{expected['category_name']}'"
        )

    @pytest.mark.parametrize("config_id", EXPECTED_PIPELINES.keys())
    def test_pipeline_definition_correct(self, roc_rc_data, config_id):
        """Verify each config has correct outlier and imputation methods."""
        expected = EXPECTED_PIPELINES[config_id]

        config = next(
            (c for c in roc_rc_data["data"]["configs"] if c["id"] == config_id),
            None,
        )
        assert config is not None, f"Config {config_id} not found"

        assert config["outlier_method"] == expected["outlier_method"], (
            f"Config {config_id} has outlier_method '{config['outlier_method']}' "
            f"but expected '{expected['outlier_method']}'"
        )

        assert config["imputation_method"] == expected["imputation_method"], (
            f"Config {config_id} has imputation_method '{config['imputation_method']}' "
            f"but expected '{expected['imputation_method']}'"
        )

    @pytest.mark.parametrize("config_id", EXPECTED_PIPELINES.keys())
    def test_auroc_in_expected_range(self, roc_rc_data, config_id):
        """Verify AUROC is in expected range for each config."""
        expected = EXPECTED_PIPELINES[config_id]

        config = next(
            (c for c in roc_rc_data["data"]["configs"] if c["id"] == config_id),
            None,
        )
        assert config is not None, f"Config {config_id} not found"

        auroc = config["roc"]["auroc"]
        low = expected["expected_auroc"] - expected["auroc_tolerance"]
        high = expected["expected_auroc"] + expected["auroc_tolerance"]

        assert low <= auroc <= high, (
            f"Config {config_id} ({expected['category_name']}) has AUROC {auroc:.4f} "
            f"but expected {expected['expected_auroc']:.3f} ± {expected['auroc_tolerance']}"
        )


class TestSelectiveClassificationDataConsistency:
    """Test selective_classification_data.json has correct pipelines and values."""

    @pytest.fixture
    def selective_data(self):
        """Load selective classification JSON data."""
        json_path = (
            PROJECT_ROOT / "data" / "r_data" / "selective_classification_data.json"
        )
        if not json_path.exists():
            pytest.skip(f"JSON not found: {json_path}. Run: make analyze")
        with open(json_path) as f:
            return json.load(f)

    def test_has_all_5_categories(self, selective_data):
        """Verify all 5 standard categories are present."""
        config_ids = {cfg["id"] for cfg in selective_data["data"]["configs"]}
        expected_ids = set(EXPECTED_PIPELINES.keys())

        missing = expected_ids - config_ids
        assert not missing, (
            f"Missing categories in selective_classification_data.json: {missing}"
        )

    @pytest.mark.parametrize("config_id", EXPECTED_PIPELINES.keys())
    def test_category_name_correct(self, selective_data, config_id):
        """Verify each config has correct category name."""
        expected = EXPECTED_PIPELINES[config_id]

        config = next(
            (c for c in selective_data["data"]["configs"] if c["id"] == config_id),
            None,
        )
        assert config is not None, f"Config {config_id} not found"

        assert config["name"] == expected["category_name"], (
            f"Config {config_id} has name '{config['name']}' "
            f"but expected '{expected['category_name']}'"
        )

    @pytest.mark.parametrize("config_id", EXPECTED_PIPELINES.keys())
    def test_baseline_auroc_in_expected_range(self, selective_data, config_id):
        """Verify baseline AUROC matches expected for each config."""
        expected = EXPECTED_PIPELINES[config_id]

        config = next(
            (c for c in selective_data["data"]["configs"] if c["id"] == config_id),
            None,
        )
        assert config is not None, f"Config {config_id} not found"

        auroc = config["baseline_metrics"]["auroc"]
        low = expected["expected_auroc"] - expected["auroc_tolerance"]
        high = expected["expected_auroc"] + expected["auroc_tolerance"]

        assert low <= auroc <= high, (
            f"Config {config_id} ({expected['category_name']}) has baseline AUROC {auroc:.4f} "
            f"but expected {expected['expected_auroc']:.3f} ± {expected['auroc_tolerance']}"
        )


class TestDatabaseAUROCMatch:
    """Verify JSON AUROC values match database computations."""

    @pytest.fixture
    def db_connection(self):
        """Get database connection."""
        db_path = PROJECT_ROOT / "data" / "public" / "foundation_plr_results.db"
        if not db_path.exists():
            pytest.skip(f"Database not found: {db_path}. Run: make extract")
        conn = duckdb.connect(str(db_path), read_only=True)
        yield conn
        conn.close()

    @pytest.mark.parametrize("config_id", EXPECTED_PIPELINES.keys())
    def test_json_auroc_matches_computed(self, db_connection, config_id):
        """Compute AUROC from database and verify it matches JSON."""
        expected = EXPECTED_PIPELINES[config_id]

        # Compute AUROC from predictions table
        query = """
            WITH dedup AS (
                SELECT subject_id, y_true, y_prob,
                       ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY mlflow_run_id) as rn
                FROM predictions
                WHERE outlier_method = ?
                  AND imputation_method = ?
                  AND classifier = 'CATBOOST'
                  AND featurization = 'simple1.0'
            )
            SELECT y_true, y_prob FROM dedup WHERE rn = 1
        """
        result = db_connection.execute(
            query,
            [expected["outlier_method"], expected["imputation_method"]],
        ).fetchall()

        assert len(result) > 0, f"No predictions found for {config_id}"

        y_true = np.array([r[0] for r in result])
        y_prob = np.array([r[1] for r in result])
        computed_auroc = roc_auc_score(y_true, y_prob)

        # Load JSON and compare
        json_path = PROJECT_ROOT / "data" / "r_data" / "roc_rc_data.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            config = next(
                (c for c in data["data"]["configs"] if c["id"] == config_id),
                None,
            )
            if config:
                json_auroc = config["roc"]["auroc"]
                assert abs(computed_auroc - json_auroc) < 0.001, (
                    f"Config {config_id}: Computed AUROC {computed_auroc:.4f} "
                    f"doesn't match JSON AUROC {json_auroc:.4f}"
                )


class TestCategoryColorConsistency:
    """Verify category colors are consistent across exports."""

    EXPECTED_COLORS = {
        "ground_truth": "--color-category-ground-truth",
        "best_ensemble": "--color-category-ensemble",
        "best_single_fm": "--color-category-foundation-model",
        "deep_learning": "--color-category-deep-learning",
        "traditional": "--color-category-traditional",
    }

    @pytest.fixture
    def roc_rc_data(self):
        """Load ROC/RC JSON data."""
        json_path = PROJECT_ROOT / "data" / "r_data" / "roc_rc_data.json"
        if not json_path.exists():
            pytest.skip(f"JSON not found: {json_path}. Run: make analyze")
        with open(json_path) as f:
            return json.load(f)

    @pytest.mark.parametrize("config_id", EXPECTED_COLORS.keys())
    def test_color_ref_correct(self, roc_rc_data, config_id):
        """Verify each config has correct color reference."""
        expected_color = self.EXPECTED_COLORS[config_id]

        config = next(
            (c for c in roc_rc_data["data"]["configs"] if c["id"] == config_id),
            None,
        )
        assert config is not None, f"Config {config_id} not found"

        assert config.get("color_ref") == expected_color, (
            f"Config {config_id} has color_ref '{config.get('color_ref')}' "
            f"but expected '{expected_color}'"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
