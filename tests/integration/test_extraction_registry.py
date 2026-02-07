"""
Integration tests for extraction → registry validation.

Validates that the EXPORTED data (CSV for R, used by figures) only contains
registry-validated methods. The raw DuckDB may contain all MLflow runs, but
the exported CSV must be filtered to only valid methods.

Run: pytest tests/integration/test_extraction_registry.py -v
"""

import pytest
import pandas as pd
from pathlib import Path

from src.data_io.registry import (
    get_valid_outlier_methods,
    get_valid_imputation_methods,
    get_valid_classifiers,
    EXPECTED_OUTLIER_COUNT,
    EXPECTED_CLASSIFIER_COUNT,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "r_data" / "essential_metrics.csv"
DB_PATH = PROJECT_ROOT / "data" / "public" / "foundation_plr_results.db"

pytestmark = pytest.mark.data


@pytest.fixture
def exported_df():
    """Load the exported CSV (filtered data used by R figures)."""
    if not CSV_PATH.exists():
        pytest.skip(f"CSV not found: {CSV_PATH}. Run: make analyze")
    return pd.read_csv(CSV_PATH)


@pytest.fixture
def db_connection():
    """Connect to extraction database."""
    import duckdb

    if not DB_PATH.exists():
        pytest.skip(f"Database not found: {DB_PATH}. Run: make extract")
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    yield conn
    conn.close()


class TestExportedDataUsesRegistry:
    """Verify exported CSV only includes registry-validated methods."""

    def test_all_outlier_methods_are_valid(self, exported_df):
        """Every exported outlier method must be in the registry."""
        extracted_methods = set(exported_df["outlier_method"].unique())
        valid_methods = set(get_valid_outlier_methods())

        invalid = extracted_methods - valid_methods
        assert not invalid, (
            f"Exported INVALID outlier methods: {invalid}. "
            f"These are not in the registry and must be excluded."
        )

    def test_all_imputation_methods_are_valid(self, exported_df):
        """Every exported imputation method must be in the registry."""
        extracted_methods = set(exported_df["imputation_method"].unique())
        valid_methods = set(get_valid_imputation_methods())

        invalid = extracted_methods - valid_methods
        assert not invalid, (
            f"Exported INVALID imputation methods: {invalid}. "
            f"These are not in the registry and must be excluded."
        )

    def test_outlier_count_matches_registry(self, exported_df):
        """Exported data should have exactly the registry-defined outlier methods."""
        count = exported_df["outlier_method"].nunique()

        assert count == EXPECTED_OUTLIER_COUNT, (
            f"Exported {count} outlier methods, registry defines {EXPECTED_OUTLIER_COUNT}."
        )

    def test_no_anomaly_method(self, exported_df):
        """'anomaly' is garbage and must NEVER appear."""
        assert "anomaly" not in exported_df["outlier_method"].values, (
            "'anomaly' found in export - this is INVALID!"
        )

    def test_no_exclude_method(self, exported_df):
        """'exclude' is garbage and must NEVER appear."""
        assert "exclude" not in exported_df["outlier_method"].values, (
            "'exclude' found in export - this is INVALID!"
        )

    def test_no_orig_variant_methods(self, exported_df):
        """'-orig-' variant methods are not in the registry."""
        orig_methods = [
            m for m in exported_df["outlier_method"].unique() if "-orig-" in m
        ]
        assert len(orig_methods) == 0, (
            f"Found '-orig-' variant methods: {orig_methods}. "
            f"These are not in the registry."
        )


class TestGroundTruthAUROC:
    """Verify ground truth AUROC matches expected value."""

    EXPECTED_GT_AUROC = 0.911
    TOLERANCE = 0.002

    def test_ground_truth_config_exists(self, exported_df):
        """Ground truth (pupil-gt + pupil-gt + CatBoost) must exist."""
        gt = exported_df[
            (exported_df["outlier_method"] == "pupil-gt")
            & (exported_df["imputation_method"] == "pupil-gt")
            & (exported_df["classifier"].str.upper() == "CATBOOST")
        ]
        assert len(gt) == 1, f"Expected 1 ground truth config, found {len(gt)}"

    def test_ground_truth_auroc_value(self, exported_df):
        """Ground truth AUROC must be 0.911 +/- 0.002."""
        gt = exported_df[
            (exported_df["outlier_method"] == "pupil-gt")
            & (exported_df["imputation_method"] == "pupil-gt")
            & (exported_df["classifier"].str.upper() == "CATBOOST")
        ]
        assert len(gt) > 0, "Ground truth config not found"

        auroc = gt["auroc"].iloc[0]
        assert abs(auroc - self.EXPECTED_GT_AUROC) < self.TOLERANCE, (
            f"Ground truth AUROC is {auroc:.4f}, expected {self.EXPECTED_GT_AUROC} +/- {self.TOLERANCE}. "
            f"DO NOT change expected value - find the extraction bug!"
        )


class TestClassifierValidation:
    """Verify classifier validation works correctly."""

    def test_all_classifiers_are_valid(self, exported_df):
        """Every exported classifier must be in the registry."""
        extracted_classifiers = {c.upper() for c in exported_df["classifier"].unique()}
        valid_classifiers = {c.upper() for c in get_valid_classifiers()}

        invalid = extracted_classifiers - valid_classifiers
        assert not invalid, (
            f"Exported INVALID classifiers: {invalid}. These are not in the registry."
        )

    def test_classifier_count_not_exceeds_registry(self, exported_df):
        """Cannot have more classifiers than registry defines."""
        count = exported_df["classifier"].nunique()

        assert count <= EXPECTED_CLASSIFIER_COUNT, (
            f"Exported {count} classifiers, registry defines {EXPECTED_CLASSIFIER_COUNT}."
        )
