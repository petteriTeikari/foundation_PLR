"""
Tests for Report Metric Verification.

Ensures all reported AUROC values and metrics match the DuckDB source of truth.
Prevents hallucinated values in reports.

Run with: uv run python -m pytest tests/test_report_metrics.py -v
"""

from pathlib import Path

import duckdb
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "public" / "foundation_plr_results_stratos.db"

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not DB_PATH.exists(),
        reason="STRATOS database not available",
    ),
]

# Expected values (verified 2026-01-29)
EXPECTED = {
    "gt_auroc": 0.9110,
    "best_auroc": 0.9130,
    "handcrafted_mean": 0.8304,
    "embedding_mean": 0.7040,
    "epv_handcrafted": 7.0,
    "n_events": 56,
    "n_features": 8,
}


class TestDatabaseValues:
    """Verify expected values match database."""

    @pytest.fixture
    def conn(self):
        """Create database connection."""
        assert DB_PATH.exists(), f"STRATOS DB missing: {DB_PATH}. Run: make extract"
        return duckdb.connect(str(DB_PATH), read_only=True)

    def test_gt_auroc_in_expected_range(self, conn):
        """Ground truth AUROC matches expected value."""
        result = conn.execute(
            """
            SELECT auroc FROM essential_metrics
            WHERE outlier_method = 'pupil-gt'
              AND imputation_method = 'pupil-gt'
              AND classifier = 'CATBOOST'
              AND featurization LIKE 'simple%'
            """
        ).fetchone()

        assert result is not None, "Ground truth config not found"
        gt_auroc = result[0]
        assert abs(gt_auroc - EXPECTED["gt_auroc"]) < 0.001, (
            f"GT AUROC {gt_auroc:.4f} != expected {EXPECTED['gt_auroc']:.4f}"
        )

    def test_best_auroc_in_expected_range(self, conn):
        """Best AUROC matches expected value."""
        result = conn.execute(
            """
            SELECT MAX(auroc) FROM essential_metrics
            WHERE classifier = 'CATBOOST'
              AND featurization LIKE 'simple%'
            """
        ).fetchone()

        assert result is not None, "No CatBoost configs found"
        best_auroc = result[0]
        assert abs(best_auroc - EXPECTED["best_auroc"]) < 0.001, (
            f"Best AUROC {best_auroc:.4f} != expected {EXPECTED['best_auroc']:.4f}"
        )

    def test_handcrafted_mean_in_expected_range(self, conn):
        """Handcrafted features mean AUROC matches expected value."""
        result = conn.execute(
            """
            SELECT AVG(auroc) FROM essential_metrics
            WHERE featurization LIKE 'simple%'
            """
        ).fetchone()

        assert result is not None
        mean_auroc = result[0]
        assert abs(mean_auroc - EXPECTED["handcrafted_mean"]) < 0.01, (
            f"Handcrafted mean {mean_auroc:.4f} != expected {EXPECTED['handcrafted_mean']:.4f}"
        )

    def test_embedding_mean_in_expected_range(self, conn):
        """MOMENT embedding mean AUROC matches expected value."""
        result = conn.execute(
            """
            SELECT AVG(auroc) FROM essential_metrics
            WHERE featurization = 'MOMENT-embedding'
            """
        ).fetchone()

        assert result is not None
        mean_auroc = result[0]
        assert abs(mean_auroc - EXPECTED["embedding_mean"]) < 0.01, (
            f"Embedding mean {mean_auroc:.4f} != expected {EXPECTED['embedding_mean']:.4f}"
        )


class TestEPVCalculation:
    """Tests for Events Per Variable calculation."""

    def test_epv_calculation_correct(self):
        """EPV = n_events / n_features = 56 / 8 = 7.0."""
        epv = EXPECTED["n_events"] / EXPECTED["n_features"]
        assert abs(epv - EXPECTED["epv_handcrafted"]) < 0.01, (
            f"EPV {epv:.1f} != expected {EXPECTED['epv_handcrafted']:.1f}"
        )

    def test_epv_uses_exact_features_not_range(self):
        """EPV should be exactly 7.0, not a range like 4.7-14."""
        # This is a reminder test - EPV is deterministic
        assert EXPECTED["n_features"] == 8, "Should use exactly 8 features"
        assert EXPECTED["n_events"] == 56, "Should have exactly 56 events"
