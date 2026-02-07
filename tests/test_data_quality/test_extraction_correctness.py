"""
Extraction Correctness Tests

Validate that the extracted DuckDB matches known ground truth values.
These tests ensure metric accuracy after extraction refactoring.

Ground truth values from results-ground-truth.json and known MLflow runs.
"""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
pytestmark = pytest.mark.data


def _get_duckdb_connection():
    """Get DuckDB connection if database exists, else skip."""
    db_paths = [
        PROJECT_ROOT / "data" / "public" / "foundation_plr_results.db",
        PROJECT_ROOT / "data" / "foundation_plr_results.db",
    ]
    db_path = None
    for p in db_paths:
        if p.exists():
            db_path = p
            break

    if db_path is None:
        pytest.skip("DuckDB not found (run extraction first)")

    import duckdb

    return duckdb.connect(str(db_path), read_only=True)


@pytest.fixture
def conn():
    """DuckDB connection fixture."""
    connection = _get_duckdb_connection()
    yield connection
    connection.close()


class TestGroundTruthAUROC:
    """Validate that ground truth AUROC matches expected values."""

    def test_ground_truth_auroc_matches(self, conn):
        """pupil-gt + pupil-gt + CatBoost (handcrafted) AUROC should be ~0.911."""
        result = conn.execute(
            """
            SELECT MAX(auroc) FROM essential_metrics
            WHERE outlier_method = 'pupil-gt'
              AND imputation_method = 'pupil-gt'
              AND classifier = 'CatBoost'
              AND featurization NOT LIKE '%embedding%'
            """
        ).fetchone()

        assert result is not None, "Ground truth config not found in essential_metrics"
        auroc = result[0]
        assert auroc is not None, "Ground truth AUROC is NULL"
        assert abs(auroc - 0.911) < 0.005, (
            f"Ground truth AUROC = {auroc}, expected ~0.911 (±0.005)"
        )

    def test_best_ensemble_auroc_matches(self, conn):
        """Ensemble + CSDI + CatBoost AUROC should be ~0.913."""
        result = conn.execute(
            """
            SELECT auroc FROM essential_metrics
            WHERE outlier_method LIKE 'ensemble%'
              AND imputation_method = 'CSDI'
              AND classifier = 'CatBoost'
            """
        ).fetchone()

        if result is None:
            pytest.skip("Best ensemble config not found (may not be extracted yet)")

        auroc = result[0]
        assert auroc is not None, "Best ensemble AUROC is NULL"
        assert abs(auroc - 0.913) < 0.005, (
            f"Best ensemble AUROC = {auroc}, expected ~0.913 (±0.005)"
        )


class TestConfigCounts:
    """Validate expected number of configurations."""

    def test_no_duplicate_configs(self, conn):
        """Each (outlier, imputation, classifier, featurization) should be unique."""
        result = conn.execute(
            """
            SELECT outlier_method, imputation_method, classifier, featurization,
                   COUNT(*) as cnt
            FROM essential_metrics
            GROUP BY outlier_method, imputation_method, classifier, featurization
            HAVING cnt > 1
            """
        ).fetchall()

        assert len(result) == 0, f"Found {len(result)} duplicate configs: " + str(
            [(r[0], r[1], r[2], r[3], r[4]) for r in result[:5]]
        )

    def test_has_reasonable_config_count(self, conn):
        """Should have a substantial number of configs."""
        result = conn.execute("SELECT COUNT(*) FROM essential_metrics").fetchone()

        count = result[0]
        assert count >= 50, f"Only {count} configs in essential_metrics, expected >= 50"


class TestSTRATOSMetricsPopulated:
    """Validate that STRATOS metrics are not NULL for most configs."""

    def test_stratos_metrics_not_null(self, conn):
        """Calibration slope, O:E ratio, net benefit should be populated."""
        # Allow some NULLs (e.g., configs with too few samples)
        # but most should be populated
        result = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                COUNT(calibration_slope) as has_slope,
                COUNT(o_e_ratio) as has_oe,
                COUNT(net_benefit_5pct) as has_nb5,
                COUNT(net_benefit_10pct) as has_nb10
            FROM essential_metrics
            """
        ).fetchone()

        total, has_slope, has_oe, has_nb5, has_nb10 = result

        if total == 0:
            pytest.skip("No configs in essential_metrics")

        # At least 80% should have calibration metrics
        assert has_slope / total >= 0.8, (
            f"Only {has_slope}/{total} configs have calibration_slope"
        )
        assert has_oe / total >= 0.8, f"Only {has_oe}/{total} configs have o_e_ratio"
        assert has_nb5 / total >= 0.8, (
            f"Only {has_nb5}/{total} configs have net_benefit_5pct"
        )

    def test_calibration_slope_range(self, conn):
        """Calibration slopes should be in reasonable range for non-degenerate models."""
        result = conn.execute(
            """
            SELECT MIN(calibration_slope), MAX(calibration_slope)
            FROM essential_metrics
            WHERE calibration_slope IS NOT NULL
              AND auroc > 0.55
            """
        ).fetchone()

        if result[0] is None:
            pytest.skip("No calibration slopes found for non-degenerate models")

        min_slope, max_slope = result
        # Degenerate models (AUROC~0.5) may have slope=0.0, excluded above
        assert min_slope >= 0.1, f"Min calibration slope {min_slope} < 0.1 (suspicious)"
        # Severely miscalibrated models (e.g., XGBoost + some pipelines) can have slopes ~20
        assert max_slope <= 25.0, (
            f"Max calibration slope {max_slope} > 25.0 (suspicious)"
        )

    def test_net_benefit_range(self, conn):
        """Net benefit values should be in reasonable range (-0.5 to 1.0)."""
        result = conn.execute(
            """
            SELECT MIN(net_benefit_5pct), MAX(net_benefit_5pct)
            FROM essential_metrics
            WHERE net_benefit_5pct IS NOT NULL
            """
        ).fetchone()

        if result[0] is None:
            pytest.skip("No net benefit values found")

        min_nb, max_nb = result
        assert min_nb >= -0.5, f"Min NB@5% = {min_nb}, expected >= -0.5"
        assert max_nb <= 1.0, f"Max NB@5% = {max_nb}, expected <= 1.0"

    def test_net_benefit_15pct_exists(self, conn):
        """net_benefit_15pct column should exist and be populated."""
        result = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'essential_metrics' AND column_name = 'net_benefit_15pct'"
        ).fetchall()

        assert len(result) > 0, (
            "net_benefit_15pct column missing from essential_metrics"
        )

        # Check it has values
        result = conn.execute(
            "SELECT COUNT(net_benefit_15pct) FROM essential_metrics WHERE net_benefit_15pct IS NOT NULL"
        ).fetchone()
        assert result[0] > 0, "net_benefit_15pct column exists but all values are NULL"


class TestNewTableData:
    """Validate data in the new extraction tables."""

    def test_retention_metrics_have_data(self, conn):
        """retention_metrics table should have data if it exists."""
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = {t[0] for t in tables}

        if "retention_metrics" not in table_names:
            pytest.fail("retention_metrics table does not exist")

        result = conn.execute("SELECT COUNT(*) FROM retention_metrics").fetchone()
        assert result[0] > 0, "retention_metrics table is empty"

    def test_cohort_metrics_have_data(self, conn):
        """cohort_metrics table should have data if it exists."""
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = {t[0] for t in tables}

        if "cohort_metrics" not in table_names:
            pytest.fail("cohort_metrics table does not exist")

        result = conn.execute("SELECT COUNT(*) FROM cohort_metrics").fetchone()
        assert result[0] > 0, "cohort_metrics table is empty"

    def test_distribution_stats_have_data(self, conn):
        """distribution_stats table should have data if it exists."""
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = {t[0] for t in tables}

        if "distribution_stats" not in table_names:
            pytest.fail("distribution_stats table does not exist")

        result = conn.execute("SELECT COUNT(*) FROM distribution_stats").fetchone()
        assert result[0] > 0, "distribution_stats table is empty"

    def test_dca_curves_have_data(self, conn):
        """dca_curves table should have data."""
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = {t[0] for t in tables}

        if "dca_curves" not in table_names:
            pytest.fail("dca_curves table does not exist")

        result = conn.execute("SELECT COUNT(*) FROM dca_curves").fetchone()
        assert result[0] > 0, "dca_curves table is empty"
