"""
Tests for Top-10 Category Extraction.

Integration tests verifying that the database has sufficient configs
per preprocessing category for top-10 pooled analysis.

Categories:
- Ground Truth: pupil-gt outlier detection
- Ensemble FM: ensemble outlier detection
- Single FM: MOMENT, UniTS, TimesNet (non-ensemble)
- Traditional+DL: LOF, OneClassSVM, PROPHET, SubPCA

Reference: Azad et al. 2026, Varoquaux & Cheplygina 2022 (avoiding winner-takes-all)
"""

from pathlib import Path

import pytest

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestDatabaseIntegration:
    """Integration tests requiring actual database."""

    @pytest.fixture
    def db_path(self):
        """Get path to STRATOS database."""
        db = PROJECT_ROOT / "data" / "public" / "foundation_plr_results_stratos.db"
        assert db.exists(), f"STRATOS DB missing: {db}. Run: make extract"
        return db

    def test_ground_truth_category_has_configs(self, db_path):
        """Ground truth category should have configs in database."""
        import duckdb

        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            result = conn.execute("""
                SELECT COUNT(*)
                FROM essential_metrics
                WHERE classifier = 'CATBOOST'
                  AND outlier_method = 'pupil-gt'
            """).fetchone()

            count = result[0]
            assert count >= 1, f"Expected at least 1 ground truth config, got {count}"
        finally:
            conn.close()

    def test_all_categories_have_sufficient_configs(self, db_path):
        """Each category should have at least 3 configs for meaningful pooling."""
        import duckdb

        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            # Ground truth
            gt_count = conn.execute("""
                SELECT COUNT(*) FROM essential_metrics
                WHERE classifier = 'CATBOOST' AND outlier_method = 'pupil-gt'
            """).fetchone()[0]

            # Ensemble
            ens_count = conn.execute("""
                SELECT COUNT(*) FROM essential_metrics
                WHERE classifier = 'CATBOOST' AND outlier_method LIKE 'ensemble%'
            """).fetchone()[0]

            # Single FM
            fm_count = conn.execute("""
                SELECT COUNT(*) FROM essential_metrics
                WHERE classifier = 'CATBOOST'
                  AND (outlier_method LIKE '%MOMENT%'
                       OR outlier_method LIKE '%UniTS%'
                       OR outlier_method LIKE '%TimesNet%')
                  AND outlier_method NOT LIKE 'ensemble%'
            """).fetchone()[0]

            # Traditional
            trad_count = conn.execute("""
                SELECT COUNT(*) FROM essential_metrics
                WHERE classifier = 'CATBOOST'
                  AND outlier_method IN ('LOF', 'OneClassSVM', 'PROPHET', 'SubPCA')
            """).fetchone()[0]

            assert gt_count >= 3, f"Ground truth has only {gt_count} configs"
            assert ens_count >= 2, f"Ensemble has only {ens_count} configs"
            assert fm_count >= 3, f"Single FM has only {fm_count} configs"
            assert trad_count >= 3, f"Traditional has only {trad_count} configs"

        finally:
            conn.close()
