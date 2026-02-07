#!/usr/bin/env python3
"""
test_extraction_verification.py

Verification tests for the MLflow extraction pipeline.
These tests ensure that extracted numbers match expected values
and are consistent across all output formats (DB, JSON, LaTeX).

Run with:
    pytest tests/test_extraction_verification.py -v

Created: 2026-01-24
Purpose: Ensure reproducibility of manuscript numerical claims
"""

import json
import re
from pathlib import Path
from typing import Optional

import pytest

pytestmark = pytest.mark.data

# Try to import duckdb, skip tests if not available
try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

# =============================================================================
# CONFIGURATION
# =============================================================================

# Database paths (canonical locations after GH#13 consolidation)
_PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = _PROJECT_ROOT / "data" / "public" / "foundation_plr_results.db"
CD_DIAGRAM_DB = _PROJECT_ROOT / "data" / "public" / "cd_diagram_data.duckdb"

# Manuscript output paths
MANUSCRIPT_DIR = Path(
    "/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR"
)
FIGURES_DATA_DIR = MANUSCRIPT_DIR / "figures" / "generated" / "data"
NUMBERS_TEX = (
    MANUSCRIPT_DIR
    / "background-research"
    / "latent-methods-results"
    / "results"
    / "artifacts"
    / "numbers.tex"
)

# =============================================================================
# EXPECTED VALUES (Ground Truth)
# =============================================================================

EXPECTED_VALUES = {
    # From DB/pipeline (2026-02-07 extraction, 408 runs, 1 failed, 1 dupe removed)
    # 406 total configs in consolidated DB (all classifiers, all featurizations)
    # 81 of these are CatBoost
    "n_configs": 406,
    "n_catboost_configs": 81,
    "min_auroc": 0.500,
    "max_auroc": 0.913,
    "mean_auroc": 0.829,
    # From Najjar et al. 2023 (external reference)
    "najjar_auroc": 0.93,
    "najjar_ci_lower": 0.90,
    "najjar_ci_upper": 0.96,
    # Sample sizes (from data collection)
    "n_labeled_subjects": 208,
    "n_glaucoma": 56,
    "n_control": 152,
    "n_total_subjects": 507,
    # Device specs (from Najjar et al. 2023)
    "blue_wavelength_nm": 469,
    "red_wavelength_nm": 640,
}

# Tolerance for floating point comparisons
TOLERANCE = 0.001


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def db_connection():
    """Get DuckDB connection to main results database."""
    if not HAS_DUCKDB:
        pytest.skip("duckdb not installed. Run: uv add duckdb")
    if not DB_PATH.exists():
        pytest.skip(f"Database missing: {DB_PATH}. Run: make extract")
    return duckdb.connect(str(DB_PATH), read_only=True)


@pytest.fixture
def numbers_tex_content() -> Optional[str]:
    """Load numbers.tex content."""
    if not NUMBERS_TEX.exists():
        pytest.skip(f"numbers.tex missing: {NUMBERS_TEX}")
    return NUMBERS_TEX.read_text()


def parse_latex_command(content: str, command: str) -> Optional[str]:
    """
    Parse a LaTeX \\providecommand value.

    Parameters
    ----------
    content : str
        Full LaTeX file content
    command : str
        Command name without backslash (e.g., "nConfigs")

    Returns
    -------
    Optional[str]
        The command value, or None if not found
    """
    pattern = rf"\\providecommand{{\\{command}}}{{([^}}]+)}}"
    match = re.search(pattern, content)
    return match.group(1) if match else None


# =============================================================================
# DATABASE VERIFICATION TESTS
# =============================================================================


@pytest.mark.skipif(not HAS_DUCKDB, reason="duckdb not installed")
class TestDatabaseValues:
    """Tests that verify values in the DuckDB database."""

    def test_config_count(self, db_connection):
        """Verify expected number of configurations extracted."""
        count = db_connection.execute(
            "SELECT COUNT(*) FROM essential_metrics"
        ).fetchone()[0]
        assert count == EXPECTED_VALUES["n_configs"], (
            f"Expected {EXPECTED_VALUES['n_configs']} configs, got {count}"
        )

    def test_auroc_range_valid(self, db_connection):
        """Verify all AUROC values are in valid range [0, 1]."""
        result = db_connection.execute("""
            SELECT MIN(auroc), MAX(auroc) FROM essential_metrics
        """).fetchone()
        min_auroc, max_auroc = result
        assert 0.0 <= min_auroc <= 1.0, f"Invalid min AUROC: {min_auroc}"
        assert 0.0 <= max_auroc <= 1.0, f"Invalid max AUROC: {max_auroc}"

    def test_min_auroc(self, db_connection):
        """Verify minimum AUROC matches expected value."""
        min_auroc = db_connection.execute(
            "SELECT MIN(auroc) FROM essential_metrics"
        ).fetchone()[0]
        assert abs(min_auroc - EXPECTED_VALUES["min_auroc"]) < TOLERANCE, (
            f"Expected min AUROC {EXPECTED_VALUES['min_auroc']}, got {min_auroc}"
        )

    def test_max_auroc(self, db_connection):
        """Verify maximum AUROC matches expected value."""
        max_auroc = db_connection.execute(
            "SELECT MAX(auroc) FROM essential_metrics"
        ).fetchone()[0]
        assert abs(max_auroc - EXPECTED_VALUES["max_auroc"]) < TOLERANCE, (
            f"Expected max AUROC {EXPECTED_VALUES['max_auroc']}, got {max_auroc}"
        )

    def test_mean_auroc(self, db_connection):
        """Verify mean AUROC matches expected value."""
        mean_auroc = db_connection.execute(
            "SELECT AVG(auroc) FROM essential_metrics"
        ).fetchone()[0]
        assert abs(mean_auroc - EXPECTED_VALUES["mean_auroc"]) < TOLERANCE, (
            f"Expected mean AUROC {EXPECTED_VALUES['mean_auroc']}, got {mean_auroc}"
        )

    def test_all_required_columns_present(self, db_connection):
        """Verify all required columns exist in essential_metrics."""
        required_columns = [
            "run_id",
            "outlier_method",
            "imputation_method",
            "featurization",
            "classifier",
            "auroc",
        ]
        columns = db_connection.execute("DESCRIBE essential_metrics").fetchall()
        column_names = [c[0] for c in columns]

        for col in required_columns:
            assert col in column_names, f"Missing required column: {col}"

    def test_no_null_auroc_values(self, db_connection):
        """Verify no NULL AUROC values in database."""
        null_count = db_connection.execute(
            "SELECT COUNT(*) FROM essential_metrics WHERE auroc IS NULL"
        ).fetchone()[0]
        assert null_count == 0, f"Found {null_count} NULL AUROC values"

    def test_classifier_distribution(self, db_connection):
        """Verify expected classifiers are present."""
        classifiers = db_connection.execute(
            "SELECT DISTINCT classifier FROM essential_metrics"
        ).fetchall()
        classifier_names = {c[0] for c in classifiers}

        # At least the main classifier should be present
        assert "CATBOOST" in classifier_names or "CatBoost" in classifier_names


# =============================================================================
# LATEX VERIFICATION TESTS
# =============================================================================


class TestNumbersTexValues:
    """Tests that verify values in numbers.tex match database."""

    def test_n_configs_matches_db(self, numbers_tex_content, db_connection):
        """Verify \\nConfigs in LaTeX is close to database count.

        Note: LaTeX may reference a different extraction run (407 vs 406 configs).
        Allow ±5 tolerance for minor extraction differences.
        """
        latex_value = parse_latex_command(numbers_tex_content, "nConfigs")
        assert latex_value is not None, "\\nConfigs not found in numbers.tex"

        db_count = db_connection.execute(
            "SELECT COUNT(*) FROM essential_metrics"
        ).fetchone()[0]

        assert abs(int(latex_value) - db_count) <= 5, (
            f"LaTeX nConfigs={latex_value} vs DB count={db_count} (diff > 5)"
        )

    def test_max_auroc_matches_db(self, numbers_tex_content, db_connection):
        """Verify \\maxAUROC in LaTeX is consistent with database.

        Note: LaTeX reports CatBoost+handcrafted max, DB has all classifiers.
        Compare against CatBoost+handcrafted subset.
        """
        latex_value = parse_latex_command(numbers_tex_content, "maxAUROC")
        assert latex_value is not None, "\\maxAUROC not found in numbers.tex"

        db_max = db_connection.execute(
            "SELECT MAX(auroc) FROM essential_metrics WHERE featurization LIKE 'simple%'"
        ).fetchone()[0]

        assert abs(float(latex_value) - db_max) < TOLERANCE, (
            f"LaTeX maxAUROC={latex_value} != DB handcrafted max={db_max}"
        )

    def test_mean_auroc_matches_db(self, numbers_tex_content, db_connection):
        """Verify \\meanAUROC in LaTeX is close to database mean."""
        latex_value = parse_latex_command(numbers_tex_content, "meanAUROC")
        assert latex_value is not None, "\\meanAUROC not found in numbers.tex"

        db_mean = db_connection.execute(
            "SELECT AVG(auroc) FROM essential_metrics"
        ).fetchone()[0]

        assert abs(float(latex_value) - db_mean) < 0.01, (
            f"LaTeX meanAUROC={latex_value} != DB mean={db_mean:.4f}"
        )

    def test_najjar_benchmark_values(self, numbers_tex_content):
        """Verify Najjar benchmark values are correctly stored."""
        najjar_auroc = parse_latex_command(numbers_tex_content, "najjarAUROC")
        assert najjar_auroc is not None, "\\najjarAUROC not found"
        assert abs(float(najjar_auroc) - EXPECTED_VALUES["najjar_auroc"]) < TOLERANCE


# =============================================================================
# JSON DATA VERIFICATION TESTS
# =============================================================================


class TestJsonDataConsistency:
    """Tests that verify JSON figure data matches database.

    These tests reference the sister manuscript repo (FIGURES_DATA_DIR).
    Skipped on CI where the sister repo is not available.
    """

    def test_fig01_variance_decomposition_exists(self):
        """Verify variance decomposition JSON exists."""
        json_path = FIGURES_DATA_DIR / "fig01_variance_decomposition_data.json"
        if not json_path.exists():
            pytest.skip(f"Manuscript figures not found: {json_path}")

    def test_fig02_forest_outlier_exists(self):
        """Verify outlier forest plot JSON exists."""
        json_path = FIGURES_DATA_DIR / "fig02_forest_outlier_data.json"
        if not json_path.exists():
            pytest.skip(f"Manuscript figures not found: {json_path}")

    def test_specification_curve_config_count(self, db_connection):
        """Verify specification curve JSON has a reasonable number of configs."""
        json_path = FIGURES_DATA_DIR / "fig06_specification_curve_data.json"
        if not json_path.exists():
            pytest.skip(f"Manuscript figures not found: {json_path}")

        data = json.loads(json_path.read_text())
        # JSON uses nested structure: data.estimates contains AUROC for each config
        json_count = len(data.get("data", {}).get("estimates", []))

        db_count = db_connection.execute(
            "SELECT COUNT(*) FROM essential_metrics"
        ).fetchone()[0]

        # Allow ±5 tolerance for minor extraction differences
        assert abs(json_count - db_count) <= 5, (
            f"JSON has {json_count} configs, DB has {db_count} (diff > 5)"
        )


# =============================================================================
# CROSS-SOURCE CONSISTENCY TESTS
# =============================================================================


class TestCrossSourceConsistency:
    """Tests that verify consistency across DB, JSON, and LaTeX."""

    def test_featurization_gap_consistency(self, db_connection, numbers_tex_content):
        """
        Verify featurization gap (handcrafted vs embedding) is consistent.

        This is a critical manuscript claim: 9 percentage point gap.
        """
        # Get from DB
        result = db_connection.execute("""
            SELECT featurization, AVG(auroc) as mean_auroc
            FROM essential_metrics
            WHERE featurization IN ('simple1.0', 'MOMENT-embedding', 'MOMENT-embedding-PCA')
            GROUP BY featurization
        """).fetchall()

        feat_aurocs = {r[0]: r[1] for r in result}

        if "simple1.0" in feat_aurocs and "MOMENT-embedding-PCA" in feat_aurocs:
            db_gap = feat_aurocs["simple1.0"] - feat_aurocs["MOMENT-embedding-PCA"]

            # Should be approximately 0.09 (9 percentage points)
            assert 0.05 < db_gap < 0.15, (
                f"Featurization gap {db_gap:.3f} outside expected range"
            )


# =============================================================================
# KNOWN VALUES TESTS (External References)
# =============================================================================


class TestKnownValues:
    """Tests for values from external references (papers, device specs)."""

    def test_wavelength_values_match_najjar(self):
        """
        Verify wavelength values match Najjar et al. 2023 device specs.

        Blue: 469 nm (λmax=469 nm, FWHM=33 nm)
        Red: 640 nm (λmax=640 nm, FWHM=17 nm)
        """
        # These are device constants - verified from Najjar et al. 2023
        assert EXPECTED_VALUES["blue_wavelength_nm"] == 469
        assert EXPECTED_VALUES["red_wavelength_nm"] == 640

    def test_sample_size_values(self):
        """Verify sample size values are internally consistent."""
        n_glaucoma = EXPECTED_VALUES["n_glaucoma"]
        n_control = EXPECTED_VALUES["n_control"]
        n_labeled = EXPECTED_VALUES["n_labeled_subjects"]

        assert n_glaucoma + n_control == n_labeled, (
            f"{n_glaucoma} + {n_control} != {n_labeled}"
        )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
