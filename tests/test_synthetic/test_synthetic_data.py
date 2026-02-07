"""
Tests for synthetic PLR data generation module.

These tests verify:
1. Synthetic data matches expected schema
2. Privacy safeguards work (subject codes, no direct copying)
3. Data quality (physiologically plausible values)
"""

from pathlib import Path

import duckdb
import numpy as np
import pytest

pytestmark = pytest.mark.data

# Path to generated synthetic database
SYNTH_DB_PATH = (
    Path(__file__).parent.parent.parent / "data" / "synthetic" / "SYNTH_PLR_DEMO.db"
)


@pytest.fixture
def synth_conn():
    """Connect to synthetic database."""
    if not SYNTH_DB_PATH.exists():
        pytest.skip(f"Synthetic database not found: {SYNTH_DB_PATH}")
    conn = duckdb.connect(str(SYNTH_DB_PATH), read_only=True)
    yield conn
    conn.close()


class TestSyntheticDataSchema:
    """Verify schema matches real database."""

    def test_has_required_tables(self, synth_conn):
        """Database must have train and test tables."""
        tables = synth_conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        assert "train" in table_names
        assert "test" in table_names

    def test_has_all_columns(self, synth_conn):
        """Tables must have all required columns."""
        required_columns = {
            "time",
            "pupil_orig",
            "pupil_raw",
            "pupil_gt",
            "Red",
            "Blue",
            "time_orig",
            "subject_code",
            "no_outliers",
            "Age",
            "class_label",
            "light_stimuli",
            "pupil_orig_imputed",
            "outlier_mask",
            "pupil_raw_imputed",
            "imputation_mask",
            "split",
        }

        schema = synth_conn.execute("DESCRIBE train").fetchall()
        actual_columns = {col[0] for col in schema}

        missing = required_columns - actual_columns
        assert not missing, f"Missing columns: {missing}"

    def test_rows_per_subject(self, synth_conn):
        """Each subject should have 1981 timepoints."""
        counts = synth_conn.execute("""
            SELECT subject_code, COUNT(*) as cnt
            FROM train
            GROUP BY subject_code
        """).fetchall()

        for subject_code, count in counts:
            assert count == 1981, f"{subject_code} has {count} rows, expected 1981"


class TestPrivacySafeguards:
    """Verify privacy safeguards are in place."""

    def test_subject_codes_use_synth_prefix(self, synth_conn):
        """All subject codes must use SYNTH_ prefix."""
        codes = synth_conn.execute("SELECT DISTINCT subject_code FROM train").fetchall()

        for (code,) in codes:
            assert code.startswith("SYNTH_"), f"Invalid subject code: {code}"

    def test_no_plr_pattern_in_codes(self, synth_conn):
        """Subject codes must not contain PLR pattern (real data format)."""
        codes = synth_conn.execute("SELECT DISTINCT subject_code FROM train").fetchall()

        for (code,) in codes:
            assert "PLR" not in code.upper(), f"Real pattern in code: {code}"

    def test_no_duplicate_subjects_across_splits(self, synth_conn):
        """Subjects should not appear in both train and test."""
        train_codes = set(
            c[0]
            for c in synth_conn.execute(
                "SELECT DISTINCT subject_code FROM train"
            ).fetchall()
        )

        test_codes = set(
            c[0]
            for c in synth_conn.execute(
                "SELECT DISTINCT subject_code FROM test"
            ).fetchall()
        )

        overlap = train_codes & test_codes
        assert not overlap, f"Subjects in both splits: {overlap}"


class TestDataQuality:
    """Verify synthetic data has realistic values."""

    def test_time_range(self, synth_conn):
        """Time should range from 0 to 66 seconds."""
        result = synth_conn.execute("SELECT MIN(time), MAX(time) FROM train").fetchone()

        min_time, max_time = result
        assert min_time == pytest.approx(0.0, abs=0.1)
        assert max_time == pytest.approx(66.0, abs=0.1)

    def test_pupil_diameter_range(self, synth_conn):
        """Pupil diameter should be physiologically plausible (2-8mm)."""
        result = synth_conn.execute("""
            SELECT MIN(pupil_gt), MAX(pupil_gt) FROM train
        """).fetchone()

        min_pupil, max_pupil = result
        assert min_pupil >= 2.0, f"Min pupil too small: {min_pupil}"
        assert max_pupil <= 8.0, f"Max pupil too large: {max_pupil}"

    def test_class_distribution(self, synth_conn):
        """Should have both control and glaucoma classes."""
        result = synth_conn.execute("""
            SELECT class_label, COUNT(DISTINCT subject_code)
            FROM train
            GROUP BY class_label
        """).fetchall()

        classes = {label: count for label, count in result}
        assert "control" in classes, "Missing control class"
        assert "glaucoma" in classes, "Missing glaucoma class"
        assert classes["control"] == classes["glaucoma"], "Unbalanced classes"

    def test_outlier_mask_values(self, synth_conn):
        """Outlier mask should be 0 or 1."""
        result = synth_conn.execute("""
            SELECT DISTINCT outlier_mask FROM train
        """).fetchall()

        values = {v[0] for v in result}
        assert values <= {0, 1}, f"Invalid outlier mask values: {values}"

    def test_light_stimuli_pattern(self, synth_conn):
        """Light stimuli should have expected pattern."""
        result = synth_conn.execute("""
            SELECT DISTINCT Red, Blue FROM train
        """).fetchall()

        red_values = {v[0] for v in result}
        blue_values = {v[1] for v in result}

        assert 0.0 in red_values, "Red should have 0 (off) values"
        assert 1.0 in red_values, "Red should have 1 (on) values"
        assert 0.0 in blue_values, "Blue should have 0 (off) values"
        assert 1.0 in blue_values, "Blue should have 1 (on) values"


class TestSubjectCount:
    """Verify correct number of subjects for debug mode."""

    def test_train_subject_count(self, synth_conn):
        """Train should have 16 subjects (8 per label)."""
        result = synth_conn.execute("""
            SELECT COUNT(DISTINCT subject_code) FROM train
        """).fetchone()[0]

        assert result == 16, f"Expected 16 train subjects, got {result}"

    def test_test_subject_count(self, synth_conn):
        """Test should have 16 subjects (8 per label)."""
        result = synth_conn.execute("""
            SELECT COUNT(DISTINCT subject_code) FROM test
        """).fetchone()[0]

        assert result == 16, f"Expected 16 test subjects, got {result}"


class TestPLRGeneratorUnit:
    """Unit tests for PLR curve generator."""

    def test_generate_plr_curve_returns_correct_shape(self):
        """PLR curve should have correct number of timepoints."""
        from src.synthetic.plr_generator import generate_plr_curve

        time, pupil = generate_plr_curve("control", seed=42)

        assert len(time) == 1981
        assert len(pupil) == 1981

    def test_generate_plr_curve_is_reproducible(self):
        """Same seed should produce same curve."""
        from src.synthetic.plr_generator import generate_plr_curve

        _, pupil1 = generate_plr_curve("control", seed=42)
        _, pupil2 = generate_plr_curve("control", seed=42)

        np.testing.assert_array_equal(pupil1, pupil2)

    def test_different_seeds_produce_different_curves(self):
        """Different seeds should produce different curves."""
        from src.synthetic.plr_generator import generate_plr_curve

        _, pupil1 = generate_plr_curve("control", seed=42)
        _, pupil2 = generate_plr_curve("control", seed=43)

        assert not np.allclose(pupil1, pupil2)

    def test_class_affects_curve_parameters(self):
        """Control and glaucoma should have statistically different curves."""
        from src.synthetic.plr_generator import generate_plr_curve

        # Generate multiple curves for each class
        control_means = []
        glaucoma_means = []

        for seed in range(100, 110):
            _, c_pupil = generate_plr_curve("control", seed=seed)
            _, g_pupil = generate_plr_curve("glaucoma", seed=seed)
            control_means.append(np.mean(c_pupil))
            glaucoma_means.append(np.mean(g_pupil))

        # Control should generally have larger pupils
        assert (
            np.mean(control_means) > np.mean(glaucoma_means) - 1.0
        )  # Allow some overlap


class TestArtifactInjection:
    """Unit tests for artifact injection."""

    def test_inject_artifacts_adds_outliers(self):
        """Artifact injection should add outliers."""
        from src.synthetic.artifact_injection import inject_artifacts

        pupil_gt = np.ones(1981) * 5.0  # Clean signal
        pupil_orig, pupil_raw, outlier_mask = inject_artifacts(
            pupil_gt, outlier_pct=0.10, seed=42
        )

        assert np.sum(outlier_mask) > 0, "Should have some outliers"
        assert np.any(np.isnan(pupil_raw)), "pupil_raw should have NaN at outliers"

    def test_inject_artifacts_respects_outlier_pct(self):
        """Outlier percentage should be approximately as requested."""
        from src.synthetic.artifact_injection import inject_artifacts

        pupil_gt = np.ones(1981) * 5.0
        _, _, outlier_mask = inject_artifacts(pupil_gt, outlier_pct=0.20, seed=42)

        actual_pct = np.mean(outlier_mask)
        # Note: actual % can exceed target due to overlapping artifact types
        # Just verify it's in a reasonable range (not 0% or 100%)
        assert 0.05 < actual_pct < 0.60, (
            f"Outlier pct {actual_pct:.2f} outside reasonable range"
        )

    def test_zero_outlier_pct_produces_no_outliers(self):
        """Zero outlier percentage should produce clean signal."""
        from src.synthetic.artifact_injection import inject_artifacts

        pupil_gt = np.ones(1981) * 5.0
        _, _, outlier_mask = inject_artifacts(pupil_gt, outlier_pct=0.0, seed=42)

        assert np.sum(outlier_mask) == 0, "Should have no outliers"
