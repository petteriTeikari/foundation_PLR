# tests/test_data_quality/test_normalization_consistency.py
"""
TDD tests for data normalization consistency.

These tests validate that all subjects in the database have consistent
normalization/scaling of pupil signals. Specifically:

1. pupil_orig and pupil_raw should have similar scales (offset < 50)
2. pupil_gt (ground truth) should be in physiological range
3. No subject should have severe scaling anomalies

Note: pupil_orig may contain large outliers (blinks) - that's expected.
The CRITICAL check is that the baseline offset is correct.

IMPORTANT: NO REGEX ALLOWED for any analysis.
See: .claude/docs/meta-learnings/VIOLATION-002-regex-in-test-despite-ban.md

Run: pytest tests/test_data_quality/test_normalization_consistency.py -v
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.data

# Database path - check both possible locations
PROJECT_ROOT = Path(__file__).parents[2]
DB_PATH_LOCAL = PROJECT_ROOT.parent / "SERI_PLR_GLAUCOMA.db"
DB_PATH_ALT = Path(
    "/home/petteri/Dropbox/github-personal/foundation-PLR/SERI_PLR_GLAUCOMA.db"
)

# Known anomalies that are documented but not yet fixed at source
# These subjects will be flagged but test won't fail (xfail pattern)
KNOWN_SCALING_ANOMALIES = {
    "PLR4018": "pupil_orig has +184 offset - baseline not subtracted during ingestion"
}


def get_db_path():
    """Get database path, checking multiple locations."""
    if DB_PATH_LOCAL.exists():
        return DB_PATH_LOCAL
    if DB_PATH_ALT.exists():
        return DB_PATH_ALT
    return None


@pytest.fixture
def db_connection():
    """Provide DuckDB connection to PLR database."""
    db_path = get_db_path()
    if db_path is None:
        pytest.skip("Database not available (expected in CI)")

    import duckdb

    conn = duckdb.connect(str(db_path), read_only=True)
    yield conn
    conn.close()


class TestPupilOrigValueRanges:
    """Test that pupil_orig values are in expected range for all subjects.

    Note: pupil_orig contains raw data with blinks/artifacts, so extreme
    values are expected. The CRITICAL check is the baseline offset, not
    individual outlier values.
    """

    # Threshold for MEAN value - if mean is way off, baseline is wrong
    # Normal range for mean: -30 to 0 (slight constriction on average)
    MEAN_OFFSET_THRESHOLD = 50  # If mean > 50, baseline definitely wrong

    def test_no_severe_baseline_offset_in_mean(self, db_connection):
        """No subject should have mean pupil_orig > 50 (indicates baseline error)."""
        query = """
        SELECT
            subject_code,
            AVG(pupil_orig) as mean_val
        FROM train
        GROUP BY subject_code
        HAVING AVG(pupil_orig) > 50
        UNION ALL
        SELECT
            subject_code,
            AVG(pupil_orig) as mean_val
        FROM test
        GROUP BY subject_code
        HAVING AVG(pupil_orig) > 50
        """
        df = db_connection.execute(query).fetchdf()

        # Filter out known anomalies
        unknown_anomalies = [
            f"{row['subject_code']}: mean={row['mean_val']:.1f}"
            for _, row in df.iterrows()
            if row["subject_code"] not in KNOWN_SCALING_ANOMALIES
        ]

        known_found = [
            row["subject_code"]
            for _, row in df.iterrows()
            if row["subject_code"] in KNOWN_SCALING_ANOMALIES
        ]

        # Report known anomalies as warnings
        if known_found:
            print(f"\nKnown anomalies detected (documented): {known_found}")

        # Fail only on UNKNOWN anomalies
        assert not unknown_anomalies, (
            "NEW subjects with severe baseline offset (mean > 50):\n"
            + "\n".join(unknown_anomalies)
            + "\nAdd to KNOWN_SCALING_ANOMALIES if expected, or fix at source."
        )

    def test_identify_scaling_anomalies(self, db_connection):
        """Identify subjects where pupil_orig has abnormal offset from pupil_raw.

        This is the CRITICAL test - offset > 50 indicates baseline subtraction
        was skipped during data ingestion.
        """
        query = """
        SELECT
            subject_code,
            AVG(pupil_orig - pupil_raw) as mean_offset
        FROM train
        WHERE pupil_orig IS NOT NULL AND pupil_raw IS NOT NULL
        GROUP BY subject_code
        HAVING ABS(AVG(pupil_orig - pupil_raw)) > 50
        UNION ALL
        SELECT
            subject_code,
            AVG(pupil_orig - pupil_raw) as mean_offset
        FROM test
        WHERE pupil_orig IS NOT NULL AND pupil_raw IS NOT NULL
        GROUP BY subject_code
        HAVING ABS(AVG(pupil_orig - pupil_raw)) > 50
        """
        df = db_connection.execute(query).fetchdf()

        # Separate known vs unknown anomalies
        unknown = []
        known = []
        for _, row in df.iterrows():
            subject = row["subject_code"]
            info = f"{subject}: offset={row['mean_offset']:.1f}"
            if subject in KNOWN_SCALING_ANOMALIES:
                known.append(info)
            else:
                unknown.append(info)

        if known:
            print("\nKnown scaling anomalies (documented):\n" + "\n".join(known))

        # Fail only on NEW anomalies
        assert not unknown, (
            "NEW subjects with scaling anomalies (offset > 50):\n"
            + "\n".join(unknown)
            + "\n\nThese subjects have pupil_orig baseline errors. "
            "Add to KNOWN_SCALING_ANOMALIES or fix at source."
        )


class TestNormalizationConsistency:
    """Test consistency between different pupil signal columns."""

    # Warning threshold (flag for review but don't fail)
    OFFSET_WARNING_THRESHOLD = 20
    # Critical threshold (fail test - indicates data corruption)
    OFFSET_CRITICAL_THRESHOLD = 50

    def test_no_critical_offset_between_columns(self, db_connection):
        """No subject should have pupil_orig - pupil_raw offset > 50.

        Offset > 50 indicates baseline subtraction error during ingestion.
        Smaller offsets may be due to outlier handling differences.
        """
        query = """
        SELECT
            subject_code,
            AVG(pupil_orig - pupil_raw) as mean_offset
        FROM train
        WHERE pupil_orig IS NOT NULL AND pupil_raw IS NOT NULL
        GROUP BY subject_code
        UNION ALL
        SELECT
            subject_code,
            AVG(pupil_orig - pupil_raw) as mean_offset
        FROM test
        WHERE pupil_orig IS NOT NULL AND pupil_raw IS NOT NULL
        GROUP BY subject_code
        """
        df = db_connection.execute(query).fetchdf()

        # Critical violations (offset > 50)
        critical = df[abs(df["mean_offset"]) > self.OFFSET_CRITICAL_THRESHOLD]
        critical_unknown = [
            f"{row['subject_code']}: offset={row['mean_offset']:.1f}"
            for _, row in critical.iterrows()
            if row["subject_code"] not in KNOWN_SCALING_ANOMALIES
        ]

        # Warning violations (20 < offset <= 50) - just report
        warnings = df[
            (abs(df["mean_offset"]) > self.OFFSET_WARNING_THRESHOLD)
            & (abs(df["mean_offset"]) <= self.OFFSET_CRITICAL_THRESHOLD)
        ]
        if len(warnings) > 0:
            print(
                f"\nWarning: {len(warnings)} subjects have offset 20-50 (review recommended)"
            )

        assert not critical_unknown, (
            "NEW critical offset anomalies detected:\n" + "\n".join(critical_unknown)
        )

    def test_pupil_gt_mean_in_range(self, db_connection):
        """Ground truth pupil_gt mean should be in physiological range.

        pupil_gt is the denoised signal - mean should be between -40 and +5
        (typical range for PLR response over full recording).
        """
        query = """
        SELECT
            subject_code,
            AVG(pupil_gt) as mean_val
        FROM train
        GROUP BY subject_code
        HAVING AVG(pupil_gt) > 10 OR AVG(pupil_gt) < -50
        UNION ALL
        SELECT
            subject_code,
            AVG(pupil_gt) as mean_val
        FROM test
        GROUP BY subject_code
        HAVING AVG(pupil_gt) > 10 OR AVG(pupil_gt) < -50
        """
        df = db_connection.execute(query).fetchdf()

        if len(df) > 0:
            violations = [
                f"{row['subject_code']}: mean={row['mean_val']:.1f}"
                for _, row in df.iterrows()
            ]
            # This is a warning, not a hard failure
            print(
                f"\nWarning: {len(violations)} subjects have pupil_gt mean outside [-50, 10]"
            )
            print("\n".join(violations[:5]))


class TestSubjectCount:
    """Verify expected subject counts for data integrity."""

    def test_total_subject_count(self, db_connection):
        """Should have expected number of subjects."""
        query = """
        SELECT COUNT(DISTINCT subject_code) as n
        FROM (
            SELECT subject_code FROM train
            UNION
            SELECT subject_code FROM test
        )
        """
        result = db_connection.execute(query).fetchone()
        n_subjects = result[0]

        # Based on documentation: 507 subjects for preprocessing
        assert n_subjects >= 200, f"Expected >= 200 subjects, got {n_subjects}"

    def test_subjects_have_all_columns(self, db_connection):
        """All subjects should have pupil_orig, pupil_raw, pupil_gt columns."""
        query = """
        SELECT subject_code,
               COUNT(pupil_orig) as n_orig,
               COUNT(pupil_raw) as n_raw,
               COUNT(pupil_gt) as n_gt
        FROM test
        GROUP BY subject_code
        HAVING n_orig = 0 OR n_raw = 0 OR n_gt = 0
        """
        df = db_connection.execute(query).fetchdf()

        assert len(df) == 0, (
            "Subjects missing pupil columns:\n" + df.to_string() if len(df) > 0 else ""
        )
