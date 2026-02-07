"""Integration tests for data import functionality.

Tests DuckDB data loading, validation, and train/test splitting
using actual demo data when available.
"""

import numpy as np
import polars as pl
import pytest

from src.data_io.data_utils import (
    import_duckdb_as_dataframes,
    get_unique_polars_rows,
    check_for_data_lengths,
)


@pytest.fixture
def demo_db_path(demo_data_path, skip_if_no_demo_data):
    """Get demo database path, skip if unavailable."""
    return str(demo_data_path)


class TestDuckDBImport:
    """Tests for DuckDB data import functionality."""

    @pytest.mark.integration
    def test_duckdb_loading(self, demo_db_path):
        """Test that DuckDB file loads successfully."""
        df_train, df_test = import_duckdb_as_dataframes(demo_db_path)

        assert df_train is not None
        assert df_test is not None
        assert isinstance(df_train, pl.DataFrame)
        assert isinstance(df_test, pl.DataFrame)

    @pytest.mark.integration
    def test_required_columns_exist(self, demo_db_path):
        """Test that required columns exist in loaded data."""
        df_train, df_test = import_duckdb_as_dataframes(demo_db_path)

        required_columns = ["subject_code", "time", "pupil_raw"]

        for col in required_columns:
            assert col in df_train.columns, f"Missing column '{col}' in train"
            assert col in df_test.columns, f"Missing column '{col}' in test"

    @pytest.mark.integration
    def test_plr_length_per_subject(self, demo_db_path, minimal_cfg):
        """Test that each subject has correct PLR length (1981 timepoints)."""
        df_train, df_test = import_duckdb_as_dataframes(demo_db_path)

        expected_length = minimal_cfg["DATA"]["PLR_length"]

        # Check train split
        for df, split_name in [(df_train, "train"), (df_test, "test")]:
            unique_subjects = df["subject_code"].unique()
            for subject in unique_subjects:
                subject_df = df.filter(pl.col("subject_code") == subject)
                assert len(subject_df) == expected_length, (
                    f"Subject {subject} in {split_name} has {len(subject_df)} "
                    f"timepoints, expected {expected_length}"
                )

    @pytest.mark.integration
    def test_train_test_no_overlap(self, demo_db_path):
        """Test that train and test splits have no overlapping subjects."""
        df_train, df_test = import_duckdb_as_dataframes(demo_db_path)

        train_subjects = set(df_train["subject_code"].unique())
        test_subjects = set(df_test["subject_code"].unique())

        overlap = train_subjects.intersection(test_subjects)
        assert (
            len(overlap) == 0
        ), f"Found {len(overlap)} overlapping subjects between train and test: {overlap}"

    @pytest.mark.integration
    def test_time_vector_consistency(self, demo_db_path):
        """Test that time vectors are consistent across subjects."""
        df_train, _ = import_duckdb_as_dataframes(demo_db_path)

        # Get time values from first subject
        first_subject = df_train["subject_code"].unique()[0]
        first_subject_df = df_train.filter(pl.col("subject_code") == first_subject)
        reference_time = first_subject_df["time"].to_numpy()

        # Check other subjects have same time values
        unique_subjects = df_train["subject_code"].unique()
        for subject in unique_subjects[:5]:  # Check first 5 subjects
            subject_df = df_train.filter(pl.col("subject_code") == subject)
            subject_time = subject_df["time"].to_numpy()
            np.testing.assert_array_almost_equal(
                reference_time,
                subject_time,
                decimal=5,
                err_msg=f"Time vector mismatch for subject {subject}",
            )

    @pytest.mark.integration
    def test_pupil_values_reasonable_range(self, demo_db_path):
        """Test that pupil values are in reasonable physiological range."""
        df_train, df_test = import_duckdb_as_dataframes(demo_db_path)

        for df, split_name in [(df_train, "train"), (df_test, "test")]:
            pupil_values = df["pupil_raw"].drop_nulls().to_numpy()

            # Pupil diameter typically 2-8mm
            min_val = np.nanmin(pupil_values)
            max_val = np.nanmax(pupil_values)

            # Allow some margin for normalized/processed data
            assert min_val >= -10, f"Pupil min too low in {split_name}: {min_val}"
            assert max_val <= 20, f"Pupil max too high in {split_name}: {max_val}"


class TestDataValidation:
    """Tests for data validation functions with real data."""

    @pytest.mark.integration
    def test_get_unique_polars_rows(self, demo_db_path):
        """Test extraction of unique rows from DataFrame."""
        df_train, _ = import_duckdb_as_dataframes(demo_db_path)

        unique_df = get_unique_polars_rows(
            df_train,
            unique_col="subject_code",
            value_col="time",
            split="train",
            df_string="PLR",
        )

        # Should have one row per subject
        expected_subjects = len(df_train["subject_code"].unique())
        assert len(unique_df) == expected_subjects

    @pytest.mark.integration
    def test_check_for_data_lengths(self, demo_db_path, minimal_cfg):
        """Test data length validation function."""
        df_train, _ = import_duckdb_as_dataframes(demo_db_path)

        # Should not raise for valid data
        check_for_data_lengths(df_train, minimal_cfg)

    @pytest.mark.integration
    def test_data_import_with_outliers(self, demo_db_path):
        """Test that data import handles outlier columns correctly."""
        df_train, df_test = import_duckdb_as_dataframes(demo_db_path)

        # Check for outlier-related columns (may or may not exist)
        outlier_cols = ["outlier_labels", "no_outliers"]

        for col in outlier_cols:
            if col in df_train.columns:
                # If column exists, verify it's numeric
                assert df_train[col].dtype in [
                    pl.Int64,
                    pl.Int32,
                    pl.Float64,
                    pl.Float32,
                ], f"Outlier column {col} should be numeric"
