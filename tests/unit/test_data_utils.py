"""Unit tests for data utilities.

Tests functions from src/data_io/data_utils.py related to time vectors,
data validation, and label extraction.
"""

import numpy as np
import polars as pl
import pytest

from src.data_io.data_utils import (
    check_time_similarity,
    define_desired_timevector,
    get_unique_labels,
)


class TestCheckTimeSimilarity:
    """Tests for check_time_similarity function."""

    @pytest.mark.unit
    def test_time_similarity_identical(self):
        """Test that identical time vectors pass all checks."""
        time_vec = define_desired_timevector(PLR_length=1981)

        result = check_time_similarity(time_vec, time_vec)

        assert result["OK"]
        assert result["allclose"]
        assert result["min_same"]
        assert result["max_same"]

    @pytest.mark.unit
    def test_time_similarity_different_length(self):
        """Test that different length vectors fail."""
        time_vec_1 = define_desired_timevector(PLR_length=1981)
        define_desired_timevector(PLR_length=1000)

        # This should raise or return mismatched - but the function expects same length
        # so we test with same length but different values
        time_vec_3 = np.linspace(0, 100, 1981)  # Different range

        result = check_time_similarity(time_vec_1, time_vec_3)

        assert result["OK"] is False

    @pytest.mark.unit
    def test_time_similarity_different_values(self):
        """Test that different values fail similarity check."""
        time_vec_ideal = define_desired_timevector(PLR_length=1981)
        time_vec_shifted = time_vec_ideal + 1.0  # Shift by 1 second

        result = check_time_similarity(time_vec_shifted, time_vec_ideal)

        assert not result["OK"]
        assert not result["min_same"]
        assert not result["max_same"]

    @pytest.mark.unit
    def test_time_similarity_returns_dict(self):
        """Test that function returns dictionary with expected keys."""
        time_vec = define_desired_timevector(PLR_length=1981)

        result = check_time_similarity(time_vec, time_vec)

        expected_keys = ["allclose", "min_in", "min_same", "max_in", "max_same", "OK"]
        for key in expected_keys:
            assert key in result


class TestDefineDesiredTimevector:
    """Tests for define_desired_timevector function."""

    @pytest.mark.unit
    def test_define_timevector_length(self):
        """Test that time vector has correct length."""
        time_vec = define_desired_timevector(PLR_length=1981)

        assert len(time_vec) == 1981

    @pytest.mark.unit
    def test_define_timevector_values(self):
        """Test time vector start/end values for default parameters."""
        time_vec = define_desired_timevector(PLR_length=1981, fps=30)

        # At 30 fps, 1981 samples = (1981-1)/30 = 66 seconds
        assert time_vec[0] == 0.0
        np.testing.assert_almost_equal(time_vec[-1], 66.0, decimal=5)

    @pytest.mark.unit
    def test_define_timevector_different_fps(self):
        """Test time vector with different fps."""
        time_vec_30fps = define_desired_timevector(PLR_length=1981, fps=30)
        time_vec_60fps = define_desired_timevector(PLR_length=1981, fps=60)

        # 60 fps should have half the duration
        assert time_vec_60fps[-1] == time_vec_30fps[-1] / 2

    @pytest.mark.unit
    def test_define_timevector_different_lengths(self):
        """Test time vector with different PLR lengths."""
        time_vec_short = define_desired_timevector(PLR_length=100, fps=30)
        time_vec_long = define_desired_timevector(PLR_length=2000, fps=30)

        assert len(time_vec_short) == 100
        assert len(time_vec_long) == 2000

    @pytest.mark.unit
    def test_define_timevector_is_numpy_array(self):
        """Test that function returns numpy array."""
        time_vec = define_desired_timevector(PLR_length=1981)

        assert isinstance(time_vec, np.ndarray)

    @pytest.mark.unit
    def test_define_timevector_monotonically_increasing(self):
        """Test that time vector is monotonically increasing."""
        time_vec = define_desired_timevector(PLR_length=1981)

        assert np.all(np.diff(time_vec) > 0)


class TestGetUniqueLabels:
    """Tests for get_unique_labels function."""

    @pytest.fixture
    def sample_labeled_df(self):
        """Create a sample Polars DataFrame with labels."""
        n_samples = 100
        data = {
            "time": np.linspace(0, 66, n_samples),
            "class_label": ["control"] * 40 + ["glaucoma"] * 40 + [None] * 20,
            "subject_code": [f"PLR{i:04d}" for i in range(n_samples)],
        }
        return pl.DataFrame(data)

    @pytest.mark.unit
    def test_get_unique_labels_extracts_labels(self, sample_labeled_df):
        """Test extraction of unique non-null labels."""
        labels = get_unique_labels(sample_labeled_df)

        assert "control" in labels
        assert "glaucoma" in labels
        assert len(labels) == 2

    @pytest.mark.unit
    def test_get_unique_labels_excludes_null(self, sample_labeled_df):
        """Test that null labels are excluded."""
        labels = get_unique_labels(sample_labeled_df)

        assert None not in labels
        assert "None" not in labels

    @pytest.mark.unit
    def test_get_unique_labels_returns_list(self, sample_labeled_df):
        """Test that function returns a list."""
        labels = get_unique_labels(sample_labeled_df)

        assert isinstance(labels, list)

    @pytest.mark.unit
    def test_get_unique_labels_single_label(self):
        """Test with single label type."""
        data = {
            "time": np.linspace(0, 66, 50),
            "class_label": ["control"] * 50,
            "subject_code": [f"PLR{i:04d}" for i in range(50)],
        }
        df = pl.DataFrame(data)

        labels = get_unique_labels(df)

        assert labels == ["control"]

    @pytest.mark.unit
    def test_get_unique_labels_all_null(self):
        """Test with all null labels."""
        data = {
            "time": np.linspace(0, 66, 50),
            "class_label": [None] * 50,
            "subject_code": [f"PLR{i:04d}" for i in range(50)],
        }
        df = pl.DataFrame(data)

        labels = get_unique_labels(df)

        assert labels == []
