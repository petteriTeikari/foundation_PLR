"""Unit tests for data conversion utilities.

Tests functions from src/data_io/data_utils.py related to data conversion,
padding, trimming, and window splitting operations.
"""

import numpy as np
import pytest

from src.data_io.data_utils import (
    trim_to_multiple_of,
    pad_glaucoma_PLR,
    unpad_glaucoma_PLR,
    get_padding_indices,
    split_subjects_to_windows_PLR,
)


class TestTrimToMultipleOf:
    """Tests for trim_to_multiple_of function."""

    @pytest.mark.unit
    def test_trim_to_multiple_of_even(self, sample_plr_array):
        """Test trimming when result is evenly divisible."""
        # 1981 -> 1920 (20 * 96) with window_size=96
        result = trim_to_multiple_of(sample_plr_array, window_size=96)

        assert result.shape[1] == 1920
        assert result.shape[1] % 96 == 0

    @pytest.mark.unit
    def test_trim_to_multiple_of_odd(self, sample_plr_array):
        """Test trimming with odd remainder."""
        # 1981 -> 1900 (19 * 100) with window_size=100
        result = trim_to_multiple_of(sample_plr_array, window_size=100)

        assert result.shape[1] == 1900
        assert result.shape[1] % 100 == 0

    @pytest.mark.unit
    def test_trim_preserves_batch(self, sample_plr_array):
        """Test that batch dimension is unchanged after trimming."""
        n_subjects = sample_plr_array.shape[0]
        result = trim_to_multiple_of(sample_plr_array, window_size=96)

        assert result.shape[0] == n_subjects

    @pytest.mark.unit
    def test_trim_symmetric(self, sample_plr_array):
        """Test that trimming is roughly symmetric (removes from both ends)."""
        # Original length 1981, window 100 -> 1900
        # Should trim 81 points: 40 from start, 41 from end (or vice versa)
        result = trim_to_multiple_of(sample_plr_array, window_size=100)

        # Calculate the offset where original data starts in result
        trim_amount = sample_plr_array.shape[1] - result.shape[1]
        start_offset = trim_amount // 2
        if trim_amount % 2 != 0:
            start_offset += 1  # Match the function's behavior for odd trim

        # Verify the trimmed data matches the expected slice from original
        np.testing.assert_array_almost_equal(
            result, sample_plr_array[:, start_offset : start_offset + result.shape[1]]
        )


class TestPadGlaucomaPLR:
    """Tests for pad_glaucoma_PLR function."""

    @pytest.mark.unit
    def test_pad_output_length(self, sample_plr_array):
        """Test that padding produces correct output length."""
        # 1981 -> 2048 (4 * 512) with trim_to_size=512
        result = pad_glaucoma_PLR(sample_plr_array, trim_to_size=512)

        assert result.shape[1] == 2048
        assert result.shape[1] % 512 == 0

    @pytest.mark.unit
    def test_pad_contains_nans(self, sample_plr_array):
        """Test that NaN padding is applied."""
        result = pad_glaucoma_PLR(sample_plr_array, trim_to_size=512)

        # Should have NaN values in the padded regions
        assert np.any(np.isnan(result))

        # Count NaNs - should be (2048 - 1981) * n_subjects
        expected_nans = (2048 - 1981) * sample_plr_array.shape[0]
        actual_nans = np.isnan(result).sum()
        assert actual_nans == expected_nans

    @pytest.mark.unit
    def test_pad_data_preserved(self, sample_plr_array):
        """Test that original data is preserved in padded output."""
        result = pad_glaucoma_PLR(sample_plr_array, trim_to_size=512)

        # Get indices where original data should be
        start_idx, end_idx = get_padding_indices(1981, 2048)

        # Check that original data matches
        np.testing.assert_array_almost_equal(
            result[:, start_idx:end_idx], sample_plr_array
        )

    @pytest.mark.unit
    def test_pad_preserves_batch(self, sample_plr_array):
        """Test that batch dimension is unchanged after padding."""
        n_subjects = sample_plr_array.shape[0]
        result = pad_glaucoma_PLR(sample_plr_array, trim_to_size=512)

        assert result.shape[0] == n_subjects


class TestUnpadGlaucomaPLR:
    """Tests for unpad_glaucoma_PLR function."""

    @pytest.mark.unit
    def test_unpad_roundtrip(self, sample_plr_array):
        """Test that pad followed by unpad returns original data."""
        padded = pad_glaucoma_PLR(sample_plr_array, trim_to_size=512)
        unpadded = unpad_glaucoma_PLR(padded, length_PLR=1981)

        np.testing.assert_array_almost_equal(unpadded, sample_plr_array)

    @pytest.mark.unit
    def test_unpad_correct_length(self, sample_plr_array):
        """Test that unpadding returns correct length."""
        padded = pad_glaucoma_PLR(sample_plr_array, trim_to_size=512)
        unpadded = unpad_glaucoma_PLR(padded, length_PLR=1981)

        assert unpadded.shape[1] == 1981

    @pytest.mark.unit
    def test_unpad_removes_nans(self, sample_plr_array):
        """Test that unpadding removes NaN padding."""
        padded = pad_glaucoma_PLR(sample_plr_array, trim_to_size=512)
        unpadded = unpad_glaucoma_PLR(padded, length_PLR=1981)

        # Original data had no NaNs, unpadded should have none either
        assert not np.any(np.isnan(unpadded))


class TestGetPaddingIndices:
    """Tests for get_padding_indices function."""

    @pytest.mark.unit
    def test_padding_indices_symmetric(self):
        """Test that padding indices are symmetric."""
        start_idx, end_idx = get_padding_indices(length_orig=1981, length_padded=2048)

        # Difference should be 67 (2048 - 1981)
        total_padding = 2048 - 1981  # 67
        start_padding = start_idx  # 33
        end_padding = 2048 - end_idx  # 34

        # Should be roughly symmetric (differ by at most 1)
        assert abs(start_padding - end_padding) <= 1
        assert start_padding + end_padding == total_padding

    @pytest.mark.unit
    def test_padding_indices_correct_span(self):
        """Test that indices span correct length."""
        start_idx, end_idx = get_padding_indices(length_orig=1981, length_padded=2048)

        assert end_idx - start_idx == 1981

    @pytest.mark.unit
    def test_padding_indices_various_sizes(self):
        """Test padding indices for various size combinations."""
        test_cases = [
            (1981, 2048),  # Standard PLR case
            (100, 128),  # Smaller example
            (500, 512),  # Close to target
        ]

        for orig, padded in test_cases:
            start, end = get_padding_indices(orig, padded)
            assert end - start == orig
            assert start >= 0
            assert end <= padded


class TestSplitSubjectsToWindowsPLR:
    """Tests for split_subjects_to_windows_PLR function."""

    @pytest.mark.unit
    def test_split_to_windows_correct_reshape(self):
        """Test that window splitting produces correct shape."""
        # Create array that's already a multiple of window size
        n_subjects = 8
        n_timepoints = 2000  # 20 * 100
        data = np.random.randn(n_subjects, n_timepoints)

        result = split_subjects_to_windows_PLR(data, window_size=100)

        # 8 subjects * 20 windows = 160 pseudo-subjects
        assert result.shape[0] == 160
        assert result.shape[1] == 100

    @pytest.mark.unit
    def test_split_preserves_total_elements(self):
        """Test that splitting preserves total number of elements."""
        n_subjects = 4
        n_timepoints = 1000
        data = np.random.randn(n_subjects, n_timepoints)

        result = split_subjects_to_windows_PLR(data, window_size=100)

        assert result.size == data.size

    @pytest.mark.unit
    def test_split_data_continuity(self):
        """Test that data continuity is preserved after splitting."""
        n_subjects = 2
        n_timepoints = 200
        data = np.arange(n_subjects * n_timepoints).reshape(n_subjects, n_timepoints)

        result = split_subjects_to_windows_PLR(data, window_size=100)

        # First window of first subject should be 0-99
        np.testing.assert_array_equal(result[0, :], np.arange(100))
        # Second window of first subject should be 100-199
        np.testing.assert_array_equal(result[1, :], np.arange(100, 200))
