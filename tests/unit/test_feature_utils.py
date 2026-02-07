"""Unit tests for feature utilities.

Tests functions from src/featurization/feature_utils.py for light timing extraction
and edge cases discovered during synthetic pipeline testing.

See: docs/planning/pipeline-robustness-plan.md for context.
"""

import numpy as np
import polars as pl
import pytest

from src.featurization.feature_utils import (
    get_light_stimuli_timings,
    get_top1_of_col,
    replace_zeros_with_null,
)


class TestReplaceZerosWithNull:
    """Tests for replace_zeros_with_null function."""

    @pytest.mark.unit
    def test_replaces_zeros_with_nan(self):
        """Should replace zeros with NaN in specified column."""
        df = pl.DataFrame({"time": [0.0, 1.0, 2.0, 3.0], "Red": [0, 1, 1, 0]})
        result = replace_zeros_with_null(df, col="Red")

        # Check that zeros are now NaN
        red_values = result["Red"].to_numpy()
        assert np.isnan(red_values[0])
        assert red_values[1] == 1.0
        assert red_values[2] == 1.0
        assert np.isnan(red_values[3])

    @pytest.mark.unit
    def test_preserves_non_zero_values(self):
        """Should preserve all non-zero values."""
        df = pl.DataFrame({"time": [0.0, 1.0, 2.0], "Blue": [0.5, 1.0, 0.8]})
        result = replace_zeros_with_null(df, col="Blue")

        np.testing.assert_array_almost_equal(result["Blue"].to_numpy(), [0.5, 1.0, 0.8])

    @pytest.mark.unit
    def test_handles_all_zeros(self):
        """Should handle column with all zeros."""
        df = pl.DataFrame({"time": [0.0, 1.0, 2.0], "Red": [0, 0, 0]})
        result = replace_zeros_with_null(df, col="Red")

        # All should be NaN
        assert all(np.isnan(result["Red"].to_numpy()))


class TestGetTop1OfCol:
    """Tests for get_top1_of_col function - critical light timing extraction."""

    @pytest.mark.unit
    def test_gets_first_timepoint_with_light_on(self):
        """Should return first timepoint where light is on (ascending sort)."""
        df = pl.DataFrame(
            {
                "time": [0.0, 5.0, 10.0, 15.0, 20.0],
                "Red": [0, 0, 1, 1, 0],  # Light on from 10.0 to 15.0
            }
        )
        result = get_top1_of_col(df, col="Red", descending=False)

        assert result.item(0, "time") == 10.0

    @pytest.mark.unit
    def test_gets_last_timepoint_with_light_on(self):
        """Should return last timepoint where light is on (descending sort)."""
        df = pl.DataFrame(
            {
                "time": [0.0, 5.0, 10.0, 15.0, 20.0],
                "Red": [0, 0, 1, 1, 0],  # Light on from 10.0 to 15.0
            }
        )
        result = get_top1_of_col(df, col="Red", descending=True)

        assert result.item(0, "time") == 15.0

    @pytest.mark.unit
    def test_sorts_by_time_not_column_value(self):
        """Critical test: Must sort by TIME, not by column value.

        This was the original bug - when all Red values are 1, sorting by
        the column value doesn't differentiate rows, leading to incorrect
        onset/offset times.
        """
        # Simulate PLR data where light column has same value (1) during light-on period
        df = pl.DataFrame(
            {
                "time": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
                "Red": [0, 1, 1, 1, 1, 0],  # Light on from 2.0 to 8.0
            }
        )

        onset_row = get_top1_of_col(df, col="Red", descending=False)
        offset_row = get_top1_of_col(df, col="Red", descending=True)

        onset = onset_row.item(0, "time")
        offset = offset_row.item(0, "time")

        # CRITICAL: onset must be before offset
        assert onset < offset, f"Light onset ({onset}) must be before offset ({offset})"
        assert onset == 2.0
        assert offset == 8.0

    @pytest.mark.unit
    def test_handles_single_timepoint_light_on(self):
        """Should handle edge case of light on for single timepoint."""
        df = pl.DataFrame(
            {
                "time": [0.0, 5.0, 10.0],
                "Red": [0, 1, 0],  # Light on only at 5.0
            }
        )

        onset_row = get_top1_of_col(df, col="Red", descending=False)
        offset_row = get_top1_of_col(df, col="Red", descending=True)

        # Both should return 5.0 (single timepoint)
        assert onset_row.item(0, "time") == 5.0
        assert offset_row.item(0, "time") == 5.0

    @pytest.mark.unit
    def test_raises_on_all_zeros(self):
        """Should raise assertion when no non-zero values exist."""
        df = pl.DataFrame({"time": [0.0, 1.0, 2.0], "Red": [0, 0, 0]})

        with pytest.raises(AssertionError, match="No samples"):
            get_top1_of_col(df, col="Red", descending=False)

    @pytest.mark.unit
    def test_handles_float_light_values(self):
        """Should handle float values in light column (not just 0/1)."""
        df = pl.DataFrame(
            {
                "time": [0.0, 5.0, 10.0, 15.0, 20.0],
                "Blue": [0.0, 0.5, 1.0, 0.8, 0.0],  # Non-zero from 5.0 to 15.0
            }
        )

        onset_row = get_top1_of_col(df, col="Blue", descending=False)
        offset_row = get_top1_of_col(df, col="Blue", descending=True)

        assert onset_row.item(0, "time") == 5.0
        assert offset_row.item(0, "time") == 15.0


class TestGetLightStimuliTimings:
    """Tests for get_light_stimuli_timings - high-level timing extraction."""

    @pytest.fixture
    def standard_plr_data(self):
        """Create standard PLR-like data with Red and Blue light stimuli."""
        # Typical PLR protocol: Red light 0-5s, Blue light 10-15s
        n_timepoints = 200
        time = np.linspace(0, 20, n_timepoints)

        red = np.zeros(n_timepoints)
        red[(time >= 2.0) & (time <= 7.0)] = 1.0

        blue = np.zeros(n_timepoints)
        blue[(time >= 10.0) & (time <= 15.0)] = 1.0

        return pl.DataFrame({"time": time, "Red": red, "Blue": blue})

    @pytest.mark.unit
    def test_extracts_correct_timings(self, standard_plr_data):
        """Should extract correct onset/offset for both colors."""
        timings = get_light_stimuli_timings(standard_plr_data)

        # Red light: onset ~2.0, offset ~7.0
        assert "Red" in timings
        assert timings["Red"]["light_onset"] < timings["Red"]["light_offset"]
        assert abs(timings["Red"]["light_onset"] - 2.0) < 0.2  # Within tolerance
        assert abs(timings["Red"]["light_offset"] - 7.0) < 0.2

        # Blue light: onset ~10.0, offset ~15.0
        assert "Blue" in timings
        assert timings["Blue"]["light_onset"] < timings["Blue"]["light_offset"]
        assert abs(timings["Blue"]["light_onset"] - 10.0) < 0.2
        assert abs(timings["Blue"]["light_offset"] - 15.0) < 0.2

    @pytest.mark.unit
    def test_computes_correct_duration(self, standard_plr_data):
        """Should compute correct light duration."""
        timings = get_light_stimuli_timings(standard_plr_data)

        # Duration should be offset - onset
        red_duration = timings["Red"]["light_offset"] - timings["Red"]["light_onset"]
        assert abs(timings["Red"]["light_duration"] - red_duration) < 0.001

        blue_duration = timings["Blue"]["light_offset"] - timings["Blue"]["light_onset"]
        assert abs(timings["Blue"]["light_duration"] - blue_duration) < 0.001

    @pytest.mark.unit
    def test_onset_always_before_offset(self, standard_plr_data):
        """Critical assertion: onset must always be before offset."""
        timings = get_light_stimuli_timings(standard_plr_data)

        for color in ["Red", "Blue"]:
            onset = timings[color]["light_onset"]
            offset = timings[color]["light_offset"]
            assert (
                onset < offset
            ), f"{color} light onset ({onset}) must be before offset ({offset})"


class TestSyntheticDataEdgeCases:
    """Tests for edge cases from synthetic data generation."""

    @pytest.mark.unit
    def test_synthetic_data_pattern(self):
        """Test with synthetic data pattern that caused original bug.

        Synthetic data has light columns with all 1s during light period.
        The bug was sorting by column value instead of time.
        """
        # Simulate synthetic PLR data (1981 timepoints, ~20 seconds)
        n_timepoints = 1981
        time = np.linspace(0, 20, n_timepoints)

        # Red light on from 2-7 seconds (all values are 1 during this period)
        red = np.zeros(n_timepoints)
        red_on_mask = (time >= 2.0) & (time <= 7.0)
        red[red_on_mask] = 1.0

        df = pl.DataFrame(
            {
                "time": time,
                "Red": red,
                "Blue": np.zeros(n_timepoints),  # Blue light not used in this test
            }
        )

        onset_row = get_top1_of_col(df, col="Red", descending=False)
        offset_row = get_top1_of_col(df, col="Red", descending=True)

        onset = onset_row.item(0, "time")
        offset = offset_row.item(0, "time")

        # This is the exact assertion that failed before the fix
        assert (
            onset < offset
        ), f"BUG REGRESSION: Light onset ({onset}) should be before offset ({offset})"

        # Additional sanity checks
        assert 1.9 < onset < 2.1  # Should be ~2.0
        assert 6.9 < offset < 7.1  # Should be ~7.0

    @pytest.mark.unit
    def test_unsorted_input_data(self):
        """Should handle input data that's not sorted by time."""
        # Create unsorted data
        df = pl.DataFrame(
            {
                "time": [10.0, 0.0, 5.0, 15.0, 20.0],
                "Red": [1, 0, 1, 1, 0],  # Light on at times 5, 10, 15
            }
        )

        onset_row = get_top1_of_col(df, col="Red", descending=False)
        offset_row = get_top1_of_col(df, col="Red", descending=True)

        onset = onset_row.item(0, "time")
        offset = offset_row.item(0, "time")

        assert onset == 5.0  # Earliest time with light on
        assert offset == 15.0  # Latest time with light on
        assert onset < offset
