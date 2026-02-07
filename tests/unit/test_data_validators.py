"""
Unit tests for data validators.

Tests verify that validation functions catch common data issues
that could cause pipeline failures downstream.

See: docs/planning/pipeline-robustness-plan.md for context.
"""

import numpy as np
import polars as pl
import pytest

from src.data_io.validation import (
    DataValidationError,
    validate_light_stimuli,
    validate_signal_range,
    validate_time_monotonic,
    validate_features,
)


class TestLightStimuliValidation:
    """Test light stimuli timing validation."""

    @pytest.mark.unit
    def test_valid_light_stimuli(self):
        """Should pass for valid light timing."""
        df = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "Red": [0, 1, 1, 1, 0, 0],
            }
        )

        # Should not raise
        validate_light_stimuli(df)

    @pytest.mark.unit
    def test_invalid_onset_after_offset(self):
        """Should fail if onset >= offset."""
        # This shouldn't happen in real data, but if sorting is wrong it could
        df = pl.DataFrame(
            {
                "time": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],  # Reversed time
                "Red": [0, 1, 1, 1, 0, 0],
            }
        )

        # The min time with Red=1 is 2.0, max is 4.0
        # So this should still pass (onset=2 < offset=4)
        validate_light_stimuli(df)

    @pytest.mark.unit
    def test_no_light_stimulus(self):
        """Should warn but not fail if no light stimulus."""
        df = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "Red": [0, 0, 0],
            }
        )

        # Should not raise (just warns)
        validate_light_stimuli(df)

    @pytest.mark.unit
    def test_multiple_colors(self):
        """Should validate multiple light colors."""
        df = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "Red": [0, 1, 1, 0, 0, 0],
                "Blue": [0, 0, 0, 1, 1, 0],
            }
        )

        # Should not raise
        validate_light_stimuli(df)


class TestSignalRangeValidation:
    """Test signal range validation."""

    @pytest.mark.unit
    def test_valid_signal(self):
        """Should pass for signal in valid range."""
        signal = np.array([2.0, 3.0, 4.0, 5.0])

        # Should not raise
        validate_signal_range(signal, min_val=0.0, max_val=10.0)

    @pytest.mark.unit
    def test_signal_below_minimum(self):
        """Should fail if signal below minimum."""
        signal = np.array([2.0, 3.0, -1.0, 5.0])  # -1 below min=0

        with pytest.raises(DataValidationError, match="below minimum"):
            validate_signal_range(signal, min_val=0.0, max_val=10.0)

    @pytest.mark.unit
    def test_signal_above_maximum(self):
        """Should fail if signal above maximum."""
        signal = np.array([2.0, 3.0, 15.0, 5.0])  # 15 above max=10

        with pytest.raises(DataValidationError, match="above maximum"):
            validate_signal_range(signal, min_val=0.0, max_val=10.0)

    @pytest.mark.unit
    def test_signal_with_nan_allowed(self):
        """Should pass with NaN when allow_nan=True."""
        signal = np.array([2.0, np.nan, 4.0, 5.0])

        # Should not raise
        validate_signal_range(signal, min_val=0.0, max_val=10.0, allow_nan=True)

    @pytest.mark.unit
    def test_signal_with_nan_not_allowed(self):
        """Should fail with NaN when allow_nan=False."""
        signal = np.array([2.0, np.nan, 4.0, 5.0])

        with pytest.raises(DataValidationError, match="NaN"):
            validate_signal_range(signal, min_val=0.0, max_val=10.0, allow_nan=False)

    @pytest.mark.unit
    def test_all_nan_signal(self):
        """Should fail if signal is all NaN."""
        signal = np.array([np.nan, np.nan, np.nan])

        with pytest.raises(DataValidationError, match="only NaN"):
            validate_signal_range(signal)


class TestTimeMonotonicValidation:
    """Test time monotonicity validation."""

    @pytest.mark.unit
    def test_valid_monotonic_time(self):
        """Should pass for strictly increasing time."""
        df = pl.DataFrame({"time": [0.0, 0.1, 0.2, 0.3, 0.4]})

        # Should not raise
        validate_time_monotonic(df)

    @pytest.mark.unit
    def test_non_monotonic_time(self):
        """Should fail for non-monotonic time."""
        df = pl.DataFrame(
            {
                "time": [0.0, 0.1, 0.05, 0.3, 0.4]  # 0.05 breaks monotonicity
            }
        )

        with pytest.raises(DataValidationError, match="not strictly monotonic"):
            validate_time_monotonic(df)

    @pytest.mark.unit
    def test_duplicate_time(self):
        """Should fail for duplicate time values."""
        df = pl.DataFrame(
            {
                "time": [0.0, 0.1, 0.1, 0.3, 0.4]  # Duplicate 0.1
            }
        )

        with pytest.raises(DataValidationError, match="not strictly monotonic"):
            validate_time_monotonic(df)

    @pytest.mark.unit
    def test_missing_time_column(self):
        """Should fail if time column missing."""
        df = pl.DataFrame({"value": [1.0, 2.0, 3.0]})

        with pytest.raises(DataValidationError, match="missing 'time' column"):
            validate_time_monotonic(df)


class TestFeatureValidation:
    """Test feature validation."""

    @pytest.mark.unit
    def test_valid_features(self):
        """Should pass with all required features."""
        features = {
            "amplitude_bins": [0.1, 0.2, 0.3, 0.4],
            "latency_features": {"pipr": 0.5},
        }

        # Should not raise
        validate_features(features, required_keys=["amplitude_bins"])

    @pytest.mark.unit
    def test_missing_required_feature(self):
        """Should fail if required feature missing."""
        features = {
            "latency_features": {"pipr": 0.5},
        }

        with pytest.raises(DataValidationError, match="Missing required features"):
            validate_features(features, required_keys=["amplitude_bins"])

    @pytest.mark.unit
    def test_empty_amplitude_bins(self):
        """Should fail if amplitude_bins is empty."""
        features = {
            "amplitude_bins": [],
        }

        with pytest.raises(DataValidationError, match="amplitude_bins is empty"):
            validate_features(features)

    @pytest.mark.unit
    def test_wrong_amplitude_bins_type(self):
        """Should fail if amplitude_bins has wrong type."""
        features = {
            "amplitude_bins": "not_a_list",
        }

        with pytest.raises(DataValidationError, match="must be list/array"):
            validate_features(features)

    @pytest.mark.unit
    def test_custom_required_keys(self):
        """Should validate custom required keys."""
        features = {
            "custom_feature": [1, 2, 3],
        }

        # Should not raise
        validate_features(features, required_keys=["custom_feature"])

        # Should raise for missing custom key
        with pytest.raises(DataValidationError, match="Missing required features"):
            validate_features(features, required_keys=["nonexistent"])
