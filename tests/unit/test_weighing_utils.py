"""Unit tests for weighing utilities.

Tests functions from src/classification/weighing_utils.py for normalization
and edge cases discovered during synthetic pipeline testing.

See: docs/planning/pipeline-robustness-plan.md for context.
"""

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.classification.weighing_utils import (
    fix_nans_in_weights,
    minmax_norm,
    normalize_to_unity,
    zscore_norm,
)


@pytest.fixture
def xgboost_cfg():
    """Standard XGBoost configuration for testing."""
    return OmegaConf.create(
        {"MODEL": {"WEIGHING": {"weights_nan_weight_fixing": "unity"}}}
    )


class TestZscoreNorm:
    """Tests for zscore_norm function."""

    @pytest.mark.unit
    def test_zscore_normal_data(self, xgboost_cfg):
        """Should correctly z-score normalize valid data."""
        weights_array = np.zeros((5, 3))
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        feature_stats = {}

        result, stats = zscore_norm(
            weights_array,
            i=0,
            feature_stats=feature_stats,
            samples=samples,
            xgboost_cfg=xgboost_cfg,
        )

        # Mean should be 0, std should be 1 after z-score
        np.testing.assert_almost_equal(np.mean(result[:, 0]), 0, decimal=5)
        np.testing.assert_almost_equal(np.std(result[:, 0]), 1, decimal=5)

    @pytest.mark.unit
    def test_zscore_handles_all_nan(self, xgboost_cfg):
        """Should handle all-NaN arrays without error."""
        weights_array = np.zeros((5, 3))
        samples = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        feature_stats = {}

        # Should not raise
        result, stats = zscore_norm(
            weights_array,
            i=0,
            feature_stats=feature_stats,
            samples=samples,
            xgboost_cfg=xgboost_cfg,
        )

        assert np.isnan(stats[0]["mean"])
        assert np.isnan(stats[0]["std"])


class TestMinmaxNorm:
    """Tests for minmax_norm function."""

    @pytest.mark.unit
    def test_minmax_scales_to_0_1(self, xgboost_cfg):
        """Should scale values to [0, 1] range."""
        weights_array = np.zeros((5, 3))
        samples = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        feature_stats = {}

        result, stats = minmax_norm(
            weights_array,
            i=0,
            feature_stats=feature_stats,
            samples=samples,
            xgboost_cfg=xgboost_cfg,
        )

        assert np.min(result[:, 0]) == 0.0
        assert np.max(result[:, 0]) == 1.0
        np.testing.assert_almost_equal(result[:, 0], [0.0, 0.25, 0.5, 0.75, 1.0])

    @pytest.mark.unit
    def test_minmax_handles_all_nan(self, xgboost_cfg):
        """Should handle all-NaN arrays without error."""
        weights_array = np.zeros((5, 3))
        samples = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        feature_stats = {}

        result, stats = minmax_norm(
            weights_array,
            i=0,
            feature_stats=feature_stats,
            samples=samples,
            xgboost_cfg=xgboost_cfg,
        )

        assert np.isnan(stats[0]["min"])
        assert np.isnan(stats[0]["max"])


class TestNormalizeToUnity:
    """Tests for normalize_to_unity - critical function with division-by-zero fix."""

    @pytest.mark.unit
    def test_normalize_valid_data(self, xgboost_cfg):
        """Should normalize valid data correctly."""
        weights_array = np.zeros((5, 3))
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # sum = 15
        feature_stats = {}

        result, stats = normalize_to_unity(
            weights_array,
            i=0,
            feature_stats=feature_stats,
            samples=samples,
            xgboost_cfg=xgboost_cfg,
        )

        # Max should be 1.0 after normalization
        assert np.max(result[:, 0]) == 1.0
        assert stats[0]["sum"] == 15.0

    @pytest.mark.unit
    def test_normalize_handles_all_nan(self, xgboost_cfg):
        """Should handle all-NaN arrays without error."""
        weights_array = np.zeros((5, 3))
        samples = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        feature_stats = {}

        # Should not raise
        result, stats = normalize_to_unity(
            weights_array,
            i=0,
            feature_stats=feature_stats,
            samples=samples,
            xgboost_cfg=xgboost_cfg,
        )

        assert np.isnan(stats[0]["sum"])

    @pytest.mark.unit
    def test_normalize_handles_zero_max(self, xgboost_cfg):
        """Critical test: Should handle case where max is zero.

        This was the original bug - ZeroDivisionError when nanmax returns 0.
        """
        weights_array = np.zeros((5, 3))
        samples = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # All zeros
        feature_stats = {}

        # Should NOT raise ZeroDivisionError
        result, stats = normalize_to_unity(
            weights_array,
            i=0,
            feature_stats=feature_stats,
            samples=samples,
            xgboost_cfg=xgboost_cfg,
        )

        # After division by zero protection, NaN fixing replaces with 1.0
        assert stats[0]["sum"] == 0.0

    @pytest.mark.unit
    def test_normalize_handles_mixed_zeros_and_values(self, xgboost_cfg):
        """Should handle mix of zeros and valid values."""
        weights_array = np.zeros((5, 3))
        samples = np.array([0.0, 0.0, 0.0, 1.0, 2.0])  # sum = 3
        feature_stats = {}

        result, stats = normalize_to_unity(
            weights_array,
            i=0,
            feature_stats=feature_stats,
            samples=samples,
            xgboost_cfg=xgboost_cfg,
        )

        assert stats[0]["sum"] == 3.0
        # Max should be 1.0
        assert np.max(result[:, 0]) == 1.0

    @pytest.mark.unit
    def test_normalize_featurewise(self, xgboost_cfg):
        """Should work correctly with samplewise=False (feature weighting)."""
        weights_array = np.zeros((3, 5))  # 3 samples, 5 features
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        feature_stats = {}

        result, stats = normalize_to_unity(
            weights_array,
            i=0,
            feature_stats=feature_stats,
            samples=samples,
            xgboost_cfg=xgboost_cfg,
            samplewise=False,
        )

        # Should normalize along row 0
        assert np.max(result[0, :]) == 1.0


class TestFixNansInWeights:
    """Tests for fix_nans_in_weights function."""

    @pytest.mark.unit
    def test_fix_nans_unity_method(self):
        """Should replace NaNs with 1.0 using unity method."""
        weights = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result = fix_nans_in_weights(weights.copy(), method="unity")

        expected = np.array([1.0, 1.0, 3.0, 1.0, 5.0])
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.unit
    def test_fix_nans_mean_method(self):
        """Should replace NaNs with mean using mean method."""
        weights = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result = fix_nans_in_weights(weights.copy(), method="mean")

        # Mean of non-NaN values = (1+3+5)/3 = 3.0
        expected = np.array([1.0, 3.0, 3.0, 3.0, 5.0])
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.unit
    def test_fix_nans_unknown_method_raises(self):
        """Should raise ValueError for unknown method."""
        weights = np.array([1.0, np.nan, 3.0])

        with pytest.raises(ValueError, match="Unknown method"):
            fix_nans_in_weights(weights, method="unknown")

    @pytest.mark.unit
    def test_fix_nans_preserves_valid_values(self):
        """Should not modify non-NaN values."""
        weights = np.array([1.5, 2.5, 3.5, 4.5])
        original = weights.copy()
        result = fix_nans_in_weights(weights, method="unity")

        np.testing.assert_array_equal(result, original)


class TestSyntheticDataEdgeCases:
    """Tests for edge cases discovered during synthetic pipeline testing."""

    @pytest.fixture
    def xgboost_cfg(self):
        return OmegaConf.create(
            {"MODEL": {"WEIGHING": {"weights_nan_weight_fixing": "unity"}}}
        )

    @pytest.mark.unit
    def test_synthetic_data_all_same_values(self, xgboost_cfg):
        """Test with synthetic data where all features have same value.

        Synthetic PLR data with random features may have degenerate cases.
        """
        weights_array = np.zeros((10, 5))
        # All samples have the same feature value
        samples = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        feature_stats = {}

        # Should not crash
        result, stats = normalize_to_unity(
            weights_array,
            i=0,
            feature_stats=feature_stats,
            samples=samples,
            xgboost_cfg=xgboost_cfg,
        )

        assert stats[0]["sum"] == 5.0

    @pytest.mark.unit
    def test_synthetic_data_near_zero_values(self, xgboost_cfg):
        """Test with very small values that could cause numerical issues."""
        weights_array = np.zeros((5, 3))
        samples = np.array([1e-15, 1e-15, 1e-15, 1e-15, 1e-15])
        feature_stats = {}

        # Should not raise or produce inf
        result, stats = normalize_to_unity(
            weights_array,
            i=0,
            feature_stats=feature_stats,
            samples=samples,
            xgboost_cfg=xgboost_cfg,
        )

        assert not np.any(np.isinf(result))

    @pytest.mark.unit
    def test_division_by_zero_regression(self, xgboost_cfg):
        """Explicit regression test for ZeroDivisionError.

        This exact scenario caused the pipeline to crash before the fix.
        """
        weights_array = np.zeros((5, 3))
        # After dividing by sum, all values are 0
        # Then dividing by max (which is 0) caused ZeroDivisionError
        samples = np.zeros(5)
        feature_stats = {}

        try:
            result, stats = normalize_to_unity(
                weights_array,
                i=0,
                feature_stats=feature_stats,
                samples=samples,
                xgboost_cfg=xgboost_cfg,
            )
        except ZeroDivisionError:
            pytest.fail("ZeroDivisionError - regression bug detected!")

    @pytest.mark.unit
    def test_empty_array_handling(self, xgboost_cfg):
        """Test handling of empty arrays."""
        weights_array = np.zeros((0, 3))
        samples = np.array([])
        feature_stats = {}

        # This should handle gracefully (not crash)
        # Note: The function may produce NaN which is acceptable
        result, stats = normalize_to_unity(
            weights_array,
            i=0,
            feature_stats=feature_stats,
            samples=samples,
            xgboost_cfg=xgboost_cfg,
        )

        # Empty sum
        assert np.isnan(stats[0]["sum"]) or stats[0]["sum"] == 0.0
