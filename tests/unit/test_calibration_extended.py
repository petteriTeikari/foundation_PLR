"""
Unit tests for extended calibration metrics.

Tests validate:
1. Calibration slope and intercept
2. Observed:Expected ratio (Van Calster 2024 standard)
3. Brier score decomposition
4. Edge cases and error handling

Cross-references:
- planning/statistics-implementation.md (Section 2.6)
"""

import numpy as np
import pytest


class TestCalibrationSlope:
    """Tests for calibration slope computation."""

    @pytest.fixture
    def perfect_calibration_data(self):
        """Data with perfect calibration (slope = 1, intercept = 0)."""
        rng = np.random.default_rng(42)
        n = 200

        # Generate well-calibrated probabilities
        y_prob = rng.uniform(0.1, 0.9, n)
        # Generate outcomes matching probabilities
        y_true = (rng.random(n) < y_prob).astype(int)

        return {"y_true": y_true, "y_prob": y_prob}

    @pytest.fixture
    def overfitted_data(self):
        """Data with overfitting (slope < 1, predictions too extreme)."""
        rng = np.random.default_rng(42)
        n = 200

        # True probabilities centered around 0.5
        true_prob = rng.uniform(0.3, 0.7, n)
        y_true = (rng.random(n) < true_prob).astype(int)

        # Predicted probabilities are more extreme (overfitting)
        y_prob = np.clip(true_prob + (true_prob - 0.5) * 2, 0.01, 0.99)

        return {"y_true": y_true, "y_prob": y_prob}

    @pytest.fixture
    def underfitted_data(self):
        """Data with underfitting (slope > 1, predictions too conservative)."""
        rng = np.random.default_rng(42)
        n = 200

        # True probabilities spread across range
        true_prob = rng.uniform(0.1, 0.9, n)
        y_true = (rng.random(n) < true_prob).astype(int)

        # Predicted probabilities are more conservative (compressed toward 0.5)
        y_prob = 0.5 + (true_prob - 0.5) * 0.3

        return {"y_true": y_true, "y_prob": y_prob}

    def test_perfect_calibration_slope_near_1(self, perfect_calibration_data):
        """Well-calibrated model should have slope ≈ 1."""
        from src.stats.calibration_extended import calibration_slope_intercept

        result = calibration_slope_intercept(
            perfect_calibration_data["y_true"], perfect_calibration_data["y_prob"]
        )

        # Slope should be close to 1.0 (within tolerance for stochastic data)
        assert 0.7 < result.slope < 1.3

    def test_overfitting_slope_less_than_1(self, overfitted_data):
        """Overfitted model should have slope < 1."""
        from src.stats.calibration_extended import calibration_slope_intercept

        result = calibration_slope_intercept(
            overfitted_data["y_true"], overfitted_data["y_prob"]
        )

        # Overfitting: predictions too extreme → slope < 1
        assert result.slope < 1.0

    def test_underfitting_slope_greater_than_1(self, underfitted_data):
        """Underfitted model should have slope > 1."""
        from src.stats.calibration_extended import calibration_slope_intercept

        result = calibration_slope_intercept(
            underfitted_data["y_true"], underfitted_data["y_prob"]
        )

        # Underfitting: predictions too conservative → slope > 1
        assert result.slope > 1.0

    def test_extreme_probabilities_clipped(self):
        """Probabilities of 0 and 1 should be handled (clipped)."""
        from src.stats.calibration_extended import calibration_slope_intercept

        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.0, 0.1, 0.9, 1.0, 0.8])  # Contains 0 and 1

        # Should not raise, extreme values clipped internally
        result = calibration_slope_intercept(y_true, y_prob)
        assert np.isfinite(result.slope)

    def test_returns_calibration_result(self, perfect_calibration_data):
        """Should return CalibrationResult type."""
        from src.stats.calibration_extended import calibration_slope_intercept
        from src.stats import CalibrationResult

        result = calibration_slope_intercept(
            perfect_calibration_data["y_true"], perfect_calibration_data["y_prob"]
        )
        assert isinstance(result, CalibrationResult)


class TestOERatio:
    """Tests for Observed:Expected ratio (Van Calster 2024 standard)."""

    def test_perfect_calibration_ratio_near_1(self):
        """Well-calibrated model should have O:E ≈ 1."""
        from src.stats.calibration_extended import calibration_slope_intercept

        rng = np.random.default_rng(42)
        n = 500

        # Expected event rate equals observed
        y_prob = rng.uniform(0.2, 0.8, n)
        y_true = (rng.random(n) < y_prob).astype(int)

        result = calibration_slope_intercept(y_true, y_prob)

        # O:E should be close to 1
        assert 0.8 < result.o_e_ratio < 1.2

    def test_overprediction_ratio_less_than_1(self):
        """Model that overpredicts should have O:E < 1 (observed < expected)."""
        from src.stats.calibration_extended import calibration_slope_intercept

        rng = np.random.default_rng(42)
        n = 200

        # Low actual event rate
        y_true = np.array([1] * 20 + [0] * 180)  # 10% prevalence

        # High predicted probabilities (overprediction)
        y_prob = np.clip(rng.uniform(0.3, 0.5, n), 0.01, 0.99)  # ~40% mean

        result = calibration_slope_intercept(y_true, y_prob)

        # O:E = mean(y_true) / mean(y_prob) = 0.1 / 0.4 = 0.25
        assert result.o_e_ratio < 0.8

    def test_underprediction_ratio_greater_than_1(self):
        """Model that underpredicts should have O:E > 1 (observed > expected)."""
        from src.stats.calibration_extended import calibration_slope_intercept

        rng = np.random.default_rng(42)
        n = 200

        # High actual event rate
        y_true = np.array([1] * 160 + [0] * 40)  # 80% prevalence

        # Low predicted probabilities (underprediction)
        y_prob = np.clip(rng.uniform(0.2, 0.4, n), 0.01, 0.99)  # ~30% mean

        result = calibration_slope_intercept(y_true, y_prob)

        # O:E = mean(y_true) / mean(y_prob) = 0.8 / 0.3 = 2.67
        assert result.o_e_ratio > 1.5

    def test_zero_observed_handled(self):
        """Handle edge case where observed event rate is 0."""
        from src.stats.calibration_extended import calibration_slope_intercept
        from src.stats import SingleClassError

        # All negative outcomes
        y_true = np.array([0, 0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3, 0.2, 0.1])

        # Should raise SingleClassError (need both classes)
        with pytest.raises(SingleClassError):
            calibration_slope_intercept(y_true, y_prob)


class TestBrierDecomposition:
    """Tests for Brier score decomposition."""

    @pytest.fixture
    def classification_data(self):
        """Standard classification data."""
        rng = np.random.default_rng(42)
        n = 200

        y_prob = rng.uniform(0.1, 0.9, n)
        y_true = (rng.random(n) < y_prob).astype(int)

        return {"y_true": y_true, "y_prob": y_prob}

    def test_brier_decomposition_sum_equals_brier(self, classification_data):
        """Reliability - Resolution + Uncertainty = Brier."""
        from src.stats.calibration_extended import brier_decomposition

        result = brier_decomposition(
            classification_data["y_true"], classification_data["y_prob"]
        )

        # Check decomposition identity
        reconstructed = (
            result.scalars["reliability"]
            - result.scalars["resolution"]
            + result.scalars["uncertainty"]
        )
        np.testing.assert_allclose(
            reconstructed, result.scalars["brier_score"], rtol=0.01
        )

    def test_reliability_is_non_negative(self, classification_data):
        """Reliability (calibration error) should be >= 0."""
        from src.stats.calibration_extended import brier_decomposition

        result = brier_decomposition(
            classification_data["y_true"], classification_data["y_prob"]
        )

        assert result.scalars["reliability"] >= 0

    def test_resolution_is_non_negative(self, classification_data):
        """Resolution (discrimination) should be >= 0."""
        from src.stats.calibration_extended import brier_decomposition

        result = brier_decomposition(
            classification_data["y_true"], classification_data["y_prob"]
        )

        assert result.scalars["resolution"] >= 0

    def test_uncertainty_depends_only_on_prevalence(self):
        """Uncertainty = prevalence × (1 - prevalence)."""
        from src.stats.calibration_extended import brier_decomposition

        rng = np.random.default_rng(42)

        # Different prevalences
        for prevalence in [0.2, 0.5, 0.8]:
            n = 200
            n_pos = int(n * prevalence)
            y_true = np.array([1] * n_pos + [0] * (n - n_pos))
            y_prob = rng.uniform(0.1, 0.9, n)

            result = brier_decomposition(y_true, y_prob)

            expected_uncertainty = prevalence * (1 - prevalence)
            np.testing.assert_allclose(
                result.scalars["uncertainty"], expected_uncertainty, rtol=0.01
            )


class TestCalibrationIntegration:
    """Integration tests for calibration metrics."""

    def test_all_metrics_consistent(self):
        """All calibration metrics should tell consistent story."""
        from src.stats.calibration_extended import (
            calibration_slope_intercept,
            brier_decomposition,
        )

        rng = np.random.default_rng(42)
        n = 500

        # Well-calibrated model
        y_prob = rng.uniform(0.1, 0.9, n)
        y_true = (rng.random(n) < y_prob).astype(int)

        slope_result = calibration_slope_intercept(y_true, y_prob)
        brier_result = brier_decomposition(y_true, y_prob)

        # Well calibrated: slope ≈ 1, O:E ≈ 1, low reliability
        assert 0.7 < slope_result.slope < 1.3
        assert 0.8 < slope_result.o_e_ratio < 1.2
        # Reliability (calibration error) should be relatively small
        assert brier_result.scalars["reliability"] < 0.05
