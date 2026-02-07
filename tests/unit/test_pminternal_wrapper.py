"""Unit tests for pminternal Python wrapper."""

import numpy as np
import pytest


class TestPmInternalFallback:
    """Test pure Python fallback implementations."""

    def test_calibration_slope_intercept_python_perfect(self):
        """Test calibration with perfectly calibrated predictions."""
        from src.stats.pminternal_wrapper import _calibration_slope_intercept_python

        np.random.seed(42)
        n = 200
        y_true = np.random.binomial(1, 0.3, n)
        # Perfect calibration: predicted probabilities match true frequencies
        y_prob = 0.3 * np.ones(n)

        result = _calibration_slope_intercept_python(y_true, y_prob)

        assert "calibration_slope" in result
        assert "calibration_intercept" in result
        assert "oe_ratio" in result
        assert "brier_score" in result

        # O:E ratio should be close to 1.0 for well-calibrated predictions
        assert 0.7 < result["oe_ratio"] < 1.3

    def test_calibration_slope_intercept_python_overconfident(self):
        """Test calibration with overconfident (too extreme) predictions."""
        from src.stats.pminternal_wrapper import _calibration_slope_intercept_python

        np.random.seed(42)
        n = 200
        y_true = np.random.binomial(1, 0.3, n)
        # Overconfident predictions
        y_prob = np.where(y_true == 1, 0.9, 0.1)

        result = _calibration_slope_intercept_python(y_true, y_prob)

        # Slope should be positive
        assert result["calibration_slope"] > 0
        assert np.isfinite(result["calibration_slope"])

    def test_calibration_metrics_safe_no_r(self):
        """Test calibration_metrics_safe falls back to Python."""
        from src.stats.pminternal_wrapper import calibration_metrics_safe

        np.random.seed(42)
        n = 100
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.clip(
            y_true * 0.7 + (1 - y_true) * 0.3 + np.random.normal(0, 0.1, n), 0, 1
        )

        result = calibration_metrics_safe(y_true, y_prob)

        # Should return all required keys
        assert "calibration_slope" in result
        assert "calibration_intercept" in result
        assert "oe_ratio" in result
        assert "brier_score" in result
        assert "n_samples" in result
        assert result["n_samples"] == n

    def test_calibration_slope_edge_cases(self):
        """Test edge cases for calibration slope computation."""
        from src.stats.pminternal_wrapper import _calibration_slope_intercept_python

        # Extreme class imbalance (but still has both classes)
        np.random.seed(42)
        y_true = np.array([1] * 45 + [0] * 5)
        y_prob = np.random.uniform(0.8, 1.0, 50)

        result = _calibration_slope_intercept_python(y_true, y_prob)

        # Should still compute valid metrics
        assert np.isfinite(result["calibration_slope"])
        assert np.isfinite(result["oe_ratio"])
        assert result["oe_ratio"] > 0


class TestResultDataclasses:
    """Test dataclass definitions."""

    def test_validation_result_fields(self):
        """Test ValidationResult has all required fields."""
        from src.stats.pminternal_wrapper import ValidationResult

        result = ValidationResult(
            c_statistic=0.85,
            c_statistic_se=0.03,
            c_statistic_ci_lower=0.79,
            c_statistic_ci_upper=0.91,
            calibration_slope=0.95,
            calibration_slope_se=0.1,
            calibration_slope_ci_lower=0.75,
            calibration_slope_ci_upper=1.15,
            calibration_intercept=-0.1,
            calibration_intercept_se=0.05,
            calibration_intercept_ci_lower=-0.2,
            calibration_intercept_ci_upper=0.0,
            oe_ratio=1.02,
            oe_ratio_se=0.08,
            oe_ratio_ci_lower=0.86,
            oe_ratio_ci_upper=1.18,
            brier_score=0.15,
            scaled_brier=0.35,
            n_samples=200,
            n_events=60,
            event_rate=0.30,
        )

        assert result.c_statistic == 0.85
        assert result.calibration_slope == 0.95
        assert result.n_samples == 200

    def test_instability_result_fields(self):
        """Test InstabilityResult has all required fields."""
        from src.stats.pminternal_wrapper import InstabilityResult

        result = InstabilityResult(
            instability_index=0.15,
            slope_sd=0.12,
            slope_mean=0.80,
            slope_cv=0.15,
            slope_bootstrap=np.random.normal(0.8, 0.12, 200),
            slope_percentile_2_5=0.56,
            slope_percentile_97_5=1.04,
            stability_rating="moderate",
            n_bootstrap=200,
            n_samples=150,
        )

        assert result.instability_index == 0.15
        assert result.stability_rating == "moderate"
        assert len(result.slope_bootstrap) == 200


class TestRIntegration:
    """Tests that require R to be installed (skipped in CI)."""

    @pytest.fixture
    def check_r_available(self):
        """Check if R and pminternal are available."""
        from src.stats.pminternal_wrapper import _check_r_pminternal

        if not _check_r_pminternal():
            pytest.skip("R/pminternal not available")

    def test_validate_model_with_r(self, check_r_available):
        """Test validate_model with actual R backend."""
        from src.stats.pminternal_wrapper import validate_model

        np.random.seed(42)
        n = 150
        y_true = np.random.binomial(1, 0.3, n)
        # Clip to (0.01, 0.99) to avoid exact 0/1 which cause
        # NA/NaN/Inf errors in R's logistic calibration models
        y_prob = np.clip(
            y_true * 0.7 + (1 - y_true) * 0.3 + np.random.normal(0, 0.15, n),
            0.01,
            0.99,
        )

        result = validate_model(y_true, y_prob)

        assert 0.5 < result.c_statistic < 1.0
        assert result.calibration_slope > 0
        assert result.n_samples == n

    def test_instability_analysis_with_r(self, check_r_available):
        """Test instability_analysis with actual R backend."""
        from src.stats.pminternal_wrapper import instability_analysis

        np.random.seed(42)
        n = 150
        y_true = np.random.binomial(1, 0.3, n)
        # Clip to (0.01, 0.99) to avoid exact 0/1 which cause
        # NA/NaN/Inf errors in R's logistic calibration models
        y_prob = np.clip(
            y_true * 0.7 + (1 - y_true) * 0.3 + np.random.normal(0, 0.15, n),
            0.01,
            0.99,
        )

        result = instability_analysis(y_true, y_prob, n_bootstrap=100, random_state=42)

        assert result.instability_index >= 0
        # With synthetic data, some bootstrap samples may fail due to convergence
        # We require at least 3 successful samples for a meaningful test
        assert len(result.slope_bootstrap) >= 3
        assert result.stability_rating in ["stable", "moderate", "unstable"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
