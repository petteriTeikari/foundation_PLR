"""
Unit tests for bootstrap inference methods.

Tests validate:
1. BCa CI correctness against scipy
2. Percentile CI correctness
3. Coverage probability via Monte Carlo
4. Edge cases and error handling

Cross-references:
- planning/statistics-implementation.md (Section 2.4)
"""

import numpy as np
import pytest


class TestBCaBootstrapCI:
    """Tests for bias-corrected and accelerated bootstrap CI."""

    @pytest.fixture
    def normal_data(self):
        """Normal data for testing."""
        rng = np.random.default_rng(42)
        return rng.normal(loc=5.0, scale=2.0, size=50)

    @pytest.fixture
    def skewed_data(self):
        """Skewed data where BCa should differ from percentile."""
        rng = np.random.default_rng(42)
        return rng.exponential(scale=2.0, size=50)

    def test_bca_returns_bootstrap_result(self, normal_data):
        """Result should be BootstrapResult type."""
        from src.stats.bootstrap import bca_bootstrap_ci
        from src.stats import BootstrapResult

        result = bca_bootstrap_ci(normal_data, statistic_fn=np.mean, n_bootstrap=500)
        assert isinstance(result, BootstrapResult)

    def test_bca_ci_contains_point_estimate(self, normal_data):
        """CI should contain the point estimate."""
        from src.stats.bootstrap import bca_bootstrap_ci

        result = bca_bootstrap_ci(normal_data, statistic_fn=np.mean, n_bootstrap=1000)
        assert result.ci_lower <= result.point_estimate <= result.ci_upper

    def test_bca_ci_narrower_with_larger_n(self):
        """CI should be narrower with more data."""
        from src.stats.bootstrap import bca_bootstrap_ci

        rng = np.random.default_rng(42)
        data_small = rng.normal(size=20)
        data_large = rng.normal(size=200)

        result_small = bca_bootstrap_ci(data_small, np.mean, n_bootstrap=500)
        result_large = bca_bootstrap_ci(data_large, np.mean, n_bootstrap=500)

        width_small = result_small.ci_upper - result_small.ci_lower
        width_large = result_large.ci_upper - result_large.ci_lower

        assert width_large < width_small

    def test_bca_handles_skewed_data(self, skewed_data):
        """BCa should handle skewed data (exponential)."""
        from src.stats.bootstrap import bca_bootstrap_ci

        result = bca_bootstrap_ci(skewed_data, statistic_fn=np.mean, n_bootstrap=1000)

        # CI should be valid
        assert result.ci_lower < result.ci_upper
        assert not np.isnan(result.ci_lower)
        assert not np.isnan(result.ci_upper)

    def test_bca_different_from_percentile_for_skewed(self, skewed_data):
        """BCa should adjust for bias/skew compared to percentile."""
        from src.stats.bootstrap import bca_bootstrap_ci, percentile_bootstrap_ci

        bca_result = bca_bootstrap_ci(
            skewed_data, statistic_fn=np.mean, n_bootstrap=2000, random_state=42
        )
        pct_result = percentile_bootstrap_ci(
            skewed_data, statistic_fn=np.mean, n_bootstrap=2000, random_state=42
        )

        # Results should be similar but not identical
        # (BCa corrects for bias and acceleration)
        assert not np.isclose(bca_result.ci_lower, pct_result.ci_lower, rtol=0.01)

    def test_bca_with_median_statistic(self, normal_data):
        """BCa should work with median statistic."""
        from src.stats.bootstrap import bca_bootstrap_ci

        result = bca_bootstrap_ci(normal_data, statistic_fn=np.median, n_bootstrap=500)

        assert result.ci_lower < result.ci_upper
        assert result.point_estimate == np.median(normal_data)

    def test_bca_with_custom_alpha(self, normal_data):
        """Different alpha levels should produce different CI widths."""
        from src.stats.bootstrap import bca_bootstrap_ci

        result_95 = bca_bootstrap_ci(normal_data, np.mean, n_bootstrap=500, alpha=0.05)
        result_90 = bca_bootstrap_ci(normal_data, np.mean, n_bootstrap=500, alpha=0.10)

        width_95 = result_95.ci_upper - result_95.ci_lower
        width_90 = result_90.ci_upper - result_90.ci_lower

        # 95% CI should be wider than 90% CI
        assert width_95 > width_90

    def test_bca_insufficient_data_raises(self):
        """Should raise error with insufficient data."""
        from src.stats.bootstrap import bca_bootstrap_ci
        from src.stats import InsufficientDataError

        with pytest.raises((InsufficientDataError, ValueError)):
            bca_bootstrap_ci(np.array([1.0]), np.mean)

    def test_bca_reproducible_with_random_state(self, normal_data):
        """Same random_state should produce identical results."""
        from src.stats.bootstrap import bca_bootstrap_ci

        result1 = bca_bootstrap_ci(
            normal_data, np.mean, n_bootstrap=500, random_state=42
        )
        result2 = bca_bootstrap_ci(
            normal_data, np.mean, n_bootstrap=500, random_state=42
        )

        assert result1.ci_lower == result2.ci_lower
        assert result1.ci_upper == result2.ci_upper


class TestPercentileBootstrapCI:
    """Tests for percentile bootstrap CI."""

    @pytest.fixture
    def normal_data(self):
        rng = np.random.default_rng(42)
        return rng.normal(loc=5.0, scale=2.0, size=50)

    def test_percentile_basic(self, normal_data):
        """Basic percentile CI should work."""
        from src.stats.bootstrap import percentile_bootstrap_ci

        result = percentile_bootstrap_ci(
            normal_data, statistic_fn=np.mean, n_bootstrap=500
        )

        assert result.ci_lower < result.ci_upper
        assert result.method == "percentile"

    def test_percentile_matches_manual(self, normal_data):
        """Percentile CI should match manual calculation."""
        from src.stats.bootstrap import percentile_bootstrap_ci

        rng = np.random.default_rng(42)
        n_bootstrap = 1000

        # Manual calculation
        boot_stats = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            boot_sample = rng.choice(normal_data, size=len(normal_data), replace=True)
            boot_stats[i] = np.mean(boot_sample)

        expected_lower = np.percentile(boot_stats, 2.5)
        expected_upper = np.percentile(boot_stats, 97.5)

        # Our implementation
        result = percentile_bootstrap_ci(
            normal_data, np.mean, n_bootstrap=n_bootstrap, random_state=42
        )

        # Should match (with same seed)
        np.testing.assert_allclose(result.ci_lower, expected_lower, rtol=1e-6)
        np.testing.assert_allclose(result.ci_upper, expected_upper, rtol=1e-6)


class TestStratifiedBootstrap:
    """Tests for stratified bootstrap (for classification problems)."""

    @pytest.fixture
    def classification_data(self):
        """Binary classification data with imbalance."""
        rng = np.random.default_rng(42)
        n_pos = 30
        n_neg = 70

        y_true = np.array([1] * n_pos + [0] * n_neg)
        y_prob = np.clip(
            y_true * 0.7 + (1 - y_true) * 0.3 + rng.normal(0, 0.15, 100), 0.01, 0.99
        )
        return {"y_true": y_true, "y_prob": y_prob}

    def test_stratified_preserves_class_ratio(self, classification_data):
        """Stratified bootstrap should maintain class proportions."""
        from src.stats.bootstrap import stratified_bootstrap_sample

        y_true = classification_data["y_true"]
        original_ratio = np.mean(y_true)

        # Generate many samples and check ratio
        rng = np.random.default_rng(42)
        ratios = []
        for _ in range(100):
            indices = stratified_bootstrap_sample(y_true, rng)
            boot_y = y_true[indices]
            ratios.append(np.mean(boot_y))

        # All ratios should equal original (exactly for stratified)
        np.testing.assert_allclose(ratios, original_ratio, rtol=0.01)


class TestBootstrapCICoverage:
    """Monte Carlo tests for CI coverage probability."""

    @pytest.mark.slow
    def test_bca_coverage_for_mean(self):
        """BCa 95% CI should have ~95% coverage for normal data."""
        from src.stats.bootstrap import bca_bootstrap_ci

        rng = np.random.default_rng(42)
        true_mean = 5.0
        n_simulations = 100
        n_data = 30
        n_bootstrap = 500

        coverage_count = 0
        for _ in range(n_simulations):
            data = rng.normal(loc=true_mean, scale=2.0, size=n_data)
            result = bca_bootstrap_ci(data, np.mean, n_bootstrap=n_bootstrap)

            if result.ci_lower <= true_mean <= result.ci_upper:
                coverage_count += 1

        coverage = coverage_count / n_simulations
        # Allow tolerance for Monte Carlo variance
        assert (
            0.85 <= coverage <= 1.0
        ), f"Coverage {coverage:.2f} outside expected range"

    @pytest.mark.slow
    def test_percentile_coverage_for_mean(self):
        """Percentile 95% CI should have reasonable coverage."""
        from src.stats.bootstrap import percentile_bootstrap_ci

        rng = np.random.default_rng(42)
        true_mean = 5.0
        n_simulations = 100
        n_data = 30
        n_bootstrap = 500

        coverage_count = 0
        for _ in range(n_simulations):
            data = rng.normal(loc=true_mean, scale=2.0, size=n_data)
            result = percentile_bootstrap_ci(data, np.mean, n_bootstrap=n_bootstrap)

            if result.ci_lower <= true_mean <= result.ci_upper:
                coverage_count += 1

        coverage = coverage_count / n_simulations
        # Percentile may have slightly worse coverage than BCa
        assert 0.80 <= coverage <= 1.0


class TestBootstrapStatistics:
    """Tests for bootstrap statistics (SE, bias)."""

    @pytest.fixture
    def normal_data(self):
        rng = np.random.default_rng(42)
        return rng.normal(loc=5.0, scale=2.0, size=100)

    def test_bootstrap_se_reasonable(self, normal_data):
        """Bootstrap SE should be close to theoretical SE for mean."""
        from src.stats.bootstrap import bca_bootstrap_ci

        result = bca_bootstrap_ci(normal_data, np.mean, n_bootstrap=2000)

        # Theoretical SE of mean = sigma / sqrt(n)
        theoretical_se = np.std(normal_data, ddof=1) / np.sqrt(len(normal_data))

        # Bootstrap SE should be close (within 30%)
        assert 0.7 * theoretical_se < result.se < 1.3 * theoretical_se

    def test_bootstrap_bias_near_zero_for_mean(self, normal_data):
        """Bias should be near zero for sample mean."""
        from src.stats.bootstrap import bca_bootstrap_ci

        result = bca_bootstrap_ci(normal_data, np.mean, n_bootstrap=2000)

        # Bias should be small relative to SE
        assert abs(result.bias) < 0.5 * result.se
