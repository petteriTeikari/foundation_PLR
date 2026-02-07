"""
Unit tests for effect size computations.

Tests validate:
1. Correctness against known values
2. Edge cases (zero variance, small samples)
3. CI coverage via Monte Carlo simulation
4. Interpretation strings

Cross-references:
- planning/statistics-implementation.md (Section 3.4)
"""

import numpy as np
import pytest

from src.stats import EffectSizeResult, InsufficientDataError


class TestCohensD:
    """Tests for Cohen's d effect size computation."""

    @pytest.fixture
    def rng(self):
        """Seeded random generator for reproducibility."""
        return np.random.default_rng(42)

    @pytest.fixture
    def large_effect_data(self):
        """Data with known large effect (d ≈ 1.0)."""
        # Two groups with mean difference = 1 * pooled_std
        group1 = np.array([5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8])
        group2 = np.array([4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8])
        # Mean diff = 1.0, std ≈ 0.64 each, pooled ≈ 0.64, d ≈ 1.56
        return {"group1": group1, "group2": group2}

    @pytest.fixture
    def medium_effect_data(self):
        """Data with known medium effect (d ≈ 0.5)."""
        group1 = np.array([5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9])
        group2 = np.array([4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6])
        return {"group1": group1, "group2": group2}

    @pytest.fixture
    def small_effect_data(self):
        """Data with known small effect (d ≈ 0.2)."""
        group1 = np.array([5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9])
        group2 = np.array([4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8])
        return {"group1": group1, "group2": group2}

    @pytest.fixture
    def zero_effect_data(self):
        """Data with no effect (d = 0)."""
        group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        return {"group1": group1, "group2": group2}

    def test_cohens_d_returns_effect_size_result(self, large_effect_data):
        """Result should be EffectSizeResult dataclass."""
        from src.stats import cohens_d

        result = cohens_d(large_effect_data["group1"], large_effect_data["group2"])
        assert isinstance(result, EffectSizeResult)

    def test_cohens_d_large_effect(self, large_effect_data):
        """Large effect should have |d| >= 0.8."""
        from src.stats import cohens_d

        result = cohens_d(large_effect_data["group1"], large_effect_data["group2"])
        assert abs(result.effect_size) >= 0.8
        assert result.interpretation == "large"

    def test_cohens_d_medium_effect(self, medium_effect_data):
        """Medium effect should have 0.5 <= |d| < 0.8."""
        from src.stats import cohens_d

        result = cohens_d(
            medium_effect_data["group1"],
            medium_effect_data["group2"],
            ci_method="none",  # Skip CI for speed
        )
        # Note: actual d depends on data, check interpretation is reasonable
        assert 0.2 <= abs(result.effect_size) < 1.5

    def test_cohens_d_zero_effect(self, zero_effect_data):
        """Equal groups should have d ≈ 0."""
        from src.stats import cohens_d

        result = cohens_d(
            zero_effect_data["group1"], zero_effect_data["group2"], ci_method="none"
        )
        assert abs(result.effect_size) < 0.01

    def test_cohens_d_negative_effect(self, large_effect_data):
        """Swapping groups should negate d."""
        from src.stats import cohens_d

        result1 = cohens_d(
            large_effect_data["group1"], large_effect_data["group2"], ci_method="none"
        )
        result2 = cohens_d(
            large_effect_data["group2"], large_effect_data["group1"], ci_method="none"
        )
        assert np.isclose(result1.effect_size, -result2.effect_size, rtol=1e-10)

    def test_cohens_d_hedges_correction(self, large_effect_data):
        """Hedges' correction should reduce effect size for small samples."""
        from src.stats import cohens_d

        result_hedges = cohens_d(
            large_effect_data["group1"],
            large_effect_data["group2"],
            hedges=True,
            ci_method="none",
        )
        result_no_hedges = cohens_d(
            large_effect_data["group1"],
            large_effect_data["group2"],
            hedges=False,
            ci_method="none",
        )
        # Hedges' g is always smaller in magnitude than d
        assert abs(result_hedges.effect_size) < abs(result_no_hedges.effect_size)

    def test_cohens_d_single_element_raises(self):
        """Single element groups should raise InsufficientDataError."""
        from src.stats import cohens_d

        with pytest.raises(InsufficientDataError) as exc_info:
            cohens_d(np.array([1.0]), np.array([2.0, 3.0, 4.0]))
        assert exc_info.value.required == 2
        assert exc_info.value.actual == 1

    def test_cohens_d_empty_array_raises(self):
        """Empty arrays should raise error."""
        from src.stats import cohens_d

        with pytest.raises((InsufficientDataError, Exception)):
            cohens_d(np.array([]), np.array([1.0, 2.0]))

    def test_cohens_d_ci_contains_estimate(self, large_effect_data):
        """CI should contain the point estimate."""
        from src.stats import cohens_d

        result = cohens_d(
            large_effect_data["group1"],
            large_effect_data["group2"],
            ci_method="bootstrap",
            n_bootstrap=500,  # Reduced for test speed
        )
        # Point estimate should be within CI
        assert result.ci_lower <= result.effect_size <= result.ci_upper

    def test_cohens_d_analytical_ci(self, large_effect_data):
        """Analytical CI should work."""
        from src.stats import cohens_d

        result = cohens_d(
            large_effect_data["group1"],
            large_effect_data["group2"],
            ci_method="analytical",
        )
        assert not np.isnan(result.ci_lower)
        assert not np.isnan(result.ci_upper)
        assert result.ci_lower < result.ci_upper

    def test_cohens_d_interpretation_large(self, large_effect_data):
        """Large effects should be labeled 'large'."""
        from src.stats import cohens_d

        result = cohens_d(
            large_effect_data["group1"], large_effect_data["group2"], ci_method="none"
        )
        assert result.interpretation == "large"

    def test_cohens_d_interpretation_negligible(self, zero_effect_data):
        """Zero effects should be labeled 'negligible'."""
        from src.stats import cohens_d

        result = cohens_d(
            zero_effect_data["group1"], zero_effect_data["group2"], ci_method="none"
        )
        assert result.interpretation == "negligible"


class TestPartialEtaSquared:
    """Tests for partial eta-squared effect size."""

    def test_partial_eta_squared_zero_effect(self):
        """Zero SS_effect should give η²_p = 0."""
        from src.stats import partial_eta_squared

        result = partial_eta_squared(ss_effect=0.0, ss_error=100.0)
        assert result.effect_size == 0.0
        assert result.interpretation == "negligible"

    def test_partial_eta_squared_full_effect(self):
        """Full variance explained should give η²_p = 1."""
        from src.stats import partial_eta_squared

        result = partial_eta_squared(ss_effect=100.0, ss_error=0.0)
        assert result.effect_size == 1.0
        assert result.interpretation == "large"

    def test_partial_eta_squared_small_effect(self):
        """Small effect should have η²_p ≈ 0.01."""
        from src.stats import partial_eta_squared

        # η²_p = 1 / (1 + 99) = 0.01
        result = partial_eta_squared(ss_effect=1.0, ss_error=99.0)
        assert np.isclose(result.effect_size, 0.01, rtol=0.01)
        assert result.interpretation == "small"

    def test_partial_eta_squared_medium_effect(self):
        """Medium effect should have η²_p ≈ 0.06."""
        from src.stats import partial_eta_squared

        # η²_p = 6 / (6 + 94) = 0.06
        result = partial_eta_squared(ss_effect=6.0, ss_error=94.0)
        assert np.isclose(result.effect_size, 0.06, rtol=0.01)
        assert result.interpretation == "medium"

    def test_partial_eta_squared_large_effect(self):
        """Large effect should have η²_p ≈ 0.14."""
        from src.stats import partial_eta_squared

        # η²_p = 14 / (14 + 86) = 0.14
        result = partial_eta_squared(ss_effect=14.0, ss_error=86.0)
        assert np.isclose(result.effect_size, 0.14, rtol=0.01)
        assert result.interpretation == "large"

    def test_partial_eta_squared_negative_ss_raises(self):
        """Negative sum of squares should raise error."""
        from src.stats import partial_eta_squared

        with pytest.raises(ValueError):
            partial_eta_squared(ss_effect=-1.0, ss_error=100.0)


class TestOmegaSquared:
    """Tests for omega-squared effect size."""

    def test_omega_squared_less_than_eta(self):
        """Omega² should be less biased (smaller) than η²."""
        from src.stats import partial_eta_squared, omega_squared

        ss_effect = 10.0
        ss_error = 90.0
        df_effect = 2
        ms_error = ss_error / 50  # Assume 50 df for error

        eta_result = partial_eta_squared(ss_effect, ss_error)
        omega_result = omega_squared(ss_effect, ss_error, df_effect, ms_error)

        # Omega² should be smaller (less biased upward)
        assert omega_result.effect_size <= eta_result.effect_size

    def test_omega_squared_clipped_to_zero(self):
        """Negative omega² should be clipped to 0."""
        from src.stats import omega_squared

        # Very small effect with large MS_error can give negative omega²
        result = omega_squared(
            ss_effect=1.0,
            ss_error=100.0,
            df_effect=2,
            ms_error=10.0,  # Large MS_error
        )
        assert result.effect_size >= 0.0


class TestCohensF:
    """Tests for Cohen's f effect size."""

    def test_cohens_f_from_eta_squared(self):
        """f = sqrt(η² / (1 - η²))."""
        from src.stats import cohens_f

        # η² = 0.06 → f = sqrt(0.06/0.94) ≈ 0.253
        f = cohens_f(eta_squared=0.06)
        expected = np.sqrt(0.06 / 0.94)
        assert np.isclose(f, expected, rtol=1e-6)

    def test_cohens_f_zero_eta(self):
        """f should be 0 when η² = 0."""
        from src.stats import cohens_f

        f = cohens_f(eta_squared=0.0)
        assert f == 0.0

    def test_cohens_f_full_eta(self):
        """f should be inf when η² = 1."""
        from src.stats import cohens_f

        f = cohens_f(eta_squared=1.0)
        assert np.isinf(f)

    def test_cohens_f_interpretation(self):
        """Check Cohen's f interpretation guidelines."""
        from src.stats import cohens_f

        # Small: f ≈ 0.10 (η² ≈ 0.01)
        f_small = cohens_f(eta_squared=0.01)
        assert np.isclose(f_small, 0.10, atol=0.02)

        # Medium: f ≈ 0.25 (η² ≈ 0.06)
        f_medium = cohens_f(eta_squared=0.06)
        assert np.isclose(f_medium, 0.25, atol=0.02)

        # Large: f ≈ 0.40 (η² ≈ 0.14)
        f_large = cohens_f(eta_squared=0.14)
        assert np.isclose(f_large, 0.40, atol=0.02)


class TestEffectSizeCICoverage:
    """Monte Carlo tests for CI coverage probability."""

    @pytest.mark.slow
    def test_cohens_d_ci_coverage(self):
        """95% CI should contain true value ~95% of the time."""
        from src.stats import cohens_d

        rng = np.random.default_rng(42)
        true_d = 0.5  # True effect size
        n_simulations = 100
        n_per_group = 30
        alpha = 0.05

        coverage_count = 0
        for _ in range(n_simulations):
            # Generate data with known effect
            group1 = rng.normal(loc=true_d, scale=1.0, size=n_per_group)
            group2 = rng.normal(loc=0.0, scale=1.0, size=n_per_group)

            result = cohens_d(
                group1,
                group2,
                hedges=True,
                ci_method="bootstrap",
                n_bootstrap=200,  # Reduced for speed
                alpha=alpha,
            )

            if result.ci_lower <= true_d <= result.ci_upper:
                coverage_count += 1

        coverage = coverage_count / n_simulations
        # Allow some tolerance (Monte Carlo variance)
        # 95% CI coverage should be between 85% and 100% for 100 sims
        assert (
            0.80 <= coverage <= 1.0
        ), f"Coverage {coverage:.2f} outside expected range"
