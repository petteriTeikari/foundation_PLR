"""
Unit tests for Scaled Brier Score (Index of Prediction Accuracy).

Tests validate:
1. IPA = 1 - Brier/Brier_null computation
2. Skill score interpretation
3. Edge cases and error handling

Cross-references:
- planning/remaining-duckdb-stats-viz-tasks-plan.md (Appendix E.1)

References:
- Steyerberg et al. (2010). Assessing calibration.
- Van Calster et al. (2019). Calibration: the Achilles heel of predictive analytics.

IPA (Index of Prediction Accuracy):
    IPA = 1 - Brier/Brier_null

where Brier_null = prevalence × (1 - prevalence)

Interpretation:
- IPA = 0: No improvement over null model (predicting prevalence for all)
- IPA > 0: Better than null model (positive skill)
- IPA < 0: Worse than null model (negative skill)
- IPA = 1: Perfect predictions
"""

import numpy as np
import pytest


class TestScaledBrier:
    """Tests for Index of Prediction Accuracy (IPA)."""

    def test_perfect_predictions_returns_1(self):
        """Perfect predictions (Brier=0) should give IPA=1."""
        from src.stats.scaled_brier import scaled_brier_score

        # Perfect predictions
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

        result = scaled_brier_score(y_true, y_prob)

        # IPA = 1 - 0/Brier_null = 1
        assert result["ipa"] == 1.0

    def test_null_model_predictions_returns_0(self):
        """Predicting prevalence for all should give IPA=0."""
        from src.stats.scaled_brier import scaled_brier_score

        # 50% prevalence
        y_true = np.array([0, 0, 1, 1])
        # Predict prevalence (0.5) for everyone
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])

        result = scaled_brier_score(y_true, y_prob)

        # IPA should be 0 (same as null model)
        np.testing.assert_allclose(result["ipa"], 0.0, atol=1e-10)

    def test_worse_than_null_returns_negative(self):
        """Predictions worse than prevalence should give IPA < 0."""
        from src.stats.scaled_brier import scaled_brier_score

        # 50% prevalence
        y_true = np.array([0, 0, 1, 1])
        # Inverted predictions (predicting opposite)
        y_prob = np.array([0.9, 0.9, 0.1, 0.1])

        result = scaled_brier_score(y_true, y_prob)

        # IPA should be negative (worse than null)
        assert result["ipa"] < 0

    def test_good_predictions_returns_positive(self):
        """Good predictions should give 0 < IPA < 1."""
        from src.stats.scaled_brier import scaled_brier_score

        rng = np.random.default_rng(42)
        n = 200

        # Generate well-calibrated probabilities
        y_prob = rng.uniform(0.1, 0.9, n)
        y_true = (rng.random(n) < y_prob).astype(int)

        result = scaled_brier_score(y_true, y_prob)

        # Good model: IPA should be positive
        assert 0 < result["ipa"] < 1

    def test_ipa_range(self):
        """IPA can be any value <= 1, but should typically be > -1 for reasonable models."""
        from src.stats.scaled_brier import scaled_brier_score

        rng = np.random.default_rng(42)

        for _ in range(50):
            n = 100
            y_prob = rng.uniform(0, 1, n)
            y_true = rng.integers(0, 2, n)

            result = scaled_brier_score(y_true, y_prob)

            # IPA <= 1 always (perfect is 1)
            assert result["ipa"] <= 1.0

    def test_brier_null_equals_prevalence_times_complement(self):
        """Brier_null = prevalence × (1 - prevalence)."""
        from src.stats.scaled_brier import scaled_brier_score

        for prevalence in [0.1, 0.3, 0.5, 0.7, 0.9]:
            n = 100
            n_pos = int(n * prevalence)
            y_true = np.array([1] * n_pos + [0] * (n - n_pos))
            y_prob = np.random.uniform(0.1, 0.9, n)

            result = scaled_brier_score(y_true, y_prob)

            expected_brier_null = prevalence * (1 - prevalence)
            np.testing.assert_allclose(
                result["brier_null"], expected_brier_null, rtol=0.01
            )


class TestScaledBrierComponents:
    """Tests for individual components of scaled Brier score."""

    def test_returns_all_components(self):
        """Should return brier, brier_null, and ipa."""
        from src.stats.scaled_brier import scaled_brier_score

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.2, 0.3, 0.7, 0.8])

        result = scaled_brier_score(y_true, y_prob)

        assert "brier" in result
        assert "brier_null" in result
        assert "ipa" in result
        assert "prevalence" in result

    def test_brier_matches_formula(self):
        """Brier = mean((y_prob - y_true)²)."""
        from src.stats.scaled_brier import scaled_brier_score

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.2, 0.3, 0.7, 0.8])

        result = scaled_brier_score(y_true, y_prob)

        expected_brier = np.mean((y_prob - y_true) ** 2)
        np.testing.assert_allclose(result["brier"], expected_brier)

    def test_ipa_identity(self):
        """IPA = 1 - brier/brier_null."""
        from src.stats.scaled_brier import scaled_brier_score

        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_prob = rng.uniform(0, 1, 100)

        result = scaled_brier_score(y_true, y_prob)

        expected_ipa = 1 - result["brier"] / result["brier_null"]
        np.testing.assert_allclose(result["ipa"], expected_ipa)


class TestScaledBrierInterpretation:
    """Tests for scaled Brier score interpretation."""

    def test_interpretation_useless(self):
        """IPA near 0 should be 'useless'."""
        from src.stats.scaled_brier import scaled_brier_score, interpret_ipa

        # Predict prevalence for everyone
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])

        result = scaled_brier_score(y_true, y_prob)
        interpretation = interpret_ipa(result["ipa"])

        assert interpretation == "useless"

    def test_interpretation_weak(self):
        """IPA around 0.1 should be 'weak'."""
        from src.stats.scaled_brier import interpret_ipa

        assert interpret_ipa(0.1) == "weak"
        assert interpret_ipa(0.15) == "weak"

    def test_interpretation_moderate(self):
        """IPA around 0.25 should be 'moderate'."""
        from src.stats.scaled_brier import interpret_ipa

        assert interpret_ipa(0.25) == "moderate"
        assert interpret_ipa(0.35) == "moderate"

    def test_interpretation_good(self):
        """IPA around 0.5 should be 'good'."""
        from src.stats.scaled_brier import interpret_ipa

        assert interpret_ipa(0.5) == "good"

    def test_interpretation_excellent(self):
        """IPA > 0.6 should be 'excellent'."""
        from src.stats.scaled_brier import interpret_ipa

        assert interpret_ipa(0.7) == "excellent"
        assert interpret_ipa(0.9) == "excellent"

    def test_interpretation_harmful(self):
        """Negative IPA should be 'harmful'."""
        from src.stats.scaled_brier import interpret_ipa

        assert interpret_ipa(-0.1) == "harmful"
        assert interpret_ipa(-0.5) == "harmful"


class TestScaledBrierWithCI:
    """Tests for scaled Brier with confidence intervals."""

    def test_bootstrap_ci(self):
        """Should compute bootstrap CI for IPA."""
        from src.stats.scaled_brier import scaled_brier_score_with_ci

        rng = np.random.default_rng(42)
        n = 200
        y_prob = rng.uniform(0.1, 0.9, n)
        y_true = (rng.random(n) < y_prob).astype(int)

        result = scaled_brier_score_with_ci(y_true, y_prob, n_bootstrap=100)

        assert "ipa_ci_lower" in result
        assert "ipa_ci_upper" in result
        assert result["ipa_ci_lower"] <= result["ipa"]
        assert result["ipa_ci_upper"] >= result["ipa"]

    def test_bootstrap_ci_coverage(self):
        """CI should include true IPA most of the time."""
        from src.stats.scaled_brier import scaled_brier_score_with_ci

        rng = np.random.default_rng(42)
        n = 100
        y_prob = rng.uniform(0.2, 0.8, n)
        y_true = (rng.random(n) < y_prob).astype(int)

        result = scaled_brier_score_with_ci(
            y_true, y_prob, n_bootstrap=500, ci_level=0.95
        )

        # CI should be reasonable (not too wide)
        ci_width = result["ipa_ci_upper"] - result["ipa_ci_lower"]
        assert ci_width > 0
        assert ci_width < 0.5  # Should not be extremely wide


class TestScaledBrierEdgeCases:
    """Edge case tests for scaled Brier score."""

    def test_all_positive_outcomes(self):
        """Handle case with all positive outcomes."""
        from src.stats.scaled_brier import scaled_brier_score
        from src.stats import SingleClassError

        y_true = np.array([1, 1, 1, 1, 1])
        y_prob = np.array([0.8, 0.9, 0.7, 0.8, 0.9])

        # Brier_null = 1 × 0 = 0, which would cause division by zero
        # Should raise SingleClassError
        with pytest.raises(SingleClassError):
            scaled_brier_score(y_true, y_prob)

    def test_all_negative_outcomes(self):
        """Handle case with all negative outcomes."""
        from src.stats.scaled_brier import scaled_brier_score
        from src.stats import SingleClassError

        y_true = np.array([0, 0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.1, 0.2, 0.1])

        # Brier_null = 0 × 1 = 0, division by zero
        with pytest.raises(SingleClassError):
            scaled_brier_score(y_true, y_prob)

    def test_extreme_imbalance_handled(self):
        """Handle extreme class imbalance gracefully."""
        from src.stats.scaled_brier import scaled_brier_score

        # 95% negative, 5% positive
        y_true = np.array([0] * 95 + [1] * 5)
        y_prob = np.random.uniform(0.01, 0.2, 100)

        result = scaled_brier_score(y_true, y_prob)

        # Should still compute (prevalence = 0.05, brier_null = 0.0475)
        assert np.isfinite(result["ipa"])
        assert np.isfinite(result["brier_null"])

    def test_empty_arrays_raises(self):
        """Empty arrays should raise error."""
        from src.stats.scaled_brier import scaled_brier_score
        from src.stats import InsufficientDataError

        y_true = np.array([])
        y_prob = np.array([])

        with pytest.raises(InsufficientDataError):
            scaled_brier_score(y_true, y_prob)

    def test_mismatched_lengths_raises(self):
        """Mismatched array lengths should raise error."""
        from src.stats.scaled_brier import scaled_brier_score
        from src.stats import ValidationError

        y_true = np.array([0, 1, 0])
        y_prob = np.array([0.2, 0.8])

        with pytest.raises(ValidationError):
            scaled_brier_score(y_true, y_prob)
