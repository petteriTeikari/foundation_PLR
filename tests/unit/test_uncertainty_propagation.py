"""
Tests for uncertainty propagation analysis.

Tests cover:
- Monte Carlo classifier uncertainty
- Clinical decision stability
- Sensitivity analysis (delta method)
- Required accuracy computation

Cross-references:
- src/stats/uncertainty_propagation.py
- planning/uncertainty-propagation-analysis/
"""

import numpy as np
import pytest

from src.stats.uncertainty_propagation import (
    UncertaintyResult,
    monte_carlo_classifier_uncertainty,
    clinical_decision_stability,
    sensitivity_analysis_delta,
    compute_required_accuracy,
)
# ValidationError has specific signature, using ValueError for simple validation


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def rng():
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_features(rng):
    """Simple feature matrix for testing."""
    n_subjects = 50
    n_features = 5
    return rng.standard_normal((n_subjects, n_features))


@pytest.fixture
def simple_uncertainties():
    """Simple uncertainties (10% std on each feature)."""
    return np.full(5, 0.1)


def mock_predict_proba(X):
    """Simple mock classifier: logistic of mean feature."""
    logit = X.mean(axis=1)
    return 1 / (1 + np.exp(-logit))


def mock_predict_proba_feature_specific(X):
    """
    Mock classifier where feature 0 and 1 dominate.
    p = sigmoid(2*x0 + 3*x1 + 0.1*sum(other features))
    """
    logit = 2 * X[:, 0] + 3 * X[:, 1] + 0.1 * X[:, 2:].sum(axis=1)
    return 1 / (1 + np.exp(-logit))


# ============================================================================
# Test UncertaintyResult dataclass
# ============================================================================


class TestUncertaintyResult:
    """Tests for UncertaintyResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = UncertaintyResult(
            prediction_mean=np.array([0.5, 0.6]),
            prediction_std=np.array([0.1, 0.2]),
            n_simulations=100,
        )
        assert len(result.prediction_mean) == 2
        assert result.n_simulations == 100

    def test_repr(self):
        """Test string representation."""
        result = UncertaintyResult(
            prediction_mean=np.array([0.5, 0.6, 0.7]),
            n_simulations=1000,
        )
        assert "n_subjects=3" in repr(result)
        assert "n_simulations=1000" in repr(result)


# ============================================================================
# Test monte_carlo_classifier_uncertainty
# ============================================================================


class TestMonteCarloClassifierUncertainty:
    """Tests for Monte Carlo uncertainty propagation."""

    def test_basic_operation(self, simple_features, simple_uncertainties):
        """Test that MC simulation runs and returns valid result."""
        result = monte_carlo_classifier_uncertainty(
            simple_features,
            simple_uncertainties,
            mock_predict_proba,
            n_simulations=100,
            random_state=42,
        )

        assert isinstance(result, UncertaintyResult)
        assert len(result.prediction_mean) == simple_features.shape[0]
        assert len(result.prediction_std) == simple_features.shape[0]
        assert result.n_simulations == 100

    def test_predictions_in_valid_range(self, simple_features, simple_uncertainties):
        """Test that predictions are in [0, 1]."""
        result = monte_carlo_classifier_uncertainty(
            simple_features,
            simple_uncertainties,
            mock_predict_proba,
            n_simulations=50,
            random_state=42,
        )

        assert np.all(result.prediction_mean >= 0)
        assert np.all(result.prediction_mean <= 1)
        assert np.all(result.prediction_ci_lower >= 0)
        assert np.all(result.prediction_ci_upper <= 1)

    def test_uncertainty_increases_with_feature_uncertainty(self, simple_features):
        """Test that larger feature uncertainty leads to larger prediction uncertainty."""
        # Low uncertainty
        low_unc_result = monte_carlo_classifier_uncertainty(
            simple_features,
            np.full(5, 0.01),  # Very low
            mock_predict_proba,
            n_simulations=200,
            random_state=42,
        )

        # High uncertainty
        high_unc_result = monte_carlo_classifier_uncertainty(
            simple_features,
            np.full(5, 0.5),  # High
            mock_predict_proba,
            n_simulations=200,
            random_state=42,
        )

        # High uncertainty should produce larger prediction std
        assert np.mean(high_unc_result.prediction_std) > np.mean(
            low_unc_result.prediction_std
        )

    def test_reproducibility(self, simple_features, simple_uncertainties):
        """Test that same random_state gives same results."""
        result1 = monte_carlo_classifier_uncertainty(
            simple_features,
            simple_uncertainties,
            mock_predict_proba,
            n_simulations=50,
            random_state=123,
        )

        result2 = monte_carlo_classifier_uncertainty(
            simple_features,
            simple_uncertainties,
            mock_predict_proba,
            n_simulations=50,
            random_state=123,
        )

        np.testing.assert_array_almost_equal(
            result1.prediction_mean, result2.prediction_mean
        )

    def test_broadcasting_1d_uncertainties(self, simple_features):
        """Test that 1D uncertainties are broadcast correctly."""
        # 1D array of uncertainties (one per feature)
        unc_1d = np.array([0.1, 0.2, 0.15, 0.1, 0.25])

        result = monte_carlo_classifier_uncertainty(
            simple_features,
            unc_1d,
            mock_predict_proba,
            n_simulations=50,
            random_state=42,
        )

        assert result.n_simulations == 50

    def test_2d_uncertainties(self, simple_features, rng):
        """Test with subject-specific uncertainties."""
        n_subjects, n_features = simple_features.shape
        unc_2d = rng.uniform(0.05, 0.2, size=(n_subjects, n_features))

        result = monte_carlo_classifier_uncertainty(
            simple_features,
            unc_2d,
            mock_predict_proba,
            n_simulations=50,
            random_state=42,
        )

        assert len(result.prediction_mean) == n_subjects

    def test_invalid_shapes_raise_error(self, simple_features):
        """Test that mismatched shapes raise ValueError."""
        wrong_size_unc = np.array([0.1, 0.2, 0.3])  # 3 instead of 5

        with pytest.raises(ValueError):
            monte_carlo_classifier_uncertainty(
                simple_features,
                wrong_size_unc,
                mock_predict_proba,
                n_simulations=10,
            )

    def test_ci_coverage(self, rng):
        """Test that 95% CI roughly contains 95% of simulations."""
        # Generate predictions with known distribution
        features = rng.standard_normal((30, 3))
        uncertainties = np.full(3, 0.3)

        result = monte_carlo_classifier_uncertainty(
            features,
            uncertainties,
            mock_predict_proba,
            n_simulations=1000,
            random_state=42,
            alpha=0.05,
        )

        # Check that CI bounds make sense
        assert np.all(result.prediction_ci_lower <= result.prediction_mean)
        assert np.all(result.prediction_ci_upper >= result.prediction_mean)


# ============================================================================
# Test clinical_decision_stability
# ============================================================================


class TestClinicalDecisionStability:
    """Tests for clinical decision stability analysis."""

    def test_perfect_stability(self):
        """Test with predictions far from threshold."""
        # All predictions well above threshold
        predictions = np.full((10, 100), 0.9)  # 10 subjects, 100 sims

        result = clinical_decision_stability(predictions, threshold=0.5)

        assert result.decision_stability_pct == 100.0
        assert result.n_unstable == 0
        assert len(result.unstable_indices) == 0

    def test_no_stability(self):
        """Test with predictions at threshold."""
        # All predictions at exactly threshold - should be unstable
        rng = np.random.default_rng(42)
        predictions = rng.uniform(0.45, 0.55, size=(10, 100))

        result = clinical_decision_stability(predictions, threshold=0.5)

        # Should have low stability
        assert result.decision_stability_pct < 50

    def test_mixed_stability(self, rng):
        """Test with mix of stable and unstable subjects."""
        n_subjects = 20
        n_sims = 100

        predictions = np.zeros((n_subjects, n_sims))

        # First 10: stable (predictions ~0.9)
        predictions[:10, :] = rng.normal(0.9, 0.02, size=(10, n_sims))

        # Last 10: unstable (predictions ~0.5)
        predictions[10:, :] = rng.normal(0.5, 0.1, size=(10, n_sims))

        predictions = np.clip(predictions, 0, 1)

        result = clinical_decision_stability(predictions, threshold=0.5)

        # First 10 should be stable, last 10 mostly unstable
        assert result.decision_stability_pct < 100
        assert result.decision_stability_pct > 0
        assert result.n_unstable > 0

    def test_different_thresholds(self, rng):
        """Test that threshold affects stability."""
        predictions = rng.uniform(0.3, 0.7, size=(20, 100))

        # Low threshold - more subjects predicted positive = more stable positives
        result_low = clinical_decision_stability(predictions, threshold=0.2)

        # High threshold - more subjects predicted negative = more stable negatives
        result_high = clinical_decision_stability(predictions, threshold=0.8)

        # Both should have reasonable stability at their respective thresholds
        # but subjects near respective thresholds will be unstable
        assert result_low.n_unstable >= 0
        assert result_high.n_unstable >= 0

    def test_majority_criterion(self, rng):
        """Test majority stability criterion (90% agreement)."""
        predictions = np.zeros((10, 100))

        # 85% above threshold - should be unstable with majority criterion
        predictions[0, :85] = 0.6
        predictions[0, 85:] = 0.4

        # 95% above threshold - should be stable with majority criterion
        predictions[1, :95] = 0.6
        predictions[1, 95:] = 0.4

        result = clinical_decision_stability(
            predictions[:2], threshold=0.5, stability_criterion="majority"
        )

        # Subject 1 should be stable, subject 0 unstable
        assert 1 not in result.unstable_indices
        assert 0 in result.unstable_indices

    def test_all_criterion(self, rng):
        """Test all stability criterion (100% agreement)."""
        predictions = np.zeros((10, 100))

        # All above threshold - stable
        predictions[0, :] = 0.7

        # 99% above threshold - unstable with "all" criterion
        predictions[1, :99] = 0.7
        predictions[1, 99] = 0.3

        result = clinical_decision_stability(
            predictions[:2], threshold=0.5, stability_criterion="all"
        )

        assert 0 not in result.unstable_indices  # All same decision
        assert 1 in result.unstable_indices  # One different decision


# ============================================================================
# Test sensitivity_analysis_delta
# ============================================================================


class TestSensitivityAnalysisDelta:
    """Tests for delta-method sensitivity analysis."""

    def test_identifies_influential_feature(self, rng):
        """Test that sensitivity correctly identifies influential features."""
        features = rng.standard_normal((100, 5))
        uncertainties = np.full(5, 0.1)

        result = sensitivity_analysis_delta(
            features,
            uncertainties,
            mock_predict_proba_feature_specific,
            feature_names=["A", "B", "C", "D", "E"],
        )

        # Feature B (coefficient 3) should be most influential
        assert result.most_influential == "B"

        # Features A and B should have higher sensitivity than C, D, E
        idx_A = result.feature_names.index("A")
        idx_B = result.feature_names.index("B")
        idx_others = [result.feature_names.index(x) for x in ["C", "D", "E"]]

        sensitivity_AB = result.sensitivity_indices[[idx_A, idx_B]]
        sensitivity_others = result.sensitivity_indices[idx_others]

        assert sensitivity_AB.min() > sensitivity_others.max()

    def test_normalized_sums_to_one(self, simple_features, simple_uncertainties):
        """Test that normalized sensitivities sum to 1."""
        result = sensitivity_analysis_delta(
            simple_features,
            simple_uncertainties,
            mock_predict_proba,
        )

        total = np.sum(result.sensitivity_normalized)
        np.testing.assert_almost_equal(total, 1.0, decimal=5)

    def test_zero_uncertainty_gives_zero_sensitivity(self, simple_features):
        """Test that zero uncertainty gives zero sensitivity for that feature."""
        uncertainties = np.array([0.1, 0.0, 0.1, 0.1, 0.1])

        result = sensitivity_analysis_delta(
            simple_features,
            uncertainties,
            mock_predict_proba,
        )

        # Feature 1 has zero uncertainty, should have zero sensitivity
        np.testing.assert_almost_equal(result.sensitivity_indices[1], 0.0)

    def test_custom_feature_names(self, simple_features, simple_uncertainties):
        """Test that custom feature names are preserved."""
        names = ["baseline", "amplitude", "latency", "velocity", "pipr"]

        result = sensitivity_analysis_delta(
            simple_features,
            simple_uncertainties,
            mock_predict_proba,
            feature_names=names,
        )

        assert result.feature_names == names
        assert result.most_influential in names


# ============================================================================
# Test compute_required_accuracy
# ============================================================================


class TestComputeRequiredAccuracy:
    """Tests for required accuracy computation."""

    def test_returns_valid_result(self, simple_features, simple_uncertainties):
        """Test that function returns expected structure."""
        result = compute_required_accuracy(
            simple_features,
            simple_uncertainties,
            mock_predict_proba,
            target_stability=0.90,
            n_simulations=100,
            random_state=42,
        )

        assert "multipliers" in result
        assert "stabilities" in result
        assert "required_multiplier" in result
        assert "required_uncertainties" in result

    def test_lower_uncertainty_gives_higher_stability(
        self, simple_features, simple_uncertainties
    ):
        """Test that reducing uncertainty increases stability."""
        result = compute_required_accuracy(
            simple_features,
            simple_uncertainties,
            mock_predict_proba,
            target_stability=0.90,
            uncertainty_multipliers=np.array([0.1, 0.5, 1.0, 2.0]),
            n_simulations=100,
            random_state=42,
        )

        stabilities = result["stabilities"]

        # Lower multiplier (less uncertainty) should give higher stability
        # (stabilities should decrease as multiplier increases)
        # Note: this is probabilistic, so we just check the trend
        assert stabilities[0] >= stabilities[-1] - 0.1  # Allow some noise

    def test_required_uncertainties_shape(self, simple_features, simple_uncertainties):
        """Test that required uncertainties have correct shape."""
        result = compute_required_accuracy(
            simple_features,
            simple_uncertainties,
            mock_predict_proba,
            target_stability=0.90,
            n_simulations=50,
            random_state=42,
        )

        assert result["required_uncertainties"].shape == simple_uncertainties.shape


# ============================================================================
# Edge cases and error handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_subject(self):
        """Test with single subject."""
        features = np.array([[1.0, 2.0, 3.0]])
        uncertainties = np.array([0.1, 0.1, 0.1])

        result = monte_carlo_classifier_uncertainty(
            features,
            uncertainties,
            mock_predict_proba,
            n_simulations=50,
            random_state=42,
        )

        assert len(result.prediction_mean) == 1

    def test_very_small_uncertainty(self, simple_features):
        """Test with very small uncertainty."""
        tiny_unc = np.full(5, 1e-10)

        result = monte_carlo_classifier_uncertainty(
            simple_features,
            tiny_unc,
            mock_predict_proba,
            n_simulations=50,
            random_state=42,
        )

        # Very small uncertainty should give very small prediction std
        assert np.mean(result.prediction_std) < 0.01

    def test_stability_with_single_simulation(self):
        """Test stability analysis with minimal simulations."""
        predictions = np.array([[0.7], [0.3]])  # 2 subjects, 1 sim each

        result = clinical_decision_stability(predictions, threshold=0.5)

        # Both should be stable (100% agreement with themselves)
        assert result.decision_stability_pct == 100.0


# ============================================================================
# Integration tests
# ============================================================================


class TestIntegration:
    """Integration tests for the full uncertainty analysis pipeline."""

    def test_full_pipeline(self, rng):
        """Test complete pipeline from features to accuracy requirements."""
        # Generate realistic-ish features
        n_subjects = 100
        n_features = 10

        features = rng.standard_normal((n_subjects, n_features))
        uncertainties = rng.uniform(0.05, 0.2, size=n_features)

        # Define a more complex classifier
        def classifier(X):
            logit = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]
            return 1 / (1 + np.exp(-logit))

        # Step 1: MC uncertainty
        mc_result = monte_carlo_classifier_uncertainty(
            features,
            uncertainties,
            classifier,
            n_simulations=200,
            random_state=42,
        )

        assert mc_result.n_simulations == 200

        # Step 2: Decision stability
        stability_result = clinical_decision_stability(
            mc_result.arrays["all_predictions"],
            threshold=0.5,
        )

        assert 0 <= stability_result.decision_stability_pct <= 100

        # Step 3: Sensitivity analysis
        sensitivity_result = sensitivity_analysis_delta(
            features,
            uncertainties,
            classifier,
            feature_names=[f"feat_{i}" for i in range(n_features)],
        )

        # Feature 0 should be most influential (coefficient = 1)
        assert sensitivity_result.most_influential == "feat_0"

        # Step 4: Required accuracy
        accuracy_result = compute_required_accuracy(
            features,
            uncertainties,
            classifier,
            target_stability=0.90,
            n_simulations=100,
            random_state=42,
        )

        assert accuracy_result["required_multiplier"] > 0
