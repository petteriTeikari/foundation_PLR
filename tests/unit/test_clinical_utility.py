"""
Unit tests for clinical utility metrics (Decision Curve Analysis).

Tests validate:
1. Net benefit computation
2. DCA across threshold ranges
3. Comparison with treat-all/treat-none strategies
4. Glaucoma-specific threshold ranges

Cross-references:
- planning/statistics-implementation.md (Section 2.7)
"""

import numpy as np
import pandas as pd
import pytest


class TestNetBenefit:
    """Tests for net benefit computation."""

    @pytest.fixture
    def classification_data(self):
        """Standard classification data."""
        rng = np.random.default_rng(42)
        n = 200
        prevalence = 0.3

        y_true = rng.binomial(1, prevalence, n)
        # Good model: higher probabilities for positive class
        noise = rng.normal(0, 0.15, n)
        y_prob = np.clip(y_true * 0.7 + (1 - y_true) * 0.3 + noise, 0.01, 0.99)

        return {"y_true": y_true, "y_prob": y_prob}

    def test_net_benefit_perfect_model(self):
        """Perfect model should have NB equal to prevalence at low thresholds."""
        from src.stats.clinical_utility import net_benefit

        # Perfect model: prob=1 for positives, prob=0 for negatives
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

        # At threshold 0.5: TP=4, FP=0, n=8
        # NB = TP/n - FP/n × (pt/(1-pt)) = 0.5 - 0 = 0.5 = prevalence
        nb = net_benefit(y_true, y_prob, threshold=0.5)
        assert np.isclose(nb, 0.5, rtol=0.01)

    def test_net_benefit_useless_model(self):
        """Random model should have low net benefit."""
        from src.stats.clinical_utility import net_benefit

        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.binomial(1, 0.3, n)
        y_prob = rng.uniform(0.0, 1.0, n)  # Random predictions

        # At threshold 0.5, random model has NB near 0
        nb = net_benefit(y_true, y_prob, threshold=0.5)
        # Should be low (could be negative)
        assert nb < 0.2

    def test_net_benefit_treat_all_baseline(self):
        """Net benefit of treat-all strategy."""
        from src.stats.clinical_utility import net_benefit

        y_true = np.array([0, 0, 0, 0, 1, 1])  # 33% prevalence
        y_prob = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Treat all

        # At threshold 0.2: NB = prevalence - (1-prevalence) × (0.2/0.8)
        # = 0.333 - 0.667 × 0.25 = 0.333 - 0.167 = 0.167
        nb = net_benefit(y_true, y_prob, threshold=0.2)
        prevalence = 2 / 6
        expected = prevalence - (1 - prevalence) * (0.2 / 0.8)
        assert np.isclose(nb, expected, rtol=0.01)

    def test_net_benefit_treat_none_zero(self):
        """Net benefit of treat-none strategy should be 0."""
        from src.stats.clinical_utility import net_benefit

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 0.0])  # Treat none

        # NB = 0 (no predictions made)
        nb = net_benefit(y_true, y_prob, threshold=0.5)
        assert nb == 0.0

    def test_net_benefit_can_be_negative(self, classification_data):
        """Net benefit can be negative when FP harm outweighs TP benefit."""
        from src.stats.clinical_utility import net_benefit

        # Create scenario with many false positives and few true positives
        # At higher thresholds where FP cost is high, NB can be negative
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 10% prevalence
        y_prob = np.array(
            [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1]
        )  # Predicts all wrong

        # At threshold 0.5 with high FP rate:
        # TP=0, FP=9, n=10
        # NB = 0/10 - 9/10 × (0.5/0.5) = 0 - 0.9 × 1 = -0.9
        nb = net_benefit(y_true, y_prob, threshold=0.5)
        assert nb < 0


class TestDecisionCurveAnalysis:
    """Tests for Decision Curve Analysis."""

    @pytest.fixture
    def classification_data(self):
        """Standard classification data."""
        rng = np.random.default_rng(42)
        n = 200
        prevalence = 0.3

        y_true = rng.binomial(1, prevalence, n)
        noise = rng.normal(0, 0.15, n)
        y_prob = np.clip(y_true * 0.7 + (1 - y_true) * 0.3 + noise, 0.01, 0.99)

        return {"y_true": y_true, "y_prob": y_prob}

    def test_dca_returns_dataframe(self, classification_data):
        """DCA should return a pandas DataFrame."""
        from src.stats.clinical_utility import decision_curve_analysis

        result = decision_curve_analysis(
            classification_data["y_true"], classification_data["y_prob"]
        )
        assert isinstance(result, pd.DataFrame)

    def test_dca_has_required_columns(self, classification_data):
        """DCA DataFrame should have required columns."""
        from src.stats.clinical_utility import decision_curve_analysis

        result = decision_curve_analysis(
            classification_data["y_true"], classification_data["y_prob"]
        )

        required_columns = ["threshold", "nb_model", "nb_all", "nb_none"]
        for col in required_columns:
            assert col in result.columns

    def test_dca_threshold_range_glaucoma(self, classification_data):
        """Default threshold range should be 1-30% for glaucoma."""
        from src.stats.clinical_utility import decision_curve_analysis

        result = decision_curve_analysis(
            classification_data["y_true"], classification_data["y_prob"]
        )

        # Check threshold range
        assert result["threshold"].min() >= 0.01
        assert result["threshold"].max() <= 0.30

    def test_dca_model_above_treat_all_for_good_model(self, classification_data):
        """Good model should have NB above treat-all at some thresholds."""
        from src.stats.clinical_utility import decision_curve_analysis

        result = decision_curve_analysis(
            classification_data["y_true"], classification_data["y_prob"]
        )

        # At some thresholds, model should beat treat-all
        model_beats_treat_all = (result["nb_model"] > result["nb_all"]).any()
        assert model_beats_treat_all

    def test_dca_treat_none_always_zero(self, classification_data):
        """Treat-none strategy should always have NB = 0."""
        from src.stats.clinical_utility import decision_curve_analysis

        result = decision_curve_analysis(
            classification_data["y_true"], classification_data["y_prob"]
        )

        assert (result["nb_none"] == 0).all()

    def test_dca_custom_threshold_range(self, classification_data):
        """Custom threshold range should work."""
        from src.stats.clinical_utility import decision_curve_analysis

        result = decision_curve_analysis(
            classification_data["y_true"],
            classification_data["y_prob"],
            threshold_range=(0.05, 0.50),
        )

        assert result["threshold"].min() >= 0.05
        assert result["threshold"].max() <= 0.50


class TestStandardizedNetBenefit:
    """Tests for standardized net benefit."""

    def test_snb_range_0_to_1(self):
        """Standardized NB should be in [0, 1]."""
        from src.stats.clinical_utility import standardized_net_benefit

        rng = np.random.default_rng(42)
        y_true = rng.binomial(1, 0.3, 100)
        y_prob = np.clip(
            y_true * 0.7 + (1 - y_true) * 0.3 + rng.normal(0, 0.1, 100), 0.01, 0.99
        )

        snb = standardized_net_benefit(y_true, y_prob, threshold=0.2)
        assert 0.0 <= snb <= 1.0


class TestOptimalThreshold:
    """Tests for optimal threshold computation."""

    def test_optimal_threshold_equal_costs(self):
        """Equal costs should give threshold = 0.5."""
        from src.stats.clinical_utility import optimal_threshold_cost_sensitive

        threshold = optimal_threshold_cost_sensitive(cost_fp=1.0, cost_fn=1.0)
        assert np.isclose(threshold, 0.5)

    def test_optimal_threshold_higher_fn_cost(self):
        """Higher FN cost should give lower threshold."""
        from src.stats.clinical_utility import optimal_threshold_cost_sensitive

        # For glaucoma: missing disease (FN) is worse than false alarm (FP)
        threshold = optimal_threshold_cost_sensitive(cost_fp=1.0, cost_fn=5.0)
        # p_t = C_FP / (C_FP + C_FN) = 1 / 6 = 0.167
        expected = 1.0 / 6.0
        assert np.isclose(threshold, expected, rtol=0.01)
        assert threshold < 0.5

    def test_optimal_threshold_higher_fp_cost(self):
        """Higher FP cost should give higher threshold."""
        from src.stats.clinical_utility import optimal_threshold_cost_sensitive

        threshold = optimal_threshold_cost_sensitive(cost_fp=5.0, cost_fn=1.0)
        # p_t = 5 / 6 = 0.833
        assert threshold > 0.5


class TestClinicalUtilityIntegration:
    """Integration tests for clinical utility metrics."""

    def test_dca_consistent_with_net_benefit(self):
        """DCA results should match individual net_benefit calculations."""
        from src.stats.clinical_utility import decision_curve_analysis, net_benefit

        rng = np.random.default_rng(42)
        y_true = rng.binomial(1, 0.3, 100)
        y_prob = np.clip(y_true * 0.7 + rng.normal(0, 0.15, 100), 0.01, 0.99)

        dca_result = decision_curve_analysis(y_true, y_prob, n_thresholds=5)

        # Check first threshold matches
        first_threshold = dca_result["threshold"].iloc[0]
        expected_nb = net_benefit(y_true, y_prob, first_threshold)

        np.testing.assert_allclose(
            dca_result["nb_model"].iloc[0], expected_nb, rtol=0.01
        )
