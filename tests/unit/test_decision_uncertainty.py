"""
Unit tests for Decision Uncertainty (DU) metric.

Tests validate:
1. DU computation from bootstrap samples
2. DU at different thresholds
3. Edge cases (certainty, maximum uncertainty)
4. Per-subject DU computation

Cross-references:
- planning/remaining-duckdb-stats-viz-tasks-plan.md (Appendix E.1)

References:
- Barrenada et al. (2025). The fundamental problem of providing uncertainty
  in individual risk predictions using clinical prediction models. BMJ Medicine.
"""

import numpy as np
import pytest


class TestDecisionUncertainty:
    """Tests for Decision Uncertainty computation."""

    def test_certain_above_threshold_returns_zero(self):
        """If all bootstrap samples above threshold, DU = 0."""
        from src.stats.decision_uncertainty import decision_uncertainty

        # All samples well above threshold
        bootstrap_samples = np.array([0.8, 0.85, 0.9, 0.82, 0.88])
        threshold = 0.5

        du = decision_uncertainty(bootstrap_samples, threshold)

        # 100% above, 0% below → DU = min(1.0, 0.0) = 0
        assert du == 0.0

    def test_certain_below_threshold_returns_zero(self):
        """If all bootstrap samples below threshold, DU = 0."""
        from src.stats.decision_uncertainty import decision_uncertainty

        # All samples well below threshold
        bootstrap_samples = np.array([0.1, 0.15, 0.2, 0.12, 0.18])
        threshold = 0.5

        du = decision_uncertainty(bootstrap_samples, threshold)

        # 0% above, 100% below → DU = min(0.0, 1.0) = 0
        assert du == 0.0

    def test_maximum_uncertainty_returns_half(self):
        """If 50% above and 50% below threshold, DU = 0.5."""
        from src.stats.decision_uncertainty import decision_uncertainty

        # Half above, half below threshold
        bootstrap_samples = np.array([0.3, 0.4, 0.6, 0.7])
        threshold = 0.5

        du = decision_uncertainty(bootstrap_samples, threshold)

        # 50% above, 50% below → DU = min(0.5, 0.5) = 0.5
        assert du == 0.5

    def test_du_range_is_0_to_half(self):
        """DU should always be in [0, 0.5]."""
        from src.stats.decision_uncertainty import decision_uncertainty

        rng = np.random.default_rng(42)

        for _ in range(100):
            bootstrap_samples = rng.uniform(0, 1, 100)
            threshold = rng.uniform(0.1, 0.9)

            du = decision_uncertainty(bootstrap_samples, threshold)

            assert 0 <= du <= 0.5

    def test_higher_spread_gives_higher_du(self):
        """Bootstrap samples with more spread should have higher DU."""
        from src.stats.decision_uncertainty import decision_uncertainty

        threshold = 0.5

        # Tight distribution around 0.7 (low uncertainty)
        tight_samples = np.array([0.68, 0.69, 0.70, 0.71, 0.72])
        du_tight = decision_uncertainty(tight_samples, threshold)

        # Wide distribution spanning threshold (high uncertainty)
        wide_samples = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        du_wide = decision_uncertainty(wide_samples, threshold)

        assert du_wide > du_tight

    def test_threshold_affects_du(self):
        """DU should change with different thresholds."""
        from src.stats.decision_uncertainty import decision_uncertainty

        # Samples centered around 0.6
        bootstrap_samples = np.array([0.55, 0.58, 0.60, 0.62, 0.65])

        # Low threshold: all above → DU ≈ 0
        du_low_thresh = decision_uncertainty(bootstrap_samples, 0.3)

        # Threshold near mean: high uncertainty
        du_mid_thresh = decision_uncertainty(bootstrap_samples, 0.6)

        # High threshold: all below → DU ≈ 0
        du_high_thresh = decision_uncertainty(bootstrap_samples, 0.9)

        assert du_low_thresh < du_mid_thresh
        assert du_high_thresh < du_mid_thresh


class TestDecisionUncertaintyPerSubject:
    """Tests for per-subject DU computation."""

    def test_per_subject_du_shape(self):
        """Per-subject DU should have shape (n_subjects,)."""
        from src.stats.decision_uncertainty import decision_uncertainty_per_subject

        n_subjects = 50
        n_bootstrap = 100

        # Shape: (n_subjects, n_bootstrap)
        bootstrap_matrix = np.random.uniform(0, 1, (n_subjects, n_bootstrap))
        threshold = 0.5

        du_per_subject = decision_uncertainty_per_subject(bootstrap_matrix, threshold)

        assert du_per_subject.shape == (n_subjects,)

    def test_per_subject_du_range(self):
        """All per-subject DU values should be in [0, 0.5]."""
        from src.stats.decision_uncertainty import decision_uncertainty_per_subject

        n_subjects = 100
        n_bootstrap = 50

        bootstrap_matrix = np.random.uniform(0, 1, (n_subjects, n_bootstrap))
        threshold = 0.5

        du_per_subject = decision_uncertainty_per_subject(bootstrap_matrix, threshold)

        assert np.all(du_per_subject >= 0)
        assert np.all(du_per_subject <= 0.5)

    def test_subjects_with_different_certainty(self):
        """Different subjects should have different DU based on their bootstrap spread."""
        from src.stats.decision_uncertainty import decision_uncertainty_per_subject

        n_bootstrap = 100
        threshold = 0.5

        # Subject 1: certain (all above threshold)
        subject1_samples = np.random.uniform(0.7, 0.9, n_bootstrap)

        # Subject 2: uncertain (spans threshold)
        subject2_samples = np.random.uniform(0.3, 0.7, n_bootstrap)

        # Subject 3: certain (all below threshold)
        subject3_samples = np.random.uniform(0.1, 0.3, n_bootstrap)

        bootstrap_matrix = np.vstack(
            [subject1_samples, subject2_samples, subject3_samples]
        )

        du_per_subject = decision_uncertainty_per_subject(bootstrap_matrix, threshold)

        # Subject 2 should have highest DU
        assert du_per_subject[1] > du_per_subject[0]
        assert du_per_subject[1] > du_per_subject[2]


class TestDecisionUncertaintySummary:
    """Tests for DU summary statistics."""

    def test_summary_stats_computation(self):
        """Should compute mean, median, and proportion above threshold."""
        from src.stats.decision_uncertainty import decision_uncertainty_summary

        n_subjects = 100
        n_bootstrap = 50

        rng = np.random.default_rng(42)
        bootstrap_matrix = rng.uniform(0, 1, (n_subjects, n_bootstrap))
        threshold = 0.5
        du_threshold = 0.3  # Threshold for "high uncertainty"

        summary = decision_uncertainty_summary(
            bootstrap_matrix, threshold, du_threshold=du_threshold
        )

        assert "mean_du" in summary
        assert "median_du" in summary
        assert "pct_above_threshold" in summary
        assert 0 <= summary["mean_du"] <= 0.5
        assert 0 <= summary["pct_above_threshold"] <= 100


class TestDecisionUncertaintyEdgeCases:
    """Edge case tests for DU computation."""

    def test_single_bootstrap_sample(self):
        """DU with single sample should be 0 (certain)."""
        from src.stats.decision_uncertainty import decision_uncertainty

        bootstrap_samples = np.array([0.6])
        threshold = 0.5

        du = decision_uncertainty(bootstrap_samples, threshold)

        # Single sample: either all above or all below
        assert du == 0.0

    def test_samples_exactly_at_threshold(self):
        """Samples exactly at threshold should be handled."""
        from src.stats.decision_uncertainty import decision_uncertainty

        # Some samples exactly at threshold
        bootstrap_samples = np.array([0.5, 0.5, 0.5, 0.6, 0.4])
        threshold = 0.5

        du = decision_uncertainty(bootstrap_samples, threshold)

        # Should not crash, DU should be defined
        assert 0 <= du <= 0.5

    def test_empty_samples_raises(self):
        """Empty bootstrap samples should raise error."""
        from src.stats.decision_uncertainty import decision_uncertainty
        from src.stats import ValidationError

        bootstrap_samples = np.array([])
        threshold = 0.5

        with pytest.raises(ValidationError):
            decision_uncertainty(bootstrap_samples, threshold)
