"""
Unit tests for multiple comparison correction methods.

Tests validate:
1. Exact match with statsmodels implementation
2. Edge cases (all significant, none significant, ties)
3. Correct ordering preservation
4. Comparison between methods

Cross-references:
- planning/statistics-implementation.md (Section 3.4)
"""

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests


class TestBenjaminiHochberg:
    """Tests for Benjamini-Hochberg FDR correction."""

    def test_matches_statsmodels_exact(self):
        """Results should exactly match statsmodels implementation."""
        from src.stats.fdr_correction import benjamini_hochberg

        p_values = np.array([0.001, 0.01, 0.02, 0.04, 0.05, 0.1, 0.5])

        # Our implementation
        result = benjamini_hochberg(p_values)

        # statsmodels reference
        reject_ref, pvals_ref, _, _ = multipletests(
            p_values, alpha=0.05, method="fdr_bh"
        )

        np.testing.assert_array_equal(result.reject, reject_ref)
        np.testing.assert_allclose(result.p_adjusted, pvals_ref, rtol=1e-10)

    def test_all_significant(self):
        """All p-values below threshold should remain significant."""
        from src.stats.fdr_correction import benjamini_hochberg

        p_values = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
        result = benjamini_hochberg(p_values, alpha=0.05)

        assert np.all(result.reject)
        assert result.n_rejected == 5

    def test_none_significant(self):
        """All p-values above threshold should remain non-significant."""
        from src.stats.fdr_correction import benjamini_hochberg

        p_values = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        result = benjamini_hochberg(p_values, alpha=0.05)

        assert not np.any(result.reject)
        assert result.n_rejected == 0

    def test_some_significant(self):
        """Partial significance should be handled correctly."""
        from src.stats.fdr_correction import benjamini_hochberg

        p_values = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
        result = benjamini_hochberg(p_values, alpha=0.05)

        # First two should definitely be significant
        assert result.reject[0] and result.reject[1]
        # Last one should not be significant
        assert not result.reject[4]

    def test_ordering_preserved(self):
        """Output arrays should maintain input order."""
        from src.stats.fdr_correction import benjamini_hochberg

        # Unsorted p-values
        p_values = np.array([0.5, 0.001, 0.1, 0.01, 0.05])
        result = benjamini_hochberg(p_values)

        # Check that original order is preserved
        assert result.p_values[0] == 0.5
        assert result.p_values[1] == 0.001
        assert result.p_values[2] == 0.1

        # Smallest p-value (index 1) should be significant
        assert result.reject[1]
        # Largest p-value (index 0) should not be significant
        assert not result.reject[0]

    def test_alpha_0_01(self):
        """Different alpha levels should work correctly."""
        from src.stats.fdr_correction import benjamini_hochberg

        p_values = np.array([0.001, 0.005, 0.01, 0.02, 0.05])

        result_05 = benjamini_hochberg(p_values, alpha=0.05)
        result_01 = benjamini_hochberg(p_values, alpha=0.01)

        # More rejections at α=0.05 than α=0.01
        assert result_05.n_rejected >= result_01.n_rejected

    def test_single_pvalue(self):
        """Single p-value should work correctly."""
        from src.stats.fdr_correction import benjamini_hochberg

        # Significant single p-value
        result1 = benjamini_hochberg(np.array([0.01]), alpha=0.05)
        assert result1.reject[0]
        assert result1.p_adjusted[0] == 0.01  # No adjustment needed

        # Non-significant single p-value
        result2 = benjamini_hochberg(np.array([0.1]), alpha=0.05)
        assert not result2.reject[0]

    def test_ties_in_pvalues(self):
        """Tied p-values should be handled correctly."""
        from src.stats.fdr_correction import benjamini_hochberg

        p_values = np.array([0.01, 0.01, 0.01, 0.5, 0.5])
        result = benjamini_hochberg(p_values)

        # All 0.01 values should have same adjusted p and rejection status
        assert result.reject[0] == result.reject[1] == result.reject[2]
        # All 0.5 values should have same status
        assert result.reject[3] == result.reject[4]

    def test_empty_array_raises(self):
        """Empty input should raise error."""
        from src.stats.fdr_correction import benjamini_hochberg
        from src.stats import ValidationError

        with pytest.raises((ValidationError, ValueError)):
            benjamini_hochberg(np.array([]))

    def test_invalid_pvalues_raises(self):
        """P-values outside [0,1] should raise error."""
        from src.stats.fdr_correction import benjamini_hochberg
        from src.stats import ValidationError

        with pytest.raises((ValidationError, ValueError)):
            benjamini_hochberg(np.array([0.01, -0.1, 0.5]))

        with pytest.raises((ValidationError, ValueError)):
            benjamini_hochberg(np.array([0.01, 1.5, 0.5]))


class TestBonferroni:
    """Tests for Bonferroni correction."""

    def test_more_conservative_than_fdr(self):
        """Bonferroni should reject fewer hypotheses than FDR."""
        from src.stats.fdr_correction import benjamini_hochberg, bonferroni

        p_values = np.array([0.001, 0.005, 0.01, 0.02, 0.04])

        fdr_result = benjamini_hochberg(p_values, alpha=0.05)
        bonf_result = bonferroni(p_values, alpha=0.05)

        # Bonferroni is more conservative
        assert bonf_result.n_rejected <= fdr_result.n_rejected

    def test_exact_correction(self):
        """Bonferroni adjustment: p_adj = p × m."""
        from src.stats.fdr_correction import bonferroni

        p_values = np.array([0.01, 0.02, 0.03])
        result = bonferroni(p_values)

        # p_adj = p × 3 (number of tests)
        expected = np.minimum(p_values * 3, 1.0)
        np.testing.assert_allclose(result.p_adjusted, expected, rtol=1e-10)

    def test_matches_statsmodels(self):
        """Should match statsmodels Bonferroni implementation."""
        from src.stats.fdr_correction import bonferroni

        p_values = np.array([0.001, 0.01, 0.05, 0.1, 0.5])

        result = bonferroni(p_values, alpha=0.05)
        reject_ref, pvals_ref, _, _ = multipletests(
            p_values, alpha=0.05, method="bonferroni"
        )

        np.testing.assert_array_equal(result.reject, reject_ref)
        np.testing.assert_allclose(result.p_adjusted, pvals_ref, rtol=1e-10)


class TestHolm:
    """Tests for Holm-Bonferroni step-down procedure."""

    def test_step_down_procedure(self):
        """Holm should apply step-down correction."""
        from src.stats.fdr_correction import holm

        p_values = np.array([0.001, 0.01, 0.03, 0.05, 0.1])
        result = holm(p_values, alpha=0.05)

        # Should match statsmodels
        reject_ref, pvals_ref, _, _ = multipletests(p_values, alpha=0.05, method="holm")

        np.testing.assert_array_equal(result.reject, reject_ref)
        np.testing.assert_allclose(result.p_adjusted, pvals_ref, rtol=1e-10)

    def test_less_conservative_than_bonferroni(self):
        """Holm should reject at least as many as Bonferroni."""
        from src.stats.fdr_correction import bonferroni, holm

        p_values = np.array([0.001, 0.005, 0.01, 0.02, 0.04])

        bonf_result = bonferroni(p_values, alpha=0.05)
        holm_result = holm(p_values, alpha=0.05)

        # Holm is uniformly more powerful than Bonferroni
        assert holm_result.n_rejected >= bonf_result.n_rejected


class TestMethodComparison:
    """Compare different correction methods."""

    def test_power_ordering(self):
        """FDR > Holm > Bonferroni in rejections (generally)."""
        from src.stats.fdr_correction import benjamini_hochberg, bonferroni, holm

        # P-values where differences are visible
        p_values = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05])

        fdr = benjamini_hochberg(p_values, alpha=0.05)
        holm_result = holm(p_values, alpha=0.05)
        bonf = bonferroni(p_values, alpha=0.05)

        # Expected ordering (with >= to handle edge cases)
        assert fdr.n_rejected >= holm_result.n_rejected >= bonf.n_rejected

    def test_all_methods_return_fdr_result(self):
        """All methods should return FDRResult objects."""
        from src.stats.fdr_correction import benjamini_hochberg, bonferroni, holm
        from src.stats import FDRResult

        p_values = np.array([0.01, 0.05, 0.1])

        for method in [benjamini_hochberg, bonferroni, holm]:
            result = method(p_values)
            assert isinstance(result, FDRResult)
            assert result.method in ["benjamini-hochberg", "bonferroni", "holm"]
