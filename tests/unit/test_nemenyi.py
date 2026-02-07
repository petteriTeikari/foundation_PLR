#!/usr/bin/env python
"""
test_nemenyi.py - Unit tests for Nemenyi post-hoc test statistical values.

Tests that the q-values from Demšar (2006) Table 5 are correctly implemented.
This is a CRITICAL test because incorrect q-values affect statistical significance claims.

Reference: Demšar, J. (2006). Statistical comparisons of classifiers over
           multiple data sets. JMLR, 7(1), 1-30, Table 5.
"""

import pytest
import numpy as np

# Add project root to path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.viz.cd_diagram_preprocessing import get_nemenyi_q_alpha


class TestNemenyiQValues:
    """Test Nemenyi critical values from Demšar 2006 Table 5."""

    # Expected q-values from Demšar 2006 Table 5 (α = 0.05)
    EXPECTED_Q_VALUES_005 = {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
        11: 3.219,
        12: 3.268,
        13: 3.313,
        14: 3.354,
        15: 3.399,
        16: 3.439,
        17: 3.476,
        18: 3.511,
        19: 3.544,
        20: 3.575,
    }

    # Expected q-values from Demšar 2006 Table 5 (α = 0.10)
    EXPECTED_Q_VALUES_010 = {
        2: 1.645,
        3: 2.052,
        4: 2.291,
        5: 2.459,
        6: 2.589,
        7: 2.693,
        8: 2.780,
        9: 2.855,
        10: 2.920,
    }

    def test_q_values_alpha_005(self):
        """Test that all q-values at α=0.05 match Demšar Table 5."""
        for k, expected in self.EXPECTED_Q_VALUES_005.items():
            actual = get_nemenyi_q_alpha(k, alpha=0.05)
            assert actual == pytest.approx(
                expected, rel=1e-3
            ), f"q-value for k={k} at α=0.05 should be {expected}, got {actual}"

    def test_q_values_alpha_010(self):
        """Test q-values at α=0.10 (less conservative)."""
        for k, expected in self.EXPECTED_Q_VALUES_010.items():
            actual = get_nemenyi_q_alpha(k, alpha=0.10)
            assert actual == pytest.approx(
                expected, rel=1e-3
            ), f"q-value for k={k} at α=0.10 should be {expected}, got {actual}"

    def test_k_equals_12_specific(self):
        """
        Test the specific case of k=12 which was incorrectly 2.576.

        The previous INCORRECT value was 2.576.
        The CORRECT value from Demšar 2006 Table 5 is 3.268.
        """
        k = 12
        expected = 3.268
        actual = get_nemenyi_q_alpha(k, alpha=0.05)

        # Assert it's NOT the incorrect value
        assert actual != pytest.approx(
            2.576, rel=0.01
        ), "q-value for k=12 should NOT be 2.576 (this was the bug)"

        # Assert it IS the correct value
        assert actual == pytest.approx(
            expected, rel=1e-3
        ), f"q-value for k=12 should be {expected}, got {actual}"

    def test_q_values_increase_with_k(self):
        """Test that q-values increase monotonically with k."""
        previous = 0
        for k in range(2, 21):
            current = get_nemenyi_q_alpha(k, alpha=0.05)
            assert (
                current > previous
            ), f"q-value should increase with k, but q({k}) = {current} <= q({k - 1}) = {previous}"
            previous = current

    def test_invalid_alpha_raises_error(self):
        """Test that invalid alpha values raise ValueError."""
        with pytest.raises(ValueError):
            get_nemenyi_q_alpha(10, alpha=0.01)  # Not supported

    def test_invalid_k_raises_error(self):
        """Test that k < 2 raises ValueError."""
        with pytest.raises(ValueError):
            get_nemenyi_q_alpha(1, alpha=0.05)

    def test_extrapolation_for_large_k(self):
        """Test that k > 20 uses reasonable extrapolation."""
        q_20 = get_nemenyi_q_alpha(20, alpha=0.05)
        q_25 = get_nemenyi_q_alpha(25, alpha=0.05)

        # Should be larger than k=20
        assert q_25 > q_20, "q-value for k=25 should be > k=20"

        # Should be reasonable (not way off)
        assert 3.5 < q_25 < 4.0, f"q-value for k=25 should be in [3.5, 4.0], got {q_25}"


class TestCriticalDifferenceFormula:
    """Test the Critical Difference formula."""

    def test_cd_formula_with_known_values(self):
        """Test CD calculation with known values."""
        # CD = q_alpha * sqrt(k(k+1) / (6*N))
        # For k=12, N=1000, alpha=0.05:
        # q_alpha = 3.268
        # CD = 3.268 * sqrt(12*13 / 6000) = 3.268 * sqrt(0.026) = 3.268 * 0.1612 = 0.527

        k = 12
        N = 1000
        q_alpha = get_nemenyi_q_alpha(k)
        cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))

        expected_cd = 0.527
        assert cd == pytest.approx(
            expected_cd, rel=0.01
        ), f"CD for k=12, N=1000 should be ~{expected_cd}, got {cd}"

    def test_cd_increases_with_k(self):
        """Test that CD increases with number of algorithms."""
        N = 1000
        previous_cd = 0

        for k in [5, 10, 15, 20]:
            q_alpha = get_nemenyi_q_alpha(k)
            cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
            assert (
                cd > previous_cd
            ), f"CD should increase with k, but CD(k={k}) <= CD(k-prev)"
            previous_cd = cd

    def test_cd_decreases_with_N(self):
        """Test that CD decreases with more samples."""
        k = 12
        q_alpha = get_nemenyi_q_alpha(k)

        cd_100 = q_alpha * np.sqrt(k * (k + 1) / (6 * 100))
        cd_1000 = q_alpha * np.sqrt(k * (k + 1) / (6 * 1000))
        cd_10000 = q_alpha * np.sqrt(k * (k + 1) / (6 * 10000))

        assert cd_100 > cd_1000 > cd_10000, "CD should decrease with more samples"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
