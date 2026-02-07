"""
Unit tests for CI aggregation - must use conservative bounds, NOT averaging.

TDD: Write FIRST, then fix R scripts.

The key insight: Averaging CI bounds is STATISTICALLY INVALID.
- CIs represent uncertainty around a point estimate
- Averaging bounds does not preserve coverage probability
- Conservative bounds (min of lower, max of upper) guarantee coverage >= 1-alpha
"""

import pytest
import pandas as pd
import numpy as np


def aggregate_ci_conservative(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate CIs using conservative (envelope) bounds.

    This is the CORRECT method - averaging CI bounds is INVALID.
    """
    return (
        df.groupby("outlier_method")
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_ci_lo=("auroc_ci_lo", "min"),  # Conservative: minimum of lower bounds
            auroc_ci_hi=("auroc_ci_hi", "max"),  # Conservative: maximum of upper bounds
            n_configs=("auroc", "count"),
        )
        .reset_index()
    )


def aggregate_ci_wrong(df: pd.DataFrame) -> pd.DataFrame:
    """WRONG method - averaging CI bounds. DO NOT USE."""
    return (
        df.groupby("outlier_method")
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_ci_lo=("auroc_ci_lo", "mean"),  # WRONG!
            auroc_ci_hi=("auroc_ci_hi", "mean"),  # WRONG!
        )
        .reset_index()
    )


class TestCIAggregation:
    """Test that CI aggregation uses conservative bounds."""

    @pytest.fixture
    def sample_data(self):
        """Sample data with multiple configs per outlier method."""
        return pd.DataFrame(
            {
                "outlier_method": ["LOF", "LOF", "LOF", "pupil-gt", "pupil-gt"],
                "imputation_method": ["SAITS", "CSDI", "TimesNet", "SAITS", "CSDI"],
                "auroc": [0.85, 0.87, 0.84, 0.91, 0.90],
                "auroc_ci_lo": [0.80, 0.82, 0.79, 0.88, 0.87],
                "auroc_ci_hi": [0.90, 0.92, 0.89, 0.94, 0.93],
            }
        )

    def test_conservative_uses_min_for_lower_bound(self, sample_data):
        """Conservative CI uses MIN of lower bounds."""
        result = aggregate_ci_conservative(sample_data)
        lof_row = result[result["outlier_method"] == "LOF"].iloc[0]

        # LOF has CI_lo values: 0.80, 0.82, 0.79 -> min = 0.79
        assert lof_row["auroc_ci_lo"] == 0.79

    def test_conservative_uses_max_for_upper_bound(self, sample_data):
        """Conservative CI uses MAX of upper bounds."""
        result = aggregate_ci_conservative(sample_data)
        lof_row = result[result["outlier_method"] == "LOF"].iloc[0]

        # LOF has CI_hi values: 0.90, 0.92, 0.89 -> max = 0.92
        assert lof_row["auroc_ci_hi"] == 0.92

    def test_wrong_method_averages_bounds(self, sample_data):
        """Verify the WRONG method produces different (incorrect) results."""
        correct = aggregate_ci_conservative(sample_data)
        wrong = aggregate_ci_wrong(sample_data)

        lof_correct = correct[correct["outlier_method"] == "LOF"].iloc[0]
        lof_wrong = wrong[wrong["outlier_method"] == "LOF"].iloc[0]

        # Wrong method: mean([0.80, 0.82, 0.79]) = 0.803...
        # Correct method: min([0.80, 0.82, 0.79]) = 0.79
        assert lof_wrong["auroc_ci_lo"] != lof_correct["auroc_ci_lo"]

    def test_conservative_ci_is_wider(self, sample_data):
        """Conservative CI should be WIDER than averaged CI."""
        correct = aggregate_ci_conservative(sample_data)
        wrong = aggregate_ci_wrong(sample_data)

        for method in ["LOF", "pupil-gt"]:
            correct_row = correct[correct["outlier_method"] == method].iloc[0]
            wrong_row = wrong[wrong["outlier_method"] == method].iloc[0]

            correct_width = correct_row["auroc_ci_hi"] - correct_row["auroc_ci_lo"]
            wrong_width = wrong_row["auroc_ci_hi"] - wrong_row["auroc_ci_lo"]

            assert correct_width >= wrong_width, (
                f"{method}: Conservative CI width ({correct_width:.3f}) should be >= "
                f"averaged width ({wrong_width:.3f})"
            )

    def test_mean_auroc_is_same_for_both_methods(self, sample_data):
        """Point estimate (mean AUROC) should be the same regardless of CI method."""
        correct = aggregate_ci_conservative(sample_data)
        wrong = aggregate_ci_wrong(sample_data)

        for method in ["LOF", "pupil-gt"]:
            correct_mean = correct[correct["outlier_method"] == method].iloc[0][
                "auroc_mean"
            ]
            wrong_mean = wrong[wrong["outlier_method"] == method].iloc[0]["auroc_mean"]

            assert np.isclose(correct_mean, wrong_mean), (
                f"{method}: Mean AUROC differs between methods "
                f"(correct: {correct_mean:.4f}, wrong: {wrong_mean:.4f})"
            )

    def test_handles_single_config(self):
        """When only one config exists, conservative = original CI."""
        single_config = pd.DataFrame(
            {
                "outlier_method": ["LOF"],
                "auroc": [0.85],
                "auroc_ci_lo": [0.80],
                "auroc_ci_hi": [0.90],
            }
        )
        result = aggregate_ci_conservative(single_config)
        row = result.iloc[0]

        assert row["auroc_ci_lo"] == 0.80
        assert row["auroc_ci_hi"] == 0.90


class TestCIAggregationEdgeCases:
    """Test edge cases in CI aggregation."""

    def test_handles_nan_values(self):
        """CI aggregation should handle NaN values gracefully."""
        data_with_nan = pd.DataFrame(
            {
                "outlier_method": ["LOF", "LOF", "LOF"],
                "auroc": [0.85, np.nan, 0.84],
                "auroc_ci_lo": [0.80, np.nan, 0.79],
                "auroc_ci_hi": [0.90, np.nan, 0.89],
            }
        )
        result = aggregate_ci_conservative(data_with_nan)
        row = result.iloc[0]

        # Should use min/max of non-NaN values
        assert row["auroc_ci_lo"] == 0.79
        assert row["auroc_ci_hi"] == 0.90

    def test_preserves_all_groups(self):
        """All outlier methods should be present in output."""
        data = pd.DataFrame(
            {
                "outlier_method": ["LOF", "LOF", "pupil-gt", "MOMENT-gt-finetune"],
                "auroc": [0.85, 0.86, 0.91, 0.88],
                "auroc_ci_lo": [0.80, 0.81, 0.88, 0.84],
                "auroc_ci_hi": [0.90, 0.91, 0.94, 0.92],
            }
        )
        result = aggregate_ci_conservative(data)

        assert len(result) == 3  # LOF, pupil-gt, MOMENT-gt-finetune
        assert set(result["outlier_method"]) == {
            "LOF",
            "pupil-gt",
            "MOMENT-gt-finetune",
        }
