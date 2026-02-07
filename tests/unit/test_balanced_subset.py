"""
Unit tests for balanced factorial subset filtering.

The experimental design is unbalanced - different outlier methods have
different numbers of imputation methods tested. For valid comparison,
we must filter to a balanced subset.

Key insight: Comparing mean AUROC across outlier methods when they have
different imputation coverage is STATISTICALLY INVALID.
"""

import pytest
import pandas as pd


def get_balanced_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to balanced factorial subset.

    Returns only (outlier, imputation) combinations where EVERY outlier
    method has been tested with EVERY imputation method in the subset.
    """
    # Find imputation methods available for ALL outlier methods
    imps_per_outlier = df.groupby("outlier_method")["imputation_method"].apply(set)
    common_imps = set.intersection(*imps_per_outlier.values)

    if not common_imps:
        raise ValueError("No common imputation methods across all outlier methods!")

    # Filter to balanced subset
    balanced = df[df["imputation_method"].isin(common_imps)].copy()

    # Verify balance
    counts = balanced.groupby("outlier_method").size()
    if counts.nunique() != 1:
        raise ValueError(f"Imbalanced result: {counts.to_dict()}")

    return balanced


class TestBalancedSubset:
    """Test balanced factorial subset filtering."""

    @pytest.fixture
    def unbalanced_data(self):
        """Unbalanced design: pupil-gt has extra imputation."""
        return pd.DataFrame(
            {
                "outlier_method": [
                    "pupil-gt",
                    "pupil-gt",
                    "pupil-gt",
                    "pupil-gt",  # 4 imputations
                    "LOF",
                    "LOF",
                    "LOF",  # 3 imputations (missing pupil-gt imp)
                    "MOMENT-gt-finetune",
                    "MOMENT-gt-finetune",
                    "MOMENT-gt-finetune",  # 3
                ],
                "imputation_method": [
                    "SAITS",
                    "CSDI",
                    "TimesNet",
                    "pupil-gt",  # pupil-gt has extra
                    "SAITS",
                    "CSDI",
                    "TimesNet",
                    "SAITS",
                    "CSDI",
                    "TimesNet",
                ],
                "auroc": [0.91, 0.90, 0.89, 0.92, 0.85, 0.86, 0.84, 0.88, 0.87, 0.86],
            }
        )

    def test_returns_balanced_counts(self, unbalanced_data):
        """All outlier methods should have same count after filtering."""
        balanced = get_balanced_subset(unbalanced_data)
        counts = balanced.groupby("outlier_method").size()

        assert counts.nunique() == 1, f"Unequal counts: {counts.to_dict()}"

    def test_excludes_non_common_imputations(self, unbalanced_data):
        """pupil-gt imputation should be excluded (not available for LOF)."""
        balanced = get_balanced_subset(unbalanced_data)

        assert "pupil-gt" not in balanced["imputation_method"].values

    def test_keeps_common_imputations(self, unbalanced_data):
        """SAITS, CSDI, TimesNet should be kept (available for all)."""
        balanced = get_balanced_subset(unbalanced_data)
        kept = set(balanced["imputation_method"].unique())

        assert kept == {"SAITS", "CSDI", "TimesNet"}

    def test_balanced_has_fewer_rows(self, unbalanced_data):
        """Balanced subset should have fewer rows than original."""
        balanced = get_balanced_subset(unbalanced_data)

        assert len(balanced) < len(unbalanced_data)
        assert len(balanced) == 9  # 3 outliers x 3 imputations

    def test_preserves_auroc_values(self, unbalanced_data):
        """AUROC values should be preserved (not modified)."""
        balanced = get_balanced_subset(unbalanced_data)

        # Check a specific value is preserved
        lof_saits = balanced[
            (balanced["outlier_method"] == "LOF")
            & (balanced["imputation_method"] == "SAITS")
        ]
        assert len(lof_saits) == 1
        assert lof_saits.iloc[0]["auroc"] == 0.85


class TestBalancedSubsetEdgeCases:
    """Test edge cases in balanced subset filtering."""

    def test_already_balanced_returns_same(self):
        """Already balanced data should return the same."""
        balanced_data = pd.DataFrame(
            {
                "outlier_method": ["LOF", "LOF", "pupil-gt", "pupil-gt"],
                "imputation_method": ["SAITS", "CSDI", "SAITS", "CSDI"],
                "auroc": [0.85, 0.86, 0.91, 0.90],
            }
        )
        result = get_balanced_subset(balanced_data)

        assert len(result) == len(balanced_data)

    def test_raises_on_no_common_imputations(self):
        """Should raise if no imputation is common to all outlier methods."""
        disjoint_data = pd.DataFrame(
            {
                "outlier_method": ["LOF", "LOF", "pupil-gt", "pupil-gt"],
                "imputation_method": ["SAITS", "CSDI", "TimesNet", "MOMENT-finetune"],
                "auroc": [0.85, 0.86, 0.91, 0.90],
            }
        )

        with pytest.raises(ValueError, match="No common imputation methods"):
            get_balanced_subset(disjoint_data)

    def test_handles_single_outlier_method(self):
        """Single outlier method should return all its configs."""
        single_outlier = pd.DataFrame(
            {
                "outlier_method": ["LOF", "LOF", "LOF"],
                "imputation_method": ["SAITS", "CSDI", "TimesNet"],
                "auroc": [0.85, 0.86, 0.84],
            }
        )
        result = get_balanced_subset(single_outlier)

        # All imputations are "common" since there's only one outlier method
        assert len(result) == 3

    def test_handles_single_imputation(self):
        """All methods having only one common imputation should work."""
        single_common = pd.DataFrame(
            {
                "outlier_method": ["LOF", "LOF", "pupil-gt"],
                "imputation_method": ["SAITS", "CSDI", "SAITS"],  # Only SAITS is common
                "auroc": [0.85, 0.86, 0.91],
            }
        )
        result = get_balanced_subset(single_common)

        assert len(result) == 2  # LOF-SAITS and pupil-gt-SAITS
        assert set(result["imputation_method"]) == {"SAITS"}


class TestBalancedSubsetWithRealData:
    """Test with data structure similar to real experiment."""

    @pytest.fixture
    def realistic_data(self):
        """Realistic unbalanced data similar to actual experiment."""
        # Simulates: pupil-gt has 7 imputations (including pupil-gt)
        # Other methods have 5-6 imputations
        rows = []

        # pupil-gt: has all 7 imputations including itself
        for imp in [
            "pupil-gt",
            "SAITS",
            "CSDI",
            "TimesNet",
            "MOMENT-finetune",
            "MOMENT-zeroshot",
            "ensemble",
        ]:
            rows.append(
                {"outlier_method": "pupil-gt", "imputation_method": imp, "auroc": 0.91}
            )

        # LOF: has 5 standard imputations (no pupil-gt, no ensemble)
        for imp in ["SAITS", "CSDI", "TimesNet", "MOMENT-finetune", "MOMENT-zeroshot"]:
            rows.append(
                {"outlier_method": "LOF", "imputation_method": imp, "auroc": 0.85}
            )

        # MOMENT-gt-finetune: has 6 imputations (no pupil-gt)
        for imp in [
            "SAITS",
            "CSDI",
            "TimesNet",
            "MOMENT-finetune",
            "MOMENT-zeroshot",
            "ensemble",
        ]:
            rows.append(
                {
                    "outlier_method": "MOMENT-gt-finetune",
                    "imputation_method": imp,
                    "auroc": 0.88,
                }
            )

        return pd.DataFrame(rows)

    def test_finds_common_5_imputations(self, realistic_data):
        """Should find 5 common imputations in realistic data."""
        balanced = get_balanced_subset(realistic_data)
        common_imps = set(balanced["imputation_method"].unique())

        # Common to all: SAITS, CSDI, TimesNet, MOMENT-finetune, MOMENT-zeroshot
        expected = {"SAITS", "CSDI", "TimesNet", "MOMENT-finetune", "MOMENT-zeroshot"}
        assert common_imps == expected

    def test_balanced_has_15_rows(self, realistic_data):
        """3 outliers x 5 imputations = 15 rows."""
        balanced = get_balanced_subset(realistic_data)
        assert len(balanced) == 15

    def test_each_outlier_has_5_configs(self, realistic_data):
        """Each outlier method should have exactly 5 configs."""
        balanced = get_balanced_subset(realistic_data)
        counts = balanced.groupby("outlier_method").size()

        assert all(count == 5 for count in counts)
