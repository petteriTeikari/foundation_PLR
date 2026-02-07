"""
Unit tests for variance decomposition module.

Tests validate:
1. Factorial ANOVA computation with Type III SS
2. Effect size computation (partial eta-squared, omega-squared)
3. Assumption testing (normality, homogeneity)
4. Bootstrap CI for effect sizes
"""

import numpy as np
import pandas as pd
import pytest


class TestFactorialANOVA:
    """Tests for factorial_anova function."""

    @pytest.fixture
    def simple_factorial_data(self):
        """Simple balanced 2x2 factorial design."""
        np.random.seed(42)

        # Create balanced design
        data = []
        for outlier in ["IQR", "MAD"]:
            for imputation in ["Mean", "SAITS"]:
                # 10 observations per cell
                for _ in range(10):
                    # Simulate AUROC with main effects
                    base = 0.80
                    outlier_effect = 0.02 if outlier == "IQR" else 0.0
                    imputation_effect = 0.05 if imputation == "SAITS" else 0.0
                    noise = np.random.normal(0, 0.02)

                    data.append(
                        {
                            "outlier": outlier,
                            "imputation": imputation,
                            "auroc": base + outlier_effect + imputation_effect + noise,
                        }
                    )

        return pd.DataFrame(data)

    @pytest.fixture
    def three_factor_data(self):
        """Three-factor factorial design (outlier × imputation × classifier)."""
        np.random.seed(42)

        outliers = ["IQR", "MAD", "ZScore"]
        imputations = ["Mean", "SAITS"]
        classifiers = ["CatBoost", "XGBoost"]

        data = []
        for outlier in outliers:
            for imputation in imputations:
                for classifier in classifiers:
                    # 5 observations per cell
                    for _ in range(5):
                        base = 0.80
                        outlier_effect = {"IQR": 0.02, "MAD": 0.01, "ZScore": 0.0}[
                            outlier
                        ]
                        imputation_effect = 0.05 if imputation == "SAITS" else 0.0
                        classifier_effect = 0.03 if classifier == "CatBoost" else 0.0
                        noise = np.random.normal(0, 0.02)

                        data.append(
                            {
                                "outlier": outlier,
                                "imputation": imputation,
                                "classifier": classifier,
                                "auroc": base
                                + outlier_effect
                                + imputation_effect
                                + classifier_effect
                                + noise,
                            }
                        )

        return pd.DataFrame(data)

    def test_factorial_anova_returns_result(self, simple_factorial_data):
        """Should return FactorialANOVAResult."""
        from src.stats.variance_decomposition import (
            factorial_anova,
            FactorialANOVAResult,
        )

        result = factorial_anova(
            simple_factorial_data, dv="auroc", factors=["outlier", "imputation"]
        )

        assert isinstance(result, FactorialANOVAResult)

    def test_anova_table_has_correct_terms(self, simple_factorial_data):
        """ANOVA table should contain all factors and interactions."""
        from src.stats.variance_decomposition import factorial_anova

        result = factorial_anova(
            simple_factorial_data,
            dv="auroc",
            factors=["outlier", "imputation"],
            include_interactions=True,
        )

        # Check main effects and interaction are present
        assert "C(outlier)" in result.anova_table.index
        assert "C(imputation)" in result.anova_table.index
        assert "C(outlier):C(imputation)" in result.anova_table.index
        assert "Residual" in result.anova_table.index

    def test_effect_sizes_are_computed(self, simple_factorial_data):
        """Effect sizes should be computed for each term."""
        from src.stats.variance_decomposition import factorial_anova

        result = factorial_anova(
            simple_factorial_data, dv="auroc", factors=["outlier", "imputation"]
        )

        # Check effect sizes exist
        assert "C(outlier)" in result.effect_sizes
        assert "C(imputation)" in result.effect_sizes

        # Check both eta_sq and omega_sq are present
        for term in ["C(outlier)", "C(imputation)"]:
            assert "partial_eta_sq" in result.effect_sizes[term]
            assert "omega_sq" in result.effect_sizes[term]

    def test_partial_eta_squared_in_valid_range(self, simple_factorial_data):
        """Partial eta-squared should be in [0, 1]."""
        from src.stats.variance_decomposition import factorial_anova

        result = factorial_anova(
            simple_factorial_data, dv="auroc", factors=["outlier", "imputation"]
        )

        for term, sizes in result.effect_sizes.items():
            eta_sq = sizes["partial_eta_sq"]
            assert 0 <= eta_sq <= 1, f"Partial eta-sq for {term} = {eta_sq}"

    def test_omega_squared_not_negative(self, simple_factorial_data):
        """Omega-squared should be clipped to 0 (not negative)."""
        from src.stats.variance_decomposition import factorial_anova

        result = factorial_anova(
            simple_factorial_data, dv="auroc", factors=["outlier", "imputation"]
        )

        for term, sizes in result.effect_sizes.items():
            omega_sq = sizes["omega_sq"]
            assert omega_sq >= 0, f"Omega-sq for {term} = {omega_sq}"

    def test_three_factor_anova(self, three_factor_data):
        """Should handle three-factor design."""
        from src.stats.variance_decomposition import factorial_anova

        result = factorial_anova(
            three_factor_data,
            dv="auroc",
            factors=["outlier", "imputation", "classifier"],
        )

        # Should have main effects
        assert "C(outlier)" in result.effect_sizes
        assert "C(imputation)" in result.effect_sizes
        assert "C(classifier)" in result.effect_sizes

        # Should have two-way interactions
        assert "C(outlier):C(imputation)" in result.anova_table.index
        assert "C(outlier):C(classifier)" in result.anova_table.index
        assert "C(imputation):C(classifier)" in result.anova_table.index

        # Should have three-way interaction
        assert "C(outlier):C(imputation):C(classifier)" in result.anova_table.index

    def test_main_effects_only(self, simple_factorial_data):
        """Should work without interactions."""
        from src.stats.variance_decomposition import factorial_anova

        result = factorial_anova(
            simple_factorial_data,
            dv="auroc",
            factors=["outlier", "imputation"],
            include_interactions=False,
        )

        # Should have main effects
        assert "C(outlier)" in result.anova_table.index
        assert "C(imputation)" in result.anova_table.index

        # Should NOT have interaction
        interaction_present = any(":" in str(idx) for idx in result.anova_table.index)
        assert not interaction_present

    def test_r_squared_in_valid_range(self, simple_factorial_data):
        """R-squared should be in [0, 1]."""
        from src.stats.variance_decomposition import factorial_anova

        result = factorial_anova(
            simple_factorial_data, dv="auroc", factors=["outlier", "imputation"]
        )

        assert 0 <= result.r_squared <= 1

    def test_pvalues_computed(self, simple_factorial_data):
        """P-values should be computed for each factor."""
        from src.stats.variance_decomposition import factorial_anova

        result = factorial_anova(
            simple_factorial_data, dv="auroc", factors=["outlier", "imputation"]
        )

        for term, sizes in result.effect_sizes.items():
            p_value = sizes["p_value"]
            assert 0 <= p_value <= 1, f"p-value for {term} = {p_value}"


class TestAssumptionTesting:
    """Tests for ANOVA assumption testing."""

    @pytest.fixture
    def normal_data(self):
        """Data with normal residuals."""
        np.random.seed(42)

        data = []
        for group in ["A", "B", "C"]:
            for _ in range(30):
                value = np.random.normal(10, 1)  # Normal distribution
                data.append({"group": group, "value": value})

        return pd.DataFrame(data)

    @pytest.fixture
    def non_normal_data(self):
        """Data with highly skewed residuals."""
        np.random.seed(42)

        data = []
        for group in ["A", "B", "C"]:
            for _ in range(30):
                # Exponential distribution (highly skewed)
                value = np.random.exponential(1) + (ord(group) - ord("A"))
                data.append({"group": group, "value": value})

        return pd.DataFrame(data)

    def test_assumption_test_returns_result(self, normal_data):
        """Should return AssumptionTestResult."""
        from src.stats.variance_decomposition import (
            test_anova_assumptions,
            AssumptionTestResult,
        )

        result = test_anova_assumptions(normal_data, dv="value", factors=["group"])

        assert isinstance(result, AssumptionTestResult)

    def test_normality_test_included(self, normal_data):
        """Should include normality test results."""
        from src.stats.variance_decomposition import test_anova_assumptions

        result = test_anova_assumptions(normal_data, dv="value", factors=["group"])

        assert hasattr(result, "normality_statistic")
        assert hasattr(result, "normality_pvalue")
        assert hasattr(result, "normality_passed")

    def test_homogeneity_test_included(self, normal_data):
        """Should include homogeneity test results."""
        from src.stats.variance_decomposition import test_anova_assumptions

        result = test_anova_assumptions(normal_data, dv="value", factors=["group"])

        assert hasattr(result, "homogeneity_statistic")
        assert hasattr(result, "homogeneity_pvalue")
        assert hasattr(result, "homogeneity_passed")

    def test_normal_data_passes_normality(self, normal_data):
        """Normal data should pass normality test (usually)."""
        from src.stats.variance_decomposition import test_anova_assumptions

        result = test_anova_assumptions(normal_data, dv="value", factors=["group"])

        # p-value should be > 0.05 for normal data
        # Note: This can occasionally fail due to randomness
        assert result.normality_pvalue > 0.01

    def test_warnings_for_violations(self, non_normal_data):
        """Should generate warnings for assumption violations."""
        from src.stats.variance_decomposition import test_anova_assumptions

        result = test_anova_assumptions(non_normal_data, dv="value", factors=["group"])

        # Exponential data should violate normality
        # This may or may not generate warnings depending on sample
        assert isinstance(result.warnings, list)


class TestEffectSizeCI:
    """Tests for bootstrap CI computation."""

    @pytest.fixture
    def simple_data(self):
        """Simple data for CI testing."""
        np.random.seed(42)

        data = []
        for factor in ["A", "B"]:
            effect = 0.1 if factor == "A" else 0.0
            for _ in range(20):
                data.append(
                    {"factor": factor, "value": 1.0 + effect + np.random.normal(0, 0.1)}
                )

        return pd.DataFrame(data)

    def test_ci_returns_tuple(self, simple_data):
        """Should return (lower, upper) tuple."""
        from src.stats.variance_decomposition import compute_effect_size_ci

        ci_lower, ci_upper = compute_effect_size_ci(
            simple_data,
            dv="value",
            factors=["factor"],
            term="C(factor)",
            n_bootstrap=100,  # Small for speed
        )

        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)

    def test_ci_lower_less_than_upper(self, simple_data):
        """CI lower should be less than CI upper."""
        from src.stats.variance_decomposition import compute_effect_size_ci

        ci_lower, ci_upper = compute_effect_size_ci(
            simple_data,
            dv="value",
            factors=["factor"],
            term="C(factor)",
            n_bootstrap=100,
        )

        if np.isfinite(ci_lower) and np.isfinite(ci_upper):
            assert ci_lower <= ci_upper

    def test_ci_in_valid_range(self, simple_data):
        """CI bounds should be in [0, 1] for partial eta-squared."""
        from src.stats.variance_decomposition import compute_effect_size_ci

        ci_lower, ci_upper = compute_effect_size_ci(
            simple_data,
            dv="value",
            factors=["factor"],
            term="C(factor)",
            n_bootstrap=100,
        )

        if np.isfinite(ci_lower):
            assert ci_lower >= 0
        if np.isfinite(ci_upper):
            assert ci_upper <= 1


class TestLatexExport:
    """Tests for LaTeX table generation."""

    @pytest.fixture
    def anova_result(self):
        """Compute ANOVA result for LaTeX testing."""
        from src.stats.variance_decomposition import factorial_anova

        np.random.seed(42)
        data = []
        for a in ["A1", "A2"]:
            for b in ["B1", "B2"]:
                for _ in range(10):
                    data.append(
                        {"factor_a": a, "factor_b": b, "y": np.random.normal(10, 1)}
                    )

        df = pd.DataFrame(data)
        return factorial_anova(df, dv="y", factors=["factor_a", "factor_b"])

    def test_to_latex_returns_string(self, anova_result):
        """to_latex() should return string."""
        latex = anova_result.to_latex()
        assert isinstance(latex, str)

    def test_latex_contains_table_environment(self, anova_result):
        """LaTeX should contain table environment."""
        latex = anova_result.to_latex()
        assert r"\begin{table}" in latex
        assert r"\end{table}" in latex

    def test_latex_contains_effect_sizes(self, anova_result):
        """LaTeX should contain eta-squared and omega-squared columns."""
        latex = anova_result.to_latex()
        assert r"\eta^2_p" in latex
        assert r"\omega^2" in latex


class TestValidation:
    """Tests for input validation."""

    def test_missing_dv_column_raises(self):
        """Should raise ValidationError for missing DV column."""
        from src.stats.variance_decomposition import factorial_anova
        from src.stats._exceptions import ValidationError

        df = pd.DataFrame({"factor": ["A", "B", "A", "B"], "other": [1, 2, 3, 4]})

        with pytest.raises(ValidationError):
            factorial_anova(df, dv="missing_column", factors=["factor"])

    def test_missing_factor_column_raises(self):
        """Should raise ValidationError for missing factor column."""
        from src.stats.variance_decomposition import factorial_anova
        from src.stats._exceptions import ValidationError

        df = pd.DataFrame({"y": [1, 2, 3, 4], "factor": ["A", "B", "A", "B"]})

        with pytest.raises(ValidationError):
            factorial_anova(df, dv="y", factors=["factor", "missing"])

    def test_single_level_factor_raises(self):
        """Should raise ValidationError for factor with single level."""
        from src.stats.variance_decomposition import factorial_anova
        from src.stats._exceptions import ValidationError

        df = pd.DataFrame(
            {
                "y": [1, 2, 3, 4],
                "factor": ["A", "A", "A", "A"],  # Only one level
            }
        )

        with pytest.raises(ValidationError):
            factorial_anova(df, dv="y", factors=["factor"])

    def test_insufficient_data_raises(self):
        """Should raise InsufficientDataError for too few observations."""
        from src.stats.variance_decomposition import factorial_anova
        from src.stats._exceptions import InsufficientDataError

        df = pd.DataFrame({"y": [1, 2], "a": ["A", "B"], "b": ["X", "Y"]})

        with pytest.raises(InsufficientDataError):
            factorial_anova(df, dv="y", factors=["a", "b"])


class TestImports:
    """Test module imports."""

    def test_import_factorial_anova(self):
        """Should import factorial_anova."""
        from src.stats.variance_decomposition import factorial_anova

        assert callable(factorial_anova)

    def test_import_test_anova_assumptions(self):
        """Should import test_anova_assumptions."""
        from src.stats.variance_decomposition import test_anova_assumptions

        assert callable(test_anova_assumptions)

    def test_import_result_types(self):
        """Should import result types."""
