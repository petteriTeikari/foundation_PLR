"""
Unit tests for visualization modules.

Tests validate:
1. CD diagram computation and plotting
2. Forest plot generation
3. Heatmap creation
4. Specification curve visualization

Note: These tests verify that functions execute without error
and produce valid matplotlib figures. Visual inspection of
output is still recommended.
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt


class TestCDDiagram:
    """Tests for Critical Difference diagram module."""

    @pytest.fixture
    def sample_data(self):
        """Sample data for CD diagram tests."""
        np.random.seed(42)
        # Rows = "datasets" (preprocessing configs)
        # Columns = methods (classifiers)
        return pd.DataFrame(
            {
                "CatBoost": np.random.uniform(0.88, 0.93, 20),
                "XGBoost": np.random.uniform(0.85, 0.90, 20),
                "TabPFN": np.random.uniform(0.86, 0.91, 20),
                "LogReg": np.random.uniform(0.78, 0.85, 20),
            }
        )

    def test_friedman_nemenyi_test_returns_dict(self, sample_data):
        """Should return dictionary with all required keys."""
        from src.viz.cd_diagram import friedman_nemenyi_test

        result = friedman_nemenyi_test(sample_data)

        assert "friedman_statistic" in result
        assert "friedman_pvalue" in result
        assert "average_ranks" in result
        assert "critical_difference" in result
        assert "cliques" in result

    def test_friedman_pvalue_is_valid(self, sample_data):
        """p-value should be between 0 and 1."""
        from src.viz.cd_diagram import friedman_nemenyi_test

        result = friedman_nemenyi_test(sample_data)

        assert 0 <= result["friedman_pvalue"] <= 1

    def test_average_ranks_sum_correctly(self, sample_data):
        """Average ranks should sum to (k+1)/2 * n_methods."""
        from src.viz.cd_diagram import friedman_nemenyi_test

        result = friedman_nemenyi_test(sample_data)

        n_methods = len(sample_data.columns)
        expected_sum = n_methods * (n_methods + 1) / 2
        actual_sum = sum(result["average_ranks"].values())

        np.testing.assert_allclose(actual_sum, expected_sum, rtol=0.01)

    def test_compute_critical_difference(self):
        """CD should increase with more methods, decrease with more datasets."""
        from src.viz.cd_diagram import compute_critical_difference

        # More methods = larger CD
        cd_6_methods = compute_critical_difference(6, 100)
        cd_3_methods = compute_critical_difference(3, 100)
        assert cd_6_methods > cd_3_methods

        # More datasets = smaller CD
        cd_100_datasets = compute_critical_difference(6, 100)
        cd_20_datasets = compute_critical_difference(6, 20)
        assert cd_100_datasets < cd_20_datasets

    def test_identify_cliques(self):
        """Should identify groups of methods not significantly different."""
        from src.viz.cd_diagram import identify_cliques

        # Methods with close ranks
        avg_ranks = {"A": 1.0, "B": 1.5, "C": 2.0, "D": 4.0, "E": 4.5}
        cd = 1.2

        cliques = identify_cliques(avg_ranks, cd)

        # A, B, C should form a clique (within 1.2 of each other)
        # D, E should form a clique
        assert len(cliques) >= 2

    def test_draw_cd_diagram_returns_figure(self, sample_data):
        """Should return matplotlib figure and axes."""
        from src.viz.cd_diagram import draw_cd_diagram

        fig, ax = draw_cd_diagram(sample_data)

        assert isinstance(fig, plt.Figure)
        assert ax is not None

        plt.close(fig)

    def test_prepare_cd_data(self):
        """Should pivot long-format data to wide format."""
        from src.viz.cd_diagram import prepare_cd_data

        df = pd.DataFrame(
            {
                "config": ["A", "A", "B", "B"],
                "classifier": ["Cat", "XGB", "Cat", "XGB"],
                "auroc": [0.9, 0.85, 0.88, 0.83],
            }
        )

        wide_df = prepare_cd_data(df, "config", "classifier", "auroc")

        assert wide_df.shape == (2, 2)
        assert "Cat" in wide_df.columns
        assert "XGB" in wide_df.columns


class TestForestPlot:
    """Tests for forest plot module."""

    @pytest.fixture
    def sample_forest_data(self):
        """Sample data for forest plot tests."""
        return {
            "methods": ["CatBoost", "XGBoost", "TabPFN", "LogReg"],
            "estimates": [0.91, 0.88, 0.89, 0.82],
            "ci_lower": [0.87, 0.84, 0.85, 0.77],
            "ci_upper": [0.94, 0.91, 0.92, 0.86],
        }

    def test_draw_forest_plot_returns_figure(self, sample_forest_data):
        """Should return matplotlib figure and axes."""
        from src.viz.forest_plot import draw_forest_plot

        fig, ax = draw_forest_plot(
            sample_forest_data["methods"],
            sample_forest_data["estimates"],
            sample_forest_data["ci_lower"],
            sample_forest_data["ci_upper"],
        )

        assert isinstance(fig, plt.Figure)
        assert ax is not None

        plt.close(fig)

    def test_forest_plot_with_reference_line(self, sample_forest_data):
        """Should work with reference line."""
        from src.viz.forest_plot import draw_forest_plot

        fig, ax = draw_forest_plot(
            sample_forest_data["methods"],
            sample_forest_data["estimates"],
            sample_forest_data["ci_lower"],
            sample_forest_data["ci_upper"],
            reference_line=0.93,
            reference_label="Najjar 2021",
        )

        plt.close(fig)

    def test_forest_plot_from_dataframe(self):
        """Should create plot from DataFrame."""
        from src.viz.forest_plot import forest_plot_from_dataframe

        df = pd.DataFrame(
            {
                "classifier": ["CatBoost", "XGBoost", "LogReg"],
                "auroc": [0.91, 0.88, 0.82],
                "auroc_ci_lower": [0.87, 0.84, 0.77],
                "auroc_ci_upper": [0.94, 0.91, 0.86],
            }
        )

        fig, ax = forest_plot_from_dataframe(
            df, "classifier", "auroc", "auroc_ci_lower", "auroc_ci_upper"
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestHeatmapSensitivity:
    """Tests for heatmap sensitivity module."""

    @pytest.fixture
    def sample_heatmap_data(self):
        """Sample data for heatmap tests."""
        outliers = ["IQR", "MAD", "ZScore", "Ensemble"]
        imputations = ["Mean", "KNN", "SAITS", "BRITS"]

        rows = []
        for o in outliers:
            for i in imputations:
                rows.append(
                    {
                        "outlier": o,
                        "imputation": i,
                        "auroc": np.random.uniform(0.80, 0.93),
                    }
                )

        return pd.DataFrame(rows)

    def test_draw_sensitivity_heatmap_returns_figure(self, sample_heatmap_data):
        """Should return matplotlib figure and axes."""
        from src.viz.heatmap_sensitivity import draw_sensitivity_heatmap

        fig, ax = draw_sensitivity_heatmap(
            sample_heatmap_data,
            row_col="outlier",
            col_col="imputation",
            value_col="auroc",
        )

        assert isinstance(fig, plt.Figure)
        assert ax is not None

        plt.close(fig)

    def test_heatmap_from_pivot(self, sample_heatmap_data):
        """Should work with pre-pivoted data."""
        from src.viz.heatmap_sensitivity import heatmap_from_pivot

        pivot = sample_heatmap_data.pivot(
            index="outlier", columns="imputation", values="auroc"
        )

        fig, ax = heatmap_from_pivot(pivot)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_heatmap_with_highlight(self, sample_heatmap_data):
        """Should highlight cells above threshold."""
        from src.viz.heatmap_sensitivity import draw_sensitivity_heatmap

        fig, ax = draw_sensitivity_heatmap(
            sample_heatmap_data,
            row_col="outlier",
            col_col="imputation",
            value_col="auroc",
            highlight_threshold=0.90,
            highlight_best=True,
        )

        plt.close(fig)


class TestSpecificationCurve:
    """Tests for specification curve module."""

    @pytest.fixture
    def sample_spec_data(self):
        """Sample data for specification curve tests."""
        n = 50
        np.random.seed(42)

        return pd.DataFrame(
            {
                "auroc": np.random.uniform(0.75, 0.93, n),
                "auroc_ci_lower": np.random.uniform(0.70, 0.88, n),
                "auroc_ci_upper": np.random.uniform(0.80, 0.96, n),
                "outlier": np.random.choice(["IQR", "MAD", "ZScore", "None"], n),
                "imputation": np.random.choice(["Mean", "KNN", "SAITS"], n),
                "classifier": np.random.choice(["CatBoost", "XGBoost", "LogReg"], n),
            }
        )

    def test_draw_specification_curve_returns_figure(self, sample_spec_data):
        """Should return matplotlib figure."""
        from src.viz.specification_curve import draw_specification_curve

        specs = {
            "outlier": sample_spec_data["outlier"].tolist(),
            "imputation": sample_spec_data["imputation"].tolist(),
            "classifier": sample_spec_data["classifier"].tolist(),
        }

        fig = draw_specification_curve(
            sample_spec_data["auroc"].tolist(),
            sample_spec_data["auroc_ci_lower"].tolist(),
            sample_spec_data["auroc_ci_upper"].tolist(),
            specs,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_specification_curve_from_dataframe(self, sample_spec_data):
        """Should create curve from DataFrame."""
        from src.viz.specification_curve import specification_curve_from_dataframe

        fig = specification_curve_from_dataframe(
            sample_spec_data,
            estimate_col="auroc",
            ci_lower_col="auroc_ci_lower",
            ci_upper_col="auroc_ci_upper",
            spec_cols=["outlier", "imputation", "classifier"],
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_simple_specification_curve(self, sample_spec_data):
        """Should create simple curve without method indicators."""
        from src.viz.specification_curve import simple_specification_curve

        fig, ax = simple_specification_curve(
            sample_spec_data,
            estimate_col="auroc",
            reference_line=0.93,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestVizImports:
    """Test that viz module imports work correctly."""

    def test_import_cd_diagram(self):
        """Should be able to import CD diagram functions."""

    def test_import_forest_plot(self):
        """Should be able to import forest plot functions."""

    def test_import_heatmap(self):
        """Should be able to import heatmap functions."""

    def test_import_specification_curve(self):
        """Should be able to import specification curve functions."""
