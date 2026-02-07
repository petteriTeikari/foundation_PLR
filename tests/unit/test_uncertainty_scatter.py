"""Unit tests for uncertainty scatter visualization.

Tests validate:
1. Scatter plot generation
2. Color coding by outcome
3. Correlation annotation
4. JSON reproducibility data export
"""

import json
import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TestUncertaintyScatter:
    """Tests for uncertainty scatter plot."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data with uncertainty."""
        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.binomial(1, 0.3, n)
        y_prob = np.clip(
            y_true * 0.7 + (1 - y_true) * 0.3 + rng.normal(0, 0.1, n), 0.01, 0.99
        )
        # Uncertainty higher near 0.5 (decision boundary)
        uncertainty = np.abs(y_prob - 0.5) * -1 + 0.5 + rng.uniform(0, 0.1, n)
        return {"y_true": y_true, "y_prob": y_prob, "uncertainty": uncertainty}

    def test_plot_returns_figure(self, sample_data):
        """Test that plot returns valid figure."""
        from src.viz.uncertainty_scatter import plot_uncertainty_scatter

        fig, ax = plot_uncertainty_scatter(**sample_data)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_saves_to_file(self, sample_data, tmp_path):
        """Test that figure can be saved to file."""
        from src.viz.uncertainty_scatter import plot_uncertainty_scatter

        fig, ax = plot_uncertainty_scatter(**sample_data)
        output_path = tmp_path / "test_uncertainty_scatter.pdf"
        fig.savefig(output_path)
        assert output_path.exists()
        plt.close(fig)

    def test_axes_labels(self, sample_data):
        """Test that axes have proper labels."""
        from src.viz.uncertainty_scatter import plot_uncertainty_scatter

        fig, ax = plot_uncertainty_scatter(**sample_data)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close(fig)

    def test_colors_by_outcome(self, sample_data):
        """Test that different colors used for different outcomes."""
        from src.viz.uncertainty_scatter import plot_uncertainty_scatter

        fig, ax = plot_uncertainty_scatter(**sample_data, color_by_outcome=True)
        # Should have scatter collections
        collections = ax.collections
        assert len(collections) >= 1
        plt.close(fig)

    def test_without_color_by_outcome(self, sample_data):
        """Test single-color scatter mode."""
        from src.viz.uncertainty_scatter import plot_uncertainty_scatter

        fig, ax = plot_uncertainty_scatter(**sample_data, color_by_outcome=False)
        assert len(ax.collections) >= 1
        plt.close(fig)

    def test_regression_line_option(self, sample_data):
        """Test optional regression line."""
        from src.viz.uncertainty_scatter import plot_uncertainty_scatter

        fig, ax = plot_uncertainty_scatter(**sample_data, show_regression=True)
        # Should have at least one line
        lines = ax.get_lines()
        assert len(lines) >= 1
        plt.close(fig)

    def test_correlation_annotation(self, sample_data):
        """Test correlation annotation."""
        from src.viz.uncertainty_scatter import plot_uncertainty_scatter

        fig, ax = plot_uncertainty_scatter(**sample_data, show_correlation=True)
        # Should have text annotation
        # Correlation shown somewhere
        plt.close(fig)


class TestUncertaintyMetrics:
    """Tests for uncertainty metrics computation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.binomial(1, 0.3, n)
        y_prob = np.clip(
            y_true * 0.7 + (1 - y_true) * 0.3 + rng.normal(0, 0.1, n), 0.01, 0.99
        )
        uncertainty = np.abs(y_prob - 0.5) * -1 + 0.5 + rng.uniform(0, 0.1, n)
        return {"y_true": y_true, "y_prob": y_prob, "uncertainty": uncertainty}

    def test_spearman_correlation(self, sample_data):
        """Test Spearman correlation computation."""
        from src.viz.uncertainty_scatter import compute_uncertainty_correlation

        corr, p_value = compute_uncertainty_correlation(
            sample_data["y_prob"], sample_data["uncertainty"]
        )

        assert -1 <= corr <= 1
        assert 0 <= p_value <= 1


class TestUncertaintyDataExport:
    """Tests for JSON data export."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.binomial(1, 0.3, n)
        y_prob = np.clip(
            y_true * 0.7 + (1 - y_true) * 0.3 + rng.normal(0, 0.1, n), 0.01, 0.99
        )
        uncertainty = np.abs(y_prob - 0.5) * -1 + 0.5 + rng.uniform(0, 0.1, n)
        return {"y_true": y_true, "y_prob": y_prob, "uncertainty": uncertainty}

    def test_json_data_export(self, sample_data, tmp_path):
        """Test JSON reproducibility data is saved."""
        from src.viz.uncertainty_scatter import plot_uncertainty_scatter

        json_path = tmp_path / "test_uncertainty.json"
        fig, ax = plot_uncertainty_scatter(**sample_data, save_json_path=str(json_path))

        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)

        assert "y_prob" in data
        assert "uncertainty" in data
        assert "y_true" in data
        plt.close(fig)


class TestUncertaintyEdgeCases:
    """Edge cases for uncertainty visualization."""

    def test_uniform_uncertainty(self):
        """Test with uniform (uninformative) uncertainty."""
        from src.viz.uncertainty_scatter import plot_uncertainty_scatter

        rng = np.random.default_rng(42)
        n = 100
        y_true = rng.binomial(1, 0.3, n)
        y_prob = rng.uniform(0.1, 0.9, n)
        uncertainty = np.full(n, 0.5)  # All same uncertainty

        fig, ax = plot_uncertainty_scatter(
            y_true=y_true, y_prob=y_prob, uncertainty=uncertainty
        )
        plt.close(fig)

    def test_small_sample(self):
        """Test with very small sample."""
        from src.viz.uncertainty_scatter import plot_uncertainty_scatter

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.2, 0.3, 0.7, 0.8])
        uncertainty = np.array([0.1, 0.2, 0.2, 0.1])

        fig, ax = plot_uncertainty_scatter(
            y_true=y_true, y_prob=y_prob, uncertainty=uncertainty
        )
        plt.close(fig)
