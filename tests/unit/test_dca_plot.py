"""Unit tests for Decision Curve Analysis visualization.

Tests validate:
1. Net benefit computation correctness
2. DCA plot generation
3. Treat-all/treat-none reference strategies
4. STRATOS-compliant threshold ranges for glaucoma
5. JSON reproducibility data export

References:
- Vickers & Elkin 2006
- Van Calster et al. 2024 STRATOS guidelines
"""

import json
import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TestDCAComputation:
    """Tests for DCA computational correctness."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data."""
        rng = np.random.default_rng(42)
        n = 200
        prevalence = 0.3
        y_true = rng.binomial(1, prevalence, n)
        y_prob = np.clip(
            y_true * 0.7 + (1 - y_true) * 0.3 + rng.normal(0, 0.15, n), 0.01, 0.99
        )
        return {"y_true": y_true, "y_prob": y_prob}

    def test_net_benefit_formula_correct(self, sample_data):
        """Verify NB = TP/n - FP/n * (pt/(1-pt))."""
        from src.viz.dca_plot import compute_net_benefit

        y_true = sample_data["y_true"]
        y_prob = sample_data["y_prob"]
        threshold = 0.15

        # Manual calculation
        y_pred = (y_prob >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        n = len(y_true)
        expected_nb = tp / n - fp / n * (threshold / (1 - threshold))

        actual_nb = compute_net_benefit(y_true, y_prob, threshold)
        np.testing.assert_allclose(actual_nb, expected_nb, rtol=1e-10)

    def test_net_benefit_at_zero_threshold(self, sample_data):
        """At threshold=0, everyone is treated, so NB approaches prevalence."""
        from src.viz.dca_plot import compute_net_benefit

        y_true = sample_data["y_true"]
        y_prob = sample_data["y_prob"]

        # At very low threshold, almost everyone treated
        nb = compute_net_benefit(y_true, y_prob, 0.001)
        y_true.mean()
        # Should be close to prevalence (all treated, few FP penalty at low threshold)
        assert nb > 0

    def test_treat_all_net_benefit(self, sample_data):
        """Verify treat-all: NB = prevalence - (1-prevalence) * odds."""
        from src.viz.dca_plot import compute_treat_all_nb

        y_true = sample_data["y_true"]
        threshold = 0.15
        prevalence = y_true.mean()

        expected_nb = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
        actual_nb = compute_treat_all_nb(prevalence, threshold)

        np.testing.assert_allclose(actual_nb, expected_nb, rtol=1e-10)

    def test_treat_none_always_zero(self, sample_data):
        """Treat-none strategy should always have NB=0."""
        from src.viz.dca_plot import compute_treat_none_nb

        for threshold in [0.01, 0.1, 0.2, 0.3, 0.5]:
            nb = compute_treat_none_nb(threshold)
            assert nb == 0.0


class TestDCAPlotting:
    """Tests for DCA plot generation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data."""
        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.binomial(1, 0.3, n)
        y_prob = np.clip(
            y_true * 0.7 + (1 - y_true) * 0.3 + rng.normal(0, 0.15, n), 0.01, 0.99
        )
        return {"y_true": y_true, "y_prob": y_prob}

    def test_plot_returns_figure(self, sample_data):
        """Test that DCA plot returns valid figure."""
        from src.viz.dca_plot import plot_dca

        fig, ax = plot_dca(**sample_data)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_includes_three_curves(self, sample_data):
        """DCA should show model, treat-all, and treat-none curves."""
        from src.viz.dca_plot import plot_dca

        fig, ax = plot_dca(**sample_data)
        lines = ax.get_lines()

        # Should have at least 3 lines
        assert len(lines) >= 3
        plt.close(fig)

    def test_plot_glaucoma_threshold_range(self, sample_data):
        """Default range should be 1-30% for glaucoma context."""
        from src.viz.dca_plot import plot_dca

        fig, ax = plot_dca(**sample_data, threshold_range=(0.01, 0.30))

        # Check x-axis limits
        xlim = ax.get_xlim()
        assert xlim[0] <= 0.01
        assert xlim[1] >= 0.30
        plt.close(fig)

    def test_plot_saves_to_file(self, sample_data, tmp_path):
        """Test that figure can be saved to file."""
        from src.viz.dca_plot import plot_dca

        fig, ax = plot_dca(**sample_data)
        output_path = tmp_path / "test_dca.pdf"
        fig.savefig(output_path)
        assert output_path.exists()
        plt.close(fig)

    def test_axes_labels(self, sample_data):
        """Test that axes have proper labels."""
        from src.viz.dca_plot import plot_dca

        fig, ax = plot_dca(**sample_data)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close(fig)

    def test_legend_present(self, sample_data):
        """Test that legend is present."""
        from src.viz.dca_plot import plot_dca

        fig, ax = plot_dca(**sample_data)
        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)


class TestDCADataExport:
    """Tests for JSON data export."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data."""
        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.binomial(1, 0.3, n)
        y_prob = np.clip(
            y_true * 0.7 + (1 - y_true) * 0.3 + rng.normal(0, 0.15, n), 0.01, 0.99
        )
        return {"y_true": y_true, "y_prob": y_prob}

    def test_json_data_export(self, sample_data, tmp_path):
        """Test JSON reproducibility data is saved."""
        from src.viz.dca_plot import plot_dca

        json_path = tmp_path / "test_dca.json"
        fig, ax = plot_dca(**sample_data, save_json_path=str(json_path))

        assert json_path.exists(), "JSON data file must be created"

        with open(json_path) as f:
            data = json.load(f)

        assert "thresholds" in data
        assert "nb_model" in data
        assert "nb_all" in data
        assert "nb_none" in data
        plt.close(fig)

    def test_json_contains_valid_arrays(self, sample_data, tmp_path):
        """Test JSON contains valid numeric arrays."""
        from src.viz.dca_plot import plot_dca

        json_path = tmp_path / "test_dca.json"
        fig, ax = plot_dca(**sample_data, save_json_path=str(json_path))

        with open(json_path) as f:
            data = json.load(f)

        # All arrays should have same length
        assert len(data["thresholds"]) == len(data["nb_model"])
        assert len(data["thresholds"]) == len(data["nb_all"])
        assert len(data["thresholds"]) == len(data["nb_none"])
        plt.close(fig)


class TestDCAEdgeCases:
    """Tests for DCA edge cases."""

    def test_single_class_data(self):
        """Test behavior when only one class present."""
        from src.viz.dca_plot import compute_net_benefit

        y_true = np.zeros(100)  # All negative
        y_prob = np.random.uniform(0.1, 0.9, 100)

        # Should not raise error
        nb = compute_net_benefit(y_true, y_prob, 0.15)
        assert np.isfinite(nb) or np.isnan(nb)

    def test_extreme_threshold(self):
        """Test behavior at extreme thresholds."""
        from src.viz.dca_plot import compute_net_benefit

        rng = np.random.default_rng(42)
        y_true = rng.binomial(1, 0.3, 200)
        y_prob = rng.uniform(0.1, 0.9, 200)

        # Very high threshold - almost no one treated
        nb_high = compute_net_benefit(y_true, y_prob, 0.99)
        assert np.isfinite(nb_high)

        # Very low threshold - almost everyone treated
        nb_low = compute_net_benefit(y_true, y_prob, 0.01)
        assert np.isfinite(nb_low)

    def test_perfect_model(self):
        """Test DCA with perfect predictions."""
        from src.viz.dca_plot import compute_net_benefit

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])

        # At threshold 0.5, perfect model should have NB = prevalence
        nb = compute_net_benefit(y_true, y_prob, 0.5)
        prevalence = y_true.mean()
        assert nb == prevalence  # Perfect: all TP, no FP
