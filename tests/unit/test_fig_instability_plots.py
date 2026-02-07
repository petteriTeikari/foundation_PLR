"""
Tests for instability visualization figures.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.stats.pminternal_analysis import BootstrapPredictionData


@pytest.fixture
def mock_bootstrap_data():
    """Create mock bootstrap prediction data."""
    np.random.seed(42)
    n_subjects = 30
    n_bootstrap = 100

    y_true = np.random.binomial(1, 0.3, n_subjects)
    y_original = np.clip(
        y_true * 0.7 + (1 - y_true) * 0.3 + np.random.normal(0, 0.1, n_subjects),
        0.01,
        0.99,
    )

    y_bootstrap = np.zeros((n_bootstrap, n_subjects))
    for i in range(n_bootstrap):
        y_bootstrap[i] = np.clip(
            y_original + np.random.normal(0, 0.08, n_subjects), 0.01, 0.99
        )

    return BootstrapPredictionData(
        combo_id="test_combo",
        y_true=y_true,
        y_pred_proba_original=y_original,
        y_pred_proba_bootstrap=y_bootstrap,
        n_subjects=n_subjects,
        n_bootstrap=n_bootstrap,
    )


class TestPredictionInstabilityPlot:
    """Tests for Riley 2023 style instability plot."""

    def test_plot_creates_figure(self, mock_bootstrap_data):
        """Test that plot function creates a figure."""
        from src.viz.fig_instability_plots import plot_prediction_instability

        fig, ax, plot_data = plot_prediction_instability(mock_bootstrap_data)

        assert fig is not None
        assert ax is not None
        assert "scatter" in plot_data
        assert "percentile_lines" in plot_data

    def test_plot_data_structure(self, mock_bootstrap_data):
        """Test that plot data has correct structure."""
        from src.viz.fig_instability_plots import plot_prediction_instability

        _, _, plot_data = plot_prediction_instability(mock_bootstrap_data, subsample=50)

        # Check scatter data
        assert len(plot_data["scatter"]["x"]) == 50 * 30  # subsample * n_subjects
        assert len(plot_data["scatter"]["y"]) == 50 * 30

        # Check percentile lines
        assert len(plot_data["percentile_lines"]["x"]) == 30
        assert len(plot_data["percentile_lines"]["p_2_5"]) == 30
        assert len(plot_data["percentile_lines"]["p_97_5"]) == 30


class TestPerPatientUncertaintyPlot:
    """Tests for Kompa 2021 style uncertainty plot."""

    def test_plot_creates_figure(self, mock_bootstrap_data):
        """Test that plot function creates a figure."""
        from src.viz.fig_instability_plots import plot_per_patient_uncertainty

        fig, ax, plot_data = plot_per_patient_uncertainty(mock_bootstrap_data)

        assert fig is not None
        assert ax is not None
        assert "patients" in plot_data

    def test_plot_with_specific_patients(self, mock_bootstrap_data):
        """Test plot with specific patient indices."""
        from src.viz.fig_instability_plots import plot_per_patient_uncertainty

        _, _, plot_data = plot_per_patient_uncertainty(
            mock_bootstrap_data, patient_indices=[0, 5, 10]
        )

        assert len(plot_data["patients"]) == 3
        assert plot_data["patients"][0]["index"] == 0
        assert plot_data["patients"][1]["index"] == 5
        assert plot_data["patients"][2]["index"] == 10


class TestMAPEHistogram:
    """Tests for MAPE histogram comparison."""

    def test_histogram_creates_figure(self, mock_bootstrap_data):
        """Test that histogram creates a figure."""
        from src.viz.fig_instability_plots import plot_mape_histogram

        data_list = [
            ("Method A", mock_bootstrap_data),
            ("Method B", mock_bootstrap_data),
        ]

        fig, ax, plot_data = plot_mape_histogram(data_list)

        assert fig is not None
        assert ax is not None
        assert "methods" in plot_data
        assert len(plot_data["methods"]) == 2


class TestInstabilityComparison:
    """Tests for multi-panel instability comparison."""

    def test_comparison_creates_panels(self, mock_bootstrap_data):
        """Test that comparison creates multiple panels."""
        from src.viz.fig_instability_plots import plot_instability_comparison

        data_list = [
            ("Ground Truth", mock_bootstrap_data),
            ("FM Pipeline", mock_bootstrap_data),
            ("Traditional", mock_bootstrap_data),
        ]

        fig, axes, plot_data = plot_instability_comparison(data_list)

        assert fig is not None
        assert len(axes) == 3
        assert len(plot_data["panels"]) == 3


class TestSaveFigureWithData:
    """Tests for figure saving."""

    def test_save_creates_files(self, mock_bootstrap_data):
        """Test that save function creates all expected files.

        save_figure_with_data delegates to save_figure(), which reads formats
        from configs/VISUALIZATION/figure_layouts.yaml. The default config
        specifies formats: ["png"] only (no PDF). We test for PNG + JSON.
        """
        import matplotlib.pyplot as plt
        from src.viz.fig_instability_plots import (
            plot_prediction_instability,
            save_figure_with_data,
        )

        fig, _, plot_data = plot_prediction_instability(mock_bootstrap_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_figure_with_data(fig, "test_figure", output_dir, plot_data)

            assert (output_dir / "test_figure.png").exists()
            assert (output_dir / "data" / "test_figure.json").exists()

        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
