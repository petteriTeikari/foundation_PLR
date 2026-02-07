"""
Tests for instability figure generation from MLflow data.
"""

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
        metadata={"display_name": "Test Combo"},
    )


class TestInstabilityFigureGeneration:
    """Tests for instability figure generation functions."""

    def test_generate_instability_comparison(self, mock_bootstrap_data):
        """Test instability comparison figure generation."""
        from src.viz.generate_instability_figures import (
            generate_instability_comparison_figure,
        )

        data_dict = {
            "combo_a": mock_bootstrap_data,
            "combo_b": mock_bootstrap_data,
        }

        fig, plot_data = generate_instability_comparison_figure(data_dict)

        assert fig is not None
        assert "panels" in plot_data
        assert len(plot_data["panels"]) == 2

    def test_generate_per_patient_uncertainty(self, mock_bootstrap_data):
        """Test per-patient uncertainty figure generation."""
        from src.viz.generate_instability_figures import (
            generate_per_patient_uncertainty_figure,
        )

        fig, plot_data = generate_per_patient_uncertainty_figure(mock_bootstrap_data)

        assert fig is not None
        assert "patients" in plot_data
        assert len(plot_data["patients"]) >= 2

    def test_generate_mape_comparison(self, mock_bootstrap_data):
        """Test MAPE comparison figure generation."""
        from src.viz.generate_instability_figures import (
            generate_mape_comparison_figure,
        )

        data_dict = {
            "combo_a": mock_bootstrap_data,
            "combo_b": mock_bootstrap_data,
        }

        fig, plot_data = generate_mape_comparison_figure(data_dict)

        assert fig is not None
        assert "methods" in plot_data
        assert len(plot_data["methods"]) == 2


class TestBootstrapDataLoading:
    """Tests for MLflow data loading functions."""

    def test_load_bootstrap_data_from_run_missing_dir(self):
        """Test loading from non-existent directory."""
        from src.viz.generate_instability_figures import load_bootstrap_data_from_run

        result = load_bootstrap_data_from_run(Path("/nonexistent/path"))
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
