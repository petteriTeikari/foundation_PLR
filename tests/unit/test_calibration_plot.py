"""Unit tests for calibration plot visualization."""

import json
import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for CI
import matplotlib.pyplot as plt


class TestCalibrationPlot:
    """Tests for calibration plot visualization."""

    @pytest.fixture
    def sample_data(self):
        """Generate realistic sample data for testing."""
        rng = np.random.default_rng(42)
        n = 200
        # Simulate predictions with ~80% discrimination
        y_true = rng.binomial(1, 0.3, n)  # 30% prevalence like our data
        # Create reasonably calibrated predictions
        y_prob = np.clip(
            y_true * 0.7 + (1 - y_true) * 0.3 + rng.normal(0, 0.15, n), 0.01, 0.99
        )
        return {"y_true": y_true, "y_prob": y_prob}

    @pytest.fixture
    def perfectly_calibrated_data(self):
        """Generate perfectly calibrated data."""
        rng = np.random.default_rng(42)
        n = 500
        y_prob = rng.uniform(0.05, 0.95, n)
        y_true = rng.binomial(1, y_prob)
        return {"y_true": y_true, "y_prob": y_prob}

    @pytest.fixture
    def overconfident_data(self):
        """Generate overconfident (miscalibrated) data."""
        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.binomial(1, 0.3, n)
        # Overconfident: predictions pushed to extremes
        y_prob = np.clip(
            y_true * 0.9 + (1 - y_true) * 0.1 + rng.normal(0, 0.05, n), 0.01, 0.99
        )
        return {"y_true": y_true, "y_prob": y_prob}

    def test_returns_figure(self, sample_data):
        """Test that function returns a valid matplotlib Figure."""
        from src.viz.calibration_plot import plot_calibration_curve

        fig, ax = plot_calibration_curve(**sample_data)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_output_file_created(self, sample_data, tmp_path):
        """Test that figure can be saved to file."""
        from src.viz.calibration_plot import plot_calibration_curve

        fig, ax = plot_calibration_curve(**sample_data)
        output_path = tmp_path / "test_calibration.pdf"
        fig.savefig(output_path)
        assert output_path.exists()
        plt.close(fig)

    def test_loess_smoothing(self, sample_data):
        """Test LOESS smoothing is applied."""
        from src.viz.calibration_plot import compute_loess_calibration

        y_true = sample_data["y_true"]
        y_prob = sample_data["y_prob"]

        x_smooth, y_smooth = compute_loess_calibration(y_true, y_prob)

        # Should return smoothed values
        assert len(x_smooth) > 0
        assert len(y_smooth) == len(x_smooth)
        # Smoothed values should be in valid range
        assert np.all((y_smooth >= 0) & (y_smooth <= 1))

    def test_reference_line_45_degree(self, sample_data):
        """Test that 45-degree reference line is included."""
        from src.viz.calibration_plot import plot_calibration_curve

        fig, ax = plot_calibration_curve(**sample_data)

        # Check for diagonal line
        lines = ax.get_lines()
        has_diagonal = False
        for line in lines:
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            if len(x_data) == 2 and len(y_data) == 2:
                if (
                    x_data[0] == 0
                    and x_data[1] == 1
                    and y_data[0] == 0
                    and y_data[1] == 1
                ):
                    has_diagonal = True
                    break

        assert has_diagonal, "45-degree reference line should be included"
        plt.close(fig)

    def test_histogram_rug(self, sample_data):
        """Test that histogram rug for distribution is included."""
        from src.viz.calibration_plot import plot_calibration_curve

        fig, ax = plot_calibration_curve(**sample_data, show_rug=True)

        # Check for rug marks or histogram
        # The plot should have more than just the main line
        children = ax.get_children()
        # There should be scatter points or bars for the rug
        assert len(children) > 3  # At least lines + rug elements
        plt.close(fig)

    def test_confidence_intervals(self, sample_data):
        """Test confidence interval computation."""
        from src.viz.calibration_plot import compute_calibration_ci

        y_true = sample_data["y_true"]
        y_prob = sample_data["y_prob"]

        x_vals, y_lower, y_upper = compute_calibration_ci(
            y_true, y_prob, n_bootstrap=50
        )

        # CI should be valid
        assert len(x_vals) > 0
        assert np.all(y_lower <= y_upper)
        assert np.all((y_lower >= 0) & (y_upper <= 1))

    def test_multiple_models(self, sample_data):
        """Test plotting multiple models on same axes."""
        from src.viz.calibration_plot import plot_calibration_multi_model

        rng = np.random.default_rng(123)

        models_data = {
            "Model A": sample_data,
            "Model B": {
                "y_true": sample_data["y_true"],
                "y_prob": np.clip(
                    sample_data["y_prob"]
                    + rng.normal(0, 0.1, len(sample_data["y_prob"])),
                    0.01,
                    0.99,
                ),
            },
        }

        fig, ax = plot_calibration_multi_model(models_data)

        # Should have multiple lines
        lines = ax.get_lines()
        assert len(lines) >= 3  # At least 2 models + reference line
        plt.close(fig)

    def test_plot_with_precomputed_metrics(self, sample_data):
        """Test that plot_calibration_curve accepts pre-computed metrics from DuckDB."""
        from src.viz.calibration_plot import plot_calibration_curve

        # Simulate metrics that would come from DuckDB
        metrics = {
            "calibration_slope": 0.95,
            "calibration_intercept": -0.02,
        }

        fig, ax = plot_calibration_curve(
            **sample_data, show_metrics=True, metrics=metrics
        )
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_without_metrics_skips_annotation(self, sample_data):
        """Test that show_metrics=True with metrics=None skips annotation."""
        from src.viz.calibration_plot import plot_calibration_curve

        # Should not raise even though metrics is None
        fig, ax = plot_calibration_curve(**sample_data, show_metrics=True, metrics=None)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_axes_labels(self, sample_data):
        """Test that axes have proper labels."""
        from src.viz.calibration_plot import plot_calibration_curve

        fig, ax = plot_calibration_curve(**sample_data)

        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close(fig)

    def test_json_data_export(self, sample_data, tmp_path):
        """Test JSON reproducibility data is saved."""
        from src.viz.calibration_plot import plot_calibration_curve

        output_path = tmp_path / "test_calibration.pdf"

        fig, ax = plot_calibration_curve(**sample_data, save_path=str(output_path))
        fig.savefig(output_path)

        json_path = tmp_path / "test_calibration.json"
        assert json_path.exists(), "JSON data file must be created for reproducibility"

        with open(json_path) as f:
            data = json.load(f)

        assert "y_true" in data
        assert "y_prob" in data
        plt.close(fig)

    def test_perfectly_calibrated_close_to_diagonal(self, perfectly_calibrated_data):
        """Test that perfectly calibrated data lies close to diagonal."""
        from src.viz.calibration_plot import compute_loess_calibration

        y_true = perfectly_calibrated_data["y_true"]
        y_prob = perfectly_calibrated_data["y_prob"]

        x_smooth, y_smooth = compute_loess_calibration(y_true, y_prob)

        # For well-calibrated data, smoothed y should be close to x
        residuals = np.abs(y_smooth - x_smooth)
        mean_deviation = np.mean(residuals)
        assert mean_deviation < 0.15, (
            f"Well-calibrated data should be close to diagonal, got mean deviation {mean_deviation}"
        )

    def test_overconfident_detection_via_loess(self, overconfident_data):
        """Test that overconfident predictions show deviation from diagonal via LOESS."""
        from src.viz.calibration_plot import compute_loess_calibration

        y_true = overconfident_data["y_true"]
        y_prob = overconfident_data["y_prob"]

        x_smooth, y_smooth = compute_loess_calibration(y_true, y_prob)

        # Overconfident model: LOESS curve deviates from diagonal
        # ICI (Integrated Calibration Index) = mean |y_smooth - x_smooth|
        ici = np.mean(np.abs(y_smooth - x_smooth))
        assert ici > 0.01, "Should detect miscalibration via LOESS deviation"
