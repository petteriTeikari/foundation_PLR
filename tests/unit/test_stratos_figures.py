"""Unit tests for STRATOS-compliant figure generation."""

import json
from pathlib import Path

import numpy as np
import pytest


class TestSTRATOSMetrics:
    """Test STRATOS metrics computation."""

    def test_calibration_slope_intercept_perfect(self):
        """Test calibration metrics with perfectly calibrated predictions."""
        from src.stats.calibration_extended import calibration_slope_intercept

        # Perfect calibration: predicted == actual frequencies
        np.random.seed(42)
        n = 200
        y_true = np.random.binomial(1, 0.3, n)
        # Perfect prediction would be y_true probabilities
        y_prob = 0.3 * np.ones(n)

        result = calibration_slope_intercept(y_true, y_prob)

        assert hasattr(result, "slope")
        assert hasattr(result, "intercept")
        assert hasattr(result, "o_e_ratio")
        assert hasattr(result, "brier_score")

    def test_calibration_slope_intercept_overfitting(self):
        """Test calibration with overfitting (predictions too extreme)."""
        from src.stats.calibration_extended import calibration_slope_intercept

        np.random.seed(42)
        n = 200
        y_true = np.random.binomial(1, 0.3, n)
        # Overconfident predictions (too extreme)
        y_prob = np.where(y_true == 1, 0.9, 0.1)

        result = calibration_slope_intercept(y_true, y_prob)

        # Should return valid slope (high slope indicates overconfident predictions)
        assert result.slope > 0  # Positive slope
        assert np.isfinite(result.slope)

    def test_net_benefit_basic(self):
        """Test net benefit computation at a threshold."""
        from src.stats.clinical_utility import net_benefit

        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.6, 0.8, 0.9])

        nb = net_benefit(y_true, y_prob, threshold=0.5)

        # With threshold 0.5: TP=3, FP=0
        # NB = 3/5 - 0/5 * (0.5/0.5) = 0.6
        assert isinstance(nb, float)
        assert nb >= 0

    def test_decision_curve_analysis_output_format(self):
        """Test DCA returns correct DataFrame format."""
        from src.stats.clinical_utility import decision_curve_analysis

        np.random.seed(42)
        n = 100
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.random.uniform(0, 1, n)

        dca_df = decision_curve_analysis(y_true, y_prob, n_thresholds=20)

        assert "threshold" in dca_df.columns
        assert "nb_model" in dca_df.columns
        assert "nb_all" in dca_df.columns
        assert "sensitivity" in dca_df.columns
        assert "specificity" in dca_df.columns
        assert "model_useful" in dca_df.columns
        assert len(dca_df) == 20


class TestSTRATOSFigures:
    """Test STRATOS figure generation."""

    @pytest.fixture
    def mock_combos_data(self):
        """Create mock combos data for testing."""
        np.random.seed(42)
        n = 63

        def make_combo():
            y_true = np.random.binomial(1, 0.27, n)
            y_prob = np.clip(
                y_true * np.random.uniform(0.6, 0.9, n)
                + (1 - y_true) * np.random.uniform(0.1, 0.4, n),
                0,
                1,
            )
            return {
                "y_true": y_true,
                "y_prob": y_prob,
                "n_samples": n,
                "calibration_slope": np.random.uniform(0.8, 1.2),
                "calibration_intercept": np.random.uniform(-0.2, 0.2),
                "o_e_ratio": np.random.uniform(0.9, 1.1),
                "auroc": np.random.uniform(0.85, 0.95),
                "net_benefit_10pct": np.random.uniform(0.1, 0.3),
            }

        return {
            "ground_truth": make_combo(),
            "best_ensemble": make_combo(),
            "best_single_fm": make_combo(),
            "traditional": make_combo(),
        }

    def test_calibration_figure_generation(self, mock_combos_data, tmp_path):
        """Test calibration figure generation."""
        from src.viz.stratos_figures import generate_calibration_stratos_figure

        pdf_path, json_path = generate_calibration_stratos_figure(
            mock_combos_data, output_dir=str(tmp_path), filename="test_calibration"
        )

        # Check files exist
        assert Path(pdf_path).exists()
        assert Path(json_path).exists()

        # Check JSON format
        with open(json_path) as f:
            data = json.load(f)

        assert "combos" in data
        assert len(data["combos"]) == 4
        for combo_id, combo_data in data["combos"].items():
            assert "n_samples" in combo_data
            assert "y_true" in combo_data
            assert "y_prob" in combo_data

    def test_dca_figure_generation(self, mock_combos_data, tmp_path):
        """Test DCA figure generation."""
        from src.viz.stratos_figures import generate_dca_stratos_figure

        pdf_path, json_path = generate_dca_stratos_figure(
            mock_combos_data, output_dir=str(tmp_path), filename="test_dca"
        )

        # Check files exist
        assert Path(pdf_path).exists()
        assert Path(json_path).exists()

        # Check JSON format
        with open(json_path) as f:
            data = json.load(f)

        assert "thresholds" in data
        assert "nb_all" in data
        assert "nb_none" in data
        assert "combos" in data
        assert len(data["thresholds"]) > 0

    def test_calibration_scatter_generation(self, mock_combos_data, tmp_path):
        """Test calibration metrics scatter plot generation."""
        from src.viz.stratos_figures import generate_calibration_metrics_scatter

        pdf_path, json_path = generate_calibration_metrics_scatter(
            mock_combos_data, output_dir=str(tmp_path), filename="test_scatter"
        )

        # Check files exist
        assert Path(pdf_path).exists()
        assert Path(json_path).exists()

        # Check JSON format
        with open(json_path) as f:
            data = json.load(f)

        assert "ideal_slope" in data
        assert data["ideal_slope"] == 1.0
        assert "ideal_oe_ratio" in data
        assert data["ideal_oe_ratio"] == 1.0
        assert "combos" in data

    def test_probability_distribution_generation(self, mock_combos_data, tmp_path):
        """Test probability distribution figure generation."""
        from src.viz.stratos_figures import generate_probability_distribution

        pdf_path, json_path = generate_probability_distribution(
            mock_combos_data, output_dir=str(tmp_path), filename="test_prob_dist"
        )

        # Check files exist
        assert Path(pdf_path).exists()
        assert Path(json_path).exists()

        # Check JSON format
        with open(json_path) as f:
            data = json.load(f)

        assert len(data) == 4  # 4 combos
        for combo_id, combo_data in data.items():
            assert "bins" in combo_data
            assert "controls_probs" in combo_data
            assert "cases_probs" in combo_data


class TestMetricsConfig:
    """Test STRATOS metrics configuration."""

    def test_stratos_metrics_in_config(self):
        """Test that STRATOS metrics are defined in config."""
        from src.viz.config_loader import get_config_loader

        config = get_config_loader()
        metrics_config = config.get_metrics_config()

        # Check STRATOS metrics exist
        stratos_metrics = [
            "smooth_ece",
            "calibration_slope",
            "calibration_intercept",
            "o_e_ratio",
            "net_benefit_5pct",
            "net_benefit_10pct",
            "net_benefit_20pct",
        ]

        for metric in stratos_metrics:
            assert (
                metric in metrics_config["metrics"]
            ), f"Missing STRATOS metric: {metric}"

    def test_stratos_combo_exists(self):
        """Test that STRATOS combo is defined."""
        from src.viz.config_loader import get_config_loader

        config = get_config_loader()
        stratos_metrics = config.get_metric_combo("stratos")

        assert len(stratos_metrics) == 4
        assert "smooth_ece" in stratos_metrics
        assert "calibration_slope" in stratos_metrics

    def test_dca_combo_exists(self):
        """Test that DCA combo is defined."""
        from src.viz.config_loader import get_config_loader

        config = get_config_loader()
        dca_metrics = config.get_metric_combo("dca")

        assert len(dca_metrics) == 4
        assert "net_benefit" in dca_metrics


class TestDuckDBExport:
    """Test DuckDB export with STRATOS metrics."""

    def test_stratos_metrics_schema(self):
        """Test that metrics_per_fold has STRATOS columns."""
        from src.data_io.duckdb_export import RESULTS_SCHEMA

        stratos_cols = [
            "calibration_slope",
            "calibration_intercept",
            "e_o_ratio",
            "net_benefit_5pct",
            "net_benefit_10pct",
            "net_benefit_20pct",
        ]

        for col in stratos_cols:
            assert col in RESULTS_SCHEMA, f"Missing STRATOS column in schema: {col}"

    def test_compute_stratos_metrics(self):
        """Test _compute_stratos_metrics function."""
        from src.data_io.duckdb_export import _compute_stratos_metrics

        np.random.seed(42)
        n = 100
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.clip(
            y_true * 0.7 + (1 - y_true) * 0.3 + np.random.normal(0, 0.1, n), 0, 1
        )

        result = _compute_stratos_metrics(y_true, y_prob)

        assert "calibration_slope" in result
        assert "calibration_intercept" in result
        assert "e_o_ratio" in result
        assert "net_benefit_5pct" in result
        assert "net_benefit_10pct" in result
        assert "net_benefit_20pct" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
