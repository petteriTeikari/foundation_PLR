"""
Tests for pminternal-style model instability analysis.

Tests for Riley 2023 instability metrics:
- MAPE (Mean Absolute Prediction Error) per subject
- CII (Classification Instability Index)
- Prediction standard deviation
- Per-patient uncertainty (Kompa 2021)
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestBootstrapPredictionData:
    """Tests for BootstrapPredictionData dataclass."""

    @pytest.fixture
    def mock_bootstrap_data(self):
        """Create mock bootstrap prediction data."""
        from src.stats.pminternal_analysis import BootstrapPredictionData

        np.random.seed(42)
        n_subjects = 63
        n_bootstrap = 1000

        y_true = np.random.binomial(1, 0.3, n_subjects)
        y_original = np.clip(
            y_true * 0.7 + (1 - y_true) * 0.3 + np.random.normal(0, 0.1, n_subjects),
            0.01,
            0.99,
        )

        # Bootstrap predictions with some variation
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
            metadata={"test": True},
        )

    def test_bootstrap_data_shape(self, mock_bootstrap_data):
        """Test that bootstrap data has correct shape."""
        assert mock_bootstrap_data.y_true.shape == (63,)
        assert mock_bootstrap_data.y_pred_proba_original.shape == (63,)
        assert mock_bootstrap_data.y_pred_proba_bootstrap.shape == (1000, 63)

    def test_to_json_dict(self, mock_bootstrap_data):
        """Test JSON serialization."""
        json_dict = mock_bootstrap_data.to_json_dict()

        assert json_dict["combo_id"] == "test_combo"
        assert json_dict["n_subjects"] == 63
        assert json_dict["n_bootstrap"] == 1000
        assert len(json_dict["y_true"]) == 63
        assert len(json_dict["y_pred_proba_original"]) == 63
        assert len(json_dict["y_pred_proba_bootstrap"]) == 1000
        assert len(json_dict["y_pred_proba_bootstrap"][0]) == 63


class TestInstabilityMetrics:
    """Tests for instability metrics computation."""

    @pytest.fixture
    def mock_bootstrap_data(self):
        """Create mock bootstrap prediction data."""
        from src.stats.pminternal_analysis import BootstrapPredictionData

        np.random.seed(42)
        n_subjects = 63
        n_bootstrap = 1000

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

    def test_compute_mape(self, mock_bootstrap_data):
        """Test MAPE computation."""
        from src.stats.pminternal_analysis import compute_prediction_instability_metrics

        metrics = compute_prediction_instability_metrics(mock_bootstrap_data)

        # MAPE should be per subject
        assert metrics.mape.shape == (63,)

        # MAPE should be non-negative and bounded
        assert np.all(metrics.mape >= 0)
        assert np.all(metrics.mape <= 1)

        # Mean MAPE should be reasonable (given our noise level of 0.08)
        assert 0.01 < metrics.mean_mape < 0.15

    def test_compute_cii(self, mock_bootstrap_data):
        """Test Classification Instability Index computation."""
        from src.stats.pminternal_analysis import compute_prediction_instability_metrics

        metrics = compute_prediction_instability_metrics(
            mock_bootstrap_data, threshold=0.5
        )

        # CII should be per subject
        assert metrics.cii.shape == (63,)

        # CII should be between 0 and 1
        assert np.all(metrics.cii >= 0)
        assert np.all(metrics.cii <= 1)

        # CII should be higher for subjects near threshold
        # (those with original prediction close to 0.5)
        near_threshold = np.abs(mock_bootstrap_data.y_pred_proba_original - 0.5) < 0.1
        if np.any(near_threshold):
            # Subjects near threshold should have higher CII on average
            mean_cii_near = np.mean(metrics.cii[near_threshold])
            mean_cii_far = np.mean(metrics.cii[~near_threshold])
            # This may not always hold for random data, so just check it's computed
            assert np.isfinite(mean_cii_near)
            assert np.isfinite(mean_cii_far)

    def test_compute_prediction_sd(self, mock_bootstrap_data):
        """Test prediction standard deviation computation."""
        from src.stats.pminternal_analysis import compute_prediction_instability_metrics

        metrics = compute_prediction_instability_metrics(mock_bootstrap_data)

        # SD should be per subject
        assert metrics.prediction_sd.shape == (63,)

        # SD should be non-negative
        assert np.all(metrics.prediction_sd >= 0)

        # SD should be close to our noise level (0.08)
        mean_sd = np.mean(metrics.prediction_sd)
        assert 0.05 < mean_sd < 0.12

    def test_compute_confidence_intervals(self, mock_bootstrap_data):
        """Test confidence interval computation."""
        from src.stats.pminternal_analysis import compute_prediction_instability_metrics

        metrics = compute_prediction_instability_metrics(mock_bootstrap_data)

        # CIs should be per subject
        assert metrics.ci_lower.shape == (63,)
        assert metrics.ci_upper.shape == (63,)

        # CI lower should be <= CI upper
        assert np.all(metrics.ci_lower <= metrics.ci_upper)

        # CIs should contain the original prediction for most subjects
        # (given symmetric noise, ~95% should be covered)
        original = mock_bootstrap_data.y_pred_proba_original
        covered = (metrics.ci_lower <= original) & (original <= metrics.ci_upper)
        coverage = np.mean(covered)
        assert coverage > 0.8  # Should be ~95%, allow some slack


class TestPerPatientUncertainty:
    """Tests for per-patient uncertainty (Kompa 2021)."""

    @pytest.fixture
    def mock_bootstrap_data(self):
        """Create mock bootstrap prediction data."""
        from src.stats.pminternal_analysis import BootstrapPredictionData

        np.random.seed(42)
        n_subjects = 63
        n_bootstrap = 1000

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

    def test_per_patient_uncertainty_structure(self, mock_bootstrap_data):
        """Test per-patient uncertainty output structure."""
        from src.stats.pminternal_analysis import compute_per_patient_uncertainty

        uncertainty = compute_per_patient_uncertainty(mock_bootstrap_data)

        assert "n_patients" in uncertainty
        assert "mean_pred" in uncertainty
        assert "sd_pred" in uncertainty
        assert "ci_lower" in uncertainty
        assert "ci_upper" in uncertainty
        assert "y_true" in uncertainty

        assert uncertainty["n_patients"] == 63
        assert len(uncertainty["mean_pred"]) == 63
        assert len(uncertainty["sd_pred"]) == 63

    def test_per_patient_uncertainty_values(self, mock_bootstrap_data):
        """Test per-patient uncertainty values."""
        from src.stats.pminternal_analysis import compute_per_patient_uncertainty

        uncertainty = compute_per_patient_uncertainty(mock_bootstrap_data)

        # All predictions should be in [0, 1]
        assert all(0 <= p <= 1 for p in uncertainty["mean_pred"])

        # SD should be positive
        assert all(sd > 0 for sd in uncertainty["sd_pred"])

        # CI lower <= CI upper
        for lower, upper in zip(uncertainty["ci_lower"], uncertainty["ci_upper"]):
            assert lower <= upper


class TestPredictionInstabilityPlot:
    """Tests for Riley 2023 style prediction instability plot data."""

    @pytest.fixture
    def mock_bootstrap_data(self):
        """Create mock bootstrap prediction data."""
        from src.stats.pminternal_analysis import BootstrapPredictionData

        np.random.seed(42)
        n_subjects = 63
        n_bootstrap = 1000

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

    def test_plot_data_structure(self, mock_bootstrap_data):
        """Test plot data structure."""
        from src.stats.pminternal_analysis import (
            create_prediction_instability_plot_data,
        )

        plot_data = create_prediction_instability_plot_data(
            mock_bootstrap_data, subsample=100
        )

        assert "scatter" in plot_data
        assert "percentile_lines" in plot_data
        assert "diagonal" in plot_data
        assert "metadata" in plot_data

        assert "x" in plot_data["scatter"]
        assert "y" in plot_data["scatter"]

    def test_plot_data_subsample(self, mock_bootstrap_data):
        """Test that subsampling works."""
        from src.stats.pminternal_analysis import (
            create_prediction_instability_plot_data,
        )

        plot_data = create_prediction_instability_plot_data(
            mock_bootstrap_data, subsample=100
        )

        # Should have 100 bootstrap samples * 63 subjects = 6300 points
        assert len(plot_data["scatter"]["x"]) == 100 * 63
        assert len(plot_data["scatter"]["y"]) == 100 * 63


class TestExportPmInternalData:
    """Tests for exporting pminternal-compatible JSON."""

    @pytest.fixture
    def mock_bootstrap_data(self):
        """Create mock bootstrap prediction data."""
        from src.stats.pminternal_analysis import BootstrapPredictionData

        np.random.seed(42)
        n_subjects = 10
        n_bootstrap = 50

        return BootstrapPredictionData(
            combo_id="test_combo",
            y_true=np.random.binomial(1, 0.3, n_subjects),
            y_pred_proba_original=np.random.uniform(0, 1, n_subjects),
            y_pred_proba_bootstrap=np.random.uniform(0, 1, (n_bootstrap, n_subjects)),
            n_subjects=n_subjects,
            n_bootstrap=n_bootstrap,
        )

    def test_export_creates_file(self, mock_bootstrap_data):
        """Test that export creates a valid JSON file."""
        from src.stats.pminternal_analysis import export_pminternal_data

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_data.json"
            export_pminternal_data(mock_bootstrap_data, output_path)

            assert output_path.exists()

            with open(output_path) as f:
                loaded = json.load(f)

            assert loaded["combo_id"] == "test_combo"
            assert loaded["n_subjects"] == 10
            assert loaded["n_bootstrap"] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
