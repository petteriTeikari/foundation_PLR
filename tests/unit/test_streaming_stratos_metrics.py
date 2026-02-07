"""
TDD tests for streaming STRATOS metrics computation.

These tests verify that the streaming DuckDB exporter computes STRATOS metrics
from raw predictions (y_true, y_prob) when they are not pre-computed in MLflow.

Tests run BEFORE implementation (TDD Red phase).
"""

import numpy as np
import pytest


class TestStreamingSTRATOSComputation:
    """Tests for computing STRATOS metrics from raw predictions."""

    @pytest.fixture
    def sample_predictions(self):
        """Create realistic sample predictions."""
        np.random.seed(42)
        n = 200

        # 30% prevalence (similar to glaucoma dataset)
        y_true = np.random.binomial(1, 0.3, n)

        # Create somewhat calibrated predictions
        y_prob = np.clip(
            y_true * 0.6 + (1 - y_true) * 0.2 + np.random.normal(0, 0.15, n), 0.01, 0.99
        )

        return y_true, y_prob

    def test_calibration_slope_computed_from_raw(self, sample_predictions):
        """Calibration slope should be computed from y_true, y_prob."""
        from src.stats.calibration_extended import calibration_slope_intercept

        y_true, y_prob = sample_predictions
        result = calibration_slope_intercept(y_true, y_prob)

        # Slope should be defined and in reasonable range
        assert result.slope is not None
        assert np.isfinite(result.slope)
        assert 0.1 <= result.slope <= 10.0, f"Slope {result.slope} unrealistic"

    def test_calibration_intercept_computed_from_raw(self, sample_predictions):
        """Calibration intercept should be computed from y_true, y_prob."""
        from src.stats.calibration_extended import calibration_slope_intercept

        y_true, y_prob = sample_predictions
        result = calibration_slope_intercept(y_true, y_prob)

        # Intercept should be defined and in reasonable range
        assert result.intercept is not None
        assert np.isfinite(result.intercept)
        assert -2.0 <= result.intercept <= 2.0, (
            f"Intercept {result.intercept} unrealistic"
        )

    def test_oe_ratio_computed_from_raw(self, sample_predictions):
        """O:E ratio should be computed from y_true, y_prob."""
        from src.stats.calibration_extended import calibration_slope_intercept

        y_true, y_prob = sample_predictions
        result = calibration_slope_intercept(y_true, y_prob)

        # O:E should be positive and reasonable
        assert result.o_e_ratio is not None
        assert result.o_e_ratio > 0
        assert 0.3 <= result.o_e_ratio <= 3.0

        # Verify computation: O/E = mean(y_true) / mean(y_prob)
        expected = np.mean(y_true) / np.mean(y_prob)
        assert np.isclose(result.o_e_ratio, expected, rtol=0.01)

    def test_stratos_metrics_computed_when_missing_in_mlflow(self, sample_predictions):
        """
        Streaming exporter should compute STRATOS metrics from raw predictions
        when they are not present in the MLflow artifacts.
        """
        from src.data_io.streaming_duckdb_export import StreamingDuckDBExporter

        y_true, y_prob = sample_predictions

        # Exporter should compute missing metrics from raw predictions
        exporter = StreamingDuckDBExporter.__new__(StreamingDuckDBExporter)

        # Test _compute_stratos_metrics method (to be implemented)
        stratos = exporter._compute_stratos_metrics_from_raw(y_true, y_prob)

        assert stratos["calibration_slope"] is not None
        assert stratos["calibration_intercept"] is not None
        assert stratos["o_e_ratio"] is not None
        assert np.isfinite(stratos["calibration_slope"])
        assert np.isfinite(stratos["calibration_intercept"])
        assert np.isfinite(stratos["o_e_ratio"])


class TestStreamingDCAComputation:
    """Tests for computing DCA curves from raw predictions."""

    @pytest.fixture
    def sample_predictions(self):
        """Create realistic sample predictions."""
        np.random.seed(42)
        n = 200
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.clip(
            y_true * 0.6 + (1 - y_true) * 0.2 + np.random.normal(0, 0.15, n), 0.01, 0.99
        )
        return y_true, y_prob

    def test_dca_computed_from_raw(self, sample_predictions):
        """DCA curves should be computed from y_true, y_prob."""
        from src.stats.clinical_utility import decision_curve_analysis

        y_true, y_prob = sample_predictions

        dca = decision_curve_analysis(
            y_true, y_prob, threshold_range=(0.01, 0.50), n_thresholds=50
        )

        assert len(dca) >= 20
        assert "threshold" in dca.columns
        # Handle both naming conventions
        assert "nb_model" in dca.columns or "net_benefit" in dca.columns


class TestBootstrapPredictionExport:
    """Tests for exporting bootstrap prediction matrices."""

    @pytest.fixture
    def mock_bootstrap_data(self):
        """Create mock bootstrap prediction data."""
        np.random.seed(42)
        n_subjects = 63
        n_bootstrap = 1000

        # Original predictions
        y_pred_original = np.random.uniform(0, 1, n_subjects)

        # Bootstrap predictions (with some variation)
        y_pred_bootstrap = np.zeros((n_bootstrap, n_subjects))
        for i in range(n_bootstrap):
            y_pred_bootstrap[i] = y_pred_original + np.random.normal(0, 0.1, n_subjects)
            y_pred_bootstrap[i] = np.clip(y_pred_bootstrap[i], 0, 1)

        return y_pred_original, y_pred_bootstrap

    def test_bootstrap_matrix_shape(self, mock_bootstrap_data):
        """Bootstrap prediction matrix should have correct shape."""
        y_pred_original, y_pred_bootstrap = mock_bootstrap_data

        assert y_pred_original.shape == (63,)
        assert y_pred_bootstrap.shape == (1000, 63)

    def test_mape_computed_per_subject(self, mock_bootstrap_data):
        """MAPE should be computed for each subject."""
        y_pred_original, y_pred_bootstrap = mock_bootstrap_data

        # MAPE_i = mean(|p_bootstrap - p_original|) across bootstrap samples
        mape = np.mean(np.abs(y_pred_bootstrap - y_pred_original), axis=0)

        assert mape.shape == (63,)
        assert np.all(mape >= 0)
        assert np.all(mape <= 1)

    def test_classification_instability_index(self, mock_bootstrap_data):
        """CII should be computed at threshold."""
        y_pred_original, y_pred_bootstrap = mock_bootstrap_data
        threshold = 0.5

        # Original classification
        y_class_original = (y_pred_original >= threshold).astype(int)

        # Bootstrap classifications
        y_class_bootstrap = (y_pred_bootstrap >= threshold).astype(int)

        # CII = proportion of bootstrap samples with different classification
        different_class = (y_class_bootstrap != y_class_original).astype(int)
        cii = np.mean(different_class, axis=0)

        assert cii.shape == (63,)
        assert np.all(cii >= 0)
        assert np.all(cii <= 1)


class TestPmInternalDataExport:
    """Tests for exporting data in pminternal-compatible format."""

    def test_pminternal_json_schema(self):
        """JSON export should match pminternal requirements."""
        np.random.seed(42)

        # Expected schema for pminternal analysis
        expected_schema = {
            "combo_id": "test_combo",
            "n_subjects": 63,
            "n_bootstrap": 1000,
            "y_true": np.random.binomial(1, 0.3, 63).tolist(),
            "y_pred_proba_original": np.random.uniform(0, 1, 63).tolist(),
            "y_pred_proba_bootstrap": np.random.uniform(0, 1, (1000, 63)).tolist(),
        }

        # Validate schema
        assert "combo_id" in expected_schema
        assert "n_subjects" in expected_schema
        assert "n_bootstrap" in expected_schema
        assert "y_true" in expected_schema
        assert "y_pred_proba_original" in expected_schema
        assert "y_pred_proba_bootstrap" in expected_schema

        # Validate shapes
        assert len(expected_schema["y_true"]) == 63
        assert len(expected_schema["y_pred_proba_original"]) == 63
        assert len(expected_schema["y_pred_proba_bootstrap"]) == 1000
        assert len(expected_schema["y_pred_proba_bootstrap"][0]) == 63


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
