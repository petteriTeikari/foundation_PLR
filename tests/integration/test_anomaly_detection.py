"""Integration tests for anomaly detection functionality.

Tests outlier detection algorithms including sklearn LOF
and Prophet wrapper on synthetic and real data.
"""

import numpy as np
import pytest
from sklearn.neighbors import LocalOutlierFactor


class TestLocalOutlierFactor:
    """Tests for sklearn LocalOutlierFactor detector."""

    @pytest.fixture
    def synthetic_data_with_outliers(self):
        """Create synthetic data with known outliers."""
        np.random.seed(42)
        n_samples = 200

        # Normal data (cluster around 0)
        normal_data = np.random.randn(n_samples, 2) * 0.5

        # Add some outliers (far from center)
        n_outliers = 10
        outliers = np.random.randn(n_outliers, 2) * 3 + 5

        data = np.vstack([normal_data, outliers])
        labels = np.array([1] * n_samples + [-1] * n_outliers)  # 1=inlier, -1=outlier

        return data, labels

    @pytest.mark.integration
    def test_lof_detector_runs(self, synthetic_data_with_outliers):
        """Test that LOF detector runs without errors."""
        data, _ = synthetic_data_with_outliers

        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        predictions = lof.fit_predict(data)

        assert predictions is not None
        assert len(predictions) == len(data)

    @pytest.mark.integration
    def test_lof_detects_outliers(self, synthetic_data_with_outliers):
        """Test that LOF detects at least some outliers."""
        data, true_labels = synthetic_data_with_outliers

        # Use contamination that matches our outlier ratio
        contamination = 10 / 210  # 10 outliers out of 210 total

        lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        predictions = lof.fit_predict(data)

        # Check that some outliers are detected
        n_detected_outliers = np.sum(predictions == -1)
        assert n_detected_outliers > 0, "LOF should detect at least some outliers"

    @pytest.mark.integration
    def test_lof_output_format(self, synthetic_data_with_outliers):
        """Test LOF output format matches expected structure."""
        data, _ = synthetic_data_with_outliers

        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        predictions = lof.fit_predict(data)

        # Predictions should be -1 (outlier) or 1 (inlier)
        unique_values = np.unique(predictions)
        assert set(unique_values).issubset({-1, 1})

    @pytest.mark.integration
    def test_lof_on_plr_like_data(self, sample_plr_array):
        """Test LOF on PLR-like time series data."""
        # Flatten to 2D for LOF (subjects x timepoints)
        data_2d = sample_plr_array

        # Add some artificial outliers
        data_with_outliers = data_2d.copy()
        data_with_outliers[0, 500:520] += 10  # Spike in first subject

        # Transpose to have timepoints as samples for anomaly detection
        # This tests point-wise anomaly detection
        data_transposed = data_with_outliers.T  # (timepoints, subjects)

        lof = LocalOutlierFactor(n_neighbors=50, contamination=0.02)
        predictions = lof.fit_predict(data_transposed)

        assert len(predictions) == data_transposed.shape[0]


class TestOutlierMaskFormat:
    """Tests for outlier mask format and structure."""

    @pytest.mark.integration
    def test_outlier_mask_binary(self, sample_outlier_mask):
        """Test that outlier mask contains only 0s and 1s."""
        unique_values = np.unique(sample_outlier_mask)
        assert set(unique_values).issubset({0, 1})

    @pytest.mark.integration
    def test_outlier_mask_shape_matches_data(
        self, sample_plr_array, sample_outlier_mask
    ):
        """Test that outlier mask shape matches data shape."""
        assert sample_outlier_mask.shape == sample_plr_array.shape

    @pytest.mark.integration
    def test_outlier_mask_reasonable_ratio(self, sample_outlier_mask):
        """Test that outlier ratio is reasonable (not too high or low)."""
        outlier_ratio = np.mean(sample_outlier_mask)

        # Typical outlier ratios are 1-10%
        assert outlier_ratio < 0.2, "Outlier ratio seems too high"
        assert outlier_ratio > 0.001, "Outlier ratio seems too low"


class TestProphetAnomalyDetection:
    """Tests for Prophet-based anomaly detection (if available)."""

    @pytest.fixture
    def prophet_available(self):
        """Check if Prophet is available."""
        try:
            from prophet import Prophet  # noqa: F401

            return True
        except ImportError:
            return False

    @pytest.fixture
    def simple_time_series(self):
        """Create simple time series for Prophet testing."""
        import pandas as pd

        n_points = 100
        dates = pd.date_range(start="2020-01-01", periods=n_points, freq="D")

        # Create trend + seasonality + noise
        trend = np.linspace(0, 10, n_points)
        seasonality = 2 * np.sin(np.linspace(0, 4 * np.pi, n_points))
        noise = np.random.randn(n_points) * 0.5

        y = trend + seasonality + noise

        # Add outliers
        outlier_indices = [20, 50, 80]
        y[outlier_indices] += 10

        df = pd.DataFrame({"ds": dates, "y": y})

        return df, outlier_indices

    @pytest.mark.integration
    @pytest.mark.slow
    def test_prophet_runs(self, prophet_available, simple_time_series):
        """Test that Prophet model runs without errors."""
        if not prophet_available:
            pytest.skip("Prophet not installed")

        from prophet import Prophet

        df, _ = simple_time_series

        model = Prophet(
            interval_width=0.95,
            yearly_seasonality=False,
            weekly_seasonality=False,
        )
        model.fit(df)

        forecast = model.predict(df)

        assert forecast is not None
        assert len(forecast) == len(df)
        assert "yhat" in forecast.columns
        assert "yhat_lower" in forecast.columns
        assert "yhat_upper" in forecast.columns

    @pytest.mark.integration
    @pytest.mark.slow
    def test_prophet_detects_anomalies(self, prophet_available, simple_time_series):
        """Test that Prophet can identify anomalies outside prediction intervals."""
        if not prophet_available:
            pytest.skip("Prophet not installed")

        from prophet import Prophet

        df, outlier_indices = simple_time_series

        model = Prophet(
            interval_width=0.95,
            yearly_seasonality=False,
            weekly_seasonality=False,
        )
        model.fit(df)

        forecast = model.predict(df)

        # Find points outside prediction interval
        actual = df["y"].values
        below = actual < forecast["yhat_lower"].values
        above = actual > forecast["yhat_upper"].values
        detected_anomalies = below | above

        # At least one outlier should be detected
        n_detected = np.sum(detected_anomalies)
        assert n_detected > 0, "Prophet should detect at least some anomalies"
