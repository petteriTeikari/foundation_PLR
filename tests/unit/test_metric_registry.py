"""Unit tests for metric registry module."""

import numpy as np
import pytest

from src.viz.metric_registry import (
    MetricDefinition,
    MetricRegistry,
    _compute_auroc,
    _compute_brier,
    _compute_scaled_brier,
    _compute_net_benefit,
    _compute_sensitivity,
    _compute_specificity,
)


class TestMetricDefinition:
    """Tests for MetricDefinition dataclass."""

    def test_format_value_decimal(self):
        """Test formatting decimal values."""
        metric = MetricDefinition(name="test", display_name="Test", format_str=".3f")
        assert metric.format_value(0.12345) == "0.123"

    def test_format_value_percentage(self):
        """Test formatting percentage values."""
        metric = MetricDefinition(
            name="test", display_name="Test", format_str=".1f", unit="%"
        )
        assert metric.format_value(0.856) == "85.6%"

    def test_is_better_higher_is_better(self):
        """Test comparison when higher is better."""
        metric = MetricDefinition(
            name="test", display_name="Test", higher_is_better=True
        )
        assert metric.is_better(0.9, 0.8)
        assert not metric.is_better(0.7, 0.8)

    def test_is_better_lower_is_better(self):
        """Test comparison when lower is better."""
        metric = MetricDefinition(
            name="test", display_name="Test", higher_is_better=False
        )
        assert metric.is_better(0.1, 0.2)
        assert not metric.is_better(0.3, 0.2)


class TestMetricRegistry:
    """Tests for MetricRegistry class."""

    def test_register_and_get(self):
        """Test registering and retrieving a metric."""
        metric = MetricDefinition(name="custom_test", display_name="Custom Test")
        MetricRegistry.register(metric)
        retrieved = MetricRegistry.get("custom_test")
        assert retrieved.name == "custom_test"
        assert retrieved.display_name == "Custom Test"

    def test_get_unknown_metric_raises(self):
        """Test that getting unknown metric raises KeyError."""
        with pytest.raises(KeyError, match="Unknown metric"):
            MetricRegistry.get("nonexistent_metric_xyz")

    def test_get_or_default_known(self):
        """Test get_or_default with known metric."""
        metric = MetricRegistry.get_or_default("auroc")
        assert metric.name == "auroc"

    def test_get_or_default_unknown(self):
        """Test get_or_default creates default for unknown metric."""
        metric = MetricRegistry.get_or_default("some_unknown_column")
        assert metric.name == "some_unknown_column"
        assert metric.display_name == "Some Unknown Column"  # Auto-formatted

    def test_list_metrics(self):
        """Test listing all metrics."""
        metrics = MetricRegistry.list_metrics()
        assert "auroc" in metrics
        assert "brier" in metrics
        assert len(metrics) >= 5

    def test_has_metric(self):
        """Test checking if metric exists."""
        assert MetricRegistry.has("auroc")
        assert not MetricRegistry.has("definitely_not_a_metric")

    def test_standard_metrics_registered(self):
        """Test that standard metrics are pre-registered."""
        expected_metrics = [
            "auroc",
            "brier",
            "scaled_brier",
            "net_benefit",
            "f1",
            "sensitivity",
            "specificity",
            "mae",
        ]
        for name in expected_metrics:
            assert MetricRegistry.has(name), f"Standard metric '{name}' not registered"


class TestMetricComputeFunctions:
    """Tests for metric compute functions."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample prediction data."""
        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.binomial(1, 0.3, n)
        # Create predictions correlated with true labels
        y_prob = np.clip(y_true * 0.6 + rng.uniform(0.2, 0.4, n), 0.01, 0.99)
        return y_true, y_prob

    def test_auroc_perfect(self):
        """Test AUROC with perfect predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert _compute_auroc(y_true, y_prob) == 1.0

    def test_auroc_random(self):
        """Test AUROC with random predictions is around 0.5."""
        rng = np.random.default_rng(42)
        y_true = rng.binomial(1, 0.5, 1000)
        y_prob = rng.uniform(0, 1, 1000)
        auroc = _compute_auroc(y_true, y_prob)
        assert 0.4 < auroc < 0.6

    def test_auroc_single_class_returns_nan(self):
        """Test AUROC returns NaN when only one class present."""
        y_true = np.array([1, 1, 1, 1])
        y_prob = np.array([0.5, 0.6, 0.7, 0.8])
        assert np.isnan(_compute_auroc(y_true, y_prob))

    def test_brier_perfect(self):
        """Test Brier score with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        assert _compute_brier(y_true, y_prob) == 0.0

    def test_brier_worst(self):
        """Test Brier score with worst predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0])
        assert _compute_brier(y_true, y_prob) == 1.0

    def test_scaled_brier_perfect(self):
        """Test scaled Brier (IPA) with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        assert _compute_scaled_brier(y_true, y_prob) == 1.0

    def test_scaled_brier_null_model(self):
        """Test scaled Brier (IPA) with null model predictions."""
        rng = np.random.default_rng(42)
        y_true = rng.binomial(1, 0.3, 1000)
        y_prob = np.full(1000, 0.3)  # Predict prevalence
        ipa = _compute_scaled_brier(y_true, y_prob)
        assert abs(ipa) < 0.1  # Should be close to 0

    def test_net_benefit_all_positive(self):
        """Test net benefit when all predictions are positive."""
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.6])  # All above threshold 0.15
        nb = _compute_net_benefit(y_true, y_prob, threshold=0.15)
        # TP=2, FP=2, n=4
        # NB = 2/4 - 2/4 * (0.15/0.85) = 0.5 - 0.5*0.176 = 0.412
        assert 0.3 < nb < 0.5

    def test_sensitivity_perfect(self):
        """Test sensitivity with perfect recall."""
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.2, 0.1])
        sens = _compute_sensitivity(y_true, y_prob, threshold=0.5)
        assert sens == 1.0

    def test_sensitivity_zero(self):
        """Test sensitivity when no positives are detected."""
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4])  # All below threshold
        sens = _compute_sensitivity(y_true, y_prob, threshold=0.5)
        assert sens == 0.0

    def test_specificity_perfect(self):
        """Test specificity with perfect true negative rate."""
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.2, 0.1])
        spec = _compute_specificity(y_true, y_prob, threshold=0.5)
        assert spec == 1.0

    def test_metric_definition_with_compute_fn(self, sample_data):
        """Test MetricDefinition with compute function."""
        y_true, y_prob = sample_data
        metric = MetricRegistry.get("auroc")
        assert metric.compute_fn is not None
        result = metric.compute_fn(y_true, y_prob)
        assert 0.5 < result <= 1.0  # Should be better than random
