"""
Test Suite for Configuration Loading.

Tests the config loader in src/viz/config_loader.py.
Run with: pytest tests/unit/test_config_loader.py -v
"""

import pytest
from pathlib import Path

# Add project root to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.viz.config_loader import (
    ConfigLoader,
    get_config_loader,
)


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================


class TestConfigLoader:
    """Test ConfigLoader class."""

    def test_get_config_loader_returns_instance(self):
        """Test that get_config_loader returns a ConfigLoader instance."""
        loader = get_config_loader()
        assert isinstance(loader, ConfigLoader)

    def test_get_config_loader_is_singleton(self):
        """Test that get_config_loader returns the same instance."""
        loader1 = get_config_loader()
        loader2 = get_config_loader()
        assert loader1 is loader2

    def test_get_defaults_returns_dict(self):
        """Test that get_defaults returns a dictionary."""
        loader = get_config_loader()
        defaults = loader.get_defaults()
        assert isinstance(defaults, dict)
        assert len(defaults) > 0

    def test_get_colors_returns_dict(self):
        """Test that get_colors returns a dictionary."""
        loader = get_config_loader()
        colors = loader.get_colors()
        assert isinstance(colors, dict)

    def test_get_combos_returns_dict(self):
        """Test that get_combos returns a dictionary."""
        loader = get_config_loader()
        combos = loader.get_combos()
        assert isinstance(combos, dict)


class TestPrevalenceConfig:
    """Test prevalence configuration."""

    def test_get_prevalence_returns_float(self):
        """Test that get_prevalence returns a float."""
        loader = get_config_loader()
        prevalence = loader.get_prevalence()
        assert isinstance(prevalence, float)

    def test_prevalence_is_reasonable(self):
        """Test that prevalence is in reasonable range (0-1)."""
        loader = get_config_loader()
        prevalence = loader.get_prevalence()
        assert 0 < prevalence < 1, f"Prevalence {prevalence} should be between 0 and 1"

    def test_prevalence_is_tham_2014_value(self):
        """Test that prevalence matches Tham 2014 (3.54% for 40-80 age group)."""
        loader = get_config_loader()
        prevalence = loader.get_prevalence()
        # Should be approximately 0.0354 (3.54%)
        assert (
            0.03 < prevalence < 0.04
        ), f"Prevalence {prevalence} should be ~0.0354 per Tham 2014"


class TestHyperparamCombos:
    """Test hyperparameter combo configuration."""

    def test_get_standard_hyperparam_combos_returns_list(self):
        """Test that get_standard_hyperparam_combos returns a list."""
        loader = get_config_loader()
        combos = loader.get_standard_hyperparam_combos()
        assert isinstance(combos, list)

    def test_standard_combos_has_at_least_4(self):
        """Test that there are at least 4 standard combos."""
        loader = get_config_loader()
        combos = loader.get_standard_hyperparam_combos()
        assert (
            len(combos) >= 4
        ), f"Expected at least 4 standard combos, got {len(combos)}"

    def test_standard_combos_have_required_keys(self):
        """Test that standard combos have required keys."""
        loader = get_config_loader()
        combos = loader.get_standard_hyperparam_combos()

        required_keys = {"id", "outlier_method", "imputation_method", "classifier"}

        for combo in combos:
            missing = required_keys - set(combo.keys())
            assert (
                not missing
            ), f"Combo {combo.get('id', 'UNKNOWN')} missing keys: {missing}"

    def test_ground_truth_combo_exists(self):
        """Test that ground_truth combo exists."""
        loader = get_config_loader()
        combos = loader.get_standard_hyperparam_combos()
        combo_ids = [c.get("id") for c in combos]
        assert (
            "ground_truth" in combo_ids
        ), f"ground_truth combo should exist, found: {combo_ids}"


class TestMetricCombos:
    """Test metric combo configuration."""

    def test_get_metric_combo_returns_list(self):
        """Test that get_metric_combo returns a list of metric names."""
        loader = get_config_loader()
        metrics = loader.get_metric_combo()
        assert isinstance(metrics, list)

    def test_default_metric_combo_has_metrics(self):
        """Test that default metric combo has at least one metric."""
        loader = get_config_loader()
        metrics = loader.get_metric_combo()
        assert len(metrics) > 0, "Default metric combo should have at least one metric"

    def test_get_specific_metric_combo(self):
        """Test getting a specific metric combo."""
        loader = get_config_loader()
        # Should not raise an error for 'default' combo
        metrics = loader.get_metric_combo("default")
        assert isinstance(metrics, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
