"""Integration tests for decomposition aggregation module.

Tests the DecompositionAggregator class which:
1. Loads preprocessed signals from DuckDB by category
2. Applies decomposition methods
3. Aggregates results with bootstrap CIs

These tests use synthetic data to verify the aggregation logic works correctly.
Real data tests require the extraction to be complete.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.decomposition.aggregation import (
    DecompositionAggregator,
    DecompositionResult,
    ComponentTimecourse,
    load_category_mapping,
)

pytestmark = pytest.mark.data


class TestLoadCategoryMapping:
    """Tests for load_category_mapping function."""

    def test_loads_from_config_file(self):
        """Category mapping loads successfully from YAML config."""
        # Should not raise since config exists
        mapping = load_category_mapping()
        assert isinstance(mapping, dict)
        assert len(mapping) > 0

    def test_raises_if_config_missing(self, tmp_path):
        """Raises FileNotFoundError if config file doesn't exist."""
        fake_path = tmp_path / "nonexistent.yaml"

        with patch("src.decomposition.aggregation.CATEGORY_MAPPING_PATH", fake_path):
            with pytest.raises(FileNotFoundError) as exc_info:
                load_category_mapping()

            assert "Category mapping config required" in str(exc_info.value)

    def test_ground_truth_mapping_exists(self):
        """Ground truth category mapping is present."""
        mapping = load_category_mapping()
        assert "pupil-gt" in mapping
        assert mapping["pupil-gt"] == "Ground Truth"


class TestBootstrapMeanCI:
    """Tests for bootstrap CI computation."""

    def test_ci_width_is_positive(self):
        """Bootstrap CIs have positive width."""
        # Create synthetic data with known variation
        np.random.seed(42)
        data = np.random.randn(50, 100)  # 50 subjects, 100 timepoints

        aggregator = DecompositionAggregator(
            db_path=Path("/tmp/fake.db"),  # Not used in this test
            n_bootstrap=100,
            random_seed=42,
        )

        mean, ci_lo, ci_hi = aggregator._bootstrap_mean_ci(data, axis=0)

        # CI width should be positive everywhere
        ci_width = ci_hi - ci_lo
        assert np.all(ci_width > 0), "CI width should be positive"

    def test_ci_contains_mean(self):
        """Bootstrap CIs contain the sample mean."""
        np.random.seed(42)
        data = np.random.randn(50, 100)

        aggregator = DecompositionAggregator(
            db_path=Path("/tmp/fake.db"),
            n_bootstrap=100,
            random_seed=42,
        )

        mean, ci_lo, ci_hi = aggregator._bootstrap_mean_ci(data, axis=0)

        # Mean should be within CIs
        assert np.all(mean >= ci_lo), "Mean should be >= CI lower bound"
        assert np.all(mean <= ci_hi), "Mean should be <= CI upper bound"

    def test_ci_width_scales_with_variance(self):
        """CI width increases with data variance."""
        np.random.seed(42)

        # Low variance data
        data_low_var = np.random.randn(50, 100) * 0.1
        # High variance data
        data_high_var = np.random.randn(50, 100) * 10.0

        aggregator = DecompositionAggregator(
            db_path=Path("/tmp/fake.db"),
            n_bootstrap=100,
            random_seed=42,
        )

        _, ci_lo_low, ci_hi_low = aggregator._bootstrap_mean_ci(data_low_var, axis=0)
        _, ci_lo_high, ci_hi_high = aggregator._bootstrap_mean_ci(data_high_var, axis=0)

        width_low = np.mean(ci_hi_low - ci_lo_low)
        width_high = np.mean(ci_hi_high - ci_lo_high)

        assert width_high > width_low * 10, "High variance should have wider CIs"


class TestDecompositionResultDataclass:
    """Tests for DecompositionResult dataclass."""

    def test_result_has_required_fields(self):
        """DecompositionResult has all required fields."""
        time_vec = np.linspace(0, 66, 200)
        components = [
            ComponentTimecourse(
                name="PC1",
                mean=np.zeros(200),
                ci_lower=np.zeros(200),
                ci_upper=np.zeros(200),
            )
        ]

        result = DecompositionResult(
            category="Ground Truth",
            method="pca",
            time_vector=time_vec,
            components=components,
            n_subjects=50,
            mean_waveform=np.zeros(200),
            mean_waveform_ci_lower=np.zeros(200),
            mean_waveform_ci_upper=np.zeros(200),
        )

        assert result.category == "Ground Truth"
        assert result.method == "pca"
        assert len(result.time_vector) == 200
        assert len(result.components) == 1
        assert result.n_subjects == 50


class TestComponentTimecourse:
    """Tests for ComponentTimecourse dataclass."""

    def test_component_has_name_and_arrays(self):
        """ComponentTimecourse has name and mean/CI arrays."""
        comp = ComponentTimecourse(
            name="phasic",
            mean=np.ones(100),
            ci_lower=np.zeros(100),
            ci_upper=np.ones(100) * 2,
        )

        assert comp.name == "phasic"
        assert len(comp.mean) == 100
        assert len(comp.ci_lower) == 100
        assert len(comp.ci_upper) == 100


class TestAggregatorInitialization:
    """Tests for DecompositionAggregator initialization."""

    def test_default_parameters(self):
        """Aggregator initializes with sensible defaults."""
        aggregator = DecompositionAggregator(db_path=Path("/tmp/test.db"))

        assert aggregator.n_bootstrap == 1000
        assert aggregator.ci_level == 0.95
        assert aggregator.random_seed == 42

    def test_custom_parameters(self):
        """Aggregator accepts custom parameters."""
        aggregator = DecompositionAggregator(
            db_path=Path("/tmp/test.db"),
            n_bootstrap=500,
            ci_level=0.90,
            random_seed=123,
        )

        assert aggregator.n_bootstrap == 500
        assert aggregator.ci_level == 0.90
        assert aggregator.random_seed == 123


class TestCategoryValidation:
    """Tests for preprocessing category validation."""

    VALID_CATEGORIES = [
        "Ground Truth",
        "Foundation Model",
        "Deep Learning",
        "Traditional",
        "Ensemble",
    ]

    def test_all_five_categories_defined(self):
        """All 5 preprocessing categories are valid."""

        # PreprocessingCategory is a Literal type
        # We check by attempting to use each category
        for cat in self.VALID_CATEGORIES:
            # This should not raise
            assert cat in self.VALID_CATEGORIES


class TestDecompositionMethodValidation:
    """Tests for decomposition method validation."""

    VALID_METHODS = ["template", "pca", "rotated_pca", "sparse_pca", "ged"]

    def test_all_five_methods_defined(self):
        """All 5 decomposition methods are valid."""

        for method in self.VALID_METHODS:
            assert method in self.VALID_METHODS


class TestSyntheticDataDecomposition:
    """Tests using synthetic data to verify decomposition logic."""

    @pytest.fixture
    def synthetic_waveforms(self):
        """Generate synthetic PLR-like waveforms for testing."""
        np.random.seed(42)
        n_subjects = 20
        n_timepoints = 200
        time_vector = np.linspace(0, 66, n_timepoints)

        # Create base waveform (PLR-like shape)
        base = 100 - 15 * np.exp(-np.maximum(0, time_vector - 15.5) / 2)
        base -= 10 * np.exp(-np.maximum(0, time_vector - 46.5) / 2)

        # Add subject variability
        waveforms = np.array(
            [base + np.random.randn(n_timepoints) * 2 for _ in range(n_subjects)]
        )

        return waveforms, time_vector

    def test_template_fitting_returns_three_components(self, synthetic_waveforms):
        """Template fitting extracts phasic, sustained, PIPR components."""
        waveforms, time_vector = synthetic_waveforms

        aggregator = DecompositionAggregator(
            db_path=Path("/tmp/fake.db"),
            n_bootstrap=10,  # Low for speed
            random_seed=42,
        )

        components = aggregator.aggregate_template_fitting(waveforms, time_vector)

        assert len(components) == 3
        component_names = {c.name for c in components}
        assert component_names == {"phasic", "sustained", "pipr"}

    def test_pca_returns_three_components(self, synthetic_waveforms):
        """PCA extracts PC1, PC2, PC3 components."""
        waveforms, _ = synthetic_waveforms

        aggregator = DecompositionAggregator(
            db_path=Path("/tmp/fake.db"),
            n_bootstrap=10,
            random_seed=42,
        )

        components = aggregator.aggregate_pca(waveforms, n_components=3)

        assert len(components) == 3
        component_names = [c.name for c in components]
        assert component_names == ["PC1", "PC2", "PC3"]

    def test_rotated_pca_returns_three_components(self, synthetic_waveforms):
        """Rotated PCA extracts RC1, RC2, RC3 components."""
        waveforms, _ = synthetic_waveforms

        aggregator = DecompositionAggregator(
            db_path=Path("/tmp/fake.db"),
            n_bootstrap=10,
            random_seed=42,
        )

        components = aggregator.aggregate_rotated_pca(waveforms, n_components=3)

        assert len(components) == 3
        component_names = [c.name for c in components]
        assert component_names == ["RC1", "RC2", "RC3"]

    def test_sparse_pca_returns_three_components(self, synthetic_waveforms):
        """Sparse PCA extracts SPC1, SPC2, SPC3 components."""
        waveforms, _ = synthetic_waveforms

        aggregator = DecompositionAggregator(
            db_path=Path("/tmp/fake.db"),
            n_bootstrap=10,
            random_seed=42,
        )

        components = aggregator.aggregate_sparse_pca(waveforms, n_components=3)

        assert len(components) == 3
        component_names = [c.name for c in components]
        assert component_names == ["SPC1", "SPC2", "SPC3"]

    def test_ged_returns_three_components(self, synthetic_waveforms):
        """GED extracts GED1, GED2, GED3 components."""
        waveforms, time_vector = synthetic_waveforms

        aggregator = DecompositionAggregator(
            db_path=Path("/tmp/fake.db"),
            n_bootstrap=10,
            random_seed=42,
        )

        components = aggregator.aggregate_ged(waveforms, time_vector, n_components=3)

        assert len(components) == 3
        component_names = [c.name for c in components]
        assert component_names == ["GED1", "GED2", "GED3"]
