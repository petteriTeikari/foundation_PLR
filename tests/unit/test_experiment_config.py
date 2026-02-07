"""
Unit tests for experiment configuration system.

Tests verify:
1. Experiment configs load correctly
2. Validation catches invalid configs
3. Pydantic models enforce type safety
"""

import pytest
from pathlib import Path


class TestExperimentLoading:
    """Test experiment configuration loading."""

    @pytest.mark.unit
    def test_list_experiments_returns_available(self):
        """Should list available experiment configs."""
        from src.config import list_experiments

        experiments = list_experiments()

        assert isinstance(experiments, list)
        assert "paper_2026" in experiments
        assert "synthetic" in experiments

    @pytest.mark.unit
    def test_load_paper_2026_experiment(self):
        """Should load paper_2026 experiment config."""
        from src.config import load_experiment

        cfg = load_experiment("paper_2026")

        assert cfg.name == "Foundation PLR Paper 2026"
        assert cfg.version == "1.0.0"
        assert cfg.is_frozen is True

    @pytest.mark.unit
    def test_load_synthetic_experiment(self):
        """Should load synthetic experiment config."""
        from src.config import load_experiment

        cfg = load_experiment("synthetic")

        assert cfg.name == "Synthetic Pipeline Test"
        assert cfg.is_frozen is False

    @pytest.mark.unit
    def test_load_experiment_with_yaml_extension(self):
        """Should handle .yaml extension in name."""
        from src.config import load_experiment

        cfg = load_experiment("paper_2026.yaml")

        assert cfg.name == "Foundation PLR Paper 2026"

    @pytest.mark.unit
    def test_load_nonexistent_experiment_raises(self):
        """Should raise FileNotFoundError for missing experiment."""
        from src.config import load_experiment

        with pytest.raises(FileNotFoundError):
            load_experiment("nonexistent_experiment")


class TestExperimentValidation:
    """Test experiment configuration validation."""

    @pytest.mark.unit
    def test_validate_valid_config(self):
        """Should validate existing config files."""
        from src.config import validate_experiment_config

        config_dir = Path(__file__).parent.parent.parent / "configs" / "experiment"

        # All existing configs should validate
        for config_path in config_dir.glob("*.yaml"):
            assert validate_experiment_config(config_path) is True

    @pytest.mark.unit
    def test_validate_missing_file_raises(self):
        """Should raise FileNotFoundError for missing file."""
        from src.config import validate_experiment_config

        with pytest.raises(FileNotFoundError):
            validate_experiment_config("/nonexistent/path.yaml")


class TestExperimentConfigModel:
    """Test ExperimentConfig Pydantic model."""

    @pytest.mark.unit
    def test_experiment_config_minimal(self):
        """Should accept minimal valid config."""
        from src.config.experiment import ExperimentConfig

        cfg = ExperimentConfig(experiment={"name": "Test", "version": "1.0.0"})

        assert cfg.name == "Test"
        assert cfg.version == "1.0.0"

    @pytest.mark.unit
    def test_experiment_config_frozen_default(self):
        """Frozen should default to False."""
        from src.config.experiment import ExperimentConfig

        cfg = ExperimentConfig(experiment={"name": "Test", "version": "1.0.0"})

        assert cfg.is_frozen is False

    @pytest.mark.unit
    def test_experiment_config_with_factorial_design(self):
        """Should parse factorial_design section."""
        from src.config.experiment import ExperimentConfig

        cfg = ExperimentConfig(
            experiment={"name": "Test", "version": "1.0.0"},
            factorial_design={
                "outlier_methods": 11,
                "imputation_methods": 8,
                "classifiers": 5,
            },
        )

        assert cfg.factorial_design is not None
        assert cfg.factorial_design.outlier_methods == 11


class TestYAMLConfigLoader:
    """Test YAML configuration loader."""

    @pytest.mark.unit
    def test_loader_caches_files(self):
        """Should cache loaded files."""
        from src.config import YAMLConfigLoader

        loader = YAMLConfigLoader()

        # First load
        combos1 = loader.load_combos()

        # Second load should use cache
        combos2 = loader.load_combos()

        # Data should be equivalent (MappingProxyType wraps cached dict)
        assert dict(combos1) == dict(combos2)

    @pytest.mark.unit
    def test_loader_clear_cache(self):
        """Should clear cache when requested."""
        from src.config import YAMLConfigLoader

        loader = YAMLConfigLoader()

        # Load and cache
        combos1 = loader.load_combos()

        # Clear cache
        loader.clear_cache()

        # Reload (new object)
        combos2 = loader.load_combos()

        # Different object reference
        assert combos1 is not combos2

    @pytest.mark.unit
    def test_get_combo_by_id(self):
        """Should get specific combo by ID."""
        from src.config import YAMLConfigLoader

        loader = YAMLConfigLoader()

        gt_combo = loader.get_combo_by_id("ground_truth")

        assert gt_combo is not None
        assert gt_combo["outlier_method"] == "pupil-gt"

    @pytest.mark.unit
    def test_get_combo_by_id_not_found(self):
        """Should return None for unknown ID."""
        from src.config import YAMLConfigLoader

        loader = YAMLConfigLoader()

        combo = loader.get_combo_by_id("nonexistent_combo")

        assert combo is None
