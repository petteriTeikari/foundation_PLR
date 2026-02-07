"""Integration tests for MLflow synthetic data isolation.

Tests that synthetic runs are properly prefixed and tagged in MLflow,
and that they can be reliably filtered out during production extraction.

Part of Phase 2 of the 4-gate isolation architecture.
"""

import pytest
from omegaconf import OmegaConf


class TestMlflowRunNaming:
    """Tests for MLflow run naming with synthetic prefixes."""

    @pytest.mark.integration
    def test_experiment_name_wrapper_adds_synthetic_prefix(self):
        """Test that experiment_name_wrapper adds synthetic prefix when configured."""
        from src.log_helpers.log_naming_uris_and_dirs import experiment_name_wrapper

        cfg = OmegaConf.create(
            {
                "EXPERIMENT": {
                    "use_demo_data": False,
                    "debug": False,
                    "is_synthetic": True,
                }
            }
        )

        result = experiment_name_wrapper("PLR_Classification", cfg)

        assert result.startswith("synth_") or "__SYNTHETIC_" in result

    @pytest.mark.integration
    def test_experiment_name_wrapper_no_prefix_for_production(self):
        """Test that production runs don't get synthetic prefix."""
        from src.log_helpers.log_naming_uris_and_dirs import experiment_name_wrapper

        cfg = OmegaConf.create(
            {
                "EXPERIMENT": {
                    "use_demo_data": False,
                    "debug": False,
                    "is_synthetic": False,
                }
            }
        )

        result = experiment_name_wrapper("PLR_Classification", cfg)

        assert not result.startswith("synth_")
        assert "__SYNTHETIC_" not in result


class TestMlflowTagging:
    """Tests for MLflow tag utilities."""

    @pytest.mark.integration
    def test_get_run_tags_for_synthetic_mode(self):
        """Test that synthetic mode produces correct tags."""
        from src.utils.data_mode import get_mlflow_tags_for_mode

        tags = get_mlflow_tags_for_mode(synthetic=True)

        assert tags["is_synthetic"] == "true"
        assert tags["data_source"] == "synthetic"

    @pytest.mark.integration
    def test_get_run_tags_for_production_mode(self):
        """Test that production mode produces correct tags."""
        from src.utils.data_mode import get_mlflow_tags_for_mode

        tags = get_mlflow_tags_for_mode(synthetic=False)

        assert tags["is_synthetic"] == "false"
        assert tags["data_source"] == "production"


class TestRunNameValidation:
    """Tests for run name validation in extraction context."""

    @pytest.mark.integration
    def test_validate_run_for_production_rejects_synthetic_prefix(self):
        """Test that production extraction rejects __SYNTHETIC_ prefix."""
        from src.utils.data_mode import validate_run_for_production_extraction

        result = validate_run_for_production_extraction(
            run_name="__SYNTHETIC_LOF",
            experiment_name="PLR_Classification",
        )

        assert result is False

    @pytest.mark.integration
    def test_validate_run_for_production_rejects_synthetic_experiment(self):
        """Test that production extraction rejects synth_ experiment prefix."""
        from src.utils.data_mode import validate_run_for_production_extraction

        result = validate_run_for_production_extraction(
            run_name="LOF",
            experiment_name="synth_PLR_Classification",
        )

        assert result is False

    @pytest.mark.integration
    def test_validate_run_for_production_rejects_synthetic_tag(self):
        """Test that production extraction rejects is_synthetic=true tag."""
        from src.utils.data_mode import validate_run_for_production_extraction

        result = validate_run_for_production_extraction(
            run_name="LOF",
            experiment_name="PLR_Classification",
            tags={"is_synthetic": "true"},
        )

        assert result is False

    @pytest.mark.integration
    def test_validate_run_for_production_accepts_production_run(self):
        """Test that production extraction accepts valid production runs."""
        from src.utils.data_mode import validate_run_for_production_extraction

        result = validate_run_for_production_extraction(
            run_name="LOF",
            experiment_name="PLR_Classification",
            tags={"is_synthetic": "false", "data_source": "production"},
        )

        assert result is True


class TestSyntheticRunConfig:
    """Tests for synthetic_run.yaml configuration."""

    @pytest.fixture
    def synthetic_config(self):
        """Load synthetic run configuration."""
        import yaml
        from pathlib import Path

        config_path = (
            Path(__file__).parent.parent.parent / "configs" / "synthetic_run.yaml"
        )
        with open(config_path) as f:
            return yaml.safe_load(f)

    @pytest.mark.integration
    def test_synthetic_config_has_experiment_prefix(self, synthetic_config):
        """Test that synthetic_run.yaml has synth_ prefix."""
        assert synthetic_config["EXPERIMENT"]["experiment_prefix"] == "synth_"

    @pytest.mark.integration
    def test_synthetic_config_experiment_name(self, synthetic_config):
        """Test that synthetic_run.yaml has synth experiment name."""
        assert "synth" in synthetic_config["MLFLOW"]["experiment_name"].lower()

    @pytest.mark.integration
    def test_synthetic_config_data_path(self, synthetic_config):
        """Test that synthetic_run.yaml points to synthetic data."""
        assert "synthetic" in synthetic_config["DATA"]["data_path"].lower()


class TestExperimentNameParsing:
    """Tests for parsing experiment names to detect synthetic data."""

    @pytest.mark.integration
    def test_is_synthetic_experiment_name_true(self):
        """Test detection of synthetic experiment names."""
        from src.utils.data_mode import is_synthetic_experiment_name

        assert is_synthetic_experiment_name("synth_PLR_Classification") is True
        assert is_synthetic_experiment_name("SYNTH_PLR_Imputation") is True

    @pytest.mark.integration
    def test_is_synthetic_experiment_name_false(self):
        """Test detection of production experiment names."""
        from src.utils.data_mode import is_synthetic_experiment_name

        assert is_synthetic_experiment_name("PLR_Classification") is False
        assert is_synthetic_experiment_name("PLR_Imputation") is False
        # Note: __DEBUG_ is not the same as synthetic
        assert is_synthetic_experiment_name("__DEBUG_PLR_Classification") is False


class TestDataModeIsolation:
    """Tests for data mode isolation config loading."""

    @pytest.fixture
    def isolation_config(self):
        """Load data isolation configuration."""
        import yaml
        from pathlib import Path

        config_path = (
            Path(__file__).parent.parent.parent / "configs" / "data_isolation.yaml"
        )
        with open(config_path) as f:
            return yaml.safe_load(f)

    @pytest.mark.integration
    def test_isolation_config_run_prefix(self, isolation_config):
        """Test that isolation config has correct run prefix."""
        assert isolation_config["mlflow"]["run_prefix"] == "__SYNTHETIC_"

    @pytest.mark.integration
    def test_isolation_config_experiment_prefix(self, isolation_config):
        """Test that isolation config has correct experiment prefix."""
        assert isolation_config["mlflow"]["experiment_prefix"] == "synth_"

    @pytest.mark.integration
    def test_isolation_config_rejection_criteria(self, isolation_config):
        """Test that isolation config has proper rejection criteria."""
        criteria = isolation_config["extraction"]["rejection_criteria"]

        # Should reject based on run name, experiment name, and tags
        prefixes = [
            c.get("run_name_prefix") for c in criteria if "run_name_prefix" in c
        ]
        assert "__SYNTHETIC_" in prefixes

        exp_prefixes = [
            c.get("experiment_name_prefix")
            for c in criteria
            if "experiment_name_prefix" in c
        ]
        assert "synth_" in exp_prefixes
