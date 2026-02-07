"""Unit tests for data mode detection infrastructure.

Tests the synthetic vs production data mode detection logic from
src/utils/data_mode.py.

The data mode detection is critical for ensuring synthetic data never
contaminates production artifacts. See CRITICAL-FAILURE-001 for context.
"""

import os

import pytest


class TestIsSyntheticMode:
    """Tests for is_synthetic_mode() function."""

    @pytest.fixture(autouse=True)
    def clear_env(self):
        """Clear environment variables before each test."""
        env_vars = [
            "FOUNDATION_PLR_SYNTHETIC",
            "FOUNDATION_PLR_DATA_SOURCE",
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]
        yield
        # Cleanup after test
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

    @pytest.mark.unit
    def test_synthetic_from_env_var_true(self):
        """Test detection when FOUNDATION_PLR_SYNTHETIC=1."""
        os.environ["FOUNDATION_PLR_SYNTHETIC"] = "1"
        from src.utils.data_mode import is_synthetic_mode

        assert is_synthetic_mode() is True

    @pytest.mark.unit
    def test_synthetic_from_env_var_false(self):
        """Test detection when FOUNDATION_PLR_SYNTHETIC=0."""
        os.environ["FOUNDATION_PLR_SYNTHETIC"] = "0"
        from src.utils.data_mode import is_synthetic_mode

        assert is_synthetic_mode() is False

    @pytest.mark.unit
    def test_synthetic_from_env_var_yes(self):
        """Test detection when FOUNDATION_PLR_SYNTHETIC=yes."""
        os.environ["FOUNDATION_PLR_SYNTHETIC"] = "yes"
        from src.utils.data_mode import is_synthetic_mode

        assert is_synthetic_mode() is True

    @pytest.mark.unit
    def test_production_default_no_env(self):
        """Test that production mode is default when no env var set."""
        # Ensure env var is not set
        if "FOUNDATION_PLR_SYNTHETIC" in os.environ:
            del os.environ["FOUNDATION_PLR_SYNTHETIC"]

        from src.utils.data_mode import is_synthetic_mode

        assert is_synthetic_mode() is False


class TestIsSyntheticFromConfig:
    """Tests for is_synthetic_from_config() function."""

    @pytest.mark.unit
    def test_synthetic_from_config_true(self):
        """Test detection from config with is_synthetic=true."""
        from omegaconf import OmegaConf

        from src.utils.data_mode import is_synthetic_from_config

        cfg = OmegaConf.create({"EXPERIMENT": {"is_synthetic": True}})
        assert is_synthetic_from_config(cfg) is True

    @pytest.mark.unit
    def test_synthetic_from_config_false(self):
        """Test detection from config with is_synthetic=false."""
        from omegaconf import OmegaConf

        from src.utils.data_mode import is_synthetic_from_config

        cfg = OmegaConf.create({"EXPERIMENT": {"is_synthetic": False}})
        assert is_synthetic_from_config(cfg) is False

    @pytest.mark.unit
    def test_synthetic_from_config_missing_key(self):
        """Test detection from config when key is missing (default=False)."""
        from omegaconf import OmegaConf

        from src.utils.data_mode import is_synthetic_from_config

        cfg = OmegaConf.create({"EXPERIMENT": {}})
        assert is_synthetic_from_config(cfg) is False

    @pytest.mark.unit
    def test_synthetic_from_config_experiment_prefix(self):
        """Test detection from config with synth_ experiment prefix."""
        from omegaconf import OmegaConf

        from src.utils.data_mode import is_synthetic_from_config

        cfg = OmegaConf.create({"EXPERIMENT": {"experiment_prefix": "synth_"}})
        assert is_synthetic_from_config(cfg) is True

    @pytest.mark.unit
    def test_synthetic_from_config_data_path(self):
        """Test detection from config with synthetic data path."""
        from omegaconf import OmegaConf

        from src.utils.data_mode import is_synthetic_from_config

        cfg = OmegaConf.create({"DATA": {"data_path": "data/synthetic"}})
        assert is_synthetic_from_config(cfg) is True


class TestIsSyntheticFromFilename:
    """Tests for is_synthetic_from_filename() function."""

    @pytest.mark.unit
    def test_synthetic_from_filename_prefix(self):
        """Test detection from filename with SYNTH_ prefix."""
        from src.utils.data_mode import is_synthetic_from_filename

        assert is_synthetic_from_filename("SYNTH_PLR_DEMO.db") is True

    @pytest.mark.unit
    def test_synthetic_from_filename_contains(self):
        """Test detection from filename containing 'synthetic'."""
        from src.utils.data_mode import is_synthetic_from_filename

        assert is_synthetic_from_filename("synthetic_data.db") is True

    @pytest.mark.unit
    def test_synthetic_from_filename_path(self):
        """Test detection from path containing 'synthetic'."""
        from src.utils.data_mode import is_synthetic_from_filename

        assert is_synthetic_from_filename("/data/synthetic/results.db") is True

    @pytest.mark.unit
    def test_production_from_filename(self):
        """Test production detection from normal filename."""
        from src.utils.data_mode import is_synthetic_from_filename

        assert is_synthetic_from_filename("SERI_PLR_GLAUCOMA.db") is False

    @pytest.mark.unit
    def test_production_from_results_path(self):
        """Test production detection from results path."""
        from src.utils.data_mode import is_synthetic_from_filename

        assert (
            is_synthetic_from_filename("data/public/foundation_plr_results.db") is False
        )


class TestGetDataMode:
    """Tests for get_data_mode() function."""

    @pytest.fixture(autouse=True)
    def clear_env(self):
        """Clear environment variables before each test."""
        if "FOUNDATION_PLR_SYNTHETIC" in os.environ:
            del os.environ["FOUNDATION_PLR_SYNTHETIC"]
        yield
        if "FOUNDATION_PLR_SYNTHETIC" in os.environ:
            del os.environ["FOUNDATION_PLR_SYNTHETIC"]

    @pytest.mark.unit
    def test_get_data_mode_production(self):
        """Test get_data_mode returns 'production' for normal mode."""
        from src.utils.data_mode import get_data_mode

        assert get_data_mode() == "production"

    @pytest.mark.unit
    def test_get_data_mode_synthetic(self):
        """Test get_data_mode returns 'synthetic' when env var set."""
        os.environ["FOUNDATION_PLR_SYNTHETIC"] = "1"
        from src.utils.data_mode import get_data_mode

        assert get_data_mode() == "synthetic"

    @pytest.mark.unit
    def test_get_data_mode_from_filename(self):
        """Test get_data_mode detects synthetic from filename."""
        from src.utils.data_mode import get_data_mode

        assert get_data_mode(filename="synthetic_results.db") == "synthetic"
        assert get_data_mode(filename="production_results.db") == "production"

    @pytest.mark.unit
    def test_get_data_mode_from_config(self):
        """Test get_data_mode detects synthetic from config."""
        from omegaconf import OmegaConf

        from src.utils.data_mode import get_data_mode

        cfg = OmegaConf.create({"EXPERIMENT": {"is_synthetic": True}})
        assert get_data_mode(cfg=cfg) == "synthetic"

    @pytest.mark.unit
    def test_get_data_mode_priority_env_over_config(self):
        """Test that env var takes priority over config."""
        from omegaconf import OmegaConf

        from src.utils.data_mode import get_data_mode

        # Config says production, but env says synthetic
        os.environ["FOUNDATION_PLR_SYNTHETIC"] = "1"
        cfg = OmegaConf.create({"EXPERIMENT": {"is_synthetic": False}})
        assert get_data_mode(cfg=cfg) == "synthetic"


class TestSyntheticPrefixes:
    """Tests for synthetic prefix constants and utilities."""

    @pytest.mark.unit
    def test_get_synthetic_run_prefix(self):
        """Test that synthetic run prefix is correct."""
        from src.utils.data_mode import SYNTHETIC_RUN_PREFIX

        assert SYNTHETIC_RUN_PREFIX == "__SYNTHETIC_"

    @pytest.mark.unit
    def test_get_synthetic_experiment_prefix(self):
        """Test that synthetic experiment prefix is correct."""
        from src.utils.data_mode import SYNTHETIC_EXPERIMENT_PREFIX

        assert SYNTHETIC_EXPERIMENT_PREFIX == "synth_"

    @pytest.mark.unit
    def test_add_synthetic_prefix_to_run_name(self):
        """Test adding synthetic prefix to run name."""
        from src.utils.data_mode import add_synthetic_prefix_to_run_name

        result = add_synthetic_prefix_to_run_name("LOF")
        assert result == "__SYNTHETIC_LOF"

    @pytest.mark.unit
    def test_add_synthetic_prefix_idempotent(self):
        """Test that adding prefix twice doesn't double-prefix."""
        from src.utils.data_mode import add_synthetic_prefix_to_run_name

        result = add_synthetic_prefix_to_run_name("__SYNTHETIC_LOF")
        assert result == "__SYNTHETIC_LOF"

    @pytest.mark.unit
    def test_remove_synthetic_prefix(self):
        """Test removing synthetic prefix from run name."""
        from src.utils.data_mode import remove_synthetic_prefix_from_run_name

        result = remove_synthetic_prefix_from_run_name("__SYNTHETIC_LOF")
        assert result == "LOF"

    @pytest.mark.unit
    def test_remove_synthetic_prefix_no_prefix(self):
        """Test removing prefix when not present returns unchanged."""
        from src.utils.data_mode import remove_synthetic_prefix_from_run_name

        result = remove_synthetic_prefix_from_run_name("LOF")
        assert result == "LOF"

    @pytest.mark.unit
    def test_is_synthetic_run_name_true(self):
        """Test detection of synthetic run name."""
        from src.utils.data_mode import is_synthetic_run_name

        assert is_synthetic_run_name("__SYNTHETIC_LOF") is True

    @pytest.mark.unit
    def test_is_synthetic_run_name_false(self):
        """Test detection of production run name."""
        from src.utils.data_mode import is_synthetic_run_name

        assert is_synthetic_run_name("LOF") is False


class TestOutputPaths:
    """Tests for synthetic output path utilities."""

    @pytest.mark.unit
    def test_get_synthetic_output_dir(self):
        """Test that synthetic output dir is separate from production."""
        from src.utils.data_mode import get_synthetic_output_dir

        output_dir = get_synthetic_output_dir()
        assert "synthetic" in str(output_dir)

    @pytest.mark.unit
    def test_get_synthetic_figures_dir(self):
        """Test that synthetic figures dir is separate from production."""
        from src.utils.data_mode import get_synthetic_figures_dir

        figures_dir = get_synthetic_figures_dir()
        assert "synthetic" in str(figures_dir)

    @pytest.mark.unit
    def test_get_results_db_path_production(self):
        """Test production database path."""
        from src.utils.data_mode import get_results_db_path_for_mode

        path = get_results_db_path_for_mode(synthetic=False)
        assert "synthetic" not in str(path)

    @pytest.mark.unit
    def test_get_results_db_path_synthetic(self):
        """Test synthetic database path."""
        from src.utils.data_mode import get_results_db_path_for_mode

        path = get_results_db_path_for_mode(synthetic=True)
        assert "synthetic" in str(path)


class TestMlflowTags:
    """Tests for MLflow tag utilities."""

    @pytest.mark.unit
    def test_get_synthetic_tags(self):
        """Test that synthetic tags include is_synthetic=true."""
        from src.utils.data_mode import get_synthetic_mlflow_tags

        tags = get_synthetic_mlflow_tags()
        assert tags["is_synthetic"] == "true"
        assert tags["data_source"] == "synthetic"

    @pytest.mark.unit
    def test_get_production_tags(self):
        """Test that production tags include is_synthetic=false."""
        from src.utils.data_mode import get_production_mlflow_tags

        tags = get_production_mlflow_tags()
        assert tags["is_synthetic"] == "false"
        assert tags["data_source"] == "production"


class TestValidation:
    """Tests for data mode validation utilities."""

    @pytest.mark.unit
    def test_validate_not_synthetic_passes(self):
        """Test that validation passes for production data."""
        from src.utils.data_mode import validate_not_synthetic

        # Should not raise
        validate_not_synthetic("LOF", "SERI_PLR_GLAUCOMA.db")

    @pytest.mark.unit
    def test_validate_not_synthetic_fails_on_run_name(self):
        """Test that validation fails for synthetic run names."""
        from src.utils.data_mode import SyntheticDataError, validate_not_synthetic

        with pytest.raises(SyntheticDataError, match="Synthetic"):
            validate_not_synthetic("__SYNTHETIC_LOF", "SERI_PLR_GLAUCOMA.db")

    @pytest.mark.unit
    def test_validate_not_synthetic_fails_on_db_path(self):
        """Test that validation fails for synthetic database paths."""
        from src.utils.data_mode import SyntheticDataError, validate_not_synthetic

        with pytest.raises(SyntheticDataError, match="synthetic"):
            validate_not_synthetic("LOF", "/data/synthetic/results.db")
