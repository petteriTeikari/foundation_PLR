"""Integration tests for figure generation synthetic data isolation.

Tests that figures generated from synthetic data are routed to
figures/synthetic/ and marked appropriately.

Part of Phase 4 of the 4-gate isolation architecture.
"""

import os
from pathlib import Path

import pytest


class TestFigureOutputRouting:
    """Tests for figure output directory routing."""

    @pytest.mark.integration
    def test_get_figures_dir_production(self):
        """Test that production figures go to figures/generated/."""
        from src.utils.data_mode import get_figures_dir_for_mode

        figures_dir = get_figures_dir_for_mode(synthetic=False)

        assert "synthetic" not in str(figures_dir).lower()
        assert "generated" in str(figures_dir).lower()

    @pytest.mark.integration
    def test_get_figures_dir_synthetic(self):
        """Test that synthetic figures go to figures/synthetic/."""
        from src.utils.data_mode import get_figures_dir_for_mode

        figures_dir = get_figures_dir_for_mode(synthetic=True)

        assert "synthetic" in str(figures_dir).lower()
        assert figures_dir.name == "synthetic"

    @pytest.mark.integration
    def test_synthetic_figures_dir_separate_from_generated(self):
        """Test that synthetic and generated dirs are separate."""
        from src.utils.data_mode import get_figures_dir_for_mode

        production_dir = get_figures_dir_for_mode(synthetic=False)
        synthetic_dir = get_figures_dir_for_mode(synthetic=True)

        assert production_dir != synthetic_dir
        assert "generated" in str(production_dir)
        assert "synthetic" in str(synthetic_dir)


class TestJSONMetadata:
    """Tests for JSON metadata marking synthetic data."""

    @pytest.fixture
    def isolation_config(self):
        """Load isolation config."""
        import yaml

        config_path = (
            Path(__file__).parent.parent.parent / "configs" / "data_isolation.yaml"
        )
        with open(config_path) as f:
            return yaml.safe_load(f)

    @pytest.mark.integration
    def test_json_metadata_synthetic_warning(self, isolation_config):
        """Test that synthetic JSON has warning flag."""
        synthetic_meta = isolation_config["json_metadata"]["synthetic"]

        assert synthetic_meta["_synthetic_warning"] is True
        assert synthetic_meta["_do_not_publish"] is True
        assert synthetic_meta["_data_source"] == "synthetic"

    @pytest.mark.integration
    def test_json_metadata_production_no_warning(self, isolation_config):
        """Test that production JSON has no warning flag."""
        production_meta = isolation_config["json_metadata"]["production"]

        assert production_meta["_synthetic_warning"] is False
        assert production_meta["_do_not_publish"] is False
        assert production_meta["_data_source"] == "production"


class TestDatabaseSourceDetection:
    """Tests for detecting synthetic data from database metadata."""

    @pytest.mark.integration
    def test_is_synthetic_from_filename_synth_db(self):
        """Test detection from synthetic database filename."""
        from src.utils.data_mode import is_synthetic_from_filename

        assert is_synthetic_from_filename("synthetic_foundation_plr_results.db") is True
        assert is_synthetic_from_filename("outputs/synthetic/results.db") is True

    @pytest.mark.integration
    def test_is_synthetic_from_filename_production_db(self):
        """Test detection from production database filename."""
        from src.utils.data_mode import is_synthetic_from_filename

        assert is_synthetic_from_filename("foundation_plr_results.db") is False
        assert (
            is_synthetic_from_filename("data/public/foundation_plr_results.db") is False
        )


class TestPlotConfigSyntheticDetection:
    """Tests for synthetic detection in plot_config.py."""

    @pytest.mark.integration
    def test_plot_config_has_database_finder(self):
        """Test that plot_config has database finding logic."""
        from src.viz.plot_config import _find_database

        # Function should exist
        assert callable(_find_database)

    @pytest.mark.integration
    def test_plot_config_get_output_dir(self):
        """Test that plot_config has output directory logic."""
        from src.viz.plot_config import get_output_dir

        # Function should exist and return a path
        output_dir = get_output_dir()
        assert isinstance(output_dir, Path)


class TestDataIsolationConfig:
    """Tests for data isolation config paths."""

    @pytest.fixture
    def isolation_config(self):
        """Load isolation config."""
        import yaml

        config_path = (
            Path(__file__).parent.parent.parent / "configs" / "data_isolation.yaml"
        )
        with open(config_path) as f:
            return yaml.safe_load(f)

    @pytest.mark.integration
    def test_figure_paths_separate(self, isolation_config):
        """Test that figure paths are properly separated."""
        paths = isolation_config["paths"]["figures"]

        assert paths["production"] == "figures/generated/"
        assert "synthetic" in paths["synthetic"].lower()
        assert paths["production"] != paths["synthetic"]

    @pytest.mark.integration
    def test_gitignore_patterns_include_synthetic(self, isolation_config):
        """Test that gitignore patterns include synthetic directories."""
        patterns = isolation_config["gitignore_patterns"]

        # Should ignore synthetic outputs
        assert any("synthetic" in p for p in patterns)
        assert "figures/synthetic/" in patterns or any(
            "figures/synthetic" in p for p in patterns
        )


class TestEnvVarOverride:
    """Tests for environment variable overrides."""

    @pytest.fixture(autouse=True)
    def clear_env(self):
        """Clear relevant env vars before/after each test."""
        env_vars = [
            "FOUNDATION_PLR_SYNTHETIC",
            "FOUNDATION_PLR_DB_PATH",
            "FOUNDATION_PLR_FIGURES_DIR",
        ]
        original = {v: os.environ.get(v) for v in env_vars}

        # Clear
        for v in env_vars:
            if v in os.environ:
                del os.environ[v]

        yield

        # Restore
        for v, val in original.items():
            if val is not None:
                os.environ[v] = val
            elif v in os.environ:
                del os.environ[v]

    @pytest.mark.integration
    def test_synthetic_mode_from_env_var(self):
        """Test that FOUNDATION_PLR_SYNTHETIC triggers synthetic mode."""
        os.environ["FOUNDATION_PLR_SYNTHETIC"] = "1"

        from src.utils.data_mode import is_synthetic_mode

        assert is_synthetic_mode() is True

    @pytest.mark.integration
    def test_db_path_override_works(self):
        """Test that FOUNDATION_PLR_DB_PATH overrides database location."""
        test_path = "/tmp/test_db.db"
        os.environ["FOUNDATION_PLR_DB_PATH"] = test_path

        # Import should use env var
        from src.viz import plot_config

        # Re-import to pick up env var
        from importlib import reload

        reload(plot_config)

        # The _find_database function should check env var first
        # We're testing the behavior, not creating the file
        # so we expect FileNotFoundError with our test path in the message
        with pytest.raises(FileNotFoundError) as exc_info:
            plot_config._find_database()

        assert test_path in str(exc_info.value)
