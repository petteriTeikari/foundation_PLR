"""End-to-end tests for synthetic data isolation.

Tests the complete 4-gate isolation architecture:
- Gate 0: Data mode detection
- Gate 1: MLflow naming and tagging
- Gate 2: DuckDB extraction isolation
- Gate 3: Figure output routing

These tests verify that:
1. Synthetic pipeline outputs to synthetic directories
2. Production extraction rejects synthetic runs
3. Figure generation routes based on data mode
4. Pre-commit hooks catch violations
"""

import json
import os
import tempfile
from pathlib import Path

import pytest


class TestGate0DataModeDetection:
    """End-to-end tests for Gate 0: Data mode detection."""

    @pytest.fixture(autouse=True)
    def clear_env(self):
        """Clear environment variables before each test."""
        if "FOUNDATION_PLR_SYNTHETIC" in os.environ:
            del os.environ["FOUNDATION_PLR_SYNTHETIC"]
        yield
        if "FOUNDATION_PLR_SYNTHETIC" in os.environ:
            del os.environ["FOUNDATION_PLR_SYNTHETIC"]

    @pytest.mark.e2e
    def test_env_var_triggers_synthetic_mode(self):
        """Test that FOUNDATION_PLR_SYNTHETIC=1 triggers synthetic mode."""
        os.environ["FOUNDATION_PLR_SYNTHETIC"] = "1"

        from src.utils.data_mode import get_data_mode, is_synthetic_mode

        assert is_synthetic_mode() is True
        assert get_data_mode() == "synthetic"

    @pytest.mark.e2e
    def test_config_triggers_synthetic_mode(self):
        """Test that config with is_synthetic=true triggers synthetic mode."""
        from omegaconf import OmegaConf

        from src.utils.data_mode import get_data_mode, is_synthetic_from_config

        cfg = OmegaConf.create(
            {
                "EXPERIMENT": {"is_synthetic": True},
                "DATA": {"data_path": "data/normal"},
            }
        )

        assert is_synthetic_from_config(cfg) is True
        assert get_data_mode(cfg=cfg) == "synthetic"

    @pytest.mark.e2e
    def test_production_is_default(self):
        """Test that production mode is the default."""
        from src.utils.data_mode import get_data_mode, is_synthetic_mode

        # No env var, no config
        assert is_synthetic_mode() is False
        assert get_data_mode() == "production"


class TestGate1MlflowIsolation:
    """End-to-end tests for Gate 1: MLflow naming and tagging."""

    @pytest.mark.e2e
    def test_synthetic_run_prefix(self):
        """Test that synthetic runs get __SYNTHETIC_ prefix."""
        from src.utils.data_mode import (
            SYNTHETIC_RUN_PREFIX,
            add_synthetic_prefix_to_run_name,
            is_synthetic_run_name,
        )

        run_name = "LOF"
        synthetic_name = add_synthetic_prefix_to_run_name(run_name)

        assert synthetic_name == f"{SYNTHETIC_RUN_PREFIX}{run_name}"
        assert is_synthetic_run_name(synthetic_name) is True
        assert is_synthetic_run_name(run_name) is False

    @pytest.mark.e2e
    def test_experiment_name_wrapper_adds_prefix(self):
        """Test that experiment_name_wrapper adds synthetic prefix."""
        from omegaconf import OmegaConf

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

        assert result.startswith("synth_")

    @pytest.mark.e2e
    def test_mlflow_tags_for_synthetic(self):
        """Test that synthetic mode produces correct MLflow tags."""
        from src.utils.data_mode import get_mlflow_tags_for_mode

        tags = get_mlflow_tags_for_mode(synthetic=True)

        assert tags["is_synthetic"] == "true"
        assert tags["data_source"] == "synthetic"


class TestGate2ExtractionIsolation:
    """End-to-end tests for Gate 2: DuckDB extraction isolation."""

    @pytest.mark.e2e
    def test_validate_run_rejects_synthetic(self):
        """Test that production extraction rejects synthetic runs."""
        from src.utils.data_mode import validate_run_for_production_extraction

        # Synthetic run name
        assert (
            validate_run_for_production_extraction(
                run_name="__SYNTHETIC_LOF", experiment_name="PLR_Classification"
            )
            is False
        )

        # Synthetic experiment
        assert (
            validate_run_for_production_extraction(
                run_name="LOF", experiment_name="synth_PLR_Classification"
            )
            is False
        )

        # Synthetic tag
        assert (
            validate_run_for_production_extraction(
                run_name="LOF",
                experiment_name="PLR_Classification",
                tags={"is_synthetic": "true"},
            )
            is False
        )

    @pytest.mark.e2e
    def test_validate_run_accepts_production(self):
        """Test that production extraction accepts production runs."""
        from src.utils.data_mode import validate_run_for_production_extraction

        assert (
            validate_run_for_production_extraction(
                run_name="LOF",
                experiment_name="PLR_Classification",
                tags={"is_synthetic": "false"},
            )
            is True
        )

    @pytest.mark.e2e
    def test_output_paths_separate(self):
        """Test that synthetic and production output paths are separate."""
        from src.utils.data_mode import get_results_db_path_for_mode

        prod_path = get_results_db_path_for_mode(synthetic=False)
        synth_path = get_results_db_path_for_mode(synthetic=True)

        assert prod_path != synth_path
        assert "synthetic" in str(synth_path).lower()
        assert "synthetic" not in str(prod_path).lower()


class TestGate3FigureIsolation:
    """End-to-end tests for Gate 3: Figure output routing."""

    @pytest.fixture(autouse=True)
    def clear_env(self):
        """Clear environment variables."""
        if "FOUNDATION_PLR_SYNTHETIC" in os.environ:
            del os.environ["FOUNDATION_PLR_SYNTHETIC"]
        yield
        if "FOUNDATION_PLR_SYNTHETIC" in os.environ:
            del os.environ["FOUNDATION_PLR_SYNTHETIC"]

    @pytest.mark.e2e
    def test_figure_dir_routing_production(self):
        """Test that production figures go to figures/generated/."""
        from src.utils.data_mode import get_figures_dir_for_mode

        figures_dir = get_figures_dir_for_mode(synthetic=False)

        assert "generated" in str(figures_dir)
        assert "synthetic" not in str(figures_dir).lower()

    @pytest.mark.e2e
    def test_figure_dir_routing_synthetic(self):
        """Test that synthetic figures go to figures/synthetic/."""
        from src.utils.data_mode import get_figures_dir_for_mode

        figures_dir = get_figures_dir_for_mode(synthetic=True)

        assert "synthetic" in str(figures_dir).lower()

    @pytest.mark.e2e
    def test_save_figure_adds_json_metadata(self):
        """Test that save_figure adds synthetic metadata to JSON."""
        import matplotlib.pyplot as plt

        from src.viz.plot_config import save_figure

        # Create a simple figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Save with synthetic=True
            save_figure(
                fig,
                "test_figure",
                data={"test": "data"},
                output_dir=output_dir,
                synthetic=True,
            )

            # Check JSON metadata
            json_path = output_dir / "data" / "test_figure.json"
            assert json_path.exists()

            with open(json_path) as f:
                data = json.load(f)

            assert data["_synthetic_warning"] is True
            assert data["_data_source"] == "synthetic"
            assert data["_do_not_publish"] is True

        plt.close(fig)


class TestPreCommitHooks:
    """End-to-end tests for pre-commit hooks."""

    @pytest.mark.e2e
    def test_extraction_isolation_check_exists(self):
        """Test that extraction isolation check script exists."""
        script = (
            Path(__file__).parent.parent.parent
            / "scripts"
            / "validation"
            / "check_extraction_isolation.py"
        )
        assert script.exists()

    @pytest.mark.e2e
    def test_figure_isolation_check_exists(self):
        """Test that figure isolation check script exists."""
        script = (
            Path(__file__).parent.parent.parent
            / "scripts"
            / "validation"
            / "check_figure_isolation.py"
        )
        assert script.exists()

    @pytest.mark.e2e
    def test_extraction_check_passes(self):
        """Test that extraction isolation check passes."""
        import subprocess

        result = subprocess.run(
            ["uv", "run", "python", "scripts/validation/check_extraction_isolation.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0, (
            f"Extraction check failed:\n{result.stdout}\n{result.stderr}"
        )

    @pytest.mark.e2e
    def test_figure_check_passes(self):
        """Test that figure isolation check passes."""
        import subprocess

        result = subprocess.run(
            ["uv", "run", "python", "scripts/validation/check_figure_isolation.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0, (
            f"Figure check failed:\n{result.stdout}\n{result.stderr}"
        )


class TestFullIsolationChain:
    """End-to-end tests for the complete isolation chain."""

    @pytest.fixture(autouse=True)
    def clear_env(self):
        """Clear environment variables."""
        if "FOUNDATION_PLR_SYNTHETIC" in os.environ:
            del os.environ["FOUNDATION_PLR_SYNTHETIC"]
        yield
        if "FOUNDATION_PLR_SYNTHETIC" in os.environ:
            del os.environ["FOUNDATION_PLR_SYNTHETIC"]

    @pytest.mark.e2e
    def test_synthetic_mode_propagates_through_chain(self):
        """Test that synthetic mode propagates through all gates."""
        os.environ["FOUNDATION_PLR_SYNTHETIC"] = "1"

        from src.utils.data_mode import (
            get_data_mode,
            get_figures_dir_for_mode,
            get_mlflow_tags_for_mode,
            get_results_db_path_for_mode,
            is_synthetic_mode,
        )

        # Gate 0: Detection
        assert is_synthetic_mode() is True
        assert get_data_mode() == "synthetic"

        # Gate 1: MLflow
        tags = get_mlflow_tags_for_mode(synthetic=is_synthetic_mode())
        assert tags["is_synthetic"] == "true"

        # Gate 2: DuckDB
        db_path = get_results_db_path_for_mode(synthetic=is_synthetic_mode())
        assert "synthetic" in str(db_path).lower()

        # Gate 3: Figures
        figures_dir = get_figures_dir_for_mode(synthetic=is_synthetic_mode())
        assert "synthetic" in str(figures_dir).lower()

    @pytest.mark.e2e
    def test_production_mode_propagates_through_chain(self):
        """Test that production mode propagates through all gates."""
        # Ensure no synthetic env var
        assert "FOUNDATION_PLR_SYNTHETIC" not in os.environ

        from src.utils.data_mode import (
            get_data_mode,
            get_figures_dir_for_mode,
            get_mlflow_tags_for_mode,
            get_results_db_path_for_mode,
            is_synthetic_mode,
        )

        # Gate 0: Detection
        assert is_synthetic_mode() is False
        assert get_data_mode() == "production"

        # Gate 1: MLflow
        tags = get_mlflow_tags_for_mode(synthetic=is_synthetic_mode())
        assert tags["is_synthetic"] == "false"

        # Gate 2: DuckDB
        db_path = get_results_db_path_for_mode(synthetic=is_synthetic_mode())
        assert "synthetic" not in str(db_path).lower()

        # Gate 3: Figures
        figures_dir = get_figures_dir_for_mode(synthetic=is_synthetic_mode())
        assert "synthetic" not in str(figures_dir).lower()
