"""Integration tests for DuckDB extraction synthetic data isolation.

Tests that the extraction pipeline properly rejects synthetic runs and
routes synthetic/production outputs to separate directories.

Part of Phase 3 of the 4-gate isolation architecture.
"""

from pathlib import Path

import pytest


class TestExtractionPathRouting:
    """Tests for extraction output path routing."""

    @pytest.mark.integration
    def test_get_results_db_path_production(self):
        """Test that production DB path does not contain 'synthetic'."""
        from src.utils.data_mode import get_results_db_path_for_mode

        path = get_results_db_path_for_mode(synthetic=False)
        assert "synthetic" not in str(path).lower()
        assert "foundation_plr_results" in str(path)

    @pytest.mark.integration
    def test_get_results_db_path_synthetic(self):
        """Test that synthetic DB path contains 'synthetic'."""
        from src.utils.data_mode import get_results_db_path_for_mode

        path = get_results_db_path_for_mode(synthetic=True)
        assert "synthetic" in str(path).lower()

    @pytest.mark.integration
    def test_synthetic_output_dir_is_separate(self):
        """Test that synthetic output dir is separate from outputs/."""
        from src.utils.data_mode import get_synthetic_output_dir

        output_dir = get_synthetic_output_dir()

        # Should be outputs/synthetic/, not outputs/
        assert output_dir.name == "synthetic"
        assert output_dir.parent.name == "outputs"


class TestRunValidationForExtraction:
    """Tests for run-level validation during extraction."""

    @pytest.mark.integration
    def test_validate_run_rejects_synthetic_run_name(self):
        """Test that extraction rejects runs with __SYNTHETIC_ prefix."""
        from src.utils.data_mode import validate_run_for_production_extraction

        result = validate_run_for_production_extraction(
            run_name="__SYNTHETIC_LOF",
            experiment_name="PLR_Classification",
        )
        assert result is False

    @pytest.mark.integration
    def test_validate_run_rejects_synthetic_experiment(self):
        """Test that extraction rejects runs from synth_ experiments."""
        from src.utils.data_mode import validate_run_for_production_extraction

        result = validate_run_for_production_extraction(
            run_name="LOF",
            experiment_name="synth_PLR_Classification",
        )
        assert result is False

    @pytest.mark.integration
    def test_validate_run_rejects_synthetic_tag(self):
        """Test that extraction rejects runs with is_synthetic=true tag."""
        from src.utils.data_mode import validate_run_for_production_extraction

        result = validate_run_for_production_extraction(
            run_name="LOF",
            experiment_name="PLR_Classification",
            tags={"is_synthetic": "true"},
        )
        assert result is False

    @pytest.mark.integration
    def test_validate_run_accepts_production_run(self):
        """Test that extraction accepts valid production runs."""
        from src.utils.data_mode import validate_run_for_production_extraction

        result = validate_run_for_production_extraction(
            run_name="LOF",
            experiment_name="PLR_Classification",
            tags={"is_synthetic": "false"},
        )
        assert result is True

    @pytest.mark.integration
    def test_validate_run_accepts_no_tags(self):
        """Test that extraction accepts runs without tags (legacy runs)."""
        from src.utils.data_mode import validate_run_for_production_extraction

        result = validate_run_for_production_extraction(
            run_name="LOF",
            experiment_name="PLR_Classification",
            tags=None,
        )
        assert result is True


class TestSyntheticDataError:
    """Tests for SyntheticDataError exception."""

    @pytest.mark.integration
    def test_validate_not_synthetic_raises_for_run_name(self):
        """Test that validate_not_synthetic raises for synthetic run names."""
        from src.utils.data_mode import SyntheticDataError, validate_not_synthetic

        with pytest.raises(SyntheticDataError) as exc_info:
            validate_not_synthetic(
                run_name="__SYNTHETIC_LOF", context="production extraction"
            )

        assert "Synthetic run detected" in str(exc_info.value)
        assert "__SYNTHETIC_" in str(exc_info.value)

    @pytest.mark.integration
    def test_validate_not_synthetic_raises_for_db_path(self):
        """Test that validate_not_synthetic raises for synthetic DB paths."""
        from src.utils.data_mode import SyntheticDataError, validate_not_synthetic

        with pytest.raises(SyntheticDataError) as exc_info:
            validate_not_synthetic(
                db_path="/outputs/synthetic/synthetic_results.db",
                context="production extraction",
            )

        assert "Synthetic database detected" in str(exc_info.value)

    @pytest.mark.integration
    def test_validate_not_synthetic_raises_for_experiment_name(self):
        """Test that validate_not_synthetic raises for synthetic experiment names."""
        from src.utils.data_mode import SyntheticDataError, validate_not_synthetic

        with pytest.raises(SyntheticDataError) as exc_info:
            validate_not_synthetic(
                experiment_name="synth_PLR_Classification",
                context="production extraction",
            )

        assert "Synthetic experiment detected" in str(exc_info.value)

    @pytest.mark.integration
    def test_validate_not_synthetic_passes_for_production(self):
        """Test that validate_not_synthetic passes for production data."""
        from src.utils.data_mode import validate_not_synthetic

        # Should not raise
        validate_not_synthetic(
            run_name="LOF",
            db_path="/data/public/foundation_plr_results.db",
            experiment_name="PLR_Classification",
            context="production extraction",
        )


class TestExtractSyntheticScript:
    """Tests for the existence and structure of synthetic extraction script."""

    @pytest.mark.integration
    def test_synthetic_extraction_script_exists(self):
        """Test that the synthetic extraction script exists."""
        from pathlib import Path

        script_path = (
            Path(__file__).parent.parent.parent
            / "scripts"
            / "extraction"
            / "extract_synthetic_to_duckdb.py"
        )

        assert script_path.exists(), (
            f"Synthetic extraction script not found: {script_path}"
        )


class TestDatabaseMetadata:
    """Tests for database metadata marking synthetic/production."""

    @pytest.mark.integration
    def test_isolation_config_database_naming(self):
        """Test that isolation config has proper database naming."""
        import yaml

        config_path = (
            Path(__file__).parent.parent.parent / "configs" / "data_isolation.yaml"
        )
        with open(config_path) as f:
            config = yaml.safe_load(f)

        db_config = config["extraction"]["database"]

        assert db_config["production"] == "foundation_plr_results.db"
        assert "synthetic" in db_config["synthetic"].lower()

    @pytest.mark.integration
    def test_isolation_config_output_paths(self):
        """Test that isolation config has separate output paths."""
        import yaml

        config_path = (
            Path(__file__).parent.parent.parent / "configs" / "data_isolation.yaml"
        )
        with open(config_path) as f:
            config = yaml.safe_load(f)

        paths = config["paths"]["outputs"]

        assert paths["production"] == "outputs/"
        assert "synthetic" in paths["synthetic"].lower()
