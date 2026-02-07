"""Smoke tests for the PLR pipeline.

Minimal end-to-end tests that verify the pipeline components
can be loaded and run without errors. These tests are designed
to catch major integration issues quickly.
"""

import numpy as np
import pytest


class TestPipelineImports:
    """Test that all major pipeline modules can be imported."""

    @pytest.mark.unit
    def test_import_data_utils(self):
        """Test that data utilities can be imported."""
        from src.data_io import data_utils

        assert data_utils is not None

    @pytest.mark.unit
    def test_import_metrics_utils(self):
        """Test that metrics utilities can be imported."""
        from src.metrics import metrics_utils

        assert metrics_utils is not None

    @pytest.mark.unit
    def test_import_orchestration(self):
        """Test that orchestration modules can be imported."""
        from src.orchestration import hyperparameter_sweep_utils
        from src.orchestration import hyperparamer_list_utils

        assert hyperparameter_sweep_utils is not None
        assert hyperparamer_list_utils is not None

    @pytest.mark.unit
    def test_import_utils(self):
        """Test that main utils can be imported."""
        from src import utils

        assert utils is not None

    @pytest.mark.unit
    def test_import_log_helpers(self):
        """Test that log helpers can be imported."""
        from src.log_helpers import log_naming_uris_and_dirs

        assert log_naming_uris_and_dirs is not None


class TestMinimalDataPipeline:
    """Minimal tests for data pipeline without requiring demo data."""

    @pytest.mark.unit
    def test_time_vector_creation(self):
        """Test that time vector can be created."""
        from src.data_io.data_utils import define_desired_timevector

        time_vec = define_desired_timevector(PLR_length=1981, fps=30)

        assert len(time_vec) == 1981
        assert time_vec[0] == 0.0

    @pytest.mark.unit
    def test_padding_roundtrip(self):
        """Test that padding and unpadding work correctly."""
        from src.data_io.data_utils import (
            pad_glaucoma_PLR,
            unpad_glaucoma_PLR,
        )

        # Create synthetic data
        np.random.seed(42)
        data = np.random.randn(4, 1981)

        # Pad and unpad
        padded = pad_glaucoma_PLR(data, trim_to_size=512)
        unpadded = unpad_glaucoma_PLR(padded, length_PLR=1981)

        np.testing.assert_array_almost_equal(data, unpadded)

    @pytest.mark.unit
    def test_config_creation(self, minimal_cfg):
        """Test that minimal config can be created and accessed."""
        assert minimal_cfg["DATA"]["PLR_length"] == 1981
        assert minimal_cfg["EXPERIMENT"]["use_demo_data"] is True
        assert minimal_cfg["DEVICE"]["device"] == "cpu"


class TestDemoDataPipeline:
    """End-to-end smoke tests using demo data."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_demo_data_load_and_validate(
        self, demo_data_path, demo_data_available, minimal_cfg
    ):
        """Test loading and validating demo data."""
        assert demo_data_available, (
            f"Demo data not available: {demo_data_path}. Run: make synthetic"
        )

        from src.data_io.data_utils import (
            import_duckdb_as_dataframes,
            check_for_data_lengths,
        )

        # Load data
        df_train, df_test = import_duckdb_as_dataframes(str(demo_data_path))

        # Basic validation
        assert df_train is not None
        assert df_test is not None
        assert len(df_train) > 0
        assert len(df_test) > 0

        # Check data lengths
        check_for_data_lengths(df_train, minimal_cfg)
        check_for_data_lengths(df_test, minimal_cfg)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_demo_data_to_numpy_conversion(self, demo_data_path, demo_data_available):
        """Test converting demo data to numpy arrays."""
        assert demo_data_available, (
            f"Demo data not available: {demo_data_path}. Run: make synthetic"
        )

        from src.data_io.data_utils import import_duckdb_as_dataframes

        df_train, _ = import_duckdb_as_dataframes(str(demo_data_path))

        # Convert to numpy
        pupil_data = df_train["pupil_raw"].to_numpy()

        assert isinstance(pupil_data, np.ndarray)
        assert len(pupil_data) > 0


class TestConfigurationValidation:
    """Tests for configuration validation."""

    @pytest.mark.unit
    def test_omegaconf_serialization(self, minimal_cfg):
        """Test that config can be serialized and deserialized."""
        from omegaconf import OmegaConf

        # Serialize to YAML string
        yaml_str = OmegaConf.to_yaml(minimal_cfg)
        assert isinstance(yaml_str, str)
        assert "PLR_length" in yaml_str

        # Deserialize back
        cfg_back = OmegaConf.create(yaml_str)
        assert cfg_back["DATA"]["PLR_length"] == 1981

    @pytest.mark.unit
    def test_drop_other_models_all_tasks(self, minimal_cfg):
        """Test model dropping for all task types."""
        from src.orchestration.hyperparameter_sweep_utils import drop_other_models

        # Test outlier detection
        result = drop_other_models(
            minimal_cfg.copy(), model="LOF", task="outlier_detection"
        )
        assert len(result["OUTLIER_MODELS"]) == 1

        # Test imputation
        result = drop_other_models(
            minimal_cfg.copy(), model="MissForest", task="imputation"
        )
        assert len(result["MODELS"]) == 1

        # Test classification
        result = drop_other_models(
            minimal_cfg.copy(), model="XGBOOST", task="classification"
        )
        assert len(result["CLS_MODELS"]) == 1


class TestExternalDependencies:
    """Tests verifying external dependencies are available."""

    @pytest.mark.unit
    def test_numpy_available(self):
        """Test that numpy is available and working."""
        import numpy as np

        arr = np.array([1, 2, 3])
        assert np.sum(arr) == 6

    @pytest.mark.unit
    def test_polars_available(self):
        """Test that polars is available and working."""
        import polars as pl

        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert len(df) == 3

    @pytest.mark.unit
    def test_sklearn_available(self):
        """Test that scikit-learn is available."""
        from sklearn.neighbors import LocalOutlierFactor

        lof = LocalOutlierFactor(n_neighbors=5)
        assert lof is not None

    @pytest.mark.unit
    def test_omegaconf_available(self):
        """Test that OmegaConf is available."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"key": "value"})
        assert cfg["key"] == "value"

    @pytest.mark.unit
    def test_duckdb_available(self):
        """Test that DuckDB is available."""
        import duckdb

        con = duckdb.connect(":memory:")
        result = con.execute("SELECT 1 as test").fetchone()
        assert result[0] == 1
        con.close()
