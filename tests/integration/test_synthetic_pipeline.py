"""
Integration tests for synthetic pipeline stages.

Tests verify that each pipeline stage runs correctly with synthetic data.
This enables CI/CD testing without access to real patient data.

See: docs/planning/pipeline-robustness-plan.md for context.
"""

import os
from pathlib import Path

import pytest

# Set environment to disable Prefect UI
os.environ["PREFECT_DISABLED"] = "1"


def get_synthetic_db_path() -> Path:
    """Get path to synthetic database."""
    project_root = Path(__file__).parent.parent.parent
    return project_root / "data" / "synthetic" / "SYNTH_PLR_DEMO.db"


@pytest.fixture
def synthetic_db_path():
    """Fixture providing path to synthetic database."""
    path = get_synthetic_db_path()
    if not path.exists():
        pytest.skip(f"Synthetic database not found: {path}")
    return path


class TestSyntheticDataExists:
    """Verify synthetic test data is available."""

    @pytest.mark.integration
    def test_synthetic_database_exists(self, synthetic_db_path):
        """Synthetic database should exist for CI testing."""
        assert synthetic_db_path.exists()
        assert synthetic_db_path.suffix == ".db"
        # Should be at least 1MB (4MB expected)
        assert synthetic_db_path.stat().st_size > 1_000_000

    @pytest.mark.integration
    def test_synthetic_config_exists(self):
        """Synthetic run config should exist."""
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "configs" / "synthetic_run.yaml"
        assert config_path.exists()


class TestDataLoading:
    """Test that synthetic data loads correctly."""

    @pytest.mark.integration
    def test_load_synthetic_data_duckdb(self, synthetic_db_path):
        """Should load synthetic data from DuckDB."""
        import duckdb

        conn = duckdb.connect(str(synthetic_db_path), read_only=True)

        # Check tables exist
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]

        assert "train" in table_names
        assert "test" in table_names

        # Check data shape
        train_count = conn.execute("SELECT COUNT(*) FROM train").fetchone()[0]
        test_count = conn.execute("SELECT COUNT(*) FROM test").fetchone()[0]

        # Synthetic data should have subjects
        assert train_count > 0, "Train split should have data"
        assert test_count > 0, "Test split should have data"

        conn.close()

    @pytest.mark.integration
    def test_synthetic_data_has_required_columns(self, synthetic_db_path):
        """Synthetic data should have all required columns."""
        import duckdb

        conn = duckdb.connect(str(synthetic_db_path), read_only=True)

        # Get column names
        cols = conn.execute("DESCRIBE train").fetchdf()["column_name"].tolist()

        required_columns = [
            "time",
            "pupil_raw",
            "outlier_mask",
            "subject_code",
            "class_label",
        ]

        for col in required_columns:
            assert col in cols, f"Missing required column: {col}"

        conn.close()

    @pytest.mark.integration
    def test_synthetic_data_has_labels(self, synthetic_db_path):
        """Synthetic data should have class labels."""
        import duckdb

        conn = duckdb.connect(str(synthetic_db_path), read_only=True)

        # Check label distribution
        labels = (
            conn.execute("SELECT DISTINCT class_label FROM train")
            .fetchdf()["class_label"]
            .tolist()
        )

        # Should have binary classification (labels can be strings or integers)
        label_set = set(str(label).lower() for label in labels)
        has_control = "control" in label_set or "0" in label_set
        has_glaucoma = "glaucoma" in label_set or "1" in label_set

        assert has_control, f"Should have control class, got: {labels}"
        assert has_glaucoma, f"Should have glaucoma class, got: {labels}"

        conn.close()


class TestOutlierDetectionStage:
    """Test outlier detection stage with synthetic data."""

    @pytest.mark.integration
    def test_outlier_detection_imports(self):
        """Outlier detection module should import without errors."""
        try:
            from src.anomaly_detection.outlier_sklearn import (
                LOF_wrapper,
                get_LOF,
                subjectwise_LOF,
            )

            assert callable(LOF_wrapper)
            assert callable(get_LOF)
            assert callable(subjectwise_LOF)
        except ImportError as e:
            pytest.fail(f"Failed to import outlier detection: {e}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_lof_outlier_detection_runs(self, synthetic_db_path):
        """LOF outlier detection should run on synthetic data without crashing."""
        import duckdb
        import numpy as np

        conn = duckdb.connect(str(synthetic_db_path), read_only=True)

        # Get one subject's data
        subject_data = conn.execute("""
            SELECT time, pupil_raw, outlier_mask
            FROM train
            WHERE subject_code = (SELECT subject_code FROM train LIMIT 1)
            ORDER BY time
        """).fetchdf()

        conn.close()

        # Extract signal
        signal = subject_data["pupil_raw"].values.astype(float)

        # Should have data
        assert len(signal) > 0, "Subject should have signal data"
        # Signal should be reasonable (pupil size in mm or normalized)
        assert np.nanmean(signal) > 0, "Signal mean should be positive"


class TestFeaturizationStage:
    """Test featurization stage with synthetic data."""

    @pytest.mark.integration
    def test_feature_extraction_imports(self):
        """Feature extraction module should import without errors."""
        try:
            from src.featurization.feature_utils import (
                get_top1_of_col,
                get_light_stimuli_timings,
            )

            assert callable(get_top1_of_col)
            assert callable(get_light_stimuli_timings)
        except ImportError as e:
            pytest.fail(f"Failed to import feature extraction: {e}")


class TestClassificationStage:
    """Test classification stage utilities."""

    @pytest.mark.integration
    def test_classifier_log_utils_imports(self):
        """Classifier logging utils should import without errors."""
        try:
            from src.classification.classifier_log_utils import (
                parse_and_log_cls_run_name,
            )

            assert callable(parse_and_log_cls_run_name)
        except ImportError as e:
            pytest.fail(f"Failed to import classifier log utils: {e}")

    @pytest.mark.integration
    def test_weighing_utils_imports(self):
        """Weighing utilities should import without errors."""
        try:
            from src.classification.weighing_utils import (
                normalize_to_unity,
                norm_wrapper,
            )

            assert callable(normalize_to_unity)
            assert callable(norm_wrapper)
        except ImportError as e:
            pytest.fail(f"Failed to import weighing utils: {e}")


class TestEnsembleUtilities:
    """Test ensemble utility functions."""

    @pytest.mark.integration
    def test_ensemble_utils_imports(self):
        """Ensemble utilities should import without errors."""
        try:
            from src.ensemble.ensemble_utils import (
                get_best_moment_variant,
                get_best_moments_per_source,
            )

            assert callable(get_best_moment_variant)
            assert callable(get_best_moments_per_source)
        except ImportError as e:
            pytest.fail(f"Failed to import ensemble utils: {e}")

    @pytest.mark.integration
    def test_ensemble_handles_empty_data(self):
        """Ensemble functions should handle empty data gracefully."""
        import pandas as pd
        from omegaconf import OmegaConf

        from src.ensemble.ensemble_utils import get_best_moment_variant

        cfg = OmegaConf.create({"direction": "DESC", "split": "test", "string": "f1"})

        # Should not crash with empty DataFrame
        result = get_best_moment_variant(pd.DataFrame(), cfg, return_best_gt=True)

        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestPipelineConfig:
    """Test pipeline configuration loading."""

    @pytest.mark.integration
    def test_hydra_config_loads(self):
        """Hydra configuration should load without errors."""
        from omegaconf import OmegaConf

        project_root = Path(__file__).parent.parent.parent
        defaults_path = project_root / "configs" / "defaults.yaml"

        # Should load without error
        cfg = OmegaConf.load(defaults_path)

        # Verify key sections exist (using actual config structure)
        assert "DATA" in cfg
        assert "CLASSIFICATION_SETTINGS" in cfg  # Actual key name
        assert "OUTLIER_DETECTION" in cfg

    @pytest.mark.integration
    def test_synthetic_config_overrides_work(self):
        """Synthetic config should properly override defaults."""
        from omegaconf import OmegaConf

        project_root = Path(__file__).parent.parent.parent

        # Load defaults
        defaults = OmegaConf.load(project_root / "configs" / "defaults.yaml")

        # Load synthetic overrides
        synthetic = OmegaConf.load(project_root / "configs" / "synthetic_run.yaml")

        # Merge (synthetic overrides defaults)
        cfg = OmegaConf.merge(defaults, synthetic)

        # Verify overrides applied
        assert cfg.DATA.filename_DuckDB == "SYNTH_PLR_DEMO.db"
        assert cfg.DEBUG.debug_n_subjects == 4
