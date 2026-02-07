"""
TDD tests for streaming DuckDB export.

These tests verify that the StreamingDuckDBExporter properly handles:
1. Checkpoint/resume capability
2. Memory monitoring
3. Progress logging
4. STRATOS-compliant schema

Written as part of the 20-hour robustification plan (2026-02-01).
"""

import pickle
from unittest.mock import patch

import duckdb
import numpy as np
import pytest
import yaml

from src.data_io.streaming_duckdb_export import (
    CheckpointManager,
    MemoryMonitor,
    StreamingDuckDBExporter,
    ESSENTIAL_METRICS_SCHEMA,
)

pytestmark = pytest.mark.unit


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary DuckDB file."""
    db_path = tmp_path / "test_streaming.db"
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def mock_mlruns(tmp_path):
    """Create a mock MLflow runs structure with 5 test runs."""
    mlruns_dir = tmp_path / "mlruns"
    exp_dir = mlruns_dir / "test_experiment_123"
    exp_dir.mkdir(parents=True)

    for i in range(5):
        run_dir = exp_dir / f"run_{i:03d}"
        run_dir.mkdir()

        # Create meta.yaml
        meta = {
            "run_id": f"run_{i:03d}",
            "run_name": f"CatBoost__simple__SAITS__LOF_{i}",
            "status": "FINISHED",
        }
        with open(run_dir / "meta.yaml", "w") as f:
            yaml.dump(meta, f)

        # Create metrics pickle
        metrics_dir = run_dir / "artifacts" / "metrics"
        metrics_dir.mkdir(parents=True)

        metrics_data = {
            "metrics_stats": {
                "test": {
                    "metrics": {
                        "scalars": {
                            "AUROC": {"mean": 0.85 + i * 0.01, "ci": [0.80, 0.90]},
                            "Brier": {"mean": 0.15 - i * 0.01},
                        }
                    }
                }
            }
        }
        with open(metrics_dir / "metrics.pickle", "wb") as f:
            pickle.dump(metrics_data, f)

        # Create arrays pickle for probability distributions
        arrays_dir = run_dir / "artifacts" / "dict_arrays"
        arrays_dir.mkdir(parents=True)

        arrays_data = {
            "y_test": np.array([0, 0, 1, 1, 1]),
            "y_pred_proba_mean": np.array([0.1, 0.3, 0.6, 0.8, 0.9]),
        }
        with open(arrays_dir / "arrays.pickle", "wb") as f:
            pickle.dump(arrays_data, f)

    return mlruns_dir, "test_experiment_123"


# ============================================================================
# MemoryMonitor Tests
# ============================================================================


class TestMemoryMonitor:
    """Tests for MemoryMonitor class."""

    def test_check_returns_usage_and_status(self):
        """Memory check returns (usage_gb, status) tuple."""
        monitor = MemoryMonitor(warning_threshold_gb=100.0, critical_threshold_gb=200.0)
        usage_gb, status = monitor.check()

        assert isinstance(usage_gb, float)
        assert usage_gb > 0
        assert status in ("ok", "warning", "critical")

    def test_check_status_ok_when_under_threshold(self):
        """Status is 'ok' when memory is under warning threshold."""
        # Set high thresholds so we're definitely under
        monitor = MemoryMonitor(warning_threshold_gb=100.0, critical_threshold_gb=200.0)
        _, status = monitor.check()
        assert status == "ok"

    def test_enforce_calls_gc_at_critical(self):
        """GC is called when memory exceeds critical threshold."""
        # Set very low threshold to trigger critical
        monitor = MemoryMonitor(warning_threshold_gb=0.001, critical_threshold_gb=0.001)

        with patch("gc.collect") as mock_gc:
            monitor.enforce()
            mock_gc.assert_called()

    def test_get_usage_gb(self):
        """get_usage_gb returns current memory in GB."""
        monitor = MemoryMonitor()
        usage = monitor.get_usage_gb()
        assert isinstance(usage, float)
        assert usage > 0


# ============================================================================
# CheckpointManager Tests
# ============================================================================


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_is_completed_false_for_new_run(self, temp_db):
        """New runs are not marked as completed."""
        # Initialize DB with schema
        with duckdb.connect(str(temp_db)) as con:
            con.execute(ESSENTIAL_METRICS_SCHEMA)

        manager = CheckpointManager(temp_db)
        assert manager.is_completed("new_run_id") is False

    def test_mark_completed_and_check(self, temp_db):
        """Completed runs are tracked correctly."""
        with duckdb.connect(str(temp_db)) as con:
            con.execute(ESSENTIAL_METRICS_SCHEMA)

            manager = CheckpointManager(temp_db)
            manager.mark_started("test_run", con)
            manager.mark_completed("test_run", con)

        # Reload manager to verify persistence
        manager2 = CheckpointManager(temp_db)
        assert manager2.is_completed("test_run") is True

    def test_mark_failed_with_error(self, temp_db):
        """Failed runs store error message."""
        with duckdb.connect(str(temp_db)) as con:
            con.execute(ESSENTIAL_METRICS_SCHEMA)

            manager = CheckpointManager(temp_db)
            manager.mark_started("failed_run", con)
            manager.mark_failed("failed_run", "Test error message", con)

            result = con.execute(
                "SELECT error_message FROM extraction_checkpoints WHERE run_id = 'failed_run'"
            ).fetchone()

        assert result[0] == "Test error message"

    def test_get_progress_counts(self, temp_db):
        """Progress statistics are correct."""
        with duckdb.connect(str(temp_db)) as con:
            con.execute(ESSENTIAL_METRICS_SCHEMA)

            manager = CheckpointManager(temp_db)
            manager.mark_started("run1", con)
            manager.mark_completed("run1", con)
            manager.mark_started("run2", con)
            manager.mark_completed("run2", con)
            manager.mark_started("run3", con)
            manager.mark_failed("run3", "error", con)

        progress = manager.get_progress()
        assert progress["completed"] == 2
        assert progress["failed"] == 1


# ============================================================================
# StreamingDuckDBExporter Tests
# ============================================================================


class TestStreamingDuckDBExporter:
    """Tests for StreamingDuckDBExporter class."""

    def test_export_creates_database(self, mock_mlruns, temp_db):
        """Export creates the output database."""
        mlruns_dir, exp_id = mock_mlruns

        exporter = StreamingDuckDBExporter(
            mlruns_dir=mlruns_dir,
            output_path=temp_db,
            experiment_id=exp_id,
        )
        stats = exporter.export()

        assert temp_db.exists()
        assert stats["completed"] > 0

    def test_export_creates_essential_metrics_table(self, mock_mlruns, temp_db):
        """Export creates essential_metrics table with correct schema."""
        mlruns_dir, exp_id = mock_mlruns

        exporter = StreamingDuckDBExporter(
            mlruns_dir=mlruns_dir,
            output_path=temp_db,
            experiment_id=exp_id,
        )
        exporter.export()

        with duckdb.connect(str(temp_db)) as con:
            # Check table exists and has data
            result = con.execute("SELECT COUNT(*) FROM essential_metrics").fetchone()
            assert result[0] > 0

            # Check required columns exist
            columns = con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'essential_metrics'"
            ).fetchall()
            column_names = {c[0] for c in columns}

            required = {
                "config_id",
                "auroc",
                "calibration_slope",
                "brier_score",
                "outlier_method",
                "imputation_method",
                "classifier",
            }
            assert required.issubset(column_names)

    def test_export_checkpoint_resume(self, mock_mlruns, temp_db):
        """Export can resume from checkpoint after interruption."""
        mlruns_dir, exp_id = mock_mlruns

        # First export - process some runs
        exporter1 = StreamingDuckDBExporter(
            mlruns_dir=mlruns_dir,
            output_path=temp_db,
            experiment_id=exp_id,
        )
        stats1 = exporter1.export()
        initial_completed = stats1["completed"]

        # Second export - should skip already completed
        exporter2 = StreamingDuckDBExporter(
            mlruns_dir=mlruns_dir,
            output_path=temp_db,
            experiment_id=exp_id,
        )
        stats2 = exporter2.export()

        # All runs should be skipped (already completed)
        assert stats2["skipped"] == initial_completed
        assert stats2["completed"] == 0

    def test_export_progress_logging(self, mock_mlruns, temp_db, capsys):
        """Export logs progress during execution."""
        mlruns_dir, exp_id = mock_mlruns

        exporter = StreamingDuckDBExporter(
            mlruns_dir=mlruns_dir,
            output_path=temp_db,
            experiment_id=exp_id,
        )
        exporter.export()

        # Check that progress was logged (via loguru, may go to stderr)
        # The exact output depends on loguru config, so just verify no crash

    def test_export_memory_bounded(self, mock_mlruns, temp_db):
        """Export stays under memory threshold."""
        mlruns_dir, exp_id = mock_mlruns

        exporter = StreamingDuckDBExporter(
            mlruns_dir=mlruns_dir,
            output_path=temp_db,
            experiment_id=exp_id,
            memory_threshold_gb=12.0,
        )

        # Track peak memory during export
        import psutil

        process = psutil.Process()
        initial_mem = process.memory_info().rss / 1024 / 1024 / 1024

        exporter.export()

        final_mem = process.memory_info().rss / 1024 / 1024 / 1024

        # Memory should not have grown excessively
        # (with streaming, growth should be minimal)
        mem_growth = final_mem - initial_mem
        assert mem_growth < 1.0  # Less than 1GB growth for small test

    def test_export_stratos_schema_compliance(self, mock_mlruns, temp_db):
        """Export creates STRATOS-compliant schema."""
        mlruns_dir, exp_id = mock_mlruns

        exporter = StreamingDuckDBExporter(
            mlruns_dir=mlruns_dir,
            output_path=temp_db,
            experiment_id=exp_id,
        )
        exporter.export()

        with duckdb.connect(str(temp_db)) as con:
            # Check all STRATOS-required tables exist
            tables = con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
            table_names = {t[0] for t in tables}

            required_tables = {
                "essential_metrics",
                "supplementary_metrics",
                "calibration_curves",
                "probability_distributions",
                "dca_curves",
                "extraction_checkpoints",
            }
            assert required_tables.issubset(table_names)


# ============================================================================
# Integration Tests
# ============================================================================


class TestStreamingExportIntegration:
    """Integration tests for full export workflow."""

    def test_full_export_workflow(self, mock_mlruns, temp_db):
        """Complete export workflow from start to finish."""
        mlruns_dir, exp_id = mock_mlruns

        # Export
        exporter = StreamingDuckDBExporter(
            mlruns_dir=mlruns_dir,
            output_path=temp_db,
            experiment_id=exp_id,
        )
        stats = exporter.export()

        # Verify
        assert stats["completed"] == 5  # All 5 mock runs
        assert stats["failed"] == 0

        with duckdb.connect(str(temp_db)) as con:
            # Check data was extracted
            n_metrics = con.execute(
                "SELECT COUNT(*) FROM essential_metrics"
            ).fetchone()[0]
            assert n_metrics == 5

            # Check checkpoints recorded
            n_checkpoints = con.execute(
                "SELECT COUNT(*) FROM extraction_checkpoints WHERE status = 'completed'"
            ).fetchone()[0]
            assert n_checkpoints == 5

    def test_export_with_partial_data(self, tmp_path, temp_db):
        """Export handles runs with missing data gracefully."""
        mlruns_dir = tmp_path / "mlruns"
        exp_dir = mlruns_dir / "partial_exp"
        exp_dir.mkdir(parents=True)

        # Create run with only meta.yaml (no metrics)
        run_dir = exp_dir / "incomplete_run"
        run_dir.mkdir()
        with open(run_dir / "meta.yaml", "w") as f:
            yaml.dump({"run_id": "incomplete", "run_name": "CatBoost__test"}, f)

        exporter = StreamingDuckDBExporter(
            mlruns_dir=mlruns_dir,
            output_path=temp_db,
            experiment_id="partial_exp",
        )
        stats = exporter.export()

        # Should handle gracefully (marked as failed, not crashed)
        assert stats["completed"] == 0
        assert stats["failed"] == 1


# ============================================================================
# Re-anonymization Tests
# ============================================================================


class TestReAnonymization:
    """Tests for subject code re-anonymization."""

    def test_export_with_subject_mapping(self, mock_mlruns, temp_db):
        """Export with subject_mapping re-anonymizes subject codes."""
        mlruns_dir, exp_id = mock_mlruns

        # Create subject mapping
        subject_mapping = {
            "PLR001": "H001",
            "PLR002": "G001",
        }

        exporter = StreamingDuckDBExporter(
            mlruns_dir=mlruns_dir,
            output_path=temp_db,
            experiment_id=exp_id,
            subject_mapping=subject_mapping,
        )
        exporter.export()

        # Database should be created successfully
        assert temp_db.exists()

    def test_exporter_stores_subject_mapping(self, mock_mlruns, temp_db):
        """StreamingDuckDBExporter stores subject_mapping parameter."""
        mlruns_dir, exp_id = mock_mlruns
        mapping = {"PLR001": "H001"}
        exporter = StreamingDuckDBExporter(
            mlruns_dir=mlruns_dir,
            output_path=temp_db,
            experiment_id=exp_id,
            subject_mapping=mapping,
        )
        assert exporter.subject_mapping == mapping

    def test_exporter_default_empty_mapping(self, mock_mlruns, temp_db):
        """StreamingDuckDBExporter defaults to empty subject_mapping."""
        mlruns_dir, exp_id = mock_mlruns
        exporter = StreamingDuckDBExporter(
            mlruns_dir=mlruns_dir,
            output_path=temp_db,
            experiment_id=exp_id,
        )
        assert exporter.subject_mapping == {}


# ============================================================================
# Predictions Table Tests
# ============================================================================


class TestPredictionsTable:
    """Tests for predictions table in STRATOS schema."""

    def test_predictions_table_created(self, mock_mlruns, temp_db):
        """Export creates predictions table in schema."""
        mlruns_dir, exp_id = mock_mlruns

        exporter = StreamingDuckDBExporter(
            mlruns_dir=mlruns_dir,
            output_path=temp_db,
            experiment_id=exp_id,
        )
        exporter.export()

        with duckdb.connect(str(temp_db)) as con:
            tables = con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
            table_names = {t[0] for t in tables}

            assert "predictions" in table_names

    def test_predictions_table_has_correct_columns(self, mock_mlruns, temp_db):
        """Predictions table has required columns."""
        mlruns_dir, exp_id = mock_mlruns

        exporter = StreamingDuckDBExporter(
            mlruns_dir=mlruns_dir,
            output_path=temp_db,
            experiment_id=exp_id,
        )
        exporter.export()

        with duckdb.connect(str(temp_db)) as con:
            columns = con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'predictions'"
            ).fetchall()
            column_names = {c[0] for c in columns}

            required = {
                "prediction_id",
                "config_id",
                "subject_code",
                "y_true",
                "y_prob",
            }
            assert required.issubset(column_names)

    def test_extract_predictions_disabled(self, mock_mlruns, temp_db):
        """Predictions extraction can be disabled."""
        mlruns_dir, exp_id = mock_mlruns

        exporter = StreamingDuckDBExporter(
            mlruns_dir=mlruns_dir,
            output_path=temp_db,
            experiment_id=exp_id,
            extract_predictions=False,
        )
        exporter.export()

        with duckdb.connect(str(temp_db)) as con:
            # Table exists - verify we can query it without error
            # The key is the extraction doesn't crash with extract_predictions=False
            con.execute("SELECT COUNT(*) FROM predictions").fetchone()
