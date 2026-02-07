"""
Unit tests for streaming DuckDB export module.

Tests validate:
1. Memory monitoring
2. Checkpoint management
3. Schema creation
4. Metric extraction
"""

import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pytest


class TestMemoryMonitor:
    """Tests for MemoryMonitor class."""

    def test_memory_monitor_initialization(self):
        """Should initialize with correct thresholds."""
        from src.data_io.streaming_duckdb_export import MemoryMonitor

        monitor = MemoryMonitor(warning_threshold_gb=10.0, critical_threshold_gb=12.0)

        assert monitor.warning_threshold_gb == 10.0
        assert monitor.critical_threshold_gb == 12.0

    def test_memory_check_returns_tuple(self):
        """check() should return (usage_gb, status)."""
        from src.data_io.streaming_duckdb_export import MemoryMonitor

        monitor = MemoryMonitor()
        usage_gb, status = monitor.check()

        assert isinstance(usage_gb, float)
        assert status in ("ok", "warning", "critical")

    def test_memory_get_usage_gb(self):
        """get_usage_gb() should return positive float."""
        from src.data_io.streaming_duckdb_export import MemoryMonitor

        monitor = MemoryMonitor()
        usage = monitor.get_usage_gb()

        assert isinstance(usage, float)
        assert usage > 0

    def test_memory_enforce_does_not_crash(self):
        """enforce() should not raise exceptions."""
        from src.data_io.streaming_duckdb_export import MemoryMonitor

        monitor = MemoryMonitor()
        # Should not raise
        monitor.enforce()


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Initialize schema (matching the production schema)
            with duckdb.connect(str(db_path)) as con:
                con.execute("""
                    CREATE TABLE IF NOT EXISTS extraction_checkpoints (
                        run_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        started_at TEXT,
                        completed_at TEXT,
                        error_message TEXT
                    )
                """)

            yield db_path

    def test_checkpoint_manager_initialization(self, temp_db):
        """Should initialize with empty completed set."""
        from src.data_io.streaming_duckdb_export import CheckpointManager

        manager = CheckpointManager(temp_db)

        assert len(manager._completed_runs) == 0

    def test_is_completed_false_for_new_run(self, temp_db):
        """is_completed() should return False for new runs."""
        from src.data_io.streaming_duckdb_export import CheckpointManager

        manager = CheckpointManager(temp_db)

        assert manager.is_completed("run123") is False

    def test_mark_completed_updates_state(self, temp_db):
        """mark_completed() should update internal state."""
        from src.data_io.streaming_duckdb_export import CheckpointManager

        manager = CheckpointManager(temp_db)

        with duckdb.connect(str(temp_db)) as con:
            manager.mark_started("run123", con)
            manager.mark_completed("run123", con)

        assert manager.is_completed("run123") is True

    def test_get_progress_returns_dict(self, temp_db):
        """get_progress() should return statistics dict."""
        from src.data_io.streaming_duckdb_export import CheckpointManager

        manager = CheckpointManager(temp_db)
        progress = manager.get_progress()

        assert isinstance(progress, dict)
        assert "completed" in progress
        assert "failed" in progress


class TestEssentialMetricsSchema:
    """Tests for database schema."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database with schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            from src.data_io.streaming_duckdb_export import ESSENTIAL_METRICS_SCHEMA

            with duckdb.connect(str(db_path)) as con:
                con.execute(ESSENTIAL_METRICS_SCHEMA)

            yield db_path

    def test_essential_metrics_table_exists(self, temp_db):
        """essential_metrics table should exist."""
        with duckdb.connect(str(temp_db)) as con:
            result = con.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='essential_metrics'
            """).fetchone()

        assert result is not None

    def test_supplementary_metrics_table_exists(self, temp_db):
        """supplementary_metrics table should exist."""
        with duckdb.connect(str(temp_db)) as con:
            result = con.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='supplementary_metrics'
            """).fetchone()

        assert result is not None

    def test_can_insert_essential_metrics(self, temp_db):
        """Should be able to insert essential metrics."""
        with duckdb.connect(str(temp_db)) as con:
            con.execute("""
                INSERT INTO essential_metrics (
                    config_id, outlier_method, imputation_method,
                    classifier, source_name, auroc
                ) VALUES (1, 'IQR', 'SAITS', 'CatBoost', 'test_run', 0.91)
            """)

            result = con.execute(
                "SELECT auroc FROM essential_metrics WHERE config_id = 1"
            ).fetchone()

        # Use approximate comparison for float
        np.testing.assert_allclose(result[0], 0.91, rtol=1e-5)

    def test_can_insert_calibration_curves(self, temp_db):
        """Should be able to insert calibration curve data."""
        with duckdb.connect(str(temp_db)) as con:
            # First insert a config
            con.execute("""
                INSERT INTO essential_metrics (
                    config_id, outlier_method, imputation_method,
                    classifier, source_name
                ) VALUES (1, 'IQR', 'SAITS', 'CatBoost', 'test_run')
            """)

            # Then insert calibration data (include curve_id)
            con.execute("""
                INSERT INTO calibration_curves (
                    curve_id, config_id, bin_index, bin_midpoint, observed_proportion, n_samples
                ) VALUES (1, 1, 0, 0.05, 0.03, 10)
            """)

            result = con.execute(
                "SELECT bin_midpoint FROM calibration_curves WHERE config_id = 1"
            ).fetchone()

        np.testing.assert_allclose(result[0], 0.05, rtol=1e-5)


class TestStreamingExporterUnit:
    """Unit tests for StreamingDuckDBExporter helper methods."""

    def test_parse_run_name_simple(self):
        """Should parse simple run names."""
        from src.data_io.streaming_duckdb_export import StreamingDuckDBExporter

        # Create minimal instance for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            mlruns = Path(tmpdir) / "mlruns"
            mlruns.mkdir()
            exp_dir = mlruns / "1"
            exp_dir.mkdir()
            run_dir = exp_dir / "run1"
            run_dir.mkdir()
            (run_dir / "meta.yaml").write_text("run_name: test")

            exporter = StreamingDuckDBExporter(
                mlruns, Path(tmpdir) / "test.db", experiment_id="1"
            )

            config = exporter._parse_run_name("CatBoost_auroc__handcrafted__SAITS__IQR")

            assert config["classifier"] == "CatBoost"
            assert config["imputation_method"] == "SAITS"
            assert config["outlier_method"] == "IQR"


class TestImports:
    """Test that module imports work correctly."""

    def test_import_streaming_duckdb_export(self):
        """Should be able to import module."""

    def test_import_schema(self):
        """Should be able to import schema."""
        from src.data_io.streaming_duckdb_export import ESSENTIAL_METRICS_SCHEMA

        assert "essential_metrics" in ESSENTIAL_METRICS_SCHEMA
        assert "supplementary_metrics" in ESSENTIAL_METRICS_SCHEMA
