"""
Tests for extraction guardrails module.

These tests verify that the guardrails properly detect and fail on:
- Memory limit exceeded
- Stall detection (no progress)
- Low disk space
- Unresponsive paths
"""

import tempfile
import time
from pathlib import Path

import pytest

from src.extraction.guardrails import (
    ExtractionConfig,
    ExtractionGuardrails,
    ProgressTracker,
    StallDetector,
    check_disk_space,
    check_memory,
)


class TestCheckMemory:
    """Tests for memory limit checking."""

    def test_check_memory_returns_current_usage(self):
        """check_memory should return current memory usage."""
        mem_gb = check_memory(max_gb=100.0)  # High limit to not trigger
        assert isinstance(mem_gb, float)
        assert mem_gb > 0

    def test_check_memory_raises_on_limit_exceeded(self):
        """check_memory should raise MemoryError when limit exceeded."""
        # Set limit impossibly low (0.001 GB = 1MB)
        with pytest.raises(MemoryError, match="MEMORY LIMIT EXCEEDED"):
            check_memory(max_gb=0.001)


class TestCheckDiskSpace:
    """Tests for disk space checking."""

    def test_check_disk_space_returns_free_space(self):
        """check_disk_space should return free space in GB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            free_gb = check_disk_space(Path(tmpdir), min_free_gb=0.001)
            assert isinstance(free_gb, float)
            assert free_gb > 0

    def test_check_disk_space_raises_on_low_space(self):
        """check_disk_space should raise IOError when space is low."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set min_free impossibly high
            with pytest.raises(IOError, match="DISK SPACE LOW"):
                check_disk_space(Path(tmpdir), min_free_gb=1_000_000)  # 1 PB


class TestProgressTracker:
    """Tests for progress tracking."""

    def test_progress_tracker_updates(self):
        """ProgressTracker should track progress correctly."""
        tracker = ProgressTracker(total=100, heartbeat_seconds=1000)
        assert tracker.current == 0

        tracker.update(10)
        assert tracker.current == 10

        tracker.update(5)
        assert tracker.current == 15

    def test_progress_tracker_heartbeat(self, capsys):
        """ProgressTracker should emit heartbeat."""
        tracker = ProgressTracker(total=100, heartbeat_seconds=0)  # Force heartbeat
        tracker.update(50)
        tracker.heartbeat()

        captured = capsys.readouterr()
        assert "[HEARTBEAT]" in captured.out
        assert "50/100" in captured.out

    def test_progress_tracker_finish(self, capsys):
        """ProgressTracker should emit completion message."""
        tracker = ProgressTracker(total=100)
        tracker.current = 100
        tracker.finish()

        captured = capsys.readouterr()
        assert "[COMPLETE]" in captured.out


class TestStallDetector:
    """Tests for stall detection."""

    def test_stall_detector_passes_when_active(self):
        """StallDetector should not raise when progress is being made."""
        detector = StallDetector(max_stall_seconds=1)
        detector.mark_progress()
        detector.check()  # Should not raise

    def test_stall_detector_raises_on_stall(self):
        """StallDetector should raise TimeoutError after stall."""
        detector = StallDetector(max_stall_seconds=0)  # Immediate timeout
        detector.last_progress = time.time() - 10  # Pretend 10s ago

        with pytest.raises(TimeoutError, match="EXTRACTION STALLED"):
            detector.check()


class TestExtractionConfig:
    """Tests for ExtractionConfig dataclass."""

    def test_default_config(self):
        """Default config should have reasonable values."""
        config = ExtractionConfig()
        assert config.max_memory_gb == 4.0
        assert config.heartbeat_seconds == 60
        assert config.max_stall_seconds == 90
        assert config.min_disk_space_gb == 5.0

    def test_custom_config(self):
        """Custom config should override defaults."""
        config = ExtractionConfig(
            max_memory_gb=8.0,
            heartbeat_seconds=30,
            max_stall_seconds=60,
        )
        assert config.max_memory_gb == 8.0
        assert config.heartbeat_seconds == 30
        assert config.max_stall_seconds == 60


class TestExtractionGuardrails:
    """Tests for the unified ExtractionGuardrails class."""

    def test_guardrails_init_runs_preflight(self, capsys):
        """ExtractionGuardrails should run preflight checks on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExtractionConfig(max_memory_gb=100.0)  # High limit
            ExtractionGuardrails(
                config=config,
                total_items=10,
                output_path=Path(tmpdir) / "test.db",
            )

            captured = capsys.readouterr()
            assert "[PREFLIGHT]" in captured.out

    def test_guardrails_on_item_complete(self):
        """on_item_complete should update progress and mark stall detector."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExtractionConfig(max_memory_gb=100.0)
            guardrails = ExtractionGuardrails(
                config=config,
                total_items=10,
                output_path=Path(tmpdir) / "test.db",
            )

            assert guardrails.progress.current == 0
            guardrails.on_item_complete()
            assert guardrails.progress.current == 1

    def test_guardrails_finish(self, capsys):
        """finish() should emit final summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExtractionConfig(max_memory_gb=100.0)
            guardrails = ExtractionGuardrails(
                config=config,
                total_items=10,
                output_path=Path(tmpdir) / "test.db",
            )
            guardrails.progress.current = 10
            guardrails.finish()

            captured = capsys.readouterr()
            assert "[COMPLETE]" in captured.out or "[FINAL]" in captured.out
