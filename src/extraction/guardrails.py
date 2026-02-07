"""
Production guardrails for MLflow extraction scripts.

Created after CRITICAL-FAILURE-005: A script ran for 24 hours stuck in swap
thrashing without being detected. This module provides guardrails to prevent
such failures.

Usage:
    from src.extraction.guardrails import ExtractionGuardrails, ExtractionConfig

    config = ExtractionConfig(
        max_memory_gb=4.0,
        heartbeat_seconds=60,
        max_stall_seconds=90,
    )
    guardrails = ExtractionGuardrails(config, total_items=136, output_path=OUTPUT_DB)

    for i, item in enumerate(items):
        process(item)
        guardrails.on_item_complete()

    guardrails.finish()
"""

import gc
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import psutil


@dataclass
class ExtractionConfig:
    """Configuration for extraction guardrails."""

    # Memory limits
    max_memory_gb: float = 4.0

    # Progress tracking
    heartbeat_seconds: int = 60

    # Stall detection - per-item timeout (should be shorter than total)
    max_stall_seconds: int = 90  # Fail if no progress for 90 seconds

    # Disk space
    min_disk_space_gb: float = 5.0

    # Batch processing
    batch_size: int = 1000

    # Network path validation timeout
    path_timeout_seconds: float = 5.0


@dataclass
class ProgressTracker:
    """Track progress with heartbeat logging."""

    total: int
    heartbeat_seconds: int = 60
    current: int = 0
    start_time: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)

    def update(self, n: int = 1) -> None:
        """Update progress and emit heartbeat if needed."""
        self.current += n
        if time.time() - self.last_heartbeat > self.heartbeat_seconds:
            self.heartbeat()

    def heartbeat(self) -> None:
        """Emit progress heartbeat with memory usage."""
        mem_gb = psutil.Process().memory_info().rss / (1024**3)
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        eta_seconds = (self.total - self.current) / rate if rate > 0 else float("inf")
        eta_min = eta_seconds / 60

        pct = self.current / self.total * 100 if self.total > 0 else 0

        print(
            f"[HEARTBEAT] {self.current}/{self.total} ({pct:.1f}%) "
            f"| Rate: {rate:.2f}/sec | ETA: {eta_min:.1f} min | Mem: {mem_gb:.2f}GB",
            flush=True,
        )
        self.last_heartbeat = time.time()

    def finish(self) -> None:
        """Emit final summary."""
        elapsed = time.time() - self.start_time
        mem_gb = psutil.Process().memory_info().rss / (1024**3)
        print(
            f"[COMPLETE] {self.current}/{self.total} items in {elapsed:.1f}s "
            f"| Final mem: {mem_gb:.2f}GB",
            flush=True,
        )


@dataclass
class StallDetector:
    """Detect when extraction is stuck (no progress for too long)."""

    max_stall_seconds: int = 90
    last_progress: float = field(default_factory=time.time)

    def mark_progress(self) -> None:
        """Mark that progress was made."""
        self.last_progress = time.time()

    def check(self) -> None:
        """Raise TimeoutError if no progress for too long."""
        stall_time = time.time() - self.last_progress
        if stall_time > self.max_stall_seconds:
            raise TimeoutError(
                f"EXTRACTION STALLED: No progress for {stall_time:.0f} seconds "
                f"(limit: {self.max_stall_seconds}s). Script appears stuck!"
            )


def check_memory(max_gb: float = 4.0) -> float:
    """Check memory usage and fail if exceeded.

    Returns current memory usage in GB.
    Raises MemoryError if limit exceeded.
    """
    process = psutil.Process()
    mem_gb = process.memory_info().rss / (1024**3)

    if mem_gb > max_gb:
        raise MemoryError(
            f"MEMORY LIMIT EXCEEDED: {mem_gb:.2f}GB > {max_gb}GB limit. "
            "Script is accumulating too much data in memory. "
            "Use streaming inserts instead of batch accumulation."
        )

    return mem_gb


def check_disk_space(path: Path, min_free_gb: float = 5.0) -> float:
    """Check disk space and fail if critically low.

    Returns free space in GB.
    Raises IOError if below minimum.
    """
    # Find the first existing parent directory
    check_path = path
    while not check_path.exists() and check_path.parent != check_path:
        check_path = check_path.parent

    usage = shutil.disk_usage(check_path)
    free_gb = usage.free / (1024**3)

    if free_gb < min_free_gb:
        raise IOError(
            f"DISK SPACE LOW: {free_gb:.1f}GB free < {min_free_gb}GB minimum. "
            "Cannot safely continue extraction."
        )

    return free_gb


def validate_mlflow_path(path: Path, timeout: float = 5.0) -> None:
    """Validate MLflow path is accessible and responsive.

    This catches network drive issues (NFS, SMB, Dropbox) before they
    cause the script to hang indefinitely.

    Raises:
        FileNotFoundError: If path doesn't exist
        TimeoutError: If path is unresponsive (network issue)
    """
    import signal

    if not path.exists():
        raise FileNotFoundError(f"MLflow path not found: {path}")

    # Check path is responsive with timeout
    def timeout_handler(_signum, _frame) -> None:
        raise TimeoutError(
            f"MLflow path not responding within {timeout}s: {path}. "
            "Check network drive connectivity."
        )

    # Set up alarm (Unix only)
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout))

    try:
        # Quick read test - list first few items
        list(path.iterdir())[:3]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class ExtractionGuardrails:
    """Unified guardrails for extraction scripts.

    Usage:
        guardrails = ExtractionGuardrails(config, total_items=136, output_path=OUTPUT_DB)

        for item in items:
            process(item)
            guardrails.on_item_complete()

        guardrails.finish()
    """

    def __init__(
        self,
        config: Optional[ExtractionConfig] = None,
        total_items: int = 0,
        output_path: Optional[Path] = None,
        mlflow_path: Optional[Path] = None,
    ) -> None:
        self.config = config or ExtractionConfig()
        self.output_path = output_path
        self.mlflow_path = mlflow_path

        # Initialize trackers
        self.progress = ProgressTracker(
            total=total_items,
            heartbeat_seconds=self.config.heartbeat_seconds,
        )
        self.stall_detector = StallDetector(
            max_stall_seconds=self.config.max_stall_seconds,
        )

        # Run initial checks
        self._preflight_checks()

    def _preflight_checks(self) -> None:
        """Run checks before extraction starts."""
        print("[PREFLIGHT] Running extraction guardrail checks...", flush=True)

        # Check MLflow path
        if self.mlflow_path:
            validate_mlflow_path(self.mlflow_path, self.config.path_timeout_seconds)
            print(f"  MLflow path OK: {self.mlflow_path}", flush=True)

        # Check disk space
        if self.output_path:
            free_gb = check_disk_space(self.output_path, self.config.min_disk_space_gb)
            print(f"  Disk space OK: {free_gb:.1f}GB free", flush=True)

        # Check current memory
        mem_gb = check_memory(self.config.max_memory_gb)
        print(f"  Memory OK: {mem_gb:.2f}GB used", flush=True)

        print("[PREFLIGHT] All checks passed. Starting extraction.", flush=True)

    def check_all(self) -> None:
        """Run all runtime checks."""
        check_memory(self.config.max_memory_gb)
        self.stall_detector.check()

        if self.output_path:
            check_disk_space(self.output_path, self.config.min_disk_space_gb)

    def on_item_complete(self, n_items: int = 1, force_gc: bool = True) -> None:
        """Call after processing each item (or batch of items).

        Args:
            n_items: Number of items completed (default 1)
            force_gc: Whether to force garbage collection
        """
        self.progress.update(n_items)
        self.stall_detector.mark_progress()

        if force_gc:
            gc.collect()

        # Run periodic checks (every heartbeat interval)
        if time.time() - self.progress.last_heartbeat > self.config.heartbeat_seconds:
            self.check_all()

    def on_batch_complete(self, batch_size: int) -> None:
        """Call after completing a batch insert."""
        self.on_item_complete(n_items=batch_size, force_gc=True)

    def finish(self) -> None:
        """Call when extraction is complete."""
        self.progress.finish()

        # Final memory check
        mem_gb = psutil.Process().memory_info().rss / (1024**3)
        print(f"[FINAL] Peak memory usage: {mem_gb:.2f}GB", flush=True)
