# Prefect MLflow-to-DuckDB Extraction: Test Suite Improvement Plan

**Status**: REVIEWED - Critical issues identified
**Created**: 2026-02-01
**Triggered by**: 6+ hour extraction with NO VISIBLE PROGRESS OUTPUT
**Related**: CRITICAL-FAILURE-005 (24-hour stuck extraction)

---

## ⚠️ CRITICAL REVIEW FINDINGS (2026-02-01)

### The REAL Root Cause (NOT progress visibility)

The plan initially treated "no progress visibility" as the root cause. **WRONG.**

The actual root cause is in `extraction_flow.py` lines 239-243:
```python
return {
    "predictions": predictions,      # Accumulates ALL 410 runs
    "metrics_aggregate": metrics_aggregate,
    "run_metadata": run_metadata,
}
```

**This is the EXACT SAME PATTERN as CRITICAL-FAILURE-005** - batch accumulation before write.

### Critical Architecture Issues

| Issue | Location | Impact |
|-------|----------|--------|
| Batch accumulation | Lines 210-243 | OOM after ~300 runs |
| No streaming inserts | Lines 450-468 | Row-by-row insert after full accumulation |
| Progress logging inside batch task | Line 215-216 | Only logs every 50 runs, task doesn't yield |
| No checkpoint resume | Entire flow | Kill = start from scratch |

### Corrected Priority Order

```
P-1: Debug first (identify where stalls occur)
P0.0: Architecture decision (per-run streaming vs batch)
P0.1: FIX STREAMING INSERTS (this alone fixes OOM)
P0.2: Add checkpoint table
P0.3: Add resume logic
P0.4: Add progress logging (nice-to-have)
```

### Chosen Approach: Option C (Use Existing Code - The Right Move)

**CRITICAL DISCOVERY**: `src/data_io/streaming_duckdb_export.py` ALREADY HAS EVERYTHING:

| Feature | Status | Location |
|---------|--------|----------|
| CheckpointManager | ✅ IMPLEMENTED | Lines 225-314 |
| MemoryMonitor | ✅ IMPLEMENTED | Lines 151-217 |
| StreamingDuckDBExporter | ✅ IMPLEMENTED | Lines 322-1101 |
| STRATOS schema | ✅ IMPLEMENTED | Lines 43-143 |
| Per-run writes | ✅ IMPLEMENTED | `_extract_run()` method |
| Progress logging | ✅ IMPLEMENTED | Every 10 runs |
| CLI entry point | ✅ IMPLEMENTED | Lines 1103-1136 |

**The correct approach is Option C**:
- **Effort**: 20 hours
- **Solves**: Everything (OOM, checkpoints, resume, progress, STRATOS compliance)
- **Approach**: Refactor `extraction_flow.py` to USE `StreamingDuckDBExporter`
- **Benefit**: Single source of truth, no duplicate code

This was already built and sitting unused while 30+ hours were wasted.

---

## 0. Implementation Plan (Option C)

### Step 1: Verify StreamingDuckDBExporter Works (1 hour)

```bash
# Test the existing streaming exporter directly
uv run python -m src.data_io.streaming_duckdb_export \
    /home/petteri/mlruns \
    data/test_streaming_export.db \
    --experiment 253031330985650090
```

Expected output:
- Progress logging every 10 runs
- Memory monitoring
- Checkpoint table populated
- Can kill and resume

### Step 2: Refactor extraction_flow.py (4 hours)

Replace the batch accumulation approach with:

```python
from src.data_io.streaming_duckdb_export import StreamingDuckDBExporter

@task(name="extract_mlflow_with_streaming")
def extract_mlflow_with_streaming(
    mlruns_dir: Path,
    experiment_id: str,
    output_path: Path,
) -> Dict[str, Any]:
    """Use the production-grade streaming exporter."""
    exporter = StreamingDuckDBExporter(
        mlruns_dir=mlruns_dir,
        output_path=output_path,
        experiment_id=experiment_id,
        memory_threshold_gb=12.0,
    )
    stats = exporter.export()
    return stats
```

### Step 3: Integrate Re-anonymization (4 hours)

The current StreamingDuckDBExporter doesn't handle re-anonymization.
Options:
1. Add `original_to_anon` parameter to StreamingDuckDBExporter
2. Post-process the DB to re-anonymize subject codes
3. Pre-process subject codes in pickle loading

**Recommendation**: Option 1 - add parameter to StreamingDuckDBExporter

### Step 4: Add Tests (4 hours)

```python
# tests/test_data_io/test_streaming_export.py
class TestStreamingExporter:
    def test_checkpoint_resume(self, mock_mlruns, tmp_db):
        """Kill and resume completes extraction."""

    def test_memory_bounded(self, mock_mlruns):
        """Peak memory stays under threshold."""

    def test_progress_logging(self, mock_mlruns, capsys):
        """Progress appears in output."""
```

### Step 5: Update Prefect Flow (4 hours)

- Replace `extract_mlflow_runs()` batch task with streaming version
- Remove `export_to_duckdb()` task (now integrated)
- Keep demo traces extraction separate

### Step 6: Validation (4 hours)

- Run full extraction
- Compare output with expected schema
- Verify STRATOS metrics populated
- Test interrupt/resume

**Total: 20 hours**

---

## 1. Current State Analysis

### 1.1 What's Broken

| Issue | Impact | Root Cause |
|-------|--------|------------|
| **No visible progress output** | User has no idea if extraction is running or stuck | Guardrails exist (`src/extraction/guardrails.py`) but are NOT integrated into the Prefect flow |
| **No checkpoint/resume** | If killed, must restart from scratch | `extraction_flow.py` uses Prefect tasks but no checkpointing strategy |
| **Guardrails module unused** | 30+ hours wasted compute | `ExtractionGuardrails` class exists but is not called in `extraction_flow.py` |
| **Inconsistent architectures** | Two parallel systems | `scripts/extract_*.py` (standalone) vs `src/orchestration/flows/extraction_flow.py` (Prefect) |

### 1.2 Architecture Mismatch

We have TWO extraction systems that are not aligned:

**System A: Standalone Scripts** (`scripts/extract_*.py`)
- 10 extraction scripts
- Some use `ExtractionGuardrails` (after CRITICAL-FAILURE-005 fix)
- No Prefect integration
- No checkpoint/resume

**System B: Prefect Flow** (`src/orchestration/flows/extraction_flow.py`)
- Uses Prefect `@task` and `@flow` decorators
- Has retries (`retries=2`) but NO progress logging
- Does NOT use `ExtractionGuardrails`
- Does NOT have checkpoint/resume

**System C: Streaming Exporter** (`src/data_io/streaming_duckdb_export.py`)
- Has `CheckpointManager` class (resume capability)
- Has `MemoryMonitor` class
- Has per-run streaming (not batch)
- NOT integrated with Prefect or the extraction flow

### 1.3 Guardrails Module Analysis

The guardrails module at `src/extraction/guardrails.py` is well-designed but UNUSED in the Prefect flow:

```python
# Available but NOT USED:
class ExtractionGuardrails:
    - preflight_checks()     # Validates paths, disk space, memory
    - on_item_complete()     # Updates progress, checks stall
    - check_all()            # Memory + stall + disk checks
    - finish()               # Emits summary

class ProgressTracker:
    - heartbeat()            # "[HEARTBEAT] 50/136 (36.8%) | Rate: 0.5/sec | ETA: 3.2 min"

class StallDetector:
    - check()                # TimeoutError if no progress for 90 seconds
```

### 1.4 Existing Tests

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `tests/test_extraction/test_guardrails.py` | 14 | Memory, stall, progress, preflight |
| `tests/test_extraction/test_no_memory_accumulation.py` | 2 | AST pattern detection |
| `tests/integration/test_extraction_registry.py` | 8 | Registry validation |
| `tests/test_orchestration_flows.py` | 28 | Prefect flow structure |

**Missing Tests:**
- Integration of guardrails WITH Prefect flow
- Crash recovery/resume
- Per-pickle progress visibility
- End-to-end extraction with monitoring

---

## 2. Architecture for Crash-Resistant Extraction

### 2.1 Target Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Prefect Extraction Flow                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  Preflight   │───>│  Per-Pickle  │───>│   Finalize   │               │
│  │    Task      │    │    Tasks     │    │    Task      │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│         │                   │                   │                        │
│         v                   v                   v                        │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              ExtractionGuardrails (Unified)                       │   │
│  │  - Memory monitoring (4GB limit)                                  │   │
│  │  - Stall detection (90 second timeout)                            │   │
│  │  - Heartbeat logging (every 60s)                                  │   │
│  │  - Disk space checks                                              │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│         │                   │                   │                        │
│         v                   v                   v                        │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              CheckpointManager (Resume Support)                   │   │
│  │  - extraction_checkpoints table in DuckDB                         │   │
│  │  - is_completed(run_id) check before processing                   │   │
│  │  - mark_started/completed/failed per pickle                       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Design Decisions

1. **Merge streaming_duckdb_export.py checkpoint logic into extraction_flow.py**
   - Use `CheckpointManager` class from streaming module
   - DuckDB table tracks: run_id, status, started_at, completed_at, error_message

2. **Per-Pickle Task Granularity**
   - Each pickle is a separate Prefect task (for state tracking)
   - Can resume from any failure point

3. **Unified Guardrails Integration**
   - `ExtractionGuardrails` initialized at flow start
   - `on_item_complete()` called after each pickle
   - Heartbeat output every 60 seconds with ETA

4. **Streaming Inserts (NOT Batch)**
   - Write to DuckDB per pickle, not batch at end
   - Memory bounded to ~400MB instead of 13GB

### 2.3 Checkpoint/Resume Strategy

```python
# DuckDB Table: extraction_checkpoints
CREATE TABLE IF NOT EXISTS extraction_checkpoints (
    run_id TEXT PRIMARY KEY,
    pickle_path TEXT,
    status TEXT CHECK (status IN ('pending', 'started', 'completed', 'failed')),
    started_at TEXT,
    completed_at TEXT,
    error_message TEXT
);

# Resume Logic:
def should_process(pickle_path: Path, checkpoint_mgr: CheckpointManager) -> bool:
    run_id = pickle_path.stem  # Or compute stable ID
    return not checkpoint_mgr.is_completed(run_id)
```

---

## 3. Test Suite Specification

### 3.1 New Test File: `tests/test_extraction/test_prefect_integration.py`

```python
"""
Tests for Prefect extraction flow with guardrails integration.
"""

class TestProgressVisibility:
    """Every pickle processed should log with ETA."""

    def test_heartbeat_emitted_during_extraction(self, capsys, mock_mlflow):
        """Verify heartbeat appears in output during extraction."""

    def test_progress_includes_eta(self, capsys, mock_mlflow):
        """Verify ETA is calculated and displayed."""

    def test_per_pickle_progress_log(self, capsys, mock_mlflow):
        """Verify each pickle completion is logged."""

    def test_progress_includes_memory_usage(self, capsys, mock_mlflow):
        """Verify memory usage is reported."""


class TestCrashRecovery:
    """Checkpoint/resume capability tests."""

    def test_resume_skips_completed_pickles(self, tmp_db, mock_mlflow):
        """If extraction is restarted, completed pickles are skipped."""

    def test_checkpoint_table_created(self, tmp_db, mock_mlflow):
        """Checkpoint table is created on first run."""

    def test_failed_pickles_retried_on_restart(self, tmp_db, mock_mlflow):
        """Failed pickles are retried on restart (not skipped)."""

    def test_checkpoint_status_transitions(self, tmp_db, mock_mlflow):
        """Verify status: pending -> started -> completed."""

    def test_graceful_shutdown_saves_checkpoint(self, tmp_db, mock_mlflow):
        """SIGINT during extraction saves checkpoint."""


class TestMemoryBounds:
    """Memory monitoring tests."""

    def test_extraction_stays_under_memory_limit(self, mock_mlflow):
        """Extraction should not exceed 4GB RAM."""

    def test_memory_warning_logged_at_threshold(self, capsys, mock_mlflow):
        """Warning emitted when memory exceeds warning threshold."""

    def test_gc_collect_called_after_each_pickle(self, mock_mlflow):
        """gc.collect() is called after processing each pickle."""


class TestStallDetection:
    """Stall detection tests."""

    def test_timeout_raised_on_stall(self, mock_mlflow):
        """TimeoutError raised if no progress for 90 seconds."""

    def test_stall_detector_reset_on_progress(self, mock_mlflow):
        """Stall timer resets after each successful pickle."""
```

### 3.2 New Test File: `tests/test_extraction/test_guardrails_prefect_bridge.py`

```python
"""
Tests for guardrails integration with Prefect tasks.
"""

class TestGuardrailsPrefectBridge:
    """Verify guardrails work within Prefect task context."""

    def test_guardrails_work_in_task_decorator(self):
        """ExtractionGuardrails works inside @task function."""

    def test_guardrails_heartbeat_with_prefect_logger(self):
        """Heartbeat uses Prefect logger when available."""

    def test_guardrails_checkpoint_with_task_state(self):
        """Checkpoint integrates with Prefect task state."""

    def test_memory_error_triggers_task_retry(self):
        """MemoryError from guardrails triggers Prefect retry."""
```

### 3.3 Pre-commit Validation

Extend `scripts/check_extraction_patterns.py` to also check Prefect flow files.

### 3.4 Integration Test: `tests/integration/test_extraction_end_to_end.py`

Full pipeline tests with mock MLflow data for speed.

---

## 4. Implementation Checklist with Priorities

### Priority 0: Critical Blockers (Do First)

- [ ] **P0.1**: Add `ExtractionGuardrails` to `extraction_flow.py`
  - Import from `src.extraction.guardrails`
  - Initialize in flow with `total_items=len(pickle_files)`
  - Call `on_item_complete()` after each pickle

- [ ] **P0.2**: Add heartbeat logging to extraction tasks
  - Every pickle should log: `Processed 1/136 | ETA: 45 min | Mem: 0.4GB`
  - Use `ProgressTracker.heartbeat()` from guardrails

- [ ] **P0.3**: Add `CheckpointManager` for resume
  - Port from `streaming_duckdb_export.py` or create new
  - Check `is_completed()` before processing each pickle
  - Mark `completed` after successful processing

### Priority 1: Core Test Coverage

- [ ] **P1.1**: Create `tests/test_extraction/test_prefect_integration.py`
- [ ] **P1.2**: Create `tests/test_extraction/test_guardrails_prefect_bridge.py`
- [ ] **P1.3**: Add integration test for full pipeline

### Priority 2: Pre-commit Safety

- [ ] **P2.1**: Add pre-commit hook to verify guardrails usage
- [ ] **P2.2**: Extend `check_extraction_patterns.py` to check flows
- [ ] **P2.3**: Add test for pattern checker on Prefect flow

### Priority 3: Documentation and Cleanup

- [ ] **P3.1**: Update `docs/planning/robustify-mlruns-extraction.md`
- [ ] **P3.2**: Add docstrings to new test files
- [ ] **P3.3**: Deprecate standalone scripts (optional)

---

## 5. Success Criteria

### Must Have (For Plan Completion)

| Criterion | Verification |
|-----------|--------------|
| **Visible progress every 60 seconds** | Heartbeat log with pickle count, ETA, memory |
| **Resume from checkpoint** | Kill and restart → continues from last pickle |
| **Memory bounded** | Peak memory < 4GB during extraction |
| **Stall detection** | TimeoutError if no progress for 90s |
| **All tests pass** | `pytest tests/test_extraction/ -v` green |

### Metrics for Success

| Metric | Before | After |
|--------|--------|-------|
| Visible progress | 0 logs in 6 hours | Log every 60s |
| Crash recovery | Start from scratch | Resume from checkpoint |
| Wasted compute | 30+ hours | < 1 hour on crash |
| Test coverage | 0% for Prefect+guardrails | 90%+ |

---

## 6. Key Files for Implementation

| File | Purpose | Action |
|------|---------|--------|
| `src/orchestration/flows/extraction_flow.py` | Main Prefect flow | Add guardrails + checkpoint |
| `src/extraction/guardrails.py` | Guardrails module | No changes, just USE IT |
| `src/data_io/streaming_duckdb_export.py` | Has CheckpointManager | Port to extraction flow |
| `tests/test_extraction/test_prefect_integration.py` | NEW | Create with tests above |
| `scripts/check_extraction_patterns.py` | AST checker | Add Prefect flow files |

---

## 7. Timeline Estimate

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| P0: Critical fixes | 2-4 hours | Working extraction with visibility |
| P1: Core tests | 2-3 hours | Test coverage for new functionality |
| P2: Pre-commit | 1-2 hours | Safety guardrails enforced |
| P3: Documentation | 1 hour | Plan marked complete |
| **Total** | **6-10 hours** | Crash-resistant extraction pipeline |

---

## 8. Review Checklist

- [ ] Does the plan address the root cause?
- [ ] Is the architecture feasible?
- [ ] Are the tests comprehensive?
- [ ] Is the timeline realistic?
- [ ] Are dependencies clear?
