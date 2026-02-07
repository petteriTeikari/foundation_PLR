# CRITICAL-FAILURE-005: Extraction Script Stuck for 24 Hours Without Detection

## Incident Summary

**Date**: 2026-01-31 to 2026-02-01
**Duration**: 21+ hours of wasted compute
**Impact**: Blocked Phase 2 (per-subject signal extraction), wasted user time

## What Happened

1. `scripts/extract_decomposition_signals.py` was launched to extract per-subject preprocessed signals to DuckDB
2. Script collected ALL data in memory before any database writes (design flaw)
3. With 136 pickle files × 507 subjects × large numpy arrays = 13.4GB RAM usage
4. System swap thrashed: **18TB of disk writes** over 21 hours (detected via `/proc/<pid>/io`)
5. Script appeared "running" (99.4% CPU) but was stuck in memory pressure loop
6. NO monitoring, NO progress checks, NO timeouts detected the issue

## Root Cause Analysis

### 1. Script Design Flaw

```python
# BROKEN: Accumulates ALL rows in memory before insert
all_rows = []
for pickle_path in pickle_files:
    signals = extract_signals_from_pickle(pickle_path)
    for sig in signals:
        all_rows.append(...)  # Memory grows unbounded

# Only writes AFTER collecting everything
conn.executemany(..., all_rows)
```

### 2. Missing Guardrails

| Expected Guardrail | Status |
|--------------------|--------|
| Memory usage monitoring | MISSING |
| Progress heartbeat | MISSING |
| Timeout detection | MISSING |
| DB size growth check | MISSING |
| Test for memory efficiency | MISSING |

### 3. Oversight During Monitoring

Claude did not:
- Check `/proc/<pid>/io` for swap activity
- Verify DB file was growing (it wasn't)
- Question why DB was 3GB after 21 hours (should have been done in ~1 hour)
- Set up proper progress monitoring when launching background task

## The Fix Applied

```python
# FIXED: Stream inserts per-pickle, not batch at end
for pickle_path in pickle_files:
    signals = extract_signals_from_pickle(pickle_path)
    rows = [(config_id, ..., sig["signal"].tolist()) for sig in signals]
    conn.executemany(..., rows)  # Insert immediately
    del signals, rows
    gc.collect()  # Force memory cleanup
```

**Result**: Memory usage dropped from 13.4GB to ~400MB

## Lessons Learned

### L1: Fail Fast, Not Slow

Long-running scripts MUST have:
- **Heartbeat logging**: Every N seconds, log "Still running, processed X/Y"
- **Progress metrics**: Track rows/sec, estimated time remaining
- **Stall detection**: If no progress for 5 minutes, FAIL loudly

### L2: Memory-Bounded Design

Large data processing scripts MUST:
- Stream results, not batch
- Explicitly call `gc.collect()` after processing each chunk
- Have memory limit checks (`psutil.Process().memory_info().rss`)

### L3: Proactive Monitoring Setup

When launching background tasks, ALWAYS:
- Set up heartbeat monitoring
- Check DB/output file growth periodically
- Verify `/proc/<pid>/io` shows expected I/O patterns (not swap thrashing)

### L4: Never Trust "99% CPU"

High CPU usage ≠ progress. Check:
- Output file growth
- Read vs write bytes (18TB writes for 20GB input = BAD)
- Memory vs expected memory

## Required Changes

### Immediate

1. [x] Fix `extract_decomposition_signals.py` to use streaming inserts
2. [x] Add memory monitoring to all extraction scripts - **DONE: src/extraction/guardrails.py**
3. [x] Add progress heartbeat with ETA - **DONE: ProgressTracker class**
4. [x] Add stall detection (no progress for N minutes → FAIL) - **DONE: StallDetector class**

### Short-Term

1. [x] Create `tests/test_extraction/test_guardrails.py` - 14 tests for guardrails - **DONE**
2. [x] Add pre-commit hook to detect unbounded accumulation patterns - **DONE: scripts/check_extraction_patterns.py**
3. [x] Fix `extract_cd_diagram_data.py` (CRITICAL - 542K rows) - **DONE**

### Long-Term

1. [ ] Use Prefect for all extraction flows with built-in monitoring
2. [ ] Dashboard for extraction job health (memory, progress, ETA)

## Detection Checklist for Future Scripts

Before running ANY long-running extraction:

- [ ] Does the script stream results or batch?
- [ ] Is there a progress log every N items?
- [ ] Is there memory monitoring?
- [ ] Is there a timeout?
- [ ] Is the output file growing during execution?

## Related Files

- `scripts/extract_decomposition_signals.py` - FIXED with streaming inserts
- `scripts/extract_all_configs_to_duckdb.py` - AUDITED: accumulates ~300 rows (acceptable) + added memory monitoring
- `scripts/extract_cd_diagram_data.py` - FIXED: was accumulating 542K rows, now streaming
- `src/extraction/guardrails.py` - NEW: Unified guardrails module
- `scripts/check_extraction_patterns.py` - NEW: AST-based pattern checker
- `tests/test_extraction/test_guardrails.py` - NEW: 14 tests
- `tests/test_extraction/test_no_memory_accumulation.py` - NEW: 2 tests
