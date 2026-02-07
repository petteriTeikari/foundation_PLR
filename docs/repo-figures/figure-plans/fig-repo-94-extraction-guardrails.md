# fig-repo-94: Guardrails: Memory, Disk, and Stall Protection

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-94 |
| **Title** | Guardrails: Memory, Disk, and Stall Protection |
| **Complexity Level** | L4 |
| **Target Persona** | ML Engineer |
| **Location** | `src/extraction/README.md`, `docs/explanation/extraction-pipeline.md` |
| **Priority** | P3 (Medium) |

## Purpose

Document the three runtime guardrails that protect the extraction pipeline (MLflow to DuckDB) from resource exhaustion. Processing 410 MLflow runs with 1000 bootstrap iterations each is memory-intensive; these guardrails were created after CRITICAL-FAILURE-005 where an extraction script ran stuck in swap thrashing for 24 hours without detection.

## Key Message

Three independent monitors (memory, disk, stall) protect extraction from silent resource exhaustion. If any monitor triggers, the pipeline aborts with a checkpoint so work is not lost and can be resumed.

## Content Specification

### Panel 1: ExtractionGuardrails Architecture

```
+----------------------------------------------------------------------+
|                     ExtractionGuardrails                               |
|                                                                        |
|  Created after CRITICAL-FAILURE-005:                                   |
|  "A script ran for 24 hours stuck in swap thrashing"                   |
|                                                                        |
|  +-----------------------+  +-----------------------+                  |
|  |    MemoryMonitor      |  |     DiskMonitor       |                  |
|  |                       |  |                       |                  |
|  | check_memory()        |  | check_disk_space()    |                  |
|  |                       |  |                       |                  |
|  | Threshold:            |  | Threshold:            |                  |
|  |   max_memory_gb=4.0   |  |   min_disk_space_gb   |                  |
|  |   (configurable)      |  |   =5.0 (configurable) |                  |
|  |                       |  |                       |                  |
|  | Action on breach:     |  | Action on breach:     |                  |
|  |   gc.collect()        |  |   IOError with        |                  |
|  |   MemoryError with    |  |   diagnostic message  |                  |
|  |   diagnostic message  |  |                       |                  |
|  +-----------------------+  +-----------------------+                  |
|                                                                        |
|  +-----------------------+  +-----------------------+                  |
|  |    StallDetector      |  |   ProgressTracker     |                  |
|  |                       |  |                       |                  |
|  | mark_progress()       |  | update(n=1)           |                  |
|  | check()               |  | heartbeat()           |                  |
|  |                       |  | finish()              |                  |
|  | Timeout:              |  |                       |                  |
|  |   max_stall_seconds   |  | Heartbeat interval:   |                  |
|  |   =90 (configurable)  |  |   60 seconds          |                  |
|  |                       |  |                       |                  |
|  | Action on breach:     |  | Output:               |                  |
|  |   TimeoutError        |  |   [HEARTBEAT] 42/410  |                  |
|  |   "Script appears     |  |   (10.2%) Rate: 0.5/s |                  |
|  |    stuck!"            |  |   ETA: 12.3 min       |                  |
|  +-----------------------+  |   Mem: 2.1GB          |                  |
|                              +-----------------------+                  |
+----------------------------------------------------------------------+
```

### Panel 2: Lifecycle Flow

```
ExtractionGuardrails.__init__(config, total_items=410, output_path=DB)
  |
  v
_preflight_checks()
  +-- validate_mlflow_path() --> FileNotFoundError / TimeoutError if bad
  +-- check_disk_space()     --> IOError if < 5GB free
  +-- check_memory()         --> MemoryError if > 4GB used
  |
  v
[PREFLIGHT] All checks passed. Starting extraction.
  |
  v
for each MLflow run (410 total):
  |
  +-- process(run)              <-- actual extraction work
  |
  +-- guardrails.on_item_complete()
  |     +-- progress.update(1)       --> heartbeat if interval elapsed
  |     +-- stall_detector.mark_progress()  --> reset stall timer
  |     +-- gc.collect()             --> force garbage collection
  |     +-- check_all()             --> memory + disk + stall check
  |           (runs every heartbeat_seconds interval)
  |
  v (after all runs)
guardrails.finish()
  +-- progress.finish()
  +-- [COMPLETE] 410/410 items in 1234.5s | Final mem: 2.3GB
```

### Panel 3: ExtractionConfig (Configurable Parameters)

```
@dataclass
class ExtractionConfig:
    max_memory_gb: float = 4.0          # Per-process memory ceiling
    heartbeat_seconds: int = 60          # Progress log interval
    max_stall_seconds: int = 90          # No-progress timeout
    min_disk_space_gb: float = 5.0       # Minimum free disk space
    batch_size: int = 1000               # Items per batch insert
    path_timeout_seconds: float = 5.0    # MLflow path responsiveness
```

### Panel 4: CheckpointManager Integration

```
FULL PIPELINE                        RESUME FROM CHECKPOINT
+--------------------------+         +--------------------------+
| make reproduce           |         | make reproduce-from-     |
|                          |         |       checkpoint         |
| 1. make extract          |         |                          |
|    (Block 1: MLflow-->DB)|         | Skip extraction entirely |
|    410 runs, ~45 min     |         | Use existing DuckDB      |
|                          |         |                          |
| 2. make analyze          |         | 1. make analyze          |
|    (Block 2: DB-->Figs)  |         |    (Block 2: DB-->Figs)  |
+--------------------------+         +--------------------------+

If extraction aborts mid-way (memory/disk/stall):
  - DuckDB has all completed runs (streaming inserts)
  - Re-run make extract: processes only missing runs
  - No work is lost
```

### Panel 5: Why This Exists (Context)

```
Processing scale:
  410 MLflow runs x 1000 bootstrap samples = 410,000 metric computations
  Each run: load model, load predictions, resample, compute 7 metrics
  Memory: pickle deserialization can spike to several GB

Without guardrails (CRITICAL-FAILURE-005):
  - Script stuck in swap thrashing for 24 hours
  - No heartbeat output, no progress indication
  - User discovered it by accident the next day
  - 21+ hours of wasted compute time

With guardrails:
  - Stall detected within 90 seconds
  - Memory spike caught before swap thrashing
  - Progress logged every 60 seconds (visible in terminal/CI logs)
  - Disk space checked before DB grows unboundedly
```

## Spatial Anchors

```yaml
layout_flow: "Top-down: architecture overview, lifecycle flow, config, checkpoint, context"
spatial_anchors:
  architecture:
    x: 0.5
    y: 0.2
    content: "Four components: MemoryMonitor, DiskMonitor, StallDetector, ProgressTracker"
  lifecycle:
    x: 0.5
    y: 0.45
    content: "Preflight -> per-item checks -> finish"
  config:
    x: 0.25
    y: 0.65
    content: "ExtractionConfig dataclass with defaults"
  checkpoint:
    x: 0.75
    y: 0.65
    content: "Checkpoint and resume flow"
  context:
    x: 0.5
    y: 0.88
    content: "Scale of 410 runs x 1000 bootstraps"
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `src/extraction/guardrails.py` | `ExtractionConfig` dataclass with all defaults |
| `Makefile` | `reproduce`, `reproduce-from-checkpoint`, `extract` targets |

## Code Paths

| Module | Role |
|--------|------|
| `src/extraction/guardrails.py` | `ExtractionGuardrails`, `ExtractionConfig`, `ProgressTracker`, `StallDetector` |
| `src/extraction/guardrails.py` | `check_memory()`, `check_disk_space()`, `validate_mlflow_path()` |
| `src/data_io/streaming_duckdb_export.py` | `StreamingDuckDBExporter` (streaming inserts, enables checkpoint recovery) |
| `src/orchestration/flows/extraction_flow.py` | Block 1 entry point that uses guardrails |

## Extension Guide

To add a new guardrail monitor:
1. Create a new function or class in `src/extraction/guardrails.py`
2. Add configurable thresholds to `ExtractionConfig` dataclass
3. Call the new check in `ExtractionGuardrails.check_all()`
4. Add preflight validation in `_preflight_checks()`
5. Raise a descriptive error with diagnostic message on breach
6. Add unit test in `tests/unit/test_guardrails.py`

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-94",
    "title": "Guardrails: Memory, Disk, and Stall Protection"
  },
  "content_architecture": {
    "primary_message": "Three independent monitors (memory, disk, stall) protect extraction from silent resource exhaustion, with checkpoint recovery if any triggers.",
    "layout_flow": "Top-down: architecture, lifecycle, config, checkpoint, context",
    "spatial_anchors": {
      "architecture": {"x": 0.5, "y": 0.2},
      "lifecycle": {"x": 0.5, "y": 0.45},
      "config": {"x": 0.25, "y": 0.65},
      "checkpoint": {"x": 0.75, "y": 0.65},
      "context": {"x": 0.5, "y": 0.88}
    },
    "key_structures": [
      {
        "name": "MemoryMonitor",
        "role": "abnormal_warning",
        "is_highlighted": true,
        "labels": ["max_memory_gb=4.0", "gc.collect()", "MemoryError"]
      },
      {
        "name": "DiskMonitor",
        "role": "abnormal_warning",
        "is_highlighted": true,
        "labels": ["min_disk_space_gb=5.0", "IOError"]
      },
      {
        "name": "StallDetector",
        "role": "abnormal_warning",
        "is_highlighted": true,
        "labels": ["max_stall_seconds=90", "TimeoutError"]
      },
      {
        "name": "ProgressTracker",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["heartbeat every 60s", "rate + ETA + memory"]
      },
      {
        "name": "CheckpointManager",
        "role": "healthy_normal",
        "is_highlighted": true,
        "labels": ["make reproduce-from-checkpoint", "No work lost"]
      }
    ],
    "callout_boxes": [
      {"heading": "ORIGIN STORY", "body_text": "Created after CRITICAL-FAILURE-005: extraction script stuck in swap thrashing for 24 hours without detection."},
      {"heading": "SCALE", "body_text": "410 MLflow runs x 1000 bootstrap samples = 410,000 metric computations."}
    ]
  }
}
```

## Alt Text

Architecture diagram of ExtractionGuardrails showing four components (memory monitor, disk monitor, stall detector, progress tracker) with their thresholds, actions, and checkpoint recovery flow.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
