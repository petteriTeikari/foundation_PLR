# Extraction

Production guardrails for MLflow extraction scripts.

## Overview

Created after CRITICAL-FAILURE-005 (a script ran 24 hours stuck in swap thrashing without detection). This module provides memory monitoring, heartbeat/stall detection, disk space checks, and progress tracking to prevent runaway extraction jobs.

## Modules

| Module | Purpose |
|--------|---------|
| `guardrails.py` | `ExtractionGuardrails`, `ExtractionConfig`, `ProgressTracker`, `StallDetector`, memory/disk checks |

## See Also

- `src/orchestration/flows/extraction_flow.py` -- Uses these guardrails during MLflow-to-DuckDB extraction
- `src/data_io/` -- Data import/export utilities
