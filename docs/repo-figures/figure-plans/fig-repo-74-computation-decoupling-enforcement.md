# fig-repo-74: The Import Ban: What src/viz/ Cannot Touch

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-74 |
| **Title** | The Import Ban: What src/viz/ Cannot Touch |
| **Complexity Level** | L3 |
| **Target Persona** | ML Engineer / Research Scientist |
| **Location** | `src/viz/README.md`, `ARCHITECTURE.md` |
| **Priority** | P1 (Critical) |

## Purpose

Show the strict boundary between Block 1 (Extraction: MLflow to DuckDB) and Block 2 (Visualization: DuckDB to Figures). Developers must understand which imports are BANNED in `src/viz/` and why, what enforcement mechanisms exist, and what the correct pattern looks like.

## Key Message

Visualization code (src/viz/) is BANNED from importing computation modules (sklearn.metrics, scipy.stats, src/stats/*). All metrics must be pre-computed in extraction and read from DuckDB. Two enforcement mechanisms catch violations at commit-time and test-time.

## Content Specification

### Panel 1: Two-Block Architecture

```
┌──────────────────────────────────┐     ┌──────────────────────────────────┐
│       BLOCK 1: EXTRACTION         │     │       BLOCK 2: VISUALIZATION      │
│       (ALL computation here)      │     │       (READ-ONLY from DuckDB)     │
│                                    │     │                                    │
│  ALLOWED imports:                  │     │  ALLOWED imports:                  │
│  ┌──────────────────────────────┐ │     │  ┌──────────────────────────────┐ │
│  │ from sklearn.metrics import   │ │     │  │ import duckdb               │ │
│  │   roc_auc_score,              │ │     │  │ import pandas as pd         │ │
│  │   brier_score_loss            │ │     │  │ import matplotlib.pyplot    │ │
│  │                               │ │     │  │ from plot_config import     │ │
│  │ from scipy.stats import       │ │     │  │   setup_style, COLORS,     │ │
│  │   bootstrap                   │ │     │  │   save_figure              │ │
│  │                               │ │     │  └──────────────────────────────┘ │
│  │ from src.stats import         │ │     │                                    │
│  │   calibration_extended,       │ │     │  BANNED imports:                   │
│  │   clinical_utility,           │ │     │  ┌──────────────────────────────┐ │
│  │   scaled_brier                │ │     │  │ X sklearn.metrics            │ │
│  └──────────────────────────────┘ │     │  │ X sklearn.linear_model       │ │
│                                    │     │  │ X sklearn.calibration        │ │
│  Writes to DuckDB:                │     │  │ X scipy.stats                │ │
│  ┌──────────────────────────────┐ │     │  │ X src.stats.calibration_*    │ │
│  │ essential_metrics             │ │     │  │ X src.stats.clinical_utility │ │
│  │ calibration_curves            │ │     │  │ X src.stats.scaled_brier    │ │
│  │ dca_curves                    │──────│──│                               │ │
│  │ predictions                   │ │     │  │ Also banned functions:       │ │
│  │ retention_metrics             │ │     │  │ X roc_auc_score             │ │
│  │ distribution_stats            │ │     │  │ X brier_score_loss          │ │
│  │ cohort_metrics                │ │     │  │ X calibration_curve         │ │
│  └──────────────────────────────┘ │     │  │ X LogisticRegression        │ │
│                                    │     │  │ X net_benefit               │ │
└──────────────────────────────────┘     │  └──────────────────────────────┘ │
                                          │                                    │
                                          │  Correct pattern:                  │
                                          │  ┌──────────────────────────────┐ │
                                          │  │ conn = duckdb.connect(db)    │ │
                                          │  │ df = conn.execute(           │ │
                                          │  │   "SELECT auroc,             │ │
                                          │  │    calibration_slope         │ │
                                          │  │    FROM essential_metrics"   │ │
                                          │  │ ).fetchdf()                  │ │
                                          │  └──────────────────────────────┘ │
                                          └──────────────────────────────────┘
```

### Panel 2: Enforcement Mechanisms

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ENFORCEMENT: TWO INDEPENDENT CHECKS                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. PRE-COMMIT HOOK (commit-time)                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  .pre-commit-config.yaml → id: computation-decoupling             │  │
│  │  entry: scripts/check_computation_decoupling.py                   │  │
│  │  files: src/viz/                                                  │  │
│  │                                                                    │  │
│  │  HOW IT WORKS:                                                    │  │
│  │  1. Parses each .py file in src/viz/ with Python AST              │  │
│  │  2. Walks AST to find Import and ImportFrom nodes                 │  │
│  │  3. Checks against BANNED_SKLEARN_IMPORTS (10 functions)          │  │
│  │  4. Checks against BANNED_SKLEARN_MODULES (3 modules)            │  │
│  │  5. Checks against BANNED_STATS_IMPORTS (3 functions)            │  │
│  │  6. Skips ALLOWED_COMPUTATION_FILES: metric_registry.py,         │  │
│  │     metrics_utils.py, __init__.py                                │  │
│  │  7. Exit code 1 → commit BLOCKED                                 │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  2. PYTEST SUITE (test-time)                                            │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  tests/test_no_hardcoding/test_computation_decoupling.py          │  │
│  │  tests/test_guardrails/test_computation_decoupling.py             │  │
│  │                                                                    │  │
│  │  HOW IT WORKS:                                                    │  │
│  │  1. Scans all .py files in src/viz/                               │  │
│  │  2. Checks for banned import patterns                             │  │
│  │  3. Verifies no metric computation functions are called           │  │
│  │  4. Test FAILURE → CI pipeline rejects PR                        │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  EXCEPTION: metric_registry.py                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  src/viz/metric_registry.py is DUAL-USE:                          │  │
│  │  ├── Viz code: reads .display_name, .higher_is_better (OK)       │  │
│  │  └── Extraction code: calls .compute_fn (which imports sklearn)   │  │
│  │  The file is in ALLOWED_COMPUTATION_FILES whitelist               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 3: Historical Context

```
┌─────────────────────────────────────────────────────────────────────────┐
│  WHY THIS RULE EXISTS: CRITICAL-FAILURE-003                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  BEFORE (violation):                                                     │
│  src/viz/calibration_plot.py imported sklearn.metrics.roc_auc_score     │
│  src/viz/dca_plot.py imported src.stats.clinical_utility.net_benefit    │
│  → Figures could silently recompute metrics with different parameters   │
│  → No single source of truth for metric values                          │
│  → Extraction results != Visualization results (data drift)             │
│                                                                          │
│  AFTER (enforced):                                                       │
│  Block 1 computes ALL metrics → stores in DuckDB                        │
│  Block 2 reads ONLY from DuckDB → plots pre-computed values            │
│  → One computation, one truth, many figures                              │
│                                                                          │
│  Full history: .claude/docs/meta-learnings/                             │
│  CRITICAL-FAILURE-003-computation-decoupling-violation.md               │
└─────────────────────────────────────────────────────────────────────────┘
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `.pre-commit-config.yaml` | Hook definition: `computation-decoupling` |
| `configs/data_isolation.yaml` | Data isolation configuration |

## Code Paths

| Module | Role |
|--------|------|
| `scripts/check_computation_decoupling.py` | Pre-commit hook: AST-based banned import detection |
| `tests/test_no_hardcoding/test_computation_decoupling.py` | pytest: Scans src/viz/ for banned imports |
| `tests/test_guardrails/test_computation_decoupling.py` | pytest: Additional guardrail checks |
| `src/viz/metric_registry.py` | Dual-use exception: metadata for viz, compute_fn for extraction |
| `src/viz/plot_config.py` | Visualization utilities (COLORS, save_figure, setup_style) |
| `src/data_io/streaming_duckdb_export.py` | Block 1: Writes metrics to DuckDB |

## Extension Guide

To add a new pre-computed metric for visualization:
1. Compute the metric in extraction code (`src/stats/` or `scripts/extract_*.py`)
2. Add a column to the appropriate DuckDB table (e.g., `essential_metrics`)
3. Register in `src/viz/metric_registry.py` (metadata only -- display name, format)
4. Read from DuckDB in your viz module: `conn.execute("SELECT new_metric FROM ...")`
5. NEVER import the computation function in `src/viz/` -- the pre-commit hook will block it

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "repo-74",
    "title": "The Import Ban: What src/viz/ Cannot Touch"
  },
  "content_architecture": {
    "primary_message": "Visualization code is BANNED from importing computation modules. All metrics are pre-computed in extraction and read from DuckDB.",
    "layout_flow": "Two-column layout: Block 1 (Extraction) on left, Block 2 (Visualization) on right, enforcement below",
    "spatial_anchors": {
      "block_1": {"x": 0.25, "y": 0.3},
      "block_2": {"x": 0.75, "y": 0.3},
      "duckdb_bridge": {"x": 0.5, "y": 0.3},
      "enforcement": {"x": 0.5, "y": 0.75}
    },
    "key_structures": [
      {
        "name": "Block 1: Extraction",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["sklearn allowed", "DuckDB WRITE"]
      },
      {
        "name": "Block 2: Visualization",
        "role": "secondary_pathway",
        "is_highlighted": true,
        "labels": ["sklearn BANNED", "DuckDB READ"]
      },
      {
        "name": "Pre-commit hook",
        "role": "abnormal_warning",
        "is_highlighted": true,
        "labels": ["AST-based scan"]
      },
      {
        "name": "pytest suite",
        "role": "abnormal_warning",
        "is_highlighted": false,
        "labels": ["CI enforcement"]
      }
    ],
    "callout_boxes": [
      {"heading": "CRITICAL-FAILURE-003", "body_text": "This rule exists because viz code once recomputed metrics with different parameters, causing data drift."}
    ]
  }
}
```

## Alt Text

Two-column diagram showing extraction (allowed imports) and visualization (banned imports) blocks separated by DuckDB, with enforcement mechanisms below.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
