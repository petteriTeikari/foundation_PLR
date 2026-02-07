# fig-repo-57: DuckDB Schema Diagram

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-57 |
| **Title** | DuckDB Archival Database Schema |
| **Complexity Level** | L2 |
| **Target Persona** | ML Engineer / Research Scientist |
| **Location** | `src/data_io/README.md`, `docs/user-guide/pipeline-overview.md` |
| **Priority** | P1 (Critical) |

## Purpose

Show the complete schema of the DuckDB archival database (`data/public/foundation_plr_results.db`) — the permanent, portable artifact that replaces ephemeral MLflow runs. Developers need to understand what tables exist, how they relate, and what each column means for downstream visualization.

## Key Concept

The DuckDB is the SINGLE archival artifact (~50 MB). MLflow runs (~200 GB) are ephemeral. All figures, statistics, and LaTeX macros are generated from DuckDB alone.

## Content Specification

### Panel 1: Table Overview (Entity-Relationship)

```
┌─────────────────────────┐
│    essential_metrics     │  ← Main results table (406 rows)
│─────────────────────────│
│ config_id (PK)          │
│ outlier_method           │
│ imputation_method        │
│ classifier               │
│ featurization            │
│ auroc, auroc_ci_lower/upper │
│ calibration_slope/intercept │
│ o_e_ratio                │
│ brier, scaled_brier      │
│ net_benefit_5/10/15/20pct│
└──────────┬──────────────┘
           │ config_id
    ┌──────┴──────┬──────────────┬──────────────┬──────────────┐
    ▼             ▼              ▼              ▼              ▼
┌────────┐ ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌──────────────┐
│dca_    │ │calibra-  │ │retention_ │ │cohort_   │ │distribution_ │
│curves  │ │tion_     │ │metrics    │ │metrics   │ │stats         │
│        │ │curves    │ │           │ │          │ │              │
│threshold│ │predicted │ │retention_ │ │cohort_   │ │auroc         │
│net_ben │ │observed  │ │rate       │ │fraction  │ │median_cases  │
│treat_  │ │ci_lower  │ │metric_    │ │metric_   │ │median_ctrls  │
│all_nb  │ │ci_upper  │ │name/value │ │name/value│ │n_cases/ctrls │
└────────┘ └──────────┘ └───────────┘ └──────────┘ └──────────────┘
```

### Panel 2: Data Volume

| Table | Rows | Columns | Source |
|-------|------|---------|--------|
| essential_metrics | 406 | ~20 | MLflow per-fold metrics |
| predictions | 25,452 | 5 | Per-subject probabilities |
| dca_curves | ~8,000 | 5 | Threshold sweep 5%-40% |
| calibration_curves | ~4,000 | 5 | LOESS smoothed |
| retention_metrics | ~64,000 | 4 | 20 rates x 8 metrics x 406 configs |
| cohort_metrics | ~32,000 | 4 | 10 fractions x 8 metrics x 406 configs |
| distribution_stats | ~400 | 8 | Per-config summary stats |

### Panel 3: Data Provenance Flow

```
MLflow Runs (200 GB)
  └→ StreamingDuckDBExporter
       └→ DuckDB (50 MB)
            └→ src/viz/ (READ ONLY)
                 └→ Figures + JSON + LaTeX
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `src/data_io/streaming_duckdb_export.py` | Schema definitions, extraction logic |
| `configs/CLS_EVALUATION.yaml` | Bootstrap params, threshold ranges |

## Code Paths

| Module | Role |
|--------|------|
| `src/data_io/streaming_duckdb_export.py` | Defines all table schemas, writes DuckDB |
| `src/extraction/extract_all_configs_to_duckdb.py` | Legacy extraction script |
| `src/orchestration/flows/extraction_flow.py` | Prefect flow for extraction |
| `src/viz/*.py` | Read-only consumers of DuckDB tables |

## Extension Guide

To add a new DuckDB table:
1. Add schema to `StreamingDuckDBExporter.SCHEMAS` dict
2. Add extraction method `_extract_<table_name>()`
3. Add test in `tests/test_data_quality/test_extraction_correctness.py`
4. Add guardrail test in `tests/test_guardrails/test_computation_decoupling.py`
5. Re-run extraction: `make extract`

Note: Performance comparisons are in the manuscript, not this repository.
