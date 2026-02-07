# fig-repo-79: DuckDB Schema: Tables That Power Every Figure

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-79 |
| **Title** | DuckDB Schema: Tables That Power Every Figure |
| **Complexity Level** | L3 |
| **Target Persona** | Research Scientist / ML Engineer |
| **Location** | `data/README.md`, `ARCHITECTURE.md`, `src/viz/README.md` |
| **Priority** | P1 (Critical) |

## Purpose

Show the entity-relationship structure of `foundation_plr_results.db`, the central database that powers all figures. Developers and researchers need to understand which tables exist, how they relate via composite keys, what columns each contains, and which visualization module reads from which table.

## Key Message

The foundation_plr_results.db database contains tables centered around essential_metrics (one row per configuration). Auxiliary tables (calibration_curves, dca_curves, predictions, retention_metrics, distribution_stats) link via composite key (outlier_method, imputation_method, classifier).

## Content Specification

### Panel 1: Entity-Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│              foundation_plr_results.db  (data/public/)                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  essential_metrics  (316 rows, 1 per config)                      │   │
│  │  ═══════════════════════════════════════════                      │   │
│  │  PK: (outlier_method, imputation_method, classifier)              │   │
│  │                                                                    │   │
│  │  Discrimination:                                                   │   │
│  │    auroc              FLOAT    [0.0 - 1.0]                        │   │
│  │    auroc_ci_lower     FLOAT    bootstrap 2.5th percentile         │   │
│  │    auroc_ci_upper     FLOAT    bootstrap 97.5th percentile        │   │
│  │                                                                    │   │
│  │  Calibration:                                                      │   │
│  │    calibration_slope      FLOAT    ideal = 1.0                    │   │
│  │    calibration_intercept  FLOAT    ideal = 0.0                    │   │
│  │    o_e_ratio              FLOAT    observed/expected, ideal = 1.0 │   │
│  │                                                                    │   │
│  │  Overall:                                                          │   │
│  │    brier                  FLOAT    [0.0 - 0.25], lower = better   │   │
│  │    scaled_brier           FLOAT    IPA, 0=null, 1=perfect         │   │
│  │                                                                    │   │
│  │  Clinical Utility:                                                 │   │
│  │    net_benefit_5pct       FLOAT    NB at 5% threshold             │   │
│  │    net_benefit_10pct      FLOAT    NB at 10% threshold            │   │
│  │    net_benefit_15pct      FLOAT    NB at 15% threshold            │   │
│  │    net_benefit_20pct      FLOAT    NB at 20% threshold            │   │
│  └────────┬─────────────────────────────────────────────────────────┘   │
│            │                                                              │
│       1:N  │  (linked via outlier_method, imputation_method, classifier) │
│            │                                                              │
│  ┌─────────┴──────────────────────────────────────────────────────────┐ │
│  │                                                                      │ │
│  │  ┌──────────────────────────┐  ┌──────────────────────────────┐   │ │
│  │  │  predictions              │  │  calibration_curves           │   │ │
│  │  │  ────────────             │  │  ──────────────────           │   │ │
│  │  │  FK → essential_metrics   │  │  FK → essential_metrics      │   │ │
│  │  │  subject_id    VARCHAR    │  │  bin_midpoint   FLOAT        │   │ │
│  │  │  y_true        INT (0/1) │  │  observed_freq  FLOAT        │   │ │
│  │  │  y_prob        FLOAT     │  │  predicted_freq FLOAT        │   │ │
│  │  │  (316 x 208 rows)       │  │                               │   │ │
│  │  └──────────────────────────┘  └──────────────────────────────┘   │ │
│  │                                                                      │ │
│  │  ┌──────────────────────────┐  ┌──────────────────────────────┐   │ │
│  │  │  dca_curves               │  │  retention_metrics            │   │ │
│  │  │  ──────────               │  │  ──────────────────           │   │ │
│  │  │  FK → essential_metrics   │  │  FK → essential_metrics      │   │ │
│  │  │  threshold     FLOAT     │  │  threshold       FLOAT       │   │ │
│  │  │  net_benefit   FLOAT     │  │  metric_value    FLOAT       │   │ │
│  │  │  treat_all     FLOAT     │  │                               │   │ │
│  │  │  treat_none    FLOAT     │  │                               │   │ │
│  │  └──────────────────────────┘  └──────────────────────────────┘   │ │
│  │                                                                      │ │
│  │  ┌──────────────────────────┐  ┌──────────────────────────────┐   │ │
│  │  │  distribution_stats       │  │  cohort_metrics               │   │ │
│  │  │  ───────────────────      │  │  ──────────────               │   │ │
│  │  │  FK → essential_metrics   │  │  FK → essential_metrics      │   │ │
│  │  │  class_label   VARCHAR   │  │  cohort comparison data      │   │ │
│  │  │  mean          FLOAT     │  │                               │   │ │
│  │  │  std           FLOAT     │  │                               │   │ │
│  │  │  quantiles     FLOAT[]   │  │                               │   │ │
│  │  └──────────────────────────┘  └──────────────────────────────┘   │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 2: Table-to-Visualization Mapping

```
┌─────────────────────────────────────────────────────────────────────────┐
│  WHICH VIZ MODULE READS WHICH TABLE                                      │
├──────────────────────────┬──────────────────────────────────────────────┤
│  DuckDB Table            │  Visualization Consumer                      │
├──────────────────────────┼──────────────────────────────────────────────┤
│  essential_metrics       │  src/viz/metric_registry.py (metadata)      │
│                          │  src/viz/fig_decomposition_grid.py           │
│                          │  src/viz/metric_vs_cohort.py                │
│                          │  (most viz modules)                          │
├──────────────────────────┼──────────────────────────────────────────────┤
│  predictions             │  src/viz/prob_distribution.py                │
│                          │  src/viz/uncertainty_scatter.py              │
├──────────────────────────┼──────────────────────────────────────────────┤
│  calibration_curves      │  src/viz/calibration_plot.py                │
├──────────────────────────┼──────────────────────────────────────────────┤
│  dca_curves              │  src/viz/dca_plot.py                        │
├──────────────────────────┼──────────────────────────────────────────────┤
│  retention_metrics       │  src/viz/retained_metric.py                 │
├──────────────────────────┼──────────────────────────────────────────────┤
│  distribution_stats      │  src/viz/prob_distribution.py                │
├──────────────────────────┼──────────────────────────────────────────────┤
│  cohort_metrics          │  src/viz/metric_vs_cohort.py                │
└──────────────────────────┴──────────────────────────────────────────────┘
```

### Panel 3: Companion Database

```
┌─────────────────────────────────────────────────────────────────────────┐
│  COMPANION: cd_diagram_data.duckdb  (data/public/)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Used by: R scripts for Critical Difference (CD) diagrams              │
│  Contains: Rank data for Friedman/Nemenyi post-hoc tests              │
│  Consumer: src/r/figures/cd_diagram.R                                  │
│                                                                          │
│  Note: Separate from foundation_plr_results.db because CD diagrams    │
│  require a different data structure (ranks, not raw metrics)           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 4: Key Counts

```
┌─────────────────────────────────────────────────────────────────────────┐
│  KEY DATABASE COUNTS                                                     │
├──────────────────────────┬──────────────────────────────────────────────┤
│  essential_metrics rows  │  316 (available configurations)              │
│  predictions rows        │  316 x 208 = 65,728 (per-subject)          │
│  Unique outlier methods  │  11 (from registry)                         │
│  Unique imputation       │  8 (from registry)                          │
│  Unique classifiers      │  5 (from registry)                          │
│  Full grid               │  11 x 8 x 5 = 440 (theoretical max)       │
│  Available configs       │  316 (not all combos ran)                   │
│  Subjects per config     │  208 (152 control + 56 glaucoma)           │
│  Bootstrap iterations    │  1000 per config                            │
└──────────────────────────┴──────────────────────────────────────────────┘
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `configs/CLS_EVALUATION.yaml` | Bootstrap parameters: n_iterations=1000, alpha_CI=0.95 |
| `configs/CLS_EVALUATION.yaml` | glaucoma_params.prevalence=0.0354 (for net benefit) |

## Code Paths

| Module | Role |
|--------|------|
| `src/data_io/streaming_duckdb_export.py` | Creates and populates DuckDB tables during extraction |
| `scripts/extract_all_configs_to_duckdb.py` | Extraction script: MLflow runs to essential_metrics |
| `scripts/extract_curve_data_to_duckdb.py` | Extraction script: calibration/DCA curves to DuckDB |
| `src/viz/calibration_plot.py` | Reads calibration_curves table |
| `src/viz/dca_plot.py` | Reads dca_curves table |
| `src/viz/prob_distribution.py` | Reads predictions and distribution_stats tables |
| `src/viz/retained_metric.py` | Reads retention_metrics table |
| `src/viz/metric_vs_cohort.py` | Reads cohort_metrics table |
| `src/viz/fig_decomposition_grid.py` | Reads essential_metrics table |
| `src/viz/uncertainty_scatter.py` | Reads predictions table |

## Extension Guide

To add a new DuckDB table for a new visualization:
1. Define the schema in `src/data_io/streaming_duckdb_export.py`
2. Populate during extraction in `scripts/extract_*.py`
3. Read from the table in your `src/viz/` module (SELECT only, never compute)
4. Add the table to this figure plan for documentation
5. Add integration tests verifying the table exists and has expected columns

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "repo-79",
    "title": "DuckDB Schema: Tables That Power Every Figure"
  },
  "content_architecture": {
    "primary_message": "foundation_plr_results.db contains tables centered around essential_metrics (316 configs). Auxiliary tables link via composite key (outlier, imputation, classifier).",
    "layout_flow": "Central essential_metrics table at top, 1:N relationships fan out to auxiliary tables below, table-to-viz mapping on the right",
    "spatial_anchors": {
      "essential_metrics": {"x": 0.5, "y": 0.2},
      "predictions": {"x": 0.2, "y": 0.5},
      "calibration_curves": {"x": 0.5, "y": 0.5},
      "dca_curves": {"x": 0.8, "y": 0.5},
      "other_tables": {"x": 0.5, "y": 0.7},
      "viz_mapping": {"x": 0.5, "y": 0.9}
    },
    "key_structures": [
      {
        "name": "essential_metrics",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["316 rows", "PK: outlier+imputation+classifier"]
      },
      {
        "name": "predictions",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["65K rows", "subject_id, y_true, y_prob"]
      },
      {
        "name": "calibration_curves",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["bin_midpoint, observed, predicted"]
      },
      {
        "name": "dca_curves",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["threshold, net_benefit, treat_all"]
      },
      {
        "name": "retention_metrics",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["threshold, metric_value"]
      },
      {
        "name": "distribution_stats",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["class_label, mean, std"]
      }
    ],
    "callout_boxes": [
      {"heading": "READ-ONLY", "body_text": "All src/viz/ modules SELECT from these tables. No computation happens in visualization code."}
    ]
  }
}
```

## Alt Text

Entity-relationship diagram of DuckDB database with essential_metrics as central table linked to predictions, calibration curves, DCA curves, and other auxiliary tables.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
