# fig-repo-89: R Figure Pipeline: renv -> rocker -> ggplot2

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-89 |
| **Title** | R Figure Pipeline: renv -> rocker -> ggplot2 |
| **Complexity Level** | L3 |
| **Target Persona** | Research Scientist / Biostatistician |
| **Location** | `src/r/README.md`, `docs/explanation/r-figures.md` |
| **Priority** | P2 (High) |

## Purpose

Show how the R figure generation system works end-to-end: from pinned package versions (renv.lock) through Docker-based execution (rocker) to final ggplot2 figures. The R system runs in parallel with the Python matplotlib system and shares the same data sources (DuckDB) and color definitions (colors.yaml). Enforcement via pre-commit hooks prevents common mistakes.

## Key Message

R figures use a reproducible pipeline: renv.lock pins exact package versions, rocker Docker images provide the runtime, and `src/r/figure_system/` provides shared helper functions (theme, colors, save) that enforce consistency across 28 R figure scripts.

## Content Specification

### Panel 1: End-to-End Pipeline

```
PACKAGE MANAGEMENT                     SHARED FIGURE SYSTEM
═══════════════════                    ═══════════════════════
renv.lock (pinned versions)            src/r/figure_system/
├── ggplot2 3.5.x                      ├── common.R
├── pminternal 0.2.x                   │   └── Project-wide constants
├── dcurves 0.4.x                      ├── config_loader.R
├── pROC 1.18.x                        │   └── Load YAML configs
├── data.table 1.15.x                  ├── load_style.R
├── ggrepel                            │   └── theme_foundation_plr()
├── patchwork                          ├── category_loader.R
└── cowplot                            │   └── Load color definitions from YAML
                                       ├── save_figure.R
                                       │   └── save_publication_figure()
                                       ├── figure_factory.R
                                       │   └── Compose multi-panel figures
                                       ├── cd_diagram.R
                                       │   └── Critical difference diagrams
                                       └── compose_figures.R
                                           └── Panel arrangement utilities
         │                                          │
         ▼                                          ▼
DOCKER EXECUTION                       FIGURE SCRIPTS (28 scripts)
═══════════════════                    ═══════════════════════════
Dockerfile.r                           src/r/figures/
├── FROM rocker/tidyverse:4.5          ├── fig_calibration_stratos.R
├── COPY renv.lock                     ├── fig_dca_stratos.R
├── RUN renv::restore()                ├── fig_prob_dist_by_outcome.R
└── ~1GB image                         ├── fig02_forest_outlier.R
                                       ├── fig03_forest_imputation.R
Run via:                               ├── fig04_variance_decomposition.R
  make r-docker-run                    ├── fig05_shap_beeswarm.R
  make r-docker-test                   ├── fig06_specification_curve.R
  make r-docker-shell                  ├── fig07_heatmap_preprocessing.R
                                       ├── cd_preprocessing.R
Or locally:                            ├── fig_M3_factorial_matrix.R
  make r-figures-all                   ├── fig_R7_featurization_comparison.R
  make r-figures-sprint1               ├── fig_R8_fm_dashboard.R
                                       └── ... (28 total)
```

### Panel 2: Data Handoff (Python -> R)

```
PYTHON (Block 2 extraction)              R (figure generation)
═══════════════════════════              ═══════════════════════

DuckDB (foundation_plr_results.db)       outputs/r_data/*.csv
        │                                       │
        ▼                                       ▼
scripts/export_data_for_r.py             src/r/figures/*.R
scripts/export_predictions_for_r.py      ├── df <- fread("outputs/r_data/
scripts/export_roc_rc_data.py            │            essential_metrics.csv")
scripts/export_shap_for_r.py             ├── source("src/r/figure_system/
scripts/export_subject_traces_for_r.py   │            load_style.R")
scripts/export_selective_classification  │
        _data.py                         ├── theme_foundation_plr()
        │                                ├── load_color_definitions()
        ▼                                └── save_publication_figure(p, "name")
outputs/r_data/                                  │
├── essential_metrics.csv                        ▼
├── predictions.csv                      figures/generated/ggplot2/
├── roc_curves.csv                       ├── fig_calibration_stratos.pdf
├── shap_feature_importance.json         ├── fig_dca_stratos.pdf
├── vif_analysis.json                    ├── fig02_forest_outlier.pdf
└── ...                                  └── ... (28 outputs)
```

### Panel 3: Enforcement Mechanisms

```
PRE-COMMIT HOOK: r-hardcoding-check
════════════════════════════════════
Script: scripts/check_r_hardcoding.py
Scans: All .R files in src/r/

  BANNED PATTERNS:
  ├── "#[0-9a-fA-F]{6}"          No hex color literals
  │   Use: load_color_definitions() from colors.yaml
  │
  ├── ggsave(                     No direct ggsave()
  │   Use: save_publication_figure()
  │
  └── theme_minimal/classic/bw    No built-in themes
      Use: theme_foundation_plr()

DOCKER ISOLATION:
  Dockerfile.r provides identical R environment
  regardless of developer's local R installation.

TEST SUITE:
  tests/test_r_figures/           R figure validation
  tests/test_r_guardrails/        R code quality checks
  tests/test_r_environment.py     R package availability
```

### Panel 4: Makefile Sprint Structure

```
make r-figures-all
├── Sprint 1 (STRATOS + Main Results): 7 figures
│   ├── fig_calibration_stratos.pdf
│   ├── fig_dca_stratos.pdf
│   ├── fig_prob_dist_by_outcome.pdf
│   ├── fig02_forest_outlier.pdf
│   ├── fig03_forest_imputation.pdf
│   ├── fig06_specification_curve.pdf
│   └── cd_preprocessing.pdf
│
├── Sprint 2 (Enhanced): 3 figures
│   ├── fig04_variance_decomposition.pdf
│   ├── fig05_shap_beeswarm.pdf
│   └── fig07_heatmap_preprocessing.pdf
│
└── Sprint 3 (Complete): 6 figures
    ├── fig_M3_factorial_matrix.pdf
    ├── fig_R7_featurization_comparison.pdf
    ├── fig_R8_fm_dashboard.pdf
    ├── fig_shap_gt_vs_ensemble.pdf
    ├── fig_shap_heatmap.pdf
    └── fig_raincloud_auroc.pdf

Parallel execution: make -j4 r-figures-all
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `renv.lock` | Pinned R package versions |
| `configs/VISUALIZATION/colors.yaml` | Shared color definitions (Python + R) |
| `configs/VISUALIZATION/figure_style.yaml` | Shared style parameters |
| `Dockerfile.r` | R Docker image definition |
| `.pre-commit-config.yaml` | `r-hardcoding-check` hook |
| `Makefile` | R figure targets (lines 260-375) |

## Code Paths

| Module | Role |
|--------|------|
| `src/r/figure_system/common.R` | Project-wide R constants |
| `src/r/figure_system/config_loader.R` | YAML config loading for R |
| `src/r/figure_system/load_style.R` | `theme_foundation_plr()` definition |
| `src/r/figure_system/category_loader.R` | `load_color_definitions()` |
| `src/r/figure_system/save_figure.R` | `save_publication_figure()` |
| `src/r/figure_system/figure_factory.R` | Multi-panel figure composition |
| `src/r/figure_system/cd_diagram.R` | Critical difference diagram helper |
| `src/r/figure_system/compose_figures.R` | Panel arrangement utilities |
| `src/r/figures/` | 28 individual R figure scripts |
| `src/r/color_palettes.R` | Color palette definitions |
| `src/r/load_data.R` | Data loading utilities |
| `src/r/setup.R` | R environment setup |
| `scripts/check_r_hardcoding.py` | Pre-commit hook scanner |
| `scripts/export_data_for_r.py` | Python -> CSV export for R |
| `tests/test_r_figures/` | R figure test suite |
| `tests/test_r_guardrails/` | R code quality tests |
| `tests/test_r_environment.py` | R package availability test |

## Extension Guide

To add a new R figure:
1. Create script in `src/r/figures/new_figure.R`
2. Use figure system: `source("src/r/figure_system/load_style.R")`
3. Load colors: `colors <- load_color_definitions()`
4. Apply theme: `p + theme_foundation_plr()`
5. Save: `save_publication_figure(p, "fig_new_analysis")`
6. Add Makefile target: `$(R_OUTPUT)/fig_new_analysis.pdf: $(R_FIGURES)/new_figure.R`
7. Add to appropriate sprint in Makefile (`R_SPRINT1`/`R_SPRINT2`/`R_SPRINT3`)
8. If new data needed: add export script in `scripts/export_*_for_r.py`

To update R package versions:
1. In R session: `renv::install("package@version")`
2. Run `renv::snapshot()` to update `renv.lock`
3. Test in Docker: `make r-docker-test`

Note: This is a repo documentation figure - shows HOW the R figure system works, NOT research results.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "repo-89",
    "title": "R Figure Pipeline: renv -> rocker -> ggplot2"
  },
  "content_architecture": {
    "primary_message": "R figures use a reproducible pipeline: renv.lock pins packages, rocker Docker provides runtime, src/r/figure_system/ provides shared helpers for 28 figure scripts.",
    "layout_flow": "Four-panel layout: package management (top-left), figure system (top-right), data handoff (bottom-left), enforcement (bottom-right)",
    "spatial_anchors": {
      "packages": {"x": 0.05, "y": 0.05, "width": 0.4, "height": 0.4},
      "figure_system": {"x": 0.55, "y": 0.05, "width": 0.4, "height": 0.4},
      "data_handoff": {"x": 0.05, "y": 0.5, "width": 0.4, "height": 0.4},
      "enforcement": {"x": 0.55, "y": 0.5, "width": 0.4, "height": 0.4}
    },
    "key_structures": [
      {
        "name": "renv.lock",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Pinned versions"]
      },
      {
        "name": "figure_system/",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Shared helpers"]
      },
      {
        "name": "Data handoff",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["DuckDB -> CSV -> R"]
      },
      {
        "name": "Pre-commit",
        "role": "abnormal_warning",
        "is_highlighted": true,
        "labels": ["No hex, no ggsave"]
      }
    ],
    "callout_boxes": [
      {"heading": "DATA HANDOFF", "body_text": "Python exports DuckDB to CSV (outputs/r_data/). R reads CSV and generates ggplot2 figures."},
      {"heading": "ENFORCEMENT", "body_text": "Pre-commit hook bans hex colors, ggsave(), and custom themes in R code."}
    ]
  }
}
```

## Alt Text

Four-panel diagram showing R figure pipeline: renv.lock package management, shared figure system helpers, Python-to-R data handoff via CSV, and pre-commit enforcement of coding standards across 28 R figure scripts.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
