# fig-repo-80: 11 x 8 x 5 = 440: The Full Configuration Space

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-80 |
| **Title** | 11 x 8 x 5 = 440: The Full Configuration Space |
| **Complexity Level** | L2 |
| **Target Persona** | Biostatistician / Research Scientist |
| **Location** | `configs/mlflow_registry/README.md`, `ARCHITECTURE.md` |
| **Priority** | P2 (High) |

## Purpose

Show the full factorial experiment design: 11 outlier detection methods, 8 imputation methods, and 5 classifiers form a 3D grid. The main analysis fixes CatBoost (11 x 8 x 1 = 88 configs) because the research question is about preprocessing, not classifier comparison. The actual DuckDB contains 316 configs (not all 440 combinations were run).

## Key Message

The experiment grid is 11 outlier x 8 imputation x 5 classifiers = 440 configs. CatBoost is FIXED for the main analysis (88 configs) because the research question varies preprocessing, not the classifier.

## Content Specification

### Panel 1: 3D Configuration Grid

```
┌─────────────────────────────────────────────────────────────────────────┐
│              THE FULL CONFIGURATION SPACE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│              Imputation Methods (8)                                      │
│             ┌─────────────────────────────────────┐                     │
│            /│ pupil-gt                             │/│                   │
│           / │ CSDI                                 │ │                   │
│          /  │ SAITS                                │ │                   │
│         /   │ TimesNet                             │ │                   │
│        /    │ NuwaTS                               │ │ Classifiers (5)  │
│       /     │ MissForest                           │ │ ┌──────────────┐│
│      /      │ linear_interpolation                 │ │ │ CatBoost     ││
│     /       │ mean_imputation                      │ │ │ XGBoost      ││
│    /        └─────────────────────────────────────┘ │ │ LogReg       ││
│   /         │                                       │ │ TabPFN       ││
│  /          │                                       │ │ TabM         ││
│ /           │                                       │ └──────────────┘│
│┌────────────────────────────────────────────┐       │                  │
││ Outlier Detection Methods (11)              │───────┘                  │
││ ┌──────────────────────────────────────────┐│                          │
││ │  1. pupil-gt  (ground truth)              ││                          │
││ │  2. MOMENT-gt-finetune                    ││                          │
││ │  3. MOMENT-gt-zeroshot                    ││                          │
││ │  4. UniTS-gt-finetune                     ││                          │
││ │  5. TimesNet-gt                           ││                          │
││ │  6. LOF                                   ││                          │
││ │  7. OneClassSVM                           ││                          │
││ │  8. PROPHET                               ││                          │
││ │  9. SubPCA                                ││                          │
││ │ 10. Ensemble (7-method)                   ││                          │
││ │ 11. EnsembleThresholded (4 FM)            ││                          │
││ └──────────────────────────────────────────┘│                          │
│└────────────────────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 2: Analysis Slices

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ANALYSIS SLICES                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  MAIN ANALYSIS (research question):                                     │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Fix CatBoost → 11 outlier x 8 imputation x 1 = 88 configs       │  │
│  │  Question: How do preprocessing choices affect downstream AUROC?   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  FULL GRID (sensitivity analysis):                                      │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  All classifiers → 11 x 8 x 5 = 440 configs (theoretical)        │  │
│  │  Confirms findings are not classifier-specific                     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  IN DUCKDB (available runs):                                            │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  316 configs in essential_metrics table                            │  │
│  │  (not all 440 combos ran — some failed or were not attempted)     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  STANDARD 4 COMBOS (main figures):                                      │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  From plot_hyperparam_combos.yaml:                                │  │
│  │  ┌──────────────────┬─────────────┬──────────────┬─────────────┐ │  │
│  │  │ ground_truth     │ best_       │ best_single_ │ traditional │ │  │
│  │  │ pupil-gt +       │ ensemble    │ fm           │ LOF +       │ │  │
│  │  │ pupil-gt         │ Ensemble +  │ MOMENT-gt +  │ SAITS       │ │  │
│  │  │                  │ CSDI        │ SAITS        │             │ │  │
│  │  └──────────────────┴─────────────┴──────────────┴─────────────┘ │  │
│  │  These 4 represent the key comparisons for main manuscript figs  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 3: Why CatBoost Is Fixed

```
┌─────────────────────────────────────────────────────────────────────────┐
│  WHY CatBoost IS FIXED                                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Research Question:                                                     │
│  "How do PREPROCESSING choices affect downstream classification?"       │
│                                                                          │
│  FIX the classifier  →  CatBoost (already established)                 │
│  VARY the preprocessing →  11 outlier x 8 imputation                   │
│  MEASURE the effect  →  On ALL STRATOS metrics                         │
│                                                                          │
│  NOT about: comparing classifiers (CatBoost vs XGBoost)                │
│  NOT about: maximizing AUROC                                            │
│  NOT about: generic ML benchmarking                                     │
│                                                                          │
│  Code: FIXED_CLASSIFIER = "CatBoost"  (src/viz/plot_config.py)         │
│  Config: plot_hyperparam_combos.yaml   (all combos use CatBoost)       │
└─────────────────────────────────────────────────────────────────────────┘
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `configs/mlflow_registry/parameters/classification.yaml` | Defines all 11 outlier, 8 imputation, 5 classifier methods |
| `configs/VISUALIZATION/plot_hyperparam_combos.yaml` | Standard 4 combos for main figures |
| `configs/registry_canary.yaml` | Frozen reference counts (11/8/5) |

## Code Paths

| Module | Role |
|--------|------|
| `src/data_io/registry.py` | EXPECTED_OUTLIER_COUNT=11, EXPECTED_IMPUTATION_COUNT=8, EXPECTED_CLASSIFIER_COUNT=5 |
| `src/viz/plot_config.py` | FIXED_CLASSIFIER = "CatBoost" |
| `scripts/extract_all_configs_to_duckdb.py` | Extracts 316 available configs to DuckDB |

## Extension Guide

To add a new outlier detection method:
1. Add to `configs/mlflow_registry/parameters/classification.yaml` (outlier_methods list)
2. Increment count in `configs/registry_canary.yaml` (11 to 12)
3. Update `EXPECTED_OUTLIER_COUNT` in `src/data_io/registry.py`
4. Update `tests/test_registry.py` assertions
5. Run MLflow experiments for the new method
6. Re-extract: `make extract` (populates DuckDB with new configs)
7. The grid becomes 12 x 8 x 5 = 480 configs

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "repo-80",
    "title": "11 x 8 x 5 = 440: The Full Configuration Space"
  },
  "content_architecture": {
    "primary_message": "The experiment grid is 11 outlier x 8 imputation x 5 classifiers = 440 configs. CatBoost is FIXED for the main analysis (88 configs).",
    "layout_flow": "3D grid at top showing the full space, analysis slices below showing main vs full vs available",
    "spatial_anchors": {
      "grid_3d": {"x": 0.5, "y": 0.25},
      "main_analysis": {"x": 0.25, "y": 0.6},
      "full_grid": {"x": 0.5, "y": 0.6},
      "duckdb_count": {"x": 0.75, "y": 0.6},
      "standard_4": {"x": 0.5, "y": 0.85}
    },
    "key_structures": [
      {
        "name": "Outlier Methods (11)",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["pupil-gt to EnsThreshold"]
      },
      {
        "name": "Imputation Methods (8)",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["pupil-gt to mean"]
      },
      {
        "name": "Classifiers (5)",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["CatBoost FIXED"]
      },
      {
        "name": "Standard 4 Combos",
        "role": "highlight_accent",
        "is_highlighted": true,
        "labels": ["GT, Ensemble, FM, Traditional"]
      }
    ],
    "callout_boxes": [
      {"heading": "RESEARCH DESIGN", "body_text": "Fix the classifier (CatBoost). Vary the preprocessing. Measure the effect on ALL STRATOS metrics."}
    ]
  }
}
```

## Alt Text

Three-dimensional grid showing 11 outlier methods, 8 imputation methods, and 5 classifiers forming 440 total configs, with CatBoost fixed for the 88-config main analysis.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
