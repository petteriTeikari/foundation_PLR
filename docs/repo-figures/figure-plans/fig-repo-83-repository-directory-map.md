# fig-repo-83: Repository at a Glance: What's Where

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-83 |
| **Title** | Repository at a Glance: What's Where |
| **Complexity Level** | L1 |
| **Target Persona** | All (first thing a new developer sees) |
| **Location** | `README.md`, `docs/onboarding/` |
| **Priority** | P1 (Critical) |

## Purpose

Provide a single visual that orients new developers to the repository structure. After seeing this figure, a developer should know exactly which directory to look in for any type of file (source code, configuration, tests, data, outputs).

## Key Message

The repo has 7 main directories: `src/` (code), `configs/` (settings), `tests/` (validation), `data/` (databases), `figures/` (outputs), `docs/` (documentation), and `.claude/` (AI agent instructions).

## Content Specification

### Panel 1: Annotated Directory Tree

```
foundation_PLR/
├── src/                           [SOURCE CODE]
│   ├── anomaly_detection/         11 outlier detection methods
│   │   ├── momentfm_outlier/      MOMENT foundation model
│   │   ├── units/                 UniTS foundation model
│   │   ├── outlier_sklearn.py     LOF, OneClassSVM
│   │   ├── outlier_prophet.py     PROPHET
│   │   └── timesnet_wrapper.py    TimesNet
│   ├── classification/            5 classifiers
│   │   ├── catboost/              CatBoost (FIXED for main analysis)
│   │   ├── xgboost_cls/           XGBoost
│   │   ├── tabpfn/                TabPFN (foundation tabular)
│   │   ├── tabm/                  TabM
│   │   └── sklearn_simple_classifiers.py  LogReg
│   ├── extraction/                Block 1: MLflow -> DuckDB
│   ├── stats/                     STRATOS metric computation
│   │   ├── classifier_metrics.py  Bootstrap loop
│   │   ├── calibration_extended.py Calibration slope/intercept
│   │   ├── clinical_utility.py    Net benefit / DCA
│   │   ├── pminternal_wrapper.py  R interop (model stability)
│   │   └── _defaults.py           Constants (n_bootstrap, ci_level)
│   ├── viz/                       Block 2: Figure generation (READ-ONLY)
│   │   ├── plot_config.py         COLORS dict, setup_style(), save_figure()
│   │   ├── metric_registry.py     STRATOS metric groups
│   │   ├── generate_all_figures.py Entry point (--figure ID)
│   │   └── *.py                   Individual figure scripts
│   ├── r/                         R figure scripts (ggplot2)
│   │   ├── figure_system/         theme, colors, save helpers
│   │   └── figures/               28 R figure scripts
│   ├── orchestration/             Prefect flows (extract, analyze)
│   │   └── flows/                 extraction_flow.py, analysis_flow.py
│   ├── data_io/                   Registry, data loading
│   │   ├── registry.py            Method validation (11/8/5)
│   │   └── streaming_duckdb_export.py  DuckDB writer
│   ├── featurization/             Handcrafted features + embeddings
│   ├── ensemble/                  Ensemble outlier construction
│   └── synthetic/                 Test data generators
│
├── configs/                       [CONFIGURATION - Hydra]
│   ├── mlflow_registry/           SINGLE SOURCE OF TRUTH (11/8/5)
│   │   └── parameters/            classification.yaml
│   ├── VISUALIZATION/             Figure configs
│   │   ├── plot_hyperparam_combos.yaml  Standard 4 + extended combos
│   │   ├── colors.yaml            Color definitions
│   │   ├── figure_registry.yaml   Figure specifications
│   │   └── figure_style.yaml      DPI, fonts, dimensions
│   ├── CLS_MODELS/                Classifier configs (CatBoost.yaml, ...)
│   ├── CLS_HYPERPARAMS/           HPO search spaces
│   ├── OUTLIER_MODELS/            Outlier detector configs
│   ├── MODELS/                    Imputation model configs
│   ├── PLR_FEATURIZATION/         Feature window definitions
│   ├── registry_canary.yaml       Anti-cheat reference values
│   └── demo_subjects.yaml         8 demo subjects (H001-H004, G001-G004)
│
├── tests/                         [TESTS - 2000+ tests]
│   ├── conftest.py                Root fixtures (session DB connections)
│   ├── unit/                      Pure function tests
│   ├── integration/               Synthetic data tests
│   ├── e2e/                       Full pipeline tests
│   ├── test_figure_qa/            ZERO TOLERANCE figure QA
│   ├── test_no_hardcoding/        Anti-hardcoding enforcement
│   ├── test_guardrails/           Registry, decoupling checks
│   ├── test_r_figures/            R figure validation
│   └── test_registry.py           Registry count verification
│
├── data/                          [DATA - DuckDB databases]
│   ├── public/                    Extracted results (checked in)
│   │   └── foundation_plr_results.db
│   ├── private/                   Subject lookup (gitignored)
│   │   └── subject_lookup.yaml
│   └── synthetic/                 Test-only synthetic data
│       └── SYNTH_PLR_DEMO.db
│
├── figures/generated/             [OUTPUTS - figures + JSON sidecars]
│   ├── *.png                      Python matplotlib figures
│   ├── data/*.json                JSON data sidecars (reproducibility)
│   └── ggplot2/                   R ggplot2 figures (PDF)
│
├── docs/                          [DOCUMENTATION]
│   └── repo-figures/              Repository figure plans
│
├── scripts/                       [AUTOMATION]
│   ├── setup-dev-environment.sh   One-command setup
│   ├── reproduce_all_results.py   Full pipeline runner
│   ├── check_computation_decoupling.py  Pre-commit hook
│   ├── check_r_hardcoding.py      Pre-commit hook
│   └── verify_registry_integrity.py Anti-cheat check
│
├── .claude/                       [AI AGENT INSTRUCTIONS]
│   ├── CLAUDE.md                  Behavior contract
│   ├── rules/                     6 specific rules (00-25)
│   ├── domains/                   Context files (MLflow, viz, ...)
│   └── docs/meta-learnings/       Critical failure post-mortems
│
├── Makefile                       40+ targets (test, figures, reproduce)
├── pyproject.toml                 Python deps (uv managed)
├── renv.lock                      R package lockfile
└── CLAUDE.md                      Project overview (root)
```

### Panel 2: Color-Coded Legend

| Color | Category | Directories |
|-------|----------|-------------|
| Blue | Source code | `src/` |
| Green | Configuration | `configs/` |
| Red | Tests | `tests/` |
| Gold | Data | `data/` |
| Purple | Outputs | `figures/generated/` |
| Gray | Documentation | `docs/`, `.claude/` |
| Orange | Automation | `scripts/`, `Makefile` |

### Panel 3: Quick Navigation Callout

```
"Where do I find...?"

Method implementations → src/anomaly_detection/ or src/classification/
Figure scripts         → src/viz/ (Python) or src/r/figures/ (R)
Valid method names     → configs/mlflow_registry/
Figure color palette   → configs/VISUALIZATION/colors.yaml
Test fixtures          → tests/conftest.py
Extracted results      → data/public/foundation_plr_results.db
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `configs/mlflow_registry/parameters/classification.yaml` | Master list: 11 outlier, 8 imputation, 5 classifiers |
| `configs/VISUALIZATION/plot_hyperparam_combos.yaml` | Standard 4 combos for main figures |
| `configs/demo_subjects.yaml` | 8 demo subjects for visualization |
| `configs/registry_canary.yaml` | Anti-cheat reference counts |

## Code Paths

| Module | Role |
|--------|------|
| `src/viz/generate_all_figures.py` | Entry point for Python figures |
| `src/r/figures/*.R` | 28 R figure scripts |
| `src/orchestration/flows/extraction_flow.py` | Block 1 entry point |
| `src/orchestration/flows/analysis_flow.py` | Block 2 entry point |
| `src/data_io/registry.py` | Method name validation |
| `scripts/setup-dev-environment.sh` | One-command development setup |

## Extension Guide

To understand where new code should go:
1. New outlier method: `src/anomaly_detection/` + `configs/OUTLIER_MODELS/`
2. New classifier: `src/classification/` + `configs/CLS_MODELS/`
3. New figure: `src/viz/` + `configs/VISUALIZATION/figure_registry.yaml`
4. New R figure: `src/r/figures/` + update `Makefile`
5. New test: `tests/` matching the source directory structure

Note: This is a repo documentation figure - shows HOW the code is organized, NOT research results.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "repo-83",
    "title": "Repository at a Glance: What's Where"
  },
  "content_architecture": {
    "primary_message": "The repo has 7 main directories: src/ (code), configs/ (settings), tests/ (validation), data/ (databases), figures/ (outputs), docs/ (documentation), and .claude/ (AI agent instructions).",
    "layout_flow": "Top-down directory tree with color-coded annotations and quick navigation callout",
    "spatial_anchors": {
      "tree": {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.8},
      "legend": {"x": 0.65, "y": 0.1, "width": 0.3, "height": 0.3},
      "quicknav": {"x": 0.65, "y": 0.5, "width": 0.3, "height": 0.4}
    },
    "key_structures": [
      {
        "name": "src/",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["ALL Python source code"]
      },
      {
        "name": "configs/",
        "role": "foundation_model",
        "is_highlighted": true,
        "labels": ["ALL configuration (Hydra)"]
      },
      {
        "name": "tests/",
        "role": "abnormal_warning",
        "is_highlighted": true,
        "labels": ["2000+ tests"]
      },
      {
        "name": "data/",
        "role": "highlight_accent",
        "is_highlighted": true,
        "labels": ["DuckDB databases"]
      }
    ],
    "callout_boxes": [
      {"heading": "QUICK NAVIGATION", "body_text": "Method implementations in src/, configs in configs/, tests mirror src/ structure."}
    ]
  }
}
```

## Alt Text

Annotated directory tree of the foundation_PLR repository showing 7 color-coded main directories: source code, configuration, tests, data, outputs, documentation, and AI instructions.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
