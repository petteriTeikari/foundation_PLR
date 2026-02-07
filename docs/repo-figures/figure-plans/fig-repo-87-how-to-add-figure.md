# fig-repo-87: Creating a New Figure: Config -> Code -> QA -> Commit

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-87 |
| **Title** | Creating a New Figure: Config -> Code -> QA -> Commit |
| **Complexity Level** | L3 |
| **Target Persona** | Research Scientist / ML Engineer |
| **Location** | `src/viz/README.md`, `docs/contributing/` |
| **Priority** | P2 (High) |

## Purpose

Document the complete lifecycle of a figure from initial registration through QA to commit. The zero-tolerance figure QA policy (CRITICAL-FAILURE-001) means figures cannot be committed casually -- they must pass automated provenance, statistical, rendering, and accessibility checks. This figure prevents the common mistake of writing a plot script without registering it or saving JSON data.

## Key Message

Every figure follows the same lifecycle: register in YAML, write plot script with mandatory patterns (setup_style, YAML combos, DuckDB read-only, semantic colors, JSON sidecar), pass zero-tolerance QA, then commit through pre-commit hooks.

## Content Specification

### Panel 1: Five-Step Figure Lifecycle

```
STEP 1: REGISTER IN FIGURE REGISTRY
══════════════════════════════════════
File: configs/VISUALIZATION/figure_registry.yaml

  NEW_FIGURE:
    id: "NEW"
    title: "My New Analysis"
    script: "src/viz/new_figure.py"
    combos: "standard"                    ← or "extended" for supplementary
    json_privacy: "public"                ← or "private" for subject-level
    output_prefix: "fig_NEW"
    description: "Brief description of what this figure shows"
                │
                ▼
STEP 2: WRITE PLOT SCRIPT
══════════════════════════════════════
File: src/viz/new_figure.py

  from src.viz.plot_config import setup_style, COLORS, save_figure
  from src.viz.config_loader import load_combos
  import duckdb

  # 1. FIRST CALL: setup_style()
  setup_style()                           ← MANDATORY first call

  # 2. LOAD COMBOS FROM YAML (never hardcode!)
  combos = load_combos("standard")        ← From plot_hyperparam_combos.yaml
  # Returns: [ground_truth, best_ensemble, best_single_fm, traditional]

  # 3. READ DATA FROM DuckDB (READ ONLY!)
  conn = duckdb.connect("data/public/foundation_plr_results.db", read_only=True)
  df = conn.execute("""
      SELECT outlier_method, imputation_method, auroc, calibration_slope
      FROM essential_metrics
      WHERE classifier = ?
  """, [FIXED_CLASSIFIER]).fetchdf()
  #
  # BANNED in src/viz/:
  #   from sklearn.metrics import roc_auc_score       ← BANNED
  #   from src.stats.calibration_extended import ...    ← BANNED

  # 4. USE SEMANTIC COLORS (never hex!)
  fig, ax = plt.subplots()
  for combo in combos:
      ax.plot(..., color=COLORS[combo["id"]])  ← Semantic color lookup

  # 5. SAVE WITH JSON SIDECAR
  data_dict = {
      "combos_used": [c["id"] for c in combos],
      "metrics": {...},
      "n_subjects": 208
  }
  save_figure(fig, "fig_NEW_analysis", data=data_dict)
  # Creates:
  #   figures/generated/fig_NEW_analysis.png
  #   figures/generated/data/fig_NEW_analysis.json
                │
                ▼
STEP 3: GENERATE
══════════════════════════════════════
Command:
  python src/viz/generate_all_figures.py --figure NEW

  # Or generate all figures:
  python src/viz/generate_all_figures.py

  # List available figure IDs:
  python src/viz/generate_all_figures.py --list
                │
                ▼
STEP 4: QA (ZERO TOLERANCE)
══════════════════════════════════════
Command:
  pytest tests/test_figure_qa/ -v

  7 test files, organized by priority:

  P0 (CRITICAL - synthetic data detection):
  ├── test_data_provenance.py        Catches CRITICAL-FAILURE-001
  │   Verifies: JSON sidecar exists, source_db is production,
  │   no synthetic markers, data hash matches

  P1 (Statistical validity):
  ├── test_no_nan_ci.py              No NaN in confidence intervals
  │   Verifies: All CI bounds are finite numbers

  P2 (Rendering standards):
  ├── test_publication_standards.py   DPI >= 100, dimensions, fonts
  │   Verifies: Figure meets journal requirements
  ├── test_rendering_artifacts.py     No clipped content, axis labels
  │   Verifies: No visual defects

  P3 (Accessibility):
  └── (accessibility checks)         Color blindness safe palette
      Verifies: Not red-green only

  ALL MUST PASS. Zero failures allowed.
                │
                ▼
STEP 5: COMMIT
══════════════════════════════════════
Command:
  git add src/viz/new_figure.py
  git add configs/VISUALIZATION/figure_registry.yaml
  git add figures/generated/fig_NEW_analysis.png
  git add figures/generated/data/fig_NEW_analysis.json
  git commit -m "feat(figures): Add NEW analysis figure"

  Pre-commit hooks run automatically:
  ├── ruff (format + lint)
  ├── computation-decoupling          ← Scans for banned imports
  ├── figure-isolation-check          ← No synthetic data in outputs
  └── r-hardcoding-check              ← (if R figure)
```

### Panel 2: The Five Mandatory Patterns (Checklist)

```
Every figure script MUST:

  [1] setup_style()           Call before any matplotlib operations
  [2] load_combos("...")      Load from YAML, never hardcode method names
  [3] DuckDB READ ONLY        conn = duckdb.connect(path, read_only=True)
  [4] COLORS[combo["id"]]     Semantic colors from COLORS dict
  [5] save_figure(fig, name,  JSON sidecar with all numeric data
      data=data_dict)

  Every figure script must NOT:

  [x] from sklearn.metrics    Computation belongs in Block 1
  [x] color="#006BA2"          Hardcoded hex colors
  [x] "pupil-gt"              Hardcoded method names
  [x] width=14, height=6      Hardcoded dimensions
  [x] plt.savefig()           Use save_figure() instead
```

### Panel 3: JSON Sidecar Structure

```
figures/generated/data/fig_NEW_analysis.json
{
  "figure_id": "NEW",
  "generated_at": "2026-02-06T12:00:00",
  "source_db": "data/public/foundation_plr_results.db",
  "combos_used": ["ground_truth", "best_ensemble", ...],
  "data": {
    "ground_truth": { "auroc": ..., "calibration_slope": ... },
    ...
  },
  "summary_statistics": { "n_subjects": 208, "n_bootstrap": 1000 }
}
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `configs/VISUALIZATION/figure_registry.yaml` | Figure registration (ID, script, combos, privacy) |
| `configs/VISUALIZATION/plot_hyperparam_combos.yaml` | Standard 4 + extended combos |
| `configs/VISUALIZATION/colors.yaml` | Color definitions and combo color mappings |
| `configs/VISUALIZATION/figure_style.yaml` | DPI, fonts, dimensions |
| `configs/VISUALIZATION/figure_layouts.yaml` | Layout specifications |

## Code Paths

| Module | Role |
|--------|------|
| `src/viz/plot_config.py` | `setup_style()`, `COLORS` dict, `save_figure()` |
| `src/viz/config_loader.py` | `load_combos()`, `load_figure_config()` |
| `src/viz/generate_all_figures.py` | Entry point: `--figure ID`, `--list` |
| `src/viz/metric_registry.py` | STRATOS metric groups and display names |
| `src/viz/figure_dimensions.py` | Figure size calculations |
| `src/viz/figure_export.py` | JSON sidecar generation |
| `tests/test_figure_qa/` | 7 QA test files (P0-P3 priorities) |
| `tests/test_figure_qa/conftest.py` | `figure_dir`, `json_files`, `png_files` fixtures |
| `scripts/check_computation_decoupling.py` | Pre-commit: banned import scanner |
| `scripts/check_figure_isolation.py` | Pre-commit: synthetic data in outputs |

## Extension Guide

To add a new figure QA check:
1. Determine priority: P0 (data provenance), P1 (statistics), P2 (rendering), P3 (accessibility)
2. Add test to appropriate file in `tests/test_figure_qa/`
3. Use fixtures from `tests/test_figure_qa/conftest.py` (json_files, png_files)
4. Test must scan ALL figures in `figures/generated/`

To create an R figure instead of Python:
1. Register in `configs/VISUALIZATION/figure_registry.yaml` (same as Python)
2. Write script in `src/r/figures/new_figure.R`
3. Use R figure system: `theme_foundation_plr()`, `load_color_definitions()`, `save_publication_figure()`
4. Add Makefile target for the R figure
5. QA via `tests/test_r_figures/` and `scripts/check_r_hardcoding.py`

Note: This is a repo documentation figure - shows HOW to create figures, NOT research results.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "repo-87",
    "title": "Creating a New Figure: Config -> Code -> QA -> Commit"
  },
  "content_architecture": {
    "primary_message": "Every figure follows the same lifecycle: register, write script with 5 mandatory patterns, generate, pass zero-tolerance QA, commit through hooks.",
    "layout_flow": "Top-down 5-step vertical flowchart with mandatory patterns checklist sidebar",
    "spatial_anchors": {
      "step1_register": {"x": 0.05, "y": 0.02},
      "step2_write": {"x": 0.05, "y": 0.2},
      "step3_generate": {"x": 0.05, "y": 0.52},
      "step4_qa": {"x": 0.05, "y": 0.6},
      "step5_commit": {"x": 0.05, "y": 0.82},
      "checklist": {"x": 0.7, "y": 0.2, "width": 0.25, "height": 0.35}
    },
    "key_structures": [
      {
        "name": "Figure Registry YAML",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Register first"]
      },
      {
        "name": "5 Mandatory Patterns",
        "role": "abnormal_warning",
        "is_highlighted": true,
        "labels": ["setup_style, combos, DuckDB, COLORS, save_figure"]
      },
      {
        "name": "QA Tests",
        "role": "highlight_accent",
        "is_highlighted": true,
        "labels": ["Zero tolerance", "P0-P3 priorities"]
      }
    ],
    "callout_boxes": [
      {"heading": "ZERO TOLERANCE QA", "body_text": "All 7 test files must pass. P0 catches synthetic data fraud (CRITICAL-FAILURE-001)."},
      {"heading": "READ-ONLY RULE", "body_text": "src/viz/ reads DuckDB. All computation happens in src/stats/ during extraction."}
    ]
  }
}
```

## Alt Text

Five-step vertical flowchart showing figure lifecycle: register in YAML, write script with 5 mandatory patterns, generate, pass zero-tolerance QA tests at 4 priority levels, and commit through pre-commit hooks.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
