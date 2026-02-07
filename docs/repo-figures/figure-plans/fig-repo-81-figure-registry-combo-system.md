# fig-repo-81: Figure Registry: From YAML to Generated PNG

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-81 |
| **Title** | Figure Registry: From YAML to Generated PNG |
| **Complexity Level** | L3 |
| **Target Persona** | ML Engineer |
| **Location** | `configs/VISUALIZATION/README.md`, `src/viz/README.md` |
| **Priority** | P2 (High) |

## Purpose

Show the complete lifecycle of a figure from YAML definition to generated PNG with JSON sidecar. Developers need to understand how figure_registry.yaml defines figure specifications, how plot_hyperparam_combos.yaml provides the standard combos, how generate_all_figures.py orchestrates generation, and how privacy levels control what gets committed to git.

## Key Message

Every figure is defined in figure_registry.yaml, draws combos from plot_hyperparam_combos.yaml, is generated via generate_all_figures.py, and produces both a PNG and a JSON sidecar for reproducibility.

## Content Specification

### Panel 1: The Figure Generation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FIGURE GENERATION PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STEP 1: FIGURE REGISTRY                                                │
│  configs/VISUALIZATION/figure_registry.yaml                             │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  R7_calibration:                                                   │  │
│  │    script: calibration_plot.py                                    │  │
│  │    combos: standard            ← references combos YAML          │  │
│  │    json_privacy: public        ← determines git visibility       │  │
│  │    output: fig_R7_calibration.png                                 │  │
│  │                                                                    │  │
│  │  R8_dca:                                                          │  │
│  │    script: dca_plot.py                                            │  │
│  │    combos: standard                                               │  │
│  │    json_privacy: public                                           │  │
│  │    output: fig_R8_dca.png                                         │  │
│  │                                                                    │  │
│  │  R9_prob_distribution:                                            │  │
│  │    script: prob_distribution.py                                   │  │
│  │    combos: standard                                               │  │
│  │    json_privacy: private       ← subject-level, gitignored       │  │
│  │    output: fig_R9_probability.png                                 │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                    │                                                      │
│                    ▼                                                      │
│  STEP 2: COMBO RESOLUTION                                               │
│  configs/VISUALIZATION/plot_hyperparam_combos.yaml                      │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  standard_combos:                                                  │  │
│  │    ground_truth:                                                   │  │
│  │      outlier_method: pupil-gt                                     │  │
│  │      imputation_method: pupil-gt                                  │  │
│  │      classifier: CatBoost                                         │  │
│  │      color_ref: "--color-ground-truth"                            │  │
│  │    best_ensemble:                                                  │  │
│  │      outlier_method: ensemble-LOF-MOMENT-...                      │  │
│  │      imputation_method: CSDI                                      │  │
│  │      classifier: CatBoost                                         │  │
│  │      color_ref: "--color-fm-primary"                              │  │
│  │    best_single_fm:                                                 │  │
│  │      outlier_method: MOMENT-gt-finetune                           │  │
│  │      imputation_method: SAITS                                     │  │
│  │      classifier: CatBoost                                         │  │
│  │      color_ref: "--color-fm-secondary"                            │  │
│  │    traditional:                                                    │  │
│  │      outlier_method: LOF                                          │  │
│  │      imputation_method: SAITS                                     │  │
│  │      classifier: CatBoost                                         │  │
│  │      color_ref: "--color-traditional"                             │  │
│  │                                                                    │  │
│  │  color_definitions:                                                │  │
│  │    "--color-ground-truth": "#666666"                               │  │
│  │    "--color-fm-primary": "#0072B2"                                 │  │
│  │    "--color-fm-secondary": "#56B4E9"                               │  │
│  │    "--color-traditional": "#E69F00"                                │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                    │                                                      │
│                    ▼                                                      │
│  STEP 3: ORCHESTRATION                                                  │
│  src/viz/generate_all_figures.py                                        │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  CLI:                                                              │  │
│  │  python src/viz/generate_all_figures.py --figure R7                │  │
│  │  python src/viz/generate_all_figures.py --list                    │  │
│  │                                                                    │  │
│  │  1. Loads figure_registry.yaml                                    │  │
│  │  2. Resolves combos from plot_hyperparam_combos.yaml              │  │
│  │  3. Calls script (e.g., calibration_plot.py)                      │  │
│  │  4. Script calls setup_style() → DuckDB query → plot → save      │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                    │                                                      │
│                    ▼                                                      │
│  STEP 4: OUTPUT                                                         │
│  figures/generated/                                                     │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  fig_R7_calibration.png       ← The rendered figure              │  │
│  │  fig_R7_calibration.json      ← JSON sidecar (PUBLIC)            │  │
│  │                                                                    │  │
│  │  data/fig_R9_subject_*.json   ← Subject-level (PRIVATE)          │  │
│  │  (gitignored for patient privacy)                                 │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 2: Standard 4 Combos (Visual)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  THE STANDARD 4 COMBOS (loaded from YAML, never hardcoded)              │
├──────────────────┬──────────────────┬───────────────┬───────────────────┤
│  ground_truth    │  best_ensemble   │  best_single  │  traditional      │
│  ────────────    │  ──────────────  │  ───────────  │  ───────────      │
│  pupil-gt +      │  Ensemble +      │  MOMENT-gt +  │  LOF +            │
│  pupil-gt        │  CSDI            │  SAITS        │  SAITS            │
│  (CatBoost)      │  (CatBoost)      │  (CatBoost)   │  (CatBoost)       │
│                  │                  │               │                   │
│  Reference:      │  Best overall:   │  Best single  │  Traditional      │
│  Oracle upper    │  FM ensemble     │  FM pipeline  │  baseline         │
│  bound           │                  │               │                   │
├──────────────────┴──────────────────┴───────────────┴───────────────────┤
│  MAX 4 combos per main figure.  MAX 8 per supplementary figure.        │
│  Ground truth REQUIRED in every comparison.                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 3: Privacy Levels

```
┌─────────────────────────────────────────────────────────────────────────┐
│  JSON SIDECAR PRIVACY LEVELS                                             │
├────────────────┬────────────────────────────────────────────────────────┤
│  Level         │  Behavior                                              │
├────────────────┼────────────────────────────────────────────────────────┤
│  public        │  JSON committed to git                                │
│                │  Contains: aggregate metrics, summary stats            │
│                │  No subject IDs, no individual predictions             │
├────────────────┼────────────────────────────────────────────────────────┤
│  private       │  JSON gitignored (in .gitignore)                      │
│                │  Contains: per-subject predictions, individual traces  │
│                │  Pattern: **/subject_*.json, **/individual_*.json      │
│                │  Required for reproduction but not shared publicly     │
└────────────────┴────────────────────────────────────────────────────────┘
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `configs/VISUALIZATION/figure_registry.yaml` | Figure definitions: script, combos, privacy, output |
| `configs/VISUALIZATION/plot_hyperparam_combos.yaml` | Standard/extended combos, color_ref, color_definitions |
| `configs/VISUALIZATION/colors.yaml` | Supplementary color definitions |

## Code Paths

| Module | Role |
|--------|------|
| `src/viz/generate_all_figures.py` | CLI orchestrator: --figure R7, --list |
| `src/viz/plot_config.py` | setup_style(), COLORS, save_figure(), get_combo_color() |
| `src/viz/calibration_plot.py` | Example figure script (reads DuckDB, plots calibration) |
| `src/viz/dca_plot.py` | Example figure script (reads DuckDB, plots DCA curves) |
| `src/viz/figure_dimensions.py` | Figure dimension configuration |

## Extension Guide

To add a new figure to the registry:
1. Add entry to `configs/VISUALIZATION/figure_registry.yaml` (ID, script, combos, privacy)
2. Create plot script in `src/viz/` following the pattern: setup_style() then DuckDB query then save_figure()
3. Register in `generate_all_figures.py` for CLI access
4. Choose privacy level: public (aggregate data) or private (subject-level)
5. Run: `python src/viz/generate_all_figures.py --figure NEW_ID`
6. QA: `pytest tests/test_figure_qa/ -v` (ZERO TOLERANCE)

To add a new combo:
1. Add to `standard_combos:` or `extended_combos:` in `plot_hyperparam_combos.yaml`
2. Add color_ref and corresponding hex in color_definitions
3. Verify the combo exists in MLflow data before using

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "repo-81",
    "title": "Figure Registry: From YAML to Generated PNG"
  },
  "content_architecture": {
    "primary_message": "Every figure is defined in figure_registry.yaml, draws combos from plot_hyperparam_combos.yaml, and produces PNG + JSON sidecar via generate_all_figures.py.",
    "layout_flow": "Top-down pipeline: registry YAML -> combo resolution -> orchestration -> output files",
    "spatial_anchors": {
      "registry": {"x": 0.5, "y": 0.15},
      "combos": {"x": 0.5, "y": 0.35},
      "orchestrator": {"x": 0.5, "y": 0.55},
      "output": {"x": 0.5, "y": 0.75},
      "privacy": {"x": 0.5, "y": 0.9}
    },
    "key_structures": [
      {
        "name": "figure_registry.yaml",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Figure definitions"]
      },
      {
        "name": "plot_hyperparam_combos.yaml",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Standard 4 combos"]
      },
      {
        "name": "generate_all_figures.py",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["CLI orchestrator"]
      },
      {
        "name": "PNG + JSON sidecar",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Reproducibility output"]
      }
    ],
    "callout_boxes": [
      {"heading": "NEVER HARDCODE", "body_text": "Combos are always loaded from YAML. MAX 4 per main figure, ground truth required in every comparison."}
    ]
  }
}
```

## Alt Text

Pipeline diagram showing figure generation from registry YAML through combo resolution and orchestration to PNG and JSON sidecar output files.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
