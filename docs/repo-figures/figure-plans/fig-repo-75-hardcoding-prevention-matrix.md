# fig-repo-75: Four Types of Hardcoding and How They're Caught

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-75 |
| **Title** | Four Types of Hardcoding and How They're Caught |
| **Complexity Level** | L3 |
| **Target Persona** | ML Engineer / Research Scientist |
| **Location** | `CONTRIBUTING.md`, `src/viz/README.md` |
| **Priority** | P2 (High) |

## Purpose

Show the four categories of hardcoding that are banned in this repository, what detection mechanism catches each type, and what the correct pattern looks like. Developers need a quick reference to avoid violations and understand why pre-commit or pytest is rejecting their code.

## Key Message

Four types of hardcoding (hex colors, literal paths, method names, dimensions) each have a specific detection mechanism and a correct replacement pattern. Check your code against all four before committing.

## Content Specification

### Panel 1: The 4-Column Prevention Matrix

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    HARDCODING PREVENTION MATRIX                                      │
├──────────────┬──────────────────────┬─────────────────────────┬─────────────────────┤
│  TYPE        │  EXAMPLE VIOLATION   │  DETECTION              │  CORRECT PATTERN    │
├──────────────┼──────────────────────┼─────────────────────────┼─────────────────────┤
│              │                      │                         │                     │
│  HEX COLORS  │  color="#006BA2"     │  Pre-commit:            │  Python:            │
│              │  fill="#E69F00"      │  r-hardcoding-check     │  COLORS["name"]     │
│              │  col="#332288"       │                         │  resolve_color()    │
│              │  (R or Python)      │  pytest:                │  get_combo_color()  │
│              │                      │  test_r_no_hex_colors   │                     │
│              │                      │                         │  R:                 │
│              │                      │                         │  load_color_        │
│              │                      │                         │  definitions()      │
├──────────────┼──────────────────────┼─────────────────────────┼─────────────────────┤
│              │                      │                         │                     │
│  LITERAL     │  "figures/           │  pytest:                │  save_figure()      │
│  PATHS       │    generated/..."    │  test_absolute_paths    │  save_publication_  │
│              │  plt.savefig(path)   │                         │    figure()         │
│              │  ggsave(path)        │  Pre-commit:            │  Config-driven      │
│              │                      │  r-hardcoding-check     │  paths from         │
│              │                      │  (catches ggsave)       │  plot_config.py     │
├──────────────┼──────────────────────┼─────────────────────────┼─────────────────────┤
│              │                      │                         │                     │
│  METHOD      │  "CatBoost" in SQL   │  pytest:                │  FIXED_CLASSIFIER   │
│  NAMES       │  outlier = "LOF"     │  test_method_           │  from plot_config   │
│              │  classifier_map =    │    abbreviations        │                     │
│              │    {"CatBoost":...}  │                         │  Load from YAML:    │
│              │                      │  pytest:                │  plot_hyperparam_   │
│              │                      │  test_python_           │    combos.yaml      │
│              │                      │    display_names        │                     │
├──────────────┼──────────────────────┼─────────────────────────┼─────────────────────┤
│              │                      │                         │                     │
│  DIMENSIONS  │  width=14            │  pytest:                │  fig_config.        │
│              │  height=6            │  test_no_hardcoded_     │    dimensions.*     │
│              │  figsize=(14, 6)     │    values               │                     │
│              │  dpi=100             │  (tests/test_guardrails)│  Figure dimensions  │
│              │                      │                         │  from figure_       │
│              │                      │                         │    registry.yaml    │
└──────────────┴──────────────────────┴─────────────────────────┴─────────────────────┘
```

### Panel 2: Self-Check Checklist

```
┌─────────────────────────────────────────────────────────────────────┐
│  SELF-CHECK: Before EVERY code block                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [ ] Any hex colors?    → Use COLORS dict / resolve_color()         │
│  [ ] Any literal paths? → Use save_figure() / config                │
│  [ ] Any method names?  → Load from YAML combos                     │
│  [ ] Any dimensions?    → Get from figure config                    │
│                                                                      │
│  If you cannot answer "NO" to all four, REVISE before committing.   │
└─────────────────────────────────────────────────────────────────────┘
```

### Panel 3: Correct Code Examples

```
PYTHON:
┌────────────────────────────────────────────────────────────────┐
│  from src.viz.plot_config import (                              │
│      setup_style, COLORS, save_figure, FIXED_CLASSIFIER,       │
│      get_combo_color                                            │
│  )                                                              │
│                                                                  │
│  setup_style()                                    # Style first │
│  color = get_combo_color("ground_truth")          # Via YAML    │
│  ax.plot(..., color=COLORS["ground_truth"])        # Dict lookup │
│  save_figure(fig, "fig_name", data=data_dict)     # Config path │
│                                                                  │
│  # Read classifier from config, not hardcoded                   │
│  query = f"SELECT * WHERE classifier = '{FIXED_CLASSIFIER}'"   │
└────────────────────────────────────────────────────────────────┘

R:
┌────────────────────────────────────────────────────────────────┐
│  source("src/r/figure_system/load_color_definitions.R")        │
│  colors <- load_color_definitions()       # From colors.yaml  │
│  theme <- theme_foundation_plr()          # Shared theme      │
│  save_publication_figure(plot, "name")    # Config-driven     │
└────────────────────────────────────────────────────────────────┘
```

### Panel 4: Historical Context

```
┌─────────────────────────────────────────────────────────────────────┐
│  WHY: CRITICAL-FAILURE-002 + CRITICAL-FAILURE-004                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  CF-002: Mixed featurization in extraction                          │
│  └── Hardcoded method names led to incorrect data filtering         │
│                                                                      │
│  CF-004: Hardcoded values everywhere                                │
│  └── Hex colors diverged between Python and R                       │
│  └── Path changes broke figure generation silently                  │
│  └── Dimension changes required editing multiple files              │
│                                                                      │
│  RESULT: All 4 types now have automated detection                   │
│  Full history: .claude/docs/meta-learnings/                         │
└─────────────────────────────────────────────────────────────────────┘
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `configs/VISUALIZATION/colors.yaml` | Color definitions (YAML source) |
| `configs/VISUALIZATION/plot_hyperparam_combos.yaml` | Combo names, method mappings, color_ref |
| `configs/VISUALIZATION/figure_registry.yaml` | Figure dimensions, output paths, privacy levels |
| `.pre-commit-config.yaml` | Hook definitions: r-hardcoding-check |

## Code Paths

| Module | Role |
|--------|------|
| `src/viz/plot_config.py` | COLORS dict, resolve_color(), get_combo_color(), FIXED_CLASSIFIER, save_figure() |
| `scripts/check_r_hardcoding.py` | Pre-commit hook: scans R files for hex colors, ggsave() |
| `tests/test_no_hardcoding/test_r_no_hex_colors.py` | pytest: R hex color detection |
| `tests/test_no_hardcoding/test_absolute_paths.py` | pytest: Literal path detection |
| `tests/test_no_hardcoding/test_method_abbreviations.py` | pytest: Hardcoded method name detection |
| `tests/test_no_hardcoding/test_python_display_names.py` | pytest: Display name consistency |
| `tests/test_guardrails/test_no_hardcoded_values.py` | pytest: Hardcoded dimension/constant detection |
| `src/r/figure_system/` | R helper functions: load_color_definitions(), save_publication_figure(), theme_foundation_plr() |

## Extension Guide

To add a new color for a new combo:
1. Add color definition to `configs/VISUALIZATION/colors.yaml` under `combo_colors:`
2. Add `color_ref` to the combo entry in `configs/VISUALIZATION/plot_hyperparam_combos.yaml`
3. Use `get_combo_color("new_combo")` in Python code
4. Use `load_color_definitions()` in R code
5. NEVER write `"#RRGGBB"` directly in source code

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "repo-75",
    "title": "Four Types of Hardcoding and How They're Caught"
  },
  "content_architecture": {
    "primary_message": "Four types of hardcoding (colors, paths, methods, dimensions) each have a detection mechanism and a correct replacement pattern.",
    "layout_flow": "4-column matrix at top, self-check below, code examples at bottom",
    "spatial_anchors": {
      "matrix": {"x": 0.5, "y": 0.3},
      "self_check": {"x": 0.5, "y": 0.6},
      "code_examples": {"x": 0.5, "y": 0.85}
    },
    "key_structures": [
      {
        "name": "Hex Colors",
        "role": "abnormal_warning",
        "is_highlighted": true,
        "labels": ["r-hardcoding-check", "test_r_no_hex"]
      },
      {
        "name": "Literal Paths",
        "role": "abnormal_warning",
        "is_highlighted": true,
        "labels": ["test_absolute_paths"]
      },
      {
        "name": "Method Names",
        "role": "abnormal_warning",
        "is_highlighted": true,
        "labels": ["test_method_abbreviations"]
      },
      {
        "name": "Dimensions",
        "role": "abnormal_warning",
        "is_highlighted": false,
        "labels": ["test_no_hardcoded_values"]
      }
    ],
    "callout_boxes": [
      {"heading": "SELF-CHECK", "body_text": "Before every code block: no hex colors, no literal paths, no method names, no hardcoded dimensions."}
    ]
  }
}
```

## Alt Text

Four-column matrix showing banned hardcoding types (colors, paths, methods, dimensions) with detection tools and correct patterns.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
