# fig-repo-76: From YAML to Pixel: How Colors Are Resolved

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-76 |
| **Title** | From YAML to Pixel: How Colors Are Resolved |
| **Complexity Level** | L4 |
| **Target Persona** | ML Engineer |
| **Location** | `configs/VISUALIZATION/README.md`, `src/viz/README.md` |
| **Priority** | P3 (Medium) |

## Purpose

Trace the complete color resolution path from YAML definition to rendered pixel, showing both the Python and R pipelines. Developers need to understand which file is authoritative for which color, how `resolve_color()` and `get_combo_color()` work, and why the dual Python/R source of truth requires synchronization.

## Key Message

Colors flow from configs/VISUALIZATION/colors.yaml through language-specific resolution functions (Python: get_combo_color(), R: load_color_definitions()) to final rendering. The combo YAML provides color_ref identifiers that resolve to hex values.

## Content Specification

### Panel 1: The Color Data Flow (Python Path)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PYTHON COLOR RESOLUTION                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  SOURCE: configs/VISUALIZATION/                                         │
│  ┌─────────────────────────────────┐  ┌──────────────────────────────┐  │
│  │ colors.yaml                      │  │ plot_hyperparam_combos.yaml  │  │
│  │ ├── ground_truth: "#666666"      │  │ ├── standard_combos:         │  │
│  │ ├── fm_primary: "#0072B2"        │  │ │   ├── ground_truth:        │  │
│  │ ├── traditional: "#E69F00"       │  │ │   │   color_ref:           │  │
│  │ ├── ensemble: "#882255"          │  │ │   │     "--color-gt"       │  │
│  │ │                                │  │ │   ├── best_ensemble:       │  │
│  │ ├── combo_colors:                │  │ │   │   color_ref:           │  │
│  │ │   ground_truth: "#666666"      │  │ │   │     "--color-fm"      │  │
│  │ │   best_ensemble: "#0072B2"     │  │ │   └── ...                  │  │
│  │ │   best_single_fm: "#56B4E9"    │  │ └── color_definitions:       │  │
│  │ │   traditional: "#E69F00"       │  │     --color-gt: "#666666"    │  │
│  │ └─────────────────────────────── │  │     --color-fm: "#0072B2"    │  │
│  └─────────────────────────────────┘  └──────────────────────────────┘  │
│                    │                              │                       │
│                    ▼                              ▼                       │
│  RESOLVER: src/viz/plot_config.py                                       │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                                                                    │   │
│  │  COLORS = {                         ← Canonical Python dict       │   │
│  │    "ground_truth": "#666666",                                      │   │
│  │    "best_ensemble": "#0072B2",                                     │   │
│  │    "best_single_fm": "#56B4E9",                                    │   │
│  │    "traditional": "#E69F00",                                       │   │
│  │    ...                                                             │   │
│  │  }                                                                 │   │
│  │                                                                    │   │
│  │  get_combo_color("ground_truth")   ← Resolves via combos YAML    │   │
│  │  → Loads plot_hyperparam_combos.yaml                              │   │
│  │  → Finds color_ref: "--color-gt"                                  │   │
│  │  → Looks up "--color-gt" in color_definitions                     │   │
│  │  → Returns "#666666"                                              │   │
│  │                                                                    │   │
│  │  ECONOMIST_PALETTE = [...]          ← Fallback color cycle        │   │
│  │                                                                    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                    │                                                      │
│                    ▼                                                      │
│  CONSUMER: matplotlib                                                   │
│  ax.plot(..., color=COLORS["ground_truth"])                             │
│  ax.plot(..., color=get_combo_color("best_ensemble"))                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 2: The Color Data Flow (R Path)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    R COLOR RESOLUTION                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  SOURCE: configs/VISUALIZATION/colors.yaml   (SAME file as Python)      │
│                    │                                                      │
│                    ▼                                                      │
│  RESOLVER: src/r/figure_system/                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  load_color_definitions()                                         │   │
│  │  → Reads colors.yaml with yaml::yaml.load_file()                 │   │
│  │  → Returns named list: list(ground_truth="#666666", ...)          │   │
│  │                                                                    │   │
│  │  theme_foundation_plr()                                           │   │
│  │  → Shared ggplot2 theme (consistent styling)                      │   │
│  │                                                                    │   │
│  │  save_publication_figure(plot, "name")                            │   │
│  │  → Config-driven output path + DPI                                │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                    │                                                      │
│                    ▼                                                      │
│  CONSUMER: ggplot2                                                      │
│  scale_color_manual(values = colors)                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 3: Dual Source of Truth Awareness

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ARCHITECTURE NOTE: Dual Source of Truth                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Python: COLORS dict in plot_config.py      ← Canonical for Python     │
│  R:      colors.yaml via load_color_defs()  ← Canonical for R          │
│                                                                          │
│  RISK: These can drift if someone edits one but not the other.          │
│                                                                          │
│  MITIGATION:                                                             │
│  ├── colors.yaml has cross-reference comments to plot_config.py         │
│  ├── test_yaml_single_source.py checks consistency                      │
│  ├── r-hardcoding-check prevents R files from bypassing the system     │
│  └── combo_colors in colors.yaml mirrors COLORS dict values            │
│                                                                          │
│  RULE: Update BOTH when changing a color.                               │
│  Step 1: Update colors.yaml (affects R)                                 │
│  Step 2: Update COLORS dict in plot_config.py (affects Python)          │
│  Step 3: Tests verify consistency                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 4: Key Functions Reference

```
┌─────────────────────────────────────────────────────────────────────────┐
│  FUNCTION REFERENCE                                                      │
├──────────────────────────┬──────────────────────────────────────────────┤
│  Function                │  Usage                                       │
├──────────────────────────┼──────────────────────────────────────────────┤
│  COLORS["name"]          │  Direct dict lookup (fastest, Python)       │
│  get_combo_color(id)     │  Resolves via combos YAML color_ref         │
│  resolve_color(ref)      │  Resolves "--color-ref" to hex             │
│  ECONOMIST_PALETTE       │  Fallback color cycle for ad-hoc plots     │
│  load_color_definitions()│  R: loads colors.yaml into named list      │
│  theme_foundation_plr()  │  R: shared ggplot2 theme                   │
└──────────────────────────┴──────────────────────────────────────────────┘
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `configs/VISUALIZATION/colors.yaml` | Color definitions: method colors, combo_colors, palette |
| `configs/VISUALIZATION/plot_hyperparam_combos.yaml` | Combo color_ref mappings, color_definitions dict |

## Code Paths

| Module | Role |
|--------|------|
| `src/viz/plot_config.py` | COLORS dict, get_combo_color(), resolve_color(), ECONOMIST_PALETTE |
| `src/r/figure_system/` | load_color_definitions(), theme_foundation_plr(), save_publication_figure() |
| `scripts/check_r_hardcoding.py` | Pre-commit: prevents hex colors in R code |
| `tests/test_no_hardcoding/test_r_no_hex_colors.py` | pytest: ensures R uses YAML colors |
| `tests/test_no_hardcoding/test_yaml_single_source.py` | pytest: checks Python/R color consistency |

## Extension Guide

To add a new semantic color:
1. Add entry to `configs/VISUALIZATION/colors.yaml` (e.g., `new_category: "#AABBCC"`)
2. If combo-specific, add to `combo_colors:` section in the same file
3. Add to `COLORS` dict in `src/viz/plot_config.py`
4. If used in R, `load_color_definitions()` picks it up automatically from YAML
5. If combo-specific, add `color_ref` to `configs/VISUALIZATION/plot_hyperparam_combos.yaml`
6. Use `get_combo_color("new_combo")` in Python, `colors$new_combo` in R
7. NEVER write `"#AABBCC"` directly in source code

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "repo-76",
    "title": "From YAML to Pixel: How Colors Are Resolved"
  },
  "content_architecture": {
    "primary_message": "Colors flow from YAML config through language-specific resolvers (Python COLORS dict, R load_color_definitions) to rendered figures.",
    "layout_flow": "Top-down: YAML sources at top, Python resolver in middle, R parallel path on right, consumer at bottom",
    "spatial_anchors": {
      "yaml_sources": {"x": 0.5, "y": 0.15},
      "python_resolver": {"x": 0.35, "y": 0.45},
      "r_resolver": {"x": 0.75, "y": 0.45},
      "consumers": {"x": 0.5, "y": 0.75},
      "dual_source_note": {"x": 0.5, "y": 0.9}
    },
    "key_structures": [
      {
        "name": "colors.yaml",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Color definitions", "combo_colors"]
      },
      {
        "name": "plot_hyperparam_combos.yaml",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["color_ref mappings"]
      },
      {
        "name": "COLORS dict",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Python canonical source"]
      },
      {
        "name": "get_combo_color()",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["Resolves via YAML"]
      },
      {
        "name": "load_color_definitions()",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["R canonical source"]
      }
    ],
    "callout_boxes": [
      {"heading": "DUAL SOURCE", "body_text": "Python COLORS dict and R colors.yaml must stay synchronized. Tests verify consistency."}
    ]
  }
}
```

## Alt Text

Data flow diagram tracing color values from YAML config files through Python and R resolver functions to matplotlib and ggplot2 rendering.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
