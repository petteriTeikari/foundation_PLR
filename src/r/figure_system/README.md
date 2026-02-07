# Figure System API Reference

Detailed documentation for the Foundation PLR R figure system modules.

## Overview

The figure system enforces consistent, publication-quality figures through:

1. **Config-driven parameters** - Dimensions, formats, and colors from YAML
2. **Guardrail functions** - Loud failures when rules are violated
3. **Centralized saving** - One function for all output formats

## Module: config_loader.R

Central configuration loading with validation guardrails.

### `load_figure_config(figure_id)`

Load figure configuration from `figure_layouts.yaml`.

**GUARDRAIL**: Fails loudly if figure is not defined in YAML.

```r
#' @param figure_id Character: e.g., "fig_selective_classification"
#' @return List with validated config (layout, dimensions, panels, etc.)

fig_config <- load_figure_config("fig_calibration_stratos")
# Returns: list(layout, dimensions, filename, outputs, ...)
```

### `load_figure_combos(combo_source)`

Load combo definitions from `combos.yaml`.

**GUARDRAIL**: Never hardcode combos - always load from YAML.

```r
#' @param combo_source Character: Section name in YAML
#'   - "standard_combos" - 4 standard pipeline combos
#'   - "extended_combos" - 5 extended combos
#'   - "main_4" - Preset group reference
#' @return List of combo configs with colors, names, pipeline configs

combos <- load_figure_combos("standard_combos")
# Returns: list of 4 combos with id, name, color_ref, outlier, imputation, classifier
```

### `validate_data_source(data_source, required_keys)`

Validate data file exists and has correct schema.

**GUARDRAIL**: Fails with helpful message if file missing.

```r
#' @param data_source Filename in data/r_data/ or full path
#' @param required_keys Top-level keys that must exist (default: metadata, data)
#' @return Parsed JSON data as list

data <- validate_data_source("calibration_data.json")
# Returns: list(metadata = ..., data = ...)
```

### `load_color_definitions()`

Load color definitions from YAML.

```r
#' @return Named list: CSS variable style names -> hex codes

color_defs <- load_color_definitions()
# Returns: list("--color-primary" = "#006BA2", ...)
```

### `resolve_color(color_ref, color_definitions)`

Resolve a color reference to actual hex color.

**GUARDRAIL**: Only accepts `--color-xxx` refs or hex codes (with warning).

```r
#' @param color_ref Character: e.g., "--color-ground-truth" or "#666666"
#' @return Character: hex color code

hex <- resolve_color("--color-fm-primary")
# Returns: "#932834"

# Warning issued for raw hex:
hex <- resolve_color("#AABBCC")
# Warning: Using raw hex color '#AABBCC' - prefer color_ref for consistency
```

### `get_combo_colors(combos)`

Get colors for a list of combos as a named vector.

```r
#' @param combos List of combo configs
#' @return Named character vector: combo name -> hex color

combos <- load_figure_combos("standard_combos")
colors <- get_combo_colors(combos)
# Returns: c("Ground Truth" = "#2E5B8C", "Best Ensemble" = "#932834", ...)
```

### `load_figure_all(figure_id)`

Convenience function: Load everything needed for a figure in one call.

```r
#' @param figure_id Character: figure ID
#' @return List with config, combos (if applicable), color_definitions, data

all <- load_figure_all("fig_calibration_stratos")
# Returns: list(config = ..., color_definitions = ..., combos = ..., combo_colors = ..., data = ...)
```

---

## Module: save_figure.R

Publication-quality figure export with multi-format support.

### `save_publication_figure(plot, filename, ...)`

Save a publication-quality figure in multiple formats.

**CRITICAL**: Dimensions loaded from `figure_registry.yaml` by default.
Do NOT hardcode width/height in script calls!

```r
#' @param plot ggplot or patchwork object to save
#' @param filename base filename without extension (e.g., "fig_forest_combined")
#' @param output_dir directory to save files (default: from YAML config)
#' @param width figure width in inches (default: NULL = load from registry)
#' @param height figure height in inches (default: NULL = load from registry)
#' @param dpi resolution for raster formats (default: from YAML config, typically 300)
#' @param formats character vector of formats (default: NULL = load from YAML)
#' @param device_pdf PDF device: "cairo_pdf" or "pdf" (default: "cairo_pdf")
#' @param bg background color (default: "white")
#' @return invisible list of saved file paths

# Standard usage - dimensions from registry:
save_publication_figure(p, "fig_calibration_stratos")
# Output: Loaded dimensions from registry: 7 x 5.25 inches
#         Saved: figures/generated/ggplot2/fig_calibration_stratos.png

# Override for debugging (avoid in production):
save_publication_figure(p, "fig_test", width = 10, height = 8)
```

### Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| PNG | `.png` | Raster, 300 DPI default |
| PDF | `.pdf` | Vector, cairo_pdf device |
| TIFF | `.tiff` | Raster, LZW compression |
| EPS | `.eps` | Vector |
| SVG | `.svg` | Vector |
| JPEG | `.jpg` | Raster |

### `save_from_config(plot, figure_name, ...)`

Save figure using specifications from `figure_layouts.yaml`.

```r
#' @param plot ggplot or patchwork object
#' @param figure_name name of figure in config (e.g., "fig_forest_combined")

save_from_config(p, "fig_calibration_stratos")
```

---

## Module: theme_foundation_plr.R

Economist-style ggplot2 theme with off-white background.

### Color Constants

```r
ECONOMIST_BG    <- "#FBF9F3"  # Off-white background
ECONOMIST_GRID  <- "#D4D4D4"  # Grid lines
ECONOMIST_TEXT  <- "#333333"  # Body text
ECONOMIST_TITLE <- "#000000"  # Titles (black)

ECONOMIST_COLORS <- c(
  "#E3120B",  # Economist red (primary)
  "#006BA2",  # Dark blue
  "#3EBCD2",  # Light blue/cyan
  "#379A8B",  # Teal
  "#EBB434",  # Gold/yellow
  "#932834",  # Dark red
  "#999999"   # Gray
)
```

### `theme_foundation_plr(base_size, base_family)`

Main theme function. Features:
- Off-white (#FBF9F3) background
- Horizontal grid lines only
- Bold sans-serif headings
- Left-aligned captions

```r
#' @param base_size Base font size (default 11)
#' @param base_family Font family (default "Helvetica")
#' @return A ggplot2 theme object

ggplot(data, aes(x, y)) +
  geom_point() +
  theme_foundation_plr()
```

### Theme Variants

| Function | Description | Grid Lines |
|----------|-------------|------------|
| `theme_foundation_plr()` | Default | Horizontal only |
| `theme_calibration()` | Calibration plots | Both |
| `theme_forest()` | Forest plots | Vertical only |
| `theme_heatmap()` | Heatmaps | None |
| `theme_scatter()` | Scatter plots | Both |

### `gg_add(p, ...)`

S7-compatible layer addition (ggplot2 4.0+ workaround).

```r
#' @param p A ggplot object
#' @param ... Layers to add
#' @return A ggplot object with added layers

# Instead of p + layer1 + layer2 (can fail with S7):
p <- gg_add(p, geom_point(), theme_foundation_plr())
```

### Color Scales

```r
scale_color_economist()  # Discrete color scale
scale_fill_economist()   # Discrete fill scale
```

---

## Common Patterns

### Pattern 1: Standard Figure Script

```r
# ==== MANDATORY HEADER ====
PROJECT_ROOT <- (function() {
  d <- getwd()
  while (d != dirname(d)) {
    if (file.exists(file.path(d, "CLAUDE.md"))) return(d)
    d <- dirname(d)
  }
  stop("Could not find project root")
})()

source(file.path(PROJECT_ROOT, "src/r/figure_system/config_loader.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/save_figure.R"))
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))

color_defs <- load_color_definitions()
# ==== END HEADER ====

library(ggplot2)
library(dplyr)

# Load data
data <- validate_data_source("my_figure_data.json")

# Create plot
p <- ggplot(data$data, aes(x = x, y = y)) +
  geom_point(color = color_defs[["--color-primary"]]) +
  theme_foundation_plr() +
  labs(title = "My Figure")

# Save
save_publication_figure(p, "fig_my_figure")
```

### Pattern 2: Multi-Combo Figure

```r
# Load combos and colors
combos <- load_figure_combos("standard_combos")
combo_colors <- get_combo_colors(combos)

# Create named color vector for scale_color_manual
p <- ggplot(data, aes(x, y, color = pipeline)) +
  geom_line() +
  scale_color_manual(values = combo_colors) +
  theme_foundation_plr()
```

### Pattern 3: Patchwork Composition

```r
library(patchwork)

# Create panels
p1 <- ggplot(...) + theme_foundation_plr()
p2 <- ggplot(...) + theme_foundation_plr()

# Compose
combined <- p1 | p2

# Save
save_publication_figure(combined, "fig_combined")
```

---

## Guardrail Error Messages

### "GUARDRAIL VIOLATION: Figure 'X' not found in figure_layouts.yaml"

**Cause**: Trying to create a figure not registered in YAML.

**Fix**: Add the figure to `configs/VISUALIZATION/figure_layouts.yaml` first.

### "GUARDRAIL VIOLATION: Color 'X' not found in color_definitions"

**Cause**: Using a `--color-xxx` reference that doesn't exist.

**Fix**: Add the color to `configs/VISUALIZATION/combos.yaml` color_definitions.

### "GUARDRAIL VIOLATION: Data file 'X' not found"

**Cause**: JSON data file hasn't been generated yet.

**Fix**: Run the Python export script first: `python scripts/export_XXX_for_r.py`

### "Using raw hex color '#XXXXXX' - prefer color_ref for consistency"

**Cause**: Using a hardcoded hex color instead of color_ref.

**Fix**: Replace with `color_defs[["--color-xxx"]]` or add new color to YAML.

---

## Configuration Files

| File | Purpose |
|------|---------|
| `configs/VISUALIZATION/combos.yaml` | Color definitions, combo configurations |
| `configs/VISUALIZATION/figure_layouts.yaml` | Figure dimensions, output settings |
| `configs/VISUALIZATION/figure_registry.yaml` | Figure metadata, privacy settings |

---

## Pre-commit Enforcement

The hook `scripts/check_r_hardcoding.py` checks for:

1. **Hardcoded hex colors**: `#RRGGBB` patterns → must use `color_defs`
2. **ggsave() usage**: → must use `save_publication_figure()`
3. **Custom theme functions**: → must use `theme_foundation_plr()`

Violations block commit until fixed.
