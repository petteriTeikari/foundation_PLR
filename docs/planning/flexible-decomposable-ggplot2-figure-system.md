# Flexible Decomposable ggplot2 Figure System

**Status**: PLANNING
**Created**: 2026-01-27
**Goal**: Create a production-grade system for composable, journal-ready ggplot2 figures

## Problem Statement

Current figures have hardcoded titles, subtitles, and captions that:
- Are not appropriate for academic journal submissions (captions go in figure legends)
- Cannot be easily combined into multi-panel figures (A, B, C...)
- Have inconsistent styling when composed together
- Cannot switch between "infographic" mode (with annotations) and "publication" mode (clean)

## Requirements

### R1: Infographic Mode Flag
- `infographic = FALSE` by default (publication-ready)
- When `FALSE`: No title, subtitle, or caption - clean for journal
- When `TRUE`: Full annotations for presentations/infographics

### R2: Composable Figures
- Each figure function returns a ggplot object (not saves directly)
- Figures work standalone OR as subplots
- Consistent sizing and margins when composed

### R3: Multi-Panel Layouts
- Support for arbitrary layouts: 1×2, 2×1, 2×2, 3×1, etc.
- Automatic panel labels (A, B, C...)
- Shared legends when appropriate
- Configurable via YAML

### R4: Single Source of Truth
- Layout configurations in `configs/VISUALIZATION/figure_layouts.yaml`
- Panel labels, dimensions, spacing all configurable
- No hardcoded layout parameters in figure scripts

## Architecture

### Package Choice: `patchwork`

The `patchwork` package is the de facto standard for composing ggplot2 figures:
- Clean syntax: `p1 + p2` for horizontal, `p1 / p2` for vertical
- Automatic alignment of axes
- Shared legends with `plot_layout(guides = "collect")`
- Panel tags with `plot_annotation(tag_levels = "A")`

### File Structure

```
configs/VISUALIZATION/
├── figure_layouts.yaml       # NEW: Layout configurations
└── plot_hyperparam_combos.yaml  # Existing

src/r/
├── figure_system/            # NEW: Core system
│   ├── figure_factory.R      # Factory functions for creating figures
│   ├── compose_figures.R     # Composition utilities (patchwork wrappers)
│   └── save_figure.R         # Unified save function
├── figures/                  # Existing figure scripts (refactored)
│   ├── fig02_forest_outlier.R
│   ├── fig03_forest_imputation.R
│   └── ...
└── theme_foundation_plr.R    # Existing theme
```

### API Design

#### Figure Functions (Refactored)

```r
# BEFORE: Scripts that save directly
source("fig02_forest_outlier.R")  # Saves to disk

# AFTER: Functions that return ggplot objects
create_forest_outlier <- function(
  data,
  infographic = FALSE,  # NEW: Toggle annotations
  show_legend = TRUE,   # NEW: For composed figures
  x_limits = c(0.5, 1.0)
) {
  p <- ggplot(...) + ...

  if (!infographic) {
    # Publication mode: no title/subtitle/caption
    p <- p + labs(title = NULL, subtitle = NULL, caption = NULL)
  }

  if (!show_legend) {
    p <- p + theme(legend.position = "none")
  }

  return(p)
}
```

#### Composition Functions

```r
# Compose two figures vertically (1 column, 2 rows)
compose_figures(
  list(p1, p2),
  layout = "2x1",  # or "1x2", "2x2", etc.
  tag_levels = "A",  # A, B, C...
  shared_legend = TRUE,
  infographic = FALSE
)

# From YAML configuration
compose_from_config("fig_forest_combined", infographic = FALSE)
```

#### Save Function

```r
save_publication_figure(
  plot,
  name = "fig_forest_combined",
  formats = c("pdf", "png"),
  width = 180,  # mm (journal column width)
  height = 200,
  dpi = 300
)
```

## Configuration Schema

### `configs/VISUALIZATION/figure_layouts.yaml`

```yaml
# Figure Layout Configurations
# ============================
# Defines how individual figures are composed into multi-panel figures.
# All dimensions in mm (standard for journals).

version: "1.0.0"

# Journal specifications
journal:
  column_width_mm: 89      # Single column
  page_width_mm: 183       # Full page width
  max_height_mm: 247       # Maximum height

# Default settings
defaults:
  dpi: 300
  font_family: "Arial"
  base_size: 8
  tag_size: 12
  tag_face: "bold"

# Panel tag styles
tags:
  levels: ["A", "B", "C", "D", "E", "F", "G", "H"]
  prefix: ""
  suffix: ""
  sep: ""

# Composed figure definitions
figures:
  # Forest plots combined (outlier + imputation)
  fig_forest_combined:
    description: "Combined forest plots for preprocessing methods"
    layout: "2x1"  # 2 rows, 1 column
    width_mm: 183  # Full page width
    height_mm: 200
    panels:
      - source: "create_forest_outlier"
        tag: "A"
        height_ratio: 1.2  # Taller (more methods)
      - source: "create_forest_imputation"
        tag: "B"
        height_ratio: 1.0
    shared_legend: true
    legend_position: "bottom"

  # Calibration + DCA combined
  fig_stratos_combined:
    description: "STRATOS-compliant calibration and DCA"
    layout: "1x2"  # 1 row, 2 columns
    width_mm: 183
    height_mm: 90
    panels:
      - source: "create_calibration_plot"
        tag: "A"
      - source: "create_dca_plot"
        tag: "B"
    shared_legend: true

  # 2x2 grid example
  fig_preprocessing_grid:
    description: "4-panel preprocessing overview"
    layout: "2x2"
    width_mm: 183
    height_mm: 180
    panels:
      - source: "create_forest_outlier"
        tag: "A"
      - source: "create_forest_imputation"
        tag: "B"
      - source: "create_heatmap"
        tag: "C"
      - source: "create_variance_decomp"
        tag: "D"
    shared_legend: false
```

## TDD Implementation Plan

### Test Data Fixture

```r
# tests/r/fixtures/test_data.R
setup_test_data <- function() {
  data.frame(
    outlier_method = c("Method A", "Method B", "Method C"),
    outlier_display_name = c("Method A", "Method B", "Method C"),
    auroc_mean = c(0.75, 0.80, 0.85),
    auroc_ci_lo = c(0.70, 0.75, 0.80),
    auroc_ci_hi = c(0.80, 0.85, 0.90),
    category = c("Traditional", "Foundation Model", "Ground Truth"),
    stringsAsFactors = FALSE
  )
}
```

### Cycle 1: RED - Write Failing Tests

**File**: `tests/r/test_figure_system.R`

```r
library(testthat)
source("tests/r/fixtures/test_data.R")

# ==== FIXTURES ====
test_data <- setup_test_data()

# ==== UNIT TESTS: create_forest_outlier ====
describe("create_forest_outlier", {
  it("returns ggplot object", {
    p <- create_forest_outlier(test_data)
    expect_s3_class(p, "gg")
  })

  it("removes annotations when infographic=FALSE", {
    p <- create_forest_outlier(test_data, infographic = FALSE)
    expect_null(p$labels$title)
    expect_null(p$labels$subtitle)
    expect_null(p$labels$caption)
  })

  it("includes annotations when infographic=TRUE", {
    p <- create_forest_outlier(test_data, infographic = TRUE)
    expect_false(is.null(p$labels$title))
    expect_false(is.null(p$labels$caption))
  })

  it("removes legend when show_legend=FALSE", {
    p <- create_forest_outlier(test_data, show_legend = FALSE)
    expect_equal(p$theme$legend.position, "none")
  })

  # Edge cases
  it("errors on NULL data", {
    expect_error(create_forest_outlier(NULL), "cannot be NULL")
  })

  it("errors on empty data frame", {
    expect_error(create_forest_outlier(data.frame()), "must contain")
  })
})

# ==== UNIT TESTS: compose_figures ====
describe("compose_figures", {
  p1 <- ggplot2::ggplot() + ggplot2::geom_blank()
  p2 <- ggplot2::ggplot() + ggplot2::geom_blank()

  it("creates patchwork object", {
    composed <- compose_figures(list(p1, p2), layout = "2x1")
    expect_s3_class(composed, "patchwork")
  })

  it("adds panel tags", {
    composed <- compose_figures(list(p1, p2), layout = "2x1", tag_levels = "A")
    expect_s3_class(composed, "patchwork")
  })

  # Edge cases
  it("errors on empty plot list", {
    expect_error(compose_figures(list()), "at least one plot")
  })

  it("errors on invalid layout string", {
    expect_error(compose_figures(list(p1), layout = "invalid"), "Invalid layout")
  })

  it("errors when plot count mismatches layout", {
    expect_error(compose_figures(list(p1), layout = "2x2"), "requires 4 plots")
  })

  it("errors when plots list contains non-ggplot", {
    expect_error(compose_figures(list(p1, "not a plot")), "not ggplot objects")
  })
})

# ==== UNIT TESTS: compose_from_config ====
describe("compose_from_config", {
  it("loads and composes from YAML", {
    composed <- compose_from_config("fig_forest_combined")
    expect_s3_class(composed, "patchwork")
  })

  it("errors on unknown figure name", {
    expect_error(compose_from_config("nonexistent_figure"), "not found in config")
  })
})

# ==== UNIT TESTS: save_publication_figure ====
describe("save_publication_figure", {
  p <- ggplot2::ggplot() + ggplot2::geom_blank()

  it("creates PDF and PNG files", {
    tmp_dir <- tempdir()
    save_publication_figure(p, "test_fig", output_dir = tmp_dir)
    expect_true(file.exists(file.path(tmp_dir, "test_fig.pdf")))
    expect_true(file.exists(file.path(tmp_dir, "test_fig.png")))
    # Cleanup
    unlink(file.path(tmp_dir, "test_fig.pdf"))
    unlink(file.path(tmp_dir, "test_fig.png"))
  })

  it("errors on invalid dimensions", {
    expect_error(save_publication_figure(p, "test", width = -10), "positive")
    expect_error(save_publication_figure(p, "test", width = 0), "positive")
  })

  it("warns when height exceeds journal max", {
    expect_warning(
      save_publication_figure(p, "test", height = 300, output_dir = tempdir()),
      "exceeds journal max"
    )
  })
})

# ==== CONFIG VALIDATION ====
describe("figure_layouts.yaml", {
  it("is valid YAML", {
    config <- yaml::read_yaml("configs/VISUALIZATION/figure_layouts.yaml")
    expect_true("figures" %in% names(config))
  })

  it("contains fig_forest_combined", {
    config <- yaml::read_yaml("configs/VISUALIZATION/figure_layouts.yaml")
    expect_true("fig_forest_combined" %in% names(config$figures))
  })

  it("has required fields for each figure", {
    config <- yaml::read_yaml("configs/VISUALIZATION/figure_layouts.yaml")
    for (fig_name in names(config$figures)) {
      fig <- config$figures[[fig_name]]
      expect_true("layout" %in% names(fig), info = paste(fig_name, "missing layout"))
      expect_true("panels" %in% names(fig), info = paste(fig_name, "missing panels"))
    }
  })
})
```

### Cycle 1: GREEN - Implement Core Functions

Implement minimal code to make tests pass (see API Reference below).

### Cycle 1: REFACTOR - Clean Up

Extract common patterns, improve code organization.

### Cycle 2: RED - Integration Tests

Add tests for full pipeline (YAML → compose → save).

### Cycle 2: GREEN - Wire Up Pipeline

Connect all components.

### Cycle 2: REFACTOR - Optimize

Profile and optimize if needed.

### Cycle 3: RED - Visual Regression Tests

```r
# Optional: Add vdiffr for visual regression
test_that("forest plot matches baseline", {
  p <- create_forest_outlier(test_data, infographic = FALSE)
  vdiffr::expect_doppelganger("forest-outlier-baseline", p)
})
```

### Cycle 3: GREEN - Fix Visual Regressions

### Cycle 4: Refactor Existing Figures (GREEN)

1. **`src/r/figure_system/figure_factory.R`**
   - Refactor existing figure scripts into functions
   - Add `infographic` and `show_legend` parameters
   - Return ggplot objects instead of saving

2. **`src/r/figure_system/compose_figures.R`**
   - `compose_figures()` function using patchwork
   - `compose_from_config()` to read from YAML
   - Layout parsing (2x1, 1x2, 2x2, etc.)

3. **`src/r/figure_system/save_figure.R`**
   - Unified save function with journal presets
   - Support for multiple formats
   - Automatic dimension handling

### Cycle 3: Refactor Existing Figures (GREEN)

Refactor each existing figure script:
- `fig02_forest_outlier.R` → `create_forest_outlier()` function
- `fig03_forest_imputation.R` → `create_forest_imputation()` function
- etc.

### Cycle 4: Create Configuration File (GREEN)

Create `configs/VISUALIZATION/figure_layouts.yaml` with all composed figure definitions.

### Cycle 5: Integration Testing (GREEN)

- Test full pipeline: YAML → compose → save
- Verify output dimensions match journal specs
- Visual inspection of panel alignment

## API Reference

### `create_forest_outlier()`

```r
create_forest_outlier <- function(
  data = NULL,           # If NULL, loads from default path
  infographic = FALSE,   # Include title/subtitle/caption?
  show_legend = TRUE,    # Show legend? (FALSE for composed figures)
  x_limits = c(0.5, 1.0) # X-axis limits
)
```

### `compose_figures()`

```r
compose_figures <- function(
  plots,                 # List of ggplot objects

  layout = "2x1",        # Layout string: "rows x cols"
  tag_levels = "A",      # Panel tag style: "A", "a", "1", "i"
  shared_legend = TRUE,  # Collect legends?
  legend_position = "bottom",
  widths = NULL,         # Relative column widths (optional)
  heights = NULL,        # Relative row heights (optional)
  infographic = FALSE    # Pass to individual panels
)
```

### `save_publication_figure()`

```r
save_publication_figure <- function(
  plot,                  # ggplot or patchwork object
  name,                  # Figure name (without extension)
  output_dir = "figures/generated/publication",
  formats = c("pdf", "png"),
  width = NULL,          # mm (uses config default if NULL)
  height = NULL,
  dpi = 300,
  device = cairo_pdf     # For PDF
)
```

## Usage Examples

### Example 1: Standalone Figure (Publication Mode)

```r
source("src/r/figure_system/figure_factory.R")
source("src/r/figure_system/save_figure.R")

# Create figure without annotations
p <- create_forest_outlier(infographic = FALSE)

# Save
save_publication_figure(p, "fig02_forest_outlier")
```

### Example 2: Composed Figure

```r
source("src/r/figure_system/figure_factory.R")
source("src/r/figure_system/compose_figures.R")
source("src/r/figure_system/save_figure.R")

# Create individual panels
p1 <- create_forest_outlier(infographic = FALSE, show_legend = FALSE)
p2 <- create_forest_imputation(infographic = FALSE, show_legend = FALSE)

# Compose (2 rows, shared legend)
composed <- compose_figures(
  list(p1, p2),
  layout = "2x1",
  tag_levels = "A",
  shared_legend = TRUE
)

# Save
save_publication_figure(composed, "fig_forest_combined", width = 183, height = 200)
```

### Example 3: From YAML Configuration

```r
source("src/r/figure_system/compose_figures.R")
source("src/r/figure_system/save_figure.R")

# Load and compose from configuration
composed <- compose_from_config("fig_forest_combined", infographic = FALSE)

# Save with dimensions from config
save_from_config("fig_forest_combined")
```

### Example 4: Infographic Mode (Presentation)

```r
# Same figure with full annotations
p <- create_forest_outlier(infographic = TRUE)
save_publication_figure(p, "fig02_forest_outlier_infographic")
```

## Verification Commands

```bash
# Run all figure system tests
Rscript -e "testthat::test_file('tests/r/test_figure_system.R')"

# Generate combined forest plot
Rscript -e "source('src/r/figure_system/generate_composed.R'); generate('fig_forest_combined')"

# Compare publication vs infographic
Rscript -e "source('src/r/compare_modes.R')"
```

## Success Criteria

- [ ] `figure_layouts.yaml` exists with all composed figure definitions
- [ ] All existing figures refactored to return ggplot objects
- [ ] `infographic` parameter works on all figure functions
- [ ] `compose_figures()` creates valid patchwork objects
- [ ] Panel tags (A, B, C) appear correctly
- [ ] Shared legends work when specified
- [ ] Output dimensions match journal specifications
- [ ] All tests pass
- [ ] Example combined figure generates correctly

## Open Questions

1. **Legend consolidation**: When combining figures with different color scales, how to handle?
   - Option A: Keep separate legends
   - Option B: Unified legend with all categories
   - **Proposed**: Per-figure config flag `shared_legend: true/false`

2. **Axis alignment**: Should x-axes be aligned across panels?
   - Option A: Always align (may waste space)
   - Option B: Panel-specific limits
   - **Proposed**: Per-panel `align_axes: true/false` in config

3. **Caption handling**: Where do captions go in composed figures?
   - Option A: Below each panel (cluttered)
   - Option B: Single caption below composed figure
   - **Proposed**: Caption in figure legend (LaTeX), not in figure

## Error Handling Specification

### Input Validation

```r
# compose_figures() must validate:
compose_figures <- function(plots, layout = "2x1", ...) {
  # 1. Check plots is a list

if (!is.list(plots)) {
    stop("'plots' must be a list of ggplot objects", call. = FALSE)
  }

  # 2. Check list is not empty
  if (length(plots) == 0) {
    stop("'plots' list must contain at least one plot", call. = FALSE)
  }

  # 3. Check all elements are ggplot objects
  are_ggplots <- vapply(plots, inherits, logical(1), "gg")
  if (!all(are_ggplots)) {
    bad_idx <- which(!are_ggplots)
    stop(sprintf(
      "Elements at positions %s are not ggplot objects",
      paste(bad_idx, collapse = ", ")
    ), call. = FALSE)
  }

  # 4. Validate layout string format
  layout_parts <- strsplit(layout, "x")[[1]]
  if (length(layout_parts) != 2 || any(is.na(as.integer(layout_parts)))) {
    stop("Invalid layout: must be 'rows x cols' (e.g., '2x1')", call. = FALSE)
  }

  # 5. Validate plot count matches layout
  n_panels <- as.integer(layout_parts[1]) * as.integer(layout_parts[2])
  if (length(plots) != n_panels) {
    stop(sprintf(
      "Layout '%s' requires %d plots, but %d provided",
      layout, n_panels, length(plots)
    ), call. = FALSE)
  }
}

# create_forest_outlier() must validate:
create_forest_outlier <- function(data, ...) {
  if (is.null(data)) {
    stop("Data cannot be NULL", call. = FALSE)
  }
  if (nrow(data) == 0) {
    stop("Data must contain at least one row", call. = FALSE)
  }
  required_cols <- c("outlier_display_name", "auroc_mean", "auroc_ci_lo", "auroc_ci_hi")
  missing <- setdiff(required_cols, names(data))
  if (length(missing) > 0) {
    stop(sprintf("Data missing required columns: %s", paste(missing, collapse = ", ")), call. = FALSE)
  }
}

# save_publication_figure() must validate:
save_publication_figure <- function(plot, name, width = NULL, height = NULL, ...) {
  if (!is.null(width) && width <= 0) {
    stop("'width' must be positive", call. = FALSE)
  }
  if (!is.null(height) && height <= 0) {
    stop("'height' must be positive", call. = FALSE)
  }
  # Warn if exceeding journal limits
  config <- yaml::read_yaml("configs/VISUALIZATION/figure_layouts.yaml")
  max_h <- config$journal$max_height_mm
  if (!is.null(height) && height > max_h) {
    warning(sprintf(
      "Height %.0fmm exceeds journal max (%.0fmm). Consider splitting figure.",
      height, max_h
    ))
  }
}
```

### Config Validation

```r
validate_figure_config <- function(config_name) {
  config <- yaml::read_yaml("configs/VISUALIZATION/figure_layouts.yaml")

  if (!(config_name %in% names(config$figures))) {
    available <- paste(names(config$figures), collapse = ", ")
    stop(sprintf(
      "Figure '%s' not found in config. Available: %s",
      config_name, available
    ), call. = FALSE)
  }

  fig_config <- config$figures[[config_name]]
  required_fields <- c("layout", "panels")
  missing <- setdiff(required_fields, names(fig_config))
  if (length(missing) > 0) {
    stop(sprintf(
      "Figure '%s' missing required fields: %s",
      config_name, paste(missing, collapse = ", ")
    ), call. = FALSE)
  }

  invisible(TRUE)
}
```

## R Gotchas and Best Practices

### Font Embedding for PDF

```r
# GOTCHA: Cairo requires X11 on Linux
# Use this for headless environments:
Sys.setenv(DISPLAY = "")

# Alternative: Use ragg for better cross-platform font handling
save_publication_figure <- function(..., device = NULL) {
  if (is.null(device)) {
    # Prefer ragg if available, fall back to cairo
    device <- if (requireNamespace("ragg", quietly = TRUE)) {
      ragg::agg_pdf
    } else {
      cairo_pdf
    }
  }
  # ...
}
```

### Non-Standard Evaluation (NSE) in ggplot2

```r
# GOTCHA: Dynamic column names require .data pronoun
create_forest_outlier <- function(data, x_col = "auroc_mean", y_col = "outlier_display_name") {
  # WRONG: ggplot(data, aes(x = x_col, y = y_col))
  # RIGHT: Use .data pronoun from rlang
  ggplot(data, aes(x = .data[[x_col]], y = .data[[y_col]]))
}
```

### Theme Modification Order

```r
# GOTCHA: theme() calls are additive but complex elements don't merge
# Solution: Build complete theme, then add figure-specific overrides

create_forest_outlier <- function(data, infographic = FALSE, show_legend = TRUE, ...) {
  p <- ggplot(data, ...) +
    theme_foundation_plr()  # Base theme first

  # Then apply conditional modifications
  if (!show_legend) {
    p <- p + theme(legend.position = "none")
  }

  if (!infographic) {
    p <- p + labs(title = NULL, subtitle = NULL, caption = NULL)
  }

  p
}
```

### Patchwork vs ggplot Object Classes

```r
# GOTCHA: Composed figures are "patchwork", not "ggplot"
p1 + p2  # Returns class "patchwork"

# Most ggplot2 methods work, but some don't
# ggsave() works with patchwork objects
# Some theme modifications may need wrap_elements()
```

### Unit Consistency

```r
# Always use mm for journal figures
# Convert explicitly when needed:
width_inches <- width_mm / 25.4
height_inches <- height_mm / 25.4
```

## Dependencies

```r
# Required packages
install.packages(c(
  "patchwork",    # Figure composition (>= 1.2.0)
  "ggplot2",      # Base plotting (>= 3.4.0)
  "yaml",         # Config parsing (>= 2.3.0)
  "testthat",     # Testing
  "ragg"          # Optional: Better PDF rendering
))

# Optional for visual regression testing:
install.packages("vdiffr")
```

## RESOLVED DECISIONS

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Composition package | `patchwork` | De facto standard, clean syntax, good alignment |
| Default mode | `infographic = FALSE` | Publication is primary use case |
| Tag style | "A", "B", "C" | Journal convention |
| Dimensions | mm | SI units, journal standard |
| Config location | `configs/VISUALIZATION/figure_layouts.yaml` | Consistent with existing config structure |
| Caption handling | In LaTeX, not figure | Journal convention - figure legends are separate |
| DPI | 300 | Standard for publication; 600 for line art if needed |
| Color palette | Colorblind-safe | Accessibility requirement (use existing `scale_color_tol()`) |
| PDF device | `cairo_pdf` (or `ragg::agg_pdf`) | Font embedding, cross-platform support |
| Output formats | PDF + PNG | PDF for submission, PNG for previews |

## Accessibility Considerations

- **Colorblind-safe palettes**: Use existing `scale_color_tol()` from `color_palettes.R`
- **Sufficient contrast**: Minimum 4.5:1 contrast ratio for text
- **Pattern alternatives**: Consider adding patterns/shapes for key distinctions
- **Font size**: Minimum 8pt for legibility after figure scaling
