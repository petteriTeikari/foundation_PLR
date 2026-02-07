# Foundation PLR Color Palettes
# Economist-style colorblind-safe palette
# Created: 2026-01-25
#
# Primary: Economist red (#E3120B) for emphasis
# Secondary: Blues and teals for comparison
# Reference: https://altaf-ali.github.io/ggplot_tutorial/challenge.html

library(ggplot2)

# ==============================================================================
# ECONOMIST-STYLE COLOR DEFINITIONS
# ==============================================================================

# Primary Economist palette (colorblind-safe)
ECONOMIST_PALETTE <- c(
  "#E3120B",  # Economist red (primary emphasis)
  "#006BA2",  # Dark blue
  "#3EBCD2",  # Light blue/cyan
  "#379A8B",  # Teal
  "#EBB434",  # Gold
  "#932834",  # Dark red
  "#999999"   # Gray
)

# Physiological colors (for Blue/Red PLR stimuli)
# Using Economist-compatible colors
COLORS_WAVELENGTH <- c(
  "Blue (469nm)" = "#006BA2",
  "Red (640nm)" = "#E3120B",
  "Blue" = "#006BA2",
  "Red" = "#E3120B"
)

# Pipeline method colors
COLORS_PIPELINE <- c(
  "Ground truth" = "#FFD700",      # Gold
  "Ensemble" = "#4682B4",          # Steelblue
  "MOMENT" = "#2ca02c",            # Green
  "TimesNet" = "#9467bd",          # Purple
  "UniTS" = "#8c564b",             # Brown
  "SAITS" = "#e377c2",             # Pink
  "CSDI" = "#7f7f7f",              # Gray
  "Traditional" = "#bcbd22",       # Olive
  "LOF" = "#17becf",               # Cyan
  "Linear" = "#aaaaaa"             # Light gray
)

# Paul Tol's colorblind-safe palette (qualitative)
# Reference: https://personal.sron.nl/~pault/
COLORS_TOL_QUALITATIVE <- c(
  "#332288",  # Dark blue

  "#117733",  # Green
  "#44AA99",  # Teal
  "#88CCEE",  # Cyan
  "#DDCC77",  # Sand
  "#CC6677",  # Rose
  "#AA4499",  # Purple
  "#882255"   # Wine
)

# Tol bright palette (for main comparisons)
COLORS_TOL_BRIGHT <- c(
  "#4477AA",  # Blue
  "#EE6677",  # Red
  "#228833",  # Green
  "#CCBB44",  # Yellow
  "#66CCEE",  # Cyan
  "#AA3377",  # Purple
  "#BBBBBB"   # Grey
)

# VIF concern levels
COLORS_VIF <- c(
  "OK" = "#4477AA",         # Blue (VIF < 5)
  "Moderate" = "#CCBB44",   # Yellow (5 <= VIF < 10)
  "High" = "#EE6677"        # Red (VIF >= 10)
)

# STRATOS outcome classes
COLORS_OUTCOME <- c(
  "Control" = "#4477AA",    # Blue
  "Glaucoma" = "#EE6677"    # Red
)

# Pipeline type colors (for CD diagrams, raincloud, etc.)
# These map to method CATEGORIES, not individual methods
# 5 categories: Ground Truth, Ensemble, Foundation Model, Deep Learning, Traditional
COLORS_PIPELINE_TYPE <- c(
  "Ground Truth" = "#FFD700",      # Gold - reference baseline
  "Ensemble" = "#4477AA",          # Blue - combined methods
  "Foundation Model" = "#228833",  # Green - FM-based methods
  "Deep Learning" = "#9467bd",     # Purple - deep learning (non-FM)
  "Traditional" = "#BBBBBB"        # Gray - classical methods
)

# Stimulus light colors (actual wavelength approximations for PLR figures)
COLORS_STIMULUS <- c(
  "blue_480nm" = "#0072B2",  # 480nm blue light (close to Economist dark blue)
  "red_640nm" = "#FF0000"    # 640nm red light (pure red, distinct from Economist branding)
)

# Panel background tints (subtle context shading for decomposition grid)
COLORS_PANEL_BG <- c(
  "ground_truth" = "#F0EDE6",  # Slightly warmer/darker for ground truth panels
  "default" = "#FBF9F3"        # Economist off-white for all other panels
)

# Semantic annotation colors (for reference lines, text, etc.)
COLOR_REFERENCE_LINE <- "#999999"
COLOR_ANNOTATION_TEXT <- "#333333"
COLOR_ANNOTATION_SECONDARY <- "#666666"  # Lighter annotation text
COLOR_EMPHASIS <- "#E3120B"        # Economist red for critical annotations

# ==============================================================================
# GGPLOT2 SCALE FUNCTIONS
# ==============================================================================

#' Scale for wavelength (Blue/Red stimuli)
#' @export
scale_color_wavelength <- function(...) {
  scale_color_manual(values = COLORS_WAVELENGTH, ...)
}

#' Scale fill for wavelength
#' @export
scale_fill_wavelength <- function(...) {
  scale_fill_manual(values = COLORS_WAVELENGTH, ...)
}

#' Scale for pipeline methods
#' @export
scale_color_pipeline <- function(...) {
  scale_color_manual(values = COLORS_PIPELINE, ...)
}

#' Scale fill for pipeline methods
#' @export
scale_fill_pipeline <- function(...) {
  scale_fill_manual(values = COLORS_PIPELINE, ...)
}

#' Paul Tol qualitative scale
#' @export
scale_color_tol <- function(...) {
  scale_color_manual(values = COLORS_TOL_QUALITATIVE, ...)
}

#' Paul Tol qualitative fill
#' @export
scale_fill_tol <- function(...) {
  scale_fill_manual(values = COLORS_TOL_QUALITATIVE, ...)
}

#' Tol bright scale (for 4-6 categories)
#' @export
scale_color_tol_bright <- function(...) {
  scale_color_manual(values = COLORS_TOL_BRIGHT, ...)
}

#' Scale for VIF concern levels
#' @export
scale_fill_vif <- function(...) {
  scale_fill_manual(values = COLORS_VIF, ...)
}

#' Scale for outcome classes
#' @export
scale_color_outcome <- function(...) {
  scale_color_manual(values = COLORS_OUTCOME, ...)
}

#' Scale fill for outcome classes
#' @export
scale_fill_outcome <- function(...) {
  scale_fill_manual(values = COLORS_OUTCOME, ...)
}

#' Scale color for pipeline types (Ground Truth, Ensemble, FM, Traditional)
#' @export
scale_color_pipeline_type <- function(...) {
  scale_color_manual(values = COLORS_PIPELINE_TYPE, ...)
}

#' Scale fill for pipeline types
#' @export
scale_fill_pipeline_type <- function(...) {
  scale_fill_manual(values = COLORS_PIPELINE_TYPE, ...)
}

# ==============================================================================
# CONTINUOUS COLOR SCALES (viridis only)
# ==============================================================================

#' Viridis continuous scale for heatmaps
#' @export
scale_fill_auroc <- function(...) {
  scale_fill_viridis_c(
    option = "viridis",
    limits = c(0.75, 0.95),
    oob = scales::squish,
    ...
  )
}

#' Viridis diverging scale for SHAP
#' @export
scale_color_shap <- function(...) {
  scale_color_gradient2(
    low = COLORS_TOL_BRIGHT[1],   # Blue (negative SHAP)
    mid = "grey95",
    high = COLORS_TOL_BRIGHT[2],  # Red (positive SHAP)
    midpoint = 0,
    ...
  )
}

# ==============================================================================
# COLORBLIND ACCESSIBILITY
# ==============================================================================

#' Test a plot for colorblind accessibility
#' Requires: install.packages("colorblindr")
#' @export
test_colorblind <- function(p) {
  if (!requireNamespace("colorblindr", quietly = TRUE)) {
    stop("Install colorblindr: install.packages('colorblindr')")
  }
  colorblindr::cvd_grid(p)
}

# ==============================================================================
# PRESETS FOR STANDARD FIGURE TYPES
# ==============================================================================

# Standard 4 combos for main figures
STANDARD_COMBOS <- c(
  "ground_truth" = "Ground truth + Ground truth",
  "best_ensemble" = "Ensemble + CSDI",
  "best_single_fm" = "MOMENT-gt-finetune + SAITS",
  "traditional" = "LOF + SAITS"
)

# Colors for standard 4 combos
COLORS_STANDARD_COMBOS <- c(
  "ground_truth" = "#FFD700",    # Gold
  "best_ensemble" = "#4477AA",   # Blue
  "best_single_fm" = "#228833",  # Green
  "traditional" = "#BBBBBB"      # Grey
)

#' Scale for standard 4-combo comparisons
#' @export
scale_color_standard_combos <- function(...) {
  scale_color_manual(values = COLORS_STANDARD_COMBOS, ...)
}

#' Scale fill for standard 4-combo comparisons
#' @export
scale_fill_standard_combos <- function(...) {
  scale_fill_manual(values = COLORS_STANDARD_COMBOS, ...)
}

message("Foundation PLR color palettes loaded.")
message("Use scale_color_pipeline(), scale_fill_tol(), etc.")
