#!/usr/bin/env Rscript
# ==============================================================================
# pminternal Prediction Instability Plot (5-panel, 2×3 grid)
# ==============================================================================
#
# Riley 2023-style prediction instability visualization showing how individual
# predictions vary across bootstrap samples.
#
# Layout (2 rows × 3 columns):
#   Row 1: Ground Truth | Ensemble FM | Single-model FM
#   Row 2: (empty)      | Deep Learning | Traditional
#
# X-axis: Predicted risk from developed model (mean across bootstraps)
# Y-axis: Predicted risk from bootstrap models
# Each point: One bootstrap prediction for one patient
# Ribbon: 95% CI of bootstrap predictions
# Diagonal: Perfect stability (y = x)
#
# IMPORTANT: This script follows anti-hardcoding rules (CRITICAL-FAILURE-004)
# - NO hardcoded hex colors (uses color_defs from YAML)
# - NO custom theme definitions (uses theme_foundation_plr)
# - Uses save_publication_figure() from figure system
#
# Reference:
#   Riley RD et al. (2023) "Clinical prediction models and the multiverse of madness"
#   BMC Medicine 21:502. DOI: 10.1186/s12916-023-02849-7
#
# Data Source: data/r_data/pminternal_bootstrap_predictions.json
# ==============================================================================

# ==============================================================================
# MANDATORY HEADER - DO NOT REMOVE
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(jsonlite)
})

# Find project root
PROJECT_ROOT <- (function() {
  markers <- c("pyproject.toml", "CLAUDE.md", ".git")
  dir <- getwd()
  while (dir != dirname(dir)) {
    if (any(file.exists(file.path(dir, markers)))) return(dir)
    dir <- dirname(dir)
  }
  stop("Could not find project root")
})()

# Source figure system (MANDATORY)
source(file.path(PROJECT_ROOT, "src/r/figure_system/config_loader.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/save_figure.R"))
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))

# Load colors from YAML (MANDATORY - no hardcoded colors!)
color_defs <- load_color_definitions()

# ==============================================================================
# END MANDATORY HEADER
# ==============================================================================

# ==============================================================================
# Load Data
# ==============================================================================
message("[fig_instability_combined] Loading data...")

json_path <- file.path(PROJECT_ROOT, "data/r_data/pminternal_bootstrap_predictions.json")
if (!file.exists(json_path)) {
  stop("Data file not found: ", json_path, "\nRun scripts/extract_pminternal_data.py first.")
}

data <- fromJSON(json_path, simplifyVector = FALSE)
message("  Loaded ", length(data$configs), " configs")
message("  Configs: ", paste(names(data$configs), collapse = ", "))

# ==============================================================================
# Create Instability Plot Function
# ==============================================================================

#' Create a single instability plot panel
#'
#' @param config_data List with y_prob_original, y_prob_bootstrap, n_patients, n_bootstrap
#' @param title Panel title (e.g., "A  Ground Truth")
#' @param color_accent Accent color for CI ribbon (from color_defs)
#' @param sample_bootstraps Number of bootstrap samples to plot (for speed)
#' @return ggplot object
create_instability_panel <- function(config_data, title, color_accent = NULL, sample_bootstraps = 200) {
  # Default to primary color from YAML if not specified
  if (is.null(color_accent)) {
    color_accent <- color_defs[["--color-primary"]]
  }

  # Extract data
  y_original <- unlist(config_data$y_prob_original)  # Mean prediction per patient
  y_true <- unlist(config_data$y_true)
  n_patients <- config_data$n_patients
  n_bootstrap <- config_data$n_bootstrap

  # Create numeric matrix from list of lists (JSON nested arrays)
  # Shape: (n_bootstrap x n_patients)
  y_bootstrap <- matrix(
    unlist(config_data$y_prob_bootstrap),
    nrow = n_bootstrap,
    ncol = n_patients,
    byrow = TRUE
  )

  message("    Panel data: ", n_patients, " patients × ", n_bootstrap, " bootstraps")

  # Sample bootstraps for visualization (full 1000 is slow to render)
  if (n_bootstrap > sample_bootstraps) {
    sample_idx <- sort(sample(1:n_bootstrap, sample_bootstraps))
    y_bootstrap_sampled <- y_bootstrap[sample_idx, ]
  } else {
    y_bootstrap_sampled <- y_bootstrap
  }

  # Compute 95% CI per patient
  ci_lo <- apply(y_bootstrap, 2, quantile, probs = 0.025)
  ci_hi <- apply(y_bootstrap, 2, quantile, probs = 0.975)

  # Create long-format data for scatter points
  df_points <- expand.grid(
    patient = 1:n_patients,
    bootstrap = 1:nrow(y_bootstrap_sampled)
  )
  df_points$original <- y_original[df_points$patient]
  df_points$predicted <- sapply(1:nrow(df_points), function(i) {
    y_bootstrap_sampled[df_points$bootstrap[i], df_points$patient[i]]
  })

  # Create CI ribbon data (sorted by original)
  df_ci <- data.frame(
    original = y_original,
    ci_lo = ci_lo,
    ci_hi = ci_hi,
    y_true = y_true
  ) %>%
    arrange(original)

  # Calculate instability metric (95% CI width)
  ci_width <- mean(ci_hi - ci_lo)

  # Create plot
  p <- ggplot() +
    # Diagonal reference line (perfect stability)
    geom_abline(
      slope = 1, intercept = 0,
      color = color_defs[["--color-text-primary"]], linewidth = 0.5, linetype = "solid"
    ) +
    # 95% CI ribbon (sorted by original prediction)
    geom_ribbon(
      data = df_ci,
      aes(x = original, ymin = ci_lo, ymax = ci_hi),
      fill = color_accent, alpha = 0.2
    ) +
    # Bootstrap prediction points (jittered for visibility)
    geom_point(
      data = df_points,
      aes(x = original, y = predicted),
      alpha = 0.03, size = 0.4, color = color_defs[["--color-text-secondary"]]
    ) +
    # Mean prediction markers (colored by true label)
    geom_point(
      data = df_ci,
      aes(x = original, y = original, color = factor(y_true)),
      size = 1.5, alpha = 0.8
    ) +
    # Scales
    scale_x_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, 0.2),
      labels = function(x) sprintf("%.1f", x)
    ) +
    scale_y_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, 0.2),
      labels = function(x) sprintf("%.1f", x)
    ) +
    scale_color_manual(
      values = c("0" = color_defs[["--color-fm-primary"]], "1" = color_defs[["--color-negative"]]),
      labels = c("0" = "Control", "1" = "Glaucoma"),
      name = "True Class"
    ) +
    # Labels
    labs(
      title = title,
      x = "Predicted risk (developed model)",
      y = "Predicted risk (bootstrap models)"
    ) +
    # Annotation with CI width only (MAPE removed - was computed incorrectly)
    annotate(
      "text",
      x = 0.02, y = 0.98,
      label = sprintf("95%% CI width: %.3f", ci_width),
      hjust = 0, vjust = 1,
      size = 3, color = color_defs[["--color-text-primary"]]
    ) +
    # Theme
    theme_foundation_plr() +
    theme(
      legend.position = "bottom",
      legend.key.size = unit(0.8, "lines"),
      plot.title = element_text(size = 11, face = "bold")
    )

  return(p)
}

# ==============================================================================
# Create Figure (2×3 grid)
# ==============================================================================
message("[fig_instability_combined] Creating panels...")

# Consistent CI ribbon color for all panels (easier visual comparison)
ci_ribbon_color <- color_defs[["--color-fm-primary"]]  # Blue for all

# Row 1, Col 1: Ground Truth
message("  Creating panel: Ground Truth...")
p_gt <- create_instability_panel(
  config_data = data$configs$ground_truth,
  title = "A  Ground Truth",
  color_accent = ci_ribbon_color
)

# Row 1, Col 2: Ensemble FM
if (!is.null(data$configs$best_ensemble)) {
  message("  Creating panel: Ensemble FM...")
  p_ensemble <- create_instability_panel(
    config_data = data$configs$best_ensemble,
    title = "B  Ensemble FM",
    color_accent = ci_ribbon_color
  )
} else {
  stop("Missing required config 'best_ensemble'")
}

# Row 1, Col 3: Single-model FM
if (!is.null(data$configs$best_single_fm)) {
  message("  Creating panel: Single-model FM...")
  p_single_fm <- create_instability_panel(
    config_data = data$configs$best_single_fm,
    title = "C  Single-model FM",
    color_accent = ci_ribbon_color
  )
} else {
  stop("Missing required config 'best_single_fm'")
}

# Row 2, Col 1: Empty placeholder
p_empty <- ggplot() +
  theme_void() +
  theme(
    panel.background = element_rect(fill = color_defs[["--color-background"]], color = NA),
    plot.background = element_rect(fill = color_defs[["--color-background"]], color = NA)
  )

# Row 2, Col 2: Deep Learning
if (!is.null(data$configs$deep_learning)) {
  message("  Creating panel: Deep Learning...")
  p_deep <- create_instability_panel(
    config_data = data$configs$deep_learning,
    title = "D  Deep Learning",
    color_accent = ci_ribbon_color
  )
} else {
  stop("Missing required config 'deep_learning'")
}

# Row 2, Col 3: Traditional
if (!is.null(data$configs$traditional)) {
  message("  Creating panel: Traditional...")
  p_trad <- create_instability_panel(
    config_data = data$configs$traditional,
    title = "E  Traditional",
    color_accent = ci_ribbon_color
  )
} else {
  stop("Missing required config 'traditional'")
}

# Combine into 2×3 grid
# Row 1: Ground Truth | Ensemble FM | Single-model FM
# Row 2: (empty)      | Deep Learning | Traditional
combined <- (p_gt | p_ensemble | p_single_fm) /
            (p_empty | p_deep | p_trad) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

# ==============================================================================
# Save Figure
# ==============================================================================
message("[fig_instability_combined] Saving figure...")

# Ensure output directory exists (main category, not supplementary)
output_dir <- file.path(PROJECT_ROOT, "figures/generated/ggplot2/main")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Save using figure system with 2×3 layout dimensions
save_publication_figure(
  combined,
  "fig_instability_combined",
  output_dir = output_dir,
  width = 12,   # 3 columns

height = 8    # 2 rows
)

message("[fig_instability_combined] Done!")
message("  Output: ", file.path(output_dir, "fig_instability_combined.png"))
