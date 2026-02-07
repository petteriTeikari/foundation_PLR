# Figure System Demo Usage (DEVELOPMENT ONLY)
# ============================================
# This script is for DEVELOPMENT/TESTING the figure system.
# DO NOT use this for production figure generation.
#
# For production figures, use:
#   Rscript src/r/figures/generate_all_r_figures.R
#
# Output formats are controlled by: configs/VISUALIZATION/figure_layouts.yaml
#   -> output_settings.formats (currently: ["png"])
#
# Created: 2026-01-27
# Author: Foundation PLR Team

# ==============================================================================
# SETUP
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
})

# Determine project root
find_project_root <- function() {
  markers <- c("pyproject.toml", "CLAUDE.md", ".git")
  dir <- getwd()
  while (dir != dirname(dir)) {
    if (any(file.exists(file.path(dir, markers)))) {
      return(dir)
    }
    dir <- dirname(dir)
  }
  stop("Could not find project root")
}

PROJECT_ROOT <- find_project_root()

# Load figure system
source(file.path(PROJECT_ROOT, "src/r/figure_system/figure_factory.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/compose_figures.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/save_figure.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/category_loader.R"))

# ==============================================================================
# LOAD DATA
# ==============================================================================

message("Loading metrics data...")
metrics_path <- file.path(PROJECT_ROOT, "data/r_data/essential_metrics.csv")

if (!file.exists(metrics_path)) {
  stop("Data not found. Run: python scripts/export_data_for_r.py")
}

metrics <- read_csv(metrics_path, show_col_types = FALSE)
metrics <- metrics %>% filter(toupper(classifier) == "CATBOOST")

# ==============================================================================
# AGGREGATE DATA
# ==============================================================================

message("Aggregating metrics...")

# Outlier summary (grouped by outlier method)
outlier_summary <- metrics %>%
  group_by(outlier_method, outlier_display_name) %>%
  summarize(
    auroc_mean = mean(auroc, na.rm = TRUE),
    auroc_ci_lo = min(auroc_ci_lo, na.rm = TRUE),
    auroc_ci_hi = max(auroc_ci_hi, na.rm = TRUE),
    n_configs = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(auroc_mean)) %>%
  mutate(
    # Category from YAML via category_loader (Single Source of Truth)
    category = categorize_outlier_methods(outlier_method)
  )

# Imputation summary (grouped by imputation method)
imputation_summary <- metrics %>%
  group_by(imputation_method, imputation_display_name) %>%
  summarize(
    auroc_mean = mean(auroc, na.rm = TRUE),
    auroc_ci_lo = min(auroc_ci_lo, na.rm = TRUE),
    auroc_ci_hi = max(auroc_ci_hi, na.rm = TRUE),
    n_configs = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(auroc_mean)) %>%
  mutate(
    # Category from YAML via category_loader (Single Source of Truth)
    category = categorize_imputation_methods(imputation_method)
  )

# ==============================================================================
# EXAMPLE 1: Single standalone figures
# ==============================================================================
# Output format controlled by: configs/VISUALIZATION/figure_layouts.yaml
#   -> output_settings.formats

message("\n--- Example 1: Standalone Figures ---")

# Create forest plot for outlier methods
p_outlier <- create_forest_outlier(
  outlier_summary,
  infographic = FALSE,  # Journal mode (no title/caption)
  show_legend = TRUE
)

save_publication_figure(p_outlier, "fig02_forest_outlier")

# Create forest plot for imputation methods
p_imputation <- create_forest_imputation(
  imputation_summary,
  infographic = FALSE,
  show_legend = TRUE
)

save_publication_figure(p_imputation, "fig03_forest_imputation")

# ==============================================================================
# EXAMPLE 2: Composed multi-panel figure with panel titles
# ==============================================================================

message("\n--- Example 2: Composed Multi-Panel Figure with Titles ---")

# Create panels without legends (will add shared legend)
p1 <- create_forest_outlier(outlier_summary, infographic = FALSE, show_legend = FALSE)
p2 <- create_forest_imputation(imputation_summary, infographic = FALSE, show_legend = TRUE)

# Compose into 2x1 layout with panel labels AND titles
# Using Neue Haas Grotesk Display Pro font (from YAML fonts.tag_family)
composed <- compose_figures(
  list(p1, p2),
  layout = "2x1",
  tag_levels = "A",
  panel_titles = c("Outlier Detection Method", "Imputation Method"),
  tag_font = "Neue Haas Grotesk Display Pro",
  tag_size = 14
)

save_publication_figure(composed, "fig_forest_combined", width = 10, height = 12)

# ==============================================================================
# EXAMPLE 3: Infographic mode (for presentations)
# ==============================================================================

message("\n--- Example 3: Infographic Mode ---")

p_infographic <- create_forest_outlier(
  outlier_summary,
  infographic = TRUE,  # Includes title, subtitle, caption
  show_legend = TRUE
)

save_publication_figure(p_infographic, "fig_forest_outlier_infographic")

# ==============================================================================
# EXAMPLE 4: Config-driven composition with custom titles
# ==============================================================================

message("\n--- Example 4: Config-Driven with Panel Titles ---")

# First create panels
p1_cfg <- create_forest_outlier(outlier_summary, infographic = FALSE, show_legend = FALSE)
p2_cfg <- create_forest_imputation(imputation_summary, infographic = FALSE, show_legend = TRUE)

# Compose with panel titles
composed_with_titles <- compose_figures(
  list(p1_cfg, p2_cfg),
  layout = "2x1",
  tag_levels = "A",
  panel_titles = c("Outlier Detection Method", "Imputation Method"),
  tag_font = "Neue Haas Grotesk Display Pro",
  tag_size = 14
)

save_publication_figure(composed_with_titles, "fig_forest_combined_from_config", width = 10, height = 12)

# ==============================================================================
# SUMMARY
# ==============================================================================

message("\n========================================")
message("Figure System Demo Complete")
message("========================================")
message("Output formats controlled by: configs/VISUALIZATION/figure_layouts.yaml")
message("  -> output_settings.formats")
message("\nFiles created:")
list.files(file.path(PROJECT_ROOT, "figures/generated/ggplot2"), pattern = "^fig_?(02|03|forest)")
