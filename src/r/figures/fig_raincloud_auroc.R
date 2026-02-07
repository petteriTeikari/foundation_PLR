# Raincloud Plot for AUROC Distributions
# Shows distribution of AUROC across preprocessing configurations
# Task 4.2 from ggplot2-viz-remaining-plan.xml
#
# Created: 2026-01-25
# Author: Foundation PLR Team
#
# Note: Uses ggdist package if available, fallback to violin+boxplot otherwise

# ==============================================================================
# SETUP
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
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

# Source figure system (MANDATORY)
source(file.path(PROJECT_ROOT, "src/r/figure_system/config_loader.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/save_figure.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/category_loader.R"))
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))
source(file.path(PROJECT_ROOT, "src/r/color_palettes.R"))

# Load color definitions from YAML
color_defs <- load_color_definitions()

# Check for ggdist
HAS_GGDIST <- requireNamespace("ggdist", quietly = TRUE)
if (HAS_GGDIST) {
  library(ggdist)
  message("Using ggdist for raincloud plots")
} else {
  message("ggdist not available, using violin+boxplot fallback")
}

# Output directory handled by save_publication_figure()

# ==============================================================================
# LOAD AND PREPARE DATA
# ==============================================================================

# Load metrics
metrics <- read_csv("data/r_data/essential_metrics.csv", show_col_types = FALSE)

# Filter to CatBoost only
metrics <- metrics %>%
  filter(toupper(classifier) == "CATBOOST") %>%
  filter(!is.na(auroc))

# Categorize preprocessing pipelines (using YAML-driven category loader)
metrics <- metrics %>%
  mutate(pipeline_type = categorize_outlier_methods(outlier_method))

# Calculate summary stats per type
type_stats <- metrics %>%
  group_by(pipeline_type) %>%
  summarize(
    n = n(),
    mean_auroc = mean(auroc, na.rm = TRUE),
    median_auroc = median(auroc, na.rm = TRUE),
    sd_auroc = sd(auroc, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_auroc))

message("Pipeline type statistics:")
print(type_stats)

# Order factor by mean AUROC
metrics <- metrics %>%
  mutate(pipeline_type = factor(pipeline_type, levels = type_stats$pipeline_type))

# ==============================================================================
# RAINCLOUD PLOT
# ==============================================================================

# Color palette (from YAML config)
type_colors <- get_category_colors()

if (HAS_GGDIST) {
  # Full raincloud with ggdist
  p_raincloud <- ggplot(metrics, aes(x = pipeline_type, y = auroc, fill = pipeline_type))
  p_raincloud <- gg_add(p_raincloud,
    # Density ridges (half-violin)
    stat_halfeye(
      adjust = 1,
      width = 0.6,
      .width = 0,
      justification = -0.2,
      point_colour = NA,
      alpha = 0.6
    ),
    # Boxplot
    geom_boxplot(
      width = 0.15,
      outlier.shape = NA,
      alpha = 0.8
    ),
    # Jittered points
    stat_dots(
      side = "left",
      justification = 1.1,
      binwidth = 0.003,
      alpha = 0.5
    ),
    scale_fill_manual(values = type_colors, guide = "none"),
    scale_y_continuous(limits = c(0.75, 0.95), breaks = seq(0.75, 0.95, 0.05)),
    coord_flip(),
    # Labels (academic mode - no title/subtitle/caption for journal submission)
    labs(
      y = "AUROC",
      x = NULL
    ),
    theme_foundation_plr(),
    theme(
      axis.text.y = element_text(size = 11, face = "bold")
    )
  )
} else {
  # Fallback: violin + boxplot
  p_raincloud <- ggplot(metrics, aes(x = pipeline_type, y = auroc, fill = pipeline_type))
  p_raincloud <- gg_add(p_raincloud,
    # Violin (half)
    geom_violin(
      width = 0.7,
      alpha = 0.6,
      trim = TRUE
    ),
    # Boxplot overlay
    geom_boxplot(
      width = 0.15,
      outlier.shape = 21,
      outlier.fill = color_defs[["--color-white"]],
      alpha = 0.9
    ),
    # Jittered points
    geom_jitter(
      width = 0.05,
      alpha = 0.4,
      size = 1.5,
      color = resolve_color("--color-text-primary", color_defs)
    ),
    scale_fill_manual(values = type_colors, guide = "none"),
    scale_y_continuous(limits = c(0.75, 0.95), breaks = seq(0.75, 0.95, 0.05)),
    coord_flip(),
    # Labels (academic mode)
    labs(
      y = "AUROC",
      x = NULL
    ),
    theme_foundation_plr(),
    theme(
      axis.text.y = element_text(size = 11, face = "bold")
    )
  )
}

# Save using figure system (formats from YAML config)
# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(p_raincloud, "fig_raincloud_auroc")

# ==============================================================================
# SUMMARY
# ==============================================================================

message("\n========================================")
message("Raincloud Plot Complete")
message("========================================")
message("AUROC distribution by pipeline type:")
print(type_stats %>% select(pipeline_type, n, mean_auroc, median_auroc))
