# Factorial Design Matrix
# Shows the experimental design: 15 outlier × 7 imputation × 5 classifier
# Task 2.5 from ggplot2-viz-remaining-plan.xml
#
# Created: 2026-01-25
# Author: Foundation PLR Team

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
source(file.path(PROJECT_ROOT, "src/r/figure_system/common.R"))
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))
source(file.path(PROJECT_ROOT, "src/r/color_palettes.R"))

# Load color definitions from YAML
color_defs <- load_color_definitions()

# ==============================================================================
# LOAD AND PREPARE DATA
# ==============================================================================

# Load metrics to get actual methods used
metrics <- read_csv("data/r_data/essential_metrics.csv", show_col_types = FALSE)

# Get unique methods
outlier_methods <- unique(metrics$outlier_method)
imputation_methods <- unique(metrics$imputation_method)
classifiers <- unique(metrics$classifier)

# Create summary counts
n_outlier <- length(outlier_methods)
n_imputation <- length(imputation_methods)
n_classifier <- length(classifiers)
n_total <- nrow(metrics)

message(sprintf("Factorial design: %d outlier × %d imputation × %d classifier = %d configurations",
                n_outlier, n_imputation, n_classifier, n_outlier * n_imputation * n_classifier))
message(sprintf("Actual configurations in data: %d", n_total))

# ==============================================================================
# CREATE FACTORIAL DESIGN MATRIX VISUALIZATION
# ==============================================================================

# For the matrix, we'll show outlier (y) × imputation (x), with color showing
# whether that combination exists in our data

matrix_data <- metrics %>%
  select(outlier_method, imputation_method) %>%
  distinct() %>%
  mutate(exists = TRUE)

# Create full factorial grid
full_grid <- expand.grid(
  outlier_method = outlier_methods,
  imputation_method = imputation_methods,
  stringsAsFactors = FALSE
)

# Join with existing combinations
matrix_data <- full_grid %>%
  left_join(matrix_data, by = c("outlier_method", "imputation_method")) %>%
  mutate(
    exists = ifelse(is.na(exists), FALSE, exists),
    status = ifelse(exists, "Evaluated", "Missing")
  )

# Apply display names from YAML (Single Source of Truth)
matrix_data <- matrix_data %>%
  apply_display_names() %>%
  rename(
    outlier_short = outlier_display_name,
    imputation_short = imputation_display_name
  )

# Order by method type
matrix_data <- matrix_data %>%
  mutate(
    outlier_short = factor(outlier_short, levels = rev(unique(outlier_short))),
    imputation_short = factor(imputation_short, levels = unique(imputation_short))
  )

# Plot
p_factorial <- ggplot(matrix_data, aes(x = imputation_short, y = outlier_short, fill = status))
p_factorial <- gg_add(p_factorial,
  geom_tile(color = color_defs[["--color-white"]], linewidth = 0.5),
  scale_fill_manual(
    values = c("Evaluated" = color_defs[["--color-primary"]], "Missing" = color_defs[["--color-grid-light"]]),
    name = "Status"
  ),
  # Labels (academic mode - no title/subtitle/caption for journal submission)
  labs(
    x = "Imputation Method",
    y = "Outlier Detection Method"
  ),
  theme_foundation_plr(),
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
    axis.text.y = element_text(size = 9),
    legend.position = "top",
    panel.grid = element_blank()
  )
)

# Save using figure system (formats from YAML config)
# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(p_factorial, "fig_factorial_matrix")

# ==============================================================================
# SUMMARY
# ==============================================================================

message("\n========================================")
message("Factorial Design Matrix Complete")
message("========================================")
message(sprintf("Outlier methods: %d", n_outlier))
message(sprintf("Imputation methods: %d", n_imputation))
message(sprintf("Classifiers: %d", n_classifier))
message(sprintf("Evaluated combinations: %d", sum(matrix_data$status == "Evaluated")))
