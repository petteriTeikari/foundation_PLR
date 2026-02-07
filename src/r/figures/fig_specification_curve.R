# Specification Curve Analysis
# Shows AUROC for all 328 preprocessing configurations
# Task 2.9 from ggplot2-viz-remaining-plan.xml
#
# Created: 2026-01-25
# Author: Foundation PLR Team
#
# Note: Uses gg_add() instead of + operator for S7/ggplot2 4.0+ compatibility

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

# ==============================================================================
# LOAD AND PREPARE DATA
# ==============================================================================

# Load all metrics
metrics <- read_csv("data/r_data/essential_metrics.csv", show_col_types = FALSE)

# Filter to CatBoost and order by AUROC
spec_data <- metrics %>%
  filter(toupper(classifier) == "CATBOOST") %>%
  filter(!is.na(auroc)) %>%
  arrange(desc(auroc)) %>%
  mutate(
    rank = row_number(),
    # Create category labels (from YAML via category_loader)
    outlier_category = categorize_outlier_methods(outlier_method),
    imputation_category = categorize_imputation_methods(imputation_method)
  )

n_configs <- nrow(spec_data)
message(sprintf("Loaded %d configurations", n_configs))

# ==============================================================================
# SPECIFICATION CURVE PLOT
# ==============================================================================

# Categorize by outlier method type (5 standard categories)
# This matches the grouping used in other figures (instability, decomposition, ROC/RC)
spec_data <- spec_data %>%
  mutate(
    # Rename categories to match other figures
    pipeline_category = case_when(
      outlier_category == "Ground Truth" ~ "Ground Truth",
      outlier_category == "Ensemble" ~ "Ensemble FM",
      outlier_category == "Foundation Model" ~ "Single-model FM",
      outlier_category == "Deep Learning" ~ "Deep Learning",
      outlier_category == "Traditional" ~ "Traditional",
      TRUE ~ "Unknown"
    )
  )

# Define category order (best to worst, matching other figures)
category_order <- c("Ground Truth", "Ensemble FM", "Single-model FM", "Deep Learning", "Traditional")
spec_data$pipeline_category <- factor(spec_data$pipeline_category, levels = category_order)

# Colors from YAML config (Single Source of Truth) - 5 standard categories
color_palette <- c(
  "Ground Truth" = resolve_color("--color-category-ground-truth", color_defs),
  "Ensemble FM" = resolve_color("--color-category-ensemble", color_defs),
  "Single-model FM" = resolve_color("--color-category-foundation-model", color_defs),
  "Deep Learning" = resolve_color("--color-category-deep-learning", color_defs),
  "Traditional" = resolve_color("--color-category-traditional", color_defs)
)

p_spec <- ggplot(spec_data, aes(x = rank, y = auroc, color = pipeline_category))
p_spec <- gg_add(p_spec,
  # Error bars (CI)
  geom_errorbar(
    aes(ymin = auroc_ci_lo, ymax = auroc_ci_hi),
    width = 0,
    alpha = 0.3,
    linewidth = 0.3
  ),
  # Points
  geom_point(size = 1.5, alpha = 0.8),
  # Ground truth reference line
  geom_hline(
    yintercept = spec_data$auroc[spec_data$pipeline_category == "Ground Truth"][1],
    linetype = "dashed",
    color = resolve_color("--color-category-ground-truth", color_defs),
    linewidth = 0.5
  ),
  # Color scale (5 standard preprocessing categories)
  scale_color_manual(values = color_palette, name = "Preprocessing Category"),
  # Axis formatting
  scale_x_continuous(
    breaks = c(1, 50, 100, 150, 200, 250, 300, n_configs),
    expand = expansion(mult = c(0.02, 0.02))
  ),
  scale_y_continuous(
    limits = c(0.82, 0.92),
    breaks = seq(0.82, 0.92, 0.02)
  ),
  # Labels (academic mode - no title/subtitle/caption for journal submission)
  labs(
    x = "Configuration Rank (by AUROC)",
    y = "AUROC"
  ),
  theme_foundation_plr(),
  theme(
    legend.position = "bottom"
  ),
  guides(color = guide_legend(nrow = 1, override.aes = list(size = 3)))
)

# Save using figure system (formats from YAML config)
# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(p_spec, "fig_specification_curve")

# ==============================================================================
# SUMMARY STATISTICS
# ==============================================================================

message("\n========================================")
message("Specification Curve Complete")
message("========================================")
message(sprintf("Total configurations: %d", n_configs))
message(sprintf("AUROC range: %.3f - %.3f", min(spec_data$auroc), max(spec_data$auroc)))
message("\nConfigurations by category:")
print(table(spec_data$pipeline_category))
message("\nTop 5 configurations:")
print(head(spec_data %>% select(rank, pipeline_category, outlier_method, imputation_method, auroc), 5))
