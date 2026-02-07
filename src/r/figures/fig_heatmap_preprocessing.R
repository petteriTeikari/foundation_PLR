# Preprocessing Heatmap
# Shows AUROC for each outlier × imputation combination
# Task 2.11 from ggplot2-viz-remaining-plan.xml
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
source(file.path(PROJECT_ROOT, "src/r/figure_system/common.R"))
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))
source(file.path(PROJECT_ROOT, "src/r/color_palettes.R"))

# Load color definitions from YAML
color_defs <- load_color_definitions()

# ==============================================================================
# LOAD AND PREPARE DATA
# ==============================================================================

# Load metrics
metrics <- read_csv("data/r_data/essential_metrics.csv", show_col_types = FALSE)

# Filter to CatBoost (fixed classifier)
metrics <- metrics %>%
  filter(toupper(classifier) == "CATBOOST") %>%
  filter(!is.na(auroc))

# Apply display names from YAML (Single Source of Truth)
heatmap_data <- metrics %>%
  apply_display_names() %>%
  select(
    outlier_short = outlier_display_name,
    imputation_short = imputation_display_name,
    auroc
  )

# Order by mean AUROC
outlier_order <- heatmap_data %>%
  group_by(outlier_short) %>%
  summarize(mean_auroc = mean(auroc, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(mean_auroc)) %>%
  pull(outlier_short)

imputation_order <- heatmap_data %>%
  group_by(imputation_short) %>%
  summarize(mean_auroc = mean(auroc, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(mean_auroc)) %>%
  pull(imputation_short)

heatmap_data <- heatmap_data %>%
  mutate(
    outlier_short = factor(outlier_short, levels = rev(outlier_order)),
    imputation_short = factor(imputation_short, levels = imputation_order)
  )

# Get the range for the color scale
auroc_range <- range(heatmap_data$auroc, na.rm = TRUE)
auroc_mid <- mean(auroc_range)

message(sprintf("AUROC range: %.3f - %.3f", auroc_range[1], auroc_range[2]))

# ==============================================================================
# HEATMAP PLOT
# ==============================================================================

p_heatmap <- ggplot(heatmap_data, aes(x = imputation_short, y = outlier_short, fill = auroc))
p_heatmap <- gg_add(p_heatmap,
  geom_tile(color = color_defs[["--color-white"]], linewidth = 0.5),
  geom_text(
    aes(label = sprintf("%.3f", auroc)),
    size = 2.8,
    color = ifelse(heatmap_data$auroc > auroc_mid,
                   color_defs[["--color-white"]],
                   color_defs[["--color-text-primary"]])
  ),
  scale_fill_gradient2(
    low = color_defs[["--color-white"]],
    mid = color_defs[["--color-secondary"]],
    high = color_defs[["--color-primary"]],
    midpoint = auroc_mid,
    name = "AUROC",
    limits = auroc_range
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
    legend.position = "right",
    panel.grid = element_blank()
  )
)

# Save using figure system (formats from YAML config)
# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(p_heatmap, "fig_heatmap_preprocessing")

# ==============================================================================
# SUMMARY
# ==============================================================================

message("\n========================================")
message("Preprocessing Heatmap Complete")
message("========================================")
message("Best outlier methods (by mean AUROC):")
print(head(data.frame(outlier = outlier_order), 5))
message("\nBest imputation methods (by mean AUROC):")
print(head(data.frame(imputation = imputation_order), 5))
