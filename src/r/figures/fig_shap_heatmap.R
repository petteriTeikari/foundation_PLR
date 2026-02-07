# SHAP Feature Importance Heatmap Across Configs
# Shows how feature importance varies across preprocessing pipelines
# Task 3.4 from ggplot2-viz-remaining-plan.xml
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
  library(jsonlite)
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
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))
source(file.path(PROJECT_ROOT, "src/r/color_palettes.R"))

# Load color definitions from YAML
color_defs <- load_color_definitions()

# ==============================================================================
# LOAD AND PREPARE DATA
# ==============================================================================

# Load SHAP feature importance data
shap_data <- fromJSON("data/r_data/shap_feature_importance.json")

# Extract config data
configs_df <- shap_data$data$configs
feature_names <- shap_data$data$feature_names

message(sprintf("Loaded %d configs with %d features", nrow(configs_df), length(feature_names)))

# Extract feature importance for each config
importance_list <- lapply(1:nrow(configs_df), function(i) {
  fi <- configs_df$feature_importance[[i]]
  if (is.data.frame(fi)) {
    fi$config_name <- configs_df$name[i]
    fi$config_idx <- configs_df$config_idx[i]
    return(fi)
  }
  return(NULL)
})

importance_df <- bind_rows(importance_list[!sapply(importance_list, is.null)])

# Shorten feature names
importance_df <- importance_df %>%
  mutate(feature_short = gsub("_value", "", feature))

# Create heatmap data matrix
heatmap_df <- importance_df %>%
  select(config_name, feature_short, mean_abs_shap) %>%
  pivot_wider(names_from = feature_short, values_from = mean_abs_shap)

# Order configs by mean importance
config_order <- importance_df %>%
  group_by(config_name) %>%
  summarize(mean_imp = mean(mean_abs_shap, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(mean_imp)) %>%
  pull(config_name)

# Order features by mean importance
feature_order <- importance_df %>%
  group_by(feature_short) %>%
  summarize(mean_imp = mean(mean_abs_shap, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(mean_imp)) %>%
  pull(feature_short)

# Reshape for plotting
heatmap_long <- importance_df %>%
  select(config_name, feature_short, mean_abs_shap) %>%
  mutate(
    config_name = factor(config_name, levels = rev(config_order)),
    feature_short = factor(feature_short, levels = feature_order)
  )

# ==============================================================================
# HEATMAP PLOT
# ==============================================================================

p_heatmap <- ggplot(heatmap_long, aes(x = feature_short, y = config_name, fill = mean_abs_shap))
p_heatmap <- gg_add(p_heatmap,
  geom_tile(color = color_defs[["--color-white"]], linewidth = 0.5),
  geom_text(
    aes(label = sprintf("%.2f", mean_abs_shap)),
    size = 3,
    color = ifelse(heatmap_long$mean_abs_shap > 0.12, color_defs[["--color-white"]], color_defs[["--color-text-primary"]])
  ),
  scale_fill_gradient(
    low = color_defs[["--color-white"]],
    high = color_defs[["--color-primary"]],
    name = "Mean |SHAP|"
  ),
  # Labels (academic mode - no title/subtitle/caption for journal submission)
  labs(
    x = "Feature",
    y = "Preprocessing Pipeline"
  ),
  theme_foundation_plr(),
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
    axis.text.y = element_text(size = 8),
    legend.position = "right",
    panel.grid = element_blank()
  )
)

# Save using figure system (formats from YAML config)
# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(p_heatmap, "fig_shap_heatmap")

# ==============================================================================
# SUMMARY
# ==============================================================================

message("\n========================================")
message("SHAP Heatmap Complete")
message("========================================")
message("Top features by mean importance:")
feature_summary <- importance_df %>%
  group_by(feature_short) %>%
  summarize(
    mean_imp = mean(mean_abs_shap, na.rm = TRUE),
    sd_imp = sd(mean_abs_shap, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_imp))
print(head(feature_summary, 8))
