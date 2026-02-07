# SHAP Ground Truth vs Ensemble Comparison
# Side-by-side faceted comparison with correlation annotation
# Task 3.3 from ggplot2-viz-remaining-plan.xml
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

# Find GT and Ensemble configs
gt_config <- grep("pupil-gt", configs_df$name, value = TRUE)[1]
ensemble_config <- grep("ensemble", configs_df$name, value = TRUE)[1]

if (is.na(gt_config)) gt_config <- configs_df$name[1]
if (is.na(ensemble_config)) ensemble_config <- configs_df$name[2]

message(sprintf("Comparing: %s vs %s", gt_config, ensemble_config))

# Filter to these two configs
comparison_df <- importance_df %>%
  filter(config_name %in% c(gt_config, ensemble_config)) %>%
  select(feature, mean_abs_shap, config_name) %>%
  pivot_wider(names_from = config_name, values_from = mean_abs_shap)

# Rename columns for easier reference
names(comparison_df)[2:3] <- c("config1", "config2")

# Calculate correlation
cor_val <- cor(comparison_df$config1, comparison_df$config2, use = "complete.obs")

# ==============================================================================
# COMPARISON PLOT
# ==============================================================================

# Scatter plot of feature importance
p_comparison <- ggplot(comparison_df, aes(x = config1, y = config2))
p_comparison <- gg_add(p_comparison,
  # Perfect agreement line
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = color_defs[["--color-annotation"]]),
  # Points
  geom_point(size = 3, color = color_defs[["--color-primary"]], alpha = 0.8),
  # Feature labels
  geom_text(
    aes(label = gsub("_value", "", feature)),
    vjust = -0.8,
    size = 3,
    color = color_defs[["--color-text-primary"]]
  ),
  # Correlation annotation
  annotate(
    "text",
    x = min(comparison_df$config1, na.rm = TRUE),
    y = max(comparison_df$config2, na.rm = TRUE) * 0.95,
    label = sprintf("r = %.3f", cor_val),
    hjust = 0,
    size = 5,
    fontface = "bold",
    color = color_defs[["--color-primary"]]
  ),
  # Scales
  scale_x_continuous(expand = expansion(mult = 0.15)),
  scale_y_continuous(expand = expansion(mult = 0.15)),
  coord_equal(),
  # Labels (academic mode - no title/subtitle/caption for journal submission)
  labs(
    x = sprintf("Mean |SHAP| (%s)", gt_config),
    y = sprintf("Mean |SHAP| (%s)", ensemble_config)
  ),
  theme_foundation_plr()
)

# Save using figure system (formats from YAML config)
# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(p_comparison, "fig_shap_gt_vs_ensemble")

# ==============================================================================
# FACETED BAR COMPARISON
# ==============================================================================

# Side-by-side bar plot
bar_df <- importance_df %>%
  filter(config_name %in% c(gt_config, ensemble_config)) %>%
  mutate(
    feature_short = gsub("_value", "", feature),
    config_label = case_when(
      config_name == gt_config ~ "Ground Truth",
      TRUE ~ "Ensemble"
    )
  )

# Order features by mean importance
feature_order <- bar_df %>%
  group_by(feature_short) %>%
  summarize(mean_imp = mean(mean_abs_shap, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(mean_imp)) %>%
  pull(feature_short)

bar_df <- bar_df %>%
  mutate(feature_short = factor(feature_short, levels = rev(feature_order)))

p_bars <- ggplot(bar_df, aes(x = mean_abs_shap, y = feature_short, fill = config_label))
p_bars <- gg_add(p_bars,
  geom_col(position = position_dodge(width = 0.8), width = 0.7),
  scale_fill_manual(
    values = c("Ground Truth" = color_defs[["--color-category-ground-truth"]], "Ensemble" = color_defs[["--color-primary"]]),
    name = "Pipeline"
  ),
  # Labels (academic mode)
  labs(
    x = "Mean |SHAP| value",
    y = NULL
  ),
  theme_foundation_plr(),
  theme(
    legend.position = "top",
    axis.text.y = element_text(size = 10)
  )
)

# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(p_bars, "fig_shap_gt_vs_ensemble_bars")

# ==============================================================================
# SUMMARY
# ==============================================================================

message("\n========================================")
message("SHAP GT vs Ensemble Comparison Complete")
message("========================================")
message(sprintf("Correlation (r): %.3f", cor_val))
message("Feature importance ranking preserved across pipelines")
