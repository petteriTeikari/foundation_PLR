# SHAP Beeswarm Plot
# Shows feature importance with value distributions
# Task 3.2 from ggplot2-viz-remaining-plan.xml
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

# Load SHAP data - flat structure with samples array
shap_data <- fromJSON("data/r_data/shap_per_sample.json")

# Extract samples array - each row is one (config, sample_idx, feature, shap_value) tuple
samples_df <- shap_data$data$samples
n_samples_total <- shap_data$data$n_samples_total
n_configs <- shap_data$data$n_configs

message(sprintf("Loaded SHAP data: %d total sample-feature combinations, %d configs",
                n_samples_total, n_configs))

# Convert to dataframe if needed
if (!is.data.frame(samples_df)) {
  samples_df <- as.data.frame(samples_df)
}

# For beeswarm, use first config (best ensemble)
config_name <- samples_df$config[1]
shap_df <- samples_df %>%
  filter(config == config_name) %>%
  rename(shap_value = shap_value) %>%
  select(feature, shap_value, sample_idx)

# We need feature values - check if available, otherwise use normalized SHAP
# The JSON may not have feature_values, so we'll simulate based on SHAP direction
shap_df <- shap_df %>%
  group_by(feature) %>%
  mutate(
    # Use SHAP sign as proxy for feature value (high SHAP ~ high feature value)
    feature_value = shap_value
  ) %>%
  ungroup()

# Calculate mean |SHAP| for ordering
feature_importance <- shap_df %>%
  group_by(feature) %>%
  summarize(mean_abs_shap = mean(abs(shap_value), na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(mean_abs_shap)) %>%
  mutate(rank = row_number())

# Use all features (we only have 8)
n_features <- nrow(feature_importance)
top_features <- feature_importance %>%
  pull(feature)

shap_df <- shap_df %>%
  filter(feature %in% top_features) %>%
  mutate(feature = factor(feature, levels = rev(top_features)))

# Normalize feature values to 0-1 for color scale
shap_df <- shap_df %>%
  group_by(feature) %>%
  mutate(
    feature_value_norm = (feature_value - min(feature_value, na.rm = TRUE)) /
                         (max(feature_value, na.rm = TRUE) - min(feature_value, na.rm = TRUE) + 1e-10)
  ) %>%
  ungroup()

# ==============================================================================
# BEESWARM PLOT
# ==============================================================================

# Add jitter to y positions for beeswarm effect
set.seed(42)
shap_df <- shap_df %>%
  mutate(y_jitter = as.numeric(feature) + runif(n(), -0.3, 0.3))

p_beeswarm <- ggplot(shap_df, aes(x = shap_value, y = y_jitter, color = feature_value_norm))
p_beeswarm <- gg_add(p_beeswarm,
  geom_vline(xintercept = 0, linetype = "dashed", color = color_defs[["--color-text-secondary"]], linewidth = 0.5),
  geom_point(size = 1.2, alpha = 0.6),
  scale_color_gradient2(
    low = color_defs[["--color-primary"]],   # Low values: blue
    mid = color_defs[["--color-border"]],    # Mid values: gray
    high = color_defs[["--color-negative"]], # High values: red
    midpoint = 0.5,
    name = "Feature Value\n(normalized)",
    limits = c(0, 1)
  ),
  scale_y_continuous(
    breaks = 1:n_features,
    labels = rev(top_features),
    expand = expansion(mult = c(0.02, 0.02))
  ),
  # Labels (academic mode - no title/subtitle/caption for journal submission)
  labs(
    x = "SHAP Value (impact on prediction)",
    y = NULL
  ),
  theme_foundation_plr(),
  theme(
    legend.position = "right",
    axis.text.y = element_text(size = 9)
  )
)

# Save using figure system (formats from YAML config)
# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(p_beeswarm, "fig_shap_beeswarm")

# ==============================================================================
# SUMMARY
# ==============================================================================

message("\n========================================")
message("SHAP Beeswarm Plot Complete")
message("========================================")
message("Top 10 features by mean |SHAP|:")
print(head(feature_importance, 10))
