# SHAP Feature Importance Bar Plot
# Economist-style ggplot2 visualization
# Task 3.1 from ggplot2-viz-creation-plan.xml
#
# Created: 2026-01-25
# Author: Foundation PLR Team
#
# Note: Uses gg_add() instead of + operator for S7/ggplot2 4.0+ compatibility

# ==============================================================================
# SETUP
# ==============================================================================

# Load required packages
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
source(file.path(PROJECT_ROOT, "src/r/figure_system/compose_figures.R"))
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))
source(file.path(PROJECT_ROOT, "src/r/color_palettes.R"))

# Load color definitions from YAML
color_defs <- load_color_definitions()

# ==============================================================================
# LOAD DATA
# ==============================================================================

# Load SHAP feature importance
shap_data <- fromJSON("data/r_data/shap_feature_importance.json")

# Load VIF data for multicollinearity annotations
vif_path <- file.path(PROJECT_ROOT, "data/r_data/vif_analysis.json")
vif_data <- NULL
vif_df <- NULL
has_vif_warning <- FALSE

if (file.exists(vif_path)) {
  vif_data <- fromJSON(vif_path)
  vif_df <- as_tibble(vif_data$data$aggregate) %>%
    select(feature, VIF_mean, concern) %>%
    rename(vif = VIF_mean, vif_concern = concern)

  # Check if we have multicollinearity issues (VIF > 10)
  has_vif_warning <- any(vif_df$vif > 10, na.rm = TRUE)

  if (has_vif_warning) {
    message("\n⚠️  VIF WARNING: Multicollinearity detected!")
    high_vif <- vif_df %>% filter(vif > 10) %>% arrange(desc(vif))
    for (i in 1:nrow(high_vif)) {
      message(sprintf("  - %s: VIF = %.1f (%s)",
                      high_vif$feature[i], high_vif$vif[i], high_vif$vif_concern[i]))
    }
    message("SHAP values for these features should be interpreted with caution.\n")
  }
} else {
  message("Note: VIF data not found. Run: python scripts/compute_vif.py")
}

# Extract configs
configs <- shap_data$data$configs

# Prepare data for plotting (ground truth config)
gt_idx <- which(grepl("pupil-gt", configs$name))[1]
if (is.na(gt_idx)) gt_idx <- 1

gt_importance <- configs$feature_importance[[gt_idx]]
importance_df <- as_tibble(gt_importance) %>%
  mutate(
    wavelength = case_when(
      grepl("^Blue_", feature) ~ "Blue (469nm)",
      grepl("^Red_", feature) ~ "Red (640nm)",
      TRUE ~ "Combined"
    )
  )

# Merge VIF data and create annotated labels
if (!is.null(vif_df)) {
  importance_df <- importance_df %>%
    left_join(vif_df, by = "feature") %>%
    mutate(
      # Create annotated labels: feature [VIF=X] for high concern
      feature_label = case_when(
        vif_concern == "High" ~ paste0(feature, " [VIF=", round(vif, 0), "]"),
        vif_concern == "Moderate" ~ paste0(feature, " (VIF=", round(vif, 0), ")"),
        TRUE ~ as.character(feature)
      ),
      # Mark features with high VIF for visual distinction
      has_high_vif = vif_concern == "High"
    )
} else {
  importance_df <- importance_df %>%
    mutate(feature_label = as.character(feature), has_high_vif = FALSE)
}

# Set factor levels for ordering (use annotated labels)
importance_df <- importance_df %>%
  mutate(feature_label = factor(feature_label, levels = rev(feature_label)))

# ==============================================================================
# FIGURE 1: Single Config Feature Importance
# ==============================================================================

# Use feature_label (with VIF annotations) for y-axis
p_single <- ggplot(importance_df, aes(x = mean_abs_shap, y = feature_label, color = wavelength))
p_single <- gg_add(p_single,
  geom_pointrange(
    aes(xmin = ci_lo, xmax = ci_hi),
    size = 0.8,
    linewidth = 0.8
  ),
  geom_vline(xintercept = 0, linetype = "dashed", color = color_defs[["--color-annotation"]], linewidth = 0.3),
  scale_color_manual(
    values = c(
      "Blue (469nm)" = color_defs[["--color-primary"]],
      "Red (640nm)" = color_defs[["--color-negative"]]
    ),
    name = "Stimulus"
  ),
  labs(
    x = "Mean |SHAP| value",
    y = NULL
  ),
  theme_forest(),
  theme(
    legend.position = c(0.85, 0.2),
    legend.direction = "vertical",
    legend.background = element_rect(fill = color_defs[["--color-background"]], colour = NA)
  )
)

# Save using figure system (formats from YAML config)
# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(p_single, "fig_shap_importance_gt")

# ==============================================================================
# FIGURE 2: Multi-Config Comparison (YAML-Driven)
# ==============================================================================

# Load SHAP figure combos from YAML
shap_combos_yaml <- yaml::read_yaml(
  file.path(PROJECT_ROOT, "configs/VISUALIZATION/plot_hyperparam_combos.yaml")
)
shap_combos <- shap_combos_yaml$shap_figure_combos$configs

# Find matching configs from JSON data by pattern matching
config_names <- configs$name
multi_list <- list()

for (combo in shap_combos) {
  if (isTRUE(combo$is_aggregate)) {
    # Compute Top-10 Mean aggregate
    all_importance <- lapply(1:nrow(configs), function(i) {
      imp <- configs$feature_importance[[i]]
      imp$config_idx <- i
      imp
    })
    all_df <- bind_rows(all_importance)

    # Aggregate across all configs
    agg_df <- all_df %>%
      group_by(feature) %>%
      summarize(
        mean_abs_shap = mean(mean_abs_shap),
        ci_lo = mean(ci_lo),
        ci_hi = mean(ci_hi),
        .groups = "drop"
      ) %>%
      mutate(config = combo$name)

    multi_list[[combo$id]] <- agg_df
  } else {
    # Find matching config by pattern
    idx <- which(grepl(combo$config_pattern, config_names, fixed = TRUE))
    if (length(idx) > 0) {
      imp <- configs$feature_importance[[idx[1]]]
      imp$config <- combo$name
      multi_list[[combo$id]] <- imp
    } else {
      message(sprintf("  Warning: No config found matching '%s'", combo$config_pattern))
    }
  }
}

# Build color mapping from YAML - resolve color_ref to actual hex values
color_definitions <- shap_combos_yaml$color_definitions
resolve_color_ref <- function(ref) {
  if (is.null(ref)) return(color_defs[["--color-text-muted"]])  # Fallback gray
  # YAML keys include the "--" prefix, so use ref directly
  val <- color_definitions[[ref]]
  if (is.null(val)) {
    message(sprintf("  Warning: Could not resolve color ref '%s'", ref))
    return(color_defs[["--color-text-muted"]])
  }
  return(val)
}

shap_colors <- setNames(
  sapply(shap_combos, function(x) resolve_color_ref(x$color_ref)),
  sapply(shap_combos, function(x) x$name)
)

# Combine all matched configs
multi_df <- bind_rows(multi_list) %>%
  mutate(
    feature = factor(feature, levels = rev(unique(feature))),
    config = factor(config, levels = sapply(shap_combos, function(x) x$name)),
    wavelength = case_when(
      grepl("^Blue_", feature) ~ "Blue (469nm)",
      grepl("^Red_", feature) ~ "Red (640nm)",
      TRUE ~ "Combined"
    )
  )

message(sprintf("  SHAP multi-config: %d configs matched", length(unique(multi_df$config))))

p_multi <- ggplot(multi_df, aes(x = mean_abs_shap, y = feature, color = config))
p_multi <- gg_add(p_multi,
  geom_pointrange(
    aes(xmin = ci_lo, xmax = ci_hi),
    position = position_dodge(width = 0.7),
    size = 0.5,
    linewidth = 0.5
  ),
  scale_color_manual(values = shap_colors, name = "Pipeline"),
  # Labels (academic mode)
  labs(
    x = "Mean |SHAP| value",
    y = NULL
  ),
  theme_forest(),
  theme(
    legend.position = "top",
    legend.direction = "horizontal"
  ),
  guides(color = guide_legend(nrow = 2))
)

# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(p_multi, "fig_shap_importance_multi")

# ==============================================================================
# FIGURE 3: Ensemble Aggregated Importance
# ==============================================================================

# Load ensemble aggregated data
ensemble_data <- fromJSON("data/r_data/shap_ensemble_aggregated.json")

ensemble_df <- as_tibble(ensemble_data$data$features) %>%
  mutate(
    wavelength = case_when(
      grepl("^Blue_", feature) ~ "Blue (469nm)",
      grepl("^Red_", feature) ~ "Red (640nm)",
      TRUE ~ "Combined"
    )
  )

# Merge VIF data for ensemble plot too
if (!is.null(vif_df)) {
  ensemble_df <- ensemble_df %>%
    left_join(vif_df, by = "feature") %>%
    mutate(
      feature_label = case_when(
        vif_concern == "High" ~ paste0(feature, " [VIF=", round(vif, 0), "]"),
        vif_concern == "Moderate" ~ paste0(feature, " (VIF=", round(vif, 0), ")"),
        TRUE ~ as.character(feature)
      )
    )
} else {
  ensemble_df <- ensemble_df %>%
    mutate(feature_label = as.character(feature))
}

ensemble_df <- ensemble_df %>%
  mutate(feature_label = factor(feature_label, levels = rev(feature_label)))

p_ensemble <- ggplot(ensemble_df, aes(x = ensemble_mean, y = feature_label, color = wavelength))
p_ensemble <- gg_add(p_ensemble,
  geom_pointrange(
    aes(xmin = ci_lo, xmax = ci_hi),
    size = 0.8,
    linewidth = 0.8
  ),
  geom_vline(xintercept = 0, linetype = "dashed", color = color_defs[["--color-annotation"]], linewidth = 0.3),
  scale_color_manual(
    values = c(
      "Blue (469nm)" = color_defs[["--color-primary"]],
      "Red (640nm)" = color_defs[["--color-negative"]]
    ),
    name = "Stimulus"
  ),
  # Labels (academic mode)
  labs(
    x = "Mean |SHAP| value",
    y = NULL
  ),
  theme_forest(),
  theme(
    legend.position = c(0.85, 0.2),
    legend.direction = "vertical",
    legend.background = element_rect(fill = color_defs[["--color-background"]], colour = NA)
  )
)

# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(p_ensemble, "fig_shap_importance_ensemble")

# ==============================================================================
# FIGURE 4: COMBINED GT + Ensemble (1x2 layout) with VIF disclaimer
# ==============================================================================

# Add "A  Title" style headers (consistent pattern across all figures)
title_theme <- theme(
  plot.title = element_text(face = "bold", size = 14, hjust = 0),
  plot.title.position = "plot"
)

p_single_titled <- p_single + ggtitle("A  Ground Truth Pipeline") + title_theme
p_ensemble_titled <- p_ensemble + ggtitle("B  Ensemble Pipeline") + title_theme

# Compose with patchwork (no tag_levels - using manual ggtitle)
fig_shap_combined <- (p_single_titled | p_ensemble_titled) +
  patchwork::plot_layout(widths = c(1, 1))

# NOTE: VIF disclaimer caption is now in supplementary.tex figure caption
# (removed hardcoded caption from figure per publication guidelines)

# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(fig_shap_combined, "fig_shap_importance_combined")

# ==============================================================================
# SUMMARY
# ==============================================================================

message("\n========================================")
message("SHAP Feature Importance Figures Complete")
message("========================================")
message("Output files:")
message("  - fig_shap_importance_gt.pdf (Ground truth)")
message("  - fig_shap_importance_multi.pdf (Top 4 comparison)")
message("  - fig_shap_importance_ensemble.pdf (Ensemble aggregated)")
message("  - fig_shap_importance_combined.pdf (GT + Ensemble side-by-side)")
message("\nTheme: Economist-style (off-white via --color-background)")
message("Colors: Economist red (--color-negative) + blue (--color-primary)")

if (has_vif_warning) {
  message("\n⚠️  MULTICOLLINEARITY NOTE:")
  message("VIF annotations added to feature labels:")
  message("  - [VIF=X] = High concern (VIF > 20)")
  message("  - (VIF=X) = Moderate concern (VIF 10-20)")
  message("Figure caption includes disclaimer about SHAP reliability.")
}
