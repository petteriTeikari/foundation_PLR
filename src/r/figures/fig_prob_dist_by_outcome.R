# Probability Distribution by Outcome (STRATOS-Compliant)
# Van Calster 2024 core set requirement: probability distributions for each outcome
# Task 2.12 from ggplot2-viz-remaining-plan.xml
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
source(file.path(PROJECT_ROOT, "src/r/figure_system/compose_figures.R"))
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))
source(file.path(PROJECT_ROOT, "src/r/color_palettes.R"))

# Load color definitions from YAML
color_defs <- load_color_definitions()

# ==============================================================================
# LOAD DATA
# ==============================================================================

pred_data <- fromJSON("data/r_data/predictions_top4.json")

# Get configs dataframe
configs <- pred_data$data$configs

# Prepare prediction data
pred_list <- lapply(1:nrow(configs), function(i) {
  data.frame(
    config = configs$name[i],
    y_true = configs$y_true[[i]],
    y_prob = configs$y_prob[[i]],
    stringsAsFactors = FALSE
  )
})
pred_df <- bind_rows(pred_list) %>%
  mutate(
    config = factor(config, levels = unique(config)),
    outcome = factor(
      ifelse(y_true == 1, "Glaucoma (n=56)", "Control (n=152)"),
      levels = c("Control (n=152)", "Glaucoma (n=56)")
    )
  )

# Use first config for single-panel plot
pred_single <- pred_df %>%
  filter(config == levels(config)[1])

# ==============================================================================
# FIGURE 1: DENSITY PLOT (Single Config)
# ==============================================================================

# Colors: Blue for control, Red for glaucoma (Economist style)
outcome_colors <- c(
  "Control (n=152)" = color_defs[["--color-primary"]],
  "Glaucoma (n=56)" = color_defs[["--color-negative"]]
)

# Compute discrimination slope
disc_slope <- pred_single %>%
  group_by(outcome) %>%
  summarize(mean_prob = mean(y_prob), .groups = "drop")

discrimination_slope <- diff(disc_slope$mean_prob)

p_density <- ggplot(pred_single, aes(x = y_prob, fill = outcome, color = outcome))
p_density <- gg_add(p_density,
  # Density curves
  geom_density(alpha = 0.5, linewidth = 0.8),
  # Rug plot for individual observations
  geom_rug(alpha = 0.3, linewidth = 0.3),
  # Mean markers
  geom_vline(
    data = disc_slope,
    aes(xintercept = mean_prob, color = outcome),
    linetype = "dashed",
    linewidth = 0.8
  ),
  scale_fill_manual(values = outcome_colors, name = "Outcome"),
  scale_color_manual(values = outcome_colors, name = "Outcome"),
  scale_x_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.2),
    labels = scales::percent
  ),
  # Discrimination slope annotation
  annotate(
    "segment",
    x = disc_slope$mean_prob[1], xend = disc_slope$mean_prob[2],
    y = 2.5, yend = 2.5,
    arrow = arrow(ends = "both", length = unit(0.1, "inches")),
    color = color_defs[["--color-text-primary"]],
    linewidth = 0.5
  ),
  annotate(
    "text",
    x = mean(disc_slope$mean_prob), y = 2.7,
    label = paste0("Discrimination slope: ", round(discrimination_slope, 3)),
    size = 3.5,
    color = color_defs[["--color-text-primary"]]
  ),
  # Labels (academic mode - no title/subtitle/caption for journal submission)
  labs(
    x = "Predicted Probability of Glaucoma",
    y = "Density"
  ),
  theme_foundation_plr(),
  theme(
    legend.position = c(0.85, 0.85),
    legend.background = element_rect(fill = color_defs[["--color-background"]], color = NA)
  )
)

# NOTE: Standalone fig_prob_dist_by_outcome REMOVED per user request
# The combined figure (fig_prob_dist_combined) includes this as panel A
# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
# save_publication_figure(p_density, "fig_prob_dist_by_outcome")

# ==============================================================================
# FIGURE 2: FACETED BY CONFIG (Supplementary)
# ==============================================================================

p_facet <- ggplot(pred_df, aes(x = y_prob, fill = outcome, color = outcome))
p_facet <- gg_add(p_facet,
  geom_density(alpha = 0.5, linewidth = 0.6),
  geom_rug(alpha = 0.2, linewidth = 0.2),
  # Free y-axis scaling: traditional has very high density spike near 0
  # which would flatten all other curves if using shared y-axis
  facet_wrap(~ config, ncol = 2, scales = "free_y"),
  scale_fill_manual(values = outcome_colors, name = "Outcome"),
  scale_color_manual(values = outcome_colors, name = "Outcome"),
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)),
  # Labels (academic mode)
  labs(
    x = "Predicted Probability",
    y = "Density"
  ),
  theme_foundation_plr(),
  theme(
    legend.position = "bottom",
    strip.text = element_text(size = 9)
  )
)

# Save faceted version using figure system
# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(p_facet, "fig_prob_dist_faceted")

# ==============================================================================
# FIGURE 3: COMBINED (1x2 layout - single on left, faceted on right)
# ==============================================================================

# Add "A  Title" style headers to each panel (consistent pattern across all figures)
title_theme <- theme(
  plot.title = element_text(face = "bold", size = 14, hjust = 0),
  plot.title.position = "plot"
)

p_density_titled <- p_density + ggtitle("A  Representative Pipeline") + title_theme
p_facet_titled <- p_facet + ggtitle("B  All Pipelines") + title_theme

# Compose with patchwork (no tag_levels - using manual ggtitle)
fig_prob_combined <- (p_density_titled | p_facet_titled) +
  patchwork::plot_layout(widths = c(1, 1))

# Save combined figure using figure system
# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(fig_prob_combined, "fig_prob_dist_combined")

# ==============================================================================
# SUMMARY
# ==============================================================================

message("\n========================================")
message("Probability Distribution Plot Complete (STRATOS)")
message("========================================")
message("Elements included:")
message("  - Density curves for Control vs Glaucoma")
message("  - Rug plot for individual observations")
message("  - Mean probability markers")
message("  - Discrimination slope annotation")
message("\nOutput files:")
message("  - fig_prob_dist_by_outcome.pdf (main)")
message("  - fig_prob_dist_faceted.pdf (supplementary)")
