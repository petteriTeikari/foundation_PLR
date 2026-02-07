# Decision Curve Analysis (STRATOS-Compliant)
# Van Calster 2024 requirements: Net benefit curves with treat-all/treat-none references
# Task 2.4 from ggplot2-viz-remaining-plan.xml
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
# LOAD DATA
# ==============================================================================

dca_data <- fromJSON("data/r_data/dca_data.json")

# Extract configs dataframe
configs <- dca_data$data$configs
prevalence <- dca_data$data$sample_prevalence

# Prepare DCA curve data
dca_list <- lapply(1:nrow(configs), function(i) {
  data.frame(
    config = configs$name[i],
    threshold = configs$thresholds[[i]],
    nb_model = configs$nb_model[[i]],
    nb_treat_all = configs$nb_treat_all[[i]],
    nb_treat_none = configs$nb_treat_none[[i]],
    stringsAsFactors = FALSE
  )
})
dca_df <- bind_rows(dca_list) %>%
  mutate(config = factor(config, levels = unique(config)))

# Create reference strategies data (for all thresholds)
ref_thresholds <- dca_df$threshold[dca_df$config == levels(dca_df$config)[1]]
ref_df <- data.frame(
  threshold = ref_thresholds,
  nb_treat_all = prevalence - (1 - prevalence) * (ref_thresholds / (1 - ref_thresholds)),
  nb_treat_none = 0
)

# ==============================================================================
# DCA PLOT
# ==============================================================================

# Colors for models (Economist palette)
config_colors <- ECONOMIST_PALETTE[1:4]
names(config_colors) <- levels(dca_df$config)

# Create the plot
p_dca <- ggplot(dca_df, aes(x = threshold, y = nb_model, color = config))
p_dca <- gg_add(p_dca,
  # Treat none reference (horizontal at 0)
  geom_hline(yintercept = 0, linetype = "dashed", color = color_defs[["--color-annotation"]], linewidth = 0.5),
  # Treat all reference
  geom_line(
    data = ref_df,
    aes(x = threshold, y = nb_treat_all),
    inherit.aes = FALSE,
    linetype = "dotted",
    color = color_defs[["--color-text-primary"]],
    linewidth = 0.8
  ),
  # Model net benefit curves
  geom_line(linewidth = 1),
  # Prevalence marker
  geom_vline(
    xintercept = prevalence,
    linetype = "solid",
    color = color_defs[["--color-text-secondary"]],
    linewidth = 0.3
  ),
  # Key threshold markers (5%, 10%, 15%, 20%)
  geom_vline(
    xintercept = c(0.05, 0.10, 0.15, 0.20),
    linetype = "dotted",
    color = color_defs[["--color-border"]],
    linewidth = 0.3
  ),
  scale_color_manual(values = config_colors, name = "Pipeline"),
  scale_x_continuous(
    limits = c(0, 0.30),
    breaks = seq(0, 0.30, 0.05),
    labels = scales::percent
  ),
  scale_y_continuous(
    limits = c(-0.05, 0.30),
    breaks = seq(0, 0.30, 0.05)
  ),
  # Annotations
  annotate(
    "text",
    x = 0.28, y = ref_df$nb_treat_all[nrow(ref_df)] + 0.01,
    label = "Treat All",
    color = color_defs[["--color-text-primary"]],
    size = 3,
    hjust = 1,
    fontface = "italic"
  ),
  annotate(
    "text",
    x = 0.28, y = 0.01,
    label = "Treat None",
    color = color_defs[["--color-annotation"]],
    size = 3,
    hjust = 1,
    fontface = "italic"
  ),
  annotate(
    "text",
    x = prevalence + 0.005, y = 0.28,
    label = paste0("Prevalence\n", round(prevalence * 100, 1), "%"),
    color = color_defs[["--color-text-secondary"]],
    size = 2.5,
    hjust = 0,
    vjust = 1
  ),
  # Labels (academic mode - no title/subtitle/caption for journal submission)
  labs(
    x = "Threshold Probability",
    y = "Net Benefit"
  ),
  theme_foundation_plr(),
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal"
  ),
  guides(color = guide_legend(nrow = 2))
)

# Add net benefit values at key thresholds
key_thresholds <- c(0.05, 0.10, 0.15, 0.20)
nb_at_thresholds <- dca_df %>%
  filter(threshold %in% key_thresholds) %>%
  group_by(config) %>%
  summarize(
    t05 = nb_model[threshold == 0.05],
    t10 = nb_model[threshold == 0.10],
    t15 = nb_model[threshold == 0.15],
    t20 = nb_model[threshold == 0.20],
    .groups = "drop"
  )

message("\nNet Benefit at Key Thresholds:")
print(nb_at_thresholds)

# Save using figure system (formats from YAML config)
# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(p_dca, "fig_dca_stratos")

# ==============================================================================
# NET BENEFIT TABLE (supplementary)
# ==============================================================================

# Create table of NB at key thresholds for all configs
nb_table <- dca_df %>%
  filter(threshold %in% key_thresholds) %>%
  select(config, threshold, nb_model) %>%
  mutate(threshold = paste0(round(threshold * 100), "%")) %>%
  pivot_wider(names_from = threshold, values_from = nb_model)

message("\n========================================")
message("DCA Plot Complete (STRATOS)")
message("========================================")
message("Elements included:")
message("  - Net benefit curves for top-4 pipelines")
message("  - Treat-all reference (dotted)")
message("  - Treat-none reference (dashed at NB=0)")
message("  - Prevalence marker")
message("  - Key threshold markers (5%, 10%, 15%, 20%)")
message("\nNet Benefit Table:")
print(nb_table)
