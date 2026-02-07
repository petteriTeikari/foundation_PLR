# Generate All R Figures
# ======================
# Single entry point for generating ALL R/ggplot2 figures.
# Each figure is generated ONCE - no duplicates, no demos.
#
# Run from project root:
#   Rscript src/r/figures/generate_all_r_figures.R
#
# Created: 2026-01-27
# Author: Foundation PLR Team

# ==============================================================================
# SETUP
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
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

# Load figure system
source(file.path(PROJECT_ROOT, "src/r/figure_system/figure_factory.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/compose_figures.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/save_figure.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/category_loader.R"))

message("========================================")
message("Generating All R Figures")
message("========================================")
message(sprintf("Project root: %s", PROJECT_ROOT))
message("")

# ==============================================================================
# LOAD DATA
# ==============================================================================

message("Loading data...")
metrics_path <- file.path(PROJECT_ROOT, "data/r_data/essential_metrics.csv")

if (!file.exists(metrics_path)) {
  stop("Data not found. Run: python scripts/export_data_for_r.py")
}

metrics <- read_csv(metrics_path, show_col_types = FALSE)
metrics <- metrics %>% filter(toupper(classifier) == "CATBOOST")

# Aggregate for forest plots
outlier_summary <- metrics %>%
  group_by(outlier_method, outlier_display_name) %>%
  summarize(
    auroc_mean = mean(auroc, na.rm = TRUE),
    auroc_ci_lo = min(auroc_ci_lo, na.rm = TRUE),
    auroc_ci_hi = max(auroc_ci_hi, na.rm = TRUE),
    n_configs = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(auroc_mean)) %>%
  mutate(category = categorize_outlier_methods(outlier_method))

imputation_summary <- metrics %>%
  group_by(imputation_method, imputation_display_name) %>%
  summarize(
    auroc_mean = mean(auroc, na.rm = TRUE),
    auroc_ci_lo = min(auroc_ci_lo, na.rm = TRUE),
    auroc_ci_hi = max(auroc_ci_hi, na.rm = TRUE),
    n_configs = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(auroc_mean)) %>%
  mutate(category = categorize_imputation_methods(imputation_method))

message(sprintf("Loaded %d configurations", nrow(metrics)))
message("")

# ==============================================================================
# FIGURE 1: COMBINED FOREST PLOT (Outlier + Imputation)
# ==============================================================================

message("--- Figure: Combined Forest Plot ---")

p1 <- create_forest_outlier(outlier_summary, infographic = FALSE, show_legend = FALSE)
p2 <- create_forest_imputation(imputation_summary, infographic = FALSE, show_legend = TRUE)

fig_forest <- compose_figures(
  list(p1, p2),
  layout = "2x1",
  tag_levels = "A",
  panel_titles = c("Outlier Detection Method", "Imputation Method"),
  tag_font = "Neue Haas Grotesk Display Pro",
  tag_size = 14
)

# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(fig_forest, "fig_forest_combined")

# ==============================================================================
# FIGURE 2: COMBINED CALIBRATION + DCA (STRATOS)
# ==============================================================================

message("")
message("--- Figure: Combined Calibration + DCA ---")

# Load calibration data
cal_json_path <- file.path(PROJECT_ROOT, "data/r_data/calibration_data.json")
dca_json_path <- file.path(PROJECT_ROOT, "data/r_data/dca_data.json")
pred_json_path <- file.path(PROJECT_ROOT, "data/r_data/predictions_top4.json")

if (file.exists(cal_json_path) && file.exists(dca_json_path)) {
  cal_data <- fromJSON(cal_json_path)
  dca_data <- fromJSON(dca_json_path)

  # Prepare calibration curve data
  configs <- cal_data$data$configs
  cal_curves <- lapply(1:nrow(configs), function(i) {
    curve <- configs$curve
    data.frame(
      config = configs$name[i],
      bin_midpoint = curve$bin_midpoints[[i]],
      observed = curve$observed[[i]],
      predicted = curve$predicted[[i]],
      count = curve$counts[[i]],
      stringsAsFactors = FALSE
    )
  })
  cal_df <- bind_rows(cal_curves) %>%
    filter(!is.na(observed)) %>%
    mutate(config = factor(config, levels = unique(config)))

  # Get annotations for first config
  gt_idx <- which(grepl("pupil-gt|ensemble", configs$name))[1]
  if (is.na(gt_idx) || length(gt_idx) == 0) gt_idx <- 1

  cal_annotations <- list(
    calibration_slope = configs$calibration_slope[gt_idx],
    calibration_intercept = configs$calibration_intercept[gt_idx],
    o_e_ratio = configs$o_e_ratio[gt_idx],
    brier = configs$brier[gt_idx],
    ipa = configs$ipa[gt_idx],
    n = configs$n[gt_idx],
    n_events = configs$n_events[gt_idx]
  )

  # Prepare DCA data
  dca_configs <- dca_data$data$configs
  prevalence <- dca_data$data$sample_prevalence
  dca_list <- lapply(1:nrow(dca_configs), function(i) {
    data.frame(
      config = dca_configs$name[i],
      threshold = dca_configs$thresholds[[i]],
      nb_model = dca_configs$nb_model[[i]],
      nb_treat_all = dca_configs$nb_treat_all[[i]],
      nb_treat_none = dca_configs$nb_treat_none[[i]],
      stringsAsFactors = FALSE
    )
  })
  dca_df <- bind_rows(dca_list) %>%
    mutate(config = factor(config, levels = unique(config)))

  # Create individual plots
  p_cal <- create_calibration(cal_df, annotations = cal_annotations, infographic = FALSE, show_legend = FALSE)
  p_dca <- create_dca(dca_df, prevalence = prevalence, infographic = FALSE, show_legend = TRUE)

  # Compose into combined figure (1x2 = side by side)
  # Force equal column widths - legend placement shouldn't affect panel size
  fig_cal_dca <- compose_figures(
    list(p_cal, p_dca),
    layout = "1x2",
    tag_levels = "A",
    panel_titles = c("Calibration", "Decision Curve Analysis"),
    tag_font = "Neue Haas Grotesk Display Pro",
    tag_size = 14,
    widths = c(1, 1)  # Equal column widths
  )

  # Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
  save_publication_figure(fig_cal_dca, "fig_calibration_dca_combined")
} else {
  warning("Calibration/DCA data not found. Skipping combined figure.")
}

# ==============================================================================
# REMAINING FIGURES - Run individual scripts
# ==============================================================================

message("")
message("--- Running remaining figure scripts ---")

# List of figure scripts to run (excluding combined figures: forest, calibration+DCA)
figure_scripts <- c(
  "fig_variance_decomposition.R",
  "fig_shap_beeswarm.R",
  "fig_specification_curve.R",
  "fig_heatmap_preprocessing.R",
  # "fig_calibration_stratos.R",  # Now in combined calibration+DCA
  # "fig_dca_stratos.R",          # Now in combined calibration+DCA
  "fig_factorial_matrix.R",
  "fig_featurization_comparison.R",
  "fig_fm_dashboard.R",
  "fig_prob_dist_by_outcome.R",
  "fig_raincloud_auroc.R",
  "fig_selective_classification.R",
  "fig_roc_rc_combined.R",
  "fig_shap_importance.R",
  "fig_shap_gt_vs_ensemble.R",
  "fig_shap_heatmap.R",
  "fig_vif_analysis.R",
  "fig_cd_preprocessing.R"
)

for (script in figure_scripts) {
  script_path <- file.path(PROJECT_ROOT, "src/r/figures", script)
  if (file.exists(script_path)) {
    message(sprintf("  Running: %s", script))
    tryCatch({
      source(script_path, local = new.env())
    }, error = function(e) {
      warning(sprintf("  FAILED: %s - %s", script, e$message))
    })
  } else {
    warning(sprintf("  NOT FOUND: %s", script))
  }
}

# ==============================================================================
# SUMMARY
# ==============================================================================

message("")
message("========================================")
message("All R Figures Complete")
message("========================================")

# List generated figures
output_dir <- file.path(PROJECT_ROOT, "figures/generated/ggplot2")
figures <- list.files(output_dir, pattern = "\\.png$", full.names = FALSE)
message(sprintf("Generated %d figures in %s", length(figures), output_dir))
