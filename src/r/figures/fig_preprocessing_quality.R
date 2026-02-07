#!/usr/bin/env Rscript
# ==============================================================================
# Preprocessing Quality Comparison Figures
# Plan 04: Supplementary figures emphasizing TSFM preprocessing as core contribution
#
# Creates:
#   - fig_outlier_detection_quality.png: Outlier method comparison
#   - fig_imputation_quality.png: Imputation method comparison
#
# IMPORTANT: This script follows anti-hardcoding rules (CRITICAL-FAILURE-004)
# - NO hardcoded hex colors (uses color_defs from YAML)
# - NO custom theme definitions (uses theme_foundation_plr)
# - Uses save_publication_figure() from figure system
#
# Run: Rscript src/r/figures/fig_preprocessing_quality.R
# ==============================================================================

# ==============================================================================
# MANDATORY HEADER - DO NOT REMOVE
# ==============================================================================

# Find project root
PROJECT_ROOT <- (function() {
  d <- getwd()
  while (d != dirname(d)) {
    if (file.exists(file.path(d, "CLAUDE.md"))) return(d)
    d <- dirname(d)
  }
  stop("Could not find project root (no CLAUDE.md found)")
})()
setwd(PROJECT_ROOT)

# Load figure system (MANDATORY)
source(file.path(PROJECT_ROOT, "src/r/figure_system/config_loader.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/save_figure.R"))
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))

# Load colors from YAML (MANDATORY - no hardcoded colors!)
color_defs <- load_color_definitions()

# ==============================================================================
# END MANDATORY HEADER
# ==============================================================================

# --- Load Libraries ---
suppressPackageStartupMessages({
  library(ggplot2)
  library(jsonlite)
  library(dplyr)
  library(forcats)
})

# --- Category Colors from YAML ---
# Map category names to semantic color references
CATEGORY_COLORS <- c(
  "Ground Truth" = color_defs[["--color-ground-truth"]],
  "Foundation Model" = color_defs[["--color-fm-primary"]],
  "Ensemble" = color_defs[["--color-positive"]],
  "Deep Learning" = color_defs[["--color-fm-secondary"]],
  "Traditional" = color_defs[["--color-traditional"]]
)

# --- Load Data ---
cat("Loading preprocessing quality metrics...\n")
json_path <- file.path(PROJECT_ROOT, "figures/generated/data/fig_preprocessing_quality.json")

# Check alternate location
if (!file.exists(json_path)) {
  alt_path <- file.path(PROJECT_ROOT, "data/r_data/preprocessing_quality_metrics.json")
  if (file.exists(alt_path)) {
    json_path <- alt_path
  } else {
    stop("Data file not found. Run the extraction script first.")
  }
}

data <- fromJSON(json_path)

# --- Valid Methods (Registry) ---
# From configs/mlflow_registry/parameters/classification.yaml
VALID_OUTLIERS <- c(
  "pupil-gt",
  "MOMENT-gt-finetune", "MOMENT-gt-zeroshot",
  "UniTS-gt-finetune",
  "TimesNet-gt",
  "LOF", "OneClassSVM", "PROPHET", "SubPCA",
  "ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune",
  "ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune"
)

# --- Process Outlier Detection Data ---
outlier_df <- as.data.frame(data$outlier_detection)
outlier_df <- outlier_df %>%
  filter(outlier_method %in% VALID_OUTLIERS) %>%
  mutate(
    # Short names for display
    method_short = case_when(
      outlier_method == "pupil-gt" ~ "Ground Truth",
      outlier_method == "MOMENT-gt-finetune" ~ "MOMENT (ft)",
      outlier_method == "MOMENT-gt-zeroshot" ~ "MOMENT (zs)",
      outlier_method == "UniTS-gt-finetune" ~ "UniTS (ft)",
      outlier_method == "TimesNet-gt" ~ "TimesNet",
      outlier_method == "LOF" ~ "LOF",
      outlier_method == "OneClassSVM" ~ "OneClass SVM",
      outlier_method == "PROPHET" ~ "PROPHET",
      outlier_method == "SubPCA" ~ "SubPCA",
      grepl("^ensemble-LOF", outlier_method) ~ "Ensemble (all)",
      grepl("^ensembleThresholded", outlier_method) ~ "Ensemble (thresh)",
      TRUE ~ outlier_method
    ),
    # Categorize
    category = case_when(
      outlier_method == "pupil-gt" ~ "Ground Truth",
      grepl("ensemble", outlier_method, ignore.case = TRUE) ~ "Ensemble",
      grepl("MOMENT|UniTS|TimesNet", outlier_method) ~ "Foundation Model",
      TRUE ~ "Traditional"
    )
  )

cat(sprintf("Valid outlier methods: %d\n", nrow(outlier_df)))

# --- Process Imputation Data ---
imputation_df <- as.data.frame(data$imputation)
imputation_df <- imputation_df %>%
  mutate(
    method_short = case_when(
      imputation_method == "pupil-gt" ~ "Ground Truth",
      imputation_method == "MOMENT-finetune" ~ "MOMENT (ft)",
      imputation_method == "MOMENT-zeroshot" ~ "MOMENT (zs)",
      imputation_method == "SAITS" ~ "SAITS",
      imputation_method == "CSDI" ~ "CSDI",
      imputation_method == "TimesNet" ~ "TimesNet",
      grepl("ensemble", imputation_method, ignore.case = TRUE) ~ "Ensemble",
      TRUE ~ imputation_method
    ),
    category = case_when(
      imputation_method == "pupil-gt" ~ "Ground Truth",
      grepl("ensemble", imputation_method, ignore.case = TRUE) ~ "Ensemble",
      TRUE ~ "Deep Learning"
    )
  )

cat(sprintf("Imputation methods: %d\n", nrow(imputation_df)))

# ==============================================================================
# FIGURE 1: Outlier Detection Quality
# ==============================================================================
cat("\nCreating outlier detection quality figure...\n")

p_outlier <- ggplot(outlier_df, aes(x = reorder(method_short, auroc), y = auroc, fill = category)) +
  geom_col(width = 0.7) +
  geom_errorbar(
    aes(ymin = auroc_ci_lo, ymax = auroc_ci_hi),
    width = 0.2, color = color_defs[["--color-text-muted"]], linewidth = 0.4
  ) +
  geom_hline(
    yintercept = outlier_df$auroc[outlier_df$outlier_method == "pupil-gt"],
    linetype = "dashed", color = color_defs[["--color-ground-truth"]], linewidth = 0.6
  ) +
  coord_flip(ylim = c(0.82, 0.92)) +
  scale_fill_manual(values = CATEGORY_COLORS, name = "Method Type") +
  labs(
    title = "Outlier Detection Method Comparison",
    subtitle = "Downstream AUROC (CatBoost + handcrafted features)",
    x = NULL,
    y = "AUROC",
    caption = "Dashed line: Ground truth performance. Error bars: 95% CI."
  ) +
  theme_foundation_plr()

# Save using figure system
# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(p_outlier, "fig_outlier_detection_quality")
cat("Saved: fig_outlier_detection_quality\n")

# ==============================================================================
# FIGURE 2: Imputation Quality
# ==============================================================================
cat("\nCreating imputation quality figure...\n")

p_imputation <- ggplot(imputation_df, aes(x = reorder(method_short, auroc), y = auroc, fill = category)) +
  geom_col(width = 0.7) +
  geom_errorbar(
    aes(ymin = auroc_ci_lo, ymax = auroc_ci_hi),
    width = 0.2, color = color_defs[["--color-text-muted"]], linewidth = 0.4
  ) +
  geom_hline(
    yintercept = imputation_df$auroc[imputation_df$imputation_method == "pupil-gt"],
    linetype = "dashed", color = color_defs[["--color-ground-truth"]], linewidth = 0.6
  ) +
  coord_flip(ylim = c(0.85, 0.92)) +
  scale_fill_manual(values = CATEGORY_COLORS, name = "Method Type") +
  labs(
    title = "Imputation Method Comparison",
    subtitle = "Downstream AUROC (CatBoost + handcrafted features)",
    x = NULL,
    y = "AUROC",
    caption = "Dashed line: Ground truth performance. Error bars: 95% CI."
  ) +
  theme_foundation_plr()

# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(p_imputation, "fig_imputation_quality")
cat("Saved: fig_imputation_quality\n")

# ==============================================================================
# FIGURE 3: Combined Preprocessing Summary
# ==============================================================================
cat("\nCreating combined preprocessing summary figure...\n")

# Calculate category means
outlier_summary <- outlier_df %>%
  group_by(category) %>%
  summarise(
    mean_auroc = mean(auroc),
    min_auroc = min(auroc),
    max_auroc = max(auroc),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(stage = "Outlier Detection")

imputation_summary <- imputation_df %>%
  group_by(category) %>%
  summarise(
    mean_auroc = mean(auroc),
    min_auroc = min(auroc),
    max_auroc = max(auroc),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(stage = "Imputation")

combined_summary <- bind_rows(outlier_summary, imputation_summary)

p_combined <- ggplot(combined_summary, aes(x = category, y = mean_auroc, fill = category)) +
  geom_col(width = 0.7) +
  geom_errorbar(
    aes(ymin = min_auroc, ymax = max_auroc),
    width = 0.2, color = color_defs[["--color-text-muted"]], linewidth = 0.4
  ) +
  facet_wrap(~stage, scales = "free_x") +
  coord_cartesian(ylim = c(0.84, 0.92)) +
  scale_fill_manual(values = CATEGORY_COLORS, guide = "none") +
  labs(
    title = "Preprocessing Quality by Method Category",
    subtitle = "Mean AUROC with range across methods",
    x = NULL,
    y = "AUROC"
  ) +
  theme_foundation_plr() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(p_combined, "fig_preprocessing_summary")
cat("Saved: fig_preprocessing_summary\n")

# --- Save JSON metadata ---
output_dir <- file.path(PROJECT_ROOT, "figures/generated/data")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

json_out <- list(
  metadata = list(
    created = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    generator = "src/r/figures/fig_preprocessing_quality.R",
    description = "Preprocessing quality comparison figures"
  ),
  outlier_detection = outlier_df,
  imputation = imputation_df,
  category_summary = combined_summary
)

json_out_path <- file.path(output_dir, "fig_preprocessing_quality.json")
write_json(json_out, json_out_path, pretty = TRUE, auto_unbox = TRUE)
cat("Saved: data/fig_preprocessing_quality.json\n")

cat("\n=== DONE ===\n")
