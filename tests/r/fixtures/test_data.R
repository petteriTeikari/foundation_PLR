# Test Data Fixtures for Figure System Tests
# ==========================================
# Provides minimal test data for TDD testing.

#' Create minimal test data for forest outlier plots
#' @return data.frame with required columns
setup_outlier_test_data <- function() {
  data.frame(
    outlier_method = c("pupil-gt", "MOMENT-gt-finetune", "LOF", "OneClassSVM"),
    outlier_display_name = c("Ground Truth", "MOMENT Fine-tuned", "LOF", "One-Class SVM"),
    auroc_mean = c(0.91, 0.89, 0.85, 0.82),
    auroc_ci_lo = c(0.88, 0.86, 0.81, 0.78),
    auroc_ci_hi = c(0.94, 0.92, 0.89, 0.86),
    category = c("Ground Truth", "Foundation Model", "Traditional", "Traditional"),
    n_configs = c(8, 8, 8, 8),
    stringsAsFactors = FALSE
  )
}

#' Create minimal test data for forest imputation plots
#' @return data.frame with required columns
setup_imputation_test_data <- function() {
  data.frame(
    imputation_method = c("pupil-gt", "SAITS", "CSDI", "TimesNet"),
    imputation_display_name = c("Ground Truth", "SAITS", "CSDI", "TimesNet"),
    auroc_mean = c(0.91, 0.90, 0.89, 0.88),
    auroc_ci_lo = c(0.88, 0.87, 0.86, 0.85),
    auroc_ci_hi = c(0.94, 0.93, 0.92, 0.91),
    category = c("Ground Truth", "Deep Learning", "Deep Learning", "Deep Learning"),
    n_configs = c(11, 11, 11, 11),
    stringsAsFactors = FALSE
  )
}
