# Load display names from YAML
# =============================
# This ensures R plots use the same display names as Python.
# See: configs/mlflow_registry/display_names.yaml
# See: docs/planning/lookup-model-names.md for design decisions.

library(yaml)

#' Load display names configuration from YAML
#'
#' @return Named list with outlier_methods, imputation_methods, classifiers, categories
#' @export
load_display_names <- function() {
  yaml_path <- "configs/mlflow_registry/display_names.yaml"

  if (!file.exists(yaml_path)) {
    stop(paste("display_names.yaml not found at:", yaml_path,
               "\nRun: pytest tests/unit/test_display_names.py -v to check setup."))
  }

  yaml::read_yaml(yaml_path)
}

#' Get display name for an outlier detection method
#'
#' @param method Raw MLflow method name (e.g., "MOMENT-gt-finetune")
#' @return Publication-friendly display name (e.g., "MOMENT Fine-tuned")
#' @export
get_outlier_display_name <- function(method) {
  names <- load_display_names()
  display <- names$outlier_methods[[method]]

  if (is.null(display)) {
    warning(paste("No display name for outlier method:", method))
    return(method)
  }

  display
}

#' Get display name for an imputation method
#'
#' @param method Raw MLflow method name (e.g., "MOMENT-finetune")
#' @return Publication-friendly display name (e.g., "MOMENT Fine-tuned")
#' @export
get_imputation_display_name <- function(method) {
  names <- load_display_names()
  display <- names$imputation_methods[[method]]

  if (is.null(display)) {
    warning(paste("No display name for imputation method:", method))
    return(method)
  }

  display
}

#' Get display name for a classifier
#'
#' @param classifier Raw classifier name (e.g., "LogisticRegression")
#' @return Publication-friendly display name (e.g., "Logistic Regression")
#' @export
get_classifier_display_name <- function(classifier) {
  names <- load_display_names()
  display <- names$classifiers[[classifier]]

  if (is.null(display)) {
    warning(paste("No display name for classifier:", classifier))
    return(classifier)
  }

  display
}

#' Get display name for a method category
#'
#' @param category Internal category name (e.g., "foundation_model")
#' @return Publication-friendly display name (e.g., "Foundation Model")
#' @export
get_category_display_name <- function(category) {
  names <- load_display_names()
  display <- names$categories[[category]]

  if (is.null(display)) {
    warning(paste("No display name for category:", category))
    return(category)
  }

  display
}

#' Apply display names to a data frame
#'
#' Adds *_display columns for outlier_method, imputation_method, and classifier.
#'
#' @param df Data frame with outlier_method, imputation_method, and/or classifier columns
#' @return Data frame with added display name columns
#' @export
apply_display_names <- function(df) {
  names <- load_display_names()

  # Add outlier display name
  if ("outlier_method" %in% colnames(df)) {
    df$outlier_display <- sapply(df$outlier_method, function(m) {
      d <- names$outlier_methods[[m]]
      if (is.null(d)) m else d
    })
  }

  # Add imputation display name
  if ("imputation_method" %in% colnames(df)) {
    df$imputation_display <- sapply(df$imputation_method, function(m) {
      d <- names$imputation_methods[[m]]
      if (is.null(d)) m else d
    })
  }

  # Add classifier display name
  if ("classifier" %in% colnames(df)) {
    df$classifier_display <- sapply(df$classifier, function(c) {
      d <- names$classifiers[[c]]
      if (is.null(d)) c else d
    })
  }

  df
}
