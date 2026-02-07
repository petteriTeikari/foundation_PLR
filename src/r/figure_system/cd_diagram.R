# CD Diagram Wrapper for Standard Demšar-Style Critical Difference Diagrams
# =========================================================================
# Uses scmamp::plotCD() with Foundation PLR styling.
#
# Reference: Demšar J (2006) "Statistical Comparisons of Classifiers over
# Multiple Data Sets" J. Mach. Learn. Res. 7, 1-30.
#
# Created: 2026-01-27
# Author: Foundation PLR Team

suppressPackageStartupMessages({
  library(scmamp)
  library(yaml)
})

# ==============================================================================
# PROJECT ROOT FINDER
# ==============================================================================

.find_project_root <- function() {
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

# ==============================================================================
# METHOD CATEGORY COLORS (for optional post-process coloring)
# ==============================================================================

#' Load method category mappings from YAML
#' @return List with outlier_detection and imputation method info
.load_method_categories <- function() {
  project_root <- .find_project_root()
  methods_path <- file.path(project_root, "configs/VISUALIZATION/methods.yaml")
  yaml::read_yaml(methods_path)
}

#' Load color definitions from YAML
#' @return Named list of color values
.load_color_palette <- function() {
  project_root <- .find_project_root()
  colors_path <- file.path(project_root, "configs/VISUALIZATION/colors.yaml")
  yaml::read_yaml(colors_path)
}

#' Get category color for a method
#' @param method_name Full method name
#' @param method_type Either "outlier_detection" or "imputation"
#' @return Hex color string
get_method_category_color <- function(method_name, method_type = "outlier_detection") {
  methods <- .load_method_categories()
  colors <- .load_color_palette()

  # Look up method info
  method_info <- methods[[method_type]][[method_name]]

  if (is.null(method_info)) {
    # Try to infer category from name
    if (grepl("ensemble|Ensemble", method_name, ignore.case = TRUE)) {
      return(colors$ensemble)
    } else if (grepl("MOMENT|UniTS", method_name)) {
      return(colors$fm_primary)
    } else if (grepl("TimesNet", method_name)) {
      return(colors$timesnet)
    } else if (grepl("LOF|SVM|SubPCA|PROPHET", method_name)) {
      return(colors$traditional)
    } else if (grepl("SAITS|CSDI", method_name)) {
      return(colors$dl_primary)
    } else if (grepl("pupil-gt|GT", method_name)) {
      return(colors$ground_truth)
    }
    return(colors$category_default)
  }

  # Get color from color_key
  color_key <- method_info$color_key
  if (!is.null(color_key) && !is.null(colors[[color_key]])) {
    return(colors[[color_key]])
  }

  # Fallback to category color
  category <- method_info$category
  category_key <- paste0("category_", category)
  if (!is.null(colors[[category_key]])) {
    return(colors[[category_key]])
  }

  return(colors$category_default)
}

#' Get category colors for all methods in a matrix
#' @param method_names Character vector of method names
#' @param method_type Either "outlier_detection" or "imputation"
#' @return Named character vector of colors
get_method_colors <- function(method_names, method_type = "outlier_detection") {
  colors <- sapply(method_names, function(m) {
    get_method_category_color(m, method_type)
  }, USE.NAMES = TRUE)
  return(colors)
}

#' Create a category legend for CD diagrams
#' @param method_type Either "outlier_detection" or "imputation"
#' @param include_categories Vector of category names to include (NULL = all)
#' @return NULL (draws legend on current device)
draw_category_legend <- function(method_type = "outlier_detection",
                                  include_categories = NULL) {
  colors <- .load_color_palette()

  # Define categories and their colors
  all_categories <- list(
    ground_truth = list(name = "Ground Truth", color = colors$ground_truth),
    foundation_model = list(name = "Foundation Model", color = colors$fm_primary),
    traditional = list(name = "Traditional", color = colors$traditional),
    deep_learning = list(name = "Deep Learning", color = colors$dl_primary),
    ensemble = list(name = "Ensemble", color = colors$ensemble)
  )

  # Filter to requested categories
  if (!is.null(include_categories)) {
    all_categories <- all_categories[names(all_categories) %in% include_categories]
  }

  # Draw legend
  n_cats <- length(all_categories)
  legend_text <- sapply(all_categories, function(x) x$name)
  legend_colors <- sapply(all_categories, function(x) x$color)

  legend("bottomright",
         legend = legend_text,
         fill = legend_colors,
         border = NA,
         bty = "n",
         cex = 0.8,
         title = "Method Category")
}

# ==============================================================================
# METHOD NAME ABBREVIATIONS
# ==============================================================================

#' Get abbreviated method names for CD diagrams
#' @return Named character vector mapping full names to short names
get_method_abbreviations <- function() {
  c(
    # Ground truth
    "pupil-gt" = "GT",
    "pupil-gt + pupil-gt" = "GT + GT",

    # Foundation models - outlier detection
    "MOMENT-gt-finetune" = "MOMENT-ft",
    "MOMENT-gt-zeroshot" = "MOMENT-zs",
    "MOMENT-orig-finetune" = "MOMENT-orig",
    "UniTS-gt-finetune" = "UniTS-ft",
    "UniTS-orig-finetune" = "UniTS-orig",
    "UniTS-orig-zeroshot" = "UniTS-zs",
    "TimesNet-gt" = "TimesNet-gt",
    "TimesNet-orig" = "TimesNet",

    # Traditional - outlier detection
    "LOF" = "LOF",
    "OneClassSVM" = "OC-SVM",
    "SubPCA" = "SubPCA",
    "PROPHET" = "Prophet",

    # Ensembles - outlier detection
    "ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune" = "Ens-Full",
    "ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune" = "Ens-FM",

    # Imputation methods
    "SAITS" = "SAITS",
    "CSDI" = "CSDI",
    "MOMENT-finetune" = "MOMENT-imp",
    "MOMENT-zeroshot" = "MOMENT-zs-imp",
    "linear" = "Linear",
    "TimesNet" = "TimesNet",
    "ensemble-CSDI-MOMENT-SAITS-TimesNet" = "Ens-Imp",

    # Combined pipelines (outlier + imputation)
    "pupil-gt + CSDI" = "GT+CSDI",
    "pupil-gt + SAITS" = "GT+SAITS",
    "pupil-gt + TimesNet" = "GT+TN",
    "pupil-gt + pupil-gt" = "GT+GT",
    "pupil-gt + linear" = "GT+Lin",
    "pupil-gt + MOMENT-finetune" = "GT+MOM-ft",
    "pupil-gt + MOMENT-zeroshot" = "GT+MOM-zs",
    "pupil-gt + ensemble-CSDI-MOMENT-SAITS-TimesNet" = "GT+EnsImp",
    "pupil-gt + ensemble-CSDI-MOMENT-SAITS" = "GT+EnsImp",
    "ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune + CSDI" = "EnsFull+CSDI",
    "ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune + TimesNet" = "EnsFull+TN",
    "ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune + SAITS" = "EnsFull+SAITS",
    "ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune + MOMENT-finetune" = "EnsFull+MOM",
    "ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune + ensemble-CSDI-MOMENT-SAITS-TimesNet" = "EnsFull+EnsImp",
    "ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune + CSDI" = "EnsFM+CSDI",
    "ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune + TimesNet" = "EnsFM+TN",
    "ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune + SAITS" = "EnsFM+SAITS",
    "ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune + MOMENT-finetune" = "EnsFM+MOM",
    "ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune + ensemble-CSDI-MOMENT-SAITS-TimesNet" = "EnsFM+EnsImp",
    "LOF + SAITS" = "LOF+SAITS",
    "LOF + CSDI" = "LOF+CSDI",
    "LOF + TimesNet" = "LOF+TN",
    "LOF + MOMENT-finetune" = "LOF+MOM",
    "OneClassSVM + SAITS" = "SVM+SAITS",
    "OneClassSVM + CSDI" = "SVM+CSDI",
    "MOMENT-gt-finetune + SAITS" = "MOM-ft+SAITS",
    "MOMENT-gt-finetune + CSDI" = "MOM-ft+CSDI",
    "MOMENT-gt-finetune + TimesNet" = "MOM-ft+TN",
    "MOMENT-gt-finetune + MOMENT-finetune" = "MOM-ft+MOM",
    "UniTS-gt-finetune + SAITS" = "UniTS-ft+SAITS",
    "UniTS-gt-finetune + CSDI" = "UniTS-ft+CSDI",
    "UniTS-gt-finetune + TimesNet" = "UniTS-ft+TN",
    "TimesNet-gt + SAITS" = "TN-gt+SAITS",
    "TimesNet-gt + TimesNet" = "TN-gt+TN",
    "TimesNet-orig + TimesNet" = "TN+TN",
    "TimesNet-orig + SAITS" = "TN+SAITS"
  )
}

#' Truncate long method names that weren't caught by abbreviations
#' @param names Character vector of method names
#' @param max_len Maximum length (default 14 for readability)
#' @return Character vector with truncated names
truncate_long_names <- function(names, max_len = 14) {
  sapply(names, function(n) {
    if (nchar(n) > max_len) {
      paste0(substr(n, 1, max_len - 2), "..")
    } else {
      n
    }
  }, USE.NAMES = FALSE)
}

#' Abbreviate method names
#' @param names Character vector of full method names
#' @param max_len Maximum length for names not in abbreviation table (default 14)
#' @return Character vector of abbreviated names
abbreviate_methods <- function(names, max_len = 14) {
  abbrevs <- get_method_abbreviations()
  result <- sapply(names, function(n) {
    if (n %in% names(abbrevs)) {
      abbrevs[[n]]
    } else {
      n
    }
  }, USE.NAMES = FALSE)
  # Truncate any remaining long names
  truncate_long_names(result, max_len)
}

# ==============================================================================
# CD DIAGRAM WRAPPER
# ==============================================================================

#' Create a standard Demšar-style Critical Difference diagram
#'
#' Uses scmamp::plotCD() to create a proper CD diagram with:
#' - Horizontal rank axis at top
#' - Method names extending with vertical+horizontal bars
#' - CD bracket showing critical difference
#' - Clique bars connecting methods not significantly different
#'
#' @param results_matrix Matrix with methods as COLUMNS and datasets/folds as ROWS.
#'   Each cell contains the performance value (e.g., AUROC) for that method on that fold.
#' @param alpha Significance level for Nemenyi test (default 0.05)
#' @param cex Text size multiplier (default 1.0)
#' @param abbreviate Whether to abbreviate method names (default TRUE)
#' @param descending If TRUE, higher values are better (default TRUE for AUROC)
#'
#' @return Invisibly returns the plot (base R graphics)
#' @export
#'
#' @examples
#' # Create results matrix: rows = folds, columns = methods
#' results <- matrix(c(
#'   0.91, 0.90, 0.88, 0.85,  # Fold 1
#'   0.92, 0.89, 0.87, 0.84,  # Fold 2
#'   0.90, 0.91, 0.86, 0.83   # Fold 3
#' ), nrow = 3, byrow = TRUE)
#' colnames(results) <- c("GT", "Ensemble", "MOMENT", "LOF")
#'
#' create_cd_diagram(results)

# Load color definitions from YAML for CD diagrams
.load_cd_colors <- function() {
  project_root <- .find_project_root()
  combos_path <- file.path(project_root, "configs/VISUALIZATION/plot_hyperparam_combos.yaml")
  config <- yaml::read_yaml(combos_path)
  config$color_definitions
}

# Economist-style colors for CD diagrams (loaded from YAML)
.cd_color_defs <- NULL
.get_cd_colors <- function() {
  if (is.null(.cd_color_defs)) {
    .cd_color_defs <<- .load_cd_colors()
  }
  .cd_color_defs
}

# Accessor functions for CD colors
CD_BACKGROUND <- function() .get_cd_colors()[["--color-background"]]    # Off-white (Economist background)
CD_LINE_COLOR <- function() .get_cd_colors()[["--color-text-primary"]]  # Dark gray for lines
CD_TEXT_COLOR <- function() .get_cd_colors()[["--color-text-primary"]]  # Dark gray for text
CD_CLIQUE_COLOR <- function() .get_cd_colors()[["--color-primary"]]     # Economist blue for clique bars

create_cd_diagram <- function(results_matrix,
                               alpha = 0.05,
                               cex = 1.0,
                               abbreviate = TRUE,
                               descending = TRUE,
                               left_margin = 8,
                               right_margin = 8,
                               reset_par = TRUE,
                               show_category_legend = FALSE,
                               method_type = "outlier_detection") {

  # Validate input
  if (!is.matrix(results_matrix)) {
    stop("results_matrix must be a matrix")
  }

  if (is.null(colnames(results_matrix))) {
    stop("results_matrix must have column names (method names)")
  }

  if (nrow(results_matrix) < 2) {
    stop("results_matrix must have at least 2 rows (datasets/folds)")
  }

  if (ncol(results_matrix) < 2) {
    stop("results_matrix must have at least 2 columns (methods)")
  }

  # Abbreviate method names if requested
  if (abbreviate) {
    colnames(results_matrix) <- abbreviate_methods(colnames(results_matrix))
  }

  # Calculate dynamic margins based on method name lengths
  max_name_len <- max(nchar(colnames(results_matrix)))
  dynamic_margin <- max(left_margin, max_name_len * 0.6)

  # Set Economist-style graphics parameters BEFORE plotting
  # Only save/restore par if reset_par is TRUE (standalone use)
  # For multi-panel layouts, caller manages par settings
  if (reset_par) {
    old_par <- par(no.readonly = TRUE)
    on.exit(par(old_par))
  }

  par(
    bg = CD_BACKGROUND(),                      # Off-white background
    mar = c(4, dynamic_margin, 4, dynamic_margin),  # Dynamic margins for long names
    family = "sans",                           # Clean sans-serif font
    col.axis = CD_TEXT_COLOR(),                # Axis color
    col.lab = CD_TEXT_COLOR(),                 # Label color
    col.main = CD_TEXT_COLOR(),                # Title color
    fg = CD_LINE_COLOR()                       # Foreground color for lines
  )

  # scmamp::plotCD expects (from source code analysis):
  # - Rows = datasets/problems (what we call folds)
  # - Columns = algorithms/methods (what we want to compare)
  # This matches our input format, so NO TRANSPOSE needed!

  # Create the CD diagram using scmamp
  # Note: scmamp's rankMatrix ranks HIGHER values as rank 1 (best)
  # So for descending metrics (AUROC - higher is better), we use values as-is
  # For ascending metrics (lower is better), we negate to flip
  results_for_plot <- results_matrix
  if (!descending) {
    # Lower is better - negate so scmamp ranks correctly
    results_for_plot <- -results_for_plot
  }

  # Plot with scmamp
  scmamp::plotCD(
    results.matrix = results_for_plot,
    alpha = alpha,
    cex = cex
  )

  # Optionally add category legend
  if (show_category_legend) {
    # Determine which categories are present
    orig_names <- colnames(results_matrix)
    if (abbreviate) {
      # Need to map back to original names for category lookup
      abbrevs <- get_method_abbreviations()
      inv_abbrevs <- setNames(names(abbrevs), abbrevs)
    }

    draw_category_legend(method_type = method_type)
  }

  invisible(results_matrix)
}

#' Save a CD diagram to file
#'
#' Creates a standard Demšar-style CD diagram and saves it.
#'
#' @param results_matrix Matrix with methods as COLUMNS and datasets/folds as ROWS
#' @param filename Base filename (without extension)
#' @param output_dir Output directory (default: from figure system config)
#' @param width Figure width in inches (default 10)
#' @param height Figure height in inches (default 6)
#' @param alpha Significance level (default 0.05)
#' @param abbreviate Whether to abbreviate method names (default TRUE)
#' @param descending If TRUE, higher values are better (default TRUE)
#'
#' @return Invisibly returns the output path
#' @export
save_cd_diagram <- function(results_matrix,
                             filename,
                             output_dir = NULL,
                             width = 10,
                             height = 6,
                             alpha = 0.05,
                             abbreviate = TRUE,
                             descending = TRUE) {

  # Get output directory from figure system if not specified
  if (is.null(output_dir)) {
    project_root <- .find_project_root()
    # Check if figure is in a category
    config_path <- file.path(project_root, "configs/VISUALIZATION/figure_layouts.yaml")
    if (file.exists(config_path)) {
      config <- yaml::read_yaml(config_path)
      for (cat_name in names(config$figure_categories)) {
        cat_info <- config$figure_categories[[cat_name]]
        if (filename %in% cat_info$figures) {
          output_dir <- file.path(project_root, cat_info$output_dir)
          break
        }
      }
    }
    # Fallback to supplementary
    if (is.null(output_dir)) {
      output_dir <- file.path(project_root, "figures/generated/ggplot2/supplementary")
    }
  }

  # Ensure output directory exists
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  # Save as PNG
  output_path <- file.path(output_dir, paste0(filename, ".png"))

  png(output_path, width = width, height = height, units = "in", res = 300)
  create_cd_diagram(
    results_matrix = results_matrix,
    alpha = alpha,
    abbreviate = abbreviate,
    descending = descending
  )
  dev.off()

  message("Saved CD diagram: ", output_path)

  invisible(output_path)
}

# ==============================================================================
# DATA PREPARATION HELPERS
# ==============================================================================

#' Prepare results matrix from long-format data frame
#'
#' @param df Data frame with method, fold, and metric columns
#' @param method_col Name of method column
#' @param fold_col Name of fold/dataset column
#' @param metric_col Name of metric column (e.g., "auroc")
#'
#' @return Matrix suitable for create_cd_diagram()
#' @export
prepare_cd_matrix <- function(df, method_col, fold_col, metric_col) {
  # Pivot to wide format: rows = folds, columns = methods
  wide_df <- tidyr::pivot_wider(
    df,
    id_cols = all_of(fold_col),
    names_from = all_of(method_col),
    values_from = all_of(metric_col)
  )

  # Convert to matrix
  mat <- as.matrix(wide_df[, -1])
  rownames(mat) <- wide_df[[fold_col]]

  return(mat)
}

# ==============================================================================
# MAIN EXECUTION (when sourced as script)
# ==============================================================================

if (sys.nframe() == 0) {
  message("CD Diagram module loaded.")
  message("Usage: create_cd_diagram(results_matrix) or save_cd_diagram(results_matrix, 'filename')")

  # Demo with example data
  set.seed(42)
  demo_results <- matrix(
    c(
      0.91, 0.90, 0.88, 0.86, 0.85,
      0.92, 0.89, 0.87, 0.85, 0.84,
      0.90, 0.91, 0.86, 0.87, 0.83,
      0.89, 0.88, 0.89, 0.84, 0.82,
      0.91, 0.90, 0.85, 0.86, 0.84
    ),
    nrow = 5, byrow = TRUE
  )
  colnames(demo_results) <- c("pupil-gt", "MOMENT-gt-finetune", "UniTS-gt-finetune", "LOF", "OneClassSVM")
  rownames(demo_results) <- paste0("Fold_", 1:5)

  message("\nDemo matrix:")
  print(demo_results)

  message("\nCreating demo CD diagram...")
  png("/tmp/demo_cd_proper.png", width = 10, height = 6, units = "in", res = 150)
  create_cd_diagram(demo_results, abbreviate = TRUE)
  dev.off()
  message("Demo saved to /tmp/demo_cd_proper.png")
}
