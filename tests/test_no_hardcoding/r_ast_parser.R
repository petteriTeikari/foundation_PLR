# R AST Parser for Python Tests
# ==============================
# Uses R's native parse() function to extract string literals from R code.
# Called by Python tests to avoid regex-based code analysis.
#
# Usage: Rscript r_ast_parser.R <file_path> <mode>
# Modes:
#   - "strings": Extract all string literals
#   - "hex_colors": Extract strings matching hex color pattern
#   - "function_calls": Extract function call names
#   - "case_when_grepl": Detect case_when with grepl categorization
#
# Output: JSON array of {line, value} objects

suppressPackageStartupMessages({
  library(jsonlite)
})

#' Extract all string literals from an R file using AST
#'
#' @param file_path Path to R file
#' @return Data frame with columns: line, value
extract_strings <- function(file_path) {
  # Parse the file into an AST
  parsed <- tryCatch(
    parse(file_path, keep.source = TRUE),
    error = function(e) {
      return(NULL)
    }
  )

  if (is.null(parsed)) {
    return(data.frame(line = integer(), value = character(), stringsAsFactors = FALSE))
  }

  # Get source references
  src_ref <- attr(parsed, "srcref")

  # Use lists instead of data.frame rbind for better performance
  lines_list <- list()
  values_list <- list()
  idx <- 0

  # Recursive function to walk the AST
  walk_expr <- function(expr, default_line = 1) {
    if (is.null(expr)) return()

    # Get line number from srcref if available
    line_num <- default_line
    sr <- attr(expr, "srcref")
    if (!is.null(sr) && length(sr) >= 1) {
      line_num <- as.integer(sr[1])
    }

    # Check if this is a string literal
    if (is.character(expr) && length(expr) == 1 && !is.na(expr)) {
      idx <<- idx + 1
      lines_list[[idx]] <<- line_num
      values_list[[idx]] <<- expr
      return()
    }

    # Recurse into calls
    if (is.call(expr)) {
      for (i in seq_along(expr)) {
        tryCatch({
          child <- expr[[i]]
          walk_expr(child, line_num)
        }, error = function(e) NULL)
      }
    }

    # Recurse into expressions
    if (is.expression(expr)) {
      for (i in seq_along(expr)) {
        tryCatch({
          walk_expr(expr[[i]], line_num)
        }, error = function(e) NULL)
      }
    }

    # Handle pairlists (function arguments)
    if (is.pairlist(expr)) {
      for (i in seq_along(expr)) {
        tryCatch({
          walk_expr(expr[[i]], line_num)
        }, error = function(e) NULL)
      }
    }
  }

  # Walk each top-level expression
  for (i in seq_along(parsed)) {
    expr <- parsed[[i]]
    line <- 1
    if (!is.null(src_ref) && length(src_ref) >= i) {
      sr <- src_ref[[i]]
      if (!is.null(sr) && length(sr) >= 1) {
        line <- as.integer(sr[1])
      }
    }
    tryCatch({
      walk_expr(expr, line)
    }, error = function(e) NULL)
  }

  # Convert lists to data frame
  if (idx > 0) {
    return(data.frame(
      line = unlist(lines_list),
      value = unlist(values_list),
      stringsAsFactors = FALSE
    ))
  }

  return(data.frame(line = integer(), value = character(), stringsAsFactors = FALSE))
}

#' Extract hex color strings from an R file
#'
#' @param file_path Path to R file
#' @return Data frame with columns: line, value (only hex colors)
extract_hex_colors <- function(file_path) {
  strings <- extract_strings(file_path)

  if (nrow(strings) == 0) {
    return(strings)
  }

  # Filter to hex colors (case insensitive)
  hex_mask <- grepl("^#[0-9A-Fa-f]{6}$", strings$value)
  return(strings[hex_mask, , drop = FALSE])
}

#' Extract function calls from an R file
#'
#' @param file_path Path to R file
#' @return Data frame with columns: line, value (function names)
extract_function_calls <- function(file_path) {
  parsed <- tryCatch(
    parse(file_path, keep.source = TRUE),
    error = function(e) NULL
  )

  if (is.null(parsed)) {
    return(data.frame(line = integer(), value = character(), stringsAsFactors = FALSE))
  }

  src_ref <- attr(parsed, "srcref")
  lines_list <- list()
  values_list <- list()
  idx <- 0

  walk_for_calls <- function(expr, default_line = 1) {
    if (is.null(expr)) return()

    line_num <- default_line
    sr <- attr(expr, "srcref")
    if (!is.null(sr) && length(sr) >= 1) {
      line_num <- as.integer(sr[1])
    }

    if (is.call(expr)) {
      fn <- expr[[1]]
      if (is.symbol(fn)) {
        fn_name <- as.character(fn)
        idx <<- idx + 1
        lines_list[[idx]] <<- line_num
        values_list[[idx]] <<- fn_name
      }
      for (i in seq_along(expr)) {
        tryCatch({
          walk_for_calls(expr[[i]], line_num)
        }, error = function(e) NULL)
      }
    }

    if (is.expression(expr)) {
      for (i in seq_along(expr)) {
        tryCatch({
          walk_for_calls(expr[[i]], line_num)
        }, error = function(e) NULL)
      }
    }
  }

  for (i in seq_along(parsed)) {
    line <- 1
    if (!is.null(src_ref) && length(src_ref) >= i) {
      sr <- src_ref[[i]]
      if (!is.null(sr) && length(sr) >= 1) {
        line <- as.integer(sr[1])
      }
    }
    tryCatch({
      walk_for_calls(parsed[[i]], line)
    }, error = function(e) NULL)
  }

  if (idx > 0) {
    return(data.frame(
      line = unlist(lines_list),
      value = unlist(values_list),
      stringsAsFactors = FALSE
    ))
  }

  return(data.frame(line = integer(), value = character(), stringsAsFactors = FALSE))
}

#' Extract case_when calls that contain grepl patterns
#'
#' Detects the banned pattern: case_when(grepl(...) ~ "Category")
#'
#' @param file_path Path to R file
#' @return Data frame with columns: line, value (the category string)
extract_case_when_grepl <- function(file_path) {
  parsed <- tryCatch(
    parse(file_path, keep.source = TRUE),
    error = function(e) NULL
  )

  if (is.null(parsed)) {
    return(data.frame(line = integer(), value = character(), stringsAsFactors = FALSE))
  }

  src_ref <- attr(parsed, "srcref")
  lines_list <- list()
  values_list <- list()
  idx <- 0

  # Category strings we're looking for
  banned_categories <- c("Ground Truth", "Traditional", "Foundation Model",
                         "Ensemble", "Deep Learning", "FM Pipeline")

  walk_for_case_when <- function(expr, default_line = 1) {
    if (is.null(expr)) return()

    line_num <- default_line
    sr <- attr(expr, "srcref")
    if (!is.null(sr) && length(sr) >= 1) {
      line_num <- as.integer(sr[1])
    }

    if (is.call(expr)) {
      fn <- expr[[1]]
      fn_name <- if (is.symbol(fn)) as.character(fn) else ""

      # Check if this is a case_when call
      if (fn_name == "case_when") {
        # Look for grepl and category strings in the arguments
        has_grepl <- FALSE
        category_found <- NULL

        check_for_patterns <- function(e) {
          if (is.call(e)) {
            name <- if (is.symbol(e[[1]])) as.character(e[[1]]) else ""
            if (name == "grepl") {
              has_grepl <<- TRUE
            }
            for (i in seq_along(e)) {
              tryCatch({
                check_for_patterns(e[[i]])
              }, error = function(err) NULL)
            }
          }
          if (is.character(e) && length(e) == 1 && e %in% banned_categories) {
            category_found <<- e
          }
        }

        for (i in seq_along(expr)[-1]) {
          tryCatch({
            check_for_patterns(expr[[i]])
          }, error = function(err) NULL)
        }

        if (has_grepl && !is.null(category_found)) {
          idx <<- idx + 1
          lines_list[[idx]] <<- line_num
          values_list[[idx]] <<- category_found
        }
      }

      # Recurse into all sub-expressions
      for (i in seq_along(expr)) {
        tryCatch({
          walk_for_case_when(expr[[i]], line_num)
        }, error = function(e) NULL)
      }
    }

    if (is.expression(expr)) {
      for (i in seq_along(expr)) {
        tryCatch({
          walk_for_case_when(expr[[i]], line_num)
        }, error = function(e) NULL)
      }
    }
  }

  for (i in seq_along(parsed)) {
    line <- 1
    if (!is.null(src_ref) && length(src_ref) >= i) {
      sr <- src_ref[[i]]
      if (!is.null(sr) && length(sr) >= 1) {
        line <- as.integer(sr[1])
      }
    }
    tryCatch({
      walk_for_case_when(parsed[[i]], line)
    }, error = function(e) NULL)
  }

  if (idx > 0) {
    return(data.frame(
      line = unlist(lines_list),
      value = unlist(values_list),
      stringsAsFactors = FALSE
    ))
  }

  return(data.frame(line = integer(), value = character(), stringsAsFactors = FALSE))
}

# Main entry point
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  cat('{"error": "Usage: Rscript r_ast_parser.R <file_path> <mode>"}')
  quit(status = 1)
}

file_path <- args[1]
mode <- args[2]

if (!file.exists(file_path)) {
  cat(sprintf('{"error": "File not found: %s"}', file_path))
  quit(status = 1)
}

result <- switch(mode,
  "strings" = extract_strings(file_path),
  "hex_colors" = extract_hex_colors(file_path),
  "function_calls" = extract_function_calls(file_path),
  "case_when_grepl" = extract_case_when_grepl(file_path),
  {
    cat(sprintf('{"error": "Unknown mode: %s"}', mode))
    quit(status = 1)
  }
)

# Output as JSON
cat(toJSON(result, auto_unbox = TRUE))
