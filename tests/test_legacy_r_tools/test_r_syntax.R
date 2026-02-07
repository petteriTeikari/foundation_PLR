#!/usr/bin/env Rscript
# test_r_syntax.R
# Smoke tests for legacy R tools - verifies R files parse without syntax errors
#
# Usage:
#   Rscript test_r_syntax.R [project_root]
#
# Returns exit code 0 if all files parse, non-zero otherwise

# Get the project root from command line argument or determine from script location
args <- commandArgs(trailingOnly = TRUE)

if (length(args) > 0) {
  project_root <- normalizePath(args[1])
} else {
  # Try to determine from script location
  script_path <- NULL

  # Method 1: commandArgs
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", cmd_args, value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- sub("--file=", "", file_arg[1])
  }

  if (!is.null(script_path) && file.exists(script_path)) {
    script_dir <- dirname(normalizePath(script_path))
    project_root <- normalizePath(file.path(script_dir, "..", ".."))
  } else {
    # Fallback: assume running from project root
    project_root <- getwd()
  }
}

# Directory containing legacy R tools
tools_dir <- file.path(project_root, "src", "tools", "ground-truth-creation")

# Check if tools directory exists
if (!dir.exists(tools_dir)) {
  cat(sprintf("ERROR: Tools directory not found: %s\n", tools_dir))
  cat(sprintf("Project root: %s\n", project_root))
  quit(status = 1)
}

# Find all R files
r_files <- list.files(
  tools_dir,
  pattern = "\\.R$",
  recursive = TRUE,
  full.names = TRUE
)

cat("=== R Syntax Smoke Tests ===\n")
cat(sprintf("Project root: %s\n", project_root))
cat(sprintf("Tools directory: %s\n", tools_dir))
cat(sprintf("Found %d R files to test\n\n", length(r_files)))

if (length(r_files) == 0) {
  cat("WARNING: No R files found to test\n")
  quit(status = 0)
}

# Track results
passed <- 0
failed <- 0
failed_files <- c()

for (r_file in r_files) {
  relative_path <- sub(paste0(project_root, "/"), "", r_file)
  cat(sprintf("Testing: %s ... ", relative_path))

  # Try to parse the file
  result <- tryCatch({
    parse(file = r_file)
    TRUE
  }, error = function(e) {
    cat(sprintf("\n  ERROR: %s\n", e$message))
    FALSE
  })

  if (result) {
    cat("OK\n")
    passed <- passed + 1
  } else {
    cat("FAILED\n")
    failed <- failed + 1
    failed_files <- c(failed_files, relative_path)
  }
}

# Summary
cat("\n=== Summary ===\n")
cat(sprintf("Passed: %d\n", passed))
cat(sprintf("Failed: %d\n", failed))

if (failed > 0) {
  cat("\nFailed files:\n")
  for (f in failed_files) {
    cat(sprintf("  - %s\n", f))
  }
  quit(status = 1)
} else {
  cat("\nAll R files parse successfully!\n")
  quit(status = 0)
}
