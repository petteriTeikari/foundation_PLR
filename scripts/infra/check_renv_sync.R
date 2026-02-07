#!/usr/bin/env Rscript
# check_renv_sync.R - Verify renv.lock is in sync with R dependencies
#
# This script checks if any R packages are used in source files but missing from
# renv.lock. It does NOT modify renv.lock (run renv::snapshot() manually).
#
# Exit codes:
#   0 - All packages are locked
#   1 - Missing or unlocked packages found
#
# Usage:
#   Rscript scripts/check_renv_sync.R
#
# Part of pre-commit hooks for R reproducibility (R4R-inspired)
# See: manuscripts/foundationPLR/planning/r-code-reproducibility-analysis.md
#
# Created: 2026-02-01
# Author: Foundation PLR Team

# Suppress package startup messages
suppressPackageStartupMessages({
  library(renv)
})

# Find project root
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
setwd(PROJECT_ROOT)

cat("=== renv Sync Check ===\n")
cat("Project root:", PROJECT_ROOT, "\n")
cat("R version:", R.version.string, "\n\n")

# Check renv status
status <- tryCatch({
  renv::status()
}, error = function(e) {
  cat("ERROR: Failed to check renv status:", e$message, "\n")
  quit(status = 1)
})

# Analyze status
# renv::status() returns a list with synchronized, lockfile, library
if (isTRUE(status$synchronized)) {
  cat("✅ renv.lock is synchronized with project library\n")
  cat("   All R packages are properly locked.\n")
  quit(status = 0)
}

# Check for issues
has_issues <- FALSE

# Packages in lockfile but not in library (need install)
if (length(status$library$Packages) > 0) {
  missing_in_library <- status$library$Packages
  if (length(missing_in_library) > 0) {
    cat("⚠️  Packages in renv.lock but not installed:\n")
    for (pkg in names(missing_in_library)) {
      cat("   -", pkg, "\n")
    }
    cat("\n   Run: renv::restore()\n\n")
    has_issues <- TRUE
  }
}

# Packages used in project but not in lockfile (need snapshot)
if (!is.null(status$lockfile$Packages)) {
  missing_in_lockfile <- status$lockfile$Packages
  if (length(missing_in_lockfile) > 0) {
    cat("❌ Packages used in project but NOT in renv.lock:\n")
    for (pkg in names(missing_in_lockfile)) {
      cat("   -", pkg, "\n")
    }
    cat("\n   Run: renv::snapshot() to update renv.lock\n\n")
    has_issues <- TRUE
  }
}

if (has_issues) {
  cat("=== FAILED: renv.lock is out of sync ===\n")
  cat("\nTo fix:\n")
  cat("  1. Run: Rscript -e \"renv::snapshot()\"  # Update lockfile\n")
  cat("  2. Commit the updated renv.lock\n")
  cat("  3. Run this check again\n")
  quit(status = 1)
} else {
  cat("✅ renv.lock appears to be in sync\n")
  quit(status = 0)
}
