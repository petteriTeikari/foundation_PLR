#!/usr/bin/env Rscript
# ==============================================================================
# Validate R Figure Scripts for Hardcoding Anti-Patterns
# ==============================================================================
#
# Checks that all R figure scripts:
# 1. Source config_loader.R and save_figure.R
# 2. Use save_publication_figure() instead of ggsave()
# 3. Don't have hardcoded output paths
# 4. Don't have hardcoded hex colors (with some exceptions)
#
# Run: Rscript scripts/validate_r_figures.R
# Exit: 0 if all pass, 1 if violations found
#
# For help fixing violations, see:
#   - .claude/CLAUDE.md (Anti-Hardcoding section)
#   - docs/planning/ISSUE-test-documentation-improvements.md
# ==============================================================================

# Check for stringr with helpful message
if (!requireNamespace("stringr", quietly = TRUE)) {
  stop(paste0(
    "ERROR: Package 'stringr' is required but not installed.\n",
    "\n",
    "FIX: Install it in R with:\n",
    "  install.packages('stringr')\n",
    "\n",
    "Or run the setup script:\n",
    "  ./scripts/setup-dev-environment.sh"
  ))
}
library(stringr)

# Find project root
find_project_root <- function() {
  markers <- c("pyproject.toml", "CLAUDE.md", ".git")
  dir <- getwd()
  while (dir != dirname(dir)) {
    if (any(file.exists(file.path(dir, markers)))) return(dir)
    dir <- dirname(dir)
  }
  stop("Could not find project root")
}

PROJECT_ROOT <- find_project_root()
FIGURES_DIR <- file.path(PROJECT_ROOT, "src/r/figures")

# ==============================================================================
# Validation Functions with Helpful Error Messages
# ==============================================================================

check_sources_config_loader <- function(content, filename) {
  if (!grepl("source.*config_loader\\.R", content)) {
    return(paste0(
      filename, ": Missing 'source(...config_loader.R)'\n",
      "\n",
      "    WHY: config_loader.R provides:\n",
      "      - load_color_definitions() for YAML color loading\n",
      "      - validate_data_source() for data provenance tracking\n",
      "      - resolve_color() for semantic color resolution\n",
      "\n",
      "    FIX: Add this near the top of your script (after finding PROJECT_ROOT):\n",
      "      source(file.path(PROJECT_ROOT, 'src/r/figure_system/config_loader.R'))\n",
      "\n",
      "    See: .claude/CLAUDE.md (MANDATORY Pattern for R)"
    ))
  }
  NULL
}

check_sources_save_figure <- function(content, filename) {
  if (!grepl("source.*save_figure\\.R", content)) {
    return(paste0(
      filename, ": Missing 'source(...save_figure.R)'\n",
      "\n",
      "    WHY: save_figure.R provides save_publication_figure() which:\n",
      "      - Routes figures to correct directories (main/supplementary/etc)\n",
      "      - Handles PNG/PDF/TIFF output formats from YAML config\n",
      "      - Applies consistent DPI and background settings\n",
      "\n",
      "    FIX: Add this near the top of your script:\n",
      "      source(file.path(PROJECT_ROOT, 'src/r/figure_system/save_figure.R'))\n",
      "\n",
      "    See: .claude/CLAUDE.md (MANDATORY Pattern for R)"
    ))
  }
  NULL
}

check_uses_save_publication_figure <- function(content, filename) {
  if (!grepl("save_publication_figure\\s*\\(", content)) {
    return(paste0(
      filename, ": Does not use save_publication_figure()\n",
      "\n",
      "    WHY: All figures must use the figure system for:\n",
      "      - Consistent output directory routing\n",
      "      - Reproducible dimensions and formats\n",
      "      - Automatic category-based organization\n",
      "\n",
      "    FIX: Replace direct file saving with:\n",
      "      save_publication_figure(plot, 'figure_name', width = 10, height = 6)\n",
      "\n",
      "    The figure name should match an entry in:\n",
      "      configs/VISUALIZATION/figure_layouts.yaml"
    ))
  }
  NULL
}

check_no_direct_ggsave <- function(content, filename) {
  # Remove comment lines (but not inline comments for safety)
  lines <- strsplit(content, "\n")[[1]]
  code_lines <- lines[!grepl("^\\s*#", lines)]
  code <- paste(code_lines, collapse = "\n")

  if (grepl("ggsave\\s*\\(", code)) {
    return(paste0(
      filename, ": Uses ggsave() directly\n",
      "\n",
      "    WHY: ggsave() bypasses the figure system, causing:\n",
      "      - Figures saved to wrong directories\n",
      "      - Inconsistent dimensions/formats across figures\n",
      "      - Missing from figure category organization\n",
      "\n",
      "    FIX: Replace ggsave() with save_publication_figure():\n",
      "\n",
      "      # BEFORE (wrong):\n",
      "      ggsave('figures/generated/my_plot.png', p, width=10, height=6)\n",
      "\n",
      "      # AFTER (correct):\n",
      "      save_publication_figure(p, 'my_plot', width = 10, height = 6)"
    ))
  }
  NULL
}

check_no_hardcoded_output_paths <- function(content, filename) {
  if (grepl('output_dir\\s*<-\\s*file\\.path.*figures/generated', content)) {
    return(paste0(
      filename, ": Has hardcoded output path\n",
      "\n",
      "    WHY: Hardcoded paths break when:\n",
      "      - Figure categories are reorganized\n",
      "      - Project structure changes\n",
      "      - Running on different machines\n",
      "\n",
      "    FIX: Remove output_dir variable and use save_publication_figure():\n",
      "      The figure system routes to the correct directory automatically\n",
      "      based on configs/VISUALIZATION/figure_layouts.yaml"
    ))
  }
  NULL
}

check_no_hardcoded_colors <- function(content, filename) {
  # Remove comment lines
  lines <- strsplit(content, "\n")[[1]]
  code_lines <- lines[!grepl("^\\s*#", lines)]
  code <- paste(code_lines, collapse = "\n")

  # Find hex colors in assignments
  # Pattern: = "#RRGGBB" or = '#RRGGBB'
  matches <- str_extract_all(code, '[=:]\\s*["\']#[0-9A-Fa-f]{6}["\']')[[1]]

  if (length(matches) > 0) {
    # Allow if in palette/color definitions context
    if (!grepl("PALETTE|color_defs|factor_colors|scale_.*_manual", code)) {
      return(paste0(
        filename, ": Has hardcoded hex colors\n",
        "\n",
        "    WHY: Hardcoded colors break when:\n",
        "      - Color palette is updated project-wide\n",
        "      - Accessibility requirements change\n",
        "      - Journal requires different color scheme\n",
        "\n",
        "    FIX: Use resolve_color() with YAML-defined references:\n",
        "\n",
        "      # BEFORE (wrong):\n",
        "      color = '#006BA2'\n",
        "\n",
        "      # AFTER (correct):\n",
        "      color_defs <- load_color_definitions()\n",
        "      color <- resolve_color('--color-primary', color_defs)\n",
        "\n",
        "    Or use ECONOMIST_PALETTE from color_palettes.R for categorical data.\n",
        "\n",
        "    Color definitions are in: configs/VISUALIZATION/figure_colors.yaml"
      ))
    }
  }
  NULL
}

# ==============================================================================
# Main Validation
# ==============================================================================

validate_figure_scripts <- function() {
  violations <- c()

  # Get all figure scripts
  scripts <- list.files(FIGURES_DIR, pattern = "^fig.*\\.R$", full.names = TRUE)

  if (length(scripts) == 0) {
    message("No figure scripts found in ", FIGURES_DIR)
    return(0)
  }

  message("Validating ", length(scripts), " figure scripts...")

  for (script in scripts) {
    filename <- basename(script)

    # Read file with error handling
    tryCatch({
      content <- readLines(script, warn = FALSE, encoding = "UTF-8")
      content <- paste(content, collapse = "\n")
    }, error = function(e) {
      violations <<- c(violations, paste0(filename, ": Could not read file - ", e$message))
      return()
    })

    # Run all checks
    checks <- list(
      check_sources_config_loader,
      check_sources_save_figure,
      check_uses_save_publication_figure,
      check_no_direct_ggsave,
      check_no_hardcoded_output_paths,
      check_no_hardcoded_colors
    )

    for (check_fn in checks) {
      result <- tryCatch({
        check_fn(content, filename)
      }, error = function(e) {
        paste0(filename, ": Check failed with error - ", e$message)
      })

      if (!is.null(result)) {
        violations <- c(violations, result)
      }
    }
  }

  if (length(violations) > 0) {
    message("\n", paste(rep("=", 70), collapse = ""))
    message("VALIDATION FAILED - ", length(violations), " violations found:")
    message(paste(rep("=", 70), collapse = ""))
    message("")

    for (v in violations) {
      message("  - ", v)
      message("")
    }

    message(paste(rep("-", 70), collapse = ""))
    message("HELP: See the following resources for fixing these issues:")
    message("  - .claude/CLAUDE.md (Anti-Hardcoding section)")
    message("  - docs/planning/ISSUE-test-documentation-improvements.md")
    message(paste(rep("=", 70), collapse = ""))

    return(1)
  }

  message("All figure scripts pass validation!")
  return(0)
}

# Run and exit with appropriate code
exit_code <- validate_figure_scripts()
quit(save = "no", status = exit_code)
