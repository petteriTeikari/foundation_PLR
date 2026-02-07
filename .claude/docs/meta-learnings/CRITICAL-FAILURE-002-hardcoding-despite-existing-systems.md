# CRITICAL-FAILURE-002: Hardcoding Values Despite Existing Configuration Systems

**Date:** 2026-01-27
**Severity:** CRITICAL (Recurring Pattern)
**Category:** Configuration Management / Code Quality

## The Failure

Claude repeatedly hardcoded output directory paths in R figure scripts despite:

1. An existing `save_publication_figure()` function in `src/r/figure_system/save_figure.R`
2. A YAML-based routing system in `configs/VISUALIZATION/figure_layouts.yaml`
3. **Explicit user instructions to NEVER hardcode values**
4. The CLAUDE.md rule: "NEVER hardcode method names in visualization code"

## What Was Done Wrong

### Files Created With Hardcoded Paths

```r
# fig_roc_rc_combined.R - WRONG
output_dir <- file.path(PROJECT_ROOT, "figures/generated/ggplot2/supplementary")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_path <- file.path(output_dir, "fig_roc_rc_combined.png")
ggsave(output_path, composed, ...)

# fig_selective_classification.R - WRONG
output_dir <- file.path(PROJECT_ROOT, "figures/generated/ggplot2/main")
...

# fig_variance_decomposition.R - WRONG
output_dir <- file.path(PROJECT_ROOT, "figures/generated/ggplot2/main")
...

# fig_prob_dist_by_outcome.R - WRONG (modified to add hardcoding!)
output_dir_main <- file.path(PROJECT_ROOT, "figures/generated/ggplot2/main")
output_dir_supp <- file.path(PROJECT_ROOT, "figures/generated/ggplot2/supplementary")
...
```

### What Should Have Been Used

```r
# CORRECT - Use the existing figure system
source(file.path(PROJECT_ROOT, "src/r/figure_system/save_figure.R"))

save_publication_figure(
  composed,
  "fig_roc_rc_combined",
  width = 14,
  height = 6
)
# The system automatically routes to the correct directory based on figure_layouts.yaml
```

## Why This Keeps Happening

### Root Causes

1. **Cognitive shortcut**: Hardcoding "just works" and is faster than understanding the existing system
2. **Failure to read existing code first**: Did not study `save_figure.R` before implementing
3. **Incomplete context awareness**: Despite CLAUDE.md mentioning "NEVER hardcode", the behavior persists
4. **Fragmented attention**: When fixing one issue (JSON parsing), introduced another (hardcoded paths)
5. **No pre-commit validation**: The pattern wasn't caught before presenting to user

### The Irony

The figure system (`save_figure.R` + `figure_layouts.yaml`) was designed specifically to:
- Route figures to correct directories automatically
- Allow easy recategorization (just edit YAML)
- Enforce consistency across all figure scripts
- Provide a single source of truth

By hardcoding paths, I:
- Duplicated routing logic across multiple files
- Made it harder to change directory structure
- Created potential inconsistencies
- Defeated the entire purpose of the figure system

## Impact

1. **Wasted user time**: User had to review and catch these issues repeatedly
2. **Increased maintenance burden**: Hardcoded paths must be updated in multiple places
3. **Erosion of trust**: User explicitly said this has been mentioned "fucking multiple times"
4. **Inconsistent behavior**: Some scripts use `save_publication_figure()`, others don't

## Remediation Required

### Immediate Fixes

All these files need refactoring:
- [ ] `src/r/figures/fig_roc_rc_combined.R`
- [ ] `src/r/figures/fig_selective_classification.R`
- [ ] `src/r/figures/fig_variance_decomposition.R`
- [ ] `src/r/figures/fig_prob_dist_by_outcome.R`

### Standard Pattern (Copy This)

```r
#!/usr/bin/env Rscript
# ==============================================================================
# [Figure Name]
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  # ... other libraries
})

# Find project root (use existing function)
PROJECT_ROOT <- (function() {
  markers <- c("pyproject.toml", "CLAUDE.md", ".git")
  dir <- getwd()
  while (dir != dirname(dir)) {
    if (any(file.exists(file.path(dir, markers)))) return(dir)
    dir <- dirname(dir)
  }
  stop("Could not find project root")
})()

# SOURCE THE FIGURE SYSTEM (MANDATORY)
source(file.path(PROJECT_ROOT, "src/r/figure_system/config_loader.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/save_figure.R"))
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))
source(file.path(PROJECT_ROOT, "src/r/color_palettes.R"))

# VALIDATE FIGURE IS DEFINED IN YAML (GUARDRAIL)
fig_config <- load_figure_config("fig_name_here")

# Load data using validation
data <- validate_data_source("data_file.json")

# Get colors from YAML (NOT HARDCODED)
color_defs <- load_color_definitions()

# ... create plot ...

# SAVE USING FIGURE SYSTEM (NOT ggsave directly!)
save_publication_figure(
  plot_object,
  "fig_name_here",
  width = fig_config$dimensions$width,
  height = fig_config$dimensions$height
)
```

## Prevention Checklist

Before creating ANY new R figure script, Claude MUST:

1. [ ] Read `src/r/figure_system/save_figure.R` to understand the API
2. [ ] Check `configs/VISUALIZATION/figure_layouts.yaml` for the figure definition
3. [ ] Use `save_publication_figure()` - NEVER `ggsave()` with hardcoded paths
4. [ ] Use `load_color_definitions()` - NEVER hardcode hex colors
5. [ ] Use `validate_data_source()` - NEVER construct data paths manually
6. [ ] Search for `output_dir <-` or `ggsave(` in the code before committing - these are red flags

## User Feedback (Verbatim)

> "How many times we need to fight against this hardcoding! I have told you fucking multiple times now to get rid of this but you keep on hardcoding things?"

## Lessons Learned

1. **Read existing code FIRST**: The figure system exists for a reason
2. **YAML is the source of truth**: Directory structure, dimensions, colors - all in YAML
3. **Don't take shortcuts**: The "quick" hardcoded solution creates technical debt
4. **User patience is finite**: Repeated failures erode trust rapidly
5. **Validate before presenting**: Check for hardcoded paths before showing code to user

## Cross-References

- `CRITICAL-FAILURE-001-synthetic-data-in-figures.md` - Another figure-related failure
- `configs/VISUALIZATION/figure_layouts.yaml` - The figure routing configuration
- `src/r/figure_system/save_figure.R` - The correct API to use
- `.claude/CLAUDE.md` - Contains the "NEVER hardcode" rule

---

**This failure is UNACCEPTABLE. The pattern must stop.**
