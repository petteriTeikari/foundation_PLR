# CRITICAL-FAILURE-004: R Figure Hardcoding Epidemic

**Date Discovered**: 2026-01-29
**Severity**: CRITICAL - Recurring, systematic violation
**Impact**: Reproducibility destroyed, style inconsistency, maintenance nightmare

---

## The Pattern

Every time Claude generates an R figure script, it **hardcodes colors, dimensions, and method names** despite:
- Explicit CLAUDE.md rules prohibiting this
- Existing YAML config systems
- Existing color palette infrastructure
- Multiple user corrections

This is not a one-time failure - it's a **systematic pattern** that has occurred in:
- `fig_preprocessing_quality.R` (2026-01-29)
- `fig_instability_combined.R` (2026-01-29)
- `fig_subject_traces.R` (previous session)
- `fig_roc_rc_combined.R` (previous session)
- Many others...

---

## Root Cause Analysis

### Why It Keeps Happening

| Factor | Description | Impact |
|--------|-------------|--------|
| **No R Pre-Commit Hooks** | Python has ruff linting, R has NOTHING | Zero friction to commit violations |
| **No Source-Code Tests** | test_figure_qa/ checks outputs, not source | Hardcoding passes CI |
| **Optional Config System** | `config_loader.R` can be skipped | Developers take path of least resistance |
| **Pattern Copying** | Claude copies patterns from existing (bad) code | Violations propagate |
| **Context Window Loss** | After compaction, Claude "forgets" rules | Re-learns bad patterns |

### The Enforcement Gap

```
PYTHON PIPELINE (STRONG):
  Code written → ruff pre-commit → FAILS if violations → CANNOT commit

R PIPELINE (WEAK):
  Code written → NO CHECKS → commits silently → violations accumulate
```

---

## The Specific Violations

### Violation Type 1: Hardcoded Hex Colors

**What Claude Writes:**
```r
geom_line(color = "#006BA2", linewidth = 0.8)
scale_fill_manual(values = c("A" = "#FF0000", "B" = "#0000FF"))
```

**What It Should Write:**
```r
# At script start:
color_defs <- load_color_definitions()

# In plot:
geom_line(color = color_defs[["--color-primary"]], linewidth = 0.8)
scale_fill_manual(values = c(
  "A" = color_defs[["--color-fm-primary"]],
  "B" = color_defs[["--color-traditional"]]
))
```

### Violation Type 2: Using ggsave() Instead of save_publication_figure()

**What Claude Writes:**
```r
ggsave("figures/generated/fig_name.png", p, width = 10, height = 8, dpi = 150)
```

**What It Should Write:**
```r
save_publication_figure(p, "fig_name")  # Loads dimensions, DPI from YAML
```

### Violation Type 3: Hardcoded Dimensions

**What Claude Writes:**
```r
save_publication_figure(p, "fig_name", width = 12, height = 10)
```

**What It Should Write:**
```r
fig_config <- load_figure_config("fig_name")
save_publication_figure(p, "fig_name",
  width = fig_config$dimensions$width,
  height = fig_config$dimensions$height
)
```

### Violation Type 4: Own Theme Definition

**What Claude Writes:**
```r
economist_theme <- function() {
  theme_minimal(base_size = 11) +
    theme(
      plot.background = element_rect(fill = "#D7E5EC", color = NA),
      # ... 20 more hardcoded values
    )
}
```

**What It Should Write:**
```r
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))
# Then use:
p + theme_foundation_plr()
```

---

## Enforcement Solutions Implemented

### Solution 1: Pre-Commit Hook for R Files

**File**: `.pre-commit-config.yaml`

```yaml
- repo: local
  hooks:
    - id: r-hardcoding-check
      name: Check R files for hardcoded values
      entry: python scripts/check_r_hardcoding.py
      language: python
      types: [r]
      pass_filenames: true
```

### Solution 2: R Hardcoding Checker Script

**File**: `scripts/check_r_hardcoding.py`

Scans R files for:
- `color = "#[0-9A-Fa-f]{6}"` patterns
- `ggsave(` calls (should use `save_publication_figure`)
- Hardcoded dimensions in figure save calls
- Custom theme definitions (should use `theme_foundation_plr`)

### Solution 3: Test for R Source Code

**File**: `tests/test_r_figures/test_hardcoding.py`

```python
def test_no_hardcoded_hex_colors_in_r_figures():
    """Every R figure script must load colors from YAML, not hardcode."""

def test_no_ggsave_in_r_figures():
    """R figures must use save_publication_figure(), not ggsave()."""

def test_no_custom_themes_in_r_figures():
    """R figures must use theme_foundation_plr(), not custom themes."""
```

### Solution 4: Mandatory R Script Header

All R figure scripts MUST start with:

```r
# ==============================================================================
# MANDATORY HEADER - DO NOT REMOVE
# ==============================================================================
PROJECT_ROOT <- (function() {
  d <- getwd()
  while (d != dirname(d)) {
    if (file.exists(file.path(d, "CLAUDE.md"))) return(d)
    d <- dirname(d)
  }
  stop("Could not find project root")
})()

# Load figure system (MANDATORY)
source(file.path(PROJECT_ROOT, "src/r/figure_system/config_loader.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/save_figure.R"))
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))

# Load colors from YAML (MANDATORY - no hardcoded colors!)
color_defs <- load_color_definitions()

# ==============================================================================
```

---

## CLAUDE.md Updates Required

Add to `.claude/CLAUDE.md`:

```markdown
## 🚨🚨🚨 CRITICAL: R FIGURE HARDCODING IS FORBIDDEN 🚨🚨🚨

**This rule is AUTOMATICALLY ENFORCED via pre-commit hook.**

Every R figure script MUST:

1. **Load colors from YAML** - NEVER write `color = "#RRGGBB"`
   ```r
   # WRONG (will FAIL pre-commit):
   geom_point(color = "#006BA2")

   # CORRECT:
   color_defs <- load_color_definitions()
   geom_point(color = color_defs[["--color-primary"]])
   ```

2. **Use save_publication_figure()** - NEVER use `ggsave()`
   ```r
   # WRONG (will FAIL pre-commit):
   ggsave("output.png", p, width = 10, dpi = 150)

   # CORRECT:
   save_publication_figure(p, "fig_name")
   ```

3. **Use theme_foundation_plr()** - NEVER define custom themes
   ```r
   # WRONG:
   economist_theme <- function() { ... }

   # CORRECT:
   source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))
   p + theme_foundation_plr()
   ```

4. **Load dimensions from config** - NEVER hardcode width/height

**If you see yourself writing ANY hex color (#RRGGBB), STOP. You are violating this rule.**
```

---

## Prevention Checklist

Before committing ANY R figure script:

- [ ] Does the script load `color_defs <- load_color_definitions()`?
- [ ] Are ALL colors accessed via `color_defs[["--color-xxx"]]`?
- [ ] Does the script use `save_publication_figure()`, NOT `ggsave()`?
- [ ] Does the script use `theme_foundation_plr()`, NOT custom theme?
- [ ] Are dimensions loaded from config, NOT hardcoded?
- [ ] Does `python scripts/check_r_hardcoding.py <file.R>` pass?

---

## Why This Matters

1. **Reproducibility**: Hardcoded values differ between scripts, figures look inconsistent
2. **Maintenance**: Changing brand colors requires editing 50 files instead of 1 YAML
3. **Accessibility**: Centralized colors can be checked for colorblind safety; scattered hex cannot
4. **Trust**: If basic style rules aren't followed, what else is wrong?

---

## Related Failures

- `CRITICAL-FAILURE-001-synthetic-data-in-figures.md` - Also a figure generation failure
- `CRITICAL-FAILURE-002-hardcoding-despite-existing-systems.md` - Same pattern, different context
- `FAILURE-003-repeated-instruction-amnesia.md` - Root cause of pattern copying

---

## Lessons Learned

1. **Documentation alone doesn't work** - CLAUDE.md rules were clear, still violated
2. **Enforcement must be automated** - Pre-commit hooks, not manual review
3. **Tests must check source, not just output** - Output can be correct from wrong source
4. **Patterns propagate** - One bad script becomes template for future bad scripts
5. **Context loss is real** - After compaction, rules must be re-enforced

---

## Action Items

- [x] Create this meta-learning document
- [x] Create `scripts/check_r_hardcoding.py`
- [x] Add R hardcoding check to pre-commit (`.pre-commit-config.yaml`)
- [x] Create `tests/test_r_figures/test_hardcoding.py`
- [ ] Update CLAUDE.md with enforcement section
- [ ] Fix existing R scripts to comply (partially done: fig_preprocessing_quality.R, fig_instability_combined.R)
- [x] Add mandatory header template to `src/r/figures/_TEMPLATE.R`

### Files Still Needing Fixes (as of 2026-01-29)

Found by `pytest tests/test_r_figures/`:
- `fig_subject_traces.R` - uses ggsave()
- `fig_variance_decomposition.R` - defines custom theme_lollipop
- `fig_stratos_core.R` - no color_defs loading
- `fig_cd_preprocessing.R` - no color_defs loading
- `fig_multi_metric_raincloud.R` - no color_defs loading
