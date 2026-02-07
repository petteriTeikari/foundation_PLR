# FAILURE-002: Figure Aesthetic Inconsistency

**Date**: 2026-01-28
**Severity**: HIGH
**Category**: Visual QA / Configuration Management

## What Happened

Multiple figure scripts developed incrementally without centralized style enforcement, resulting in:

1. **Panel label inconsistency**: Some figures use uppercase "A, B, C" (correct), others use lowercase "a, b, c, d"
2. **Legend names hardcoded**: `fig_roc_rc_combined.R` uses inline `sprintf("%s (%.3f)", cfg$name, metric)` instead of YAML display names
3. **Metric confusion**: ROC/RC legend shows single metric value without specifying AUROC vs AURC
4. **Font inconsistency**: Some scripts hardcode "Neue Haas Grotesk Display Pro", others use "Helvetica"
5. **Color duplication**: Same hex codes appear in 8+ files instead of using centralized palette

## Root Cause Analysis

### Primary Cause: No Automated Visual Verification

Pre-commit tests validated:
- ✅ Code syntax (ruff)
- ✅ Code formatting (ruff-format)
- ✅ Registry integrity
- ❌ **NOT** rendered figure consistency
- ❌ **NOT** legend name provenance
- ❌ **NOT** panel label style

### Secondary Cause: YAML Config Exists But Not Enforced

Configuration files exist but R scripts ignore them:

| Config | Location | Enforcement |
|--------|----------|-------------|
| `tag_levels` | `figure_layouts.yaml` | ❌ Scripts hardcode own values |
| `panel_titles` | `figure_layouts.yaml` | ❌ Not used by any script |
| Pipeline names | `combos.yaml` | ❌ Scripts hardcode strings |
| Colors | `colors.yaml` | ⚠️ Partial - some scripts use, others don't |

### Tertiary Cause: Incremental Drift Without Review

Each figure script was created at different times:
- Some copied from templates with correct style
- Some written from scratch with ad-hoc choices
- No automated comparison to detect drift

## Specific Violations Found

### Files with Hardcoded Pipeline Names (3 files)

```r
# fig_raincloud_auroc.R, fig_multi_metric_raincloud.R, fig_cd_preprocessing.R
pipeline_type = case_when(
  grepl("pupil-gt", ...) ~ "Ground Truth",  # HARDCODED
  grepl("^ensemble", ...) ~ "Ensemble",      # HARDCODED
  ...
)
```

### Files with Hardcoded Colors (8 files)

- `fig_raincloud_auroc.R` - 4 colors
- `fig_fm_dashboard.R` - 5 colors
- `fig_prob_dist_by_outcome.R` - 2 colors
- `fig_heatmap_preprocessing.R` - 3 gradient colors
- `fig_featurization_comparison.R` - 2 colors
- `fig_calibration_stratos.R` - gray reference lines
- `fig_dca_stratos.R` - annotation colors
- `fig_stratos_core.R` - annotation colors

### Files with Wrong tag_levels Default (2 files)

```r
# WRONG (lowercase)
create_multi_metric_raincloud <- function(data, tag_levels = "a")
create_stratos_core_panel <- function(..., tag_levels = "a")

# CORRECT (uppercase)
tag_levels = "A"
```

## Prevention Measures

### Immediate (This Session)

1. Create `configs/VISUALIZATION/figure_style.yaml` as SINGLE SOURCE
2. Create `src/r/figure_system/load_style.R` with enforcement functions
3. Refactor all R scripts to use style loader

### Short-Term (This Week)

4. Add metadata sidecar output to `save_figure.R`
5. Create `tests/test_figure_qa/test_style_consistency.py`
6. Add pre-commit hook for figure metadata validation

### Long-Term (Ongoing)

7. Visual regression testing against golden images
8. CI pipeline runs figure generation and compares
9. Template enforcement for new figure scripts

## Guardrail Addition to CLAUDE.md

```markdown
## FIGURE DEVELOPMENT RULES

### MANDATORY: Use Centralized Style

Every R figure script MUST:

1. Source the style loader:
   ```r
   source(file.path(PROJECT_ROOT, "src/r/figure_system/load_style.R"))
   style <- load_figure_style()
   ```

2. Use style functions for display names:
   ```r
   name <- get_pipeline_name("ground_truth", style)  # Returns "Ground Truth"
   metric <- get_metric_label("auroc", style)        # Returns "AUROC"
   ```

3. Use style for panel labels:
   ```r
   plot_annotation(
     tag_levels = style$panel_labels$style,  # "A" not "a"
     theme = get_panel_label_theme(style)
   )
   ```

### BANNED: Hardcoded Strings in Figure Scripts

| Pattern | Status |
|---------|--------|
| `"Ground Truth"`, `"Ensemble"`, `"Traditional"` | ❌ BANNED |
| `tag_levels = "a"` or `tag_levels = "A"` | ❌ BANNED |
| Hex colors like `"#006BA2"` | ❌ BANNED |
| Font names like `"Neue Haas Grotesk"` | ❌ BANNED |

Use YAML config and style loader functions instead.
```

## Verification Checklist

Before committing any figure changes:

- [ ] All display names from `get_pipeline_name()` or `get_metric_label()`
- [ ] All colors from `style$colors` or centralized palette
- [ ] Panel labels use `style$panel_labels$style` (never hardcode "A" or "a")
- [ ] Fonts from `style$fonts` or theme functions
- [ ] `save_publication_figure()` emits metadata sidecar
- [ ] Pre-commit figure QA tests pass

## Fixes Applied (2026-01-28)

### Immediate Fixes

| Issue | Fix | File |
|-------|-----|------|
| calibration+DCA missing traditional | Added duckdb fallback | `scripts/export_predictions_for_r.py` |
| calibration+DCA unequal widths | Added `widths = c(1, 1)` | `src/r/figures/generate_all_r_figures.R` |
| ROC/RC showing 10 models | Added STANDARD_COMBO_IDS filter | `src/r/figures/fig_roc_rc_combined.R` |
| Density flattened by Traditional | Added `scales = "free_y"` | `src/r/figures/fig_prob_dist_by_outcome.R` |
| ROC/RC legend confusion | Separate legends per panel | `src/r/figures/fig_roc_rc_combined.R` |
| fig_prob_dist_by_outcome standalone | Removed save call | `src/r/figures/fig_prob_dist_by_outcome.R` |
| Raincloud 4th panel metric | Changed to O:E ratio | `src/r/figures/fig_multi_metric_raincloud.R` |

### Infrastructure Still Needed

| Task | Status |
|------|--------|
| `figure_style.yaml` single source | TO DO |
| `load_style.R` centralized loader | TO DO |
| Refactor hardcoded values in 8+ files | TO DO |
| Visual consistency pre-commit tests | TO DO |

## Related Documents

- `docs/planning/figure-production-grade.md` - Full implementation plan
- `configs/VISUALIZATION/figure_style.yaml` - Style config (TO BE CREATED)
- `tests/test_figure_qa/test_style_consistency.py` - Style tests (TO BE CREATED)
