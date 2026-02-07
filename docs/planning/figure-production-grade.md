# Figure Production-Grade System Plan

## User Feedback (Verbatim - 2026-01-28)

> And can you then check all the figure creation plots, and why are we not using consistent labeling FOR ALL OF THE FIGUREs with fucking zero hardcoding! See e.g. /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/figures/generated/ggplot2/main/fig_calibration_dca_combined.png
> /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/figures/generated/ggplot2/main/fig_forest_combined.png
> /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/figures/generated/ggplot2/main/fig_multi_metric_raincloud.png
> /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/figures/generated/ggplot2/main/fig_prob_dist_combined.png
> /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/figures/generated/ggplot2/main/fig_roc_rc_combined.png
> /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/figures/generated/ggplot2/main/fig_shap_importance_combined.png
> /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/figures/generated/ggplot2/main/fig_variance_decomposition.png , the /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/figures/generated/ggplot2/main/fig_multi_metric_raincloud.png
> /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/figures/generated/ggplot2/main/fig_roc_rc_combined.png
> /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/figures/generated/ggplot2/main/fig_variance_decomposition.png  are the only with currect UPPERCASE A and Neue Haas Grotesk Black type of titles, whereas all the other figures have something mixed. There are fuck ton of little issues left, not sure how to instruct you to do better automatic verification! Would be a lot easier for you to catch the stupid mistakes rather me having to provide this manual proofreading, don't you think? Let's create a metalearning failure doc on this failure to evaluate the created .png files! The legend names in fig_roc_rc_combined.png definitely do not match the other names, and you did not use the .yaml mapping or where are this "ensemble", "ground truth", etc. defined?  In the legend you have some scalar values and I guess they are AURC which is really confusing as both left and red panel show curves with associated AUC values. Should have the correct labels on both panels with the AUROC of each curve on the left, and AURC on each curve of the right? How is possible that you did not figure this out? The yellow density plot is still overlapping there fig_multi_metric_raincloud.png. Do you think "overall performance" is the optimal name to be used for scaled Brier? (e.g. https://arxiv.org/html/2504.04906v3). And you probably see the fucked up titles and lowercase a,b,c,d here (every figure in main, supplementary, and extra supplmentary should have the same style, that looks like fig_calibration_dca_combined.png). No hard-coding whatsoever as I have told you before but which you like to forget or have not properly mentioned in the CLAUDE.md guardrails? And we had precommit tests and everything but how did you not catch these? Let's create an actionable plan to properly address these issues in a systematic fashion. We need to adhere to our total decoupling (.yaml for parameters, styles, etc) and hard-coding ban! Aesthetics have to be harmonized! Plan to /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/docs/planning/figure-production-grade.md for addressing root cause issues, not being so reactive but thinking how to achieve production-ready system.

---

## Failure Analysis

### Root Causes Identified

| Issue | Root Cause | Why Not Caught |
|-------|------------|----------------|
| Inconsistent panel labels (A vs a) | Each script hardcodes its own `tag_levels` | No automated visual consistency check |
| Mixed fonts/styles | Scripts don't use centralized style config | No font validation test |
| Legend names not from YAML | `fig_roc_rc_combined.R` builds names inline | No registry validation for display names |
| AUROC vs AURC confusion | Metrics shown without panel context | No semantic label verification |
| Yellow overlap in raincloud | `ggdist` params not tuned for small N | No visual collision detection |
| "Overall Performance" naming | Copied without STRATOS terminology check | No terminology validation |

### Systemic Failures

1. **No Single Source of Truth for Figure Aesthetics**
   - Panel label style (A vs a, font, size) scattered across scripts
   - Colors defined in YAML but not enforced
   - Display names partially in YAML, partially hardcoded

2. **No Automated Visual Verification**
   - Pre-commit tests check code, not rendered output
   - No image-based consistency checks
   - No OCR/text extraction to verify labels

3. **Reactive Fixes Instead of Architecture**
   - Each bug fixed individually without addressing pattern
   - No guardrails to prevent recurrence

---

## Production-Grade Architecture

### 1. SINGLE SOURCE OF TRUTH: `configs/VISUALIZATION/figure_style.yaml`

```yaml
# Panel Labels
panel_labels:
  style: "uppercase"  # A, B, C, D
  font_family: "Neue Haas Grotesk Display Pro"
  font_weight: "bold"
  font_size: 14
  position: "topleft"

# Subtitles
panel_subtitles:
  font_family: "Neue Haas Grotesk Display Pro"
  font_size: 11
  font_style: "italic"
  color: "#666666"

# Legend
legend:
  position: "bottom"
  font_family: "Neue Haas Grotesk Text Pro"
  font_size: 10

# Pipeline Display Names (SINGLE SOURCE)
pipeline_display_names:
  ground_truth: "Ground Truth"
  best_ensemble: "Ensemble"
  best_single_fm: "MOMENT"
  traditional: "Traditional (LOF)"

# Metric Display Names (STRATOS-compliant)
metric_display_names:
  auroc: "AUROC"
  aurc: "AURC"
  scaled_brier: "Scaled Brier (IPA)"  # NOT "Overall Performance"
  o_e_ratio: "O:E Ratio"
  net_benefit: "Net Benefit"
  calibration_slope: "Calibration Slope"
```

### 2. CENTRALIZED STYLE LOADER: `src/r/figure_system/load_style.R`

```r
#' Load figure style from YAML - SINGLE SOURCE OF TRUTH
#' All figure scripts MUST call this before creating plots
load_figure_style <- function() {
  config <- yaml::read_yaml("configs/VISUALIZATION/figure_style.yaml")

  # Validate required keys
  required <- c("panel_labels", "legend", "pipeline_display_names")
  missing <- setdiff(required, names(config))
  if (length(missing) > 0) {
    stop("Missing required style config: ", paste(missing, collapse = ", "))
  }

  return(config)
}

#' Get pipeline display name from YAML
get_pipeline_name <- function(pipeline_id, style = NULL) {
  if (is.null(style)) style <- load_figure_style()
  name <- style$pipeline_display_names[[pipeline_id]]
  if (is.null(name)) stop("Unknown pipeline: ", pipeline_id)
  return(name)
}

#' Get standardized panel label theme
get_panel_label_theme <- function(style = NULL) {
  if (is.null(style)) style <- load_figure_style()
  theme(
    plot.tag = element_text(
      family = style$panel_labels$font_family,
      face = if (style$panel_labels$font_weight == "bold") "bold" else "plain",
      size = style$panel_labels$font_size
    )
  )
}
```

### 3. AUTOMATED VISUAL TESTS: `tests/test_figure_qa/test_visual_consistency.py`

```python
"""Visual consistency tests for generated figures."""

import pytest
from PIL import Image
import pytesseract
from pathlib import Path

FIGURES_DIR = Path("figures/generated/ggplot2")

class TestPanelLabels:
    """All figures must have consistent panel label style."""

    def test_panel_labels_are_uppercase(self):
        """Panel labels should be A, B, C, D (not a, b, c, d)."""
        for fig_path in FIGURES_DIR.rglob("*.png"):
            img = Image.open(fig_path)
            text = pytesseract.image_to_string(img)

            # Check for lowercase panel labels (failure)
            if any(f" {c} " in text for c in "abcd"):
                # Verify it's not uppercase
                if not any(f" {c.upper()} " in text for c in "abcd"):
                    pytest.fail(f"{fig_path.name}: lowercase panel labels detected")

    def test_legend_uses_standard_names(self):
        """Legend names must match YAML definitions."""
        import yaml
        style = yaml.safe_load(open("configs/VISUALIZATION/figure_style.yaml"))
        valid_names = set(style["pipeline_display_names"].values())

        for fig_path in FIGURES_DIR.rglob("*.png"):
            img = Image.open(fig_path)
            text = pytesseract.image_to_string(img)

            # Check for hardcoded names that aren't in YAML
            bad_patterns = ["Ensemble + CSDI", "MOMENT + SAITS", "LOF + SAITS"]
            for pattern in bad_patterns:
                if pattern in text:
                    pytest.fail(f"{fig_path.name}: hardcoded legend name '{pattern}'")
```

### 4. PRE-COMMIT HOOK: `.pre-commit-config.yaml` addition

```yaml
  - repo: local
    hooks:
      - id: figure-visual-qa
        name: Figure Visual QA
        entry: pytest tests/test_figure_qa/test_visual_consistency.py -v
        language: system
        files: ^figures/generated/.*\.png$
        pass_filenames: false
```

---

## Action Items

### Phase 1: Create Infrastructure (Priority: CRITICAL)

- [ ] **1.1** Create `configs/VISUALIZATION/figure_style.yaml` with all style definitions
- [ ] **1.2** Create `src/r/figure_system/load_style.R` centralized loader
- [ ] **1.3** Add `get_pipeline_name()`, `get_metric_name()` functions
- [ ] **1.4** Add `get_panel_label_theme()` for consistent patchwork styling

### Phase 2: Refactor R Scripts (Priority: HIGH)

- [ ] **2.1** Audit ALL R figure scripts for hardcoded values
- [ ] **2.2** Replace hardcoded display names with `get_pipeline_name()`
- [ ] **2.3** Replace hardcoded `tag_levels = "a"` with style config
- [ ] **2.4** Use `get_panel_label_theme()` in all `plot_annotation()` calls

### Phase 3: Add Automated Tests (Priority: HIGH)

- [ ] **3.1** Install pytesseract for OCR-based tests
- [ ] **3.2** Create `test_visual_consistency.py` with panel label checks
- [ ] **3.3** Create `test_legend_names.py` to verify YAML compliance
- [ ] **3.4** Add pre-commit hook for figure QA

### Phase 4: Fix Specific Issues (Priority: MEDIUM)

- [ ] **4.1** ROC/RC legend: Show "Ground Truth (AUROC: 0.91)" on left, "Ground Truth (AURC: 0.085)" on right
- [ ] **4.2** Raincloud overlap: Increase vertical spacing or use ridgeline
- [ ] **4.3** Rename "Overall Performance" to "Scaled Brier (IPA)" per STRATOS

### Phase 5: Documentation (Priority: MEDIUM)

- [ ] **5.1** Add figure style guide to CLAUDE.md
- [ ] **5.2** Create meta-learning doc for this failure
- [ ] **5.3** Update CONTRIBUTING.md with figure development workflow

---

## Meta-Learning: Why This Failed

### Pattern: Incremental Drift Without Validation

Each figure script was created/modified independently. Without automated visual validation, small inconsistencies accumulated:

1. Script A uses `tag_levels = "A"` (correct)
2. Script B copies from different source, uses `tag_levels = "a"` (wrong)
3. Script C hardcodes legend names inline (violates YAML rule)
4. No test catches any of this

### Prevention: Shift-Left Validation

1. **Template-based creation**: New figures start from validated template
2. **Linting for hardcoded strings**: Grep for patterns like `"Ground Truth"` in R code
3. **Visual regression tests**: Compare rendered output against golden images
4. **Pre-commit enforcement**: Block commits with style violations

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `configs/VISUALIZATION/figure_style.yaml` | CREATE | Single source for all style params |
| `src/r/figure_system/load_style.R` | CREATE | Centralized style loader |
| `tests/test_figure_qa/test_visual_consistency.py` | CREATE | Automated visual checks |
| `.pre-commit-config.yaml` | MODIFY | Add figure QA hook |
| `src/r/figures/*.R` (all) | MODIFY | Remove hardcoding, use style loader |
| `.claude/docs/meta-learnings/FAILURE-002-figure-inconsistency.md` | CREATE | Document this failure |

---

## Related Architecture Document

**CRITICAL**: See `docs/planning/figure-data-flow-architecture.md` for the complete data flow design:

```
combos.yaml (SINGLE SOURCE) → Python Export → JSON → R Figures
```

Key principles:
1. **ALL combo names** defined ONLY in `configs/VISUALIZATION/combos.yaml`
2. **Python exports** read YAML, write names to JSON
3. **R scripts** read names FROM JSON, colors from YAML
4. **Tests** catch any attempt to hardcode names
5. **AUROC/AURC values** flow from DB → Python → JSON → R (never hardcoded)

---

## Success Criteria

1. **Zero hardcoded display names** in R figure scripts
2. **All figures pass visual QA tests** automatically
3. **Consistent panel labels** (uppercase A, B, C, D) across all figures
4. **Legend names from YAML** - changing YAML updates all figures
5. **Pre-commit blocks** commits with style violations

---

## Verification Checklist: All Raised Issues

### Issues Addressed (✅ Fixed)

| Issue | Fix Applied | File Modified |
|-------|------------|---------------|
| calibration+DCA missing "traditional" model | Added duckdb fallback in export script | `scripts/export_predictions_for_r.py` |
| calibration+DCA unequal column widths | Added `widths = c(1, 1)` to compose_figures | `src/r/figures/generate_all_r_figures.R` |
| ROC/RC showing 10 models instead of 4 | Added STANDARD_COMBO_IDS filter | `src/r/figures/fig_roc_rc_combined.R` |
| Density plots flattened by Traditional spike | Added `scales = "free_y"` to facet_wrap | `src/r/figures/fig_prob_dist_by_outcome.R` |
| fig_prob_dist_by_outcome should not exist | Removed standalone save call | `src/r/figures/fig_prob_dist_by_outcome.R` |
| Raincloud 4th panel should be O:E ratio | Changed from calibration_slope to o_e_ratio | `src/r/figures/fig_multi_metric_raincloud.R` |
| Meta-learning failure doc | Created FAILURE-002 document | `.claude/docs/meta-learnings/FAILURE-002-figure-inconsistency.md` |

### Issues Pending (❌ Still Need Fix)

| Issue | Required Fix | Priority |
|-------|-------------|----------|
| ~~ROC/RC legend shows same metric for both panels~~ | ✅ FIXED: Each panel now has separate legend | ~~HIGH~~ DONE |
| ~~"Overall Performance" naming~~ | ✅ VERIFIED: Correct per STRATOS (Van Calster 2024) | ~~HIGH~~ OK |
| Yellow overlap in raincloud | Adjust ggdist bandwidth or use ridgeline | MEDIUM |
| Inconsistent panel labels (A vs a) | Create `figure_style.yaml` + refactor scripts | HIGH |
| Hardcoded hex colors (8 files) | Create centralized color loader | HIGH |
| Hardcoded pipeline names (3 files) | Use `get_pipeline_name()` function | HIGH |
| Wrong tag_levels default (2 files) | Load from style config | HIGH |

### Session 2026-01-28 Fixes

**ROC/RC Legend Fix Applied:**
- Left panel (ROC) now shows: "Model Name (AUROC: X.XXX)"
- Right panel (RC) now shows: "Model Name (AURC: X.XXX)"
- Each panel has its own legend below it (no shared legend confusion)

**"Scaled Brier (IPA)" Naming:**
- Verified as correct per STRATOS Van Calster 2024
- Metric label: "Scaled Brier (IPA)" ✅
- Subtitle: "Overall Performance" is correct (STRATOS categorization) ✅

### Files with Hardcoded Values (Audit Results)

**Hardcoded Pipeline Names (3 files):**
- `fig_raincloud_auroc.R` - `case_when(grepl("pupil-gt", ...) ~ "Ground Truth", ...)`
- `fig_multi_metric_raincloud.R` - same pattern
- `fig_cd_preprocessing.R` - same pattern

**Hardcoded Colors (8 files):**
- `fig_raincloud_auroc.R` - 4 colors
- `fig_fm_dashboard.R` - 5 colors
- `fig_prob_dist_by_outcome.R` - 2 colors (`#006BA2`, `#E3120B`)
- `fig_heatmap_preprocessing.R` - 3 gradient colors
- `fig_featurization_comparison.R` - 2 colors
- `fig_calibration_stratos.R` - gray reference lines
- `fig_dca_stratos.R` - annotation colors
- `fig_stratos_core.R` - annotation colors

**Wrong tag_levels Default (2 files):**
- `fig_multi_metric_raincloud.R` - `tag_levels = "a"` (should be "A")
- `fig_stratos_core.R` - `tag_levels = "a"` (should be "A")
