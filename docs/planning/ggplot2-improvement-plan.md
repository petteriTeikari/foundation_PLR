# ggplot2 Figure Improvement Plan

**Overall Assessment: 6.0/10.0** - Some good elements but significant gaps in understanding the science and having reproducible pipelines.

---

## User Feedback (Verbatim - 2026-01-27)

> 1) This fig_shap_importance_multi.png is all white with no content whatsoever! How is it possible that you let through a figure like this and think that this is okay? Create a meta-learning doc about this!
>
> 2) This fig_prob_dist_by_outcome.png is totally redundant and should be removed as this is already in fig_prob_dist_combined.png.
>
> 3) In this figure, half of the width is taken by the legend fig_selective_classification.png! What the fuck serious Claude Opus 4.5? Another test needed, and another meta-learning failure!
>
> 4) This makes zero sense to me as there are no legends in panel A, no data on Panels B and Panel C? fig_stratos_core.png.
>
> 5) This is totally abysmal plot with the bar plot, I have asked you to modify this multiple times, but somehow you manage to ignore my instructions always fig_variance_decomposition.png, wasn't there a a discussion to replace this with a better factorial design visualization with the stats? use lollipops? Add a second panel to the right including the effect of classifier choice and demonstrate how it dominates the variance, but it is not so relevant in terms of the manuscript narrative as "everyone knows that Catboost is better than Logistic Regression.
>
> 6) Combine these into a 2-col figure with the calibration on the left, and DCA on left: fig_calibration_stratos.png, fig_dca_stratos.png. Add my prompt verbatim, and let's plan how to address these failures with reviewer agents! And where is the CD diagram, or the interaction analysis from the main figures as it was really important to be in one large plot?

### Additional Feedback (CD Diagram)

> And definitely now fig_cd_diagrams.png does not look like the stylized CD diagram that people are used! It should have the rank clearly on a horizontal bar and the names of the models coming out with the vertical+horizontal bar, and then to show the CD.

**Reference: Standard Demšar (2006) CD Diagram Layout:**

```
           |-------- CD --------|
     4         3         2         1     <- Rank axis (horizontal)
     |_________|_________|_________|
                   |=========|           <- Clique bar (methods not sig. different)
     |             |         |
  Method4      Method3   Method2      Method1
     |_____________|         |_________|
           (connected = not significantly different)
```

**Current fig_cd_diagrams.png**: Uses scattered points with labels - NOT standard style.

### Proposed Solution: CD Diagram Class/Wrapper

**Existing R Packages for Standard CD Diagrams:**

| Package | Function | Output | Notes |
|---------|----------|--------|-------|
| [scmamp](https://rdrr.io/cran/scmamp/man/plotCD.html) | `plotCD()` | Base R plot | Standard Demšar style |
| [mlr](https://www.rdocumentation.org/packages/mlr/versions/2.19.1/topics/plotCritDifferences) | `plotCritDifferences()` | ggplot2 | Returns ggplot object |
| [performanceEstimation](https://www.rdocumentation.org/packages/performanceEstimation/versions/1.1.0/topics/CDdiagram.Nemenyi) | `CDdiagram.Nemenyi()` | Base R plot | Demšar-style |

**Recommended Approach:**

1. **Use `scmamp::plotCD()`** as the core implementation (proven, widely cited)
2. **Create wrapper class** `CDDiagram` in `src/r/figure_system/cd_diagram.R`:
   - Accepts our metrics data format
   - Applies Foundation PLR styling (colors, fonts)
   - Handles method name abbreviations
   - Exports to PNG/PDF with correct dimensions

```r
# Proposed CDDiagram class usage:
cd <- CDDiagram$new(
  results_matrix = auroc_by_method_and_fold,
  alpha = 0.05,
  method_labels = PIPELINE_LABELS_SHORT,
  title = "Outlier Detection Methods"
)
cd$plot()  # Renders standard Demšar-style diagram
cd$save("fig_cd_outlier.png", width = 8, height = 6)
```

**Why wrapper instead of direct use:**
- Consistent styling with other figures
- Method name abbreviation handling
- Integration with `save_publication_figure()`
- Testable interface

---

## Critical Failures Identified

### P0 - CRITICAL (Blank/Broken Figures)

| Issue | Figure | Problem | Impact |
|-------|--------|---------|--------|
| **BLANK-001** | `fig_shap_importance_multi.png` | Completely white, no content | Figure unusable |
| **NODATA-001** | `fig_stratos_core.png` | Panels B & C have no data, Panel A no legend | Figure broken |
| **WRONG-STYLE-001** | `fig_cd_diagrams.png` | Not standard Demšar CD diagram style | Violates community convention |

### P1 - HIGH (Layout/Design Failures)

| Issue | Figure | Problem | Impact |
|-------|--------|---------|--------|
| **LEGEND-001** | `fig_selective_classification.png` | Legend takes 50% of width | Poor space utilization |
| **DESIGN-001** | `fig_variance_decomposition.png` | Bar plot instead of lollipops, single panel | Ignores user instructions |

### P2 - MEDIUM (Redundancy/Missing)

| Issue | Figure | Problem | Impact |
|-------|--------|---------|--------|
| **REDUNDANT-001** | `fig_prob_dist_by_outcome.png` | Redundant with `fig_prob_dist_combined.png` | Unnecessary figure |
| **MISSING-001** | CD diagram | Not in main figures | Missing key statistical comparison |
| **MISSING-002** | Interaction analysis | Not in main figures | Missing key finding |
| **SPLIT-001** | `fig_calibration_stratos.png` + `fig_dca_stratos.png` | Should be combined 1x2 | Inefficient layout |

---

## Required Test Improvements

### New Tests Needed

1. **test_figure_not_blank.R**
   - Check that PNG file size > minimum threshold (e.g., > 10KB)
   - Check that image has actual pixel variation (not all white/black)
   - Parse image and verify content regions exist

2. **test_legend_proportion.R**
   - Check that legend does not exceed X% of figure width
   - Verify legend placement is appropriate

3. **test_panel_data_presence.R**
   - For multi-panel figures, verify each panel has data
   - Check for empty geom layers

4. **test_figure_redundancy.R**
   - Flag figures that are subsets of other figures
   - Prevent standalone versions of combined figures in same category

---

## Action Items

### Immediate Fixes Required

1. **fig_shap_importance_multi.png** - Debug why it's blank
   - Check data loading in `fig_shap_importance.R`
   - Verify SHAP values exist for multi-config comparison
   - Add error handling for missing data

2. **fig_stratos_core.png** - Fix missing data in panels
   - Debug `create_calibration_panel()` - why no calibration curve?
   - Debug `create_dca_panel()` - why no DCA data?
   - Add legend to ROC panel A

3. **fig_selective_classification.png** - Fix legend proportion
   - Move legend to bottom or inside plot
   - Reduce legend size
   - Consider removing redundant legend entries

4. **fig_variance_decomposition.png** - Complete redesign
   - Replace bar plot with lollipop chart
   - Add second panel showing classifier effect
   - Show preprocessing dominates but classifier choice is known

5. **Combine calibration + DCA** into `fig_calibration_dca_combined.png`
   - 1x2 layout: Calibration (left) + DCA (right)
   - Remove standalone versions from main

6. **Add to main figures:**
   - CD diagram (move from supplementary or create dedicated main version)
   - Interaction analysis figure

7. **Remove redundant:**
   - `fig_prob_dist_by_outcome.png` (keep only `fig_prob_dist_combined.png`)

---

## Meta-Learning Documents Created

- `.claude/docs/meta-learnings/CRITICAL-FAILURE-002-blank-figure-generation.md`
- `.claude/docs/meta-learnings/CRITICAL-FAILURE-003-legend-layout-issues.md`

---

## Expert Review Required

Spawn reviewer agents for:
1. **Figure QA Specialist** - Validate all figures have content
2. **Layout Expert** - Review space utilization and legend placement
3. **STRATOS Compliance** - Verify all required metrics are visualized
4. **Manuscript Narrative** - Ensure figures tell the research story

---

## Timeline

| Phase | Task | Status |
|-------|------|--------|
| 1 | Create meta-learning docs | DONE |
| 2 | Implement new tests | PENDING |
| 3 | Fix critical failures (P0) | PENDING |
| 4 | Fix high priority (P1) | PENDING |
| 5 | Address medium priority (P2) | PENDING |
| 6 | Expert review | DONE (see below) |
| 7 | Final validation | PENDING |

---

## Expert Review Synthesis (3 Reviewers)

### Figure QA Specialist

**Key Recommendations:**
1. Add bytes-per-pixel ratio test (catches blank 5KB figures instantly)
2. Multi-panel content validation - verify ALL panels have data
3. Fix `conftest.py` to use recursive glob (`**/*.png`)

**Priority Tests:**
```r
# Test 1: Blank detection via compression ratio
bytes_per_pixel <- file_size / pixel_count
expect_gt(bytes_per_pixel, 0.005)  # Blank PNGs have ~0.001

# Test 2: Panel data presence
for (panel in panels) {
  white_ratio <- mean(panel > 0.98)
  expect_lt(white_ratio, 0.95)
}
```

### STRATOS TG6 Expert

**STRATOS Compliance Status: INCOMPLETE**

| Requirement | Status |
|-------------|--------|
| AUROC with 95% CI | Partial |
| Calibration plot (smoothed) | **BROKEN - NO DATA** |
| Calibration slope annotation | Missing |
| O:E ratio | Missing |
| DCA curves | **BROKEN - NO DATA** |
| Probability distributions | Done (redundant) |

**Recommended Main Figure Set (6 figures):**
1. Factorial Experimental Design
2. STRATOS Core (2x2) OR Calibration+DCA (1x2)
3. CD Diagram (promote from supplementary)
4. Variance Decomposition (2-panel lollipop)
5. Featurization Comparison
6. FM Dashboard

### Layout Design Expert

**Legend Guidelines:**
- 2-3 categories: Direct labeling on lines
- 4-5 categories: Bottom legend, horizontal
- 6-8 categories: Bottom legend, 2 rows
- 9-10 categories: Dedicated legend panel

**Pipeline Name Abbreviations:**
| Full Name | Abbreviation |
|-----------|--------------|
| `pupil-gt + pupil-gt` | GT |
| `ensemble-LOF-MOMENT-...` | Ens |
| `MOMENT-gt-finetune + SAITS` | FM |
| `LOF + SAITS` | Trad |

**Layout Constants to Enforce:**
- `MAX_LEGEND_WIDTH_RATIO`: 0.25
- `MAX_LEGEND_CHARS`: 25
- `MIN_PANEL_DATA_RATIO`: 0.5

---

## Comprehensive Action Items

### Phase 1: Critical Fixes (P0)

| # | Task | File | Priority |
|---|------|------|----------|
| 1.1 | Debug blank `fig_shap_importance_multi.png` | `fig_shap_importance.R` | CRITICAL |
| 1.2 | Fix `fig_stratos_core.png` panels B & C | `fig_stratos_core.R` | CRITICAL |
| 1.3 | Replace CD diagram with `scmamp::plotCD()` wrapper | Create `cd_diagram.R` | CRITICAL |

### Phase 2: Layout Fixes (P1)

| # | Task | File | Priority |
|---|------|------|----------|
| 2.1 | Fix legend proportion in `fig_selective_classification.png` | `fig_selective_classification.R` | HIGH |
| 2.2 | Redesign `fig_variance_decomposition.png` as 2-panel lollipop | `fig_variance_decomposition.R` | HIGH |
| 2.3 | Combine calibration + DCA into 1x2 | Create `fig_calibration_dca_combined.R` | HIGH |

### Phase 3: Organization (P2)

| # | Task | File | Priority |
|---|------|------|----------|
| 3.1 | Delete redundant `fig_prob_dist_by_outcome.png` | Config + script | MEDIUM |
| 3.2 | Promote CD diagram to main figures | `figure_layouts.yaml` | MEDIUM |
| 3.3 | Add interaction analysis figure | Create new script | MEDIUM |

### Phase 4: Test Infrastructure

| # | Task | File | Priority |
|---|------|------|----------|
| 4.1 | Add blank figure detection test | `test_blank_figures.R` | HIGH |
| 4.2 | Add panel content validation test | `test_multipanel_content.R` | HIGH |
| 4.3 | Add legend proportion test | `test_legend_proportion.R` | MEDIUM |
| 4.4 | Add redundancy detection test | `test_figure_redundancy.R` | LOW |

---

## Success Criteria

Before declaring figures complete:

- [ ] All TDD tests pass (including new tests)
- [ ] No blank figures (bytes/pixel > 0.005)
- [ ] All multi-panel figures have data in ALL panels
- [ ] Legend width < 25% for side placement
- [ ] CD diagram uses standard Demšar style
- [ ] Variance decomposition uses lollipop chart with 2 panels
- [ ] Calibration + DCA combined in 1x2 layout
- [ ] No redundant figures in same category
- [ ] All main figures are STRATOS-compliant
