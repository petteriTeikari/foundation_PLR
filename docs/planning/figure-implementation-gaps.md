# Figure Implementation Gaps - TDD Checklist

**Created**: 2026-01-27
**Purpose**: Track remaining implementation work for figure grouping plan

---

## Implementation Status Overview

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| CD diagrams 1×3 layout | ❌ NOT DONE | P1 | Line 331 still uses `/` (vertical) |
| fig_preprocessing_auroc | ⚠️ EXISTS AS | P0 | `fig_raincloud_auroc.R` - rename/verify |
| fig_error_propagation | ❌ NOT DONE | P1 | New figure needed |
| Calibration metrics annotation | ✅ DONE | - | Lines 132-157 in stratos_core |
| results.tex embedding demotion | ⚠️ PARTIAL | P1 | EPV mentioned, needs full update |
| TDD tests | ⚠️ PARTIAL | P2 | Existing tests, may need updates |

---

## P0: Critical Missing Items

### 1. Verify/Rename fig_preprocessing_auroc

**Current state**: `fig_raincloud_auroc.R` exists and does exactly what's needed:
> "Shows distribution of AUROC across preprocessing configurations"

**Action needed**:
- [ ] Verify `fig_raincloud_auroc.R` generates correct output
- [ ] Either rename to `fig_preprocessing_auroc.R` OR update references
- [ ] Ensure output PNG is in `main/` directory
- [ ] Add TDD test for this figure

**Test to add**:
```r
test_that("fig_preprocessing_auroc shows AUROC by preprocessing method", {
  # Load data
  data <- load_preprocessing_auroc_data()
  # Create figure

  p <- create_preprocessing_auroc_raincloud(data)
  # Verify it's a ggplot with raincloud components
  expect_s3_class(p, "gg")
  expect_true(any(sapply(p$layers, function(l) inherits(l$geom, "GeomBoxplot"))))
})
```

---

## P1: Required Code Changes

### 2. CD Diagrams Layout Change (3×1 → 1×3)

**File**: `src/r/figures/fig_cd_diagrams.R`
**Line**: 331

**Current code**:
```r
combined <- p_outlier / p_imputation / p_combined
```

**Required change**:
```r
combined <- p_outlier + p_imputation + p_combined
```

**TDD Test**:
```r
test_that("CD diagrams use horizontal 1x3 layout", {
  p <- create_cd_combined(test_data)
  # Check patchwork uses + not /
  # Horizontal layout should have ncol=3, nrow=1
  expect_equal(p$patches$layout$ncol, 3)
  expect_equal(p$patches$layout$nrow, 1)
})
```

**Also update dimensions**:
```r
# Current: save_publication_figure(p_combined, "fig_cd_diagrams", width = 12, height = 16)
# Change to:
save_publication_figure(p_combined, "fig_cd_diagrams", width = 18, height = 6)
```

### 3. Create fig_error_propagation (NEW)

**Purpose**: Waterfall showing error cascade through pipeline stages

**Design**:
```
Stage 1: Outlier Detection
├─ Ground truth: 0% error baseline
├─ LOF: 15% false positive rate
└─ MOMENT: 8% false positive rate
    ↓
Stage 2: Imputation (degradation from outlier errors)
├─ With GT outliers: RMSE 0.05
├─ With LOF outliers: RMSE 0.12
└─ With MOMENT outliers: RMSE 0.08
    ↓
Stage 3: Classification (cumulative effect)
├─ GT preprocessing: AUROC 0.911
├─ LOF+imputation: AUROC 0.823
└─ MOMENT+imputation: AUROC 0.865
```

**Implementation approach**:
- Use `ggplot2` with `geom_segment` or `ggalluvial` for flow
- Alternative: `waterfall` chart showing degradation at each stage

**TDD Test**:
```r
test_that("fig_error_propagation shows 3-stage cascade", {
  p <- create_error_propagation_figure(test_data)
  expect_s3_class(p, "gg")
  # Should have 3 stages visible
  expect_true(length(unique(p$data$stage)) >= 3)
})
```

### 4. results.tex Updates

**File**: `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/results.tex`

**Required changes**:

a) **Replace embedding section** (around line 75-86):
```latex
% OLD: Detailed featurization comparison
% NEW:
\subsubsection{Validation of Handcrafted Features}
Our exploratory analysis of foundation model embeddings
(detailed in Supplementary Appendix B) confirmed the necessity
of domain-specific feature engineering. Given the high dimensionality
of learned representations relative to sample size (EPV $<$ 1),
this comparison serves as hypothesis-generating rather than a
primary statistical finding. All subsequent pipeline evaluations
utilize the handcrafted feature set.
```

b) **Update figure reference** (if fig_featurization demoted):
```latex
% OLD: Figure~\ref{fig:featurization} shows the 9pp gap
% NEW:
The supplementary featurization comparison (Supplementary Figure S2)
illustrates this exploratory finding.
```

### 5. discussion.tex Updates

**File**: `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/discussion.tex`

**Required change** - preface 9pp gap with exploratory framing:
```latex
% Add before the featurization gap discussion:
As shown in our exploratory analysis (Supplementary Appendix B),
foundation model embeddings underperformed handcrafted features
by approximately 9 percentage points. While this finding requires
validation in larger cohorts to meet EPV requirements, it suggests...
```

---

## P2: TDD Test Updates

### Current Test File
`tests/r/test_stratos_figures.R` - 43 tests

### Tests to Add/Update

```r
# === NEW TESTS FOR FIGURE GROUPING ===

context("Figure Grouping Compliance")

test_that("Main figures are in main/ directory", {
  main_dir <- "figures/generated/ggplot2/main"
  expected_main <- c(
    "fig_stratos_core.png",
    "fig_cd_preprocessing.png",
    "fig_fm_dashboard.png",
    "fig_heatmap_preprocessing.png",
    "fig_specification_curve.png"
  )
  for (fig in expected_main) {
    expect_true(file.exists(file.path(main_dir, fig)),
                info = paste("Missing main figure:", fig))
  }
})

test_that("Supplementary figures are in supplementary/ directory", {
  supp_dir <- "figures/generated/ggplot2/supplementary"
  expected_supp <- c(
    "fig_cd_diagrams.png",
    "fig_featurization_comparison.png",
    "fig_factorial_matrix.png"
  )
  for (fig in expected_supp) {
    expect_true(file.exists(file.path(supp_dir, fig)),
                info = paste("Missing supplementary figure:", fig))
  }
})

test_that("CD diagrams use horizontal layout (width > height)", {
  # After regeneration, check dimensions
  img <- png::readPNG("figures/generated/ggplot2/supplementary/fig_cd_diagrams.png")
  expect_gt(dim(img)[2], dim(img)[1])  # width > height
})
```

---

## Implementation Order (TDD Approach)

### Phase 1: Write Tests First
1. [ ] Add test for CD diagrams horizontal layout
2. [ ] Add test for fig_preprocessing_auroc existence
3. [ ] Add test for main/supplementary directory structure
4. [ ] Run tests - expect FAILURES

### Phase 2: Implement Changes
5. [ ] Fix CD diagrams layout (line 331: `/` → `+`)
6. [ ] Verify/rename fig_raincloud_auroc → fig_preprocessing_auroc
7. [ ] Regenerate all figures
8. [ ] Run tests - expect PASS

### Phase 3: Documentation
9. [ ] Update results.tex with embedding demotion text
10. [ ] Update discussion.tex with exploratory framing
11. [ ] Create Supplementary Appendix B content

### Phase 4: New Figure (Optional P1)
12. [ ] Design fig_error_propagation
13. [ ] Write TDD test for error propagation
14. [ ] Implement figure
15. [ ] Add to main/ directory

---

## Verification Commands

```bash
# Run R tests
cd /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR
Rscript -e "testthat::test_dir('tests/r')"

# Regenerate all figures
Rscript src/r/figures/generate_all_r_figures.R

# Verify directory structure
ls -la figures/generated/ggplot2/main/
ls -la figures/generated/ggplot2/supplementary/
ls -la figures/generated/ggplot2/extra-supplementary/
```

---

## Summary

| Phase | Items | Est. Effort |
|-------|-------|-------------|
| Tests First | 4 new tests | Small |
| Code Changes | CD layout, verify raincloud | Small |
| Documentation | results.tex, discussion.tex | Medium |
| New Figure | error_propagation (optional) | Medium |

**Minimum viable**: Phases 1-2 (tests + CD layout fix)
**Full compliance**: All 4 phases
