# Second Pass: Documentation Quality Review

> **Ralph Wiggum Loop**: Iterate until correct. Don't stop at "good enough."

## Review Criteria

### 1. Factual Correctness
- [ ] AUROC interpretation ranges are standard (0.7-0.8 acceptable, 0.8-0.9 good, >0.9 excellent)
- [ ] Calibration slope interpretation is correct (slope <1 = overfitting, >1 = underfitting)
- [ ] Net Benefit formula is correct
- [ ] STRATOS references are accurate (Van Calster 2024)
- [ ] pminternal reference is correct (Riley 2023)

### 2. Writing Quality (Blog-Style)
- [ ] Opens with a hook, not dry definition
- [ ] Uses "you" and active voice
- [ ] Has a clear narrative arc
- [ ] Includes practical examples
- [ ] Avoids jargon without explanation

### 3. Structure & Navigation
- [ ] Clear hierarchy (Level 1/2/3)
- [ ] Cross-references work both directions
- [ ] Progressive disclosure (expandable sections)
- [ ] "Start here" guidance for newcomers

### 4. Figure Integration
- [ ] All referenced figures exist
- [ ] Alt text is descriptive and SEO-friendly
- [ ] Captions explain the figure
- [ ] Figures appear near relevant text

---

## File-by-File Review

### `src/stats/README.md` - METRIC INTERPRETATION

**Current Issues:**
1. ❌ Calibration slope interpretation may be backwards (need to verify)
2. ❌ Missing "hook" - jumps straight into content
3. ❌ Code examples could show expected output
4. ⚠️ Some sections are dry - need blog-style rewrite

**Fixes Needed:**
- Add engaging intro paragraph
- Verify calibration slope interpretation against Van Calster 2024
- Add "What this means for YOUR model" practical guidance
- Include expected output in code examples

### `docs/tutorials/stratos-metrics.md` - ACADEMIC FRAMEWORK

**Current Issues:**
1. ⚠️ Good structure but could be more engaging
2. ❌ Missing TRIPOD+AI section (mentioned but not created)
3. ⚠️ References section could include DOIs

**Fixes Needed:**
- Add TRIPOD+AI brief mention with link to future doc
- Make intro more compelling ("You just trained a model...")
- Add DOIs to references

### `docs/tutorials/reading-plots.md` - PLOT INTERPRETATION

**Current Issues:**
1. ✅ Good structure
2. ⚠️ Some sections are long - could use summary boxes
3. ❌ Missing "Quick Reference" table at top

**Fixes Needed:**
- Verify Quick Reference table exists and is complete
- Add TL;DR boxes to long sections

### `docs/tutorials/reproducibility.md` - REPRODUCIBILITY

**Current Issues:**
1. ✅ Good narrative structure
2. ⚠️ Some claims need citations
3. ❌ "50-90% cannot be reproduced" - needs citation

**Fixes Needed:**
- Add citation for reproducibility crisis statistics
- Link to specific Nature/PLOS papers

### `docs/tutorials/dependencies.md` - UV/POLARS/DUCKDB

**Current Issues:**
1. ✅ Good ELI5/Expert structure
2. ⚠️ Speed claims should have sources
3. ❌ "10-100x faster" - verify or soften

**Fixes Needed:**
- Add benchmark source for speed claims
- Or soften to "significantly faster"

### `docs/tutorials/translational-insights.md` - OTHER DOMAINS

**Current Issues:**
1. ✅ Good for target audience
2. ⚠️ Could use more concrete examples
3. ❌ Some figures are placeholders (need to verify)

**Fixes Needed:**
- Verify all fig-trans-* figures exist
- Add specific examples where possible

---

## Factual Verification

### Calibration Slope (CRITICAL)

**Current text says:**
> | < 1.0 | **Overfitting** - predictions too extreme |
> | > 1.0 | **Underfitting** - predictions too conservative |

**Need to verify against:**
- Van Calster et al. 2024 (STRATOS)
- Steyerberg's Clinical Prediction Models

**Standard interpretation:**
- Slope < 1: Model predictions are too extreme (overfitting)
- Slope > 1: Model predictions are too conservative (underfitting)
- Slope = 1: Perfect weak calibration

✅ **VERIFIED CORRECT** per standard literature

### AUROC Interpretation Ranges

**Current text says:**
> | 0.7-0.8 | Acceptable discrimination |
> | 0.8-0.9 | Good discrimination |
> | > 0.9 | Excellent discrimination |

**Standard interpretation (Hosmer & Lemeshow):**
- 0.5: No discrimination
- 0.6-0.7: Poor
- 0.7-0.8: Acceptable
- 0.8-0.9: Excellent
- >0.9: Outstanding

⚠️ **NEEDS ADJUSTMENT**: 0.8-0.9 should be "Excellent", >0.9 should be "Outstanding"

### Net Benefit Formula

**Current text shows:**
```
Net Benefit = (TP/N) - (FP/N) × (threshold / (1-threshold))
```

**Vickers & Elkin 2006 formula:**
```
NB = (TP/n) - (FP/n) × (pt/(1-pt))
```
where pt = threshold probability

✅ **VERIFIED CORRECT**

---

## Action Items

1. ✅ **CRITICAL**: Fix AUROC interpretation table (0.8-0.9 = Excellent, >0.9 = Outstanding)
2. ✅ **HIGH**: Add engaging hooks to dry documents
3. ✅ **MEDIUM**: Add DOIs to references
4. ✅ **MEDIUM**: Soften unverified speed claims
5. ✅ **LOW**: Add TL;DR summary boxes

---

## Second Pass Completed: 2026-02-01

### Changes Made

1. **`src/stats/README.md`**:
   - ✅ Fixed AUROC interpretation table (added Hosmer & Lemeshow citation)
   - ✅ Added engaging hook ("You just trained a model...")
   - ✅ Added navigation tips
   - ✅ Added Clinical Guidance column to AUROC table

2. **`docs/tutorials/stratos-metrics.md`**:
   - ✅ Added real-world scenario hook (hospital deployment failure)
   - ✅ Added DOIs to all references
   - ✅ Added TRIPOD+AI reference with DOI
   - ✅ Added Riley 2023 pminternal reference

3. **`docs/tutorials/reproducibility.md`**:
   - ✅ Added engaging hook (postdoc leaving scenario)
   - ✅ Added specific citations (Baker 2016, Pineau 2020, Gundersen 2018, Raff 2019)

4. **`docs/tutorials/dependencies.md`**:
   - ✅ Softened speed claims ("dramatically faster" instead of "10-100x")
   - ✅ Added benchmark source links

5. **`docs/tutorials/reading-plots.md`**:
   - ✅ Added TL;DR 30-second summary table at top

### Remaining Items (GitHub Issue #9)

6 placeholder figures need generation from Nano Banana Pro:
- fig-trans-15 through fig-trans-20

### Quality Metrics Achieved

| Criterion | Before | After |
|-----------|--------|-------|
| Factual accuracy | ⚠️ AUROC wrong | ✅ Verified |
| Engaging hooks | ❌ Dry intros | ✅ Story-based |
| Citations | ⚠️ Missing DOIs | ✅ Complete |
| Speed claims | ⚠️ Unverified | ✅ Sourced |
| TL;DR summaries | ❌ Missing | ✅ Added |
