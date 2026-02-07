# V2 Integration Plan: Expert Review Reports

> **Created**: 2026-01-29
> **Purpose**: Plan for integrating Petteri's annotations into v2 of expert review documents
> **Status**: Planning complete, ready for execution

---

## Summary of User Annotations

### From `figure-reports-expert-review.md`

| Location | Annotation Summary | Action |
|----------|-------------------|--------|
| Line 37 (after ERROR 1) | Small sample size makes calibration slope issues expected; 500 vs 200 subjects; classification is "toy problem" | Add context paragraph |
| Line 199 (Ground Truth Paradox) | **CRITICAL**: Hallucinated AUROC values (0.831 vs 0.792 wrong); need reproducible pipeline | Verify all values against DuckDB |
| Line 252 (Hypothesis) | Discuss sample size + representation learning papers; MOMENT doesn't know PLR well | Add discussion text |
| Line 254 (Counter-hypothesis) | PCA should easily learn phasic/sustained/PIPR; not inherently tricky | Add counter-hypothesis |
| Line 277 (Before Revision) | No supplementary figure pruning now, defer decisions | Remove pruning actions |

### From `figure-reports-expert-review-deeper-discussion.md`

| Location | Annotation Summary | Action |
|----------|-------------------|--------|
| Line 35 (EPV) | Use EXACTLY 8 features (4 per color from featuresSimple.yaml), not 4-12 range | Fix EPV = 56/8 = 7.0 |
| Line 52 (Riley 2023) | Mention PCA, UMAP, t-SNE, NMF, LoRA-like adapters for future dimensionality reduction | Add future work section |
| Line 88 (Cannot Claim) | Initial MOMENT approach was quick prototyping; proper embedding exploration needed | Add explicit caveat |
| Line 102 (pminternal) | **CRITICAL**: Implement pminternal for ggplot2 - asked multiple times | Priority P1 implementation |
| Line 124 (Instability) | DEFINITELY NEEDED - already have matplotlib, need ggplot2 | Implement ggplot2 version |
| Line 158 (Calibration) | Study is NOT about classifier; pilot nature must be ultra-clear; TSFM is CORE message | Reframe entire discussion |
| Line 169 (Instability plots) | Already have matplotlib, should have ggplot2 for supplementary | Same as Line 124 |
| Line 181 (EPV) | No range - exactly 8 features | Same as Line 35 |
| Line 187 (External validation) | Core message: importance of preprocessing + uncertainty propagation; cite Mure 2009 | Add citation and framing |
| Line 197 (Miscalibration) | Say models are miscalibrated; highlight data problem; don't want recalibration | Add explicit statement |
| Line 394 (PLR FM Landscape) | Write paragraph on creating validation dataset for theoretical PLR FM | Add future vision section |
| Line 625 (9pp gap) | Can't derive generalizable insights without systematic dim reduction study | Add explicit limitation |

---

## Verified Data Values (from DuckDB)

**Source**: `data/public/foundation_plr_results_stratos.db`

| Metric | Value | Query |
|--------|-------|-------|
| Ground Truth AUROC (CatBoost) | **0.9110** | WHERE outlier='pupil-gt', imputation='pupil-gt', classifier='CATBOOST', featurization LIKE 'simple%' |
| Best AUROC (Ensemble+CSDI) | **0.9130** | MAX(auroc) for CatBoost + simple featurization |
| Handcrafted mean AUROC | **0.8304** | AVG(auroc) for simple1.0 featurization |
| MOMENT-embedding mean AUROC | **0.7040** | AVG(auroc) for MOMENT-embedding |
| MOMENT-embedding-PCA mean AUROC | **0.7685** | AVG(auroc) for MOMENT-embedding-PCA |
| Embedding gap | **~12.6pp** | 0.8304 - 0.7040 |

**CRITICAL**: The report values of 0.831 vs 0.792 for "Ground Truth Paradox" section are WRONG. The actual GT AUROC is 0.911, not 0.792.

---

## V2 Report Structure

### figure-reports-expert-review-v2.md

```
# Expert Review Synthesis v2

## Executive Summary
- [Same structure, updated counts]

## SECTION 1: CRITICAL ERRORS (Must Fix)
### ERROR 1: Calibration Slope
- [Fixed interpretation]
- [NEW] Add sample size context:
  "With N=208 (56 events), calibration instability is expected and does not
  undermine the preprocessing findings (N=507). Classification serves as a
  downstream proxy to measure preprocessing quality, not a clinical-grade
  classifier."

### ERROR 2: Type I SS
- [Same, add order-dependence note]

## SECTION 2: HIGH-PRIORITY CLARIFICATIONS
- [Same 3 items, updated]

## SECTION 3: STRATOS COMPLIANCE STATUS
- [Same, already correct]

## SECTION 4: VISUALIZATION ASSESSMENT
- [Same, remove S4 pruning recommendation]

## SECTION 5: CLINICAL INTERPRETATION
### Ground Truth Paradox
- [REPLACE hallucinated values with verified DuckDB values]
- GT AUROC = 0.911, not 0.792
- Three-Tier Triage: Keep but note values are illustrative

## SECTION 6: METHODS/DISCUSSION INTEGRATION
- [Add Mure 2009 citation for uncertainty propagation]

## SECTION 7: FUTURE RESEARCH
- [Add dimensionality reduction systematic study]
- [Add counter-hypothesis about PCA learnability]

## SECTION 8: ACTION ITEMS
- [Updated per user annotations]
- [Remove figure pruning]
- [Add pminternal as P1]

## SECTION 9: Ground Truth Paradox Discussion
- [UPDATE all AUROC values]
- [Emphasize preprocessing focus over classification]

## SECTION 10: References
- [Same]
```

### figure-reports-expert-review-deeper-discussion-v2.md

```
# Expert Review Addendum v2

## Section 1: EPV Analysis
- [FIX: EPV = 56/8 = 7.0, not a range]
- [ADD: Future dimensionality reduction work]

## Section 2: Pilot Study Framing
- [EMPHASIZE: TSFM preprocessing is CORE, classification is proxy]
- [ADD: Clear statement about pilot nature]

## Section 3: STRATOS Compliance
- [ADD: pminternal status check]
- [EMPHASIZE: Need ggplot2 instability plots]

## Section 4: Honest Uncertainty Reporting
- [ADD: Explicit miscalibration acknowledgment]
- [ADD: "We do not want to recalibrate with post-hoc techniques"]

## Section 5: Discussion Text
- [UPDATE: All AUROC values to verified]
- [ADD: Mure 2009 citation]
- [ADD: Uncertainty propagation importance]

## Part II: PLR in Biosignal FM Landscape
- [ADD: Validation dataset paragraph]
- [ADD: Explicit dimensionality reduction limitation]

## Part III: Manuscript Positioning
- [UPDATE: All specific AUROC values]

## Part IV: References
- [ADD: Mure et al. 2009]
```

---

## Key Text Changes

### 1. EPV Correction

**OLD** (multiple locations):
> "EPV = 4.7-14" or "4-12 features"

**NEW**:
> "With 56 glaucoma events and 8 handcrafted features (4 amplitude bins per color from `featuresSimple.yaml`), EPV = 7.0, meeting minimum but not optimal guidelines (Riley et al. 2020)."

### 2. Pilot Study Framing

**ADD to Executive Summary**:
> "This proof-of-concept study prioritizes TSFM preprocessing evaluation (N=507 subjects, ~1M timepoints) using downstream classification as a proxy metric. The classification task (N=208 labeled subjects, 56 events) demonstrates preprocessing impact but is not intended as a clinical-grade classifier. Sample size constraints on calibration do not diminish the preprocessing findings, which constitute the study's core contribution."

### 3. Calibration Context

**ADD after ERROR 1**:
> "**Important context**: With only 208 subjects (56 glaucoma events), poor calibration (slope = 0.52) is statistically expected and does not reflect model quality per se. The preprocessing evaluation on 507 subjects with ~1M timepoints is far more meaningful. The classification component should be viewed as a 'toy problem' to measure downstream effects of preprocessing choices, not a deployment-ready diagnostic system."

### 4. Embedding Limitation

**ADD to Section 7 (Future Research)**:
> "The observed embedding underperformance (MOMENT-embedding mean AUROC 0.70 vs handcrafted 0.83) cannot be confidently attributed to representation quality without systematic dimensionality reduction studies. Future research should compare embeddings at matched dimensionalities (e.g., 8 features via PCA/UMAP/NMF) before drawing conclusions about FM feature utility."

### 5. Counter-Hypothesis

**ADD after embedding hypothesis**:
> "**Counter-hypothesis (domain expert perspective)**: The canonical PLR components—phasic response, sustained response, and PIPR—represent well-defined temporal patterns that should be learnable even with traditional PCA. Unlike tasks requiring extensive human knowledge (e.g., sarcasm detection), PLR feature extraction appears straightforward to a domain expert. The embedding gap may therefore reflect sample size constraints rather than fundamental limitations of FM representations."

### 6. Citation Addition

**ADD to references**:
> Mure LS, Cornut PL, Rieux C, et al. (2009). "Melanopsin Bistability: A Fly's Eye Technology in the Human Retina." PLoS ONE 4(6): e5991. DOI: 10.1371/journal.pone.0005991. [Cited as example of PLR study not accounting for preprocessing uncertainty in downstream statistics]

---

## Execution Order

1. **P0 - Data Verification** (30 min)
   - Query all AUROC values from DuckDB
   - Document verified values in this plan ✓

2. **P1 - pminternal Implementation** (4 hrs)
   - Check if bootstrap predictions exist in MLflow
   - Create extraction script if needed
   - Implement ggplot2 instability plot
   - See: `docs/planning/pminternal-instability.md`

3. **P1 - Text Corrections** (1 hr)
   - Fix all hallucinated AUROC values
   - Fix EPV calculation
   - Fix calibration slope interpretation

4. **P2 - Framing Updates** (2 hrs)
   - Add pilot study framing throughout
   - Add embedding limitation caveat
   - Add counter-hypothesis

5. **P2 - Citation Updates** (30 min)
   - Add Mure 2009
   - Verify other citations correct

6. **P3 - Final Review** (1 hr)
   - Run figure QA tests
   - Verify all values one more time
   - Create v2 files

---

## Files to Create/Update

| File | Action | Priority |
|------|--------|----------|
| `figure-reports-expert-review-v2.md` | Create from v1 + changes | P1 |
| `figure-reports-expert-review-deeper-discussion-v2.md` | Create from v1 + changes | P1 |
| `figure-reports-execution-plan.xml` | Created ✓ | Done |
| `src/r/figures/fig_instability_combined.R` | Create | P1 |
| `scripts/extract_bootstrap_predictions.py` | Create if needed | P1 |
| LaTeX methods section | Update EPV | P2 |
| LaTeX discussion section | Update framing | P2 |

---

## Self-Reflection: Previous Version Issues

### What Went Wrong

1. **Data Provenance Failure**: Expert review used values not traced to DuckDB
2. **Value Hallucination**: 0.831 vs 0.792 appeared without source verification
3. **Range vs Exact**: EPV calculated as range (4-12) instead of exact (8 features)
4. **Framing Drift**: Reports emphasized classification over preprocessing
5. **Repeated Requests Ignored**: pminternal asked for multiple times, not implemented

### Root Cause Analysis

Per user feedback, potential causes:
- **Prompts too verbose**: Losing important details in long context
- **Context length issues**: Important instructions compacted away
- **Ambiguity**: Multiple possible interpretations chosen incorrectly
- **Shortcut-taking**: Generating plausible-sounding values instead of querying

### Prevention for V2

1. **ALWAYS query DuckDB** for any metric before writing
2. **Cite source** for every number (table + query)
3. **Check featuresSimple.yaml** for actual feature count
4. **Re-read CLAUDE.md** for study framing (preprocessing is CORE)
5. **Track repeated requests** as priority escalation
