# Expert Review Synthesis: Figure Reports (v2)

> **Generated**: 2026-01-29 (v2 with Petteri annotations integrated)
> **Previous version**: 2026-01-28
> **Reviewed by**: 6 specialized expert agents + author annotations
> **Documents reviewed**: main-figure-report.md, supplementary-figure-report.md, figure-bibliography.md, manuscript .tex files

---

## Executive Summary

The figure reports are **substantially compliant** with STRATOS guidelines and represent high-quality scientific visualization. However, the expert review identified:

- **2 CRITICAL ERRORS** requiring immediate correction
- **3 HIGH-priority clarifications** needed
- **6 MEDIUM-priority improvements** recommended
- **10 future experiment ideas** for research extension

**CRITICAL SCOPE REMINDER** (from author annotations):
> This study is NOT about classifiers. TSFM preprocessing is the CORE message. Classification (N=208, 56 events) is a downstream proxy to measure preprocessing quality (N=507, ~1M timepoints). Calibration instability at small sample sizes is EXPECTED and does not diminish preprocessing findings.

---

## SECTION 1: CRITICAL ERRORS (Must Fix)

### ERROR 1: Calibration Slope Interpretation is BACKWARDS

**Location**: Main Figure Report, Figure 1, lines 45-46

**Current text (WRONG)**:
> "Calibration slope <1 indicates underconfidence in predictions (observed events exceed predicted probabilities at high-risk end)."

**Correct interpretation**:
| Slope | Meaning | Interpretation |
|-------|---------|----------------|
| **slope < 1.0** | **Overfitting** | Predictions too extreme (overconfident) |
| **slope > 1.0** | **Underfitting** | Predictions too conservative (underconfident) |

**Why this matters**: The report states slope = 0.52, which indicates **overfitting/overconfidence**, not underconfidence. This is a fundamental calibration concept error.

**AUTHOR CONTEXT (CRITICAL)**:
> With such extreme small classifier sample sizes (N=208, 56 events), poor calibration is not surprising and should be expected. The preprocessing evaluation (N=507, ~1M timepoints) is far more meaningful than the classification (N=208), which serves as a downstream proxy/"toy problem" to measure preprocessing impact. This pilot study should NOT be treated as a classifier development study.

**Fix required**:
```
WRONG: "slope 0.52 indicates underconfidence"
CORRECT: "slope 0.52 indicates overfitting (predictions more extreme than warranted).
         With only 56 events for classifier development, poor calibration is
         statistically expected and does not diminish the preprocessing findings."
```

---

### ERROR 2: Type I Sequential SS is Order-Dependent (Not Acknowledged)

**Location**: Main Figure Report, Figure 6, line 323 and LaTeX caption

**Issue**: Type I sequential sums of squares attributes "shared variance" to whichever factor appears FIRST in the model. The variance decomposition percentages would change if you reordered factors.

**Current model order**: `auroc ~ classifier + outlier_method + imputation_method`

**Problem**: If you used `auroc ~ imputation_method + outlier_method + classifier`, you would get DIFFERENT η² values.

**Fix required**: Either:
1. Switch to **Type III SS** (order-independent): `car::Anova(model, type = "III")`
2. Explicitly justify the factor ordering in the caption
3. Add sensitivity analysis with different orderings

**Recommended caption addition**:
> "Type I sequential sums of squares with factor order: classifier → outlier → imputation; variance attribution is order-dependent."

---

## SECTION 2: HIGH-PRIORITY CLARIFICATIONS

### CLARIFICATION 1: CD Diagram "Replicates" Misidentified

**Location**: Supplementary Figure Report, Figure S3, LaTeX caption line 169

**Current text**: "cross-validation folds as replicates"

**Actual implementation**: The R code uses **imputation methods as pseudo-folds** (Panel A) and **outlier methods as pseudo-folds** (Panel B). This is NOT standard cross-validation replication.

**Why it matters**: The Friedman test assumes independent replicates. Using methods as pseudo-folds is a creative but non-standard approach that may violate test assumptions.

**Fix required**:
```
WRONG: "cross-validation folds as replicates"
CORRECT: "imputation methods (Panel A) / outlier methods (Panel B) as pseudo-replicates;
         see Methods for justification of this non-standard Friedman test application."
```

---

### CLARIFICATION 2: N=208 vs N=507 Sample Sizes Inconsistent

**Issue**: Figures reference N=208 for classification, but preprocessing evaluation should note N=507.

**Current confusion**:
- Forest plots (Figure 2) correctly states both
- Other figures only mention N=208

**Fix required**: Add consistent clarification to all figure captions:
> "Preprocessing evaluated on N=507 subjects (~1M timepoints); AUROC computed on N=208 labeled subjects (152 control, 56 glaucoma)."

---

### CLARIFICATION 3: EPV Concern Inadequately Flagged

**Critical statistics issue**: With N=56 glaucoma events:
- Handcrafted features (8 features from featuresSimple.yaml): **EPV = 7.0** (borderline)
- MOMENT embeddings (96 dimensions): **EPV = 0.58** (UNACCEPTABLE)

**AUTHOR CORRECTION**: The EPV for handcrafted features is EXACTLY 7.0 (56 events / 8 features), not a range of 4.7-14. We use 4 amplitude bins per color = 8 features total from `featuresSimple.yaml`.

**Fix required**: Add explicit EPV statement to methods:
> "With 56 glaucoma events and 8 handcrafted features (4 amplitude bins per color), EPV = 7.0, meeting minimum guidelines. MOMENT embeddings (96 dimensions, EPV = 0.58) should be interpreted as exploratory given severe EPV violation."

---

## SECTION 3: STRATOS COMPLIANCE STATUS

### Checklist (All Required Elements Present)

| STRATOS Core Element | Status | Location |
|---------------------|--------|----------|
| Discrimination (AUROC with CI) | COMPLETE | Fig 2, 3A, S1A |
| Calibration plot (smoothed with CI) | COMPLETE | Fig 1A |
| Calibration slope | COMPLETE | Fig 1A annotation |
| O:E ratio | COMPLETE | Fig 1A, Fig 3D |
| Net Benefit with DCA | COMPLETE | Fig 1B |
| Probability distributions per outcome | COMPLETE | Fig 4, S6 |
| Brier score / Scaled Brier | COMPLETE | Fig 1A, 3B |

### Prohibited Measures Audit: CLEAN

| Measure | Present? | STRATOS Guidance |
|---------|----------|------------------|
| F1 score | NO | NOT recommended (ignores TN) |
| AUPRC | NO | NOT recommended as primary |
| pAUC | NO | NOT recommended |
| Accuracy | NO | NOT recommended |

**Verdict**: **SUBSTANTIALLY COMPLIANT** - All required elements present, no prohibited measures.

---

## SECTION 4: VISUALIZATION ASSESSMENT

### Figures with EXCELLENT Design

| Figure | Why It Works |
|--------|--------------|
| Fig 1 (Calibration + DCA) | Standard STRATOS combined view |
| Fig 2 (Forest plot) | Gold standard for method comparison |
| Fig 3 (Raincloud) | Justified given N=57+; shows distribution shape |
| Fig 6 (Variance decomposition) | Lollipop clean and effective |
| Fig S1 (ROC + RC) | RC adds uncertainty calibration info |
| Fig S3 (CD diagrams) | Gold standard for multi-method comparison |
| Fig S8 (Specification curve) | Gold standard for multiverse analysis |

### Figures Needing Improvement

| Figure | Issue | Recommendation |
|--------|-------|----------------|
| **Fig 4** | Overlapping densities hard to read | Consider mirror/back-to-back density |
| **Fig S7** | Visual clutter with many configs | Consider heatmap alternative |

**AUTHOR NOTE**: No supplementary figure pruning at this point. Defer decisions on S4 redundancy until later.

### Color/Accessibility Recommendations

1. Run all figures through colorblind simulator (Coblis)
2. Add line types (dashed, dotted) in addition to color for 4+ curve figures
3. Ensure red-blue in Fig 4 has sufficient saturation difference

---

## SECTION 5: CLINICAL INTERPRETATION FINDINGS

### SHAP Feature Importance - Physiologically Valid

The dominance of melanopsin pathway features makes biological sense:
- **Blue_SUSTAINED_value**: Captures ipRGC function (first to die in glaucoma)
- **Blue_MAX_CONSTRICTION**: Peak melanopsin-mediated response
- **PIPR**: Post-illumination response (melanopsin hallmark)

**Conclusion**: SHAP rankings align with known glaucoma pathophysiology, increasing clinical trust.

### Ground Truth Comparison - VERIFIED VALUES

**DATA VERIFICATION** (from DuckDB `foundation_plr_results_stratos.db`):
| Configuration | AUROC | Source Query |
|---------------|-------|--------------|
| Ground Truth (pupil-gt + pupil-gt + CatBoost) | **0.9110** | WHERE outlier='pupil-gt', imputation='pupil-gt', classifier='CATBOOST', featurization LIKE 'simple%' |
| Best (Ensemble + CSDI + CatBoost) | **0.9130** | MAX(auroc) for CATBOOST + simple featurization |
| Handcrafted mean | **0.8304** | AVG(auroc) for simple1.0 featurization |
| MOMENT-embedding mean | **0.7040** | AVG(auroc) for MOMENT-embedding |

**CRITICAL**: Previous version contained unverified values (0.831 vs 0.792). All AUROC values must be sourced from DuckDB queries, not hallucinated.

### Three-Tier Triage System - Conceptual (Values Illustrative)

| Tier | Criteria | Action | Volume |
|------|----------|--------|--------|
| Low Risk | High confidence negative | Routine follow-up | ~50-60% |
| High Risk | High confidence positive | Expedited exam | ~25-35% |
| Uncertain | Uncertainty > threshold | Specialist review | ~10-15% |

*Note: Exact percentages are illustrative and depend on clinical threshold selection.*

---

## SECTION 6: METHODS/DISCUSSION INTEGRATION

### Missing Citations for Methods

| Reference | Purpose | Priority |
|-----------|---------|----------|
| Allen et al. 2019 | Raincloud plots | ESSENTIAL |
| Demšar 2006 | Critical difference diagrams | ESSENTIAL |
| Simonsohn et al. 2020 | Specification curves | ESSENTIAL |
| Vickers & Elkin 2006 | Decision curve analysis | ESSENTIAL |
| Cohen 1988 | Effect size conventions | Recommended |
| Mure et al. 2009 | PLR study example for uncertainty discussion | NEW |

**NEW CITATION** (author request):
> Mure LS et al. (2009). "Melanopsin Bistability: A Fly's Eye Technology in the Human Retina." PLoS ONE 4(6): e5991. [Example of PLR study not accounting for preprocessing uncertainty in downstream statistics]

### Discussion Points to Strengthen

1. **Calibration-discrimination tradeoff**: Traditional methods show better calibration despite lower AUROC
2. **Error propagation**: Outlier detection errors compound through imputation → features → classification
3. **Zero-shot competitiveness**: MOMENT zero-shot matching SAITS (trained) is a key finding
4. **Prevalence sensitivity**: Results at 27% sample prevalence will differ substantially at 3.54% population prevalence
5. **CORE MESSAGE**: TSFM preprocessing is the main contribution; classification is a downstream proxy

### Limitations to Add/Strengthen

1. Single-center, single-ethnicity data (Singapore, Chinese)
2. EPV = 7.0 for handcrafted (borderline); EPV = 0.58 for embeddings (unacceptable)
3. Missing severity stratification
4. Single annotator ground truth
5. Enriched prevalence limiting calibration assessment
6. **Pilot study nature** - methodology contribution, not clinical deployment

---

## SECTION 7: FUTURE RESEARCH PRIORITIES

### Top 5 Experiments (Ranked by Impact)

| Rank | Experiment | Purpose | Notes |
|------|-----------|---------|-------|
| 1 | **Dimensionality Reduction Study** | Control for embedding dimensionality | Compare embeddings at 8/16/32/64/96 features |
| 2 | **Hybrid FM + Handcrafted** | Best of both worlds | |
| 3 | **Conformal Prediction** | Calibrated uncertainty | |
| 4 | **Severity Stratification** | Early disease detection | |
| 5 | **Epistemic/Aleatoric Decomposition** | Uncertainty interpretation | |

### Key Research Question

> **Why do MOMENT embeddings (0.704 mean AUROC) underperform handcrafted features (0.830 mean AUROC)?**

**AUTHOR CAVEAT**: Cannot derive generalizable insights from the embedding gap without systematic dimensionality reduction study. Future research should compare:
- PCA, UMAP, t-SNE, NMF for embedding compression
- Various feature lengths (4, 8, 16, 32, 64, 96)
- LoRA-like adapters for dimensionality reduction
- Match feature dimensionality (8 features) between embedding and handcrafted methods

**Hypothesis**: Embeddings capture generic temporal patterns but miss disease-specific signatures (melanopsin dynamics) that require domain knowledge.

**Counter-hypothesis** (domain expert perspective): The canonical photoreceptor contributions—phasic, sustained, PIPR—should be relatively straightforward to learn even with traditional PCA, as these represent well-defined temporal patterns. Unlike tasks requiring extensive human knowledge (e.g., sarcasm detection), PLR feature extraction does not appear inherently tricky. The embedding gap may reflect sample size constraints rather than fundamental representational limitations.

---

## SECTION 8: ACTION ITEMS

### Immediate (Before Submission)

- [ ] **FIX**: Calibration slope interpretation (ERROR 1) + add sample size context
- [ ] **FIX**: Add Type I SS order-dependence note (ERROR 2)
- [ ] **CLARIFY**: CD diagram replicates definition
- [ ] **ADD**: EPV calculations to methods (8 features → EPV = 7.0)
- [ ] **ADD**: Missing citations (Allen 2019, Demšar 2006, Simonsohn 2020, Mure 2009)
- [ ] **VERIFY**: All AUROC values against DuckDB (GT=0.911, best=0.913)
- [ ] **IMPLEMENT**: pminternal instability plots (ggplot2)

### For Discussion Section

- [ ] Emphasize preprocessing as CORE contribution
- [ ] Strengthen ground truth comparison discussion (with verified values)
- [ ] Add calibration-discrimination tradeoff interpretation
- [ ] Discuss prevalence sensitivity for deployment
- [ ] Add dimensionality reduction limitation caveat

---

## SECTION 9: THE "GROUND TRUTH" COMPARISON - A DEEPER PHILOSOPHICAL DISCUSSION

### 9.1 What We Call "Ground Truth" Is Not The Truth

In the Foundation PLR study, "ground truth" for preprocessing was created through a specific human annotation protocol:

1. **Outlier Detection**: Human experts visually marked blinks, artifacts, and spurious segments
2. **Imputation**: Missing segments were reconstructed using cEEMD (Complementary Ensemble Empirical Mode Decomposition) + MissForest

This process creates a cleaned signal optimized for **smoothness and physiological plausibility**. But we must ask a fundamental question:

> **During a blink, what IS the "true" pupil diameter?**

The answer is deeply unsettling: **it is fundamentally unobservable**. The pupil is occluded by the eyelid. Any value we assign during this period is, by definition, an inference—an interpolation—a guess informed by human priors about what "should" be there.

### 9.2 The Epistemological Problem

Our "ground truth" encodes several implicit assumptions:

| Assumption | Why It Might Be Wrong |
|------------|----------------------|
| **Smooth transitions** | Real pupil dynamics during blinks may be more complex than smooth interpolation suggests |
| **Symmetric recovery** | Assuming pre-blink ≈ post-blink ignores transient pupillary responses |
| **Artifact = noise** | Some "artifacts" may carry diagnostic information (e.g., blink frequency patterns in neurological conditions) |
| **Expert consensus** | Different annotators would produce different "ground truths" |

### 9.3 Implications for Clinical Deployment

Given the comparison between automated and human preprocessing:

#### The Safety Argument for Human-Validated Preprocessing

Despite any performance differences, ground truth preprocessing is **safer** because:
1. It removes potential information leakage confounds
2. It reflects clinically-validated expectations of signal quality
3. Results are interpretable in physiological terms
4. Regulatory bodies understand human-validated pipelines

#### Recommended Resolution

1. **Use ground truth for primary analysis** (safety first)
2. **Report automated results as sensitivity analysis** (transparently explore differences)
3. **Conduct prospective validation** with independent annotators
4. **Investigate disease-artifact correlations** explicitly

### 9.4 Future Vision: End-to-End Probabilistic PLR Reconstruction

The ultimate solution is to **eliminate discrete preprocessing steps** entirely:

```
Current Approach (Discrete Pipeline):
Raw Signal → [Outlier Detection] → [Imputation] → [Featurization] → Classification
                    ↓                   ↓                ↓
              (uncertainty          (uncertainty      (uncertainty
               discarded)            discarded)        discarded)

Future Approach (End-to-End Probabilistic):
Raw Video → SAMv3 pupil segmentation (with uncertainty)
         → MOMENT reconstruction (with confidence intervals)
         → Handcrafted features (with propagated uncertainty)
         → CatBoost/TabPFN (with calibration)
         → Selective classification (abstain when uncertain)
```

---

## SECTION 10: REFERENCES

### Soft Targets and Label Uncertainty

- Beat-SSL (Tao et al. 2025): Soft targets in ECG contrastive learning
- Hinton et al. 2015: Knowledge distillation with soft labels
- STAPLE (Warfield et al. 2004): Consensus ground truth algorithm

### Medical Image Annotation Uncertainty

- Joskowicz et al. 2019: Inter-observer variability in medical image segmentation
- Kohl et al. 2018: Probabilistic U-Net for ambiguous segmentation

### Probabilistic Time Series Models

- Rasul et al. 2021: Autoregressive denoising diffusion for time series
- Salinas et al. 2020 (DeepAR): Probabilistic forecasting
- MOMENT (Goswami et al. 2024): Foundation model for time series

### PLR Preprocessing Uncertainty

- Mure et al. 2009: Example study not accounting for preprocessing uncertainty

---

*This review synthesizes findings from 6 specialized expert agents analyzing the Foundation PLR figure reports against statistical, visualization, STRATOS compliance, clinical, methodological, and future research criteria, with author annotations integrated in v2.*
