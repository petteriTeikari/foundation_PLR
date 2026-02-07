# Figure Grouping Opportunities Analysis

**Created**: 2026-01-27
**Updated**: 2026-01-27 (FINAL - 6 Expert Reviews Synthesized)
**Purpose**: Organize ggplot2 figures into main/supplementary/extra-supplementary tiers
**Goal**: Maximize information gain while maintaining STRATOS compliance

---

## Expert Review Summary (6 Reviewers)

### Round 1: Three Domain Experts
| Expert | Key Finding |
|--------|-------------|
| **Biostatistics** | Missing calibration metrics visualization; need error propagation figure |
| **Visualization** | CD diagrams need 1×3 horizontal layout; SHAP 2×2 mixes plot types |
| **Journal Requirements** | 6 main figures appropriate; raincloud redundant with stratos_core |

### Round 2: Three Specialized Experts
| Expert | Key Finding |
|--------|-------------|
| **Information Gain** | fig_stratos_core = MAX info; fig_specification_curve undervalued; raincloud RETIRE |
| **Clinical Reader** | DCA/Calibration essential; missing diagnostic performance table; SHAP tangential |
| **Research Question** | fig_preprocessing_auroc CRITICAL MISSING; SHAP off-topic; VIF should be table |

---

## CONSENSUS ACROSS ALL REVIEWERS

### UNANIMOUS: Main Figures
| Figure | Consensus | All 6 Reviewers |
|--------|-----------|-----------------|
| **fig_stratos_core** | MAIN | ✓✓✓✓✓✓ |
| **fig_featurization_comparison** | MAIN | ✓✓✓✓✓✓ |
| **fig_cd_preprocessing** (single) | MAIN | ✓✓✓✓✓✓ |
| **fig_fm_dashboard** | MAIN | ✓✓✓✓✓✓ |

### UNANIMOUS: Retire/Reorganize
| Figure | Decision | Reason |
|--------|----------|--------|
| **fig_multi_metric_raincloud** | EXTRA-SUPP | Redundant with stratos_core |
| **fig_vif_analysis** | TABLE | Numeric diagnostics, not visual pattern |
| **fig_vif_by_wavelength** | TABLE | Numeric diagnostics |
| **fig_factorial_matrix** | METHODS | Experimental design, not results |

### UNANIMOUS: Critical Gaps
| Missing Item | Priority | Notes |
|--------------|----------|-------|
| **fig_preprocessing_auroc** | P0 | Core finding, referenced but missing |
| **fig_error_propagation** | P1 | Directly answers research question Q2 |
| **Calibration metrics annotation** | P1 | Add slope/intercept/O:E to stratos_core |

### SPLIT OPINIONS
| Figure | Views | Resolution |
|--------|-------|------------|
| **fig_specification_curve** | Info expert: ELEVATE; Clinical: "zero value" | SUPPLEMENTARY (research context) |
| **fig_cd_diagrams (3-panel)** | All: SUPPLEMENTARY | Move to supplementary |
| **SHAP figures** | Clinical: tangential; Info: moderate | EXTRA-SUPPLEMENTARY |
| **fig_heatmap_preprocessing** | Moderate value | SUPPLEMENTARY (full data reference) |

---

## FINAL FIGURE ORGANIZATION

### MAIN Figures (6 slots)

| Priority | Figure | Type | Story Element |
|----------|--------|------|---------------|
| **1** | fig_stratos_core | 2×2 panel | STRATOS-mandated metrics (ROC+Cal+DCA+Prob) |
| **2** | fig_cd_preprocessing | Single | Statistical validation of preprocessing effect |
| **3** | fig_featurization_comparison | Single | Core finding: FMs fail at features (+9pp handcrafted) |
| **4** | fig_fm_dashboard | Multi-panel | FM utility by task (wins preprocessing, loses features) |
| **5** | fig_heatmap_preprocessing | Single | Full preprocessing × AUROC landscape |
| **6** | fig_specification_curve | Multi-panel | Multiverse robustness (328 configs) |

*Note: fig_preprocessing_auroc is MISSING and should be created to replace one slot if budget allows*

### METHODS Figure (1 slot)

| Figure | Purpose |
|--------|---------|
| fig_factorial_matrix | Experimental design visualization |

### SUPPLEMENTARY Figures (5 slots)

| Figure | Purpose | Value |
|--------|---------|-------|
| fig_cd_diagrams (1×3) | Detailed CD breakdown | Statistical granularity |
| fig_prob_dist_faceted | Probability distributions by outcome | STRATOS detail |
| fig_variance_decomposition | η² effect sizes | Quantifies preprocessing importance |
| fig_shap_importance_gt | Feature importance (GT preprocessing) | Interpretability |
| fig_shap_importance_ensemble | Feature importance (Ensemble) | Comparison |

### EXTRA-SUPPLEMENTARY (Preserved but Low Priority)

| Figure | Reason for Demotion |
|--------|---------------------|
| fig_multi_metric_raincloud | Redundant with fig_stratos_core |
| fig_cd_combined | Subsumed by fig_cd_diagrams |
| fig_cd_outlier | Subsumed by fig_cd_diagrams |
| fig_cd_imputation | Subsumed by fig_cd_diagrams |
| fig_shap_beeswarm | Tangential to research question |
| fig_vif_analysis | Convert to table |
| fig_vif_by_wavelength | Convert to table |

---

## IMPLEMENTATION ACTIONS

### Immediate: Organize Existing Figures

```bash
# MAIN (copy, don't move - preserve originals)
cp fig_stratos_core.png main/
cp fig_cd_preprocessing.png main/
cp fig_featurization_comparison.png main/
cp fig_fm_dashboard.png main/
cp fig_heatmap_preprocessing.png main/
cp fig_specification_curve.png main/

# SUPPLEMENTARY
cp fig_cd_diagrams.png supplementary/
cp fig_prob_dist_faceted.png supplementary/
cp fig_shap_importance_gt.png supplementary/
cp fig_shap_importance_ensemble.png supplementary/

# EXTRA-SUPPLEMENTARY (redundant/tangential)
cp fig_multi_metric_raincloud.png extra-supplementary/
cp fig_cd_combined.png extra-supplementary/
cp fig_cd_outlier.png extra-supplementary/
cp fig_cd_imputation.png extra-supplementary/
cp fig_shap_beeswarm.png extra-supplementary/
cp fig_vif_analysis.png extra-supplementary/
cp fig_vif_by_wavelength.png extra-supplementary/

# METHODS (separate category)
# fig_factorial_matrix stays in place, referenced from Methods section
```

### Future: Create Missing Figures

1. **fig_preprocessing_auroc** - Distribution of AUROC across preprocessing configs
2. **fig_error_propagation** - Waterfall: outlier errors → imputation → classification
3. **Calibration metrics annotation** - Add slope/intercept/O:E to fig_stratos_core panel (b)

### Code Changes Needed

1. **fig_cd_diagrams.R:331** - Change layout from 3×1 to 1×3:
   ```r
   # Current: combined <- p_outlier / p_imputation / p_combined
   # Change to: combined <- p_outlier + p_imputation + p_combined
   ```

2. **fig_stratos_core.R** - Add calibration metrics annotation to panel (b)

---

## INFORMATION GAIN SUMMARY

| Tier | Figures | Unique Information | Text-Substitutable? |
|------|---------|-------------------|---------------------|
| **MAIN** | 6 | High - Complex patterns, distributions, multi-dimensional | No |
| **SUPPLEMENTARY** | 5 | Moderate - Detail, granularity, robustness | Partially |
| **EXTRA-SUPP** | 7 | Low - Redundant or tangential | Mostly yes |

---

## CLINICAL RELEVANCE SUMMARY

| Tier | Clinical Value |
|------|---------------|
| **MAIN** | Essential - DCA, Calibration in stratos_core; FM comparison |
| **SUPPLEMENTARY** | Technical validation - Statistical details |
| **EXTRA-SUPP** | ML research only - No clinical decision support |

---

## RESEARCH QUESTION ALIGNMENT

| Tier | Alignment Score |
|------|-----------------|
| **MAIN** | 9-10/10 - Directly answers preprocessing → performance question |
| **SUPPLEMENTARY** | 6-8/10 - Supports but doesn't directly answer |
| **EXTRA-SUPP** | 3-5/10 - Tangential or redundant |

---

## EXPERT REVIEW DETAILS

### Information Gain Expert (Round 2)

**Key Insight**: fig_stratos_core has MAXIMUM information gain - 4 STRATOS dimensions simultaneously, impossible to convey in text.

**Undervalued Figure**: fig_specification_curve shows full distribution pattern (328 configs) - visual pattern cannot be conveyed by "mean ± SD".

**Text-Substitutable** (demote or convert):
- fig_variance_decomposition: Three η² values → one sentence
- VIF diagnostics: Precise numbers → table

### Clinical Reader Expert (Round 2)

**MUST-HAVE**: DCA curves, Calibration plots, Probability distributions
**MISSING**: Diagnostic performance table (sensitivity/specificity at thresholds)
**TANGENTIAL**: SHAP figures (interpretability, not preprocessing effect)

**Critical Question Not Answered**: "Which preprocessing method should my clinic use?"

### Research Question Expert (Round 2)

**Direct Alignment (10/10)**:
- fig_stratos_core
- fig_cd_preprocessing
- fig_preprocessing_auroc (MISSING)

**Off-Topic**:
- SHAP figures (about WHICH features, not preprocessing effects)
- VIF (technical diagnostic)

**Critical Gap**: fig_error_propagation needed to answer Q2 (error cascade)

---

## GEMINI REVIEW FEEDBACK (Round 3)

### Critical Methodological Issues Identified

1. **MOMENT Embeddings EPV Violation (FATAL)**
   - EPV = 0.58 with 96 dimensions and 56 events
   - "You cannot publish a predictive model with an EPV of 0.58"
   - **Resolution**: Move ALL embedding analysis to Supplementary Appendix B
   - Frame as "exploratory hypothesis-generating" not primary result

2. **"Ground Truth" Naming Paradox**
   - Automated methods beat "ground truth" → paradoxical
   - **Resolution**: Rename to "Expert-Curated Baseline" or "Manual Reference"

3. **fig_featurization_comparison Adjustment**
   - The 9pp gap finding remains valid as QUALITATIVE insight
   - But detailed comparison moves to Supplementary Appendix B
   - Main text: Single sentence referencing supplementary

### Updated Figure Assignment (Post-Gemini)

**DEMOTED from MAIN to SUPPLEMENTARY:**
- fig_featurization_comparison → Move to supplementary as "Supplementary Figure S2"
- Frame as "exploratory analysis" with acknowledged EPV limitations

**MAIN Figures (REVISED - 5 slots):**
| Priority | Figure | Type | Story Element |
|----------|--------|------|---------------|
| **1** | fig_stratos_core | 2×2 panel | STRATOS-mandated metrics |
| **2** | fig_cd_preprocessing | Single | Statistical validation |
| **3** | fig_fm_dashboard | Multi-panel | FM utility by task (preprocessing focus) |
| **4** | fig_heatmap_preprocessing | Single | Full preprocessing landscape |
| **5** | fig_specification_curve | Multi-panel | Multiverse robustness |

*Note: fig_featurization_comparison demoted per Gemini EPV critique*

### Manuscript Text Updates Required

**In results.tex:**
```latex
% Replace detailed embedding section with:
"Handcrafted features consistently outperformed generic embeddings
in our exploratory analysis; full comparison is provided in
Supplementary Appendix B. Given the high dimensionality of learned
representations relative to sample size (EPV < 1), this comparison
serves as hypothesis-generating rather than a primary statistical finding."
```

**In discussion.tex:**
```latex
% Preface the 9pp gap interpretation with:
"As shown in our exploratory analysis (Supplementary Appendix B),
foundation model embeddings underperformed handcrafted features by
approximately 9 percentage points. While this finding requires
validation in larger cohorts to meet EPV requirements, it suggests
that generic temporal representations do not spontaneously isolate
the melanopsin-mediated dynamics required for glaucoma classification."
```

---

## CONVERGENCE STATEMENT (FINAL)

After 7 expert reviews (3 domain + 3 specialized + Gemini methodological):

**The plan has converged with critical adjustment:**

- **5 Main Figures**: Maximum information gain, STRATOS-compliant, statistically defensible
- **6 Supplementary Figures**: Statistical detail, exploratory embedding analysis
- **7 Extra-Supplementary**: Preserved but deprioritized
- **1 Methods Figure**: Experimental design

**Key Gemini-driven change**: MOMENT embedding comparison demoted to Supplementary Appendix B due to EPV violation. This "bulletproofs" the manuscript against fatal statistical critique.

**The paper now focuses on PREPROCESSING** (where FMs excel) rather than being a "mixed bag" of wins and losses.
