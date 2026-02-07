# SHAP and Feature Importance Analysis: Handling Multicollinearity

## Status: MOSTLY COMPLETE (2026-02-02)

| Task | Status | Notes |
|------|--------|-------|
| 1.1 VIF pre-check in SHAP export | ✅ DONE | `scripts/export_shap_for_r.py` |
| 1.2 VIF annotations in SHAP figure | ✅ DONE | `src/r/figures/fig_shap_importance.R` |
| 2.1 CD diagram post-process coloring | ✅ DONE | `--color` flag added |
| 3.1 results.tex SHAP/VIF paragraph | ✅ DONE | New subsection added |
| 3.2 supplementary.tex VIF section | ✅ DONE | Full VIF analysis + alternatives |
| 3.3 SHAP figure in supplementary-data | ✅ DONE | Already present |
| 4.1 Pytest VIF test | ✅ DONE | `tests/test_vif_analysis.py` |

**Bibliography Note**: New citations (lundberg_unified_2017, molnar_interpretable_2019, chen_true_2020, covert_understanding_2020, aas_explaining_2021, apley_visualizing_2020) added to supplementary.tex thebibliography. Main references.bib is auto-generated from Zotero - user should add these to Zotero and regenerate.

---

## Executive Summary

This plan addresses the critical issue of **severe multicollinearity** (VIF > 60-114) in our PLR features and its impact on SHAP interpretation reliability. The plan covers:

1. Adding VIF pre-checks to SHAP pipeline
2. Annotating figures with VIF warnings
3. CD diagram coloring enhancement
4. Manuscript updates (main + supplementary)
5. Alternative feature importance methods

---

## Problem Statement

### Current VIF Values

| Feature | Mean VIF | Status |
|---------|----------|--------|
| Red_SUSTAINED_value | **114.0** | SEVERE |
| Red_MAX_CONSTRICTION_value | **104.3** | SEVERE |
| Blue_SUSTAINED_value | **70.9** | SEVERE |
| Blue_MAX_CONSTRICTION_value | **64.0** | SEVERE |
| Red_PHASIC_value | 9.8 | Moderate |
| Blue_PHASIC_value | 5.8 | Moderate |
| Red_PIPR_AUC_value | 4.3 | OK |
| Blue_PIPR_AUC_value | 4.0 | OK |

### Impact on SHAP

When VIF > 10, SHAP arbitrarily distributes importance among correlated features (Lundberg & Lee 2017; Molnar 2019). With VIF > 60-114, the SUSTAINED and MAX_CONSTRICTION features share so much variance that their individual SHAP rankings are **meaningless**.

### Physiological Reason for Correlation

SUSTAINED and MAX_CONSTRICTION are both derived from the same pupil constriction dynamics during PLR:
- MAX_CONSTRICTION = peak constriction amplitude
- SUSTAINED = maintained constriction after stimulus offset

These are physiologically coupled - a large MAX_CONSTRICTION typically leads to large SUSTAINED response.

---

## Task Breakdown

### Phase 1: Pipeline Pre-checks and Annotations

#### Task 1.1: Add VIF Pre-check to SHAP Export

**File**: `scripts/export_shap_for_r.py`

**Changes**:
```python
def check_vif_before_shap(vif_path: Path, threshold: float = 10.0) -> tuple[bool, list]:
    """
    Check VIF values and warn if multicollinearity is severe.

    Returns:
        (is_safe, problematic_features)
    """
    with open(vif_path) as f:
        vif_data = json.load(f)

    problematic = []
    for feat in vif_data["data"]["aggregate"]:
        if feat["VIF_mean"] and feat["VIF_mean"] > threshold:
            problematic.append({
                "feature": feat["feature"],
                "vif": feat["VIF_mean"],
                "concern": feat["concern"]
            })

    is_safe = len(problematic) == 0
    return is_safe, problematic

def main():
    # ... existing code ...

    # NEW: VIF pre-check
    vif_path = PROJECT_ROOT / "data" / "r_data" / "vif_analysis.json"
    if vif_path.exists():
        is_safe, problematic = check_vif_before_shap(vif_path)
        if not is_safe:
            print("\n" + "="*60)
            print("⚠️  VIF WARNING: Multicollinearity Detected!")
            print("="*60)
            print("The following features have VIF > 10:")
            for p in problematic:
                print(f"  - {p['feature']}: VIF = {p['vif']:.1f} ({p['concern']})")
            print("\nSHAP values for these features should be interpreted with CAUTION.")
            print("Consider using grouped SHAP or alternative methods.")
            print("="*60 + "\n")

    # Continue with SHAP export...
```

#### Task 1.2: Add VIF Annotations to SHAP Figure

**File**: `src/r/figures/fig_shap_importance.R`

**Insertion Points**:
- Insert VIF loading after line 52 (after `shap_data <- fromJSON(...)`)
- Insert VIF merge before line 62 (before creating `importance_df`)
- The `importance_with_vif` dataframe replaces `importance_df` in subsequent visualizations

**Changes**:
1. Load VIF data alongside SHAP data
2. Add VIF values as annotations to feature labels
3. Add color/symbol coding for VIF concern level
4. Add figure disclaimer annotation

```r
# Load VIF data (INSERT after line 52)
vif_data <- fromJSON("data/r_data/vif_analysis.json")
vif_df <- as_tibble(vif_data$data$aggregate)

# Merge VIF with SHAP importance
importance_with_vif <- importance_df %>%
  left_join(vif_df %>% select(feature, VIF_mean, concern), by = "feature") %>%
  mutate(
    # Add VIF to label for high-concern features
    feature_label = case_when(
      concern == "High" ~ paste0(feature, " [VIF=", round(VIF_mean, 0), "]"),
      concern == "Moderate" ~ paste0(feature, " (VIF=", round(VIF_mean, 0), ")"),
      TRUE ~ as.character(feature)
    ),
    # Symbol for concern level
    vif_symbol = case_when(
      concern == "High" ~ "†",
      concern == "Moderate" ~ "*",
      TRUE ~ ""
    )
  )

# Add disclaimer annotation to figure
p_shap <- p_shap +
  labs(
    caption = paste0(
      "† VIF > 20 (high multicollinearity): SHAP values unreliable\n",
      "* VIF 5-10 (moderate): interpret with caution"
    )
  )
```

#### Task 1.3: Create Combined SHAP + VIF Figure

**New File**: `src/r/figures/fig_shap_with_vif.R`

Create a 2-panel figure:
- Panel A: SHAP importance (existing)
- Panel B: VIF bar chart for same features

This provides visual context for interpreting SHAP reliability.

### Phase 2: CD Diagram Coloring

#### Task 2.1: Add Post-process Coloring Option

**File**: `src/r/figure_system/cd_diagram.R`

**Changes**: Add `color_labels` parameter for optional post-processing

```r
create_cd_diagram <- function(results_matrix,
                               alpha = 0.05,
                               cex = 1.0,
                               abbreviate = TRUE,
                               descending = TRUE,
                               left_margin = 8,
                               right_margin = 8,
                               reset_par = TRUE,
                               color_labels = NULL) {  # NEW: optional color mapping

  # ... existing validation and plotting code ...

  # Create base CD diagram with scmamp
  scmamp::plotCD(
    results.matrix = results_for_plot,
    alpha = alpha,
    cex = cex
  )

  # NEW: Post-process coloring if requested
  if (!is.null(color_labels)) {
    # Get method names and their colors
    method_names <- colnames(results_matrix)
    if (abbreviate) {
      method_names <- abbreviate_methods(method_names)
    }

    # Add colored text overlays
    # Note: This is hacky but works for static figures
    for (i in seq_along(method_names)) {
      name <- method_names[i]
      if (name %in% names(color_labels)) {
        # Get position from scmamp internals (approximate)
        # Add colored text annotation
        mtext(name, side = 2, line = i * 0.5, col = color_labels[[name]],
              las = 1, cex = cex * 0.8)
      }
    }
  }

  invisible(results_matrix)
}
```

#### Task 2.2: Define Category Colors for CD Diagrams

**File**: `src/r/figures/fig_cd_diagrams.R`

```r
# Load category colors from YAML
category_colors <- list(
  "GT" = color_defs[["--color-category-ground-truth"]],
  "Ens-Full" = color_defs[["--color-category-ensemble"]],
  "MOMENT-ft" = color_defs[["--color-category-foundation-model"]],
  "LOF" = color_defs[["--color-category-traditional"]],
  "TimesNet" = color_defs[["--color-category-deep-learning"]]
  # ... map all abbreviated names to category colors
)

# Pass to create_cd_diagram
create_cd_diagram(
  outlier_matrix,
  color_labels = category_colors,  # NEW
  abbreviate = TRUE
)
```

### Phase 3: Manuscript Updates

#### Task 3.1: Main Results Text (1-2 sentences)

**File**: `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/latent-methods-results/results.tex`

**Location**: After classification results section

**Text**:
```latex
\paragraph{Feature Importance Analysis}
We computed SHAP values \citep{lundberg2017unified} to assess feature contributions
to glaucoma prediction. However, variance inflation factor (VIF) analysis revealed
severe multicollinearity among temporal PLR features (VIF = 64--114 for SUSTAINED
and MAX\_CONSTRICTION features), which compromises the reliability of individual
feature importance rankings \citep{molnar2019interpretable}. Notably, the MOMENT-based
preprocessing pipeline showed substantially lower multicollinearity (VIF = 11.6 for
Red\_SUSTAINED vs.\ mean 114), suggesting that foundation model preprocessing may
partially decorrelate temporal features. The full SHAP analysis is reported in
Supplementary Materials with appropriate caveats (Supplementary Figure~X).
```

#### Task 3.2: Supplementary Materials Paragraph

**File**: `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/supplementary.tex`

**NOTE**: The supplementary.tex is at the manuscript ROOT, not in supplementary-data/ subdirectory.

**Text**:
```latex
\subsection{Feature Importance Analysis with SHAP}
\label{sec:shap-analysis}

We employed SHapley Additive exPlanations (SHAP) \citep{lundberg2017unified} to
quantify feature contributions to glaucoma prediction. SHAP values were computed
using TreeExplainer for the CatBoost classifier, with bootstrap resampling (B=1000)
to estimate confidence intervals.

\paragraph{Multicollinearity Concerns}
Prior to SHAP interpretation, we computed Variance Inflation Factors (VIF) for all
features (Supplementary Table~X). Four features exhibited severe multicollinearity:
\begin{itemize}
    \item Red\_SUSTAINED\_value: VIF = 114
    \item Red\_MAX\_CONSTRICTION\_value: VIF = 104
    \item Blue\_SUSTAINED\_value: VIF = 71
    \item Blue\_MAX\_CONSTRICTION\_value: VIF = 64
\end{itemize}

These high VIF values ($>$10) indicate that SHAP arbitrarily distributes importance
among correlated features \citep{molnar2019interpretable, chen2020trueto}.
Specifically, the marginal Shapley value formulation assumes feature independence
when computing conditional expectations, creating unrealistic feature combinations
for correlated features \citep{janzing2020feature, frye2020asymmetric}.

\paragraph{Interpretation}
Supplementary Figure~\ref{fig:shap-importance} shows Mean |SHAP| values for Ground
Truth and Ensemble preprocessing pipelines. Features are annotated with VIF values
where multicollinearity is a concern. The PIPR\_AUC features (VIF $<$ 5) provide
the most reliable importance estimates. For the temporal features (SUSTAINED,
MAX\_CONSTRICTION), the importance should be interpreted collectively rather than
individually---both feature groups capture pupil constriction dynamics that
contribute to glaucoma detection, but the exact split of importance between them
is an artifact of the SHAP formulation.

\paragraph{Alternative Approaches}
For applications requiring reliable individual feature importance with correlated
features, we recommend:
\begin{enumerate}
    \item \textbf{Grouped SHAP}: Aggregate correlated features using hierarchical
          clustering before computing Shapley values \citep{covert2020understanding}
    \item \textbf{Conditional SHAP}: Use the shapr package which respects feature
          dependencies \citep{aas2021explaining}
    \item \textbf{Permutation importance with clustering}: Select one representative
          feature per correlated cluster \citep{scikit-learn}
\end{enumerate}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{fig_shap_importance_combined.png}
    \caption{Mean |SHAP| feature importance for Ground Truth (A) and Ensemble (B)
    preprocessing pipelines. Error bars indicate 95\% bootstrap confidence intervals.
    Features annotated with VIF values in brackets (e.g., [VIF=114]) exhibit severe
    multicollinearity; their individual importance rankings should be interpreted
    with caution. Blue features (469nm) probe melanopsin pathway; red features
    (640nm) probe cone pathway.}
    \label{fig:shap-importance}
\end{figure}
```

#### Task 3.3: Copy Figure to Supplementary Data

**Option A**: Manual copy
```bash
cp figures/generated/ggplot2/supplementary/fig_shap_importance_combined.png \
   /home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/supplementary-data/
```

**Option B**: Add Makefile target (preferred for reproducibility)
```makefile
# In Makefile
copy-supplementary-figures:
	cp figures/generated/ggplot2/supplementary/fig_shap_*.png \
	   ../sci-llm-writer/manuscripts/foundationPLR/supplementary-data/
	cp figures/generated/ggplot2/supplementary/fig_vif_*.png \
	   ../sci-llm-writer/manuscripts/foundationPLR/supplementary-data/
```

### Phase 4: Alternative Methods Exploration

#### Task 4.1: Literature Review Summary

Based on research, alternative methods for correlated features:

| Method | Handles Correlation? | Python Package | Recommendation |
|--------|---------------------|----------------|----------------|
| **Grouped SHAP (PartitionExplainer)** | Yes (via grouping) | `shap` (built-in) | **RECOMMENDED** |
| **Conditional SHAP** | Yes (dependence-aware) | `shapr` (R+Python) | Best principled approach |
| **SAGE** | Partial (equal split) | `sage-importance` | Good for global importance |
| **ALE plots** | Yes (conditional) | `PyALE`, `alibi` | Effects, not rankings |
| **Permutation + clustering** | Yes (via clustering) | `sklearn` | Simple baseline |

**Key References**:
- Chen et al. 2020 "True to the Model or True to the Data?" (from your bib: `chen2020trueto`)
- Frye et al. 2020 "Asymmetric Shapley values" (from your bib: `frye2020asymmetric`)
- Covert et al. 2020 "Understanding Global Feature Contributions with SAGE" (NeurIPS)
- Aas et al. 2021 "Explaining Individual Predictions with Conditional Shapley Values"

#### Task 4.2: Optional Future Implementation

**NOT in scope for current publication**, but for future work:

1. **Grouped SHAP Implementation**:
```python
import shap

# Cluster features by correlation
clustering = shap.utils.hclust(X_train, metric="correlation")

# Use PartitionExplainer
explainer = shap.PartitionExplainer(
    model.predict_proba,
    X_train,
    clustering=clustering
)
shap_values = explainer(X_test)

# Visualize with clustering
shap.plots.bar(shap_values, clustering=clustering, clustering_cutoff=0.5)
```

2. **Conditional SHAP via shapr**:
```python
# Requires R installation
from shaprpy import explain

explanation = explain(
    model=model,
    x_explain=X_test,
    x_train=X_train,
    approach="gaussian"  # Assumes multivariate Gaussian
)
```

---

## Implementation Order

### Priority 1: Critical Path (Must Have)
1. [ ] Task 1.1: VIF pre-check in SHAP pipeline
2. [ ] Task 1.2: VIF annotations on SHAP figure
3. [ ] Task 3.1: Main results text
4. [ ] Task 3.2: Supplementary paragraph
5. [ ] Task 3.3: Copy figure to supplementary

### Priority 2: Nice to Have
6. [ ] Task 1.3: Combined SHAP + VIF figure
7. [ ] Task 2.1: CD diagram color option
8. [ ] Task 2.2: Category colors for CD

### Priority 3: Future Work
9. [ ] Task 4.2: Grouped SHAP implementation

---

## Testing Checklist

### Automated Tests
- [ ] `pytest tests/test_vif_analysis.py` - VIF concern level assignment
- [ ] `pytest tests/test_shap_extraction.py` - SHAP export with VIF warning

### Manual Verification
- [ ] VIF pre-check runs without error
- [ ] VIF warning appears when VIF > 10
- [ ] SHAP figure shows VIF annotations
- [ ] Figure caption includes multicollinearity disclaimer
- [ ] `fig_shap_importance_combined.png` copied to supplementary-data/
- [ ] LaTeX compiles without errors
- [ ] CD diagram with colors renders correctly (if implemented)

### Test File to Create

**File**: `tests/test_vif_analysis.py`

```python
import pytest
from scripts.compute_vif import get_concern_level

def test_vif_concern_temporal_ok():
    assert get_concern_level(5.0, "Blue_temporal") == "OK"
    assert get_concern_level(9.9, "Red_temporal") == "OK"

def test_vif_concern_temporal_moderate():
    assert get_concern_level(10.0, "Blue_temporal") == "Moderate"
    assert get_concern_level(15.0, "Red_temporal") == "Moderate"

def test_vif_concern_temporal_high():
    assert get_concern_level(20.0, "Blue_temporal") == "High"
    assert get_concern_level(114.0, "Red_temporal") == "High"

def test_vif_concern_other_features():
    assert get_concern_level(4.0, "Latency") == "OK"
    assert get_concern_level(7.0, "Other") == "Moderate"
    assert get_concern_level(12.0, "Other") == "High"
```

---

## Reviewer Feedback Integration

### Iteration 1: Initial Plan

**Statistical Reviewer Comments**:
- "VIF threshold of 10 is standard, but for temporal features from the same physiological process, consider a higher threshold (20)"
- "The plan correctly identifies the marginal vs conditional SHAP distinction"
- "Recommend adding reference to Janzing et al. 2020 for causal interpretation"

**Implementation Reviewer Comments**:
- "CD diagram post-process coloring is hacky but acceptable for publication figures"
- "Consider adding a simple test to verify VIF annotations appear correctly"
- "The shapr package requires R - note this in alternative methods section"

### Iteration 2: Refined Plan

**Changes Made**:
1. Added physiological context for temporal feature thresholds (Section: Problem Statement)
2. Added Janzing et al. 2020 reference (Section: Task 3.2)
3. Added R dependency note for shapr (Section: Task 4.1)
4. Added testing checklist (Section: Testing Checklist)

**Statistical Reviewer Final Comments**:
- "Plan is comprehensive. The acknowledgment of SHAP limitations is scientifically honest."
- "Recommend keeping SHAP figure in supplementary rather than removing entirely - shows due diligence."

**Implementation Reviewer Final Comments**:
- "Implementation plan is clear. VIF pre-check is a good defensive programming practice."
- "CD diagram coloring is optional enhancement - don't block publication on it."

### Iteration 3: Final Plan (Converged)

No further changes requested. Plan approved by both reviewers.

---

## References

1. Lundberg SM, Lee SI (2017). A unified approach to interpreting model predictions. NeurIPS.
2. Molnar C (2019). Interpretable Machine Learning. https://christophm.github.io/interpretable-ml-book/
3. Chen H, Janizek JD, Lundberg S, Lee SI (2020). True to the Model or True to the Data? arXiv:2006.16234.
4. Frye C, Rowat C, Feige I (2020). Asymmetric Shapley values: incorporating causal knowledge. arXiv:1910.06358.
5. Janzing D, Minorics L, Blöbaum P (2020). Feature relevance quantification in explainable AI: A causal problem. AISTATS.
6. Covert I, Lundberg S, Lee SI (2020). Understanding Global Feature Contributions With Additive Importance Measures. NeurIPS.
7. Aas K, Jullum M, Løland A (2021). Explaining individual predictions when features are dependent. AI & Statistics.
8. Hooker G, Mentch L, Zhou S (2019). Please Stop Permuting Features: An Explanation and Alternatives. arXiv:1905.03151.

---

## Appendix: VIF Interpretation Guide

### Temporal Features (Same-Stimulus: Blue_*, Red_*)

| VIF Range | Concern Level | SHAP Reliability |
|-----------|---------------|------------------|
| < 10 | OK | Reliable |
| 10-20 | Moderate | Interpret with caution |
| >= 20 | High | Only interpret as feature groups |

### Other Features (Cross-Stimulus, Latency)

| VIF Range | Concern Level | SHAP Reliability |
|-----------|---------------|------------------|
| < 5 | OK | Reliable |
| 5-10 | Moderate | Interpret with caution |
| >= 10 | High | Only interpret as feature groups |

### Physiological Context

- SUSTAINED and MAX_CONSTRICTION are physiologically coupled (same constriction dynamics)
- PIPR_AUC is relatively independent (VIF < 5) - most reliable for SHAP
- The relaxed threshold for temporal features acknowledges expected correlation structure

### Notable Finding: Preprocessing Effect on VIF

The MOMENT-gt-finetune + SAITS configuration shows substantially lower VIF:
- Red_SUSTAINED: VIF = 11.6 (vs. mean 114 across configs)
- This suggests foundation model preprocessing may partially decorrelate temporal features

This preprocessing effect on multicollinearity could be explored in future work.
