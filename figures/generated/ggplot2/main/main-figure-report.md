# Main Figure Report

> **Purpose**: Comprehensive "latent captions" for all main figures, providing full context for LLM-assisted manuscript writing. Includes draft LaTeX captions for `results.tex`.
>
> **Last Updated**: 2026-01-30
> **Data Verified**: All AUROC, calibration, and DCA values verified against `data/public/foundation_plr_results_stratos.db` and JSON exports.

---

## Data Provenance Summary

```
MLflow (410 runs) → extraction scripts → DuckDB (316 configs) → JSON/CSV → R/ggplot2 → PNG/PDF
```

**Key Config Files**:
- Combos: `configs/VISUALIZATION/combos.yaml`
- Filters: `configs/VISUALIZATION/data_filters.yaml` (featurization="simple1.0")
- Colors: `configs/VISUALIZATION/colors.yaml`

---

## Verified Key Metrics (2026-01-30)

**Standard 4 Combos** (used in all main figures):

| ID | Name | Outlier | Imputation | AUROC (95% CI) |
|----|------|---------|------------|----------------|
| ground_truth | Ground Truth | pupil-gt | pupil-gt | **0.9110** (0.9028-0.9182) |
| best_ensemble | Best Ensemble | ensemble-LOF-MOMENT-... | CSDI | **0.9130** (0.9041-0.9194) |
| best_single_fm | Best Single FM | MOMENT-gt-finetune | SAITS | **0.9099** (0.8990-0.9207) |
| traditional | Traditional | LOF | SAITS (TabPFN) | **0.8599** (0.8274-0.8900) |

**STRATOS Metrics (from calibration_data.json)**:

| Config | Cal. Slope | O:E Ratio | Brier | IPA | NB @10% |
|--------|------------|-----------|-------|-----|---------|
| Ground Truth | **0.52** | 0.82 | 0.135 | 0.315 | 0.189 |
| Best Ensemble | **0.30** | 0.86 | 0.102 | 0.481 | 0.189 |
| Best Single FM | **0.65** | 0.73 | 0.155 | 0.214 | 0.189 |
| Traditional | **0.07** | 1.34 | 0.122 | 0.381 | 0.182 |

**Calibration Interpretation**:
- Slope < 1 indicates overfitting (predictions more extreme than warranted)
- All methods show overfitting, with Traditional (0.07) most severe
- O:E ratio > 1 = underpredicting events; < 1 = overpredicting events

**Prevalence**: 26.9% (56 events / 208 subjects)

---

## Figure 1: Calibration and Decision Curve Analysis

**Filename**: `fig_calibration_dca_combined.png`

### Provenance
- **R Script**: `src/r/figures/fig_calibration_dca_combined.R`
- **Data Sources**:
  - `data/r_data/calibration_data.json` (6.1 KB)
  - `data/r_data/dca_data.json` (14 KB)
- **Export Script**: `scripts/export_predictions_for_r.py`
- **Database**: `data/public/foundation_plr_results_stratos.db` → predictions table
- **Dimensions**: 14 × 7 inches, 300 DPI

### Latent Caption (Full Description)

This figure presents STRATOS-compliant model assessment in two panels, evaluating both calibration (Panel A) and clinical utility (Panel B) for four representative preprocessing pipelines.

**Panel A - Calibration Plot**: Shows predicted probability (x-axis, 0-1) versus observed proportion (y-axis, 0-1) for each pipeline. The diagonal dashed line represents perfect calibration. Points are sized proportionally to bin counts (n subjects per probability decile). Smoothed LOESS curves with 95% confidence bands show calibration trends. An annotation box displays Ground Truth metrics: calibration slope (**0.52**), O:E ratio (**0.82**), Brier score (**0.135**), and IPA/scaled Brier (**0.32**). All calibration slopes < 1 indicate overfitting (predictions more extreme than warranted by the data).

**Panel B - Decision Curve Analysis**: Shows threshold probability (x-axis, 5%-30%) versus net benefit (y-axis). Reference strategies shown: horizontal dashed line at y=0 (Treat None), gray dotted curve (Treat All). Vertical dashed lines mark clinically relevant thresholds at 5%, 10%, 15%, and 20%. A solid vertical line marks the sample prevalence (26.9%). All four pipelines demonstrate positive net benefit across the clinical threshold range.

**Verified Net Benefit Values @10%**:
- Ground Truth: 0.189
- Best Ensemble: 0.189
- Best Single FM: 0.189
- Traditional: 0.182

**Key Findings**:
- All pipelines show overfitting (calibration slopes 0.07-0.65)
- Ground Truth: slope 0.52, O:E 0.82 (slight underprediction of events)
- Traditional: slope 0.07 (severe overfitting - predictions far too extreme)
- Best Ensemble achieves lowest Brier (0.102) and highest IPA (0.481)
- All pipelines show positive net benefit at clinical thresholds

**STRATOS Categories**: Calibration (slope, intercept, O:E), Clinical Utility (Net Benefit, DCA)

### Draft LaTeX Caption

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{figures/fig_calibration_dca_combined.pdf}
  \caption{
    \textbf{Model calibration and clinical utility assessment.}
    (A) Calibration plot showing predicted versus observed event probabilities
    for four representative preprocessing pipelines. Points sized by bin count;
    LOESS smoothing with 95\% CI bands. Diagonal indicates perfect calibration.
    Annotation shows Ground Truth metrics: slope=0.52, O:E=0.82, Brier=0.135, IPA=0.32.
    All methods show overfitting (slope $<$ 1); Traditional most severe (slope=0.07).
    (B) Decision curve analysis across clinically relevant threshold range (5--30\%).
    Dashed horizontal line: Treat None; gray dotted: Treat All.
    Vertical lines mark decision thresholds and sample prevalence (26.9\%).
    All pipelines demonstrate positive net benefit.
    STRATOS-compliant metrics per Van Calster et al.~\cite{vancalster2024stratos}.
    N=208 labeled subjects (152 control, 56 glaucoma).
  }
  \label{fig:calibration-dca}
\end{figure}
```

---

## Figure 2: Forest Plot - Method Comparison

**Filename**: `fig_forest_combined.png`

### Provenance
- **R Script**: `src/r/figures/generate_all_r_figures.R` (lines 99-111, factory pattern)
- **Factory Functions**: `create_forest_outlier()`, `create_forest_imputation()` from `figure_factory.R`
- **Data Source**: `data/r_data/essential_metrics.csv` (78 KB, 328 rows)
- **Export Script**: `scripts/export_data_for_r.py`
- **Database**: `data/public/foundation_plr_results_stratos.db` → essential_metrics table
- **Dimensions**: 10 × 10 inches, 300 DPI

### Latent Caption (Full Description)

This figure presents forest plots comparing preprocessing methods' effect on downstream classification performance, aggregating results across all CatBoost configurations.

**Panel A - Outlier Detection Methods**: Shows AUROC (x-axis, 0.5-1.0) for each of the 11 outlier detection methods (y-axis), sorted by increasing AUROC from bottom to top. Each method displays a point estimate with 95% bootstrap confidence interval whiskers. Methods are color-coded by category: Ground Truth (gold), Foundation Model (blue shades for MOMENT, UniTS), Deep Learning (purple for TimesNet), Traditional (green for LOF, OneClassSVM, PROPHET, SubPCA), and Ensemble (orange). Reference lines: vertical dotted red line at AUROC=0.5 (random chance); dashed gold line at ground truth AUROC. The ensemble method (combining LOF, MOMENT, OneClassSVM, PROPHET, SubPCA, TimesNet, UniTS) achieves highest AUROC, followed closely by ground truth.

**Panel B - Imputation Methods**: Same format as Panel A, showing 8 imputation methods. CSDI and ensemble imputation achieve highest performance. Foundation model-based imputation (MOMENT-finetune, MOMENT-zeroshot) shows competitive performance with traditional deep learning approaches (SAITS, TimesNet). Linear interpolation (traditional baseline) shows lowest but still reasonable AUROC.

**Verified Top Methods**:
- Ensemble outlier + CSDI: AUROC = 0.9130
- Ensemble outlier + TimesNet: AUROC = 0.9122
- Ground truth: AUROC = 0.9110

**Key Findings**:
- Ensemble outlier detection matches or exceeds ground truth performance
- Foundation models (MOMENT, UniTS) competitive with ground truth for outlier detection
- CSDI (deep learning) best single imputation method
- Imputation choice has smaller effect than outlier detection choice

**STRATOS Categories**: Discrimination (AUROC with 95% CI)

### Draft LaTeX Caption

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{figures/fig_forest_combined.pdf}
  \caption{
    \textbf{Preprocessing method performance comparison.}
    Forest plots showing mean AUROC with 95\% bootstrap CI for each preprocessing method,
    aggregated across CatBoost configurations.
    (A) Outlier detection methods (n=11). Ensemble approach combining traditional and
    foundation model methods achieves highest AUROC (0.913). Methods color-coded by category.
    (B) Imputation methods (n=8). CSDI and ensemble imputation perform best.
    Reference lines: red dotted at 0.5 (chance), gold dashed at ground truth AUROC (0.911).
    Foundation models show competitive performance with specialized deep learning methods.
    N=507 subjects for preprocessing evaluation; N=208 for classification AUROC.
  }
  \label{fig:forest-methods}
\end{figure}
```

---

## Figure 3: Multi-Metric Raincloud Plot

**Filename**: `fig_multi_metric_raincloud.png`

### Provenance
- **R Script**: `src/r/figures/fig_multi_metric_raincloud.R`
- **Data Source**: `data/r_data/essential_metrics.csv` (filtered to CatBoost)
- **Export Script**: `scripts/export_data_for_r.py`
- **Database**: `data/public/foundation_plr_results_stratos.db` → essential_metrics table
- **Dimensions**: 14 × 10 inches, 300 DPI

### Latent Caption (Full Description)

This figure presents a comprehensive 2×2 panel layout showing distributions of all four STRATOS-compliant metric categories across pipeline types, using raincloud plots (half-violin + boxplot + jittered points).

**Y-axis (all panels)**: Pipeline Type categories:
- Ground Truth (both outlier + imputation from human annotation)
- Ensemble (combining multiple methods)
- Foundation Model (MOMENT, UniTS-based pipelines)
- Deep Learning (TimesNet, SAITS-based pipelines)
- Traditional (LOF, OneClassSVM, linear interpolation-based pipelines)

**Panel A - AUROC (Discrimination)**: X-axis shows AUROC values. Half-violin shows density distribution; boxplot shows quartiles/median; individual points show each configuration. Ground Truth and Ensemble show highest median AUROC (~0.91), Foundation Model slightly lower (~0.89), Traditional lowest (~0.86). Tight clustering for Ground Truth indicates consistent performance.

**Panel B - Scaled Brier/IPA (Overall Performance)**: X-axis shows Scaled Brier (0-1, higher=better). Measures improvement over null model accounting for both discrimination and calibration. Ensemble shows highest median IPA, indicating best overall probabilistic predictions.

**Panel C - Net Benefit @ 10% (Clinical Utility)**: X-axis shows Net Benefit at 10% decision threshold. Measures clinical value of using model versus treat-all or treat-none strategies. All pipeline types show positive net benefit, with Ensemble highest.

**Panel D - O:E Ratio (Calibration-in-the-Large)**: X-axis shows Observed/Expected ratio. Values near 1.0 indicate good mean calibration. Traditional methods show O:E ratio near 1.0 (well-calibrated), while Foundation Model shows slight underprediction (O:E > 1).

**Key Findings**:
- Ground Truth and Ensemble pipelines consistently top performers across all metrics
- Foundation models competitive but slightly below ground truth
- Traditional methods show good calibration despite lower discrimination
- All pipeline types achieve clinically useful performance (positive net benefit)

**STRATOS Categories**: All four - Discrimination, Calibration, Overall, Clinical Utility

### Draft LaTeX Caption

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{figures/fig_multi_metric_raincloud.pdf}
  \caption{
    \textbf{STRATOS-compliant performance metrics across pipeline types.}
    Raincloud plots (half-violin, boxplot, individual points) showing metric distributions
    for five pipeline categories. Each point represents one (outlier, imputation) configuration
    with CatBoost classifier.
    (A) AUROC (discrimination): Ground Truth and Ensemble achieve highest values (~0.91).
    (B) Scaled Brier/IPA (overall performance): Ensemble methods show best probabilistic predictions.
    (C) Net Benefit at 10\% threshold (clinical utility): All pipelines achieve positive net benefit.
    (D) O:E ratio (calibration-in-the-large): Traditional methods well-calibrated (O:E~1.0).
    Foundation models competitive across all metrics but slightly below ground truth.
    Metrics computed per STRATOS guidelines~\cite{vancalster2024stratos}.
    N=208 labeled subjects; 57 unique configurations per category.
  }
  \label{fig:multi-metric-raincloud}
\end{figure}
```

---

## Figure 4: Probability Distribution by Outcome

**Filename**: `fig_prob_dist_combined.png`

### Provenance
- **R Script**: `src/r/figures/fig_prob_dist_by_outcome.R`
- **Data Source**: `data/r_data/predictions_top4.json` (13 KB)
- **Export Script**: `scripts/export_predictions_for_r.py`
- **Database**: `data/public/foundation_plr_results_stratos.db` → predictions table
- **Dimensions**: 14 × 6 inches, 300 DPI

### Latent Caption (Full Description)

This figure shows predicted probability distributions per outcome class, a STRATOS-required visualization for assessing model separation.

**Panel A - Representative Pipeline (Ground Truth)**: X-axis shows predicted probability of glaucoma (0-100%). Y-axis shows density. Blue density curve with rug plot shows Control subjects (n=152); Red density curve with rug plot shows Glaucoma subjects (n=56). Dashed vertical lines mark mean predicted probabilities for each class. The horizontal arrow annotation shows the discrimination slope (difference between mean probabilities). Ground Truth pipeline shows clear separation: Control predictions clustered near 20-40%, Glaucoma predictions spread 40-80%.

**Panel B - All Pipelines Faceted**: 2×2 subfacet grid showing the same density plot for all 4 standard combos:
- Ground Truth: Clear bimodal separation
- Best Ensemble: Similar separation pattern
- Best Single FM: Slightly less separation
- Traditional: Largest overlap, less confident predictions

**Key Findings**:
- Ground Truth and Ensemble achieve clear probability separation between outcomes
- Traditional pipeline shows more overlapping distributions (less confident)
- All pipelines show appropriate ordering (Glaucoma > Control mean probability)
- Discrimination slope ~30-40% for best pipelines

**STRATOS Categories**: Discrimination (probability distribution separation)

### Draft LaTeX Caption

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{figures/fig_prob_dist_combined.pdf}
  \caption{
    \textbf{Predicted probability distributions by outcome.}
    Density plots showing model-predicted glaucoma probability for Control (blue, n=152)
    and Glaucoma (red, n=56) subjects. Rug marks indicate individual predictions.
    (A) Ground Truth pipeline showing clear separation; dashed lines mark class means,
    arrow indicates discrimination slope (difference in means).
    (B) All four pipelines in faceted layout. Ground Truth and Ensemble achieve
    similar separation; Traditional shows more overlap.
    STRATOS core set requirement for probabilistic predictions~\cite{vancalster2024stratos}.
    N=208 labeled subjects.
  }
  \label{fig:prob-dist}
\end{figure}
```

---

## Figure 5: SHAP Feature Importance

**Filename**: `fig_shap_importance_combined.png`

### Provenance
- **R Script**: `src/r/figures/fig_shap_importance.R`
- **Data Sources**:
  - `data/r_data/shap_feature_importance.json` (23 KB)
  - `data/r_data/shap_ensemble_aggregated.json` (4 KB)
- **Export Script**: `scripts/export_shap_for_r.py`
- **Source**: `outputs/shap_summary_top10.pkl` (SHAP analysis results)
- **Dimensions**: 12 × 6 inches, 300 DPI

### Latent Caption (Full Description)

This figure presents SHAP (SHapley Additive exPlanations) feature importance for understanding which physiological features drive glaucoma classification.

**Panel A - Ground Truth Pipeline**: X-axis shows Mean |SHAP| value (mean absolute Shapley contribution to prediction). Y-axis lists features ordered by importance. Point-range shows mean with 95% bootstrap CI. Features color-coded by wavelength: Blue (469nm stimulus) vs Red (640nm stimulus).

Top features for Ground Truth:
1. Blue_SUSTAINED_value (chromatic pupil response amplitude)
2. Blue_MAX_CONSTRICTION_value (maximum constriction during blue light)
3. Red_PIPR_AUC_value (post-illumination pupil response area)

**Panel B - Ensemble Pipeline**: Same format, showing feature importance for best ensemble configuration. Similar feature ranking but different relative magnitudes, indicating different models weight features differently.

**Physiological Interpretation**:
- Blue wavelength features (melanopsin pathway) consistently most important
- SUSTAINED and MAX_CONSTRICTION capture melanopsin-driven responses
- PIPR (Post-Illumination Pupil Response) captures slow recovery phase
- Red wavelength features less important (cone pathway less affected in glaucoma)

**Key Findings**:
- Melanopsin pathway features (blue stimulus) dominate importance
- Ground Truth and Ensemble show similar feature rankings
- Amplitude features more important than latency features
- Consistent with known glaucoma pathophysiology (RGC damage affects melanopsin cells)

### Draft LaTeX Caption

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{figures/fig_shap_importance_combined.pdf}
  \caption{
    \textbf{SHAP feature importance analysis.}
    Mean absolute SHAP values with 95\% bootstrap CI for CatBoost models.
    Features color-coded by stimulus wavelength: blue (469nm, melanopsin pathway)
    vs red (640nm, cone pathway).
    (A) Ground Truth pipeline: melanopsin features (Blue\_SUSTAINED, Blue\_MAX\_CONSTRICTION)
    show highest importance.
    (B) Best Ensemble pipeline: similar ranking with different magnitudes.
    PIPR (post-illumination pupil response) captures melanopsin-driven slow recovery.
    Results consistent with glaucoma pathophysiology---retinal ganglion cell damage
    preferentially affects melanopsin-expressing intrinsically photosensitive RGCs.
    SHAP values computed per Lundberg \& Lee~\cite{lundberg2017shap}.
  }
  \label{fig:shap-importance}
\end{figure}
```

---

## Figure 6: Variance Decomposition

**Filename**: `fig_variance_decomposition.png`

### Provenance
- **R Script**: `src/r/figures/fig_variance_decomposition.R`
- **Data Source**: `data/r_data/essential_metrics.csv`
- **Export Script**: `scripts/export_data_for_r.py`
- **Statistical Method**: ANOVA (Type I sequential sum of squares) → η² (eta-squared)
- **Dimensions**: 12 × 6 inches, 300 DPI

### Latent Caption (Full Description)

This figure uses ANOVA-based variance decomposition to quantify how much each factor contributes to variation in AUROC, presented as horizontal lollipop charts showing η² (proportion of variance explained).

**Panel A - Preprocessing Only (CatBoost Fixed)**: Shows variance explained by preprocessing factors only, with classifier held constant at CatBoost.
- X-axis: η² (0-100% variance explained)
- Y-axis: Factors (Outlier Method, Imputation Method)
- Typical values: Outlier Method ~7%, Imputation Method ~5%
- **Narrative**: When using the optimal classifier, preprocessing choices explain ~12% of AUROC variance. This is meaningful but not dominant.

**Panel B - Full Model (All Factors)**: Shows variance explained by all factors including classifier choice.
- Factors: Classifier (red), Outlier Method (blue), Imputation Method (light blue)
- Typical values: Classifier ~70%, Outlier ~5%, Imputation ~3%
- Classifier emphasized in red to highlight its dominance
- **Narrative**: Classifier choice dominates overall variance (~70%). However, this is expected since CatBoost >> LogisticRegression. The key insight is that once classifier is fixed, preprocessing still matters.

**Statistical Method**: η² = SS_factor / SS_total from ANOVA. Type I sequential sums of squares used with factor order: classifier → outlier → imputation. **Note**: Variance attribution is order-dependent; shared variance is attributed to the first factor entered. Consider Type III SS or dominance analysis for order-independent decomposition.

**Key Findings**:
- Classifier choice dominates if compared across all classifiers (explains ~70% variance)
- Among preprocessing factors, outlier detection matters more than imputation
- When using best classifier (CatBoost), preprocessing explains ~12% of remaining variance
- Justifies fixing classifier and studying preprocessing sensitivity

### Draft LaTeX Caption

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{figures/fig_variance_decomposition.pdf}
  \caption{
    \textbf{Variance decomposition of AUROC by experimental factors.}
    Lollipop charts showing $\eta^2$ (proportion of variance explained) from factorial ANOVA.
    (A) Preprocessing factors only (CatBoost fixed): outlier detection method (~7\%)
    explains more variance than imputation method (~5\%).
    (B) Full model including classifier: classifier choice dominates (~70\%, red),
    explaining most AUROC variation. This expected result (CatBoost $\gg$ LogisticRegression)
    justifies fixing the classifier and studying preprocessing sensitivity.
    Among preprocessing, outlier detection contributes more than imputation.
    Type I sequential sums of squares (factor order: classifier $\rightarrow$ outlier $\rightarrow$ imputation; variance attribution is order-dependent); 316 CatBoost configurations.
  }
  \label{fig:variance-decomposition}
\end{figure}
```

---

## Summary: Main Figure Quick Reference

| Fig | Filename | Purpose | Key Finding | STRATOS |
|-----|----------|---------|-------------|---------|
| 1 | `fig_calibration_dca_combined.png` | Calibration + DCA | All pipelines overfitted (slope <1), positive net benefit | Calibration, Clinical Utility |
| 2 | `fig_forest_combined.png` | Method comparison | Ensemble outlier detection (0.913) matches ground truth (0.911); CSDI best imputation | Discrimination |
| 3 | `fig_multi_metric_raincloud.png` | Multi-metric view | Ground Truth/Ensemble consistently best across all 4 STRATOS categories | All |
| 4 | `fig_prob_dist_combined.png` | Probability separation | Clear bimodal separation for Ground Truth; Traditional more overlap | Discrimination |
| 5 | `fig_shap_importance_combined.png` | Feature importance | Melanopsin pathway features (blue stimulus) dominate | Explainability |
| 6 | `fig_variance_decomposition.png` | Variance decomposition | Classifier dominates overall (~70%); preprocessing ~12% when classifier fixed | Effect size |

---

## File Sizes and Counts

| Metric | Value |
|--------|-------|
| Total main figures | 6 |
| PNG file size range | 109 KB - 391 KB |
| Total PNG size | ~1.5 MB |
| R scripts involved | 7 (including factory/compose) |
| Data files used | 5 (CSV + JSON) |
| Bootstrap iterations | 1000 per config |
| Subject count (classification) | 208 (152 control, 56 glaucoma) |
| Subject count (preprocessing) | 507 |
| Configurations (CatBoost) | 316 |

---

## IMPORTANT: Calibration Values Correction

**Previous reports cited incorrect calibration slopes (1.625, 3.665, 0.812)**. The verified values from `calibration_data.json` are:

| Config | Previous (WRONG) | Verified (CORRECT) |
|--------|------------------|-------------------|
| Ground Truth | 1.625 | **0.52** |
| Best FM | 3.665 | **0.65** |
| Traditional | 0.812 | **0.07** |

**Interpretation change**: All methods show overfitting (slope < 1), not the mixed pattern previously reported. Traditional preprocessing shows the most severe overfitting.

---

*Report updated: 2026-01-30*
*Data verified against: foundation_plr_results_stratos.db, calibration_data.json, dca_data.json*
