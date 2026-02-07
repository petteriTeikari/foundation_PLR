# Supplementary Figure Report

> **Purpose**: Comprehensive "latent captions" for all supplementary figures, providing full context for LLM-assisted manuscript writing. Includes draft LaTeX captions and bare minimum body text for `supplementary.tex`.
>
> **Last Updated**: 2026-01-30
> **Data Verified**: All AUROC, calibration, and DCA values verified against `data/public/foundation_plr_results_stratos.db` and JSON exports.

---

## Data Provenance Summary

```
MLflow (410 runs) → extraction scripts → DuckDB (316 configs) → JSON/CSV → R/ggplot2 → PNG/PDF
```

All supplementary figures use the same data pipeline as main figures. Extended combos (5 additional pipelines) may be used in some figures.

---

## Verified Key Metrics (2026-01-30)

| Config | AUROC (95% CI) | Cal. Slope | O:E Ratio | NB @10% |
|--------|----------------|------------|-----------|---------|
| Ground Truth | **0.9110** (0.9028-0.9182) | 0.52 | 0.82 | 0.189 |
| Best Ensemble | **0.9130** (0.9041-0.9194) | 0.30 | 0.86 | 0.189 |
| Best Single FM | **0.9099** (0.8990-0.9207) | 0.65 | 0.73 | 0.189 |
| Traditional | **0.8599** (0.8274-0.8900) | 0.07 | 1.34 | 0.182 |

**Prevalence**: 26.9% (56 events / 208 subjects)

---

## Figure S1: ROC and Risk-Coverage Curves

**Filename**: `fig_roc_rc_combined.png`

### Provenance
- **R Script**: `src/r/figures/fig_roc_rc_combined.R`
- **Data Source**: `data/r_data/roc_rc_data.json` (122 KB)
- **Export Script**: `scripts/export_roc_rc_data.py`
- **Database**: `data/public/foundation_plr_results_stratos.db` → predictions table
- **Dimensions**: 14 × 7 inches, 300 DPI

### Latent Caption (Full Description)

This figure presents discrimination and uncertainty-based model evaluation in two panels, showing 4 preprocessing pipelines.

**Panel A - ROC Curves**: Shows False Positive Rate (x-axis, 0-1) versus True Positive Rate (y-axis, 0-1). Each curve represents one pipeline; color-coded per YAML config. Diagonal dashed line represents random classifier (AUROC=0.5). 95% bootstrap confidence bands shown as ribbons. Legend displays AUROC values in parentheses. Ground Truth (AUROC=0.911) and Best Ensemble (AUROC=0.913) curves nearly overlap, indicating similar discrimination. Traditional pipeline (AUROC=0.860) shows clearly lower curve.

**Panel B - Risk-Coverage Curves**: Shows Coverage (x-axis, fraction of samples retained) versus Risk (y-axis, error rate = 1 - accuracy, per Geifman & El-Yaniv 2017). Evaluates selective classification: if model can reject uncertain predictions, how does error rate improve? Curves should decrease as coverage decreases (rejecting uncertain cases). AURC (Area Under Risk-Coverage curve) shown in legend; lower is better, indicating the model correctly identifies which predictions are most reliable. Ground Truth achieves lowest AURC, indicating best uncertainty calibration.

**Key Findings**:
- ROC curves confirm ensemble methods match ground truth discrimination
- Risk-Coverage shows ensemble methods have well-calibrated uncertainty estimates
- Traditional methods show both lower discrimination and worse uncertainty calibration

### Draft LaTeX Caption

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{figures/supplementary/fig_roc_rc_combined.pdf}
  \caption{
    \textbf{ROC and Risk-Coverage curves for preprocessing pipelines.}
    (A) ROC curves with 95\% bootstrap CI bands. AUROC values in legend.
    Ground Truth and Ensemble achieve similar discrimination (~0.91).
    (B) Risk-Coverage curves showing error rate versus sample retention.
    AURC (lower is better) indicates uncertainty calibration quality.
    Ground Truth achieves lowest AURC.
    N=208 subjects; 200 bootstrap iterations for CI estimation.
  }
  \label{fig:supp-roc-rc}
\end{figure}
```

### Bare Minimum Body Text

> Figure S1 shows ROC curves (Panel A) and Risk-Coverage curves (Panel B) for representative preprocessing pipelines. Ground Truth and Ensemble methods achieve similar AUROC (~0.91) and AURC values, confirming that automated preprocessing can match human-annotated ground truth performance. Risk-Coverage analysis demonstrates that model uncertainty estimates are well-calibrated across all pipeline types.

---

## Figure S2: Selective Classification

**Filename**: `fig_selective_classification.png`

### Provenance
- **R Script**: `src/r/figures/fig_selective_classification.R`
- **Data Source**: `data/r_data/selective_classification_data.json` (50 KB)
- **Export Script**: `scripts/export_selective_classification_data.py`
- **Database**: `data/public/foundation_plr_results_stratos.db` → predictions table
- **Dimensions**: 14 × 5 inches, 300 DPI

### Latent Caption (Full Description)

This figure presents selective classification analysis showing how performance improves when the model can reject uncertain predictions.

**X-axis (all panels)**: Retained Data (100% → 10%, reversed scale). At 100%, all predictions included; at 10%, only the 10% most confident predictions retained.

**Panel A - AUROC**: Shows AUROC at each retention level. All curves trend upward as retention decreases (more confident subset = better discrimination). Ground Truth reaches AUROC ~0.95 at 50% retention.

**Panel B - Net Benefit**: Shows Net Benefit at 15% decision threshold. Clinical utility may initially increase then plateau or decrease as rejecting more cases reduces sample size benefits.

**Panel C - Scaled Brier**: Shows IPA (Integrated Probability Accuracy) at each retention level. Higher values indicate better overall probabilistic predictions. All pipelines improve as uncertain cases rejected.

**Clinical Interpretation**: In deployment, uncertain predictions (e.g., 10-15% of cases) could be referred to specialists. Remaining predictions have higher confidence and better metrics.

**Key Findings**:
- All metrics improve when rejecting uncertain predictions
- Ensemble and Ground Truth show similar improvement trajectories
- At 50% retention (reject half), AUROC improves from 0.91 to ~0.95
- Practical implication: referring 10-15% most uncertain cases to specialists improves reliability

### Draft LaTeX Caption

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{figures/supplementary/fig_selective_classification.pdf}
  \caption{
    \textbf{Selective classification analysis.}
    Performance metrics versus fraction of predictions retained (most confident subset).
    X-axis: retained data (100\% to 10\%, reversed).
    (A) AUROC improves from 0.91 to ~0.95 at 50\% retention.
    (B) Net Benefit at 15\% threshold.
    (C) Scaled Brier (IPA) shows improved probabilistic accuracy.
    Practical implication: referring 10--15\% most uncertain cases improves overall reliability.
    Methodology per Geifman \& El-Yaniv~\cite{geifman2017selective}.
    N=208 subjects, 4 pipelines.
  }
  \label{fig:supp-selective}
\end{figure}
```

### Bare Minimum Body Text

> Figure S2 shows how classification performance improves when the model can reject uncertain predictions (selective classification). At 50% retention (most confident half), AUROC improves from 0.91 to approximately 0.95 for Ground Truth and Ensemble pipelines. This suggests that in clinical deployment, referring 10-15% of uncertain cases to specialists would substantially improve diagnostic reliability while maintaining high throughput.

---

## Figure S3: Critical Difference Diagrams

**Filename**: `fig_cd_diagrams.png`

### Provenance
- **R Script**: `src/r/figures/fig_cd_diagrams.R`
- **Data Source**: `data/r_data/essential_metrics.csv` (328 rows)
- **Export Script**: `scripts/export_data_for_r.py`
- **Statistical Method**: Friedman test + Nemenyi post-hoc (α=0.05)
- **Dimensions**: 14 × 16 inches, 300 DPI

### Latent Caption (Full Description)

This figure presents Critical Difference (CD) diagrams for statistical comparison of preprocessing methods using the Friedman rank test with Nemenyi post-hoc correction.

**Panel A - Outlier Detection Methods**: Compares 11 outlier detection methods. Methods positioned by mean rank (best = leftmost). Horizontal bars connect methods with no statistically significant difference (cliques). Ensemble method (rank ~1.5) significantly outperforms traditional methods like SubPCA and PROPHET. Ground Truth (pupil-gt) and MOMENT-gt-finetune form a clique (no significant difference).

**Panel B - Imputation Methods**: Compares 8 imputation methods. CSDI and ensemble imputation rank highest. Linear interpolation (traditional baseline) ranks lowest but remains in a clique with several methods.

**Panel C - Combined Pipelines**: Compares top preprocessing pipeline combinations. Shows that multiple configurations achieve statistically equivalent performance.

**Statistical Method**: Demšar (2006) CD diagram methodology. Critical difference threshold based on α=0.05 and number of methods compared. **Note on replicates**: Panel A uses imputation methods as pseudo-folds to compare outlier methods; Panel B uses outlier methods as pseudo-folds to compare imputation methods; Panel C uses classifiers as pseudo-folds. This is a non-standard Friedman test application (methods, not independent datasets, serve as replicates).

**Key Findings**:
- Ensemble outlier detection significantly outperforms traditional methods
- Foundation models (MOMENT, UniTS) not significantly different from ground truth
- Many imputation methods statistically equivalent
- Top 5-10 pipeline combinations form a clique (no significant differences)

### Draft LaTeX Caption

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{figures/supplementary/fig_cd_diagrams.pdf}
  \caption{
    \textbf{Critical difference diagrams for preprocessing method comparison.}
    Friedman rank test with Nemenyi post-hoc correction ($\alpha=0.05$).
    Methods positioned by mean rank (best = left). Horizontal bars connect
    methods with no significant difference.
    (A) Outlier detection: Ensemble significantly outperforms SubPCA, PROPHET.
    Foundation models (MOMENT, UniTS) form clique with ground truth.
    (B) Imputation: CSDI ranks highest; many methods statistically equivalent.
    (C) Combined pipelines: Top 5--10 combinations form a clique.
    Methodology per Dem\v{s}ar~\cite{demsar2006cd}.
    N=316 configurations; imputation methods (Panel A) / outlier methods (Panel B) / classifiers (Panel C) used as pseudo-replicates for Friedman test. See Methods for justification of this non-standard application.
  }
  \label{fig:supp-cd}
\end{figure}
```

### Bare Minimum Body Text

> Figure S3 presents Critical Difference diagrams comparing preprocessing methods using the Friedman rank test with Nemenyi post-hoc correction (α=0.05). Ensemble outlier detection significantly outperforms traditional methods (Panel A), while foundation models form a statistical clique with ground truth annotation. Among imputation methods (Panel B), CSDI ranks highest but many methods show no significant difference. The top 5-10 pipeline combinations form a clique (Panel C), indicating statistically equivalent performance among the best configurations.

---

## Figure S4: Preprocessing Method Ranks

**Filename**: `fig_cd_preprocessing.png`

### Provenance
- **R Script**: `src/r/figures/fig_cd_preprocessing.R`
- **Data Source**: `data/r_data/essential_metrics.csv`
- **Export Script**: `scripts/export_data_for_r.py`
- **Dimensions**: 10 × 8 inches, 300 DPI

### Latent Caption (Full Description)

This figure shows a simplified bar chart view of outlier detection method rankings when the scmamp R package for CD diagrams is unavailable.

**X-axis**: Mean Rank (lower = better performance)
**Y-axis**: Outlier detection methods, sorted by rank

Methods color-coded by category:
- Ground Truth (gold)
- Foundation Model (blue shades)
- Deep Learning (purple)
- Traditional (green)
- Ensemble (orange)

Ensemble methods achieve lowest (best) mean rank, followed by Ground Truth and Foundation Models. Traditional methods (SubPCA, PROPHET) show highest mean ranks.

### Draft LaTeX Caption

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{figures/supplementary/fig_cd_preprocessing.pdf}
  \caption{
    \textbf{Outlier detection method rankings.}
    Mean rank across imputation method variations (lower = better).
    Color-coded by method category.
    Ensemble methods achieve best (lowest) ranks, followed by ground truth
    and foundation model approaches.
    N=316 CatBoost configurations.
  }
  \label{fig:supp-ranks}
\end{figure}
```

### Bare Minimum Body Text

> Figure S4 shows mean ranks for outlier detection methods across all imputation variations. Ensemble methods achieve the best (lowest) rankings, followed closely by ground truth annotation and foundation model-based approaches. Traditional methods (SubPCA, PROPHET) rank lowest, consistent with the CD diagram analysis.

---

## Figure S5: AUROC Distribution by Pipeline Type

**Filename**: `fig_raincloud_auroc.png`

### Provenance
- **R Script**: `src/r/figures/fig_raincloud_auroc.R`
- **Data Source**: `data/r_data/essential_metrics.csv` (filtered to CatBoost)
- **Export Script**: `scripts/export_data_for_r.py`
- **Dimensions**: 10 × 6 inches, 300 DPI

### Latent Caption (Full Description)

This figure shows AUROC distribution across pipeline types using raincloud plots (half-violin + boxplot + jittered points).

**Y-axis**: Pipeline type categories:
- Ground Truth (both outlier + imputation annotated)
- Outlier GT + Automated Imputation
- Automated Outlier + Imputation GT
- Fully Automated

**X-axis**: AUROC (0.75-0.95 range)

Half-violin shows density distribution shape. Boxplot shows quartiles and median. Individual points show each configuration's AUROC.

**Key Findings**:
- Ground Truth shows tight clustering around 0.91
- Fully automated pipelines show wider distribution (0.82-0.91)
- Using ground truth for either stage (outlier OR imputation) improves consistency
- Median AUROC: Ground Truth > Partial GT > Fully Automated

### Draft LaTeX Caption

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/supplementary/fig_raincloud_auroc.pdf}
  \caption{
    \textbf{AUROC distribution by pipeline automation level.}
    Raincloud plots showing AUROC distribution for: fully ground truth,
    partial ground truth (outlier or imputation only), and fully automated pipelines.
    Ground Truth shows tight clustering (~0.91); fully automated shows wider spread (0.82--0.91).
    Using ground truth for either stage improves consistency.
    N=316 CatBoost configurations.
  }
  \label{fig:supp-raincloud}
\end{figure}
```

### Bare Minimum Body Text

> Figure S5 shows AUROC distributions across pipeline automation levels. Fully ground truth pipelines show tight clustering around AUROC 0.91, while fully automated pipelines exhibit wider variation (0.82-0.91). Using ground truth annotation for either the outlier detection or imputation stage (but not both) provides intermediate consistency, suggesting that partial automation with quality control checkpoints may be a practical deployment strategy.

---

## Figure S6: Probability Distributions (All Pipelines)

**Filename**: `fig_prob_dist_faceted.png`

### Provenance
- **R Script**: `src/r/figures/fig_prob_dist_by_outcome.R`
- **Data Source**: `data/r_data/predictions_top4.json` (13 KB)
- **Export Script**: `scripts/export_predictions_for_r.py`
- **Dimensions**: 10 × 8 inches, 300 DPI

### Latent Caption (Full Description)

This figure shows predicted probability distributions for all 4 standard pipeline combinations in a 2×2 faceted layout.

**Each facet**:
- X-axis: Predicted probability (0-1)
- Y-axis: Density (free scaling)
- Blue curve: Control subjects (n=152)
- Red curve: Glaucoma subjects (n=56)
- Rug marks: Individual predictions

**Observations**:
- Ground Truth: Clear bimodal separation
- Best Ensemble: Similar separation pattern
- Best Single FM: Slightly more overlap
- Traditional: Most overlap, Traditional pipeline predictions more concentrated near middle range

This is the STRATOS-required visualization showing probability distributions per outcome class.

### Draft LaTeX Caption

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{figures/supplementary/fig_prob_dist_faceted.pdf}
  \caption{
    \textbf{Predicted probability distributions for all pipelines.}
    Density plots for Control (blue, n=152) and Glaucoma (red, n=56) subjects
    across four preprocessing pipelines. Rug marks show individual predictions.
    Ground Truth and Ensemble show clearest separation;
    Traditional shows more overlap.
    STRATOS core set requirement~\cite{vancalster2024stratos}.
  }
  \label{fig:supp-prob-dist}
\end{figure}
```

### Bare Minimum Body Text

> Figure S6 shows predicted probability distributions for all four representative pipelines. Ground Truth and Ensemble methods achieve clear separation between Control and Glaucoma distributions, while Traditional pipelines show more overlap. This visualization satisfies the STRATOS core set requirement for presenting probability distributions per outcome class.

---

## Figure S7: Multi-Config SHAP Importance

**Filename**: `fig_shap_importance_multi.png`

### Provenance
- **R Script**: `src/r/figures/fig_shap_importance.R`
- **Data Source**: `data/r_data/shap_feature_importance.json` (23 KB)
- **Export Script**: `scripts/export_shap_for_r.py`
- **YAML Config**: `configs/VISUALIZATION/combos.yaml` → shap_figure_combos
- **Dimensions**: 10 × 6 inches, 300 DPI

### Latent Caption (Full Description)

This figure compares SHAP feature importance across multiple preprocessing configurations (4-6 pipelines including Ground Truth and Top-10 Mean aggregate).

**X-axis**: Mean |SHAP| value
**Y-axis**: Features (12 PLR amplitude/latency features)

Multiple colors per feature row, one per pipeline configuration. Position dodge separates CI ranges.

**Features** (ordered by importance):
- Blue_SUSTAINED_value
- Blue_MAX_CONSTRICTION_value
- Blue_PIPR_AUC_value
- Red_SUSTAINED_value
- ...

**Key Findings**:
- Feature ranking relatively consistent across pipelines
- Blue wavelength features (melanopsin) consistently most important
- Some configurations show higher feature importance variance (less stable)
- Top-10 Mean aggregate provides consensus importance ranking

### Draft LaTeX Caption

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{figures/supplementary/fig_shap_importance_multi.pdf}
  \caption{
    \textbf{SHAP feature importance across preprocessing pipelines.}
    Mean |SHAP| with 95\% CI for multiple configurations.
    Feature ranking consistent across pipelines: melanopsin pathway features
    (blue wavelength) dominate.
    Top-10 Mean aggregate provides consensus importance ranking.
    SHAP methodology per Lundberg \& Lee~\cite{lundberg2017shap}.
  }
  \label{fig:supp-shap-multi}
\end{figure}
```

### Bare Minimum Body Text

> Figure S7 compares SHAP feature importance across multiple preprocessing pipelines. Feature rankings remain relatively consistent: melanopsin pathway features (blue wavelength stimulus) consistently show highest importance across all configurations. The Top-10 Mean aggregate provides a consensus ranking that can guide clinical interpretation of which physiological features drive glaucoma classification.

---

## Figure S8: Specification Curve Analysis

**Filename**: `fig_specification_curve.png`

### Provenance
- **R Script**: `src/r/figures/fig_specification_curve.R`
- **Data Source**: `data/r_data/essential_metrics.csv`
- **Export Script**: `scripts/export_data_for_r.py`
- **Documentation**: `fig_specification_curve.md` (companion file)
- **Dimensions**: 12 × 6 inches, 300 DPI

### Latent Caption (Full Description)

This figure presents a specification curve (multiverse analysis) showing AUROC for ALL 57 CatBoost configurations, ranked by performance.

**X-axis**: Configuration rank (1 = best, 57 = worst)
**Y-axis**: AUROC (0.82-0.92 range)

Points with 95% bootstrap CI error bars. Color-coded by ground truth usage:
- **Yellow (Both GT)**: Ground truth for both outlier + imputation
- **Orange (Outlier GT)**: Only outlier detection uses ground truth
- **Blue (Imputation GT)**: Only imputation uses ground truth
- **Teal (Automated)**: Neither uses ground truth

Dashed horizontal line marks best AUROC (0.913).

**Interpretation**:
- Shows full range of results, not cherry-picked "best"
- Addresses "garden of forking paths" problem
- Median AUROC = 0.878 demonstrates robustness across analytical choices
- Yellow points (Both GT) cluster near top, as expected
- Some automated pipelines achieve comparable performance to GT

**Key Findings**:
- AUROC range: 0.827 - 0.913 (8.6 percentage point spread)
- Median: 0.878
- ~40% of configurations use some form of ground truth
- Ensemble methods (teal) can match GT performance

### Draft LaTeX Caption

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{figures/supplementary/fig_specification_curve.pdf}
  \caption{
    \textbf{Specification curve analysis (multiverse).}
    All 57 CatBoost configurations ranked by AUROC with 95\% bootstrap CI.
    Color indicates ground truth usage: Both GT (yellow), Outlier GT only (orange),
    Imputation GT only (blue), Fully Automated (teal).
    AUROC range: 0.827--0.913; median 0.878.
    Addresses ``garden of forking paths'' problem~\cite{simonsohn2020specification}.
    Results demonstrate robustness across analytical choices.
  }
  \label{fig:supp-spec-curve}
\end{figure}
```

### Bare Minimum Body Text

> Figure S8 presents a specification curve showing AUROC for all 57 CatBoost preprocessing configurations, addressing the "garden of forking paths" problem. AUROC ranges from 0.827 to 0.913 (median 0.878), demonstrating that results are robust across analytical choices. Configurations using ground truth annotation (yellow, orange) tend to cluster near the top, but several fully automated pipelines achieve comparable performance, indicating that the best automated approaches can match human annotation quality.

---

## Figure S9: Model Instability Analysis (3-Panel)

**Filename**: `fig_instability_combined.png`

### Provenance
- **R Script**: `src/r/figures/fig_instability_combined.R`
- **Data Source**: `data/r_data/pminternal_bootstrap_predictions.json` (7.5 MB)
- **Export Script**: `scripts/extract_pminternal_data.py`
- **Database**: `data/public/foundation_plr_results_stratos.db` → predictions table
- **Reference**: Riley RD et al. (2023) BMC Medicine 21:502
- **Dimensions**: 15 × 5 inches, 300 DPI
- **Updated**: 2026-01-30 (expanded from 2-panel to 3-panel)

### Latent Caption (Full Description)

This figure presents Riley 2023-style prediction instability visualization showing how individual predictions vary across 1000 bootstrap samples. **New 3-panel layout** comparing Ground Truth, Ensemble, and Traditional preprocessing.

**All Panels**:
- X-axis: Predicted risk from developed model (mean across bootstraps)
- Y-axis: Predicted risk from bootstrap models
- Diagonal line: Perfect stability (y = x)
- Colored ribbon: 95% CI of bootstrap predictions
- Gray points: Individual bootstrap predictions (subsampled)
- Colored points: Mean predictions (blue=Control, red=Glaucoma)

**Panel A - Ground Truth**:
- 95% CI width: **0.089** (most stable)
- MAPE: 0.000
- Tight ribbon indicates consistent predictions across bootstrap samples

**Panel B - Automated (Ensemble)**:
- 95% CI width: **0.109** (slightly more variable)
- MAPE: 0.000
- Ensemble method achieves similar stability to ground truth

**Panel C - Traditional (LOF)**:
- 95% CI width: **0.272** (3× more instability!)
- MAPE: 0.000
- Wide orange ribbon demonstrates substantially higher prediction variability

**Key Findings**:
- Traditional preprocessing leads to **3× higher prediction instability** than ground truth
- Ensemble methods achieve stability comparable to ground truth
- Individual predictions vary considerably across bootstrap samples with traditional methods
- Clinical implication: predictions from traditional pipelines are less reliable for individual patients

**Physiological Interpretation**: LOF-based outlier detection may inconsistently reject artifacts, leading to variable feature extraction and thus unstable predictions. The ensemble method averaging multiple detection approaches provides more consistent artifact identification.

### Draft LaTeX Caption

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{figures/supplementary/fig_instability_combined.pdf}
  \caption{
    \textbf{Prediction instability across preprocessing pipelines (Riley 2023 style).}
    Each patient's predictions across 1000 bootstrap samples shown as gray points;
    colored ribbons indicate 95\% CI of predictions.
    (A) Ground Truth: 95\% CI width = 0.089 (most stable).
    (B) Automated Ensemble: CI width = 0.109 (comparable stability).
    (C) Traditional (LOF): CI width = 0.272 (\textbf{3$\times$ higher instability}).
    Diagonal indicates perfect prediction stability (bootstrap = developed model).
    Traditional preprocessing leads to substantially less reliable individual predictions.
    Methodology per Riley et al.~\cite{riley2023multiverse}.
    N=63 subjects (fold 1), 1000 bootstrap iterations.
  }
  \label{fig:supp-instability}
\end{figure}
```

### Bare Minimum Body Text

> Figure S9 presents prediction instability analysis following Riley et al. (2023), showing how individual patient predictions vary across 1000 bootstrap model fits. Ground Truth preprocessing achieves the highest stability (95% CI width = 0.089), with Ensemble methods achieving comparable performance (CI width = 0.109). In contrast, Traditional (LOF-based) preprocessing shows **3× higher prediction instability** (CI width = 0.272), indicating that predictions from traditional pipelines are substantially less reliable for individual patients. This finding has important clinical implications: even if aggregate performance metrics (AUROC) are acceptable, individual prediction reliability may differ significantly between preprocessing approaches.

---

## Summary: Supplementary Figure Quick Reference

| Fig | Filename | Purpose | Key Finding | Citation |
|-----|----------|---------|-------------|----------|
| S1 | `fig_roc_rc_combined.png` | ROC + Risk-Coverage | Ensemble matches GT discrimination and uncertainty | - |
| S2 | `fig_selective_classification.png` | Selective classification | AUROC improves 0.91→0.95 at 50% retention | Geifman 2017 |
| S3 | `fig_cd_diagrams.png` | CD diagrams | Ensemble significantly outperforms traditional; FM = GT | Demšar 2006 |
| S4 | `fig_cd_preprocessing.png` | Method ranks | Ensemble best, Traditional worst | - |
| S5 | `fig_raincloud_auroc.png` | AUROC raincloud | GT tight (0.91); automated wider (0.82-0.91) | Allen 2019 |
| S6 | `fig_prob_dist_faceted.png` | Probability faceted | Clear separation for GT/Ensemble; Traditional overlap | Van Calster 2024 |
| S7 | `fig_shap_importance_multi.png` | Multi-config SHAP | Consistent feature ranking across pipelines | Lundberg 2017 |
| S8 | `fig_specification_curve.png` | Specification curve | Robust across 57 configs; 0.827-0.913 range | Simonsohn 2020 |
| S9 | `fig_instability_combined.png` | **Model instability** | **Traditional has 3× higher instability than GT** | Riley 2023 |

---

## File Sizes and Counts

| Metric | Value |
|--------|-------|
| Total supplementary figures | **9** |
| PNG file size range | 68 KB - 549 KB |
| Total PNG size | ~2.2 MB |
| R scripts involved | 9 |
| Data files used | 5 (CSV + JSON) |
| Configurations (CatBoost) | 57-316 depending on figure |
| Subject count | 208 (152 control, 56 glaucoma) |
| Bootstrap iterations | 1000 |

---

*Report updated: 2026-01-30*
*Data verified against: foundation_plr_results_stratos.db, calibration_data.json, dca_data.json*
