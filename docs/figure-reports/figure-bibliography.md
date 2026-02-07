# Figure Bibliography from Code

> **Purpose**: References extracted from code comments in Python and R figure/statistics scripts. These links document the methodological foundations and inspirations for visualizations and statistical analyses.

Generated: 2026-01-28

---

## STRATOS Guidelines & Performance Metrics

### Van Calster 2024 - STRATOS Performance Evaluation

**Source**: `src/r/figures/fig_calibration_dca_combined.R` (lines 6-8)

```
Van Calster, Ben, Gary S. Collins, Andrew J. Vickers, et al. 2024.
"Performance Evaluation of Predictive AI Models to Support Medical Decisions:
Overview and Guidance."
arXiv:2412.10288. Preprint, arXiv, December 13.
```

**Links**:
- arXiv: https://doi.org/10.48550/arXiv.2412.10288
- Lancet Digital Health: https://doi.org/10.1016/j.landig.2025.100916

**Context**: STRATOS guidelines for calibration (smoothed curve with CI, slope, intercept, O:E, Brier) and DCA (Net benefit curves with treat-all/treat-none references)

**Used in Figures**:
- `fig_calibration_dca_combined.png` (main)
- `fig_multi_metric_raincloud.png` (main)
- `fig_prob_dist_combined.png` (main)

---

## Visualization Style & Design

### Economist/ggplot2 Style Tutorial

**Source**: `src/r/theme_foundation_plr.R` (line 5), `src/r/color_palettes.R` (line 7), `src/viz/plot_config.py` (line 393)

**Link**: https://altaf-ali.github.io/ggplot_tutorial/challenge.html

**Context**: Economist-style visualization with off-white background. Features: subtle grid lines, bold axis text, clean professional appearance.

**Used in**: All ggplot2 figures via `theme_foundation_plr()`

### Paul Tol's Colorblind-Safe Palettes

**Source**: `src/r/color_palettes.R` (line 50)

**Link**: https://personal.sron.nl/~pault/

**Context**: Qualitative color palette designed for accessibility. Used for method category coloring.

**Used in Figures**:
- `fig_forest_combined.png` (method categories)
- `fig_cd_diagrams.png` (method categories)
- `fig_raincloud_auroc.png` (pipeline types)

---

## Uncertainty Quantification & AURC

### Benchmarking Uncertainty Estimation

**Source**: `src/stats/uncertainty_quantification.py` (line 67)

**Link**: https://github.com/IdoGalil/benchmarking-uncertainty-estimation-performance

**Citation**: Ding et al. (2020): "Revisiting the Evaluation of Uncertainty Estimation and Its Application to Explore Model Similarity and Transferability"

**Context**: AURC (Area Under Risk-Coverage curve) implementation for selective classification

**Used in Figures**:
- `fig_roc_rc_combined.png` (Risk-Coverage panel)
- `fig_selective_classification.png` (all panels)

### Uncertainty in Deep Learning

**Source**: `src/stats/uncertainty_quantification.py` (lines 176-283)

**Links**:
- Heteroscedastic loss: https://shrmtmt.medium.com/beyond-average-predictions-embracing-variability-with-heteroscedastic-loss-in-deep-learning-f098244cad6f
- Predictive entropy: https://github.com/kyle-dorman/bayesian-neural-network-blogpost
- Mutual information: https://torch-uncertainty.github.io/api.html#diversity
- Uncertainty baselines: https://github.com/google/uncertainty-baselines
- GBDT uncertainty: https://github.com/yandex-research/GBDT-uncertainty
- CatBoost uncertainty: https://towardsdatascience.com/tutorial-uncertainty-estimation-with-catboost-255805ff217e
- Conformal prediction: https://github.com/PacktPublishing/Practical-Guide-to-Applied-Conformal-Prediction/blob/main/Chapter_05_TCP.ipynb

**Citation**: Nado et al. 2022, https://arxiv.org/abs/2106.04015

**Context**: Epistemic vs aleatoric uncertainty decomposition for ensemble models

**Used in Analysis**: Uncertainty metrics in `fig_selective_classification.png`

---

## Calibration Methods

### Calibration Curve Generation

**Source**: `src/stats/classifier_calibration.py` (lines 43-51), `src/stats/calibration_metrics.py` (lines 51-81)

**Links**:
- sklearn calibration: https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
- Ensemble calibration tutorial: https://github.com/huyng/incertae/blob/master/ensemble_classification.ipynb
- CatBoost conformal prediction: https://www.kaggle.com/code/banddaniel/rain-pred-catboost-conformal-prediction-f1-0-84

**Citations**:
- Alasalmi et al. 2020: "Better Classifier Calibration for Small Datasets" https://doi.org/10.1145/3385656
- Nixon et al. 2019: "Measuring Calibration in Deep Learning" https://openreview.net/forum?id=r1la7krKPS
- Wu and Gales 2020: "Should Ensemble Members Be Calibrated?" https://openreview.net/forum?id=wTWLfuDkvKp

**Context**: Calibration curve computation and ensemble calibration strategies

**Used in Figures**:
- `fig_calibration_dca_combined.png` (calibration panel)

### Beta Calibration & Platt Scaling

**Source**: `src/stats/classifier_calibration.py` (lines 63-73)

**Links**:
- Beta calibration package: https://pypi.org/project/betacal/
- Beta calibration tutorial: https://github.com/REFRAME/betacal/blob/master/python/tutorial/Python%20tutorial.ipynb
- Conformal classification: https://github.com/aangelopoulos/conformal_classification
- Ethen ML tutorial: https://ethen8181.github.io/machine-learning/model_selection/prob_calibration/prob_calibration.html

**Context**: Alternative calibration methods (not currently used but documented for future reference)

---

## Clinical Metrics & Glaucoma Screening

### Two-Way Partial AUC (tpAUC)

**Source**: `src/stats/classifier_metrics.py` (lines 27-62)

**Links**:
- Paper: https://doi.org/10.1186/s12911-023-02382-2
- Code: https://github.com/statusrank/A-Generic-Framework-for-Optimizing-Two-way-Partial-AUC
- LibAUC docs: https://docs.libauc.org/examples/pauc_sotas.html
- LibAUC paper: https://arxiv.org/abs/2306.03065

**Citation**: Neto et al. (2024): "A novel estimator for the two-way partial AUC"

**Context**: Partial AUC computation for specific sensitivity/specificity targets relevant to glaucoma screening

### PPV/NPV and Clinical Metrics

**Source**: `src/stats/classifier_metrics.py` (lines 71-91)

**Links**:
- NCBI book: https://www.ncbi.nlm.nih.gov/books/NBK430867/
- Eisenberg 1995: https://www.ccjm.org/content/ccjom/62/5/311.full.pdf
- DOIs: https://doi.org/10.1093/biostatistics/kxr008, https://doi.org/10.1177/0272989x9601600205, https://doi.org/10.1016/j.jclinepi.2008.04.007

**Context**: Prevalence-adjusted PPV/NPV calculations

### Glaucoma Screening Guidelines

**Source**: `src/stats/classifier_metrics.py` (lines 30-32)

**Links**:
- Glaucoma screening wiki: https://github.com/petteriTeikari/glaucoma_screening/wiki
- PMCID 8873198: https://pmc.ncbi.nlm.nih.gov/articles/PMC8873198/
- Nature Eye: https://doi.org/10.1038/s41433-024-03056-7

**Citation**: Najjar et al. 2021 (BJO): http://dx.doi.org/10.1136/bjophthalmol-2021-319938

**Context**: Clinical validation targets for glaucoma screening with PLR

---

## LLM-based Anomaly Detection

### SigLLM

**Source**: `scripts/sigllm_anomaly_detection.py` (lines 18-159)

**Links**:
- Original repo: https://github.com/sintel-dev/sigllm
- Fork: https://github.com/petteriTeikari/sigllm
- Orion tutorials: https://github.com/sintel-dev/Orion/blob/master/tutorials/

**Context**: LLM-based anomaly detection for PLR signals (experimental, not used in main figures)

---

## Seaborn & Matplotlib Styling

**Source**: `src/viz/viz_styling_utils.py` (lines 12, 25), `src/viz/viz_subplots.py` (line 45)

**Links**:
- Seaborn aesthetics: https://seaborn.pydata.org/tutorial/aesthetics.html#removing-axes-spines
- Font face handling: https://stackoverflow.com/a/18962217/6412152

**Context**: Python visualization styling utilities

---

## XGBoost Calibration Notes

**Source**: `src/stats/calibration_metrics.py` (lines 58-64)

**Links**:
- StackExchange Q1: https://stats.stackexchange.com/a/617182/294507
- StackExchange Q2: https://stats.stackexchange.com/a/619981/294507

**Context**: Documentation that XGBoost probabilities are not well-calibrated and require post-hoc calibration (Platt scaling, isotonic regression, beta calibration). Note: Our models use CatBoost with proper calibration.

---

## Summary: Key References by Figure

| Figure | Primary Reference | Topic |
|--------|------------------|-------|
| `fig_calibration_dca_combined` | Van Calster 2024 | STRATOS calibration + DCA |
| `fig_roc_rc_combined` | Ding et al. 2020 | Risk-Coverage curves |
| `fig_selective_classification` | Geifman & El-Yaniv 2017* | Selective classification |
| `fig_multi_metric_raincloud` | Van Calster 2024 | All STRATOS metrics |
| `fig_forest_combined` | Paul Tol palettes | Colorblind-safe method comparison |
| `fig_cd_diagrams` | Demšar 2006* | Critical difference diagrams |
| `fig_specification_curve` | Simonsohn 2020* | Multiverse analysis |
| `fig_shap_importance_*` | Lundberg & Lee 2017* | SHAP values |
| All ggplot2 figures | Altaf-Ali tutorial | Economist styling |

*Not found in code comments but documented in figure reports

---

## Integration Notes

These references should be:
1. Added to the manuscript bibliography (`biblio-pupil.bib`)
2. Cited in figure captions where appropriate
3. Discussed in Methods section when describing analysis approaches

The STRATOS guidelines (Van Calster 2024) are the primary methodological reference and should be cited for all calibration, DCA, and probability distribution figures.
