# RULE: STRATOS Reporting Compliance (Van Calster 2024)

**Reference:** Van Calster B, Collins GS, Vickers AJ, et al. "Performance evaluation of predictive AI models to support medical decisions." STRATOS Initiative Topic Group 6.
**Full text:** `/home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-pupil/vancalster-2024-performance-measures-predictive-ai.md`

## Mandatory Measures (MUST report for ALL comparisons)

| Domain | Required Measure | Source |
|--------|-----------------|--------|
| Discrimination | AUROC with 95% CI | DuckDB `auroc` |
| Calibration | Smoothed calibration plot with CI | y_true, y_prob |
| Calibration | Calibration slope | DuckDB `calibration_slope` |
| Calibration | Calibration intercept | DuckDB `calibration_intercept` |
| Calibration | O:E ratio | DuckDB `o_e_ratio` |
| Overall | Brier score, Scaled Brier (IPA) | DuckDB `brier`, `scaled_brier` |
| Clinical Utility | Net Benefit at clinical thresholds | DuckDB `net_benefit_*pct` |
| Clinical Utility | DCA curves (5%-40% range) | Threshold sweep |
| Distributions | Probability distributions per outcome | y_prob by class |

## BANNED Measures

| Measure | Problem | Use Instead |
|---------|---------|-------------|
| F1 score | Improper, ignores TN | Net Benefit |
| AUPRC | Ignores TN | AUROC + DCA |
| pAUROC | No decision-analytic basis | DCA |
| Accuracy | Improper for clinical thresholds | Calibration + DCA |
| Youden optimization | Assumes equal costs | Clinical threshold + DCA |

## Data Pipeline

Each (outlier x imputation x classifier) combination must preserve: AUROC + CI, calibration slope/intercept/O:E, Brier/scaled Brier, Net Benefit at 5/10/15/20%, DCA curves, and raw predictions (y_true, y_prob).

## Compliance Checklist

Before publishing ANY comparison: AUROC with CI, calibration plot, calibration slope, O:E ratio, Net Benefit, DCA curves, probability distributions. NO F1/AUPRC/pAUROC/Youden.

## Metric Registry

Use `src/viz/metric_registry.py` for all metric definitions. NEVER hardcode metric names.
