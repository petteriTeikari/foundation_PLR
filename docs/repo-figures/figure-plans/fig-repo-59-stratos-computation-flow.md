# fig-repo-59: STRATOS Metrics Computation Flow

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-59 |
| **Title** | STRATOS Metrics: From Predictions to Publication |
| **Complexity Level** | L3 |
| **Target Persona** | Biostatistician / Research Scientist |
| **Location** | `src/stats/README.md`, `docs/explanation/stratos-metrics.md` |
| **Priority** | P1 (Critical) |

## Purpose

Show how raw model predictions (y_true, y_prob) are transformed into the full suite of STRATOS-compliant metrics (Van Calster 2024). Developers and reviewers need to trace the computation pipeline from bootstrap resampling through metric calculation to final DuckDB storage.

## Key Concept

STRATOS requires 5 metric domains. Each (outlier x imputation x classifier) combination produces predictions that feed through the same computation pipeline. All computation happens in Block 1 (extraction), never in Block 2 (visualization).

## Content Specification

### Panel 1: The 5 STRATOS Domains

```
┌───────────────────────────────────────────────────────────┐
│              STRATOS Metric Domains (Van Calster 2024)     │
├────────────────┬──────────────────────────────────────────┤
│ Discrimination │ AUROC + 95% CI                           │
│ Calibration    │ Slope, Intercept, O:E ratio              │
│ Overall        │ Brier score, Scaled Brier (IPA)          │
│ Clinical Util  │ Net Benefit @ 5%, 10%, 15%, 20%          │
│ Distributions  │ Probability distributions per outcome    │
└────────────────┴──────────────────────────────────────────┘
```

### Panel 2: Computation Pipeline

```
For each (outlier × imputation × classifier) config:
  │
  ▼
MLflow Run → Trained model + test predictions
  │
  ▼
Bootstrap Resampling (n=1000, stratified)
  │
  ├──▶ Each bootstrap sample:
  │      │
  │      ├── roc_auc_score(y_true, y_prob)          → AUROC
  │      ├── LogisticRegression.fit(logit(y_prob))   → Slope, Intercept
  │      ├── sum(y_true) / sum(y_prob)               → O:E ratio
  │      ├── mean((y_prob - y_true)²)                → Brier score
  │      └── nb = TPR×p - FPR×(1-p)×(t/(1-t))      → Net Benefit
  │
  ▼
Aggregate across 1000 bootstraps:
  │
  ├── Point estimate = mean(bootstrap_values)
  ├── CI_lower = percentile(bootstrap_values, 2.5%)
  └── CI_upper = percentile(bootstrap_values, 97.5%)
  │
  ▼
Store in DuckDB → essential_metrics table
```

### Panel 3: Where Each Metric Is Computed

| Metric | Computation Module | DuckDB Column | Viz Consumer |
|--------|--------------------|---------------|--------------|
| AUROC | `sklearn.metrics.roc_auc_score` | `auroc` | `src/viz/metric_registry.py` |
| Cal. Slope | `sklearn.linear_model.LogisticRegression` | `calibration_slope` | `src/viz/calibration_plot.py` |
| Cal. Intercept | `sklearn.linear_model.LogisticRegression` | `calibration_intercept` | `src/viz/calibration_plot.py` |
| O:E Ratio | `sum(y) / sum(p)` | `o_e_ratio` | `src/viz/metric_registry.py` |
| Brier | `sklearn.metrics.brier_score_loss` | `brier` | `src/viz/metric_registry.py` |
| Scaled Brier | `1 - brier/brier_null` | `scaled_brier` | `src/viz/metric_registry.py` |
| Net Benefit | `src/stats/clinical_utility.py` | `net_benefit_*pct` | `src/viz/dca_plot.py` |

### Panel 4: Banned Measures (STRATOS Non-Compliant)

```
╳ F1 score       → Ignores true negatives
╳ AUPRC          → Ignores true negatives
╳ pAUROC         → No decision-analytic basis
╳ Accuracy       → Improper for clinical thresholds
╳ Youden index   → Assumes equal misclassification costs
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `configs/CLS_EVALUATION.yaml` | `BOOTSTRAP.n_iterations=1000`, `BOOTSTRAP.alpha_CI=0.95` |
| `configs/CLS_EVALUATION.yaml` | `glaucoma_params.prevalence=0.0354` (for NB calculation) |

## Code Paths

| Module | Role |
|--------|------|
| `src/stats/classifier_metrics.py` | Bootstrap loop, metric aggregation |
| `src/stats/calibration_extended.py` | Calibration slope/intercept computation |
| `src/stats/clinical_utility.py` | Net benefit and DCA curve computation |
| `src/stats/scaled_brier.py` | Scaled Brier (IPA) calculation |
| `src/data_io/streaming_duckdb_export.py` | Stores all metrics in DuckDB |
| `src/stats/_defaults.py` | Default constants (n_bootstrap, ci_level) |

## Extension Guide

To add a new STRATOS metric:
1. Implement computation in `src/stats/` (verified library or formula)
2. Add column to `StreamingDuckDBExporter.ESSENTIAL_METRICS_SCHEMA`
3. Add extraction in `_extract_essential_metrics()`
4. Register in `src/viz/metric_registry.py` with display name
5. Add test in `tests/test_data_quality/test_extraction_correctness.py`

Note: Performance comparisons are in the manuscript, not this repository.
