# Pipeline Overview

Foundation PLR implements a four-stage preprocessing sensitivity analysis pipeline.

## Research Question

> **How do preprocessing choices (outlier detection → imputation) affect downstream prediction quality when using handcrafted physiological features?**

## The Error Propagation Chain

```
Raw PLR Signal (with blinks, artifacts, noise)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ [1] OUTLIER DETECTION                                           │
│     Question: Can FMs detect artifacts as well as humans?       │
│     ERRORS HERE → propagate downstream!                         │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ [2] IMPUTATION                                                  │
│     Question: Can FMs reconstruct missing segments well?        │
│     ERRORS HERE → affect feature extraction!                    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ [3] FEATURIZATION (FIXED - handcrafted features)                │
│     - Amplitude bins (histogram of pupil size)                  │
│     - One latency feature (PIPR, MEDFA, etc.)                   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ [4] CLASSIFICATION (FIXED - CatBoost)                           │
│     - CatBoost is the best classifier (established)             │
│     - DO NOT compare classifiers - that's not the question      │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
ALL STRATOS-COMPLIANT METRICS
```

## STRATOS Compliance

We evaluate preprocessing effects on **all** STRATOS-compliant metrics:

| Category | Metrics |
|----------|---------|
| **Discrimination** | AUROC |
| **Calibration** | Slope, Intercept, O:E ratio |
| **Overall** | Brier score, Scaled Brier (IPA) |
| **Clinical Utility** | Net Benefit, Decision Curve Analysis |

!!! warning "Not AUROC-only"
    AUROC is just ONE of many metrics. We measure preprocessing effects across ALL STRATOS domains.

## Subject Counts

| Task | N Subjects | Reason |
|------|------------|--------|
| Outlier Detection | 507 | All subjects have ground truth masks |
| Imputation | 507 | All subjects have denoised signals |
| Classification | 208 | Only 152 Control + 56 Glaucoma have labels |

## Key Files

| Stage | Entry Point | Module |
|-------|-------------|--------|
| Outlier Detection | `flow_anomaly_detection.py` | `src.anomaly_detection` |
| Imputation | `flow_imputation.py` | `src.imputation` |
| Featurization | `flow_featurization.py` | `src.featurization` |
| Classification | `flow_classification.py` | `src.classification` |
