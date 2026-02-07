# Metrics

Evaluation metrics utilities for imputation quality assessment.

## Overview

Provides functions for computing imputation metrics by preparing array triplets (input with missing values, ground truth, imputed output) in the format expected by PyPOTS evaluation and subject-wise metric computation.

## Modules

| Module | Purpose |
|--------|---------|
| `evaluate_imputation_metrics.py` | Compute and log imputation metrics to MLflow |
| `metrics_utils.py` | Prepare array triplets for PyPOTS metrics, subject-wise evaluation |

## See Also

- `src/imputation/` -- Imputation methods that produce the outputs evaluated here
- `src/stats/` -- STRATOS-compliant classification metrics (separate module)
