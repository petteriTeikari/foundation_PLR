# Ensemble

Ensemble methods for combining multiple model outputs across pipeline stages.

## Overview

Provides ensemble functionality for all three pipeline tasks: anomaly detection (majority voting on outlier masks), imputation (averaging reconstructions), and classification (aggregating predictions). Results are logged to MLflow.

## Modules

| Module | Purpose |
|--------|---------|
| `tasks_ensembling.py` | High-level task orchestration across all ensemble types |
| `ensemble_anomaly_detection.py` | Combine outlier masks via majority voting, compute ensemble metrics |
| `ensemble_imputation.py` | Average imputation outputs from multiple models |
| `ensemble_classification.py` | Aggregate classification predictions with bootstrap CIs |
| `ensemble_logging.py` | MLflow logging utilities for ensemble results |
| `ensemble_utils.py` | Shared utilities: run retrieval, filtering, config helpers |

## See Also

- `src/anomaly_detection/` -- Individual outlier detection methods
- `src/imputation/` -- Individual imputation methods
- `src/classification/` -- Individual classification methods
