# Metrics Module

Evaluation metrics for imputation and classification quality assessment.

## Overview

The metrics module provides utilities for computing evaluation metrics, particularly for imputation quality assessment.

## API Reference

::: src.metrics.evaluate_imputation_metrics
    options:
      show_source: true

::: src.metrics.metrics_utils
    options:
      show_source: true

## Key Functions

| Function | Description |
|----------|-------------|
| `evaluate_imputation_metrics` | Compute MAE, RMSE for imputation |
| `compute_reconstruction_error` | Signal reconstruction quality |

## Usage Example

```python
from src.metrics import evaluate_imputation_metrics

metrics = evaluate_imputation_metrics(
    original=ground_truth_signal,
    imputed=reconstructed_signal,
    mask=outlier_mask
)
print(f"MAE: {metrics['mae']:.4f}")
```
