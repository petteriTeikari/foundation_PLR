# Classification (`src/classification/`)

**Stage 4** of the Foundation PLR pipeline: Train classifiers and evaluate with STRATOS-compliant metrics.

## Overview

This module handles:
- **Classifier training** (CatBoost primary, others for comparison)
- **Bootstrap evaluation** with 1000 iterations
- **STRATOS-compliant metrics** (AUROC, calibration, clinical utility)

**Research note**: We **fix the classifier** (CatBoost) and vary preprocessing. This is NOT about comparing classifiers.

## Entry Point

```python
from src.classification.flow_classification import flow_classification

# Run classification
flow_classification(cfg=cfg)
```

## Module Structure

```
classification/
├── flow_classification.py       # Prefect flow orchestration
├── train_classifier.py          # Main training logic
├── bootstrap_evaluation.py      # Bootstrap CI estimation
├── classifier_evaluation.py     # Evaluation pipeline
├── classifier_utils.py          # Utility functions
├── classifier_log_utils.py      # MLflow logging
├── cls_dataexport.py            # Data export
├── cls_model_utils.py           # Model utilities
├── classification_checks.py     # Validation checks
├── feature_selection.py         # Feature selection
├── hyperopt_utils.py            # Hyperparameter optimization
├── sklearn_simple_classifiers.py # sklearn classifiers
├── stats_metric_utils.py        # Metric utilities
├── weighing_utils.py            # Class weighing
├── viz_classifiers.py           # Visualization
│
├── subflow_feature_classification.py  # Feature-based classification
├── subflow_ts_classification.py       # Time series classification
│
├── tabpfn_utils.py              # TabPFN utilities
├── tabpfn_main.py               # TabPFN entry point
│
├── classification_ts/           # Time series classifiers
│
├── xgboost_cls/                 # XGBoost wrapper
│
├── tabpfn/                      # [VENDORED] TabPFN v2
│
└── tabpfn_v1/                   # [VENDORED] TabPFN v1
```

## Available Classifiers

| Classifier | Type | Primary? |
|------------|------|----------|
| `CatBoost` | Gradient Boosting | **YES** |
| `XGBoost` | Gradient Boosting | Comparison |
| `TabPFN` | Foundation Model | Comparison |
| `TabM` | Transformer | Comparison |
| `LogisticRegression` | Linear | Comparison |

**CatBoost** is the primary classifier because it achieves the best performance (AUROC 0.913 with ground truth preprocessing).

## Key Functions

### `train_classifier`

Train a classifier on features:

```python
def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: DictConfig,
    classifier_name: str,
) -> tuple:
    """
    Train and evaluate a classifier.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data.
    X_test, y_test : np.ndarray
        Test data.
    cfg : DictConfig
        Configuration.
    classifier_name : str
        Name of classifier to train.

    Returns
    -------
    tuple
        (model, predictions, metrics)
    """
```

### `bootstrap_evaluation`

Compute bootstrap confidence intervals:

```python
def bootstrap_evaluation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_iterations: int = 1000,
    alpha: float = 0.95,
) -> dict:
    """
    Bootstrap evaluation for CI estimation.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_prob : np.ndarray
        Predicted probabilities.
    n_iterations : int
        Number of bootstrap iterations.
    alpha : float
        Confidence level.

    Returns
    -------
    dict
        Metrics with confidence intervals.
    """
```

## Configuration

Configure classifiers in `configs/CLS_MODELS/`:

```yaml
# configs/CLS_MODELS/CatBoost.yaml
name: CatBoost
MODEL:
  iterations: 1000
  learning_rate: 0.03
  depth: 6
  eval_metric: 'AUC'
```

## STRATOS-Compliant Metrics

| Category | Metrics |
|----------|---------|
| **Discrimination** | AUROC with 95% CI |
| **Calibration** | Slope, intercept, O:E ratio |
| **Overall** | Brier score, Scaled Brier (IPA) |
| **Clinical Utility** | Net Benefit, DCA |

All metrics are computed via `src/stats/`.

## Bootstrap Workflow

```
For each bootstrap iteration (n=1000):
    1. Sample with replacement
    2. Compute all STRATOS metrics
    3. Store results

Final:
    - Point estimate: mean
    - CI: percentile bootstrap [2.5%, 97.5%]
```

## Vendored Code

| Directory | Source | Used For |
|-----------|--------|----------|
| `tabpfn/` | TabPFN | TabPFN v2 |
| `tabpfn_v1/` | TabPFN | TabPFN v1 |

These are **excluded from documentation**.

## Results

With ground truth preprocessing + CatBoost:
- **AUROC**: 0.913 (95% CI: 0.851-0.955)
- **Brier**: 0.131
- **Calibration slope**: ~1.0

## See Also

- [ARCHITECTURE.md](../../ARCHITECTURE.md) - Pipeline overview
- [src/stats/README.md](../stats/README.md) - STRATOS metrics implementation
- [configs/CLS_MODELS/](../../configs/CLS_MODELS/) - Classifier configurations
