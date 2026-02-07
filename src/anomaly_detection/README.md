# Anomaly Detection (`src/anomaly_detection/`)

**Stage 1** of the Foundation PLR pipeline: Detect outliers (blinks, artifacts) in raw PLR signals.

## Overview

This module implements **11 outlier detection methods** including:
- Traditional methods (LOF, OneClassSVM, SubPCA)
- Foundation models (MOMENT, UniTS, TimesNet)
- Ensemble methods

**Research question**: Can foundation models detect PLR artifacts as well as humans?

## Entry Point

```python
from src.anomaly_detection.flow_anomaly_detection import flow_anomaly_detection

# Run outlier detection
flow_anomaly_detection(cfg=cfg, df=df)
```

## Available Methods

| Method | Type | File |
|--------|------|------|
| `pupil-gt` | Ground truth | (baseline) |
| `LOF` | Traditional | `outlier_sklearn.py` |
| `OneClassSVM` | Traditional | `outlier_sklearn.py` |
| `SubPCA` | Traditional | `outlier_sklearn.py` |
| `PROPHET` | Prophet-based | `outlier_prophet.py` |
| `MOMENT-gt-finetune` | Foundation Model | `momentfm_outlier/` |
| `MOMENT-gt-zeroshot` | Foundation Model | `momentfm_outlier/` |
| `MOMENT-orig-finetune` | Foundation Model | `momentfm_outlier/` |
| `UniTS-gt-finetune` | Foundation Model | `units/` |
| `UniTS-orig-finetune` | Foundation Model | `units/` |
| `UniTS-orig-zeroshot` | Foundation Model | `units/` |
| `TimesNet-gt` | Deep Learning | `timesnet_wrapper.py` |
| `TimesNet-orig` | Deep Learning | `timesnet_wrapper.py` |
| `ensemble-*` | Ensemble | (combined) |

## Module Structure

```
anomaly_detection/
├── flow_anomaly_detection.py    # Prefect flow orchestration
├── anomaly_detection.py         # Core detection logic + method dispatch
├── anomaly_utils.py             # Utility functions
├── log_anomaly_detection.py     # MLflow logging
├── anomaly_detection_metrics_wrapper.py  # Evaluation metrics
│
├── outlier_sklearn.py           # LOF, SVM, SubPCA (sklearn-based)
├── outlier_prophet.py           # Prophet-based detection
├── outlier_tsb_ad.py            # TSB-AD benchmark wrapper
├── outlier_sigllm.py            # SigLLM wrapper
├── timesnet_wrapper.py          # TimesNet integration
│
├── momentfm_outlier/            # MOMENT foundation model
│   ├── __init__.py
│   └── momentfm_outlier.py
│
├── units/                       # UniTS foundation model
│   └── units_outlier.py
│
└── extra_eval/                  # [VENDORED] TSB-AD evaluation
    └── TSB_AD/
```

## Key Functions

### `outlier_detection_selector`

Main dispatch function selecting which method to use:

```python
def outlier_detection_selector(
    df: pl.DataFrame,
    cfg: DictConfig,
    experiment_name: str,
    run_name: str,
    model_name: str,
):
    """
    Select and run outlier detection method.

    Parameters
    ----------
    df : pl.DataFrame
        Input PLR data.
    cfg : DictConfig
        Hydra configuration.
    experiment_name : str
        MLflow experiment name.
    run_name : str
        MLflow run name.
    model_name : str
        Name of outlier detection method.
    """
```

### sklearn Methods

```python
from src.anomaly_detection.outlier_sklearn import outlier_sklearn_wrapper

# LOF, OneClassSVM, SubPCA
result = outlier_sklearn_wrapper(df, cfg, method='LOF')
```

### MOMENT Methods

```python
from src.anomaly_detection.momentfm_outlier.momentfm_outlier import momentfm_outlier_main

# MOMENT foundation model
result = momentfm_outlier_main(df, cfg)
```

## Configuration

Configure outlier detection in `configs/OUTLIER_MODELS/`:

```yaml
# configs/OUTLIER_MODELS/MOMENT.yaml
name: MOMENT
MODEL:
  train_on: 'pupil_gt'  # or 'pupil_orig_imputed'
  mode: 'finetune'      # or 'zeroshot'
  epochs: 50
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Outlier F1 | F1 score vs ground truth mask |
| Outlier Precision | Precision vs ground truth |
| Outlier Recall | Recall vs ground truth |

## Error Propagation

**Errors at this stage propagate downstream:**

```
Missed artifacts → Incorrect imputation → Corrupted features → Wrong classification
```

This is why we compare methods against human-annotated ground truth.

## Vendored Code

The `extra_eval/TSB_AD/` directory contains vendored code from the TSB-AD benchmark and is **excluded from documentation**.

## See Also

- [ARCHITECTURE.md](../../ARCHITECTURE.md) - Pipeline overview
- [configs/OUTLIER_MODELS/](../../configs/OUTLIER_MODELS/) - Method configurations
