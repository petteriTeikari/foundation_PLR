# Imputation (`src/imputation/`)

**Stage 2** of the Foundation PLR pipeline: Reconstruct missing segments after outlier removal.

## Overview

This module implements **7 imputation methods**:
- Traditional (linear interpolation)
- Deep learning (SAITS, CSDI, TimesNet)
- Foundation models (MOMENT-finetune, MOMENT-zeroshot)
- Ensemble methods

**Research question**: Can foundation models reconstruct PLR signals better than traditional methods?

## Entry Point

```python
from src.imputation.flow_imputation import flow_imputation

# Run imputation
flow_imputation(cfg=cfg)
```

## Available Methods

| Method | Type | File |
|--------|------|------|
| `pupil-gt` | Ground truth | (baseline) |
| `linear` | Traditional | (built-in) |
| `SAITS` | Deep Learning | `pypots/` |
| `CSDI` | Deep Learning | `pypots/` |
| `TimesNet` | Deep Learning | (built-in) |
| `MOMENT-finetune` | Foundation Model | `momentfm/` |
| `MOMENT-zeroshot` | Foundation Model | `momentfm/` |
| `ensemble-*` | Ensemble | (combined) |

## Module Structure

```
imputation/
├── flow_imputation.py           # Prefect flow orchestration
├── imputation_main.py           # Core imputation logic
├── impute_with_models.py        # Model dispatch
├── imputation_utils.py          # Utility functions
├── imputation_log_artifacts.py  # MLflow logging
├── eval_utils.py                # Evaluation utilities
├── train_utils.py               # Training utilities
├── train_torch_utils.py         # PyTorch training
├── missforest_main.py           # MissForest implementation
│
├── momentfm/                    # MOMENT foundation model
│   ├── __init__.py
│   └── moment_imputation_main.py
│
├── pypots/                      # [VENDORED] PyPOTS library
│   └── pypots_wrapper.py        # Wrapper for SAITS, CSDI
│
└── nuwats/                      # [VENDORED] NuwaTS model
    ├── __init__.py
    └── nuwats_main.py
```

## Key Functions

### `imputation_selector`

Main dispatch function:

```python
def imputation_selector(
    cfg: DictConfig,
    source_data: dict,
    model_name: str,
    run_name: str,
) -> dict:
    """
    Select and run imputation method.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration.
    source_data : dict
        Data from outlier detection stage.
    model_name : str
        Name of imputation method.
    run_name : str
        MLflow run name.

    Returns
    -------
    dict
        Imputed data and metrics.
    """
```

### MOMENT Imputation

```python
from src.imputation.momentfm.moment_imputation_main import moment_imputation_main

result = moment_imputation_main(data, cfg)
```

### PyPOTS Methods (SAITS, CSDI)

```python
from src.imputation.pypots.pypots_wrapper import pypots_wrapper

result = pypots_wrapper(data, cfg, method='SAITS')
```

## Configuration

Configure imputation in `configs/MODELS/`:

```yaml
# configs/MODELS/SAITS.yaml
name: SAITS
MODEL:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error vs ground truth |
| RMSE | Root Mean Square Error vs ground truth |
| MAPE | Mean Absolute Percentage Error |

## Error Propagation

**Errors at this stage propagate downstream:**

```
Imputation errors → Feature distortion → Classification degradation
```

## Vendored Code

The following directories contain vendored third-party code:

| Directory | Source | Used For |
|-----------|--------|----------|
| `pypots/` | PyPOTS | SAITS, CSDI |
| `nuwats/` | NuwaTS | NuwaTS model |

These are **excluded from documentation**.

## See Also

- [ARCHITECTURE.md](../../ARCHITECTURE.md) - Pipeline overview
- [configs/MODELS/](../../configs/MODELS/) - Method configurations
