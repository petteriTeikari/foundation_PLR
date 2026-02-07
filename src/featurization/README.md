# Featurization (`src/featurization/`)

**Stage 3** of the Foundation PLR pipeline: Extract physiological features from imputed PLR signals.

## Overview

This module extracts **handcrafted features** from PLR signals:
- **Amplitude bins**: Histogram of pupil sizes during light stimulation
- **Latency features**: PIPR, MEDFA, MAX_CONSTRICTION, etc.

**Key finding**: Handcrafted features outperform FM embeddings by **9 percentage points** AUROC.

## Entry Point

```python
from src.featurization.flow_featurization import flow_featurization

# Run featurization
flow_featurization(cfg=cfg)
```

## Module Structure

```
featurization/
├── flow_featurization.py        # Prefect flow orchestration
├── featurize_PLR.py             # Main featurization logic
├── featurizer_PLR_subject.py    # Per-subject feature extraction
├── feature_utils.py             # Utility functions
├── feature_log.py               # MLflow logging
├── visualize_features.py        # Feature visualization
│
├── subflow_handcrafted_featurization.py  # Handcrafted features subflow
│
└── embedding/                   # FM embedding alternatives
    └── __init__.py
```

## Feature Types

### 1. Amplitude Bins

Histogram of pupil diameter during each light stimulus phase:

```
Light ON (Red/Blue)
        │
        ▼
┌─────────────────────┐
│ Pupil response      │
│  ▼ constriction     │
│    ▼                │
│      ↗ recovery     │
└─────────────────────┘
        │
        ▼
[bin1, bin2, bin3, ...binN]  # Histogram of amplitudes
```

### 2. Latency Features

| Feature | Description |
|---------|-------------|
| PIPR | Post-Illumination Pupillary Response |
| MEDFA | Median Frequency of Adaptation |
| MAX_CONSTRICTION | Maximum constriction amplitude |
| TIME_TO_MAX | Time to reach maximum constriction |

## Key Functions

### `featurize_subject`

Extract features for a single subject:

```python
def featurize_subject(
    subject_dict: dict,
    subject_code: str,
    cfg: DictConfig,
    feature_cfg: DictConfig,
    i: int,
    feature_col: str = "X",
) -> dict:
    """
    Extract features from a single subject's PLR signal.

    Parameters
    ----------
    subject_dict : dict
        Subject data including PLR signal and metadata.
    subject_code : str
        Subject identifier.
    cfg : DictConfig
        Main configuration.
    feature_cfg : DictConfig
        Feature extraction configuration.
    i : int
        Subject index.
    feature_col : str
        Column name for the signal to featurize.

    Returns
    -------
    dict
        Extracted features by color (Red, Blue).
    """
```

### `get_features_per_color`

Extract features for each light stimulus color:

```python
def get_features_per_color(
    df_subject: pl.DataFrame,
    light_timing: dict,
    bin_cfg: dict,
    color: str,
    feature_col: str,
) -> dict:
    """
    Extract features for a specific light color stimulus.
    """
```

## Configuration

Configure featurization in `configs/PLR_FEATURIZATION/`:

```yaml
# configs/PLR_FEATURIZATION/featuresSimple.yaml
FEATURES:
  amplitude_bins:
    n_bins: 10
    normalize: true
  latency:
    compute_pipr: true
    compute_medfa: true
```

## FM Embeddings (Not Recommended)

The `embedding/` subdirectory contains code for extracting FM embeddings as features. **However, these underperform handcrafted features by 9pp AUROC:**

| Approach | AUROC |
|----------|-------|
| Handcrafted (amplitude bins + latency) | 0.830 |
| FM Embeddings | 0.740 |

Use handcrafted features for classification.

## Output

Features are exported as a DataFrame:

```python
# Example feature output
{
    'subject_code': 'S001',
    'Red_bin1': 0.12,
    'Red_bin2': 0.23,
    ...
    'Blue_bin1': 0.15,
    ...
    'Red_PIPR': 0.45,
    'Blue_PIPR': 0.38,
    'class_label': 'glaucoma'
}
```

## See Also

- [ARCHITECTURE.md](../../ARCHITECTURE.md) - Pipeline overview
- [configs/PLR_FEATURIZATION/](../../configs/PLR_FEATURIZATION/) - Feature configurations
