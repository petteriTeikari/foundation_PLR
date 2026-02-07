# Adding New Methods

This tutorial explains how to extend Foundation PLR with new preprocessing methods.

## Adding a New Outlier Detection Method

### Step 1: Create the Method

```python
# src/anomaly_detection/my_outlier_method.py

import numpy as np
from typing import Tuple

def detect_outliers_my_method(
    signal: np.ndarray,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers using my custom method.

    Parameters
    ----------
    signal : np.ndarray
        Input PLR signal of shape (n_timepoints,).
    threshold : float
        Detection threshold.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - outlier_mask: Boolean mask where True = outlier
        - confidence: Confidence scores per timepoint
    """
    # Your implementation here
    outlier_mask = np.zeros(len(signal), dtype=bool)
    confidence = np.zeros(len(signal))

    # Example: simple threshold-based detection
    median = np.median(signal)
    mad = np.median(np.abs(signal - median))
    z_scores = np.abs(signal - median) / (mad + 1e-8)
    outlier_mask = z_scores > threshold

    return outlier_mask, z_scores
```

### Step 2: Register the Method

Add to `src/anomaly_detection/anomaly_detection.py`:

```python
from src.anomaly_detection.my_outlier_method import detect_outliers_my_method

OUTLIER_METHODS = {
    # ... existing methods ...
    "my_method": detect_outliers_my_method,
}
```

### Step 3: Add Configuration

```yaml
# configs/outlier_methods/my_method.yaml
outlier_method: my_method
outlier_params:
  threshold: 0.5
```

### Step 4: Test

```bash
python -m src.classification.flow_classification \
    outlier_method=my_method
```

## Adding a New Imputation Method

### Step 1: Create the Method

```python
# src/imputation/my_imputation_method.py

import numpy as np

def impute_my_method(
    signal: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Impute missing values using my custom method.

    Parameters
    ----------
    signal : np.ndarray
        Input signal with missing values.
    mask : np.ndarray
        Boolean mask where True = missing/outlier.

    Returns
    -------
    np.ndarray
        Imputed signal.
    """
    imputed = signal.copy()
    # Your implementation here
    return imputed
```

### Step 2: Register and Configure

Follow the same pattern as outlier detection.

## Best Practices

1. **Follow NumPy docstring style** for automatic API docs
2. **Add type hints** for better IDE support
3. **Write tests** in `tests/` directory
4. **Update documentation** in `docs/user-guide/`
5. **Register in YAML** - Add to `configs/mlflow_registry/parameters/classification.yaml`

!!! important "Registry is Single Source of Truth"
    After adding a new method, register it in `configs/mlflow_registry/parameters/classification.yaml` to ensure it's counted correctly. Currently: 11 outlier methods, 8 imputation methods.

## See Also

- [API Reference](../api-reference/index.md)
- [Pipeline Overview](../user-guide/pipeline-overview.md)
