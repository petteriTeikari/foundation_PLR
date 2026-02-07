# Preprocess

Signal preprocessing and normalization for PLR data.

## Overview

Handles data preprocessing steps before model input, including standardization (z-scoring), normalization state management, and basic data statistics. The `NormalizationManager` provides a stateful, serializable normalization system that prevents double-transforms and handles NaN values.

## Modules

| Module | Purpose |
|--------|---------|
| `normalization_manager.py` | Stateful normalization with fit/transform, NaN handling, serialization |
| `preprocess_data.py` | Top-level preprocessing dispatch (standardization, filtering) |
| `preprocess_PLR.py` | Standardization statistics retrieval for train/test splits |
| `preprocess_utils.py` | Basic statistics computation and logging per data split |

## See Also

- `src/data_io/` -- Data loading before preprocessing
- `src/anomaly_detection/` -- Outlier detection applied after preprocessing
