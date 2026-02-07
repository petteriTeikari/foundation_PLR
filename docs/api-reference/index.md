# API Reference

Auto-generated documentation from source code docstrings.

!!! info "Automatic Generation"
    This documentation is automatically extracted from NumPy-style docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

## Modules

### Core Pipeline

| Module | Description |
|--------|-------------|
| [anomaly_detection](anomaly_detection.md) | Outlier detection methods (11 methods) |
| [imputation](imputation.md) | Signal reconstruction (8 methods) |
| [featurization](featurization.md) | Feature extraction |
| [classification](classification.md) | Model training and evaluation |
| [decomposition](decomposition.md) | PLR waveform decomposition (5 methods) |

### Support Modules

| Module | Description |
|--------|-------------|
| [data_io](data_io.md) | Data loading and preprocessing |
| [ensemble](ensemble.md) | Ensemble methods |
| [log_helpers](log_helpers.md) | Logging and MLflow utilities |
| [stats](stats.md) | Statistical analysis and metrics |
| [metrics](metrics.md) | Evaluation metrics |
| [orchestration](orchestration.md) | Pipeline orchestration |
| [preprocess](preprocess.md) | Data preprocessing |
| [summarization](summarization.md) | Results summarization |

### Visualization

| Module | Description |
|--------|-------------|
| [viz](viz.md) | Python visualization (calibration, DCA, CD diagrams) |

## Usage

Each module page shows:

- **Functions**: With parameters, return types, and examples
- **Classes**: With attributes and methods
- **Source code**: Links to the implementation

## Docstring Style

All docstrings follow NumPy style:

```python
def example_function(param1: int, param2: str) -> dict:
    """
    Short description of the function.

    Longer description with more details about what the function does.

    Parameters
    ----------
    param1 : int
        Description of first parameter.
    param2 : str
        Description of second parameter.

    Returns
    -------
    dict
        Description of return value.

    Examples
    --------
    >>> example_function(1, "test")
    {"result": "success"}
    """
```
