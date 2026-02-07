"""
Input validation utilities for statistics module.

Provides consistent validation across all statistical computations.

Cross-references:
- planning/statistics-implementation.md (Section 1.2)
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from ._exceptions import (
    DegenerateCaseError,
    InsufficientDataError,
    SingleClassError,
    ValidationError,
)


def validate_array(
    data: Union[np.ndarray, list],
    name: str = "data",
    allow_nan: bool = False,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """
    Validate and convert input to numpy array.

    Parameters
    ----------
    data : array-like
        Input data to validate
    name : str
        Name of the parameter (for error messages)
    allow_nan : bool
        Whether to allow NaN values
    dtype : np.dtype, optional
        Required dtype (will convert if not matching)

    Returns
    -------
    np.ndarray
        Validated numpy array

    Raises
    ------
    ValidationError
        If data is empty or contains invalid values
    """
    if data is None:
        raise ValidationError(name, "array-like", "None")

    arr = np.asarray(data, dtype=dtype)

    if arr.size == 0:
        raise ValidationError(name, "non-empty array", "empty array")

    if not allow_nan and np.any(np.isnan(arr)):
        n_nan = np.sum(np.isnan(arr))
        raise ValidationError(
            name, "array without NaN values", f"array with {n_nan} NaN values"
        )

    return arr


def validate_min_samples(
    data: np.ndarray,
    min_n: int,
    name: str = "data",
    context: str = "",
) -> None:
    """
    Validate minimum sample size requirement.

    Parameters
    ----------
    data : np.ndarray
        Input data
    min_n : int
        Minimum required sample size
    name : str
        Name of the parameter
    context : str
        Additional context for error message

    Raises
    ------
    InsufficientDataError
        If sample size is less than min_n
    """
    n = len(data)
    if n < min_n:
        raise InsufficientDataError(
            required=min_n, actual=n, context=f"{name}: {context}" if context else name
        )


def validate_binary_outcomes(
    y: Union[np.ndarray, list],
    name: str = "y_true",
) -> np.ndarray:
    """
    Validate binary outcome array.

    Parameters
    ----------
    y : array-like
        Binary outcomes (0 or 1)
    name : str
        Name of the parameter

    Returns
    -------
    np.ndarray
        Validated binary array (dtype int)

    Raises
    ------
    ValidationError
        If values are not binary
    SingleClassError
        If only one class is present
    """
    arr = validate_array(y, name=name)

    # Check binary
    unique_vals = np.unique(arr)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValidationError(name, "binary values (0 or 1)", f"values {unique_vals}")

    # Check both classes present
    class_counts = {0: np.sum(arr == 0), 1: np.sum(arr == 1)}
    if class_counts[0] == 0 or class_counts[1] == 0:
        raise SingleClassError(class_counts)

    return arr.astype(int)


def validate_probabilities(
    probs: Union[np.ndarray, list],
    name: str = "y_prob",
    clip: bool = True,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Validate probability array.

    Parameters
    ----------
    probs : array-like
        Predicted probabilities
    name : str
        Name of the parameter
    clip : bool
        Whether to clip values to [eps, 1-eps]
    eps : float
        Epsilon for clipping

    Returns
    -------
    np.ndarray
        Validated probability array

    Raises
    ------
    ValidationError
        If values are outside [0, 1]
    """
    arr = validate_array(probs, name=name, dtype=np.float64)

    if np.any(arr < 0) or np.any(arr > 1):
        min_val, max_val = np.min(arr), np.max(arr)
        raise ValidationError(
            name, "values in [0, 1]", f"values in [{min_val:.4f}, {max_val:.4f}]"
        )

    if clip:
        arr = np.clip(arr, eps, 1 - eps)

    return arr


def validate_positive(
    value: float,
    name: str,
    strict: bool = True,
) -> float:
    """
    Validate that a value is positive.

    Parameters
    ----------
    value : float
        Value to check
    name : str
        Name of the parameter
    strict : bool
        If True, require > 0; if False, allow >= 0

    Returns
    -------
    float
        The validated value

    Raises
    ------
    ValidationError
        If value is not positive
    """
    if strict:
        if value <= 0:
            raise ValidationError(name, "> 0", str(value))
    else:
        if value < 0:
            raise ValidationError(name, ">= 0", str(value))
    return value


def validate_in_range(
    value: float,
    name: str,
    min_val: float,
    max_val: float,
    inclusive: str = "both",
) -> float:
    """
    Validate that a value is within a specified range.

    Parameters
    ----------
    value : float
        Value to check
    name : str
        Name of the parameter
    min_val : float
        Minimum allowed value
    max_val : float
        Maximum allowed value
    inclusive : str
        'both', 'neither', 'left', or 'right'

    Returns
    -------
    float
        The validated value

    Raises
    ------
    ValidationError
        If value is outside the range
    """
    if inclusive == "both":
        valid = min_val <= value <= max_val
        expected = f"in [{min_val}, {max_val}]"
    elif inclusive == "neither":
        valid = min_val < value < max_val
        expected = f"in ({min_val}, {max_val})"
    elif inclusive == "left":
        valid = min_val <= value < max_val
        expected = f"in [{min_val}, {max_val})"
    elif inclusive == "right":
        valid = min_val < value <= max_val
        expected = f"in ({min_val}, {max_val}]"
    else:
        raise ValueError(f"Invalid inclusive parameter: {inclusive}")

    if not valid:
        raise ValidationError(name, expected, str(value))

    return value


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    name: str = "df",
) -> pd.DataFrame:
    """
    Validate DataFrame has required columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : list
        List of required column names
    name : str
        Name of the parameter

    Returns
    -------
    pd.DataFrame
        The validated DataFrame

    Raises
    ------
    ValidationError
        If DataFrame is missing required columns
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(name, "pandas DataFrame", type(df).__name__)

    if len(df) == 0:
        raise ValidationError(name, "non-empty DataFrame", "empty DataFrame")

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValidationError(
            name, f"columns {required_columns}", f"missing columns {list(missing)}"
        )

    return df


def validate_factors(
    df: pd.DataFrame,
    factors: List[str],
) -> None:
    """
    Validate factor columns for ANOVA.

    Checks that each factor has at least 2 levels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with factor columns
    factors : list
        Factor column names

    Raises
    ------
    ValidationError
        If any factor has fewer than 2 levels
    """
    for factor in factors:
        if factor not in df.columns:
            raise ValidationError("factors", f"column '{factor}' to exist", "missing")

        n_levels = df[factor].nunique()
        if n_levels < 2:
            raise ValidationError(factor, "at least 2 levels", f"{n_levels} level(s)")


def check_variance(
    data: np.ndarray,
    name: str = "data",
    min_variance: float = 1e-10,
) -> None:
    """
    Check that data has non-zero variance.

    Parameters
    ----------
    data : np.ndarray
        Data to check
    name : str
        Name of the parameter
    min_variance : float
        Minimum acceptable variance

    Raises
    ------
    DegenerateCaseError
        If variance is below threshold
    """
    var = np.var(data, ddof=1)
    if var < min_variance:
        raise DegenerateCaseError(
            name, f"variance is {var:.2e}, below minimum {min_variance:.2e}"
        )
