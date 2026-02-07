"""
Display Names Lookup - Maps MLflow names to publication-friendly labels.

This module provides the mapping from raw MLflow method names to clean
display names suitable for publication figures.

Usage:
    from src.data_io.display_names import (
        get_outlier_display_name,
        get_imputation_display_name,
        get_classifier_display_name,
    )

    # Get display name for a method
    display = get_outlier_display_name("MOMENT-gt-finetune")
    # Returns: "MOMENT Fine-tuned"

See: configs/mlflow_registry/display_names.yaml for all mappings.
See: docs/planning/lookup-model-names.md for design decisions.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:
    from loguru import logger
except ImportError:
    # Fallback for environments without loguru
    import logging

    logger = logging.getLogger(__name__)

# Public API
__all__ = [
    "get_outlier_display_name",
    "get_imputation_display_name",
    "get_classifier_display_name",
    "get_category_display_name",
    "get_all_display_names",
    "DISPLAY_NAMES_PATH",
]

# Path to the display names YAML (relative to project root)
DISPLAY_NAMES_PATH = (
    Path(__file__).parent.parent.parent
    / "configs"
    / "mlflow_registry"
    / "display_names.yaml"
)


@lru_cache(maxsize=1)
def _load_display_names() -> dict[str, Any]:
    """
    Load and cache display names from YAML.

    Returns
    -------
    dict
        The parsed display names configuration.

    Raises
    ------
    FileNotFoundError
        If display_names.yaml is missing.
    yaml.YAMLError
        If YAML parsing fails.
    """
    if not DISPLAY_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Display names configuration not found: {DISPLAY_NAMES_PATH}. "
            "Run: pytest tests/unit/test_display_names.py -v to check setup."
        )

    with open(DISPLAY_NAMES_PATH) as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Display names YAML is empty: {DISPLAY_NAMES_PATH}")

    return data


def get_outlier_display_name(method: str) -> str:
    """
    Get display name for an outlier detection method.

    Parameters
    ----------
    method : str
        Raw MLflow outlier method name (e.g., "MOMENT-gt-finetune").

    Returns
    -------
    str
        Publication-friendly display name (e.g., "MOMENT Fine-tuned").
        If method not found, returns raw name and logs WARNING.

    Examples
    --------
    >>> get_outlier_display_name("pupil-gt")
    'Ground Truth'
    >>> get_outlier_display_name("MOMENT-gt-finetune")
    'MOMENT Fine-tuned'
    """
    names = _load_display_names()
    display = names.get("outlier_methods", {}).get(method)

    if display is None:
        logger.warning(f"No display name for outlier method: {method}")
        return method

    return display


def get_imputation_display_name(method: str) -> str:
    """
    Get display name for an imputation method.

    Parameters
    ----------
    method : str
        Raw MLflow imputation method name (e.g., "MOMENT-finetune").

    Returns
    -------
    str
        Publication-friendly display name (e.g., "MOMENT Fine-tuned").
        If method not found, returns raw name and logs WARNING.

    Examples
    --------
    >>> get_imputation_display_name("pupil-gt")
    'Ground Truth'
    >>> get_imputation_display_name("SAITS")
    'SAITS'
    """
    names = _load_display_names()
    display = names.get("imputation_methods", {}).get(method)

    if display is None:
        logger.warning(f"No display name for imputation method: {method}")
        return method

    return display


def get_classifier_display_name(classifier: str) -> str:
    """
    Get display name for a classifier.

    Parameters
    ----------
    classifier : str
        Raw classifier name (e.g., "LogisticRegression").

    Returns
    -------
    str
        Publication-friendly display name (e.g., "Logistic Regression").
        If classifier not found, returns raw name and logs WARNING.

    Examples
    --------
    >>> get_classifier_display_name("CatBoost")
    'CatBoost'
    >>> get_classifier_display_name("LogisticRegression")
    'Logistic Regression'
    """
    names = _load_display_names()
    display = names.get("classifiers", {}).get(classifier)

    if display is None:
        logger.warning(f"No display name for classifier: {classifier}")
        return classifier

    return display


def get_category_display_name(category: str) -> str:
    """
    Get display name for a method category.

    Parameters
    ----------
    category : str
        Internal category name (e.g., "foundation_model").

    Returns
    -------
    str
        Publication-friendly display name (e.g., "Foundation Model").
        If category not found, returns raw name and logs WARNING.

    Examples
    --------
    >>> get_category_display_name("ground_truth")
    'Ground Truth'
    >>> get_category_display_name("foundation_model")
    'Foundation Model'
    """
    names = _load_display_names()
    display = names.get("categories", {}).get(category)

    if display is None:
        logger.warning(f"No display name for category: {category}")
        return category

    return display


def get_all_display_names() -> dict[str, str]:
    """
    Get all display name mappings combined into a single dict.

    This is useful for validation and bulk operations.

    Returns
    -------
    dict[str, str]
        Mapping from raw names to display names.
        Includes outlier_methods, imputation_methods, and classifiers.

    Notes
    -----
    If a raw name appears in multiple sections (e.g., "pupil-gt" in both
    outlier and imputation), the later one wins. This is intentional since
    they should have the same display name anyway.

    Examples
    --------
    >>> names = get_all_display_names()
    >>> len(names) >= 24  # At least 11 + 8 + 5
    True
    >>> names["pupil-gt"]
    'Ground Truth'
    """
    data = _load_display_names()
    all_names: dict[str, str] = {}

    # Combine all sections
    if "outlier_methods" in data:
        all_names.update(data["outlier_methods"])
    if "imputation_methods" in data:
        all_names.update(data["imputation_methods"])
    if "classifiers" in data:
        all_names.update(data["classifiers"])

    return all_names
