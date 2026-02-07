"""
MLflow Registry - Single Source of Truth for Experiment Parameters.

This module provides validated access to the experiment registry defined in:
    configs/mlflow_registry/parameters/classification.yaml

ALL extraction and analysis code MUST use this module to get valid method names.
NEVER parse MLflow run names or hardcode method lists.

Usage:
    from src.data_io.registry import (
        get_valid_outlier_methods,
        get_valid_imputation_methods,
        get_valid_classifiers,
        validate_outlier_method,
        validate_imputation_method,
    )

    # Get all valid methods
    outliers = get_valid_outlier_methods()  # Returns exactly 11
    imputations = get_valid_imputation_methods()  # Returns exactly 8

    # Validate a method name
    if not validate_outlier_method("anomaly"):
        raise ValueError("Invalid outlier method!")

See: configs/mlflow_registry/README.md for documentation.
See: CLAUDE.md for the rule that this MUST be used.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

try:
    from loguru import logger
except ImportError:
    # Fallback for environments without loguru
    import logging

    logger = logging.getLogger(__name__)

# Public API - explicit exports
__all__ = [
    # Exception
    "RegistryError",
    # Getters
    "get_valid_outlier_methods",
    "get_valid_imputation_methods",
    "get_valid_classifiers",
    "get_outlier_categories",
    "get_imputation_categories",
    # Validators (O(1) using cached frozensets)
    "validate_outlier_method",
    "validate_imputation_method",
    "validate_classifier",
    # Utilities
    "get_expected_factorial_size",
    "print_registry_summary",
    # Constants
    "EXPECTED_OUTLIER_COUNT",
    "EXPECTED_IMPUTATION_COUNT",
    "EXPECTED_CLASSIFIER_COUNT",
    "REGISTRY_PATH",
    "CLASSIFICATION_PARAMS",
]

# Path to the registry (relative to project root)
REGISTRY_PATH = Path(__file__).parent.parent.parent / "configs" / "mlflow_registry"
CLASSIFICATION_PARAMS = REGISTRY_PATH / "parameters" / "classification.yaml"

# ============================================================================
# HARDCODED EXPECTED COUNTS - FINAL VALUES FOR PUBLICATION
# ============================================================================
# WARNING: These counts are HARDCODED because no more experiments will be run.
# If you need to expand the experimental design, you MUST:
#   1. Update configs/mlflow_registry/parameters/classification.yaml
#   2. Update configs/registry_canary.yaml
#   3. Update these constants below
#   4. Update all tests in tests/test_registry.py
#   5. Update .claude/rules/05-registry-source-of-truth.md
#   6. Update CLAUDE.md (both root and .claude/)
#
# The expected counts are validated at runtime. If the registry YAML defines
# different counts, this module will raise RegistryError.
# ============================================================================
EXPECTED_OUTLIER_COUNT = 11
EXPECTED_IMPUTATION_COUNT = 8
EXPECTED_CLASSIFIER_COUNT = 5

# Note: Removed logger.warning at import time - it was firing on every import
# (tests, IDE autocomplete, linting) and polluting logs with expected behavior.
# The counts are documented above and validated at runtime.


class RegistryError(Exception):
    """Raised when registry validation fails."""

    pass


@lru_cache(maxsize=1)
def _load_classification_params() -> dict[str, Any]:
    """Load and cache the classification parameters registry."""
    if not CLASSIFICATION_PARAMS.exists():
        raise RegistryError(f"Registry not found: {CLASSIFICATION_PARAMS}")

    with open(CLASSIFICATION_PARAMS) as f:
        params = yaml.safe_load(f)

    return params


def get_valid_outlier_methods() -> list[str]:
    """
    Get the list of valid outlier detection methods.

    Returns exactly 11 methods as defined in the registry.
    If this returns a different count, the registry is corrupted.

    Returns
    -------
    list[str]
        List of valid outlier method names.

    Raises
    ------
    RegistryError
        If registry is missing or count doesn't match expected.
    """
    params = _load_classification_params()
    methods = params["pipeline"]["anomaly_source"]["values"]

    if len(methods) != EXPECTED_OUTLIER_COUNT:
        raise RegistryError(
            f"Registry corrupted: expected {EXPECTED_OUTLIER_COUNT} outlier methods, "
            f"got {len(methods)}. Check configs/mlflow_registry/parameters/classification.yaml"
        )

    return methods


def get_valid_imputation_methods() -> list[str]:
    """
    Get the list of valid imputation methods.

    Returns exactly 8 methods as defined in the registry.

    Returns
    -------
    list[str]
        List of valid imputation method names.

    Raises
    ------
    RegistryError
        If registry is missing or count doesn't match expected.
    """
    params = _load_classification_params()
    methods = params["pipeline"]["imputation_source"]["values"]

    if len(methods) != EXPECTED_IMPUTATION_COUNT:
        raise RegistryError(
            f"Registry corrupted: expected {EXPECTED_IMPUTATION_COUNT} imputation methods, "
            f"got {len(methods)}. Check configs/mlflow_registry/parameters/classification.yaml"
        )

    return methods


def get_valid_classifiers() -> list[str]:
    """
    Get the list of valid classifiers.

    Returns exactly 5 classifiers as defined in the registry.

    Returns
    -------
    list[str]
        List of valid classifier names.

    Raises
    ------
    RegistryError
        If registry is missing or count doesn't match expected.
    """
    params = _load_classification_params()
    classifiers = params["pipeline"]["model_name"]["values"]

    if len(classifiers) != EXPECTED_CLASSIFIER_COUNT:
        raise RegistryError(
            f"Registry corrupted: expected {EXPECTED_CLASSIFIER_COUNT} classifiers, "
            f"got {len(classifiers)}. Check configs/mlflow_registry/parameters/classification.yaml"
        )

    return classifiers


def get_outlier_categories() -> dict[str, list[str]]:
    """
    Get outlier methods grouped by category.

    Returns
    -------
    dict[str, list[str]]
        Mapping from category name to list of methods.
        Categories: ground_truth, foundation_model, deep_learning, traditional, ensemble
    """
    params = _load_classification_params()
    return params["pipeline"]["anomaly_source"]["categories"]


def get_imputation_categories() -> dict[str, list[str]]:
    """
    Get imputation methods grouped by category.

    Returns
    -------
    dict[str, list[str]]
        Mapping from category name to list of methods.
        Categories: ground_truth, foundation_model, deep_learning, ensemble
    """
    params = _load_classification_params()
    return params["pipeline"]["imputation_source"]["categories"]


# =============================================================================
# O(1) Validation with Cached Frozensets
# =============================================================================
# These internal functions cache frozensets for O(1) membership testing.
# The public validate_* functions use these for efficient validation.


@lru_cache(maxsize=1)
def _get_outlier_method_set() -> frozenset[str]:
    """Return frozenset of valid outlier methods for O(1) lookup."""
    return frozenset(get_valid_outlier_methods())


@lru_cache(maxsize=1)
def _get_imputation_method_set() -> frozenset[str]:
    """Return frozenset of valid imputation methods for O(1) lookup."""
    return frozenset(get_valid_imputation_methods())


@lru_cache(maxsize=1)
def _get_classifier_set() -> frozenset[str]:
    """Return frozenset of valid classifiers for O(1) lookup."""
    return frozenset(get_valid_classifiers())


def validate_outlier_method(method: str) -> bool:
    """
    Check if an outlier method is valid according to the registry.

    Uses O(1) frozenset lookup (cached) instead of O(n) list search.

    Parameters
    ----------
    method : str
        The outlier method name to validate.

    Returns
    -------
    bool
        True if valid, False otherwise.

    Examples
    --------
    >>> validate_outlier_method("pupil-gt")
    True
    >>> validate_outlier_method("anomaly")  # Invalid!
    False
    >>> validate_outlier_method("MOMENT-orig-finetune")  # Not in registry!
    False
    """
    return method in _get_outlier_method_set()


def validate_imputation_method(method: str) -> bool:
    """
    Check if an imputation method is valid according to the registry.

    Uses O(1) frozenset lookup (cached) instead of O(n) list search.

    Parameters
    ----------
    method : str
        The imputation method name to validate.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    return method in _get_imputation_method_set()


def validate_classifier(classifier: str) -> bool:
    """
    Check if a classifier is valid according to the registry.

    Uses O(1) frozenset lookup (cached) instead of O(n) list search.

    Parameters
    ----------
    classifier : str
        The classifier name to validate.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    return classifier in _get_classifier_set()


def get_expected_factorial_size(classifier_only: Optional[str] = None) -> int:
    """
    Get the expected number of configurations for the factorial design.

    Parameters
    ----------
    classifier_only : str | None
        If provided, return count for single classifier.

    Returns
    -------
    int
        Expected number of (outlier × imputation × classifier) combinations.
    """
    n_outliers = EXPECTED_OUTLIER_COUNT
    n_imputations = EXPECTED_IMPUTATION_COUNT
    n_classifiers = 1 if classifier_only else EXPECTED_CLASSIFIER_COUNT

    return n_outliers * n_imputations * n_classifiers


def print_registry_summary() -> None:
    """Print a summary of the registry for debugging."""
    print("=" * 60)
    print("MLflow Registry - Single Source of Truth")
    print("=" * 60)
    print(f"\nOutlier methods ({EXPECTED_OUTLIER_COUNT}):")
    for method in get_valid_outlier_methods():
        print(f"  - {method}")

    print(f"\nImputation methods ({EXPECTED_IMPUTATION_COUNT}):")
    for method in get_valid_imputation_methods():
        print(f"  - {method}")

    print(f"\nClassifiers ({EXPECTED_CLASSIFIER_COUNT}):")
    for clf in get_valid_classifiers():
        print(f"  - {clf}")

    print(f"\nExpected factorial size: {get_expected_factorial_size()}")
    print(f"Expected CatBoost configs: {get_expected_factorial_size('CatBoost')}")


# =============================================================================
# Safe Run Name Parsing
# =============================================================================


def parse_run_name(
    run_name: str,
    require_valid: bool = True,
    log_invalid: bool = True,
) -> Optional[tuple[str, str, str, str]]:
    """
    Parse an MLflow run name with registry validation.

    Run names follow the format: signal__outlier__imputation__classifier

    This function MUST be used instead of raw split("__") parsing to ensure
    only valid methods from the registry are processed.

    Parameters
    ----------
    run_name : str
        MLflow run name in format "signal__outlier__imputation__classifier"
    require_valid : bool, default=True
        If True, return None for runs with invalid methods.
        If False, return parsed values even if not in registry (use with caution).
    log_invalid : bool, default=True
        If True, log a warning for invalid methods.

    Returns
    -------
    tuple[str, str, str, str] | None
        Tuple of (signal_type, outlier, imputation, classifier) if valid,
        None if any component fails validation (when require_valid=True).

    Examples
    --------
    >>> parse_run_name("pupil-full__pupil-gt__pupil-gt__CatBoost")
    ('pupil-full', 'pupil-gt', 'pupil-gt', 'CatBoost')

    >>> parse_run_name("pupil-full__anomaly__linear__CatBoost")  # Invalid outlier
    None

    >>> # WRONG - never do this:
    >>> parts = run_name.split("__")  # BANNED PATTERN

    >>> # CORRECT - always use this:
    >>> parsed = parse_run_name(run_name)
    >>> if parsed is None:
    ...     continue  # Skip invalid run
    >>> signal, outlier, imputation, classifier = parsed

    See Also
    --------
    validate_outlier_method : Check if outlier method is valid
    validate_imputation_method : Check if imputation method is valid
    validate_classifier : Check if classifier is valid
    """
    parts = run_name.split("__")

    if len(parts) != 4:
        if log_invalid:
            logger.warning(f"Invalid run name format (expected 4 parts): {run_name}")
        return None

    signal, outlier, imputation, classifier = parts

    if require_valid:
        # Validate against registry
        if not validate_outlier_method(outlier):
            if log_invalid:
                logger.debug(f"Invalid outlier method '{outlier}' in run: {run_name}")
            return None

        if not validate_imputation_method(imputation):
            if log_invalid:
                logger.debug(
                    f"Invalid imputation method '{imputation}' in run: {run_name}"
                )
            return None

        if not validate_classifier(classifier):
            if log_invalid:
                logger.debug(f"Invalid classifier '{classifier}' in run: {run_name}")
            return None

    return signal, outlier, imputation, classifier


def parse_run_name_or_raise(run_name: str) -> tuple[str, str, str, str]:
    """
    Parse an MLflow run name, raising RegistryError if invalid.

    This is a strict version that raises instead of returning None.
    Use when you want to fail fast on invalid runs.

    Parameters
    ----------
    run_name : str
        MLflow run name in format "signal__outlier__imputation__classifier"

    Returns
    -------
    tuple[str, str, str, str]
        Tuple of (signal_type, outlier, imputation, classifier)

    Raises
    ------
    RegistryError
        If run name format is invalid or contains unregistered methods.
    """
    result = parse_run_name(run_name, require_valid=True, log_invalid=False)

    if result is None:
        parts = run_name.split("__")
        if len(parts) != 4:
            raise RegistryError(
                f"Invalid run name format: {run_name} "
                f"(expected 4 '__'-separated parts, got {len(parts)})"
            )

        _, outlier, imputation, classifier = parts
        invalid = []
        if not validate_outlier_method(outlier):
            invalid.append(f"outlier='{outlier}'")
        if not validate_imputation_method(imputation):
            invalid.append(f"imputation='{imputation}'")
        if not validate_classifier(classifier):
            invalid.append(f"classifier='{classifier}'")

        raise RegistryError(
            f"Invalid methods in run {run_name}: {', '.join(invalid)}. "
            f"Check configs/mlflow_registry/parameters/classification.yaml"
        )

    return result


# Update __all__ to include new functions
__all__.extend(["parse_run_name", "parse_run_name_or_raise"])


if __name__ == "__main__":
    # When run directly, print the registry summary
    print_registry_summary()
