"""Data mode detection and isolation infrastructure.

This module provides the central logic for detecting and enforcing
synthetic vs production data isolation. It implements GATE 0 of
the 4-gate isolation architecture.

Design Goals:
- Prevent synthetic data from contaminating production artifacts
- Provide consistent detection across all pipeline stages
- Enable clear separation of output paths

Detection Hierarchy (ANY match triggers synthetic mode):
1. Environment variable: FOUNDATION_PLR_SYNTHETIC=1/true/yes
2. Config: EXPERIMENT.is_synthetic=true
3. Config: EXPERIMENT.experiment_prefix="synth_"
4. Config: DATA.data_path contains "synthetic"
5. Filename: contains "SYNTH_" or "synthetic"

See: CRITICAL-FAILURE-001 for why this isolation is critical.

Usage:
    from src.utils.data_mode import is_synthetic_mode, get_data_mode

    if is_synthetic_mode():
        # Use synthetic output paths
        output_dir = get_synthetic_output_dir()
    else:
        # Use production output paths
        output_dir = get_production_output_dir()
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

# =============================================================================
# Constants
# =============================================================================

# Prefix added to MLflow run names for synthetic runs
# This makes synthetic runs visually distinct and filterable
SYNTHETIC_RUN_PREFIX = "__SYNTHETIC_"

# Prefix for synthetic experiment names in MLflow
SYNTHETIC_EXPERIMENT_PREFIX = "synth_"

# Environment variable for forcing synthetic mode
SYNTHETIC_ENV_VAR = "FOUNDATION_PLR_SYNTHETIC"

# =============================================================================
# Project Root (for path resolution)
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


# =============================================================================
# Custom Exceptions
# =============================================================================


class SyntheticDataError(Exception):
    """Raised when synthetic data is detected in production context.

    This exception is used by validation functions to prevent synthetic
    data from contaminating production artifacts.
    """

    pass


# =============================================================================
# Core Detection Functions
# =============================================================================


def is_synthetic_mode() -> bool:
    """Check if we're running in synthetic data mode.

    Detection is based on the FOUNDATION_PLR_SYNTHETIC environment variable.

    Returns
    -------
    bool
        True if running in synthetic mode, False for production.

    Examples
    --------
    >>> import os
    >>> os.environ["FOUNDATION_PLR_SYNTHETIC"] = "1"
    >>> is_synthetic_mode()
    True
    """
    env_value = os.environ.get(SYNTHETIC_ENV_VAR, "").lower()
    return env_value in ("1", "true", "yes")


def is_synthetic_from_config(cfg: Any) -> bool:
    """Check if config indicates synthetic data mode.

    Checks multiple config paths that indicate synthetic data:
    1. EXPERIMENT.is_synthetic = true
    2. EXPERIMENT.experiment_prefix = "synth_"
    3. DATA.data_path contains "synthetic"

    Parameters
    ----------
    cfg : DictConfig or dict
        Configuration object (OmegaConf DictConfig or plain dict).

    Returns
    -------
    bool
        True if any synthetic indicator is found in config.

    Examples
    --------
    >>> from omegaconf import OmegaConf
    >>> cfg = OmegaConf.create({"EXPERIMENT": {"is_synthetic": True}})
    >>> is_synthetic_from_config(cfg)
    True
    """
    # Check EXPERIMENT.is_synthetic
    try:
        if cfg.get("EXPERIMENT", {}).get("is_synthetic", False):
            return True
    except (AttributeError, TypeError):
        pass

    # Check EXPERIMENT.experiment_prefix
    try:
        prefix = cfg.get("EXPERIMENT", {}).get("experiment_prefix", "")
        if prefix and SYNTHETIC_EXPERIMENT_PREFIX in prefix.lower():
            return True
    except (AttributeError, TypeError):
        pass

    # Check DATA.data_path
    try:
        data_path = cfg.get("DATA", {}).get("data_path", "")
        if data_path and "synthetic" in str(data_path).lower():
            return True
    except (AttributeError, TypeError):
        pass

    return False


def is_synthetic_from_filename(filename: str) -> bool:
    """Check if filename indicates synthetic data.

    Checks if the filename or path contains synthetic indicators:
    - Starts with "SYNTH_" (case-insensitive)
    - Contains "synthetic" anywhere (case-insensitive)

    Parameters
    ----------
    filename : str
        Filename or path to check.

    Returns
    -------
    bool
        True if filename indicates synthetic data.

    Examples
    --------
    >>> is_synthetic_from_filename("SYNTH_PLR_DEMO.db")
    True
    >>> is_synthetic_from_filename("SERI_PLR_GLAUCOMA.db")
    False
    """
    filename_lower = str(filename).lower()
    basename = Path(filename).name.lower()

    # Check for SYNTH_ prefix in filename
    if basename.startswith("synth_"):
        return True

    # Check for 'synthetic' anywhere in path
    if "synthetic" in filename_lower:
        return True

    return False


def get_data_mode(cfg: Optional[Any] = None, filename: Optional[str] = None) -> str:
    """Get the current data mode.

    Checks all detection sources in order:
    1. Environment variable
    2. Config (if provided)
    3. Filename (if provided)

    Parameters
    ----------
    cfg : DictConfig, optional
        Configuration object to check.
    filename : str, optional
        Filename or path to check.

    Returns
    -------
    str
        Either "synthetic" or "production".

    Examples
    --------
    >>> import os
    >>> os.environ["FOUNDATION_PLR_SYNTHETIC"] = "1"
    >>> get_data_mode()
    'synthetic'
    """
    # Check environment variable first (highest priority)
    if is_synthetic_mode():
        return "synthetic"

    # Check config if provided
    if cfg is not None and is_synthetic_from_config(cfg):
        return "synthetic"

    # Check filename if provided
    if filename is not None and is_synthetic_from_filename(filename):
        return "synthetic"

    return "production"


# =============================================================================
# Run Name Utilities
# =============================================================================


def add_synthetic_prefix_to_run_name(run_name: str) -> str:
    """Add synthetic prefix to MLflow run name.

    Idempotent: If prefix already exists, returns unchanged.

    Parameters
    ----------
    run_name : str
        Original run name.

    Returns
    -------
    str
        Run name with __SYNTHETIC_ prefix.

    Examples
    --------
    >>> add_synthetic_prefix_to_run_name("LOF")
    '__SYNTHETIC_LOF'
    >>> add_synthetic_prefix_to_run_name("__SYNTHETIC_LOF")
    '__SYNTHETIC_LOF'
    """
    if run_name.startswith(SYNTHETIC_RUN_PREFIX):
        return run_name
    return f"{SYNTHETIC_RUN_PREFIX}{run_name}"


def remove_synthetic_prefix_from_run_name(run_name: str) -> str:
    """Remove synthetic prefix from MLflow run name.

    Parameters
    ----------
    run_name : str
        Run name potentially with prefix.

    Returns
    -------
    str
        Run name without __SYNTHETIC_ prefix.

    Examples
    --------
    >>> remove_synthetic_prefix_from_run_name("__SYNTHETIC_LOF")
    'LOF'
    >>> remove_synthetic_prefix_from_run_name("LOF")
    'LOF'
    """
    if run_name.startswith(SYNTHETIC_RUN_PREFIX):
        return run_name[len(SYNTHETIC_RUN_PREFIX) :]
    return run_name


def is_synthetic_run_name(run_name: str) -> bool:
    """Check if run name indicates synthetic data.

    Parameters
    ----------
    run_name : str
        MLflow run name to check.

    Returns
    -------
    bool
        True if run name has synthetic prefix.

    Examples
    --------
    >>> is_synthetic_run_name("__SYNTHETIC_LOF")
    True
    >>> is_synthetic_run_name("LOF")
    False
    """
    return run_name.startswith(SYNTHETIC_RUN_PREFIX)


def is_synthetic_experiment_name(experiment_name: str) -> bool:
    """Check if experiment name indicates synthetic data.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name to check.

    Returns
    -------
    bool
        True if experiment name has synthetic prefix.

    Examples
    --------
    >>> is_synthetic_experiment_name("synth_PLR_Classification")
    True
    >>> is_synthetic_experiment_name("PLR_Classification")
    False
    """
    return experiment_name.lower().startswith(SYNTHETIC_EXPERIMENT_PREFIX)


# =============================================================================
# Output Path Utilities
# =============================================================================


def get_synthetic_output_dir() -> Path:
    """Get output directory for synthetic data artifacts.

    Returns
    -------
    Path
        Path to outputs/synthetic/ directory.
    """
    output_dir = PROJECT_ROOT / "outputs" / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_synthetic_figures_dir() -> Path:
    """Get output directory for synthetic figures.

    Returns
    -------
    Path
        Path to figures/synthetic/ directory.
    """
    figures_dir = PROJECT_ROOT / "figures" / "synthetic"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def get_results_db_path_for_mode(synthetic: bool = False) -> Path:
    """Get results database path based on data mode.

    Parameters
    ----------
    synthetic : bool
        If True, return synthetic database path.

    Returns
    -------
    Path
        Path to appropriate results database.

    Examples
    --------
    >>> path = get_results_db_path_for_mode(synthetic=False)
    >>> "synthetic" in str(path)
    False
    >>> path = get_results_db_path_for_mode(synthetic=True)
    >>> "synthetic" in str(path)
    True
    """
    if synthetic:
        return get_synthetic_output_dir() / "synthetic_foundation_plr_results.db"
    return PROJECT_ROOT / "data" / "public" / "foundation_plr_results.db"


def get_figures_dir_for_mode(synthetic: bool = False) -> Path:
    """Get figures output directory based on data mode.

    Parameters
    ----------
    synthetic : bool
        If True, return synthetic figures directory.

    Returns
    -------
    Path
        Path to appropriate figures directory.
    """
    if synthetic:
        return get_synthetic_figures_dir()
    figures_dir = PROJECT_ROOT / "figures" / "generated"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


# =============================================================================
# MLflow Tag Utilities
# =============================================================================


def get_synthetic_mlflow_tags() -> Dict[str, str]:
    """Get MLflow tags for synthetic runs.

    Returns
    -------
    dict
        Tags to apply to synthetic MLflow runs.
    """
    return {
        "is_synthetic": "true",
        "data_source": "synthetic",
    }


def get_production_mlflow_tags() -> Dict[str, str]:
    """Get MLflow tags for production runs.

    Returns
    -------
    dict
        Tags to apply to production MLflow runs.
    """
    return {
        "is_synthetic": "false",
        "data_source": "production",
    }


def get_mlflow_tags_for_mode(synthetic: bool = False) -> Dict[str, str]:
    """Get appropriate MLflow tags based on data mode.

    Parameters
    ----------
    synthetic : bool
        If True, return synthetic tags.

    Returns
    -------
    dict
        Appropriate tags for the data mode.
    """
    if synthetic:
        return get_synthetic_mlflow_tags()
    return get_production_mlflow_tags()


# =============================================================================
# Validation Functions
# =============================================================================


def validate_not_synthetic(
    run_name: Optional[str] = None,
    db_path: Optional[str] = None,
    experiment_name: Optional[str] = None,
    context: str = "extraction",
) -> None:
    """Validate that data is not synthetic (for production contexts).

    This function should be called in production extraction and figure
    generation to prevent synthetic data contamination.

    Parameters
    ----------
    run_name : str, optional
        MLflow run name to check.
    db_path : str, optional
        Database path to check.
    experiment_name : str, optional
        MLflow experiment name to check.
    context : str
        Context description for error messages.

    Raises
    ------
    SyntheticDataError
        If any parameter indicates synthetic data.

    Examples
    --------
    >>> validate_not_synthetic("LOF", "SERI_PLR_GLAUCOMA.db")
    # Passes silently

    >>> validate_not_synthetic("__SYNTHETIC_LOF", "SERI_PLR_GLAUCOMA.db")
    # Raises SyntheticDataError
    """
    if run_name and is_synthetic_run_name(run_name):
        raise SyntheticDataError(
            f"Synthetic run detected in {context}: {run_name}\n"
            f"Production {context} MUST NOT include synthetic runs.\n"
            f"Run name has prefix '{SYNTHETIC_RUN_PREFIX}'."
        )

    if db_path and is_synthetic_from_filename(db_path):
        raise SyntheticDataError(
            f"Synthetic database detected in {context}: {db_path}\n"
            f"Production {context} MUST NOT use synthetic database.\n"
            f"Path contains 'synthetic' or 'SYNTH_'."
        )

    if experiment_name and is_synthetic_experiment_name(experiment_name):
        raise SyntheticDataError(
            f"Synthetic experiment detected in {context}: {experiment_name}\n"
            f"Production {context} MUST NOT include synthetic experiments.\n"
            f"Experiment name has prefix '{SYNTHETIC_EXPERIMENT_PREFIX}'."
        )


def validate_run_for_production_extraction(
    run_name: str,
    experiment_name: str,
    tags: Optional[Dict[str, str]] = None,
) -> bool:
    """Check if a run should be included in production extraction.

    Returns False (skip) instead of raising for synthetic runs,
    allowing extraction to continue with other runs.

    Parameters
    ----------
    run_name : str
        MLflow run name.
    experiment_name : str
        MLflow experiment name.
    tags : dict, optional
        MLflow run tags.

    Returns
    -------
    bool
        True if run should be included, False if it should be skipped.
    """
    # Check run name prefix
    if is_synthetic_run_name(run_name):
        logger.debug(f"Skipping synthetic run: {run_name}")
        return False

    # Check experiment name
    if is_synthetic_experiment_name(experiment_name):
        logger.debug(f"Skipping run from synthetic experiment: {experiment_name}")
        return False

    # Check tags
    if tags:
        if tags.get("is_synthetic", "").lower() == "true":
            logger.debug(f"Skipping run with synthetic tag: {run_name}")
            return False
        if tags.get("data_source", "").lower() == "synthetic":
            logger.debug(f"Skipping run with synthetic data_source: {run_name}")
            return False

    return True


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Constants
    "SYNTHETIC_RUN_PREFIX",
    "SYNTHETIC_EXPERIMENT_PREFIX",
    "SYNTHETIC_ENV_VAR",
    # Exceptions
    "SyntheticDataError",
    # Detection functions
    "is_synthetic_mode",
    "is_synthetic_from_config",
    "is_synthetic_from_filename",
    "get_data_mode",
    # Run name utilities
    "add_synthetic_prefix_to_run_name",
    "remove_synthetic_prefix_from_run_name",
    "is_synthetic_run_name",
    "is_synthetic_experiment_name",
    # Output path utilities
    "get_synthetic_output_dir",
    "get_synthetic_figures_dir",
    "get_results_db_path_for_mode",
    "get_figures_dir_for_mode",
    # MLflow tag utilities
    "get_synthetic_mlflow_tags",
    "get_production_mlflow_tags",
    "get_mlflow_tags_for_mode",
    # Validation functions
    "validate_not_synthetic",
    "validate_run_for_production_extraction",
]
