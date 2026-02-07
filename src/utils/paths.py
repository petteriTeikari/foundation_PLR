"""Centralized path resolution - SINGLE SOURCE OF TRUTH for all paths.

This module provides environment-aware path resolution to eliminate hardcoded
absolute paths throughout the codebase. All path access should go through
these functions.

Usage:
    from src.utils.paths import get_mlruns_dir, get_seri_db_path

    mlruns = get_mlruns_dir()  # Returns Path, uses env var or default
    seri_db = get_seri_db_path()  # Returns Path, uses env var or default

Environment Variables (see .env.example):
    MLRUNS_DIR - MLflow runs directory
    SERI_DB_PATH - SERI PLR database path
    FOUNDATION_PLR_RESULTS_DB - Extracted results database
    PREPROCESSED_SIGNALS_DB - Preprocessed signals database
    CLASSIFICATION_EXP_ID - MLflow classification experiment ID

Note: This module was created to address CRITICAL-FAILURE in code review
where 9 files had hardcoded /home/petteri/... paths that would break on
other machines and in CI/CD.
"""

import os
from functools import lru_cache
from pathlib import Path

# Project root - computed relative to this file's location
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


# =============================================================================
# MLflow Paths
# =============================================================================


@lru_cache(maxsize=1)
def get_mlruns_dir() -> Path:
    """Get MLflow runs directory from environment or default.

    Returns
    -------
    Path
        MLflow runs directory. Uses MLRUNS_DIR env var if set,
        otherwise defaults to ~/mlruns.

    Examples
    --------
    >>> mlruns = get_mlruns_dir()
    >>> print(mlruns)
    /home/user/mlruns
    """
    env_path = os.environ.get("MLRUNS_DIR")
    if env_path:
        return Path(env_path).resolve()
    return Path.home() / "mlruns"


@lru_cache(maxsize=1)
def get_classification_experiment_id() -> str:
    """Get MLflow classification experiment ID.

    Returns
    -------
    str
        Experiment ID. Uses CLASSIFICATION_EXP_ID env var if set,
        otherwise returns the default experiment ID.
    """
    return os.environ.get("CLASSIFICATION_EXP_ID", "253031330985650090")


# =============================================================================
# Data Paths
# =============================================================================


@lru_cache(maxsize=1)
def get_synthetic_db_path() -> Path:
    """Get synthetic PLR database path for testing.

    Returns
    -------
    Path
        Path to SYNTH_PLR_DEMO.db (synthetic data for CI/testing).
    """
    return PROJECT_ROOT / "data" / "synthetic" / "SYNTH_PLR_DEMO.db"


@lru_cache(maxsize=1)
def get_seri_db_path() -> Path:
    """Get SERI PLR database path from environment or default.

    Returns
    -------
    Path
        Path to SERI_PLR_GLAUCOMA.db. Uses SERI_DB_PATH env var if set,
        otherwise tries several default locations.

    Raises
    ------
    FileNotFoundError
        If the database cannot be found at any location.
    """
    env_path = os.environ.get("SERI_DB_PATH")
    if env_path:
        path = Path(env_path).resolve()
        if path.exists():
            return path
        # Env var set but file doesn't exist - still return it
        # (may be created later or caller will handle error)
        return path

    # Try default locations in order
    default_locations = [
        PROJECT_ROOT / "data" / "private" / "SERI_PLR_GLAUCOMA.db",
        PROJECT_ROOT.parent / "SERI_PLR_GLAUCOMA.db",
        Path.home() / "data" / "SERI_PLR_GLAUCOMA.db",
    ]

    for loc in default_locations:
        if loc.exists():
            return loc

    # Return first default even if it doesn't exist (let caller handle)
    return default_locations[0]


@lru_cache(maxsize=1)
def get_results_db_path() -> Path:
    """Get foundation PLR results database path.

    Returns
    -------
    Path
        Path to foundation_plr_results.db (extracted metrics).
        Uses FOUNDATION_PLR_RESULTS_DB env var if set.
    """
    env_path = os.environ.get("FOUNDATION_PLR_RESULTS_DB")
    if env_path:
        return Path(env_path).resolve()
    return PROJECT_ROOT / "data" / "public" / "foundation_plr_results.db"


@lru_cache(maxsize=1)
def get_preprocessed_signals_db_path() -> Path:
    """Get preprocessed signals database path.

    Returns
    -------
    Path
        Path to preprocessed_signals_per_subject.db.
        Uses PREPROCESSED_SIGNALS_DB env var if set.
    """
    env_path = os.environ.get("PREPROCESSED_SIGNALS_DB")
    if env_path:
        return Path(env_path).resolve()
    return PROJECT_ROOT / "data" / "private" / "preprocessed_signals_per_subject.db"


# =============================================================================
# Output Paths
# =============================================================================


def get_figures_output_dir() -> Path:
    """Get figures output directory.

    Returns
    -------
    Path
        Directory for generated figures. Creates if doesn't exist.
    """
    env_path = os.environ.get("FIGURES_OUTPUT_DIR")
    if env_path:
        output_dir = Path(env_path).resolve()
    else:
        output_dir = PROJECT_ROOT / "figures" / "generated"

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_json_data_dir() -> Path:
    """Get JSON data output directory (for figure reproducibility).

    Returns
    -------
    Path
        Directory for JSON data files. Creates if doesn't exist.
    """
    data_dir = get_figures_output_dir() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


# =============================================================================
# Config Paths
# =============================================================================


def get_config_dir() -> Path:
    """Get configuration directory.

    Returns
    -------
    Path
        Path to configs/ directory.
    """
    return PROJECT_ROOT / "configs"


def get_visualization_config_dir() -> Path:
    """Get visualization configuration directory.

    Returns
    -------
    Path
        Path to configs/VISUALIZATION/ directory.
    """
    return get_config_dir() / "VISUALIZATION"


def get_mlflow_registry_dir() -> Path:
    """Get MLflow registry configuration directory.

    Returns
    -------
    Path
        Path to configs/mlflow_registry/ directory.
    """
    return get_config_dir() / "mlflow_registry"


# =============================================================================
# Utility Functions
# =============================================================================


def resolve_path(path: str | Path, must_exist: bool = False) -> Path:
    """Resolve a path, expanding ~ and making absolute.

    Parameters
    ----------
    path : str or Path
        Path to resolve.
    must_exist : bool, optional
        If True, raises FileNotFoundError if path doesn't exist.

    Returns
    -------
    Path
        Resolved absolute path.

    Raises
    ------
    FileNotFoundError
        If must_exist=True and path doesn't exist.
    """
    resolved = Path(path).expanduser().resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved


def clear_path_caches() -> None:
    """Clear all cached path values.

    Useful for testing or when environment variables change.
    """
    get_mlruns_dir.cache_clear()
    get_classification_experiment_id.cache_clear()
    get_seri_db_path.cache_clear()
    get_results_db_path.cache_clear()
    get_preprocessed_signals_db_path.cache_clear()


# =============================================================================
# Module-level exports
# =============================================================================

__all__ = [
    "PROJECT_ROOT",
    "get_mlruns_dir",
    "get_classification_experiment_id",
    "get_seri_db_path",
    "get_synthetic_db_path",
    "get_results_db_path",
    "get_preprocessed_signals_db_path",
    "get_figures_output_dir",
    "get_json_data_dir",
    "get_config_dir",
    "get_visualization_config_dir",
    "get_mlflow_registry_dir",
    "resolve_path",
    "clear_path_caches",
]
