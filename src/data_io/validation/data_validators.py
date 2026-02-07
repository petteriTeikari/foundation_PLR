"""
Data validation utilities for the Foundation PLR pipeline.

These validators implement fail-fast checks to catch data issues early
in the pipeline before they propagate downstream.

See: docs/planning/pipeline-robustness-plan.md for context.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import duckdb
import numpy as np
import polars as pl
from loguru import logger


class DataValidationError(Exception):
    """Raised when data validation fails."""

    pass


# =============================================================================
# Signal Validators
# =============================================================================


def validate_light_stimuli(df: pl.DataFrame) -> None:
    """
    Validate that light stimuli have proper timing.

    The light onset must be before the light offset.

    Args:
        df: DataFrame with 'time', 'Red', and/or 'Blue' columns

    Raises:
        DataValidationError: If light timing is invalid
    """
    for color in ["Red", "Blue"]:
        if color not in df.columns:
            continue

        # Find onset and offset
        light_on = df.filter(pl.col(color) > 0)
        if light_on.is_empty():
            logger.warning(f"No {color} light stimulus found in data")
            continue

        onset = light_on.select("time").min().item()
        offset = light_on.select("time").max().item()

        if onset >= offset:
            raise DataValidationError(
                f"{color} light onset ({onset}) must be before offset ({offset})"
            )

    logger.debug("Light stimuli validation passed")


def validate_signal_range(
    signal: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 10.0,
    allow_nan: bool = True,
) -> None:
    """
    Validate that signal values are within expected range.

    Args:
        signal: Numpy array of signal values
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        allow_nan: Whether to allow NaN values

    Raises:
        DataValidationError: If signal is out of range
    """
    if not allow_nan and np.any(np.isnan(signal)):
        raise DataValidationError("Signal contains NaN values but allow_nan=False")

    # Check non-NaN values
    valid_values = signal[~np.isnan(signal)]

    if len(valid_values) == 0:
        raise DataValidationError("Signal contains only NaN values")

    if np.any(valid_values < min_val):
        min_found = np.nanmin(valid_values)
        raise DataValidationError(
            f"Signal value {min_found:.4f} below minimum {min_val}"
        )

    if np.any(valid_values > max_val):
        max_found = np.nanmax(valid_values)
        raise DataValidationError(
            f"Signal value {max_found:.4f} above maximum {max_val}"
        )

    logger.debug(
        f"Signal range validation passed: [{np.nanmin(valid_values):.4f}, {np.nanmax(valid_values):.4f}]"
    )


def validate_time_monotonic(df: pl.DataFrame) -> None:
    """
    Validate that time column is strictly monotonically increasing.

    Args:
        df: DataFrame with 'time' column

    Raises:
        DataValidationError: If time is not monotonic
    """
    if "time" not in df.columns:
        raise DataValidationError("DataFrame missing 'time' column")

    time_vals = df.get_column("time").to_numpy()
    time_diff = np.diff(time_vals)

    if not np.all(time_diff > 0):
        non_monotonic_idx = np.where(time_diff <= 0)[0]
        raise DataValidationError(
            f"Time is not strictly monotonic. Issues at indices: {non_monotonic_idx[:5].tolist()}"
        )

    logger.debug("Time monotonicity validation passed")


# =============================================================================
# Feature Validators
# =============================================================================


def validate_features(
    features: Dict[str, Any], required_keys: Optional[List[str]] = None
) -> None:
    """
    Validate that extracted features are complete.

    Args:
        features: Dictionary of extracted features
        required_keys: List of required feature keys. Defaults to standard features.

    Raises:
        DataValidationError: If required features are missing
    """
    if required_keys is None:
        required_keys = ["amplitude_bins"]

    missing = [key for key in required_keys if key not in features]

    if missing:
        raise DataValidationError(f"Missing required features: {missing}")

    # Validate amplitude bins if present
    if "amplitude_bins" in features:
        bins = features["amplitude_bins"]
        if not isinstance(bins, (list, np.ndarray)):
            raise DataValidationError(
                f"amplitude_bins must be list/array, got {type(bins)}"
            )
        if len(bins) == 0:
            raise DataValidationError("amplitude_bins is empty")

    logger.debug(f"Feature validation passed: {list(features.keys())}")


# =============================================================================
# Database Validators
# =============================================================================


def validate_database_schema(
    db_path: Union[str, Path],
    required_tables: Optional[List[str]] = None,
    required_columns: Optional[Dict[str, List[str]]] = None,
) -> None:
    """
    Validate that database has required schema.

    Args:
        db_path: Path to DuckDB database
        required_tables: List of required table names
        required_columns: Dict mapping table names to required columns

    Raises:
        DataValidationError: If schema is invalid
        FileNotFoundError: If database doesn't exist
    """
    db_path = Path(db_path)

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    if required_tables is None:
        required_tables = ["train", "test"]

    if required_columns is None:
        required_columns = {
            "train": ["time", "pupil_raw", "outlier_mask", "subject_code"],
            "test": ["time", "pupil_raw", "outlier_mask", "subject_code"],
        }

    conn = duckdb.connect(str(db_path), read_only=True)

    try:
        # Check tables exist
        tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]

        missing_tables = [t for t in required_tables if t not in tables]
        if missing_tables:
            raise DataValidationError(f"Missing required tables: {missing_tables}")

        # Check columns for each table
        for table, required_cols in required_columns.items():
            if table not in tables:
                continue

            cols = conn.execute(f"DESCRIBE {table}").fetchdf()["column_name"].tolist()

            missing_cols = [c for c in required_cols if c not in cols]
            if missing_cols:
                raise DataValidationError(
                    f"Table '{table}' missing columns: {missing_cols}"
                )

    finally:
        conn.close()

    logger.debug(f"Database schema validation passed: {db_path}")


def validate_subject_count(
    db_path: Union[str, Path],
    expected_train: Optional[int] = None,
    expected_test: Optional[int] = None,
) -> Dict[str, int]:
    """
    Validate subject counts in database.

    Args:
        db_path: Path to DuckDB database
        expected_train: Expected number of train subjects (optional)
        expected_test: Expected number of test subjects (optional)

    Returns:
        Dict with actual counts

    Raises:
        DataValidationError: If counts don't match expected
    """
    db_path = Path(db_path)

    conn = duckdb.connect(str(db_path), read_only=True)

    try:
        counts = {}

        for split in ["train", "test"]:
            try:
                count = conn.execute(
                    f"SELECT COUNT(DISTINCT subject_code) FROM {split}"
                ).fetchone()[0]
                counts[split] = count
            except Exception:
                counts[split] = 0

        # Validate if expected values provided
        if expected_train is not None and counts.get("train", 0) != expected_train:
            raise DataValidationError(
                f"Train subject count mismatch: expected {expected_train}, got {counts.get('train', 0)}"
            )

        if expected_test is not None and counts.get("test", 0) != expected_test:
            raise DataValidationError(
                f"Test subject count mismatch: expected {expected_test}, got {counts.get('test', 0)}"
            )

    finally:
        conn.close()

    logger.debug(f"Subject count validation passed: {counts}")
    return counts


# =============================================================================
# Pipeline Input Validators
# =============================================================================


def validate_pipeline_inputs(cfg: Any) -> None:
    """
    Validate pipeline inputs at startup.

    This is a fail-fast check to catch configuration errors before
    running expensive computations.

    Args:
        cfg: Hydra configuration object

    Raises:
        DataValidationError: If inputs are invalid
        FileNotFoundError: If data files don't exist
    """
    # Check data path exists
    data_path = Path(cfg.DATA.data_path)
    db_filename = cfg.DATA.filename_DuckDB

    db_path = data_path / db_filename

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    # Validate database schema
    validate_database_schema(db_path)

    logger.info(f"Pipeline input validation passed: {db_path}")
