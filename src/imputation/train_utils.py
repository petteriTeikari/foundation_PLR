import os
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from src.preprocess.preprocess_data import destandardize_numpy


def if_results_file_found(results_path: str) -> bool:
    """Check if a results file exists at the specified path.

    Parameters
    ----------
    results_path : str
        Full path to the results file.

    Returns
    -------
    bool
        True if the file exists, False otherwise.
    """
    if os.path.exists(results_path):
        return True
    else:
        dir, fname = os.path.split(results_path)
        logger.debug(
            "Could not find the results  file from the artifact_dir\n"
            "dir = {}, fname = {}".format(dir, fname)
        )
        return False


def create_imputation_dict(
    imputation_mean: np.ndarray,
    preprocess: Dict[str, Any],
    X_missing: np.ndarray,
    cfg: DictConfig,
    end_time: Optional[float] = None,
) -> Dict[str, Any]:
    """Create standardized imputation result dictionary.

    Formats imputation results into the common structure used across
    all imputation methods, with optional destandardization.

    Parameters
    ----------
    imputation_mean : np.ndarray
        Imputed values array, shape (samples, timepoints) or (samples, timepoints, features).
    preprocess : dict
        Preprocessing dictionary containing 'standardization' with mean and stdev.
    X_missing : np.ndarray
        Original array with NaN values indicating missing points.
    cfg : DictConfig
        Configuration with PREPROCESS.standardize flag.
    end_time : float, optional
        Time taken for imputation in seconds. Default is None.

    Returns
    -------
    dict
        Standardized imputation dictionary containing:
        - 'imputation_dict': Dict with 'imputation' (mean, CI bounds) and 'indicating_mask'
        - 'timing': Elapsed time if provided

    Notes
    -----
    If input is 2D, a third dimension is added to match expected (samples, timepoints, features) format.
    Destandardization is applied if configured.
    """
    # Get boolean mask of missing (NaN) values
    indicating_mask = np.isnan(X_missing)

    if len(imputation_mean.shape) == 2:
        # Add the third dimension, as the downstream code expects 3D arrays
        imputation_mean = np.expand_dims(imputation_mean, axis=2)
        logger.debug("Adding the third dimension to the imputation_mean array")

    assert len(imputation_mean.shape) == 3

    # Destandardize the data (if needed)
    if cfg["PREPROCESS"]["standardize"]:
        logger.debug("Destandardizing the imputed data")
        imputation_mean = destandardize_numpy(
            imputation_mean,
            mean=preprocess["standardization"]["mean"],
            std=preprocess["standardization"]["stdev"],
        )

    imputation_dict = {
        "imputation_dict": {
            "imputation": {
                # (no_samples, no_timepoints, no_features)
                "mean": imputation_mean,
                "imputation_ci_neg": None,
                "imputation_ci_pos": None,
            },
            "indicating_mask": indicating_mask,
        },
        "timing": end_time,
    }

    return imputation_dict


def create_imputation_dict_from_moment(
    imputation_mean: np.ndarray, indicating_mask: np.ndarray, imputation_time: float
) -> Dict[str, Any]:
    """Create imputation dictionary from MOMENT model outputs.

    Formats MOMENT-specific outputs into the common imputation structure.

    Parameters
    ----------
    imputation_mean : np.ndarray
        Imputed values array from MOMENT model.
    indicating_mask : np.ndarray
        Boolean mask indicating originally missing values.
    imputation_time : float
        Time taken for imputation in seconds.

    Returns
    -------
    dict
        Standardized imputation dictionary with imputation values,
        mask, and timing information.
    """
    imputation_dict = {
        "imputation_dict": {
            "imputation": {
                # (no_samples, no_timepoints, no_features)
                "mean": imputation_mean,
                "imputation_ci_neg": None,
                "imputation_ci_pos": None,
            },
            "indicating_mask": indicating_mask,
        },
        "timing": imputation_time,
    }

    return imputation_dict


def imputation_per_split_of_dict(
    data_dicts: Dict[str, Any],
    df: pd.DataFrame,
    preprocess: Dict[str, Any],
    model: Any,
    split: str,
    cfg: DictConfig,
) -> Dict[str, Any]:
    """Apply imputation model to a single data split.

    Transforms the input DataFrame using the trained model and creates
    a standardized imputation result dictionary.

    Parameters
    ----------
    data_dicts : dict
        Data dictionaries (unused but kept for interface consistency).
    df : pd.DataFrame
        DataFrame with missing values (NaN) to impute.
    preprocess : dict
        Preprocessing dictionary with standardization statistics.
    model : object
        Trained imputation model with transform() method.
    split : str
        Split name for logging.
    cfg : DictConfig
        Configuration for imputation settings.

    Returns
    -------
    dict
        Imputation result dictionary with imputed values and timing.
    """
    X_missing = df.to_numpy()
    logger.debug("Split = {}".format(split))
    start_time = time.time()
    imputation_mean = model.transform(x=df)
    dict_out = create_imputation_dict(
        imputation_mean=imputation_mean,
        preprocess=preprocess,
        X_missing=X_missing,
        end_time=time.time() - start_time,
        cfg=cfg,
    )

    return dict_out
