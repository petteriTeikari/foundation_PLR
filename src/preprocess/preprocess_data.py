from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from loguru import logger
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler


def preprocess_PLR_data(
    X: np.ndarray,
    preprocess_cfg: Union[Dict[str, Any], DictConfig],
    preprocess_dict: Optional[Dict[str, Any]] = None,
    data_filtering: str = "gt",
    split: str = "train",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Preprocess PLR data by applying standardization if configured.

    Parameters
    ----------
    X : np.ndarray
        Input PLR data array to preprocess.
    preprocess_cfg : dict
        Configuration dictionary containing preprocessing settings,
        including 'standardize' and 'use_gt_stats_for_raw' flags.
    preprocess_dict : dict, optional
        Dictionary to store/retrieve precomputed statistics. Default is None.
    data_filtering : str, optional
        Type of data filtering applied ('gt' for ground truth, 'raw' for raw data).
        Default is 'gt'.
    split : str, optional
        Data split identifier ('train', 'val', or 'test'). Default is 'train'.

    Returns
    -------
    tuple
        A tuple containing:
        - X : np.ndarray
            The preprocessed data array.
        - preprocess_dict : dict
            Updated preprocessing dictionary with computed statistics.
    """
    if preprocess_dict is None:
        preprocess_dict = {}

    if preprocess_cfg["standardize"]:
        use_precomputed, mean, std, filterkey = if_use_precomputed(
            preprocess_dict, preprocess_cfg, split, data_filtering
        )
        if use_precomputed:
            X = standardize_with_precomputed_stats(
                X, preprocess_dict, data_filtering, filterkey, split
            )
        else:
            preprocess_dict, X = compute_stats_and_standardize(
                preprocess_dict, X, data_filtering, split
            )

    logger.debug(
        'Number of NaNs in the "{}" data: {}'.format(data_filtering, np.isnan(X).sum())
    )

    return X, preprocess_dict


def if_use_precomputed(
    preprocess_dict: Dict[str, Any],
    preprocess_cfg: Union[Dict[str, Any], DictConfig],
    split: str,
    data_filtering: str,
) -> Tuple[bool, Optional[float], Optional[float], str]:
    """Determine whether to use precomputed standardization statistics.

    Parameters
    ----------
    preprocess_dict : dict
        Dictionary containing previously computed statistics.
    preprocess_cfg : dict
        Configuration dictionary with preprocessing settings.
    split : str
        Data split identifier ('train', 'val', or 'test').
    data_filtering : str
        Type of data filtering ('gt' or 'raw').

    Returns
    -------
    tuple
        A tuple containing:
        - use_precomputed : bool
            Whether to use precomputed statistics.
        - mean : float or None
            Precomputed mean value if available.
        - std : float or None
            Precomputed standard deviation if available.
        - filterkey : str
            The key used to retrieve statistics from the dictionary.
    """
    mean, std, filterkey = None, None, "gt"
    if len(preprocess_dict) == 0:
        return False, mean, std, filterkey
    elif "standardize" in preprocess_dict:
        if preprocess_cfg["use_gt_stats_for_raw"]:
            logger.debug("Use mean&stdev from GT for raw data")
            if "gt" in preprocess_dict["standardize"]:
                mean = preprocess_dict["standardize"]["gt"]["mean"]
                std = preprocess_dict["standardize"]["gt"]["std"]
                log_stats_msg(mean, std, split, data_filtering, "precomputed")
                filterkey = "gt"
                return True, mean, std, filterkey
            else:
                return False, mean, std, filterkey
        else:
            raise NotImplementedError("Not implemented yet")
            # if data_filtering in preprocess_dict["standardize"]:
            #     mean = preprocess_dict["standardize"][data_filtering]["mean"]
            #     std = preprocess_dict["standardize"][data_filtering]["std"]
            #     log_stats_msg(mean, std, split, data_filtering, "precomputed")
            #     filterkey = data_filtering
            #     return True, mean, std, filterkey
            # else:
            #     return False, mean, std, filterkey


def log_stats_msg(
    mean: float,
    std: float,
    split: str,
    data_filtering: str,
    call_from: str = "precomputed",
) -> None:
    """Log standardization statistics message for debugging.

    Parameters
    ----------
    mean : float
        Mean value of the data.
    std : float
        Standard deviation of the data.
    split : str
        Data split identifier ('train', 'val', or 'test').
    data_filtering : str
        Type of data filtering ('gt' or 'raw').
    call_from : str, optional
        Source of the call, either 'precomputed' or 'standardize'.
        Default is 'precomputed'.

    Raises
    ------
    NotImplementedError
        If call_from is neither 'precomputed' nor 'standardize'.
    """
    if call_from == "precomputed":
        string = "Mean&Std already precomputed"
    elif call_from == "standardize":
        string = "STATS after standardization"
    else:
        raise NotImplementedError("Unknown call_from = {}".format(call_from))

    logger.debug(
        "{}: mean = {}, std = {}, split = {}, data_filtering = {}".format(
            string, mean, std, split, data_filtering
        )
    )


def standardize_with_precomputed_stats(
    X: np.ndarray,
    preprocess_dict: Dict[str, Any],
    data_filtering: str,
    filterkey: str,
    split: str,
) -> np.ndarray:
    """Standardize data using precomputed mean and standard deviation.

    Parameters
    ----------
    X : np.ndarray
        Input data array to standardize.
    preprocess_dict : dict
        Dictionary containing precomputed standardization statistics.
    data_filtering : str
        Type of data filtering ('gt' or 'raw').
    filterkey : str
        Key to access the correct statistics in preprocess_dict.
    split : str
        Data split identifier ('train', 'val', or 'test').

    Returns
    -------
    np.ndarray
        Standardized data array with zero mean and unit variance.
    """
    X = (X - preprocess_dict["standardize"][filterkey]["mean"]) / preprocess_dict[
        "standardize"
    ][filterkey]["std"]
    logger.debug(
        "Data has been standardized, mean = {}, std = {}".format(
            np.nanmean(X), np.nanstd(X)
        )
    )
    return X


def compute_stats_and_standardize(
    preprocess_dict: Dict[str, Any],
    X: np.ndarray,
    data_filtering: str,
    split: str,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Compute standardization statistics and apply standardization to data.

    Fits a StandardScaler to the data, transforms it, and stores the
    computed mean and standard deviation in the preprocess dictionary.

    Parameters
    ----------
    preprocess_dict : dict
        Dictionary to store the computed standardization statistics.
    X : np.ndarray
        Input data array to standardize.
    data_filtering : str
        Type of data filtering ('gt' or 'raw').
    split : str
        Data split identifier ('train', 'val', or 'test').

    Returns
    -------
    tuple
        A tuple containing:
        - preprocess_dict : dict
            Updated dictionary with computed mean and std.
        - X : np.ndarray
            Standardized data array.
    """
    no_samples = X.shape[0] * X.shape[1]
    scaler = StandardScaler()
    scaler.fit(X.reshape(no_samples, -1))
    X = scaler.transform(X.reshape(no_samples, -1)).reshape(X.shape)
    preprocess_dict["standardize"] = {}
    print_stdz_stats(scaler, split, data_filtering)

    if "standardize" not in preprocess_dict:
        preprocess_dict["standardize"] = {}

    if data_filtering not in preprocess_dict["standardize"]:
        preprocess_dict["standardize"][data_filtering] = {}

    preprocess_dict["standardize"][data_filtering]["mean"] = float(scaler.mean_)
    preprocess_dict["standardize"][data_filtering]["std"] = float(scaler.scale_)

    log_stats_msg(np.nanmean(X), np.nanstd(X), split, data_filtering, "standardize")

    return preprocess_dict, X


def print_stdz_stats(scaler: StandardScaler, split: str, data_filtering: str) -> None:
    """Print standardization statistics from a fitted scaler.

    Logs the mean and scale values at INFO level for training ground truth,
    and at DEBUG level for other splits/filters to reduce log clutter.

    Parameters
    ----------
    scaler : sklearn.preprocessing.StandardScaler
        Fitted StandardScaler object containing mean_ and scale_ attributes.
    split : str
        Data split identifier ('train', 'val', or 'test').
    data_filtering : str
        Type of data filtering ('gt' or 'raw').
    """
    if split == "train" and data_filtering == "gt":
        # Print only once the standardized stats to reduce clutter
        logger.info(
            "Standardized (split = {}, split_key = {}), mean = {}, std = {}".format(
                split, data_filtering, scaler.mean_, scaler.scale_
            )
        )
    else:
        logger.debug(
            "Standardized, mean = {}, std = {}".format(scaler.mean_, scaler.scale_)
        )


def debug_triplet_stats(
    X_gt: np.ndarray,
    X_gt_missing: np.ndarray,
    X_raw: np.ndarray,
    split: str,
) -> None:
    """Log debug statistics for the data filtering triplet.

    Computes and logs mean, standard deviation, and NaN count for
    ground truth, ground truth with missing values, and raw data.

    Parameters
    ----------
    X_gt : np.ndarray
        Ground truth data array.
    X_gt_missing : np.ndarray
        Ground truth data with missing values (NaNs).
    X_raw : np.ndarray
        Raw unprocessed data array.
    split : str
        Data split identifier ('train', 'val', or 'test').

    Returns
    -------
    None
    """

    def stats_per_split(X: np.ndarray, split: str) -> Dict[str, float]:
        logger.debug(
            "{}: mean = {}, std = {}, no_NaN = {}".format(
                split, np.nanmean(X), np.nanstd(X), np.isnan(X).sum()
            )
        )
        return {"mean": np.nanmean(X), "std": np.nanstd(X), "no_NaN": np.isnan(X).sum()}

    logger.debug("DEBUG FOR THE 'FILTERING TRIPLET', split = {}:".format(split))
    stats_per_split(X_gt, "GT")
    stats_per_split(X_gt_missing, "GT_MISSING")
    stats_per_split(X_raw, "RAW")

    return None


def destandardize_for_imputation_metric(
    targets: np.ndarray,
    predictions: np.ndarray,
    stdz_dict: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Destandardize targets and predictions for computing imputation metrics.

    Reverses the standardization transformation to compute metrics in
    the original data scale.

    Parameters
    ----------
    targets : np.ndarray
        Ground truth target values (potentially standardized).
    predictions : np.ndarray
        Model predictions (potentially standardized).
    stdz_dict : dict
        Standardization dictionary containing 'standardized' boolean,
        'mean', and 'stdev' values.

    Returns
    -------
    tuple
        A tuple containing:
        - targets : np.ndarray
            Destandardized target values.
        - predictions : np.ndarray
            Destandardized prediction values.
    """
    if stdz_dict["standardized"]:
        targets = destandardize_numpy(targets, stdz_dict["mean"], stdz_dict["stdev"])
        predictions = destandardize_numpy(
            predictions, stdz_dict["mean"], stdz_dict["stdev"]
        )

    return targets, predictions


def destandardize_dict(
    imputation_dict: Dict[str, Any],
    mean: float,
    std: float,
) -> Dict[str, Any]:
    """Destandardize the mean values in an imputation results dictionary.

    Parameters
    ----------
    imputation_dict : dict
        Dictionary containing imputation results with a 'mean' key.
    mean : float
        Mean value used for original standardization.
    std : float
        Standard deviation used for original standardization.

    Returns
    -------
    dict
        Updated imputation dictionary with destandardized mean values.

    Notes
    -----
    TODO: Confidence intervals (CI) are not yet destandardized.
    """
    logger.debug(
        "De-standardizing the imputation results with mean = {} and std = {}".format(
            mean, std
        )
    )
    imputation_dict["mean"] = imputation_dict["mean"] * std + mean
    # TODO! Also for the confidence intervals (CI)
    return imputation_dict


def destandardize_numpy(X: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Reverse standardization on a numpy array.

    Applies the inverse transformation: X_original = X_standardized * std + mean.

    Parameters
    ----------
    X : np.ndarray
        Standardized data array.
    mean : float
        Mean value used for original standardization.
    std : float
        Standard deviation used for original standardization.

    Returns
    -------
    np.ndarray
        Destandardized data array in original scale.
    """
    logger.debug(
        "De-standardizing the imputation results with mean = {} and std = {}".format(
            mean, std
        )
    )
    return X * std + mean


def destandardize_for_imputation_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    preprocess_dict: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Destandardize targets and predictions with automatic scale detection.

    Detects if predictions and targets are on different scales (one
    destandardized, one not) and corrects accordingly before returning
    both in the original scale.

    Parameters
    ----------
    targets : np.ndarray
        Ground truth target values.
    predictions : np.ndarray
        Model predictions.
    preprocess_dict : dict
        Dictionary containing 'standardization' sub-dict with 'standardized',
        'mean', and 'stdev' keys.

    Returns
    -------
    tuple
        A tuple containing:
        - targets : np.ndarray
            Destandardized target values.
        - predictions : np.ndarray
            Destandardized prediction values.

    Notes
    -----
    If predictions are more than 100x larger than targets in absolute mean,
    assumes predictions were already destandardized and only destandardizes
    targets.
    """
    predictions_mean = np.nanmean(predictions)
    targets_mean = np.nanmean(targets)
    predictions_larger_ratio = abs(predictions_mean) / abs(targets_mean)
    if predictions_larger_ratio > 100:
        logger.debug(
            "Predictions are larger than targets by a factor of {}".format(
                predictions_larger_ratio
            )
        )
        logger.debug(
            "It seems that your predictions are inverse transformed (destandardized) and targets are not"
        )
        logger.debug(
            "Check if you have destandardized the predictions and targets correctly"
        )
        logger.debug("Destandardizing now the targets as well for you")
        targets = destandardize_numpy(
            targets,
            preprocess_dict["standardization"]["mean"],
            preprocess_dict["standardization"]["stdev"],
        )
    else:
        if preprocess_dict["standardization"]["standardized"]:
            targets = destandardize_numpy(
                targets,
                preprocess_dict["standardization"]["mean"],
                preprocess_dict["standardization"]["stdev"],
            )
            predictions = destandardize_numpy(
                predictions,
                preprocess_dict["standardization"]["mean"],
                preprocess_dict["standardization"]["stdev"],
            )

    return targets, predictions
