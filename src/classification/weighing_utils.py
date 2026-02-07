from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig


def zscore_norm(
    weights_array: np.ndarray,
    i: int,
    feature_stats: Dict[int, Dict[str, float]],
    samples: np.ndarray,
    xgboost_cfg: DictConfig,
    samplewise: bool = True,
) -> Tuple[np.ndarray, Dict[int, Dict[str, float]]]:
    """
    Apply z-score normalization to a feature column.

    Parameters
    ----------
    weights_array : np.ndarray
        2D array of weights with shape (n_samples, n_features).
    i : int
        Index of the feature column to normalize.
    feature_stats : dict
        Dictionary to store normalization statistics (mean, std).
    samples : np.ndarray
        1D array of sample values for the feature.
    xgboost_cfg : DictConfig
        XGBoost configuration dictionary.
    samplewise : bool, optional
        If True, normalize along sample axis. Default is True.

    Returns
    -------
    tuple
        Updated weights_array and feature_stats dictionary.
    """
    if np.isnan(samples).all():
        nanmean = np.nan
        nanstd = np.nan
    else:
        nanmean = np.nanmean(samples)
        nanstd = np.nanstd(samples)
        weights_array[:, i] = (samples - nanmean) / nanstd
    feature_stats[i] = {"mean": nanmean, "std": nanstd}
    return weights_array, feature_stats


def minmax_norm(
    weights_array: np.ndarray,
    i: int,
    feature_stats: Dict[int, Dict[str, float]],
    samples: np.ndarray,
    xgboost_cfg: DictConfig,
    samplewise: bool = True,
) -> Tuple[np.ndarray, Dict[int, Dict[str, float]]]:
    """
    Apply min-max normalization to scale values to [0, 1] range.

    Parameters
    ----------
    weights_array : np.ndarray
        2D array of weights with shape (n_samples, n_features).
    i : int
        Index of the feature column or sample row to normalize.
    feature_stats : dict
        Dictionary to store normalization statistics (min, max).
    samples : np.ndarray
        1D array of sample values for the feature.
    xgboost_cfg : DictConfig
        XGBoost configuration dictionary.
    samplewise : bool, optional
        If True, normalize along sample axis; otherwise along feature axis.
        Default is True.

    Returns
    -------
    tuple
        Updated weights_array and feature_stats dictionary.
    """
    if np.isnan(samples).all():
        min_val = np.nan
        max_val = np.nan
    else:
        min_val = np.nanmin(samples)
        max_val = np.nanmax(samples)
        if samplewise:
            weights_array[:, i] = (samples - min_val) / (max_val - min_val)
        else:
            weights_array[i, :] = (samples - min_val) / (max_val - min_val)
    feature_stats[i] = {"min": min_val, "max": max_val}
    return weights_array, feature_stats


def fix_nans_in_weights(
    sample_feature_weights: np.ndarray, method: str = "unity"
) -> np.ndarray:
    """
    Replace NaN values in feature weights with imputed values.

    Parameters
    ----------
    sample_feature_weights : np.ndarray
        1D array of feature weights for a single sample.
    method : str, optional
        Imputation method: "mean" replaces NaNs with mean weight,
        "unity" replaces NaNs with 1.0. Default is "unity".

    Returns
    -------
    np.ndarray
        Feature weights with NaN values replaced.

    Raises
    ------
    ValueError
        If method is not "mean" or "unity".
    """
    # no_features = len(sample_feature_weights)
    # Now with timing and AUC features, you do not have any std from the bin
    # quick guestimate is to use the mean feature weight for the sample to impute the nans
    weights_are_nan = pd.isnull(sample_feature_weights)
    if method == "mean":
        weight_mean = np.nanmean(sample_feature_weights)
        sample_feature_weights[weights_are_nan] = weight_mean

    # Or with the AUC, you could say that there is "no uncertainty" in the feature
    # as we don't have information at this point about the uncertainty of pupil size
    # If you start using timing features, you could re-think this or ge timing stdev
    elif method == "unity":
        sample_feature_weights[weights_are_nan] = 1.0
    else:
        logger.error(f"Unknown method {method} for fixing nans in feature weights")
        raise ValueError(f"Unknown method {method} for fixing nans in feature weights")

    return sample_feature_weights


def fix_feature_weights(
    weights_array: np.ndarray, xgboost_cfg: DictConfig
) -> np.ndarray:
    """
    Fix NaN values in all rows of the weights array.

    Parameters
    ----------
    weights_array : np.ndarray
        2D array of weights with shape (n_samples, n_features).
    xgboost_cfg : DictConfig
        XGBoost configuration containing NaN fixing method.

    Returns
    -------
    np.ndarray
        Weights array with NaN values fixed for all samples.
    """
    for sample_idx in range(weights_array.shape[0]):
        weights_array[sample_idx, :] = fix_nans_in_weights(
            sample_feature_weights=weights_array[sample_idx, :],
            method=xgboost_cfg["MODEL"]["WEIGHING"]["weights_nan_weight_fixing"],
        )

    return weights_array


def normalize_to_unity(
    weights_array: np.ndarray,
    i: int,
    feature_stats: Dict[int, Dict[str, float]],
    samples: np.ndarray,
    xgboost_cfg: DictConfig,
    samplewise: bool = True,
) -> Tuple[np.ndarray, Dict[int, Dict[str, float]]]:
    """
    Normalize values to sum to unity and scale by maximum.

    Parameters
    ----------
    weights_array : np.ndarray
        2D array of weights with shape (n_samples, n_features).
    i : int
        Index of the feature column or sample row to normalize.
    feature_stats : dict
        Dictionary to store normalization statistics (sum).
    samples : np.ndarray
        1D array of sample values for the feature.
    xgboost_cfg : DictConfig
        XGBoost configuration dictionary.
    samplewise : bool, optional
        If True, normalize along sample axis; otherwise along feature axis.
        Default is True.

    Returns
    -------
    tuple
        Updated weights_array and feature_stats dictionary.
    """
    if np.isnan(samples).all():
        nansum = np.nan
    else:
        nansum = np.nansum(samples)
        if samplewise:
            weights_array[:, i] = samples / nansum
            max_val = np.nanmax(weights_array[:, i])
            if max_val != 0 and not np.isnan(max_val):
                weights_array[:, i] /= max_val

        else:
            weights_array[i, :] = samples / nansum
            max_val = np.nanmax(weights_array[i, :])
            if max_val != 0 and not np.isnan(max_val):
                weights_array[i, :] /= max_val

        # Both for samplewise (sample weighting) and feature weighting, you need
        # to do feature-wise fixing of NaN weights that come from missing stdevs in features
        weights_array = fix_feature_weights(weights_array, xgboost_cfg)

    feature_stats[i] = {"sum": nansum}

    return weights_array, feature_stats


def norm_wrapper(
    weights_array: np.ndarray,
    xgboost_cfg: DictConfig,
    method: str = "normalize",
    samplewise: bool = True,
) -> Tuple[np.ndarray, Dict[int, Dict[str, float]]]:
    """
    Wrapper to apply normalization to weights array using specified method.

    Parameters
    ----------
    weights_array : np.ndarray
        2D array of weights with shape (n_samples, n_features).
    xgboost_cfg : DictConfig
        XGBoost configuration dictionary.
    method : str, optional
        Normalization method: "zscore", "minmax", or "normalize".
        Default is "normalize".
    samplewise : bool, optional
        If True, normalize along sample axis (per feature);
        otherwise normalize along feature axis (per sample). Default is True.

    Returns
    -------
    tuple
        Normalized weights array (float) and feature_stats dictionary.

    Raises
    ------
    ValueError
        If method is not recognized.
    """
    # Normalize the feature columns
    no_samples, no_features = weights_array.shape
    if samplewise:
        no_items = no_features
    else:
        no_items = no_samples

    feature_stats: Dict[int, Dict[str, float]] = {}
    for i in range(no_items):
        if samplewise:
            # As in now we are normalizing over the samples of one feature
            # Sample weights from here
            samples = weights_array[:, i].astype(float)
            assert samples.shape[0] == no_samples, "Sample shape mismatch"
        else:
            # Now we take subject, and have all the features when you want feature weights
            # i.e. how much more uncertain is one feature compared to another
            samples = weights_array[i, :].astype(float)
            assert samples.shape[0] == no_features, "Feature shape mismatch"

        if method == "zscore":
            weights_array, feature_stats = zscore_norm(
                weights_array,
                i,
                feature_stats,
                samples,
                xgboost_cfg,
                samplewise=samplewise,
            )

        elif method == "minmax":
            weights_array, feature_stats = minmax_norm(
                weights_array,
                i,
                feature_stats,
                samples,
                xgboost_cfg,
                samplewise=samplewise,
            )

        elif method == "normalize":
            weights_array, feature_stats = normalize_to_unity(
                weights_array,
                i,
                feature_stats,
                samples,
                xgboost_cfg,
                samplewise=samplewise,
            )

        else:
            logger.error(f"Normalization method {method} not recognized")
            raise ValueError(f"Normalization method {method} not recognized")

    return weights_array.astype(float), feature_stats


def normalize_mean(
    weights_array: np.ndarray, xgboost_cfg: DictConfig, samplewise: bool = True
) -> Tuple[np.ndarray, Dict[int, Dict[str, float]]]:
    """
    Sample weighing: Z-score mean
        We are interested in how much a given sample deviates from the mean of the feature across all subject
        first normalize the feature column and then take the average of all the features for a given sample
        to approximate "how good of a subject" this sample is.

    """
    weights_array, norm_stats = norm_wrapper(
        weights_array, xgboost_cfg, method="normalize", samplewise=samplewise
    )

    # Do not allow any np.nan weights to be passed to XGBoost
    assert not pd.isnull(weights_array).any(), "NaN weights in the weights array"

    # Take the average of all the features for a given sample
    if samplewise:
        weights = np.nanmean(weights_array, axis=1)
    else:
        weights = np.nanmean(weights_array, axis=0)

    if samplewise:
        assert weights.shape[0] == weights_array.shape[0], (
            "Sample weight shape mismatch"
        )
    else:
        assert len(weights) == weights_array.shape[1], "Feature weight shape mismatch"

    # Should not matter the scaling to XGBoost, and the sample weights don't have to add up to 1 necessarily
    # see e.g. https://stats.stackexchange.com/a/458686/294507
    return weights, norm_stats


def get_1d_sample_weights(
    weights_array: np.ndarray, xgboost_cfg: DictConfig
) -> Tuple[np.ndarray, Dict[int, Dict[str, float]]]:
    """
    Compute 1D sample weights from 2D weights array.

    Parameters
    ----------
    weights_array : np.ndarray
        2D array of weights with shape (n_samples, n_features).
    xgboost_cfg : DictConfig
        XGBoost configuration specifying weight creation method.

    Returns
    -------
    tuple
        1D sample weights array and normalization statistics dictionary.

    Raises
    ------
    ValueError
        If sample weights creation method is not recognized.
    """
    if (
        xgboost_cfg["MODEL"]["WEIGHING"]["weights_sample_creation_method"]
        == "normalize_mean"
    ):
        sample_weight, norm_stats = normalize_mean(weights_array, xgboost_cfg)
    else:
        logger.error(
            "Unknown sample weights creation method, method = {}".format(
                xgboost_cfg["MODEL"]["WEIGHING"]["weights_sample_creation_method"]
            )
        )
        raise ValueError(
            "Unknown sample weights creation method, method = {}".format(
                xgboost_cfg["MODEL"]["WEIGHING"]["weights_sample_creation_method"]
            )
        )

    return sample_weight, norm_stats


def sample_weight_wrapper(
    dict_arrays: Dict[str, np.ndarray], xgboost_cfg: DictConfig
) -> Tuple[
    Optional[np.ndarray],
    Optional[List[np.ndarray]],
    Optional[Dict[int, Dict[str, float]]],
]:
    """
    Compute sample weights for training and evaluation sets.

    Parameters
    ----------
    dict_arrays : dict
        Dictionary containing training, test, and optionally validation
        weight arrays (keys: "x_train_w", "x_test_w", "x_val_w").
    xgboost_cfg : DictConfig
        XGBoost configuration dictionary.

    Returns
    -------
    tuple
        sample_weight : np.ndarray or None
            1D sample weights for training.
        sample_weight_eval_set : list or None
            List of sample weights for each evaluation set.
        sample_stats : dict or None
            Normalization statistics.
    """
    if xgboost_cfg["MODEL"]["WEIGHING"]["weigh_the_samples"]:
        logger.debug("Using sample weighing for XGBoost")
        sample_weight, sample_stats = get_1d_sample_weights(
            weights_array=dict_arrays["x_train_w"], xgboost_cfg=xgboost_cfg
        )
        # Same order now as in for the evaluation set defined above
        sample_weight_eval_train, _ = get_1d_sample_weights(
            dict_arrays["x_train_w"], xgboost_cfg
        )
        sample_weight_eval_test, _ = get_1d_sample_weights(
            dict_arrays["x_test_w"], xgboost_cfg
        )
        if "x_val_w" in dict_arrays:  # when bootstrapping
            sample_weight_eval_val, _ = get_1d_sample_weights(
                dict_arrays["x_val_w"], xgboost_cfg
            )
            sample_weight_eval_set = [
                sample_weight_eval_train,
                sample_weight_eval_test,
                sample_weight_eval_val,
            ]
        else:
            sample_weight_eval_set = [sample_weight_eval_train, sample_weight_eval_test]
        assert len(sample_weight) == dict_arrays["x_train_w"].shape[0], (
            "Sample weight shape mismatch"
        )

    else:
        logger.debug("Skipping the use of sample weighing for XGBoost")
        sample_weight = None
        sample_weight_eval_set = None
        sample_stats = None

    return sample_weight, sample_weight_eval_set, sample_stats


def feature_weight_wrapper(
    dict_arrays: Dict[str, np.ndarray], xgboost_cfg: DictConfig
) -> Tuple[Optional[np.ndarray], Optional[Dict[int, Dict[str, float]]]]:
    """
    Compute feature weights for XGBoost training.

    Parameters
    ----------
    dict_arrays : dict
        Dictionary containing training weight array (key: "x_train_w").
    xgboost_cfg : DictConfig
        XGBoost configuration dictionary.

    Returns
    -------
    tuple
        feature_weight : np.ndarray or None
            1D feature weights array.
        feature_stats : dict or None
            Normalization statistics.
    """
    if xgboost_cfg["MODEL"]["WEIGHING"]["weigh_the_features"]:
        logger.debug("Using feature weighing for XGBoost")
        # TODO! if you want to control the normalization method, you can add it here
        feature_weight, feature_stats = normalize_mean(
            weights_array=dict_arrays["x_train_w"],
            xgboost_cfg=xgboost_cfg,
            samplewise=False,
        )
        assert len(feature_weight) == dict_arrays["x_train_w"].shape[1], (
            "Feature weight shape mismatch"
        )

        return feature_weight, feature_stats
    else:
        logger.debug("Skipping the use of feature weighing for XGBoost")
        return None, None


def class_weight_wrapper(
    dict_arrays: Dict[str, np.ndarray], xgboost_cfg: DictConfig
) -> Tuple[Optional[float], Optional[Dict[str, int]]]:
    """
    Compute class weight for handling class imbalance.

    Parameters
    ----------
    dict_arrays : dict
        Dictionary containing training labels (key: "y_train").
    xgboost_cfg : DictConfig
        XGBoost configuration dictionary.

    Returns
    -------
    tuple
        scale_pos_weight : float or None
            Ratio of negative to positive class samples.
        class_stats : dict or None
            Dictionary with class counts.
    """
    if xgboost_cfg["MODEL"]["WEIGHING"]["weigh_the_classes"]:
        # TODO! if you want to control the normalization method, you can add it here
        class_stats = {
            "sum_0": sum(dict_arrays["y_train"] == 0),
            "sum_1": sum(dict_arrays["y_train"] == 1),
        }
        scale_pos_weight = class_stats["sum_0"] / class_stats["sum_1"]
        return scale_pos_weight, class_stats
    else:
        logger.debug(
            "Skipping the use of class weighing for class imbalance mitigation"
        )
        return None, None


def get_weights_for_xgboost_fit(
    dict_arrays: Dict[str, np.ndarray],
    xgboost_cfg: DictConfig,
    write_to_mlflow: bool = False,
) -> Tuple[
    Optional[np.ndarray],
    Optional[List[np.ndarray]],
    Optional[np.ndarray],
    Optional[float],
    Dict[str, Any],
]:
    """
    Compute all weights (sample, feature, class) for XGBoost training.

    Parameters
    ----------
    dict_arrays : dict
        Dictionary containing feature arrays and labels.
    xgboost_cfg : DictConfig
        XGBoost configuration dictionary.
    write_to_mlflow : bool, optional
        If True, log weights to MLflow. Not implemented. Default is False.

    Returns
    -------
    tuple
        sample_weight : np.ndarray or None
        sample_weight_eval_set : list or None
        feature_weights : np.ndarray or None
        scale_pos_weight : float or None
        norm_stats : dict
            Combined normalization statistics.

    Raises
    ------
    NotImplementedError
        If write_to_mlflow is True.
    """
    # Get the sample weights (or return Nones if not used)
    sample_weight, sample_weight_eval_set, sample_stats = sample_weight_wrapper(
        dict_arrays, xgboost_cfg
    )

    # Same for feature weights
    feature_weights, feature_stats = feature_weight_wrapper(dict_arrays, xgboost_cfg)

    # Same for class weights
    scale_pos_weight, class_stats = class_weight_wrapper(dict_arrays, xgboost_cfg)

    # Stats dict
    norm_stats = {
        "weights_sum_sample": sample_stats,
        "weights_sum_feature": feature_stats,
        "weights_sum_class": class_stats,
    }

    if write_to_mlflow:
        raise NotImplementedError("MLflow logging not implemented yet")
        # # TODO! instead of just dumping the dicts, something nicer could be done?
        # for key, dict in norm_stats.items():
        #     mlflow.log_param(key, dict)

    return (
        sample_weight,
        sample_weight_eval_set,
        feature_weights,
        scale_pos_weight,
        norm_stats,
    )


def return_weights_as_dict(
    dict_arrays: Dict[str, np.ndarray], cls_model_cfg: DictConfig
) -> Dict[str, Any]:
    """
    Compute weights and return as a structured dictionary.

    Parameters
    ----------
    dict_arrays : dict
        Dictionary containing feature arrays and labels.
    cls_model_cfg : DictConfig
        Classifier model configuration dictionary.

    Returns
    -------
    dict
        Dictionary containing:
        - "weights": dict with "samples", "features", "classes" keys
        - "norm_stats": normalization statistics
        - "sample_weight_eval_set": evaluation set weights
    """
    # TODO! rename this and make it general for sklearn API classifiers as nothing
    #  XGBoost-specific should happen here
    # if np.all(np.isnan(dict_arrays["x_train_w"])):
    #     logger.debug("All the weights are 1, no weighing in practice")
    (
        sample_weight,
        sample_weight_eval_set,
        feature_weights,
        scale_pos_weight,
        norm_stats,
    ) = get_weights_for_xgboost_fit(dict_arrays, cls_model_cfg)

    weights_dict = {
        "weights": {
            "samples": sample_weight,
            "features": feature_weights,
            "classes": scale_pos_weight,
        },
        "norm_stats": norm_stats,
        "sample_weight_eval_set": sample_weight_eval_set,
    }

    return weights_dict


def weights_dict_wrapper(
    dict_arrays: Dict[str, np.ndarray], cls_model_cfg: DictConfig
) -> Dict[str, Any]:
    """
    Wrapper to compute all weights and package into a dictionary.

    Parameters
    ----------
    dict_arrays : dict
        Dictionary containing feature arrays and labels.
    cls_model_cfg : DictConfig
        Classifier model configuration dictionary.

    Returns
    -------
    dict
        Dictionary containing:
        - "weights": dict with "samples", "features", "classes" keys
        - "norm_stats": normalization statistics
        - "sample_weight_eval_set": evaluation set weights
    """
    (
        sample_weight,
        sample_weight_eval_set,
        feature_weights,
        scale_pos_weight,
        norm_stats,
    ) = get_weights_for_xgboost_fit(dict_arrays, cls_model_cfg)

    weights_dict = {
        "weights": {
            "samples": sample_weight,
            "features": feature_weights,
            "classes": scale_pos_weight,
        },
        "norm_stats": norm_stats,
        "sample_weight_eval_set": sample_weight_eval_set,
    }

    return weights_dict
