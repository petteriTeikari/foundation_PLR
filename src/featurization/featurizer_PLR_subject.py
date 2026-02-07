from copy import deepcopy
from typing import Any, Optional

import numpy as np
import polars as pl
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import auc

from src.featurization.feature_utils import (
    convert_relative_timing_to_absolute_timing,
    get_feature_samples,
    get_top1_of_col,
)


def nan_auc(x: np.ndarray, y: np.ndarray, method: str = "") -> float:
    """Compute AUC while handling NaN values.

    Parameters
    ----------
    x : array-like
        X values for AUC computation.
    y : array-like
        Y values for AUC computation.
    method : str, optional
        Method for handling NaNs, by default ''.

    Returns
    -------
    float
        Computed AUC score.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.
    """
    raise NotImplementedError
    # if method == "imputation":
    #     data = pd.DataFrame({"x": x, "y": y})
    #     # https://medium.com/@datasciencewizards/preprocessing-and-data-exploration-for-time-series-handling-missing-values-e5c507f6c71c
    #     # result = seasonal_decompose(data["y"], model="additive", period=7)
    # else:
    #     logger.error("Unknown method for handling NaNs in AUC computation")
    #     raise NotImplementedError("Unknown method for handling NaNs in AUC computation")
    #
    # return auc_score


def compute_AUC(y: np.ndarray, fps: int = 30, return_abs_AUC: bool = False) -> float:
    """Compute area under the curve for a time series.

    Parameters
    ----------
    y : array-like
        Y values (e.g., pupil size measurements).
    fps : int, optional
        Frames per second for time axis calculation, by default 30.
    return_abs_AUC : bool, optional
        If True, return absolute value of AUC, by default False.

    Returns
    -------
    float
        AUC value, or NaN if y contains NaN values.
    """
    x = np.linspace(0, len(y) - 1, len(y)) / fps
    if np.any(np.isnan(y)):
        # what to do with missing values?
        # the Raw PLR contains missing values giving NaNs in the AUC vanilla auc does not handle NaNs
        auc_score = np.nan  # nan_auc(x, y)
        if return_abs_AUC:
            return abs(auc_score)
    else:
        if return_abs_AUC:
            return abs(auc(x, y))
        else:
            return auc(x, y)


def compute_feature(
    feature_samples: pl.DataFrame,
    feature: str,
    feature_params: dict[str, Any],
    feature_col: str = "imputation_mean",
) -> dict[str, Any]:
    """Compute a single feature from sampled time points.

    Dispatches to amplitude or timing feature computation based on
    feature_params['measure'].

    Parameters
    ----------
    feature_samples : pl.DataFrame
        Dataframe with time points within the feature window.
    feature : str
        Feature name being computed.
    feature_params : dict
        Feature parameters including 'measure' and 'stat'.
    feature_col : str, optional
        Column name for feature values, by default 'imputation_mean'.

    Returns
    -------
    dict
        Feature dictionary with 'value', 'std', 'ci_pos', and 'ci_neg'.

    Raises
    ------
    NotImplementedError
        If feature measure type is not 'amplitude' or 'timing'.
    """

    def get_amplitude_feature(
        feature_samples: pl.DataFrame,
        feature: str,
        feature_params: dict[str, Any],
        feature_col: str,
    ) -> dict[str, Any]:
        y = feature_samples[feature_col].to_numpy()
        if "CI_pos" not in feature_samples.columns:
            logger.error("No confidence interval columns found in the feature samples")
            logger.error("Returning None for the confidence intervals")
            ci_pos = None
            ci_neg = None
        else:
            # Atm no imputation method actually estimates the CI, so we just return None
            # TODO! ensemble imputation would have this
            ci_pos = None
            ci_neg = None
            if np.isnan(feature_samples["CI_pos"].to_numpy()).all():
                ci_pos = None
            if np.isnan(feature_samples["CI_neg"].to_numpy()).all():
                ci_neg = None

        if feature_params["stat"] == "min":
            feature_dict = {
                "value": np.nanmin(y),
                "std": np.nanstd(y),
                "ci_pos": ci_pos,
                "ci_neg": ci_neg,
            }
        elif feature_params["stat"] == "max":
            feature_dict = {
                "value": np.nanmin(y),
                "std": np.nanstd(y),
                "ci_pos": ci_pos,
                "ci_neg": ci_neg,
            }
        elif feature_params["stat"] == "mean":
            feature_dict = {
                "value": np.nanmean(y),
                "std": np.nanstd(y),
                "ci_pos": ci_pos,
                "ci_neg": ci_neg,
            }
        elif feature_params["stat"] == "median":
            feature_dict = {
                "value": np.nanmedian(y),
                "std": np.nanstd(y),
                "ci_pos": ci_pos,
                "ci_neg": ci_neg,
            }
        elif feature_params["stat"] == "AUC":
            feature_dict = {
                "value": compute_AUC(y),
                "std": None,
                "ci_pos": ci_pos,
                "ci_neg": ci_neg,
            }
        else:
            logger.error("Unknown feature stat: {}".format(feature_params["stat"]))
            raise NotImplementedError(
                "Unknown feature stat: {}".format(feature_params["stat"])
            )

        return feature_dict

    def get_timing_feature(
        feature_samples: pl.DataFrame,
        feature: str,
        feature_params: dict[str, Any],
        feature_col: str,
    ) -> dict[str, Any]:
        t0 = feature_samples[0, "time"]
        min_time = get_top1_of_col(feature_samples, feature_col, descending=False).item(
            0, "time"
        )
        return {
            # TODO! Check why is this "value" is None?
            "value": min_time - t0,
            # TODO! This obviously is not None, as we have discretized frame rate, latency estimate not so great
            #  you can always think of some latency tricks as well if you feel like it?
            #  See e.g. Bergamin et al. (2003): "Latency of the Pupil Light Reflex:
            #  Sample Rate, Stimulus Intensity, and Variation in Normal Subjects" (Savitzky-Golay filter)
            #  https://doi.org/10.1167/iovs.02-0468
            #  And obviously, you could train a Neural ODE or something to reconstruct nowadays the PLR for
            #   higher temporal resolution to get better "synthetic" temporal resolution?
            "std": None,
            "ci_pos": None,
            "ci_neg": None,
        }

    if feature_params["measure"] == "amplitude":
        feature_dict = get_amplitude_feature(
            feature_samples, feature, feature_params, feature_col
        )
    elif feature_params["measure"] == "timing":
        feature_dict = get_timing_feature(
            feature_samples, feature, feature_params, feature_col
        )
    else:
        logger.error("Unknown feature measure: {}".format(feature_params["measure"]))
        raise NotImplementedError(
            "Unknown feature measure: {}".format(feature_params["measure"])
        )

    return feature_dict


def get_individual_feature(
    df_subject: pl.DataFrame,
    light_timing: dict[str, Any],
    feature_cfg: DictConfig,
    color: str,
    feature: str,
    feature_params: dict[str, Any],
    feature_col: str = "mean",
) -> Optional[dict[str, Any]]:
    """Extract a single feature for a subject at a specific light color.

    Converts relative timing to absolute, extracts samples within the
    time window, and computes the feature value.

    Parameters
    ----------
    df_subject : pl.DataFrame
        Subject dataframe with time series data.
    light_timing : dict
        Light timing information with onset/offset times.
    feature_cfg : DictConfig
        Feature configuration.
    color : str
        Light color ('Red' or 'Blue').
    feature : str
        Feature name to compute.
    feature_params : dict
        Feature parameters with timing and statistic info.
    feature_col : str, optional
        Column name for feature values, by default 'mean'.

    Returns
    -------
    dict or None
        Feature dictionary with value, std, and CI, or None on error.

    Raises
    ------
    Exception
        Re-raised if error occurs during feature extraction.
    """
    # subject_code = df_subject["subject_code"].to_numpy()[0]
    # Get the absolute timing from the recording
    feature_params_abs = convert_relative_timing_to_absolute_timing(
        light_timing, feature_params, color, feature, feature_cfg
    )

    # Get the time points within the bin
    try:
        feature_samples = get_feature_samples(
            df_subject, feature_params_abs, feature=feature
        )
        # When you have the samples, compute the feature (amplitude or timing) with desired stat (min, max, mean, median)
        feature_dict = compute_feature(
            feature_samples, feature, feature_params_abs, feature_col
        )
    except Exception as e:
        logger.error(
            "Error when getting the feature samples for feature {} and color {}: {}".format(
                feature, color, e
            )
        )
        raise e

    return feature_dict


def get_features_per_color(
    df_subject: pl.DataFrame,
    light_timing: dict[str, Any],
    bin_cfg: DictConfig,
    color: str,
    feature_col: str,
) -> dict[str, Optional[dict[str, Any]]]:
    """Compute all configured features for a specific light color.

    Parameters
    ----------
    df_subject : pl.DataFrame
        Subject dataframe with time series data.
    light_timing : dict
        Light timing information with onset/offset times.
    bin_cfg : DictConfig
        Configuration defining which features to compute.
    color : str
        Light color ('Red' or 'Blue').
    feature_col : str
        Column name for feature values.

    Returns
    -------
    dict
        Dictionary keyed by feature name containing feature dictionaries.
    """
    features = {}
    for feature in bin_cfg.keys():
        features[feature] = get_individual_feature(
            df_subject,
            light_timing,
            bin_cfg,
            color,
            feature,
            feature_params=deepcopy(bin_cfg[feature]),
            feature_col=feature_col,
        )

    return features


def check_that_features_are_not_the_same_for_colors(
    features: dict[str, dict[str, dict[str, Any]]],
) -> None:
    """Validate that red and blue light features are different.

    Ensures that features computed for different light colors are not
    identical, which would indicate a data processing error.

    Parameters
    ----------
    features : dict
        Dictionary keyed by color containing feature dictionaries.

    Raises
    ------
    ValueError
        If feature values are identical for both colors.
    """

    def compare_lists(list1: list[Any], list2: list[Any]) -> bool:
        return all([list1[i] != list2[i] for i in range(len(list1))])

    vals = {}
    colors = list(features.keys())
    for color in colors:
        vals[color] = []
        for feature in features[color].keys():
            vals[color].append(features[color][feature]["value"])

    lists_ok = compare_lists(colors[0], colors[1])
    if not lists_ok:
        logger.error("The feature values for the colors are the same!")
        logger.error("Unlikely that this would happen without a glitch in the data?")
        raise ValueError("The feature values for the colors are the same!")
