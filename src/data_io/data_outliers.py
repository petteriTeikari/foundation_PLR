from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from momentfm.utils.anomaly_detection_metrics import (
    f1_score,
)
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def get_global_stdev_and_envelopes(
    residual: np.ndarray, std_multiplier: float
) -> Tuple[float, float, float]:
    """Calculate global standard deviation and confidence interval envelopes.

    Parameters
    ----------
    residual : np.ndarray
        Residual signal array.
    std_multiplier : float
        Multiplier for standard deviation to define envelope bounds.

    Returns
    -------
    tuple
        Tuple containing (global_stdev, upper_envelope, lower_envelope).
    """
    global_stdev = np.nanstd(residual)
    upper_envelope = global_stdev * std_multiplier
    lower_envelope = -(global_stdev * std_multiplier)
    return global_stdev, upper_envelope, lower_envelope


def fill_with_median(
    array_rolling: np.ndarray, W: int, median_W: int = 5
) -> np.ndarray:
    """Fill NaN values at array edges with median of nearby values.

    Parameters
    ----------
    array_rolling : np.ndarray
        Array with potential NaN values at edges from rolling operations.
    W : int
        Window size used in rolling operation.
    median_W : int, optional
        Number of values to use for median calculation, by default 5.

    Returns
    -------
    np.ndarray
        Array with edge NaN values filled.
    """
    nonnan_idxs = np.argwhere((~np.isnan(array_rolling)))
    first_nonnan_idx = int(nonnan_idxs[0])
    last_nonnan_idx = int(nonnan_idxs[-1])

    start_values = array_rolling[first_nonnan_idx : first_nonnan_idx + median_W]
    start_median_value = np.nanmedian(start_values)
    end_values = array_rolling[last_nonnan_idx - median_W : last_nonnan_idx]
    end_median_value = np.nanmedian(end_values)

    array_rolling[:first_nonnan_idx] = start_median_value
    array_rolling[last_nonnan_idx:] = end_median_value

    return array_rolling


def get_rolling_stdev(
    X_subj: np.ndarray, W: int, fill_na_values_with_median: bool = True
) -> np.ndarray:
    """Calculate rolling standard deviation of a signal.

    Parameters
    ----------
    X_subj : np.ndarray
        Input signal array.
    W : int
        Window size for rolling calculation.
    fill_na_values_with_median : bool, optional
        Whether to fill edge NaN values with median, by default True.

    Returns
    -------
    np.ndarray
        Rolling standard deviation array.
    """
    array_rolling = np.array(pd.Series(X_subj).rolling(W, center=True).std())
    if fill_na_values_with_median:
        array_rolling = fill_with_median(array_rolling, W)
    return array_rolling


def compute_rolling_metrics(
    outliers: np.ndarray, outlier_mask: np.ndarray
) -> Tuple[float, float, float]:
    """Compute outlier detection performance metrics.

    Parameters
    ----------
    outliers : np.ndarray
        Predicted outlier mask (boolean).
    outlier_mask : np.ndarray
        Ground truth outlier mask (boolean).

    Returns
    -------
    tuple
        Tuple containing (fp_ratio, tp_ratio, f1_score).
    """
    tn, fp, fn, tp = confusion_matrix(outlier_mask, outliers).ravel()
    # False negative still okay, as it meeans that we did not get all the difficult ones, but we don't want
    # to remove the inliers (fp to zero desire)
    fp_ratio = float(fp) / len(outlier_mask)
    tp_ratio = float(tp) / len(outlier_mask)
    f1 = f1_score(outliers, outlier_mask)
    return fp_ratio, tp_ratio, f1


def plot_debug_per_subject(
    code: str,
    pupil_trend: np.ndarray,
    residual: np.ndarray,
    outliers_subj: np.ndarray,
    outlier_mask: np.ndarray,
) -> None:
    """Create debug visualization of outlier detection for a single subject.

    Parameters
    ----------
    code : str
        Subject code identifier.
    pupil_trend : np.ndarray
        Ground truth pupil trend.
    residual : np.ndarray
        Residual signal (raw - trend).
    outliers_subj : np.ndarray
        Detected outlier mask.
    outlier_mask : np.ndarray
        Ground truth outlier mask.
    """
    plt.close("all")
    raw_inputed = pupil_trend + residual
    # fp_subj, f1_subj = compute_rolling_metrics(outliers_subj, outlier_mask)
    outliers_found = raw_inputed.copy()
    outliers_found[~outliers_subj] = np.nan
    outliers_masked = raw_inputed.copy()
    outliers_masked[~outlier_mask] = np.nan
    plt.plot(
        raw_inputed,
        "k",
        pupil_trend,
        "g",
        outliers_masked,
        "bo",
        outliers_found,
        "ro",
        markersize=1,
    )
    plt.show()
    plt.savefig(f"outliers_{code}.png", dpi=300)


def find_window_size_for_dataset(
    df_raw: pl.DataFrame,
    std_multiplier: float = 1.96,
    W: int = 100,
    codes: Optional[np.ndarray] = None,
    outlier_mask: Optional[np.ndarray] = None,
    plot_debug: bool = False,
    correct_for_ground_truth: bool = True,
) -> Tuple[float, float, float, np.ndarray, List[int]]:
    """Find outliers using rolling standard deviation method for entire dataset.

    Parameters
    ----------
    df_raw : pl.DataFrame
        Raw data dataframe with pupil signals.
    std_multiplier : float, optional
        Standard deviation multiplier for outlier threshold, by default 1.96.
    W : int, optional
        Window size for rolling calculation, by default 100.
    codes : np.ndarray, optional
        Subject codes array, by default None.
    outlier_mask : np.ndarray, optional
        Ground truth outlier mask, by default None.
    plot_debug : bool, optional
        Whether to create debug plots, by default False.
    correct_for_ground_truth : bool, optional
        Whether to correct predictions using ground truth, by default True.

    Returns
    -------
    tuple
        Tuple containing (fp_ratio, tp_ratio, f1_score, outliers_flat, sum_per_subj).
    """
    subjects = sorted(df_raw["subject_code"].unique().to_numpy())
    sum_per_subj = []
    outliers = []

    for idx, code in enumerate(subjects):
        df_subj = df_raw.filter(df_raw["subject_code"] == subjects[idx])
        assert df_subj.shape[0] == 1981, "The number of rows must be 1981"

        pupil_trend = df_subj["pupil_gt"].to_numpy()
        residual = df_subj["pupil_orig_imputed"].to_numpy() - pupil_trend
        assert ~np.all(np.isnan(residual)), "All values are NaN for code = {}".format(
            codes[idx]
        )

        # Set the values from "easy" outliers to Nan
        residual[df_subj["outlier_mask_easy"]] = np.nan

        # Get rolling stdev
        rolling_stdev = get_rolling_stdev(X_subj=residual, W=W)

        # The CI envelopes
        upper_envelope = rolling_stdev * std_multiplier
        lower_envelope = -(rolling_stdev * std_multiplier)

        # Check if input data outside the CI
        outliers_subj = (residual > upper_envelope) | (residual < lower_envelope)
        sum_per_subj.append(np.sum(outliers_subj))
        outliers.append(outliers_subj)

        if plot_debug:
            plot_debug_per_subject(
                code,
                pupil_trend,
                residual,
                outliers_subj,
                outlier_mask=df_subj["outlier_mask"].to_numpy(),
            )

    # Compute the metrics
    outliers_flat = np.array(outliers).flatten()
    if correct_for_ground_truth:
        outliers_flat = outliers_flat & df_raw["outlier_mask"].to_numpy()
    fp, tp, f1 = compute_rolling_metrics(
        outliers=outliers_flat, outlier_mask=df_raw["outlier_mask"].to_numpy()
    )

    return fp, tp, f1, outliers_flat, sum_per_subj


def subjectwise_rolling_stdev(
    df_raw: pl.DataFrame, window_size: int, std_multiplier: float, codes: np.ndarray
) -> np.ndarray:
    """Find best window size for rolling stdev outlier detection.

    Searches over a range of window sizes to find the one with best F1 score.

    Parameters
    ----------
    df_raw : pl.DataFrame
        Raw data dataframe.
    window_size : int
        Initial window size (not used, searches range instead).
    std_multiplier : float
        Standard deviation multiplier for thresholds.
    codes : np.ndarray
        Subject code identifiers.

    Returns
    -------
    np.ndarray
        Boolean outlier mask using best window size.
    """
    w_sizes = np.linspace(3, 30, 28).astype(int)
    fps = []
    tps = []
    f1s = []
    outlier_sums = []

    best_f1 = 0.0
    for W in tqdm(w_sizes, total=len(w_sizes), desc="Finding best window size"):
        fp, tp, f1, outliers, sum_per_subj = find_window_size_for_dataset(
            df_raw=df_raw, std_multiplier=std_multiplier, W=W, codes=codes
        )
        fps.append(fp)
        tps.append(tp)
        f1s.append(f1)
        outlier_sums.append(sum_per_subj)
        if f1 > best_f1:
            best_f1 = f1
            best_stats = {"fp": fp, "f1": f1, "tp": tp, "W": W}
            best_outliers = outliers.copy()

    stats_dict = {"fp": np.array(fps), "f1": np.array(f1s), "W": w_sizes}

    no_of_outliers = np.sum(best_outliers)
    outlier_percentage = 100 * (no_of_outliers / len(best_outliers))
    logger.info(
        f"Best window size = {best_stats['W']}, "
        f"{no_of_outliers} marked as medium difficulty ({outlier_percentage:.2f}%)"
    )
    logger.info(f"Best stats: {best_stats}")

    logger.debug("The whole experiment:")
    for key, value in stats_dict.items():
        logger.debug(f"{key} = {value}")

    return best_outliers


def check_for_nan_subjects(X: np.ndarray, codes: Union[List[str], np.ndarray]) -> None:
    """Check for subjects with all NaN values and log warnings.

    Parameters
    ----------
    X : np.ndarray
        Data array of shape (n_subjects, n_timepoints).
    codes : list or np.ndarray
        Subject code identifiers.

    Raises
    ------
    AssertionError
        If X and codes have different lengths.
    """
    assert X.shape[0] == len(codes), "X and codes must have the same length"
    for idx in range(X.shape[0]):
        X_subj = X[idx, :]
        if np.all(np.isnan(X_subj)):
            logger.warning("All values are NaN for code = {}".format(codes[idx]))


def debug_global_outlier(
    outliers_init: np.ndarray,
    outliers: np.ndarray,
    pupil_trend: np.ndarray,
    residual: np.ndarray,
    outlier_mask: np.ndarray,
    idxs: Tuple[int, int] = (16000, 22000),
    use_indices: bool = True,
) -> None:
    """Create debug visualization of global outlier detection.

    Parameters
    ----------
    outliers_init : np.ndarray
        Initial outlier predictions before ground truth correction.
    outliers : np.ndarray
        Final outlier predictions after correction.
    pupil_trend : np.ndarray
        Ground truth pupil trend.
    residual : np.ndarray
        Residual signal.
    outlier_mask : np.ndarray
        Ground truth outlier mask.
    idxs : tuple, optional
        Index range for plotting, by default (16000, 22000).
    use_indices : bool, optional
        Whether to use index range or plot all, by default True.
    """
    plt.close("all")
    raw_imputed = pupil_trend + residual
    init_outliers = raw_imputed.copy()
    init_outliers[~outliers_init] = np.nan
    outliers_out = raw_imputed.copy()
    outliers_out[~outliers] = np.nan
    outliers_masked = raw_imputed.copy()
    outliers_masked[~outlier_mask] = np.nan

    if use_indices:
        plt.plot(
            raw_imputed[idxs[0] : idxs[1]],
            "k",
            outliers_masked[idxs[0] : idxs[1]],
            "go",
            outliers_out[idxs[0] : idxs[1]],
            "bo",
            init_outliers[idxs[0] : idxs[1]],
            "ro",
            markersize=1,
        )
    else:
        plt.plot(
            raw_imputed,
            "k",
            outliers_masked,
            "go",
            outliers_out,
            "bo",
            init_outliers,
            "ro",
            markersize=1,
        )

    plt.savefig("outliers_global.png", dpi=300)


def reject_outliers(
    df_raw: pl.DataFrame,
    std_multiplier: float = 1.96,
    window_size: Optional[int] = None,
    method: str = "global",
    codes: Optional[np.ndarray] = None,
    plot_on: bool = False,
) -> np.ndarray:
    """Detect outliers using global or rolling standard deviation method.

    Parameters
    ----------
    df_raw : pl.DataFrame
        Raw data dataframe with pupil signals.
    std_multiplier : float, optional
        Standard deviation multiplier for threshold, by default 1.96.
    window_size : int, optional
        Window size for rolling method, by default None.
    method : str, optional
        Detection method ("global" or "rolling"), by default "global".
    codes : np.ndarray, optional
        Subject code identifiers, by default None.
    plot_on : bool, optional
        Whether to create debug plots, by default False.

    Returns
    -------
    np.ndarray
        Boolean outlier mask.

    Raises
    ------
    ValueError
        If method is unknown.
    """
    if method == "global":
        pupil_trend = df_raw["pupil_gt"].to_numpy()
        residual = df_raw["pupil_orig_imputed"].to_numpy() - pupil_trend
        if plot_on:
            plt.close("all")
            plt.plot(df_raw["pupil_orig"].to_numpy()[210000:220000])
            plt.savefig("pupil_orig.png", dpi=300)
        global_stdev, upper_envelope, lower_envelope = get_global_stdev_and_envelopes(
            residual, std_multiplier
        )
        outliers_init = (residual > upper_envelope) | (residual < lower_envelope)
        no_outliers_init = np.sum(outliers_init)
        outlier_init_percentage = 100 * (no_outliers_init / len(outliers_init))
        logger.info(
            f"With method = {method}, initially {no_outliers_init} marked as easy (blinks) ({outlier_init_percentage:.2f}%)"
        )
        # correct with the ground truth, so that you did not remove any inliers
        outliers = outliers_init & df_raw["outlier_mask"].to_numpy()
        no_outliers = np.sum(outliers)
        outlier_percentage = 100 * (no_outliers / len(outliers))
        logger.info(
            f"With method = {method}, after GT correction {no_outliers} marked as easy (blinks) ({outlier_percentage:.2f}%)"
        )
        if plot_on:
            debug_global_outlier(
                outliers_init,
                outliers,
                pupil_trend,
                residual,
                outlier_mask=df_raw["outlier_mask"].to_numpy(),
            )

    elif method == "rolling":
        # doing this subject-by-subject as outliers can have significant inter-subject variation
        outliers = subjectwise_rolling_stdev(df_raw, window_size, std_multiplier, codes)
    else:
        logger.error("Unknown method = {}".format(method))
        raise ValueError("Unknown method = {}".format(method))

    return outliers


def granularize_outlier_labels(
    df_raw: pl.DataFrame, cfg: Union[DictConfig, Dict[str, Any]]
) -> pl.DataFrame:
    """Split outlier labels into easy (blinks) and medium difficulty categories.

    Uses global method for easy outliers and rolling method for medium
    difficulty outliers that are closer to the true signal.

    Parameters
    ----------
    df_raw : pl.DataFrame
        Raw data dataframe with pupil signals and outlier_mask.
    cfg : DictConfig
        Configuration with granular_outliers_stdev_factor and
        granular_outlier_window_size settings.

    Returns
    -------
    pl.DataFrame
        Dataframe with added outlier_mask_easy and outlier_mask_medium columns.
    """
    # The most conservative rejection
    outliers = reject_outliers(
        df_raw=df_raw,
        std_multiplier=cfg["DATA"]["granular_outliers_stdev_factor"],
        method="global",
        codes=np.unique(df_raw["subject_code"].to_numpy()),
    )

    # This would probably require some manual proofreading in order to be so useful :S
    df_raw = df_raw.with_columns(pl.Series(name="outlier_mask_easy", values=outliers))

    # Slightly more aggressive for the medium difficulty outliers
    outliers = reject_outliers(
        df_raw=df_raw,
        std_multiplier=cfg["DATA"]["granular_outliers_stdev_factor"],
        window_size=cfg["DATA"]["granular_outlier_window_size"],
        method="rolling",
        codes=np.unique(df_raw["subject_code"].to_numpy()),
    )

    df_raw = df_raw.with_columns(pl.Series(name="outlier_mask_medium", values=outliers))

    granular_masks = (
        df_raw["outlier_mask_easy"] | df_raw["outlier_mask_medium"]
    ).to_numpy()
    no_masked = np.sum(granular_masks)
    mask_percentage = 100 * (no_masked / len(granular_masks))
    # Granularized outliers: 26093 marked as easy or medium (2.60%)
    logger.info(
        f"Granularized outliers: {no_masked} marked as easy or medium ({mask_percentage:.2f}%)"
    )
    outlier_percentage = (
        100 * (np.sum(df_raw["outlier_mask"].to_numpy())) / len(granular_masks)
    )
    # The hard outliers that are close to the signal: 5.07%
    logger.info(
        f"The hard outliers that are close to the signal: "
        f"{outlier_percentage - mask_percentage:.2f}% of the total outliers labeled then"
    )

    return df_raw
