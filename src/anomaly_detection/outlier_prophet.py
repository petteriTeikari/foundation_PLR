import datetime
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from omegaconf import DictConfig
from prophet import Prophet
from scipy import signal

from src.anomaly_detection.anomaly_utils import get_data_for_sklearn_anomaly_models
from src.anomaly_detection.outlier_sklearn import (
    get_outlier_metrics,
    log_outlier_mlflow_artifacts,
)
from src.featurization.feature_utils import get_top1_of_col


def create_ds(X_subj: np.ndarray, fps: int = 30) -> List[datetime.datetime]:
    """
    Create datetime series for Prophet from sample indices.

    Converts sample indices to datetime objects based on the sampling rate,
    which Prophet requires for time series modeling.

    Parameters
    ----------
    X_subj : np.ndarray
        Subject data array (used for length).
    fps : int, optional
        Sampling rate in frames per second. Default is 30.

    Returns
    -------
    list
        List of datetime objects representing each timepoint.
    """
    # in seconds
    time_vector = np.linspace(0, X_subj.shape[0] - 1, X_subj.shape[0]) / fps

    # add seconds to a dummy date
    a = datetime.datetime(2000, 1, 1, 12, 00, 00)
    ds = [a + datetime.timedelta(0, t) for t in time_vector]

    return ds


def reject_outliers(
    pred: pd.DataFrame, model_cfg: DictConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify outliers based on Prophet prediction uncertainty.

    Points where the prediction error exceeds a factor of the uncertainty
    interval are flagged as outliers.

    Parameters
    ----------
    pred : pd.DataFrame
        Prophet prediction DataFrame with columns 'y', 'yhat', 'yhat_upper', 'yhat_lower'.
    model_cfg : DictConfig
        Model configuration containing 'uncertainty_factor'.

    Returns
    -------
    tuple
        A tuple containing:
        - y : np.ndarray
            Original values with outliers set to NaN.
        - pred_mask : np.ndarray
            Binary mask (1 = outlier, 0 = inlier).

    References
    ----------
    https://medium.com/@reza.rajabi/outlier-and-anomaly-detection-using-facebook-prophet-in-python-3a83d58b1bdf
    """
    # We calculate the prediction error here and uncertainty
    pred["error"] = pred["y"] - pred["yhat"]
    pred["uncertainty"] = pred["yhat_upper"] - pred["yhat_lower"]

    # We this factor we can identify the outlier or anomaly.
    # This factor can be customized based on the data
    factor = model_cfg["MODEL"]["uncertainty_factor"]
    pred["anomaly"] = pred.apply(
        lambda x: 1 if (np.abs(x["error"]) > factor * x["uncertainty"]) else 0,
        axis=1,
    )
    # no_anomalies = pred['anomaly'].sum()

    # Set the outliers to NaN
    pred.loc[pred["anomaly"] == 1, "y"] = np.nan
    pred_mask = pred["anomaly"]

    # Remove one future timestep
    y = pred["y"].values[:-1]
    pred_mask = pred_mask.values[:-1]

    return y, pred_mask


def pad_input(X_subj: np.ndarray) -> np.ndarray:
    """
    Pad input array by duplicating the last value.

    Parameters
    ----------
    X_subj : np.ndarray
        Input array of shape (n_timepoints, 1).

    Returns
    -------
    np.ndarray
        Padded array of shape (n_timepoints + 1, 1).
    """
    X_new = np.zeros((X_subj.shape[0] + 1, 1))
    X_new[0:-1] = X_subj
    X_new[-1] = X_new[-2]
    return X_new


def plot_fitted_model(model: Prophet, pred: pd.DataFrame) -> None:
    """
    Display Prophet model fit visualization.

    Parameters
    ----------
    model : Prophet
        Fitted Prophet model.
    pred : pd.DataFrame
        Prophet prediction DataFrame.
    """
    _ = model.plot(pred)
    plt.show()


def create_prophet_df(X: np.ndarray) -> pd.DataFrame:
    """
    Create Prophet-compatible DataFrame from time series data.

    Parameters
    ----------
    X : np.ndarray
        Input data array of shape (n_timepoints, 1).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'ds' (datetime) and 'y' (values).

    Raises
    ------
    NotImplementedError
        If X has more than one channel (multivariate not supported).
    """
    # Prophet requires the time series to be in a DataFrame
    ds = create_ds(X)
    if X.shape[1] == 1:
        df = pd.DataFrame({"ds": ds, "y": X.flatten()})
    else:
        logger.error("Multiple timeseries, is it possible to use Prophet?")
        # Not multivariate per se, but multiple timeseries (different subjects
        raise NotImplementedError("Multiple timeseries, is it possible to use Prophet?")

    return df


def get_changepoints_from_light(light: Dict[str, Any], df: pd.DataFrame) -> pd.Series:
    """
    Extract changepoints from light stimulus timing for Prophet.

    Identifies key physiological events (light onsets/offsets and maximum
    constriction points) to use as manual changepoints in Prophet.

    Parameters
    ----------
    light : dict
        Light stimulus data with 'Red' and 'Blue' channels.
    df : pd.DataFrame
        Prophet DataFrame with 'ds' and 'y' columns.

    Returns
    -------
    pd.Series
        Sorted datetime changepoints including light onsets/offsets
        and maximum constriction times.

    References
    ----------
    https://facebook.github.io/prophet/docs/trend_changepoints.html
    https://github.com/facebook/prophet/issues/697
    """

    def get_color_onset_offset(light_df: pl.DataFrame, color: str) -> List[Any]:
        """
        Get onset and offset timestamps for a specific light color.

        Parameters
        ----------
        light_df : pl.DataFrame
            DataFrame containing light stimulus data with 'ds' column.
        color : str
            Light color channel ('Red' or 'Blue').

        Returns
        -------
        list
            Two-element list with onset and offset timestamps.
        """
        changepoints = []
        light_onset_row = get_top1_of_col(df=light_df, col=color, descending=False)
        # If you want to add some delay (offset), here is your change, e.g. 200 ms delay
        changepoints.append(light_onset_row["ds"].to_numpy()[0])
        light_offset_row = get_top1_of_col(df=light_df, col=color, descending=True)
        changepoints.append(light_offset_row["ds"].to_numpy()[0])
        return changepoints

    def get_light_onsets_and_offsets(light_df: pl.DataFrame) -> List[Any]:
        """
        Get all light onset and offset timestamps for Red and Blue channels.

        Parameters
        ----------
        light_df : pl.DataFrame
            DataFrame containing light stimulus data.

        Returns
        -------
        list
            Four-element list with Red onset, Red offset, Blue onset, Blue offset.
        """
        changepoints = []
        for color in ["Red", "Blue"]:
            changepoints += get_color_onset_offset(light_df, color)
        assert len(changepoints) == 4
        return changepoints

    def smooth_pupil(x: np.ndarray, pupil_signal: np.ndarray) -> np.ndarray:
        """
        Apply Savitzky-Golay filter to smooth pupil signal.

        Parameters
        ----------
        x : np.ndarray
            Time vector (unused, kept for interface consistency).
        pupil_signal : np.ndarray
            Raw pupil signal values.

        Returns
        -------
        np.ndarray
            Smoothed pupil signal.
        """
        smooth = signal.savgol_filter(
            pupil_signal,
            53,  # window size used for filtering
            3,  # order of fitted polynomial
        )
        return smooth

    def get_color_df(
        df_pd: pd.DataFrame,
        onset: datetime.datetime,
        offset: datetime.datetime,
        col: str = "ds",
    ) -> pd.DataFrame:
        """
        Filter DataFrame to time window between onset and offset.

        Parameters
        ----------
        df_pd : pd.DataFrame
            DataFrame with datetime column.
        onset : datetime
            Start of time window.
        offset : datetime
            End of time window.
        col : str, optional
            Column name for datetime values. Default is 'ds'.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame containing only rows within the time window.
        """
        df_pd = df_pd[(df_pd[col] >= onset) & (df_pd[col] <= offset)]
        return df_pd

    def get_max_constriction(df: pl.DataFrame) -> List[Any]:
        """
        Find timestamps of maximum pupil constriction for each light color.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with 'ds' and 'y' columns.

        Returns
        -------
        list
            Two-element list with max constriction timestamps for Red and Blue.
        """
        changepoints_out = []
        pupil_signal = df["y"].to_numpy()
        df["y"] = smooth_pupil(x=df["ds"].to_numpy(), pupil_signal=pupil_signal)
        for color in ["Red", "Blue"]:
            changepoints = get_color_onset_offset(light_df, color)
            values_per_color = get_color_df(
                df_pd=df, onset=changepoints[0], offset=changepoints[1]
            )
            min_row = values_per_color[
                values_per_color["y"] == values_per_color["y"].min()
            ]
            if min_row.shape[0] > 1:
                changepoints_out.append(min_row["ds"].to_numpy()[0])
        assert len(changepoints_out) == 2
        return changepoints_out

    # https://facebook.github.io/prophet/docs/trend_changepoints.html
    # https://github.com/facebook/prophet/issues/697
    # These in practice are the timestamps from ds
    light["ds"] = df["ds"]
    light_df = pl.DataFrame(light)
    changepoints_offset_onset = get_light_onsets_and_offsets(light_df)
    changepoints_max_constriction = get_max_constriction(df)

    sorted_list = sorted(changepoints_offset_onset + changepoints_max_constriction)
    changepoints: pd.Series = pd.Series(sorted_list)

    return changepoints


def add_manual_changepoints(
    auto_changepoints: pd.Series, changepoints: List[Any], df: pd.DataFrame
) -> pd.Series:
    """
    Combine automatic and manual changepoints.

    Parameters
    ----------
    auto_changepoints : pd.Series
        Changepoints automatically detected by Prophet.
    changepoints : list
        Manually specified changepoints.
    df : pd.DataFrame
        Prophet DataFrame to match timestamps.

    Returns
    -------
    pd.Series
        Combined unique changepoints.
    """
    # do you need the correct index? anyway picking the ds values from original df
    df_out = pd.DataFrame()
    for changepoint in changepoints:
        # find the matching timestamp from df["ds"]
        row = df[df["ds"] == changepoint]
        df_out = pd.concat([df_out, row])

    # same for the auto changepoints
    for changepoint in auto_changepoints:
        row = df[df["ds"] == changepoint]
        df_out = pd.concat([df_out, row])

    return pd.Series(pd.Series(df_out["ds"]).unique())


def get_prophet_model(
    df: pd.DataFrame, light: Dict[str, Any], model_cfg: DictConfig
) -> Prophet:
    """
    Fit a Prophet model with optional manual changepoints.

    Parameters
    ----------
    df : pd.DataFrame
        Prophet DataFrame with 'ds' and 'y' columns.
    light : dict
        Light stimulus data for manual changepoint extraction.
    model_cfg : DictConfig
        Model configuration with 'changepoint_prior_scale' and
        'manual_changepoints' settings.

    Returns
    -------
    Prophet
        Fitted Prophet model.

    Raises
    ------
    ValueError
        If manual_changepoints method is not recognized.

    Notes
    -----
    If manual_changepoints is 'light', extracts changepoints from light
    stimulus timing and refits the model with combined changepoints.
    """
    # Get the changepoints
    if model_cfg["MODEL"]["manual_changepoints"] is not None:
        if model_cfg["MODEL"]["manual_changepoints"] == "light":
            changepoints = get_changepoints_from_light(light, deepcopy(df))
        else:
            logger.error(
                "Manual changepoints method not recognized, = {}".format(
                    model_cfg["MODEL"]["manual_changepoints"]
                )
            )
            raise ValueError(
                "Manual changepoints method not recognized, = {}".format(
                    model_cfg["MODEL"]["manual_changepoints"]
                )
            )

    # TODO! Could you get a better trend here by creating a trend model from the "pupil_gt" that is pretty
    #  smooth and then use that as a prior for the trend?
    model = Prophet(
        changepoint_prior_scale=model_cfg["MODEL"]["changepoint_prior_scale"]
    ).fit(df)
    auto_changepoints = model.changepoints
    # I guess you just have to deal with the excessive log prints?
    # https://stackoverflow.com/a/76233910/6412152

    if model_cfg["MODEL"]["manual_changepoints"] == "light":
        auto_changepoints = model.changepoints
        # Add the manual changepoints to auto changepoints
        changepoints_new = add_manual_changepoints(auto_changepoints, changepoints, df)

        # Refit the model
        model = Prophet(
            changepoint_prior_scale=model_cfg["MODEL"]["changepoint_prior_scale"],
            changepoints=changepoints_new,
        ).fit(df)

    return model


def prophet_per_X(
    X: np.ndarray,
    y: np.ndarray,
    light: Dict[str, Any],
    model_cfg: DictConfig,
    model: Optional[Prophet] = None,
    plot: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, Prophet]:
    """
    Run Prophet outlier detection on a single time series.

    Parameters
    ----------
    X : np.ndarray
        Input time series of shape (n_timepoints, 1).
    y : np.ndarray
        Ground truth labels (not used in detection, for consistency).
    light : np.ndarray
        Light stimulus data for changepoint extraction.
    model_cfg : DictConfig
        Model configuration.
    model : Prophet, optional
        Pre-fitted model to use. If None, fits a new model. Default is None.
    plot : bool, optional
        Whether to display the fit visualization. Default is False.

    Returns
    -------
    tuple
        A tuple containing:
        - y : np.ndarray
            Original values with outliers set to NaN.
        - pred_mask : np.ndarray
            Binary outlier mask.
        - no_outliers : int
            Number of detected outliers.
        - model : Prophet
            The fitted Prophet model.
    """
    # Create the dataframe
    df = create_prophet_df(X)

    # Fit the model
    if model is None:
        model = get_prophet_model(df, light, model_cfg)

    # predict 1-second in the future
    future = model.make_future_dataframe(periods=1, freq="s")
    pred = model.predict(future)
    pred["y"] = pad_input(X)
    if plot:
        # when you are debugging, and want to visualize
        plot_fitted_model(model, pred)

    # Reject outliers
    # This is in practice quite conservative, and removes clear outliers but keeps a lot of the outliers
    y, pred_mask = reject_outliers(pred, model_cfg)
    no_outliers = np.sum(pred_mask)

    return y, pred_mask, no_outliers, model


def prophet_per_split(
    X: np.ndarray,
    y: np.ndarray,
    light: Dict[str, Any],
    model_cfg: DictConfig,
    model: Optional[Prophet] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Prophet outlier detection on all subjects in a split.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_subjects, n_timepoints).
    y : np.ndarray
        Ground truth labels of shape (n_subjects, n_timepoints).
    light : dict
        Light stimulus data.
    model_cfg : DictConfig
        Model configuration.
    model : Prophet, optional
        Pre-fitted model to use for all subjects. Default is None.

    Returns
    -------
    tuple
        A tuple containing:
        - X_cleaned : np.ndarray
            Cleaned signals with outliers as NaN.
        - pred_masks : np.ndarray
            Binary outlier masks for all subjects.
        - no_outliers_per_subject : np.ndarray
            Number of outliers detected per subject.
    """
    no_subjects = X.shape[0]
    no_outliers_per_subject = np.zeros(no_subjects)
    X_cleaned = None
    pred_masks = None
    for subj_idx in range(no_subjects):
        X_subj = X[subj_idx, :].reshape(-1, 1)
        y_subj = y[subj_idx, :].reshape(-1, 1)
        assert X_subj.shape[0] == y_subj.shape[0]
        X_out, pred_mask, no_outliers, _ = prophet_per_X(
            X=X_subj, y=y_subj, light=light, model_cfg=model_cfg, model=model
        )

        no_outliers_per_subject[subj_idx] = no_outliers
        if X_cleaned is None:
            X_cleaned = X_out
            pred_masks = pred_mask
        else:
            X_cleaned = np.vstack((X_cleaned, X_out))
            pred_masks = np.vstack((pred_masks, pred_mask))

    return X_cleaned, pred_masks, no_outliers_per_subject


def prophet_dataset_per_split(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    light: Dict[str, Any],
    model_cfg: DictConfig,
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, Prophet]:
    """
    Train Prophet on training data and apply to test set.

    Parameters
    ----------
    X : np.ndarray
        Training data.
    y : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test data.
    y_test : np.ndarray
        Test labels.
    light : dict
        Light stimulus data.
    model_cfg : DictConfig
        Model configuration.

    Returns
    -------
    tuple
        A tuple containing:
        - pred_mask : np.ndarray
            Training outlier mask.
        - no_outliers : int
            Number of training outliers.
        - pred_masks_test : np.ndarray
            Test outlier masks.
        - no_outliers_test : np.ndarray
            Number of outliers per test subject.
        - model : Prophet
            The trained Prophet model.
    """
    _, pred_mask, no_outliers, model = prophet_per_X(
        X=X, y=y, light=light, model_cfg=model_cfg
    )
    _, pred_masks_test, no_outliers_test = prophet_per_split(
        X_test, y_test, light, model_cfg, model=model
    )

    return pred_mask, no_outliers, pred_masks_test, no_outliers_test, model


def outlier_prophet_wrapper(
    df: pl.DataFrame,
    cfg: DictConfig,
    model_cfg: DictConfig,
    experiment_name: str,
    run_name: str,
) -> Tuple[Dict[str, Dict[str, Any]], Optional[Prophet]]:
    """
    Run Prophet-based outlier detection on PLR data.

    Uses Facebook Prophet to model the time series trend and identifies
    outliers based on prediction uncertainty intervals.

    Parameters
    ----------
    df : pl.DataFrame
        Input PLR data.
    cfg : DictConfig
        Full Hydra configuration.
    model_cfg : DictConfig
        Prophet model configuration.
    experiment_name : str
        MLflow experiment name.
    run_name : str
        MLflow run name.

    Returns
    -------
    tuple
        A tuple containing:
        - metrics : dict
            Outlier detection metrics for train and test splits.
        - model : Prophet or None
            The trained Prophet model (None for per_subject mode).

    Raises
    ------
    NotImplementedError
        If train_method is 'datasetwise' (not yet implemented).
    ValueError
        If train_method is not recognized.
    """
    train_on = model_cfg["MODEL"]["train_on"]
    X, y, X_test, y_test, light = get_data_for_sklearn_anomaly_models(
        df=df, cfg=cfg, train_on=train_on
    )

    # Get outliers
    if model_cfg["MODEL"]["train_method"] == "per_subject":
        # No training per se, just per subject
        logger.info("Subject-wise PROPHET")
        _, pred_masks_train, no_outliers_train = prophet_per_split(
            X, y, light, model_cfg
        )
        _, pred_masks_test, no_outliers_test = prophet_per_split(
            X_test, y_test, light, model_cfg
        )
        model = None
    elif model_cfg["MODEL"]["train_method"] == "datasetwise":
        raise NotImplementedError("Dataset-wise training not implemented yet")
    else:
        logger.error(
            "Unrecognized train method = {}".format(model_cfg["MODEL"]["train_method"])
        )
        raise ValueError(
            "Unrecognized train method = {}".format(model_cfg["MODEL"]["train_method"])
        )

    # Get metrics
    metrics = {}
    metrics["train"] = get_outlier_metrics(
        None, pred_masks_train, y, df=df, cfg=cfg, split="train"
    )
    metrics["test"] = get_outlier_metrics(
        None, pred_masks_test, y_test, df=df, cfg=cfg, split="test"
    )

    metrics["outlier_train"] = metrics["train"]
    metrics["outlier_test"] = metrics["test"]

    # Log the metrics and the results
    log_outlier_mlflow_artifacts(metrics, model, "PROPHET")
    mlflow.end_run()

    return metrics, model
