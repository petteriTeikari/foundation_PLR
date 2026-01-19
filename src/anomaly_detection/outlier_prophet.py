from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from omegaconf import DictConfig
from prophet import Prophet
import datetime
import mlflow
from loguru import logger
from scipy import signal

from src.anomaly_detection.anomaly_utils import get_data_for_sklearn_anomaly_models
from src.anomaly_detection.outlier_sklearn import (
    get_outlier_metrics,
    log_outlier_mlflow_artifacts,
)
from src.featurization.feature_utils import get_top1_of_col


def create_ds(X_subj, fps: int = 30):
    # in seconds
    time_vector = np.linspace(0, X_subj.shape[0] - 1, X_subj.shape[0]) / fps

    # add seconds to a dummy date
    a = datetime.datetime(2000, 1, 1, 12, 00, 00)
    ds = [a + datetime.timedelta(0, t) for t in time_vector]

    return ds


def reject_outliers(pred, model_cfg):
    """
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


def pad_input(X_subj):
    X_new = np.zeros((X_subj.shape[0] + 1, 1))
    X_new[0:-1] = X_subj
    X_new[-1] = X_new[-2]
    return X_new


def plot_fitted_model(model, pred):
    _ = model.plot(pred)
    plt.show()


def create_prophet_df(X):
    # Prophet requires the time series to be in a DataFrame
    ds = create_ds(X)
    if X.shape[1] == 1:
        df = pd.DataFrame({"ds": ds, "y": X.flatten()})
    else:
        logger.error("Multiple timeseries, is it possible to use Prophet?")
        # Not multivariate per se, but multiple timeseries (different subjects
        raise NotImplementedError("Multiple timeseries, is it possible to use Prophet?")

    return df


def get_changepoints_from_light(light, df):
    def get_color_onset_offset(light_df: pl.DataFrame, color: str):
        changepoints = []
        light_onset_row = get_top1_of_col(df=light_df, col=color, descending=False)
        # If you want to add some delay (offset), here is your change, e.g. 200 ms delay
        changepoints.append(light_onset_row["ds"].to_numpy()[0])
        light_offset_row = get_top1_of_col(df=light_df, col=color, descending=True)
        changepoints.append(light_offset_row["ds"].to_numpy()[0])
        return changepoints

    def get_light_onsets_and_offsets(light_df: pl.DataFrame):
        changepoints = []
        for color in ["Red", "Blue"]:
            changepoints += get_color_onset_offset(light_df, color)
        assert len(changepoints) == 4
        return changepoints

    def smooth_pupil(x, pupil_signal):
        smooth = signal.savgol_filter(
            pupil_signal,
            53,  # window size used for filtering
            3,  # order of fitted polynomial
        )
        return smooth

    def get_color_df(df_pd, onset, offset, col="ds"):
        df_pd = df_pd[(df_pd[col] >= onset) & (df_pd[col] <= offset)]
        return df_pd

    def get_max_constriction(df: pl.DataFrame):
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


def add_manual_changepoints(auto_changepoints, changepoints, df):
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


def get_prophet_model(df, light: dict, model_cfg: DictConfig):
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
    light: np.ndarray,
    model_cfg: DictConfig,
    model=None,
    plot: bool = False,
):
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


def prophet_per_split(X, y, light, model_cfg: DictConfig, model=None):
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


def prophet_dataset_per_split(X, y, X_test, y_test, light, model_cfg: DictConfig):
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
):
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
        logger.info(
            "Train PROPHET on the train set, and apply subjectwise on the test set"
        )
        pred_mask, no_outliers, pred_masks_test, no_outliers_test, model = (
            prophet_dataset_per_split(X, y, X_test, y_test, light, model_cfg)
        )
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
