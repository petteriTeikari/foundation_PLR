import warnings
from copy import deepcopy
from typing import Any

import numpy as np
from loguru import logger
from omegaconf import DictConfig
from scipy.interpolate import interp1d

from src.classification.catboost.catboost_ensemble import ensemble_uncertainties
from src.classification.catboost.catboost_utils import (
    get_catboost_preds_from_results_for_bootstrap,
)
from src.classification.tabm.tabm_utils import get_tabm_preds_from_results_for_bootstrap
from src.stats._defaults import DEFAULT_CI_LEVEL
from src.stats.calibration_metrics import get_calibration_metrics
from src.stats.classifier_metrics import get_classifier_metrics
from src.stats.uncertainty_quantification import (
    uncertainty_wrapper,
    uncertainty_wrapper_from_subject_codes,
)


def interpolation_wrapper(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray,
    n_samples: int,
    metric: str,
    _kind: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate metric curves to a fixed number of points.

    Parameters
    ----------
    x : array-like
        Original x values.
    y : array-like
        Original y values.
    x_new : array-like
        New x values to interpolate to.
    n_samples : int
        Number of samples for interpolation.
    metric : str
        Metric name ('AUROC', 'AUPR', 'calibration_curve').
    kind : str, default 'linear'
        Interpolation method.

    Returns
    -------
    tuple
        (x_new, y_new) interpolated arrays.
    """

    def clip_illegal_calibration_values(y_new: np.ndarray) -> np.ndarray:
        y_new[y_new < 0] = 0
        y_new[y_new > 1] = 1
        return y_new

    warnings.simplefilter("ignore")
    assert len(x) == len(y), "x and y must have the same length"
    if metric == "calibration_curve":
        # so few points and does not behave well
        x_new = np.linspace(0, 1, 10)
        f = interp1d(x, y, kind="linear", fill_value="extrapolate")
        y_new = f(x_new)
        y_new = clip_illegal_calibration_values(y_new)

    else:
        if len(x) > 1:
            f = interp1d(x, y, kind="linear")
            y_new = f(x_new)
        else:
            # you got only one value, not a vector
            y_new = np.repeat(y, n_samples)
    warnings.resetwarnings()

    return x_new, y_new


def bootstrap_get_array_axis_names(metric: str) -> tuple[str, str]:
    """
    Get x and y axis names for a given metric curve.

    Parameters
    ----------
    metric : str
        Metric name ('AUROC', 'AUPR', 'calibration_curve').

    Returns
    -------
    tuple
        (x_name, y_name) axis names for the metric.

    Raises
    ------
    ValueError
        If unknown metric specified.
    """
    if metric == "AUROC":
        x_name = "fpr"
        y_name = "tpr"
    elif metric == "AUPR":
        x_name = "recall"
        y_name = "precision"
    elif metric == "calibration_curve":
        # https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
        x_name = "prob_pred"
        y_name = "prob_true"
    else:
        logger.error(
            f"Unknown metric: {metric} (only AUROC, AUPR and calibration curve supported)"
        )
        raise ValueError

    return x_name, y_name


def bootstrap_interpolate_metric_arrays(
    arrays: dict[str, Any], n_samples: int = 200
) -> dict[str, Any]:
    """
    Interpolate all metric arrays to a fixed number of samples.

    Enables aggregation of ROC/PR curves across bootstrap iterations
    by standardizing the x-axis.

    Parameters
    ----------
    arrays : dict
        Dictionary of metric arrays with varying lengths.
    n_samples : int, default 200
        Number of points for interpolation.

    Returns
    -------
    dict
        Dictionary of interpolated metric arrays.
    """
    arrays_out = {}
    for metric in arrays.keys():
        arrays_out[metric] = {}
        x_name, y_name = bootstrap_get_array_axis_names(metric)

        # Interpolate (actual metric)
        arrays_out[metric][x_name], arrays_out[metric][y_name] = interpolation_wrapper(
            x=arrays[metric][x_name],
            y=arrays[metric][y_name],
            x_new=np.linspace(
                arrays[metric][x_name][0], arrays[metric][y_name][-1], n_samples
            ),
            n_samples=n_samples,
            metric=metric,
        )

        # Interpolate (thresholds as well)
        if "thresholds" in arrays[metric]:
            # exist for AUROC and AUPR
            _, arrays_out[metric]["thresholds"] = interpolation_wrapper(
                x=np.linspace(0, 1, len(arrays[metric]["thresholds"])),
                y=arrays[metric]["thresholds"],
                x_new=np.linspace(0, 1, n_samples),
                n_samples=n_samples,
                metric="thresholds",
            )

    return arrays_out


def bootstrap_aggregate_arrays(
    arrays: dict[str, Any], metrics_per_split: dict[str, Any], main_key: str = "metrics"
) -> dict[str, Any]:
    """
    Aggregate array metrics across bootstrap iterations.

    Stacks interpolated curves (ROC, PR, calibration) horizontally
    for later statistical analysis.

    Parameters
    ----------
    arrays : dict
        Interpolated metric arrays from current iteration.
    metrics_per_split : dict
        Accumulated metrics from previous iterations.
    main_key : str, default "metrics"
        Key for storing metrics in output dict.

    Returns
    -------
    dict
        Updated metrics_per_split with new arrays appended.
    """
    # For first bootstrap iteration, initialize the metrics_per_split dict
    if main_key not in metrics_per_split.keys():
        metrics_per_split[main_key] = {}
    if "arrays" not in metrics_per_split[main_key].keys():
        metrics_per_split[main_key]["arrays"] = {}

    # Go through all the scalar metrics and aggregate them, flexible, so you can add new metrics without changing this
    for metric in arrays.keys():
        if metric not in metrics_per_split[main_key]["arrays"].keys():
            # For first iteration
            metrics_per_split[main_key]["arrays"][metric] = {}
        values = arrays[metric]
        for variable in values.keys():
            array_var: np.ndarray = values[variable][:, np.newaxis]  # e,g, (200, 1)
            if variable not in metrics_per_split[main_key]["arrays"][metric].keys():
                # For first iteration, (curve_length, no_iterations)
                # print(1, ' ', metric, variable, array_var.shape)
                metrics_per_split[main_key]["arrays"][metric][variable] = array_var
            else:
                # When there is something already here
                # print(2, " ", metric, variable, array_var.shape)
                try:
                    metrics_per_split[main_key]["arrays"][metric][variable] = np.hstack(
                        (
                            metrics_per_split[main_key]["arrays"][metric][variable],
                            array_var,
                        )
                    )
                except Exception as e:
                    logger.error(f"Could not stack arrays: {e}")
                    logger.error(
                        f"previous shape: "
                        f"{metrics_per_split[main_key]['arrays'][metric][variable].shape[0]}, "
                        f"and current shape: "
                        f"{array_var.shape}"
                    )
                    raise e

    return metrics_per_split


def bootstrap_aggregate_scalars(
    metrics_dict: dict[str, Any], metrics_per_split: dict[str, Any]
) -> dict[str, Any]:
    """
    Aggregate scalar metrics across bootstrap iterations.

    Appends scalar values (AUROC, Brier, etc.) to lists for later
    statistical analysis.

    Parameters
    ----------
    metrics_dict : dict
        Metrics from current iteration.
    metrics_per_split : dict
        Accumulated metrics from previous iterations.

    Returns
    -------
    dict
        Updated metrics_per_split with new scalars appended.
    """
    # For first bootstrap iteration, initialize the metrics_per_split dict
    if "metrics" not in metrics_per_split.keys():
        metrics_per_split["metrics"] = {}
    if "scalars" not in metrics_per_split["metrics"].keys():
        metrics_per_split["metrics"]["scalars"] = {}

    # Go through all the scalar metrics and aggregate them, flexible, so you can add new metrics without changing this
    for metric in metrics_dict["metrics"]["scalars"].keys():
        if metric not in metrics_per_split["metrics"]["scalars"].keys():
            # For first iteration
            metrics_per_split["metrics"]["scalars"][metric] = [
                metrics_dict["metrics"]["scalars"][metric]
            ]
        else:
            # When there is something already here
            metrics_per_split["metrics"]["scalars"][metric].append(
                metrics_dict["metrics"]["scalars"][metric]
            )

    return metrics_per_split


def bootstrap_aggregate_by_subject_per_split(
    arrays: dict[str, Any],
    metrics_per_split: dict[str, Any],
    codes_per_split: np.ndarray,
    main_key: str,
    subkey: str = "predictions",
    is_init_with_correct_codes: bool = False,
) -> dict[str, Any]:
    """
    Aggregate predictions by subject code across bootstrap iterations.

    Used for train/val splits where subjects vary between iterations
    due to resampling. Stores predictions keyed by subject code.

    Parameters
    ----------
    arrays : dict
        Predictions from current iteration.
    metrics_per_split : dict
        Accumulated predictions from previous iterations.
    codes_per_split : np.ndarray
        Subject codes for current iteration.
    main_key : str
        Key for storing predictions.
    subkey : str, default "predictions"
        Subkey within arrays.
    is_init_with_correct_codes : bool, default False
        If True, expects codes to already exist.

    Returns
    -------
    dict
        Updated metrics_per_split with predictions aggregated by subject.
    """
    # For first bootstrap iteration, initialize the metrics_per_split dict
    if main_key not in metrics_per_split.keys():
        metrics_per_split[main_key] = {}
    if "arrays" not in metrics_per_split[main_key].keys():
        metrics_per_split[main_key]["arrays"] = {}
    code_warnings = []

    # Aggregate in a dict based on the subject code. Remember that in the bootstrap resample, you
    # get the same code multiple times in each iteration, so you get less unique codes per iteration
    # than you have samples (if you are counting the number of codes added per iteration and are confused)
    for metric in arrays[subkey].keys():
        if metric not in metrics_per_split[main_key]["arrays"].keys():
            metrics_per_split[main_key]["arrays"][metric] = {}
        values: np.ndarray = arrays[subkey][metric]
        assert len(values) == len(
            codes_per_split
        ), "Values and codes must have the same length"

        for i, code in enumerate(codes_per_split):
            value = values[i]
            if code not in metrics_per_split[main_key]["arrays"][metric].keys():
                # When this code has not yet been added
                if is_init_with_correct_codes:
                    # if you are trying to add a code that is already there, you are doing something wrong
                    logger.debug(
                        "You are trying to add a code that is not in the split?, code = {}".format(
                            code
                        )
                    )
                    code_warnings.append(code)
                else:
                    # if you have an empty the dict, then you can append
                    metrics_per_split[main_key]["arrays"][metric][code] = [value]
            else:
                # Just keep on appending then over the iterations
                metrics_per_split[main_key]["arrays"][metric][code].append(value)

    if len(code_warnings) > 0:
        logger.warning(
            "You tried to add {} codes that are not in the split".format(
                len(code_warnings)
            )
        )
        logger.warning(f"Codes that are not in the split: {code_warnings}")

    return metrics_per_split


def bootstrap_aggregate_subjects(
    metrics_per_split: dict[str, Any],
    codes_per_split: np.ndarray,
    split: str,
    preds: dict[str, np.ndarray],
) -> dict[str, Any]:
    """
    Aggregate subject predictions based on split type.

    For test split, predictions are stacked as arrays (same subjects each iter).
    For train/val, predictions are stored in dicts keyed by subject code.

    Parameters
    ----------
    metrics_per_split : dict
        Accumulated metrics and predictions.
    codes_per_split : np.ndarray
        Subject codes for current split.
    split : str
        Split name ('train', 'val', 'test').
    preds : dict
        Predictions from current iteration.

    Returns
    -------
    dict
        Updated metrics_per_split with aggregated predictions.

    Raises
    ------
    ValueError
        If unknown split specified.
    """
    # Aggregate the predictions as well so you could get average probabilty per patient,
    # and some uncertainty quantification (aleatoric and epistemic uncertainty)?

    # Add a "dummy key" so that the array code above works for this without modifications
    arrays_preds = {"predictions": preds}
    if split == "test":
        # The subjects are always the same for each iteration
        metrics_per_split = bootstrap_aggregate_arrays(
            arrays=arrays_preds, metrics_per_split=metrics_per_split, main_key="preds"
        )
        from src.ensemble.ensemble_classification import check_metrics_iter_preds

        check_metrics_iter_preds(
            dict_arrays=metrics_per_split["preds"]["arrays"]["predictions"]
        )
    elif split == "train" or split == "val":
        # Subjects are not the same now for each iterations, so we need to aggregate them
        # by the subject code
        metrics_per_split = bootstrap_aggregate_by_subject_per_split(
            arrays=arrays_preds,
            metrics_per_split=metrics_per_split,
            codes_per_split=codes_per_split,
            main_key="preds_dict",
        )
        from src.ensemble.ensemble_classification import check_metrics_iter_preds_dict

        check_metrics_iter_preds_dict(
            dict_arrays=metrics_per_split["preds_dict"]["arrays"]
        )
    else:
        logger.error(f"Unknown split: {split}")
        raise ValueError

    return metrics_per_split


def bootstrap_metrics_per_split(
    X: np.ndarray,
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    model: Any,
    model_name: str,
    metrics_per_split: dict[str, Any],
    codes_per_split: np.ndarray,
    method_cfg: DictConfig,
    cfg: DictConfig,
    split: str,
    skip_mlflow: bool = False,
    recompute_for_ensemble: bool = False,
) -> dict[str, Any]:
    """
    Compute and aggregate metrics for a single split in bootstrap iteration.

    Calculates classifier metrics, calibration metrics, interpolates curves,
    and aggregates all results across iterations.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix for the split.
    y_true : np.ndarray
        True labels.
    preds : dict
        Model predictions with 'y_pred', 'y_pred_proba'.
    model : object
        Trained classifier model.
    model_name : str
        Name of the classifier.
    metrics_per_split : dict
        Accumulated metrics from previous iterations.
    codes_per_split : np.ndarray
        Subject codes for this split.
    method_cfg : DictConfig
        Bootstrap method configuration.
    cfg : DictConfig
        Full Hydra configuration.
    split : str
        Split name ('train', 'val', 'test').
    skip_mlflow : bool, default False
        Skip MLflow logging.
    recompute_for_ensemble : bool, default False
        If True, skip subject aggregation (already done).

    Returns
    -------
    dict
        Updated metrics_per_split with new iteration's results.
    """
    assert len(y_true) == len(
        preds["y_pred"]
    ), "y_true and y_pred must have the same length"
    assert len(y_true) == len(
        codes_per_split
    ), "y_true and codes_per_split must have the same length"

    # Get the basic metrics that you want
    metrics_dict = get_classifier_metrics(
        y_true, preds=preds, cfg=cfg, skip_mlflow=skip_mlflow, model_name=model_name
    )

    # Note with such few samples for calibration&uncertainty, results might be crap, do after the bootstrap?
    # Doing now both
    # Get calibration metrics
    metrics_dict = get_calibration_metrics(model, metrics_dict, y_true, preds=preds)

    # Interpolate the ROC and PR curves to a shared fixed length so you can aggregate them and do stats
    interpolated_arrays = bootstrap_interpolate_metric_arrays(
        arrays=metrics_dict["metrics"]["arrays"], n_samples=method_cfg["curve_x_length"]
    )

    # aggregate the scalar metrics
    metrics_per_split = bootstrap_aggregate_scalars(
        metrics_dict=metrics_dict, metrics_per_split=metrics_per_split
    )

    # aggregate the array curves
    metrics_per_split = bootstrap_aggregate_arrays(
        arrays=interpolated_arrays, metrics_per_split=metrics_per_split
    )

    # Aggregate the subject predictions (i.e. preds)
    if not recompute_for_ensemble:
        # during a "live" bootstrapping (as in when not ensembling from results loaded from MLflow)
        # ENSEMBLING: this is basically created already by the previous ensembling functions
        metrics_per_split = bootstrap_aggregate_subjects(
            metrics_per_split=metrics_per_split,
            codes_per_split=codes_per_split,
            split=split,
            preds=preds,
        )

    return metrics_per_split


def bootstrap_predict(
    model: Any, X: np.ndarray, i: int, split: str, debug_aggregation: bool = True
) -> dict[str, np.ndarray]:
    """
    Get predictions from model for bootstrap iteration.

    Parameters
    ----------
    model : object
        Trained classifier with predict_proba() method.
    X : np.ndarray
        Feature matrix.
    i : int
        Bootstrap iteration index.
    split : str
        Split name for logging.
    debug_aggregation : bool, default True
        If True, log debug info for test split.

    Returns
    -------
    dict
        Predictions with 'y_pred_proba' and 'y_pred' keys.

    Raises
    ------
    Exception
        If model prediction fails.
    """
    try:
        predict_probs = model.predict_proba(X)  # (n_samples, n_classes), e.g. (72,2)
        preds = {
            "y_pred_proba": predict_probs[
                :, 1
            ],  # (n_samples,), e.g. (72,) for the class 1 (e.g. glaucoma)
            "y_pred": model.predict(X),  # (n_samples,), e.g. (72,)
        }
    except Exception as e:
        logger.error(f"Could not get prediction from the model: {e}")
        raise e

    if debug_aggregation:
        if split == "test":
            logger.info(
                f"DEBUG iter #{i + 1}: 1st sample probs the test split: {preds['y_pred_proba'][0]}"
            )

    return preds


def tabm_demodata_fix(preds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Fix TabM prediction array length mismatch on demo data.

    Parameters
    ----------
    preds : dict
        Predictions dictionary with 'y_pred_proba', 'y_pred', 'label'.

    Returns
    -------
    dict
        Fixed predictions dictionary.

    Raises
    ------
    RuntimeError
        If prediction length doesn't match expected ratio.
    """
    no_pred_length = len(preds["y_pred_proba"])
    no_pred_labels = len(preds["label"])

    if no_pred_length != no_pred_labels:
        if no_pred_length == 2 * no_pred_labels:
            preds["y_pred"] = preds["y_pred"][0:no_pred_labels]
            preds["y_pred_proba"] = preds["y_pred_proba"][0:no_pred_labels]
        else:
            logger.error("Number of predictions does not match number of labels")
            raise RuntimeError("Number of predictions does not match number of labels")

    return preds


def bootstrap_metrics(
    i: int,
    model: Any,
    dict_splits: dict[str, dict[str, np.ndarray]],
    metrics: dict[str, dict],
    results_per_iter: dict[str, dict] | None,
    method_cfg: DictConfig,
    cfg: DictConfig,
    debug_aggregation: bool = False,
    model_name: str | None = None,
) -> dict[str, dict]:
    """
    i: int
        which bootstrap iteration, or a submodel of the ensemble
    dict_splits: dict
        test: dict
            X: np.ndarray
            y: np.ndarray
            w: np.ndarray
            codes: np.ndarray
    metrics: dict, e.g. {} on i=0
    results_per_iter, e.g. None on i=0
    """
    for split in dict_splits.keys():
        if split not in metrics:
            metrics[split] = {}
        X = dict_splits[split]["X"]
        y_true = dict_splits[split]["y"]
        if results_per_iter is None:
            # for sklearn API like models, we can do predict() here
            preds = bootstrap_predict(
                model, X, i, split, debug_aggregation=debug_aggregation
            )
        else:
            # more custom models like TabM, have already the preds done
            if model_name is not None:
                if model_name == "TabM":
                    preds = get_tabm_preds_from_results_for_bootstrap(
                        split_results=results_per_iter[split]
                    )
                elif model_name == "CATBOOST":
                    preds = get_catboost_preds_from_results_for_bootstrap(
                        split_results=results_per_iter[split], split=split
                    )
                elif model_name == "TabPFN":
                    preds = results_per_iter[split]
                else:
                    logger.error(
                        "Some novel model name for bootstrap? model_name = {}".format(
                            model_name
                        )
                    )
                    raise ValueError(
                        "Some novel model name for bootstrap? model_name = {}".format(
                            model_name
                        )
                    )

        # add the label here (if you want to plot some probs distributions as a function of label, apply threshold
        # tuning, or whatever)
        preds["label"] = y_true

        # hacky fix if you are running TabM, on demo data
        if model_name == "TabM":
            preds = tabm_demodata_fix(preds)

        assert preds["label"].shape[0] == preds["y_pred_proba"].shape[0], (
            f"label ({preds['label'].shape[0]}) and "
            f"probs ({preds['y_pred_proba'].shape[0]}) should have the same length"
        )

        # TODO! Add the "granular label", when you add it to the input data?
        #  early, moderate, severe, etc. for glaucoma, or whatever you have

        warnings.simplefilter("ignore")
        metrics[split] = bootstrap_metrics_per_split(
            X,
            y_true,
            preds,
            model,
            model_name,
            metrics_per_split=metrics[split],
            codes_per_split=dict_splits[split]["codes"],
            method_cfg=method_cfg,
            cfg=cfg,
            split=split,
        )
        warnings.resetwarnings()

    return metrics


def get_p_from_alpha(alpha: float = DEFAULT_CI_LEVEL) -> float:
    """
    Convert confidence level alpha to percentile value.

    Parameters
    ----------
    alpha : float, default DEFAULT_CI_LEVEL
        Confidence level (e.g., 0.95 for 95% CI).

    Returns
    -------
    float
        Percentile value for lower bound (e.g., 2.5 for alpha=0.95).
    """
    return np.round(
        ((1.0 - alpha) / 2.0) * 100, 1
    )  # e.g. 2.5 with alpha=0.95 (2.5 - 97.5%)


def bootstrap_scalar_stats_per_metric(
    values: np.ndarray, method_cfg: DictConfig
) -> dict[str, int | float | np.ndarray]:
    """
    Compute summary statistics for a scalar metric across bootstrap iterations.

    Parameters
    ----------
    values : np.ndarray
        Array of metric values from all iterations.
    method_cfg : DictConfig
        Bootstrap configuration with 'alpha_CI'.

    Returns
    -------
    dict
        Statistics with 'n', 'mean', 'std', 'ci' keys.
    """
    dict_out = {}
    dict_out["n"] = len(values)
    warnings.simplefilter("ignore")
    dict_out["mean"] = np.nanmean(values)
    dict_out["std"] = np.nanstd(values)

    # Confidence intervals
    p = get_p_from_alpha(alpha=method_cfg["alpha_CI"])
    if dict_out["std"] != 0:
        # all the values are the same, no point in trying to estimate CI, and get a warning clogging up your logs
        dict_out["ci"] = np.nanpercentile(values, [p, 100 - p])
    else:
        dict_out["ci"] = np.array((np.nan, np.nan))
    warnings.resetwarnings()

    return dict_out


def convert_inf_to_nan(values: np.ndarray) -> np.ndarray:
    """
    Replace infinite values with NaN in array.

    Parameters
    ----------
    values : np.ndarray
        2D array possibly containing inf values.

    Returns
    -------
    np.ndarray
        Array with inf replaced by NaN.

    Raises
    ------
    NotImplementedError
        If array is not 2D.
    """
    # vector-based for 2D arrays? instead of the loop
    if np.any(np.isinf(values)):
        if len(values.shape) == 2:
            for row in range(values.shape[0]):
                for col in range(values.shape[1]):
                    is_inf = np.isinf(values[row, col])
                    if is_inf:
                        values[row, col] = np.nan
        else:
            raise NotImplementedError(f"Not implemented, {values.shape}dim array")
    return values


def get_array_stats_per_metric(
    values: np.ndarray, method_cfg: DictConfig, inf_to_nan: bool = True
) -> dict[str, np.ndarray]:
    """
    Compute summary statistics for array metrics across bootstrap iterations.

    Parameters
    ----------
    values : np.ndarray
        2D array of shape (curve_length, n_iterations).
    method_cfg : DictConfig
        Bootstrap configuration with 'alpha_CI'.
    inf_to_nan : bool, default True
        Convert infinite values to NaN before computing stats.

    Returns
    -------
    dict
        Statistics with 'mean', 'std', 'ci' arrays.
    """
    if inf_to_nan:
        values = convert_inf_to_nan(values)
    warnings.simplefilter("ignore")
    dict_out = {}
    dict_out["mean"] = np.mean(values, axis=1)
    dict_out["std"] = np.mean(values, axis=1)

    # Confidence intervals
    p = round(((1.0 - method_cfg["alpha_CI"]) / 2.0) * 100, 1)
    if ~np.all(values.flatten() == values.flatten()[0]):
        dict_out["ci"] = np.nanpercentile(values, [p, 100 - p], axis=1)
    else:
        # if all the values are the same, no point in trying to estimate CI
        dict_out["ci"] = np.array((np.nan, np.nan))
    warnings.resetwarnings()
    return dict_out


def bootstrap_scalar_stats(
    metrics_per_split: dict[str, np.ndarray], method_cfg: DictConfig, split: str
) -> dict[str, dict[str, int | float | np.ndarray]]:
    """
    Compute statistics for all scalar metrics in a split.

    Parameters
    ----------
    metrics_per_split : dict
        Accumulated scalar metrics per metric name.
    method_cfg : DictConfig
        Bootstrap configuration.
    split : str
        Split name (unused, for signature compatibility).

    Returns
    -------
    dict
        Statistics per metric with mean, std, CI.
    """
    metrics_out = {}
    for metric in metrics_per_split.keys():
        metrics_out[metric] = bootstrap_scalar_stats_per_metric(
            values=metrics_per_split[metric],
            method_cfg=method_cfg,
        )
    return metrics_out


def bootstrap_array_stats(
    metrics_per_split: dict[str, dict[str, np.ndarray]], method_cfg: DictConfig
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """
    Compute statistics for all array metrics (curves) in a split.

    Parameters
    ----------
    metrics_per_split : dict
        Accumulated array metrics per metric name.
    method_cfg : DictConfig
        Bootstrap configuration.

    Returns
    -------
    dict
        Statistics per metric and variable with mean, std, CI arrays.
    """
    metrics_out = {}
    for metric in metrics_per_split.keys():
        metrics_out[metric] = {}
        for variable in metrics_per_split[metric].keys():
            metrics_out[metric][variable] = get_array_stats_per_metric(
                values=metrics_per_split[metric][variable], method_cfg=method_cfg
            )

    return metrics_out


def check_bootstrap_probability_predictions(metrics: dict[str, dict]) -> None:
    """
    Validate that bootstrap predictions vary across iterations.

    Warns if all predictions are identical, which indicates a bug
    in model retraining or bootstrap resampling.

    Parameters
    ----------
    metrics : dict
        Accumulated metrics with predictions per split.
    """

    def check_probs_array(probs_array, split):
        if isinstance(probs_array, list):
            # These are list if coming from train/val
            if len(probs_array) > 1:
                # (n_samples, n_bootstrap_iters)
                probs_array = np.array(probs_array)[np.newaxis, :]
            else:
                # if you run only for couple of iterations, you might have just
                # one sample per subject, and stdev will obviously be zero
                return
        # stdev = np.nanstd(probs_array, axis=1)
        probs_are_the_same = np.all(probs_array.flatten() == probs_array.flatten()[0])
        if probs_are_the_same:
            logger.warning(
                f"Class probabilities across all the {probs_array.shape[1]} "
                f"bootstrap iteration seem to be the same"
            )
            logger.warning(
                "Either your model is very deterministic on different bootstrap iters or you have a bug"
            )
            logger.error(
                "Either the model does not get updated on each iteration, "
                "you do not use the bootstrap samples?"
            )
            if np.all(probs_array == 0.5):
                logger.warning(
                    "All probabilities across the bootstrap iteration seem to be 0.5 with the model"
                    "failing to learning anything?"
                )
            logger.error(
                "Not raising an exception here as I guess with garbage outlier detection + imputation, "
                'you might get "unlearnable input data"?'
            )

    for split, split_dict in metrics.items():
        if "preds_dict" in split_dict:  # train/val
            probs_dict = split_dict["preds_dict"]["arrays"]["y_pred_proba"]
            for code, probs_array in probs_dict.items():
                check_probs_array(probs_array, split)
        elif "preds" in split_dict:
            probs_array = split_dict["preds"]["arrays"]["predictions"]["y_pred_proba"]
            check_probs_array(probs_array, split)
        else:
            logger.error("How come an error here?")
            raise ValueError


def bootstrap_compute_stats(
    metrics: dict[str, dict],
    method_cfg: DictConfig,
    call_from: str,
    verbose: bool = True,
) -> dict[str, dict[str, dict]]:
    """
    Compute final statistics from all bootstrap iterations.

    Aggregates scalar and array metrics into mean, std, and CI values.

    Parameters
    ----------
    metrics : dict
        Accumulated metrics from all bootstrap iterations.
    method_cfg : DictConfig
        Bootstrap configuration.
    call_from : str
        Caller identifier for conditional checks.
    verbose : bool, default True
        Enable logging.

    Returns
    -------
    dict
        Statistics per split with scalars and arrays.
    """
    if verbose:
        logger.info("Compute Bootstrap statistics (AUROC, ROC Curves, etc.)")
    warnings.simplefilter("ignore")
    if call_from != "ts_ensemble":
        check_bootstrap_probability_predictions(metrics)
    else:
        logger.info("Skip bootstrap check")
    metrics_stats = {}
    for split in metrics.keys():
        metrics_stats[split] = {}
        metrics_stats[split]["metrics"] = {}
        if "metrics" not in metrics[split]:
            logger.info(
                f"Skip the split {split} as no metrics found (e.g. with ensembling val is skipped)"
            )
            metrics_stats[split]["metrics"]["scalars"] = None
            metrics_stats[split]["metrics"]["arrays"] = None
        else:
            metrics_stats[split]["metrics"]["scalars"] = bootstrap_scalar_stats(
                metrics_per_split=metrics[split]["metrics"]["scalars"],
                method_cfg=method_cfg,
                split=split,
            )
            metrics_stats[split]["metrics"]["arrays"] = bootstrap_array_stats(
                metrics_per_split=metrics[split]["metrics"]["arrays"],
                method_cfg=method_cfg,
            )

    warnings.resetwarnings()

    return metrics_stats


def bootstrap_subject_stats_numpy_array(
    preds_per_key: np.ndarray, labels: np.ndarray, key: str
) -> dict[str, np.ndarray]:
    """
    Compute per-subject statistics from 2D prediction array.

    Used for test split where subjects are consistent across iterations.

    Parameters
    ----------
    preds_per_key : np.ndarray
        Predictions of shape (n_subjects, n_iterations).
    labels : np.ndarray
        True labels (unused, for signature consistency).
    key : str
        Prediction key (unused, for signature consistency).

    Returns
    -------
    dict
        Statistics with 'mean' and 'std' arrays (n_subjects,).
    """
    dict_out = {}
    warnings.simplefilter("ignore")
    dict_out["mean"] = np.mean(preds_per_key, axis=1)
    dict_out["std"] = np.std(preds_per_key, axis=1)
    warnings.resetwarnings()

    return dict_out


def aggregate_dict_subjects(
    dict_out: dict[str, np.ndarray], stats_per_code: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """
    Aggregate subject statistics by horizontal stacking.

    Parameters
    ----------
    dict_out : dict
        Accumulated statistics.
    stats_per_code : dict
        Statistics for a single subject code.

    Returns
    -------
    dict
        Updated statistics with new subject appended.
    """
    for key, scalar_in_array in stats_per_code.items():
        if key not in dict_out.keys():
            dict_out[key] = scalar_in_array
        else:
            dict_out[key] = np.hstack((dict_out[key], scalar_in_array))

    return dict_out


def bootstrap_subject_stats_dict(
    preds_per_key: dict[str, list[float]],
    labels: np.ndarray,
    _codes_train: np.ndarray,
    key: str,
    split: str = "train",
    verbose: bool = True,
    check_preds: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    Compute per-subject statistics from dictionary of predictions.

    Used for train/val splits where predictions are stored by subject code.
    Also computes uncertainty quantification for probability predictions.

    Parameters
    ----------
    preds_per_key : dict
        Predictions keyed by subject code.
    labels : np.ndarray
        True labels for subjects.
    codes_train : np.ndarray
        Subject codes for ordering.
    key : str
        Prediction key (e.g., 'y_pred_proba').
    split : str
        Split name for logging.
    verbose : bool, default True
        Enable logging.
    check_preds : bool, default False
        Validate predictions vary per subject.

    Returns
    -------
    tuple
        (stats_dict, uq_dict) with per-subject statistics and uncertainty.
    """
    warnings.simplefilter("ignore")
    dict_out = {}
    for code in preds_per_key.keys():
        array_per_code = np.array(preds_per_key[code])[np.newaxis, :]
        if check_preds:
            check_indiv_code_for_different_preds(code, list(array_per_code), key)
        stats_per_code = bootstrap_subject_stats_numpy_array(
            array_per_code, labels=labels, key=key
        )
        dict_out = aggregate_dict_subjects(dict_out, stats_per_code)

    # Uncertainty here for the train/val splits
    if key == "y_pred_proba":
        if verbose:
            logger.info(
                "Compute uncertainty quantification, split = {}, key = {}".format(
                    split, key
                )
            )
        assert len(labels) == len(
            preds_per_key
        ), "label and prediction lengths do not match"
        uq_dict = uncertainty_wrapper_from_subject_codes(
            p_mean=dict_out["mean"], p_std=dict_out["std"], y_true=labels, split=split
        )
        warnings.resetwarnings()
    else:
        uq_dict = {}

    return dict_out, uq_dict


def sort_dict_keys_based_on_list(
    dict_to_sort: dict[str, Any], list_to_sort_by: list[str], sort_list: bool = True
) -> dict[str, Any]:
    """
    Sort dictionary keys to match a reference list order.

    Parameters
    ----------
    dict_to_sort : dict
        Dictionary to reorder.
    list_to_sort_by : list
        Reference list defining key order.
    sort_list : bool, default True
        If True, reorder to match list. If False, just sort alphabetically.

    Returns
    -------
    dict
        Reordered dictionary.

    Raises
    ------
    Exception
        If keys don't match reference list.
    """
    # sort the keys based on original train codes as you now get arrays for the stats
    dict_to_sort = dict(sorted(dict_to_sort.items()))

    if sort_list:
        # Only need to sort when you are bootstrapping, does not matter for CatBoost Ensemble
        try:
            return {k: dict_to_sort[k] for k in list_to_sort_by}
        except Exception as e:
            logger.error(f"Could not sort the dict: {e}")
            raise e
    else:
        return dict_to_sort


def bootstrap_check_that_samples_different(
    preds_per_key: dict, key: str, check_preds: bool = False
):
    """
    Verify predictions vary across bootstrap iterations per subject.

    Parameters
    ----------
    preds_per_key : dict
        Predictions keyed by subject code.
    key : str
        Prediction type key.
    check_preds : bool, default False
        If True, perform detailed validation.
    """
    for subject_code in list(preds_per_key.keys()):
        preds_per_code = preds_per_key[subject_code]
        if check_preds:
            check_indiv_code_for_different_preds(subject_code, preds_per_code, key)


def check_indiv_code_for_different_preds(
    subject_code: str, preds_per_code: list, key: str
):
    """
    Check if predictions for a subject vary across iterations.

    Note: May raise false alarms for garbage input data where model
    consistently outputs same predictions.

    Parameters
    ----------
    subject_code : str
        Subject identifier.
    preds_per_code : list
        List of predictions for this subject across iterations.
    key : str
        Prediction type (e.g., 'y_pred_proba').

    Raises
    ------
    ValueError
        If all predictions are identical for probability predictions.
    """
    if key == "y_pred_proba":
        # could happen that class labels are same for tiny bootstraps?
        # or badly functioning model maybe just outputs all the classes the same?
        if len(np.unique(preds_per_code)) == 1:
            logger.warning(f"Subject {subject_code} has the same predictions")
            logger.warning(preds_per_code)
            raise ValueError


def bootstrap_compute_subject_stats(
    metrics_iter,
    dict_arrays,
    method_cfg,
    sort_list: bool = True,
    call_from: str = None,
    verbose: bool = True,
):
    """
    Compute per-subject statistics from bootstrap iterations.

    Aggregates predictions across bootstrap iterations to compute
    mean predictions and uncertainty per subject.

    Parameters
    ----------
    metrics_iter : dict
        Accumulated metrics from all iterations.
    dict_arrays : dict
        Original data arrays with labels and codes.
    method_cfg : DictConfig
        Bootstrap configuration.
    sort_list : bool, default True
        Sort results to match original code order.
    call_from : str, optional
        Caller identifier for special handling.
    verbose : bool, default True
        Enable logging.

    Returns
    -------
    dict
        Per-subject statistics per split.
    """
    if verbose:
        logger.info(
            "Compute subject-wise Bootstrap statistics (class probabiities, uncertainty quantification, etc."
        )
    subject_stats = {}
    for split in metrics_iter.keys():
        labels, _ = get_labels_and_codes(split, dict_arrays, call_from)
        subject_stats[split] = {}
        subject_stats[split]["preds"] = {}

        dict_per_split = metrics_iter[split]
        if "preds_dict" in dict_per_split.keys():
            if call_from == "CATBOOST":
                codes = dict_arrays[f"subject_codes_{split}"]
            elif call_from == "classification_ensemble":
                # examine later why the codes do not seem to match, if you want to use the classification ensembles
                # which do not seem to provide much, just use "the internal ensembling" of Tree-based methods
                codes = None  # dict_arrays[f"subject_codes_{split}"]
            else:
                codes = dict_arrays["subject_codes_train"]
            # train/val split with the subject codes
            preds_per_split = dict_per_split["preds_dict"]["arrays"]
            for key in preds_per_split.keys():
                if codes is None:
                    # don't use the metadata codes (there is some glitch?) for the ensemble
                    # use directly the prediction codes for sorting
                    codes = sorted(list(set(list(preds_per_split[key].keys()))))
                preds_per_key = sort_dict_keys_based_on_list(
                    preds_per_split[key], list(codes), sort_list=sort_list
                )
                assert len(labels) == len(preds_per_key), (
                    f"label ({len(labels)}) and pred ({len(preds_per_key)}) "
                    f"lengths do not match"
                )
                bootstrap_check_that_samples_different(preds_per_key, key)
                if key == "y_pred_proba":
                    subject_stats[split]["preds"][key], subject_stats[split]["uq"] = (
                        bootstrap_subject_stats_dict(
                            preds_per_key=preds_per_key,
                            labels=labels,
                            codes_train=codes,
                            key=key,
                            split=split,
                            verbose=verbose,
                        )
                    )
                else:
                    subject_stats[split]["preds"][key], _ = (
                        bootstrap_subject_stats_dict(
                            preds_per_key=preds_per_key,
                            labels=labels,
                            codes_train=codes,
                            key=key,
                            split=split,
                            verbose=verbose,
                        )
                    )
            subject_stats[split]["subject_code"] = codes
            subject_stats[split]["labels"] = labels
            assert len(subject_stats[split]["subject_code"]) == len(
                preds_per_key
            ), "Codes and predictions must have the same length"
            assert len(subject_stats[split]["subject_code"]) == len(
                labels
            ), "Codes and labels must have the same length"

        elif "preds" in dict_per_split.keys():
            # test split
            codes = dict_arrays["subject_codes_test"]
            preds_per_split = dict_per_split["preds"]["arrays"]["predictions"]
            for key in preds_per_split.keys():
                preds_per_key = preds_per_split[key]
                subject_stats[split]["preds"][key] = (
                    bootstrap_subject_stats_numpy_array(
                        preds_per_key=preds_per_key, labels=labels, key=key
                    )
                )
                # Uncertainty Quantification
                if key == "y_pred_proba":
                    subject_stats[split]["uq"] = uncertainty_wrapper(
                        preds=preds_per_key, y_true=labels, key=key, split=split
                    )

            subject_stats[split]["subject_code"] = codes
            subject_stats[split]["labels"] = labels
            assert len(subject_stats[split]["subject_code"]) == len(
                preds_per_key
            ), "Codes and predictions must have the same length"
            assert len(subject_stats[split]["subject_code"]) == len(
                labels
            ), "Codes and labels must have the same length"

        else:
            logger.error(f"Where are the predictions now? {dict_per_split.keys()}")
            raise ValueError

    return subject_stats


def global_subject_stats(
    values: np.ndarray,
    labels: np.ndarray,
    key: str,
    variable: str,
    method_cfg: DictConfig,
):
    """
    Compute global statistics stratified by class label.

    Parameters
    ----------
    values : np.ndarray
        Per-subject values to aggregate.
    labels : np.ndarray
        Class labels for stratification.
    key : str
        Prediction key (unused, for logging).
    variable : str
        Variable name (unused, for logging).
    method_cfg : DictConfig
        Bootstrap configuration.

    Returns
    -------
    dict
        Statistics per class label with mean, std, CI.
    """
    dict_out = {}
    # not much point in averaging all the subject probabilities together without accounting for the label
    unique_labels = np.unique(labels)
    for label in unique_labels:
        values_per_label = values[labels == label]
        dict_out[label] = bootstrap_scalar_stats_per_metric(
            values_per_label, method_cfg=method_cfg
        )

    return dict_out


def get_labels_and_codes(split, dict_arrays, call_from):
    """
    Get labels and codes for a split, handling bootstrap vs ensemble cases.

    Parameters
    ----------
    split : str
        Split name ('train', 'val', 'test').
    dict_arrays : dict
        Data arrays with labels and codes.
    call_from : str or None
        Caller identifier for special handling.

    Returns
    -------
    tuple
        (labels, codes) arrays for the split.

    Raises
    ------
    ValueError
        If unknown split specified.
    """
    # These splits are now "bootstrapping splits" so the labels, codes are from the original Train
    if split == "train" or split == "val":
        if call_from is None or call_from == "classification_ensemble":
            labels, codes = dict_arrays["y_train"], dict_arrays["subject_codes_train"]
        elif call_from == "CATBOOST":
            # This is actually a call for the ensemble evaluation, the Catboost bootstrap call is handled
            # normally as the None condition above
            labels, codes = (
                dict_arrays[f"y_{split}"],
                dict_arrays[f"subject_codes_{split}"],
            )
    elif split == "test":
        labels, codes = dict_arrays["y_test"], dict_arrays["subject_codes_test"]
    else:
        logger.error(f"Unknown split: {split}")
        raise ValueError
    assert len(labels) == len(codes), "Labels and codes must have the same length"

    return labels, codes


def bootstrap_compute_global_subject_stats(
    subjectwise_stats, method_cfg, verbose: bool = True
):
    """
    Compute global subject-level statistics across all subjects.

    Aggregates per-subject statistics (e.g., mean probability, uncertainty)
    into population-level summaries stratified by class.

    Parameters
    ----------
    subjectwise_stats : dict
        Per-subject statistics from bootstrap_compute_subject_stats.
    method_cfg : DictConfig
        Bootstrap configuration.
    verbose : bool, default True
        Enable logging.

    Returns
    -------
    dict
        Global statistics per split, key, variable, and class.
    """
    # Compute the "mean response" of the subjects, e.g. scalar mean UQ metric to describe the whole model uncertainty

    subject_global_stats = {}
    for split in subjectwise_stats.keys():
        subject_global_stats[split] = {}
        labels = subjectwise_stats[split]["labels"]
        for key in subjectwise_stats[split]["preds"].keys():
            preds_per_key = subjectwise_stats[split]["preds"][key]
            subject_global_stats[split][key] = {}
            for variable in preds_per_key.keys():
                # Note! Now we just compute stats for whatever variables you have here, not like UQ epistemic
                # uncertainty computed from y_pred (class) is necesssarily what you need, but pick the correct combos
                # when visualizing and logging to MLflow. Easier than cherrypicking here what makes sense and not
                values: np.ndarray = preds_per_key[variable]
                subject_global_stats[split][key][variable] = global_subject_stats(
                    values, labels, key, variable, method_cfg
                )

    return subject_global_stats


def compute_uq_unks_from_dict_of_subjects(probs_dict: dict):
    """
    Compute uncertainty metrics from subject-keyed probability dictionary.

    Used for train/val splits where different subjects appear in different
    bootstrap iterations. Computes ensemble-style uncertainty metrics.

    Parameters
    ----------
    probs_dict : dict
        Probabilities keyed by subject code, each a list of predictions.

    Returns
    -------
    dict
        Uncertainty metrics (confidence, entropy, mutual_information) per subject.

    Notes
    -----
    Uses ensemble_uncertainties from CatBoost tutorial code for metrics like
    total uncertainty, data uncertainty, and knowledge uncertainty.
    """
    uq = None
    iters = []
    for code, list_of_probs in probs_dict.items():
        probs_code = np.array(list_of_probs)[
            :, np.newaxis, np.newaxis
        ]  # (esize/no_iter, 1, 1)
        probs_code = np.concatenate(
            [probs_code, 1 - probs_code], axis=2
        )  # (n_samples, 1, n_classes=2)
        iters.append(probs_code.shape[0])
        uq_code: dict = ensemble_uncertainties(probs=probs_code)
        # TODO! compute AURC here as well? instead of before?
        if uq is None:
            uq = deepcopy(uq_code)
        else:
            for key in uq.keys():
                uq[key] = np.vstack((uq[key], uq_code[key]))

    assert uq["confidence"].shape[0] == len(
        probs_dict
    ), "You did not compute probs ({}) for all the subjects ({})".format(
        uq["confidence"].shape[0], len(probs_dict)
    )

    return uq


def compute_uq_for_subjectwise_stats(
    metrics_iter, subjectwise_stats, verbose: bool = True
):
    """
    Compute and merge uncertainty quantification into subject-wise stats.

    Computes ensemble-based uncertainty metrics (confidence, entropy,
    mutual information) and adds them to subjectwise_stats.

    Parameters
    ----------
    metrics_iter : dict
        Accumulated metrics with predictions per split.
    subjectwise_stats : dict
        Per-subject statistics to augment.
    verbose : bool, default True
        Enable logging.

    Returns
    -------
    dict
        Updated subjectwise_stats with uncertainty metrics added.
    """
    from src.classification.catboost.catboost_main import (
        combine_unks_with_subjectwise_stats,
    )

    uq = {}
    for split in metrics_iter.keys():
        if "preds_dict" in metrics_iter[split].keys():
            probs_dict = metrics_iter[split]["preds_dict"]["arrays"]["y_pred_proba"]
            assert isinstance(probs_dict, dict), "Probs must be a dictionary"
            uq[split] = compute_uq_unks_from_dict_of_subjects(probs_dict)
        elif "preds" in metrics_iter[split].keys():
            probs = metrics_iter[split]["preds"]["arrays"]["predictions"][
                "y_pred_proba"
            ]
            assert isinstance(probs, np.ndarray), "Probs must be a numpy array"
            probs = probs.T[:, :, np.newaxis]  # (n_iter, n_samples, 1)
            probs = np.concatenate(
                [probs, 1 - probs], axis=2
            )  # (n_iter, n_samples, n_classes=2)
            uq[split] = ensemble_uncertainties(probs)
        else:
            logger.error(f"Where are the predictions now? {metrics_iter[split].keys()}")
            raise ValueError

        subjectwise_stats = combine_unks_with_subjectwise_stats(
            subjectwise_stats, unks=uq[split], split=split
        )

    return subjectwise_stats
