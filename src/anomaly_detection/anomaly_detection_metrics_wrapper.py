from copy import deepcopy

import numpy as np
from loguru import logger
from momentfm.utils.anomaly_detection_metrics import (
    adjust_predicts,
    f1_score,
)
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix

from src.imputation.momentfm.moment_utils import reshape_np_array_windows


def get_best_outlier_metric(scalar_metrics, outlier_model_cfg) -> float:
    """
    Retrieve the best outlier detection metric value based on configuration.

    Parameters
    ----------
    scalar_metrics : dict
        Dictionary containing computed metrics with a 'global' key for global metrics.
    outlier_model_cfg : dict or DictConfig
        Outlier model configuration containing EVALUATION.best_metric.

    Returns
    -------
    float
        The value of the specified best metric from global metrics.

    Raises
    ------
    ValueError
        If the specified metric is not found in scalar_metrics.
    """
    metric_name = outlier_model_cfg["EVALUATION"]["best_metric"]
    if metric_name in scalar_metrics["global"]:
        return scalar_metrics["global"][metric_name]
    else:
        logger.error(f"Metric {metric_name} not found in scalar_metrics")
        raise ValueError(f"Metric {metric_name} not found in scalar_metrics")


def anomaly_MSE_metric(trues, preds):
    """
    Compute element-wise Mean Squared Error as anomaly score.

    Parameters
    ----------
    trues : np.ndarray
        Ground truth values (observed time series).
    preds : np.ndarray
        Predicted values from the reconstruction model.

    Returns
    -------
    np.ndarray
        Element-wise squared error between true and predicted values.
    """
    return (trues - preds) ** 2


def adjbestf1_with_threshold(
    y_true: np.array,
    y_scores: np.array,
    n_splits: int = 100,
    adj_f1_threshold: float = None,
):
    """
    Compute adjusted best F1 score with optimal threshold selection.

    Finds the threshold that maximizes the adjusted F1 score for anomaly detection.
    The adjustment accounts for temporal nature by crediting detection of any
    timestep within an anomaly sequence as correct detection of the full sequence.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (1 for anomaly, 0 for normal).
    y_scores : np.ndarray
        Anomaly scores (higher indicates more anomalous).
    n_splits : int, optional
        Number of threshold candidates to evaluate. Default is 100.
    adj_f1_threshold : float, optional
        If provided, use this fixed threshold instead of the optimal one.
        Useful for applying a global threshold to subjectwise evaluation.

    Returns
    -------
    best_adjusted_f1 : float
        The best (or fixed threshold) adjusted F1 score.
    best_threshold : float
        The threshold that achieved the best F1 (always returns optimal,
        even when adj_f1_threshold is provided for comparison).
    """
    # modified from the original MOMENT code
    thresholds = np.linspace(y_scores.min(), y_scores.max(), n_splits)
    adjusted_f1 = np.zeros(thresholds.shape)

    for i, threshold in enumerate(thresholds):
        y_pred = y_scores >= threshold
        y_pred = adjust_predicts(
            score=y_scores,
            label=(y_true > 0),
            pred=y_pred,
            threshold=None,
            calc_latency=False,
        )
        adjusted_f1[i] = f1_score(y_pred, y_true)

    best_adjusted_f1 = np.max(adjusted_f1)
    best_threshold = thresholds[np.argmax(adjusted_f1)]

    if adj_f1_threshold is not None:
        # for example, when you have already computed "global stats" as in across the whole dataset
        # and you want you use that value instead of defining a new threshold for each subject when
        # doing the "subjectwise stats"
        y_pred = y_scores >= adj_f1_threshold
        best_adjusted_f1 = f1_score(y_pred, y_true)
        # return either way the subject-specific thresholds so that you have an idea of the spread?
        # might get confusing to remember why you have different thresholds returned to MLflow
        # even though you always used a fixed threshold when computing the F1

    return best_adjusted_f1, best_threshold


def metrics_per_split_v2(split_results, split, cfg, outlier_model_cfg):
    """
    Compute anomaly detection metrics for a single data split (version 2).

    This version reshapes windowed arrays before computing metrics and uses
    the extended outlier detection evaluation module.

    Parameters
    ----------
    split_results : dict
        Dictionary containing 'arrays' with 'trues', 'preds', and 'labels'.
    split : str
        Name of the data split (e.g., 'train', 'test').
    cfg : DictConfig
        Main configuration object.
    outlier_model_cfg : DictConfig
        Outlier model specific configuration.

    Returns
    -------
    metrics : dict
        Dictionary containing 'arrays_flat' and 'scalars' with computed metrics.
    pred_mask : dict
        Dictionary containing predicted anomaly mask in 'arrays.pred_mask'.
    """
    from src.anomaly_detection.extra_eval.eval_outlier_detection import (
        get_scalar_outlier_metrics,
    )

    split_results_orig_windowed = deepcopy(split_results)
    split_results = deepcopy(split_results)
    split_results["arrays"] = reshape_np_array_windows(
        split_results["arrays"], cfg, outlier_model_cfg
    )
    trues = split_results["arrays"]["trues"]
    recon = split_results["arrays"]["preds"]
    gt = split_results["arrays"]["labels"]
    score = anomaly_MSE_metric(trues, recon)
    _, threshold = adjbestf1_with_threshold(
        y_true=gt.flatten(), y_scores=score.flatten()
    )
    pred = (score > threshold).astype(int)

    # downstream code assumes that the data is windowed, and it is reshaped again
    score_window = anomaly_MSE_metric(
        split_results_orig_windowed["arrays"]["trues"],
        split_results_orig_windowed["arrays"]["preds"],
    )
    pred_out = (score_window > threshold).astype(int)

    metrics = {}
    metrics["arrays_flat"] = {}
    metrics["scalars"] = get_scalar_outlier_metrics(
        preds=pred,
        gt=gt,
        score=score,
        use_detection_adjustment=False,
        threshold=threshold,
    )
    pred_mask = {"arrays": {"pred_mask": pred_out}}

    return metrics, pred_mask


def metrics_per_split(split_results, split):
    """
    Compute anomaly detection metrics for a single data split.

    Computes MSE-based anomaly scores and adjusted best F1 with optimal threshold.
    Uses the same scalar vs array split as imputation metrics for easy MLflow logging.

    Parameters
    ----------
    split_results : dict
        Dictionary containing 'arrays' with shape (no_batches, batch_size, n_timesteps)
        and 'arrays_flat' with flattened valid data.
    split : str
        Name of the data split (e.g., 'train', 'test').

    Returns
    -------
    metrics : dict
        Dictionary with 'scalars' (f1, adjbestf1_threshold, fp),
        'arrays' (MSE), and 'arrays_flat' keys.
    preds : dict
        Dictionary containing predicted anomaly mask in 'arrays.pred_mask'.
    """
    # Using the same scalar vs. array split here as for imputation metrics
    # Easy to log all the scalar metrics to MLflow, and then arrays can be useful for plotting
    metrics = {}
    metrics["scalars"] = {}
    metrics["arrays"] = {}
    metrics["arrays_flat"] = {}

    # (no_batches, batch_size, n_timesteps)
    assert (
        split_results["arrays"]["trues"].shape[0]
        == split_results["arrays"]["preds"].shape[0]
        == split_results["arrays"]["labels"].shape[0]
    )

    # (no_batches*batch_size*n_timesteps)
    flat_trues = split_results["arrays_flat"]["trues_valid"].flatten()
    flat_preds = split_results["arrays_flat"]["preds_valid"].flatten()
    flat_labels = split_results["arrays_flat"]["labels_valid"].flatten()
    assert flat_trues.shape[0] == flat_preds.shape[0] == flat_labels.shape[0]

    # We will use the Mean Squared Error (MSE) between the observed values and MOMENT's predictions as the anomaly score
    # (no_batches*batch_size*n_timesteps)
    anomaly_scores = anomaly_MSE_metric(trues=flat_trues, preds=flat_preds)

    # Get the anomaly detection metrics
    # (no_batches, batch_size, n_timesteps)
    metrics["arrays"]["MSE"] = anomaly_MSE_metric(
        trues=split_results["arrays"]["trues"], preds=split_results["arrays"]["preds"]
    )

    # From MOMENT
    # We will use adjusted best F1, a metric which is frequently used in practice Goswami et al., 2023,
    # "Unsupervised Model Selection for Time-series Anomaly Detection" https://arxiv.org/abs/2210.01078
    # to evaluate the anomaly detection performance. MOMENT uses the mean squarred error between its
    # predictions and the observed time series as an anomaly score. To convert anomaly score to
    # binary predictions, we must find a threshold, such that time steps with an anomaly score
    # exceeding this threshold are considered anomalous. Adjusted best F1 finds the best threshold
    # which maximizes the F1 of the anomaly detection model. To account for the temporal nature of this problem,
    # a model is said to have correctly identified the complete anomaly sequence as long
    # as it detects any anomalous timestep.
    metrics["scalars"]["f1"], metrics["scalars"]["adjbestf1_threshold"] = (
        adjbestf1_with_threshold(
            y_true=flat_labels,
            y_scores=anomaly_scores,
        )
    )

    # TimesNet/UniTS defines the adjustment a bit differently
    # see eval_outlier_detection() (in practice these seemed to be very close to each other)

    # Get some other metrics as well?
    # MOMENT routine for these does not work :(
    # metrics["scalars"]["rAUCROC"] =
    # metrics["scalars"]["rAUCPR"] =

    pred_mask = metrics["arrays"]["MSE"] >= metrics["scalars"]["adjbestf1_threshold"]
    preds = {"arrays": {"pred_mask": pred_mask}}

    flat_preds = anomaly_scores >= metrics["scalars"]["adjbestf1_threshold"]
    _, fp, _, _ = confusion_matrix(flat_labels, flat_preds).ravel()
    metrics["scalars"]["fp"] = float(fp) / len(flat_labels.flatten())

    return metrics, preds


def compute_outlier_detection_metrics(
    outlier_results: dict, cfg: DictConfig, outlier_model_cfg: DictConfig
):
    """
    Compute outlier detection metrics across all data splits.

    Iterates over all splits in outlier_results and computes per-split metrics
    using MSE-based anomaly scoring and adjusted best F1.

    Parameters
    ----------
    outlier_results : dict
        Dictionary mapping split names to split results containing arrays.
    cfg : DictConfig
        Main configuration object.
    outlier_model_cfg : DictConfig
        Outlier model specific configuration.

    Returns
    -------
    metrics : dict
        Dictionary mapping split names to their computed metrics.
    preds : dict
        Dictionary mapping split names to their predicted anomaly masks.

    References
    ----------
    https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/anomaly_detection.ipynb
    """
    metrics, preds = {}, {}
    for split, split_results in outlier_results.items():
        metrics[split], preds[split] = metrics_per_split(split_results, split)

    return metrics, preds
