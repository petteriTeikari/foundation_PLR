from typing import Dict, Tuple

import mlflow
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from src.log_helpers.mlflow_utils import (
    log_metrics_as_mlflow_artifact,
    log_mlflow_imputation_metrics,
    mlflow_imputation_metrics_logger,
)
from src.metrics.metrics_utils import (
    get_array_triplet_for_pypots_metrics_from_imputer,
    get_subjectwise_arrays,
)
from src.preprocess.preprocess_data import (
    destandardize_for_imputation_metrics,
)


def get_imputation_metric_dict(
    model_name: str,
    imputation_artifacts: Dict,
    cfg: DictConfig,
) -> Dict[str, Dict]:
    """
    Compute imputation metrics for all data splits.

    Parameters
    ----------
    model_name : str
        Name of the imputation model being evaluated.
    imputation_artifacts : dict
        Dictionary containing model artifacts and source data with imputation results.
    cfg : DictConfig
        Hydra configuration object.

    Returns
    -------
    dict
        Dictionary mapping split names to their computed metrics.
    """
    metrics = {}
    for split in imputation_artifacts["model_artifacts"]["imputation"]:
        logger.debug(f"Computing the metrics for the '{split}' split")
        metrics[split] = compute_metrics_by_split(
            split_imputation=imputation_artifacts["model_artifacts"]["imputation"][
                split
            ],
            preprocess_dict=imputation_artifacts["source_data"]["preprocess"],
            split_data=imputation_artifacts["source_data"]["df"][split],
            model_name=model_name,
            split=split,
            cfg=cfg,
        )

    return metrics


def log_metrics_per_split_as_mlflow_artifact(
    metrics_global: Dict,
    model_name: str,
    split: str,
    model_artifacts: Dict,
    cfg: DictConfig,
) -> None:
    """
    Log global imputation metrics to MLflow as an artifact.

    Parameters
    ----------
    metrics_global : dict
        Dictionary of global metrics (e.g., MAE, MSE, MRE).
    model_name : str
        Name of the imputation model.
    split : str
        Data split name (e.g., 'train', 'val', 'test').
    model_artifacts : dict
        Dictionary containing model artifacts.
    cfg : DictConfig
        Hydra configuration object.

    Returns
    -------
    None
    """
    # Log the metrics to MLflow (and subjectwise metrics as a pickled artifact)
    log_mlflow_imputation_metrics(
        metrics_global=metrics_global,
        split=split,
        model_artifacts=model_artifacts,
        model_name=model_name,
        cfg=cfg,
    )


def recompute_submodel_imputation_metrics(
    run_id: str,
    submodel_mlflow_run: "mlflow.entities.Run",
    model_name: str,
    gt_dict: Dict,
    gt_preprocess: Dict,
    reconstructions_submodel: Dict[str, np.ndarray],
    cfg: DictConfig,
) -> Dict[str, Dict]:
    """
    Recompute and re-log imputation metrics for a submodel to MLflow.

    Parameters
    ----------
    run_id : str
        MLflow run ID to log metrics to.
    submodel_mlflow_run : mlflow.entities.Run
        MLflow run object for the submodel.
    model_name : str
        Name of the imputation model.
    gt_dict : dict
        Ground truth data dictionary with labels and data per split.
    gt_preprocess : dict
        Preprocessing parameters used for destandardization.
    reconstructions_submodel : dict
        Dictionary mapping splits to imputation arrays.
    cfg : DictConfig
        Hydra configuration object.

    Returns
    -------
    dict
        Dictionary mapping splits to their computed metrics.

    See Also
    --------
    compute_granular_metrics : For anomaly detection recomputation.
    """
    metrics = {}
    for split, imputation_array in reconstructions_submodel.items():
        # missingness_mask = gt_dict[split]['labels']['imputation_mask']
        # true_pupil = gt_dict[split]['data']['X_GT']
        if len(imputation_array.shape) == 2:
            imputation_array = np.expand_dims(imputation_array, 2)
        split_imputation = {
            "imputation_dict": {"imputation": {"mean": imputation_array}}
        }
        metrics[split] = compute_metrics_by_split(
            split_imputation=split_imputation,
            preprocess_dict=gt_preprocess,
            split_data=gt_dict[split],
            model_name=model_name,
            split=split,
            cfg=cfg,
        )

    with mlflow.start_run(run_id=run_id):
        logger.info("Re-logging the metrics to MLflow")
        for split, split_metrics in metrics.items():
            metrics_global = split_metrics["global"]
            mlflow_imputation_metrics_logger(metrics_global, split)
        mlflow.end_run()

    return metrics


# @task(
#     log_prints=True,
#     name="Compute Imputation Metrics",
#     description="Quantify the imputation (regression) performance of the models",
# )
def compute_metrics_by_model(
    model_name: str,
    imputation_artifacts: Dict,
    cfg: DictConfig,
    _log_if_improved: bool = True,
    log_mlflow: bool = True,
) -> Dict[str, Dict]:
    """
    Compute and log imputation metrics for a given model across all splits.

    Parameters
    ----------
    model_name : str
        Name of the imputation model being evaluated.
    imputation_artifacts : dict
        Dictionary containing model artifacts, source data, and optionally pre-computed metrics.
    cfg : DictConfig
        Hydra configuration object.
    _log_if_improved : bool, optional
        Unused parameter for future model registry logging (default True).
    log_mlflow : bool, optional
        Whether to log subjectwise metrics as MLflow artifact (default True).

    Returns
    -------
    dict
        Dictionary mapping splits to their computed metrics (global, subjectwise).
    """
    # MLflow log the metrics
    if "metrics" in imputation_artifacts["model_artifacts"]:
        logger.info("Using metrics already computed during training (e.g. MOMENT)")
        metrics = imputation_artifacts["model_artifacts"]["metrics"]
        split = list(metrics.keys())[0]
    else:
        logger.info("Computing the metrics from the imputed data")
        metrics = get_imputation_metric_dict(model_name, imputation_artifacts, cfg)

    # Log global metrics to MLflow (i.e. MAE)
    for split in metrics.keys():
        log_metrics_per_split_as_mlflow_artifact(
            metrics_global=metrics[split]["global"],
            model_name=model_name,
            split=split,
            model_artifacts=imputation_artifacts["model_artifacts"],
            cfg=cfg,
        )

    # Log the subjectwise metrics as a pickled artifact
    if log_mlflow:
        log_metrics_as_mlflow_artifact(
            metrics_subjectwise=metrics[split]["subjectwise"],
            model_name=model_name,
            model_artifacts=imputation_artifacts["model_artifacts"],
            cfg=cfg,
        )
    else:
        logger.info("Skipping logging of the subjectwise metrics")

    # if log_if_improved:
    #     try:
    #         # TODO! fix some glitches here, not used at the moment really for anything so not urgent
    #         #  in preparation if you need to be pushing the improved models to model registry
    #         post_imputation_model_training_mlflow_log(
    #             metrics_model=metrics,
    #             model_artifacts=imputation_artifacts["model_artifacts"],
    #             cfg=cfg,
    #         )
    #     except Exception as e:
    #         logger.error(f"Failed to log the metrics to MLflow: {e}")
    #         raise e
    # else:
    #     logger.info("Skipping logging (printing) about whether the model improved")

    return metrics


def compute_metrics_by_split(
    split_imputation: Dict,
    preprocess_dict: Dict,
    split_data: Dict,
    model_name: str,
    split: str,
    cfg: DictConfig,
) -> Dict:
    """
    Compute imputation metrics for a single data split.

    Parameters
    ----------
    split_imputation : dict
        Imputation results for the split.
    preprocess_dict : dict
        Preprocessing parameters for destandardization.
    split_data : dict
        Original data for the split including metadata.
    model_name : str
        Name of the imputation model.
    split : str
        Data split name (e.g., 'train', 'val', 'test').
    cfg : DictConfig
        Hydra configuration object.

    Returns
    -------
    dict
        Dictionary with 'global', 'subjectwise', and 'subjectwise_arrays' metrics.
    """
    # Get the arrays for the metrics computation
    X, targets, predictions, indicating_mask = (
        get_array_triplet_for_pypots_metrics_from_imputer(
            split_imputation, split_data, split, cfg=cfg
        )
    )

    # Destandardize the arrays if they were standardized
    targets, predictions = destandardize_for_imputation_metrics(
        targets, predictions, preprocess_dict
    )

    # Compute the metrics (global, and subject-wise)
    metrics = compute_imputation_metrics(
        targets,
        predictions,
        indicating_mask,
        cfg=cfg,
        metadata_dict=split_data["metadata"],
    )

    # TODO! you could compute stdevs of mae for example from the subjectwise metrics
    #  to have an idea how much of a spread there is? weigh with missing_rate?
    #  e.g. PLR1002 does not have any missing values

    return metrics


def compute_imputation_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    indicating_mask: np.ndarray,
    cfg: DictConfig,
    metadata_dict: Dict,
    checks_on: bool = False,
) -> Dict:
    """
    Compute global and subjectwise imputation metrics using BenchPOTS methodology.

    Uses the BenchPOTS suite for fair evaluation of imputation algorithms.
    See https://arxiv.org/pdf/2406.12747 and https://github.com/WenjieDu/BenchPOTS.

    Parameters
    ----------
    targets : np.ndarray
        Ground truth values, shape (n_subjects, n_timepoints, n_features).
    predictions : np.ndarray
        Imputed predictions, shape (n_subjects, n_timepoints, n_features).
    indicating_mask : np.ndarray
        Binary mask indicating missing values, shape (n_subjects, n_timepoints, n_features).
    cfg : DictConfig
        Hydra configuration object.
    metadata_dict : dict
        Metadata including subject codes for subjectwise metrics.
    checks_on : bool, optional
        Whether to run prechecks for NaN removal and validation (default False).

    Returns
    -------
    dict
        Dictionary with 'global', 'subjectwise', and 'subjectwise_arrays' keys.
    """

    # Get the metrics for each subject, useful for hunting down the outliers
    metrics_subjectwise = subjectwise_metrics_wrapper(
        predictions=predictions,
        targets=targets,
        masks=indicating_mask,
        cfg=cfg,
        metadata_dict=metadata_dict,
        checks_on=checks_on,
    )

    # Get global metrics (as in averaged over all the subjects)
    metrics_global = imputation_metrics_wrapper(
        predictions=predictions,
        targets=targets,
        masks=indicating_mask,
        subject_code="global",
    )

    # Compute CIs from subjectwise metrics as we did with the anomaly detection
    metrics_subjectwise_arrays, metrics_global = compute_CI_imputation_metrics(
        metrics_subjectwise, metrics_global
    )

    return {
        "global": metrics_global,
        "subjectwise": metrics_subjectwise,
        "subjectwise_arrays": metrics_subjectwise_arrays,
    }


def compute_CI_imputation_metrics(
    metrics_subjectwise: Dict, metrics_global: Dict, p: float = 0.05
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Compute confidence intervals for imputation metrics from subjectwise values.

    Parameters
    ----------
    metrics_subjectwise : dict
        Dictionary mapping subject codes to their metric dictionaries.
    metrics_global : dict
        Global metrics dictionary to augment with CI values.
    p : float, optional
        Percentile for CI bounds (default 0.05 for 5th and 95th percentiles).

    Returns
    -------
    tuple
        (metrics_subjectwise_arrays, metrics_global) where arrays contain per-metric
        numpy arrays and global dict includes CI bounds.
    """
    metrics_subjectwise_arrays = get_arrays_from_subject_dicts(metrics_subjectwise)
    for metric_key, value_array in metrics_subjectwise_arrays.items():
        ci = np.nanpercentile(value_array, [p, 100 - p])
        metrics_global[f"{metric_key}_CI"] = ci

    return metrics_subjectwise_arrays, metrics_global


def get_arrays_from_subject_dicts(metrics_subjectwise: Dict) -> Dict[str, np.ndarray]:
    """
    Convert subjectwise metric dictionaries to arrays per metric.

    Parameters
    ----------
    metrics_subjectwise : dict
        Dictionary mapping subject codes to their metric dictionaries.

    Returns
    -------
    dict
        Dictionary mapping metric names to numpy arrays of values across subjects.
    """
    metrics = {}
    for i, (code, metric_dict) in enumerate(metrics_subjectwise.items()):
        if i == 0:
            for metric in metric_dict.keys():
                metrics[metric] = [metric_dict[metric]]
        else:
            for metric in metric_dict.keys():
                metrics[metric].append(metric_dict[metric])

    for metric in metrics.keys():
        metrics[metric] = np.array(metrics[metric])

    return metrics


def subjectwise_metrics_wrapper(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    cfg: DictConfig,
    metadata_dict: Dict,
    checks_on: bool = False,
) -> Dict[str, Dict]:
    """
    Compute imputation metrics for each subject individually.

    Parameters
    ----------
    predictions : np.ndarray
        Imputed predictions, shape (n_subjects, n_timepoints, n_features).
    targets : np.ndarray
        Ground truth values, shape (n_subjects, n_timepoints, n_features).
    masks : np.ndarray
        Binary mask indicating missing values, shape (n_subjects, n_timepoints, n_features).
    cfg : DictConfig
        Hydra configuration object.
    metadata_dict : dict
        Metadata containing subject codes.
    checks_on : bool, optional
        Whether to run prechecks for NaN handling (default False).

    Returns
    -------
    dict
        Dictionary mapping subject codes to their metric dictionaries.
    """
    no_subjects = predictions.shape[0]
    assert (
        metadata_dict["subject_code"].shape[0] == no_subjects
    ), "Number of subjects should match, {} subjects imputed, and {} metadata subjects".format(
        no_subjects, metadata_dict["subject_code"].shape[0]
    )

    metrics_subjectwise = {}
    for i in range(no_subjects):
        subject_code = str(metadata_dict["subject_code"][i, 0])
        X, Y, mask = get_subjectwise_arrays(predictions, targets, masks, i)
        metrics_subjectwise[subject_code] = imputation_metrics_wrapper(
            predictions=X,
            targets=Y,
            masks=mask,
            subject_code=subject_code,
            prechecks=checks_on,
        )

    return metrics_subjectwise


def check_target_pred_ratio(
    targets: np.ndarray, predictions: np.ndarray, subject_code: str
) -> None:
    """
    Check for scale mismatch between targets and predictions.

    Logs warnings if the ratio between prediction and target means is infinite
    or NaN, which may indicate standardization issues.

    Parameters
    ----------
    targets : np.ndarray
        Ground truth values.
    predictions : np.ndarray
        Imputed predictions.
    subject_code : str
        Subject identifier for logging purposes.

    Returns
    -------
    None
    """
    # if the other is standardized, and the other is destandardized
    target_mean = np.mean(targets)
    predictions_mean = np.mean(predictions)
    ratio = predictions_mean / target_mean

    if np.isinf(ratio):
        logger.debug(
            "Ratio between prediction mean ({}), and target mean ({}) is {}".format(
                predictions_mean, target_mean, ratio
            )
        )
        if np.isinf(predictions_mean):
            logger.debug("Predictions did not come out okay with the infinite values!")
            no_of_infs = np.sum(np.isinf(predictions))
            logger.warning(
                "{} | {:.2f}% out of predictions are np.inf".format(
                    subject_code, 100 * (no_of_infs / predictions.size)
                )
            )

    if np.isnan(ratio):
        logger.warning(
            "Ratio between prediction mean ({}), and target mean ({}) is {}".format(
                predictions_mean, target_mean, ratio
            )
        )


def remove_NaNs_from_triplet(
    X: np.ndarray, Y: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove NaN values from predictions, targets, and mask arrays by cropping.

    Handles NaNs that may occur from padding in models like MOMENT.

    Parameters
    ----------
    X : np.ndarray
        Predictions array.
    Y : np.ndarray
        Targets array.
    mask : np.ndarray
        Indicating mask array.

    Returns
    -------
    tuple
        (X, Y, mask) with NaN regions cropped out.
    """

    def crop_arrays(x: np.ndarray, nonnan_mask: np.ndarray) -> np.ndarray:
        # Crop the arrays based on the non-NaN mask from the X
        coords = np.argwhere(nonnan_mask)
        x_min, y_min, _ = coords.min(axis=0)
        x_max, y_max, _ = coords.max(axis=0)
        cropped = x[:, y_min : y_max + 1]
        return cropped

    # There might be NaNs in the predictions if you used e.g. Moment and had to trim and pad the data
    # Trim the arrays baaed on the non-NaN mask from the X
    nonnan_mask = ~np.isnan(X)
    X = crop_arrays(X, nonnan_mask)
    Y = crop_arrays(Y, nonnan_mask)
    mask = crop_arrays(mask, nonnan_mask)

    return X, Y, mask


def check_for_nan_subjects(
    X: np.ndarray, Y: np.ndarray, mask: np.ndarray, return_nanfree: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Check for and optionally remove subjects with NaN values.

    Parameters
    ----------
    X : np.ndarray
        Predictions array, shape (n_subjects, n_timepoints, n_features).
    Y : np.ndarray
        Targets array, shape (n_subjects, n_timepoints, n_features).
    mask : np.ndarray
        Indicating mask array, shape (n_subjects, n_timepoints, n_features).
    return_nanfree : bool, optional
        If True, return arrays with NaN subjects removed (default False).

    Returns
    -------
    tuple
        (X, Y, mask) optionally filtered to exclude subjects with NaNs.
    """

    def get_nan_subjects(X: np.ndarray) -> np.ndarray:
        squeezed_X = np.squeeze(X)  # e.g. (1981,) when just one subject
        if len(squeezed_X.shape) == 1:
            subject_sums = np.array((np.count_nonzero(np.isnan(squeezed_X))))
        elif len(squeezed_X.shape) == 2:
            subject_sums = np.count_nonzero(np.isnan(squeezed_X), axis=1)
        else:
            logger.error(
                "Why do you have more than 2 dimensions, multiple channels/features?"
            )
        subject_is_nanfree = subject_sums == 0
        return subject_is_nanfree

    subject_is_nanfree = get_nan_subjects(X)
    number_of_nan_subjects = np.sum(~subject_is_nanfree)
    if number_of_nan_subjects > 0:
        logger.warning(
            f"Found {number_of_nan_subjects} subjects with NaNs, removing them before computing metrics"
        )
        logger.warning(
            "Try to figure out why this happened, now your metrics are not obviously "
            "comparable to other methods as you do not use all the samples!"
        )
    # return only these, 1st dimension is the subjects from a 3d array
    if return_nanfree:
        X = X[subject_is_nanfree, :, :]
        Y = Y[subject_is_nanfree, :, :]
        mask = mask[subject_is_nanfree, :, :]

    return X, Y, mask


def imputation_metrics_wrapper(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    subject_code: str,
    prechecks: bool = False,
) -> Dict:
    """
    Compute imputation metrics (MAE, MSE, MRE) using PyPOTS utilities.

    Parameters
    ----------
    predictions : np.ndarray
        Imputed predictions array.
    targets : np.ndarray
        Ground truth values array.
    masks : np.ndarray
        Binary mask indicating missing values.
    subject_code : str
        Subject identifier or 'global' for aggregate metrics.
    prechecks : bool, optional
        Whether to run NaN removal and validation checks (default False).

    Returns
    -------
    dict
        Dictionary with 'mae', 'mse', 'mre', 'missing_rate' keys.
    """
    # This will import the annoying Timeseries ASCII logo so keep it here
    # TODO! replace these to get rid of the logo after each imputation method (not just PyPots)
    from pypots.utils.metrics import calc_mae, calc_mre, calc_mse

    if prechecks:
        try:
            predictions, targets, masks = remove_NaNs_from_triplet(
                X=predictions, Y=targets, mask=masks
            )
            predictions, targets, masks = check_for_nan_subjects(
                X=predictions, Y=targets, mask=masks
            )
            check_target_pred_ratio(targets, predictions, subject_code)

        except Exception as e:
            logger.error(f"Failed to run the prechecks, {e}")
            raise ValueError(f"Failed to run the prechecks, {e}")

    metrics = {}
    try:
        # MAE (Mean Absolute Error)
        metrics["mae"] = calc_mae(predictions=predictions, targets=targets, masks=masks)
        if np.isnan(metrics["mae"]):
            logger.warning("MAE is NaN for subject_code = {}".format(subject_code))
        # MSE (Mean Square Error)
        metrics["mse"] = calc_mse(predictions=predictions, targets=targets, masks=masks)
        # MRE (Mean Relative Error)
        metrics["mre"] = calc_mre(predictions=predictions, targets=targets, masks=masks)
        # Simply add here your favorite metrics, and save with a new key to the metrics dict
        metrics["missing_rate"] = np.mean(masks)
    except Exception as e:
        logger.error(f"Failed to compute the metrics: {e}")
        metrics["failed_metrics"] = 101010101010

    return metrics


def if_recompute_metrics(metrics_path: str, _metrics_cfg: DictConfig) -> bool:
    """
    Determine whether to recompute imputation metrics.

    Currently a placeholder that always returns True.

    Parameters
    ----------
    metrics_path : str
        Path to existing metrics file.
    _metrics_cfg : DictConfig
        Metrics configuration (unused).

    Returns
    -------
    bool
        Always returns True (recompute metrics).
    """
    logger.debug("Placeholder, always recompute the metrics")
    return True
