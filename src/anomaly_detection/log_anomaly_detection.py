from pathlib import Path

import mlflow
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from src.anomaly_detection.anomaly_utils import (
    check_outlier_detection_artifact,
    get_no_subjects_in_outlier_artifacts,
)
from src.log_helpers.local_artifacts import save_array_as_csv, save_results_dict
from src.log_helpers.log_naming_uris_and_dirs import (
    get_outlier_csv_name,
    get_outlier_pickle_name,
)
from src.utils import get_artifacts_dir


def log_anomaly_metrics(metrics: dict, cfg: DictConfig):
    """
    Log outlier detection metrics to MLflow.

    Parameters
    ----------
    metrics : dict
        Dictionary of metrics per split containing 'scalars' with 'global' values.
    cfg : DictConfig
        Hydra configuration (currently unused).

    Notes
    -----
    For array values (confidence intervals), logs separate _lo and _hi metrics.
    """
    logger.info("Logging Outlier detection metrics to MLflow")
    for split, split_metrics in metrics.items():
        if "global" in split_metrics["scalars"]:
            global_scalars = split_metrics["scalars"]["global"]
            for scalar_key, value in global_scalars.items():
                if value is not None:
                    metric_key = f"{split}/{scalar_key}"
                    if isinstance(value, np.ndarray):
                        mlflow.log_metric(metric_key + "_lo", value[0])
                        mlflow.log_metric(metric_key + "_hi", value[1])
                    else:
                        mlflow.log_metric(metric_key, value)


def log_losses(best_outlier_results: dict, cfg: DictConfig, best_epoch: int):
    """
    Log reconstruction losses to MLflow.

    Parameters
    ----------
    best_outlier_results : dict
        Results from the best epoch containing per-split loss arrays.
    cfg : DictConfig
        Hydra configuration (currently unused).
    best_epoch : int
        Index of the best training epoch.

    Raises
    ------
    ValueError
        If no losses are found for a split.
    """
    logger.info("Logging Outlier detection losses (MSE) to MLflow")
    for split, split_metrics in best_outlier_results.items():
        flat_arrays = split_metrics["results_dict"]["metrics"]["arrays_flat"]
        if "losses" in flat_arrays.keys():
            metric_key = f"{split}/{'loss'}"
            array = flat_arrays["losses"]
            best_loss = np.mean(array)
            assert isinstance(best_loss, float), (
                f"best_loss is not a float: {best_loss}"
            )
            mlflow.log_metric(metric_key, best_loss)

        else:
            logger.error(f"No losses found for split {split}, nothing logged!")
            raise ValueError(f"No losses found for split {split}, nothing logged!")


def log_anomaly_predictions(
    model_name: str, preds: dict, cfg: DictConfig, transpose: bool = True
):
    """
    Log outlier detection predictions to MLflow as CSV files.

    Parameters
    ----------
    model_name : str
        Name of the model for filename generation.
    preds : dict
        Dictionary of predictions per split containing 'arrays'.
    cfg : DictConfig
        Hydra configuration (currently unused).
    transpose : bool, optional
        Whether to transpose arrays before saving. Default is True.
    """
    logger.info("Logging Outlier detection predictions to MLflow as CSV")
    for split, split_preds in preds.items():
        artifacts_dir = Path(get_artifacts_dir(service_name="outlier_detection"))
        for key, array in split_preds["arrays"].items():
            csv_path = artifacts_dir / get_outlier_csv_name(model_name, split, key)
            if array is not None:
                if transpose:
                    array = array.T
                if csv_path.exists():
                    csv_path.unlink()
                save_array_as_csv(array=array, path=str(csv_path))
                mlflow.log_artifact(str(csv_path), "outlier_detection")


def check_debug_n_subjects_outlier_artifacts(outlier_artifacts, cfg):
    """
    Validate subject count in debug mode.

    Parameters
    ----------
    outlier_artifacts : dict
        Outlier detection artifacts to validate.
    cfg : DictConfig
        Hydra configuration with debug settings.

    Raises
    ------
    ValueError
        If subject count doesn't match expected debug count.
    """
    no_subjects_out = get_no_subjects_in_outlier_artifacts(outlier_artifacts)
    if cfg["EXPERIMENT"]["debug"]:
        if no_subjects_out != cfg["DEBUG"]["debug_n_subjects"] * 2:
            logger.error(
                "Number of subjects in the outlier artifacts ({}) does not match the "
                "number of subjects in the "
                "experiment ({})".format(
                    no_subjects_out, cfg["EXPERIMENT"]["no_subjects"]
                )
            )
            raise ValueError(
                "Number of subjects in the outlier artifacts ({}) does not match the "
                "number of subjects in the "
                "experiment ({})".format(
                    no_subjects_out, cfg["EXPERIMENT"]["no_subjects"]
                )
            )


def log_outlier_artifacts_dict(
    model_name, outlier_artifacts, cfg, checks_on: bool = True
):
    """
    Save and log outlier detection artifacts to MLflow.

    Parameters
    ----------
    model_name : str
        Name of the model for filename generation.
    outlier_artifacts : dict
        Complete outlier detection results to save.
    cfg : DictConfig
        Hydra configuration.
    checks_on : bool, optional
        Whether to run validation checks. Default is True.
    """
    artifact_dir = Path(get_artifacts_dir(service_name="outlier_detection"))
    results_path = artifact_dir / get_outlier_pickle_name(model_name)
    logger.debug(
        "Saving the imputation results as a pickled artifact: {}".format(results_path)
    )
    if checks_on:
        # these for Moment basically
        check_outlier_detection_artifact(outlier_artifacts)
        # check_debug_n_subjects_outlier_artifacts(outlier_artifacts, cfg)
    save_results_dict(
        results_dict=outlier_artifacts,
        results_path=str(results_path),
        name="outlier_detection",
    )
    logger.debug("And logging to MLflow as an artifact")
    mlflow.log_artifact(str(results_path), "outlier_detection")


def log_anomaly_detection_to_mlflow(
    model_name: str, run_name: str, outlier_artifacts: dict, cfg: DictConfig
):
    """
    Log complete anomaly detection results to MLflow.

    Orchestrates logging of metrics, losses, predictions, and artifacts.

    Parameters
    ----------
    model_name : str
        Name of the model.
    run_name : str
        MLflow run name.
    outlier_artifacts : dict
        Complete outlier detection results containing:
        - 'metadata': with 'best_epoch'
        - 'outlier_results': per-epoch results
        - 'metrics': evaluation metrics
        - 'preds': predictions
    cfg : DictConfig
        Hydra configuration.

    Notes
    -----
    Ends the MLflow run after logging all artifacts.
    """
    best_epoch = outlier_artifacts["metadata"]["best_epoch"]
    if best_epoch is not None:
        # finetune
        best_outlier_results = outlier_artifacts["outlier_results"][best_epoch]
    else:
        # zero-shot
        first_key = list(outlier_artifacts["outlier_results"].keys())[0]
        best_outlier_results = outlier_artifacts["outlier_results"][first_key]

    log_anomaly_metrics(metrics=outlier_artifacts["metrics"], cfg=cfg)
    log_losses(
        best_outlier_results=best_outlier_results,
        cfg=cfg,
        best_epoch=best_epoch,
    )
    log_anomaly_predictions(model_name, preds=outlier_artifacts["preds"], cfg=cfg)
    log_outlier_artifacts_dict(model_name, outlier_artifacts, cfg)
    # TODO! https://medium.com/@ij_82957/how-to-reduce-mlflow-logging-overhead-by-using-log-batch-b61301cc540f
    #  you could loop through the dict for each epoch and get the average loss of the batch and have the training
    #  history, or just Tensorboard?
    mlflow.end_run()
