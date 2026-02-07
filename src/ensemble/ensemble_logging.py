"""
Ensemble MLflow logging module.

Provides utilities for logging ensemble results to MLflow,
including metrics, artifacts, and run naming.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from src.classification.classifier_log_utils import (
    classifier_log_cls_evaluation_to_mlflow,
)
from src.ensemble.ensemble_anomaly_detection import write_granular_outlier_metrics
from src.log_helpers.local_artifacts import save_results_dict
from src.utils import get_artifacts_dir


def get_ensemble_pickle_name(ensemble_name: str) -> str:
    """
    Generate pickle filename for ensemble results.

    Parameters
    ----------
    ensemble_name : str
        Name of the ensemble.

    Returns
    -------
    str
        Filename in format 'ensemble_{name}_results.pickle'.
    """
    return f"ensemble_{ensemble_name}_results.pickle"


def get_source_runs(
    ensemble_mlflow_runs_per_name: Union[Dict[str, Dict[str, Any]], pd.DataFrame],
) -> List[str]:
    """
    Extract run IDs from ensemble submodel data.

    Parameters
    ----------
    ensemble_mlflow_runs_per_name : dict or pd.DataFrame
        Submodel data (dict of dicts or DataFrame).

    Returns
    -------
    list
        List of MLflow run IDs.
    """
    run_ids = []
    if isinstance(ensemble_mlflow_runs_per_name, dict):
        for model_name, model_dict in ensemble_mlflow_runs_per_name.items():
            run_ids.append(model_dict["run_id"])
    elif isinstance(ensemble_mlflow_runs_per_name, pd.DataFrame):
        for idx, row in ensemble_mlflow_runs_per_name.iterrows():
            run_ids.append(row["run_id"])

    return run_ids


def get_ensemble_name(
    runs_per_name: Union[pd.DataFrame, Dict[str, Any]],
    ensemble_name_base: str,
    ensemble_prefix_str: str,
    sort_name: str = "params.model",
) -> str:
    """
    Generate ensemble run name from submodel names.

    Parameters
    ----------
    runs_per_name : pd.DataFrame or dict
        Submodel runs data.
    ensemble_name_base : str
        Base name (e.g., source like 'pupil_gt').
    ensemble_prefix_str : str
        Prefix (e.g., 'ensemble' or 'ensembleThresholded').
    sort_name : str, default 'params.model'
        Column/key to sort models by.

    Returns
    -------
    str
        Ensemble name in format '{prefix}-{model1}-{model2}...{__{base}}'.
    """
    if isinstance(runs_per_name, pd.DataFrame):
        ensemble_name = ""
        runs_per_name = runs_per_name.sort_values(by=sort_name)
        # runs_Series = runs_per_name.iloc[0]
        model_names = runs_per_name[sort_name].unique()
        for i, name in enumerate(model_names):
            if i == len(model_names) - 1:
                ensemble_name += name
            else:
                ensemble_name += name + "-"
    elif isinstance(runs_per_name, dict):
        runs_per_name = dict(sorted(runs_per_name.items()))
        ensemble_name = "-".join([model_name for model_name in runs_per_name.keys()])
    else:
        raise ValueError("runs_per_name must be a DataFrame or a dictionary")

    # following steps expect imputer_model__data_source type of logic
    return f"{ensemble_prefix_str}-" + ensemble_name + "__" + ensemble_name_base


def log_ensemble_metrics(metrics: Dict[str, Any], task: str) -> None:
    """
    Log ensemble metrics to MLflow.

    Handles different metric structures for anomaly detection and imputation.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary.
    task : str
        Task type ('anomaly_detection' or 'imputation').
    """
    if task == "anomaly_detection":
        write_granular_outlier_metrics(metrics)
        # for metric, value in metrics[split].items():
        #     if value is not None:
        #         mlflow.log_metric(f"{split}/{metric}", value)
    elif task == "imputation":
        splits = metrics.keys()
        for split in splits:
            metric_dict = metrics[split]["global"]
            for metric, value in metric_dict.items():
                if value is not None:
                    try:
                        if isinstance(value, np.ndarray):
                            mlflow.log_metric(f"{split}/{metric}_lo", value[0])
                            mlflow.log_metric(f"{split}/{metric}_hi", value[1])
                        else:
                            mlflow.log_metric(f"{split}/{metric}", value)
                    except Exception as e:
                        logger.error(f"Failed to log metric {metric}: {e}")
                        raise e


def log_ensemble_arrays(
    pred_masks: Dict[str, Any], task: str, ensemble_name: str
) -> None:
    """
    Save and log ensemble arrays as MLflow artifact.

    Parameters
    ----------
    pred_masks : dict
        Prediction data to save.
    task : str
        Task type for artifact subdirectory.
    ensemble_name : str
        Name for the pickle file.
    """
    artifact_dir = Path(get_artifacts_dir(service_name=task))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    results_path = artifact_dir / get_ensemble_pickle_name(ensemble_name)
    save_results_dict(results_dict=pred_masks, results_path=str(results_path))
    if task == "anomaly_detection":
        mlflow.log_artifact(
            results_path, "outlier_detection"
        )  # TODO! pick which to call this
    elif task == "imputation":
        mlflow.log_artifact(results_path, task)
    else:
        raise NotImplementedError("Classification metrics not yet implemented")


def ensemble_is_empty(
    ensemble_mlflow_runs: Dict[str, Union[Dict[str, Any], pd.DataFrame]],
    ensemble_name: str,
) -> bool:
    """
    Check if ensemble has no submodels.

    Parameters
    ----------
    ensemble_mlflow_runs : dict
        Dictionary of ensemble runs.
    ensemble_name : str
        Name of ensemble to check.

    Returns
    -------
    bool
        True if ensemble is empty.
    """
    if isinstance(ensemble_mlflow_runs[ensemble_name], dict):
        ensemble_is_empty = len(ensemble_mlflow_runs[ensemble_name]) == 0
    elif isinstance(ensemble_mlflow_runs[ensemble_name], pd.DataFrame):
        ensemble_is_empty = ensemble_mlflow_runs[ensemble_name].empty
    else:
        logger.error(
            "ensemble_mlflow_runs[ensemble_name] must be a DataFrame or a dictionary"
        )
        raise ValueError(
            "ensemble_mlflow_runs[ensemble_name] must be a DataFrame or a dictionary"
        )

    return ensemble_is_empty


def get_sort_name(task: str) -> str:
    """
    Get parameter name for sorting models by task.

    Parameters
    ----------
    task : str
        Task type.

    Returns
    -------
    str
        MLflow parameter column name for model name.
    """
    if task == "classification":
        sort_name = "params.model_name"
    else:
        sort_name = "params.model"

    return sort_name


def get_ensemble_quality_threshold(task: str, cfg: DictConfig) -> Optional[float]:
    """
    Get quality threshold for ensemble submodel selection.

    Parameters
    ----------
    task : str
        Task type.
    cfg : DictConfig
        Main Hydra configuration.

    Returns
    -------
    float or None
        Quality threshold value, or None if not applicable.
    """
    if task == "anomaly_detection":
        ensemble_quality_threshold = cfg["OUTLIER_DETECTION"]["best_metric"][
            "ensemble_quality_threshold"
        ]
    elif task == "imputation":
        ensemble_quality_threshold = cfg["IMPUTATION_METRICS"]["best_metric"][
            "ensemble_quality_threshold"
        ]
    else:
        ensemble_quality_threshold = None

    return ensemble_quality_threshold


def get_ensemble_prefix(task: str, cfg: DictConfig) -> str:
    """
    Get prefix string for ensemble name based on quality thresholding.

    Parameters
    ----------
    task : str
        Task type.
    cfg : DictConfig
        Main Hydra configuration.

    Returns
    -------
    str
        'ensembleThresholded' if threshold set, 'ensemble' otherwise.
    """
    threshold = get_ensemble_quality_threshold(task, cfg)
    if threshold is not None:
        ensemble_prefix_str = "ensembleThresholded"
    else:
        ensemble_prefix_str = "ensemble"

    return ensemble_prefix_str


def get_mlflow_ensemble_name(
    task: str,
    ensemble_mlflow_runs: Dict[str, Union[Dict[str, Any], pd.DataFrame]],
    ensemble_name: str,
    cfg: DictConfig,
) -> Optional[str]:
    """
    Generate full MLflow run name for ensemble.

    Parameters
    ----------
    task : str
        Task type.
    ensemble_mlflow_runs : dict
        Dictionary of ensemble submodel runs.
    ensemble_name : str
        Source name (e.g., 'pupil_gt').
    cfg : DictConfig
        Main Hydra configuration.

    Returns
    -------
    str or None
        Full ensemble name (e.g., 'ensemble-CatBoost-XGBoost__pupil_gt'),
        or None if ensemble is empty.
    """
    sort_name = get_sort_name(task)
    ensemble_prefix_str = get_ensemble_prefix(task, cfg)

    if not ensemble_is_empty(ensemble_mlflow_runs, ensemble_name):
        mlflow_ensemble_name = get_ensemble_name(
            runs_per_name=ensemble_mlflow_runs[ensemble_name],
            ensemble_name_base=ensemble_name,
            sort_name=sort_name,
            ensemble_prefix_str=ensemble_prefix_str,
        )
    else:
        mlflow_ensemble_name = None

    return mlflow_ensemble_name


def get_existing_runs(
    experiment_name: str, mlflow_ensemble_name: str
) -> Tuple[pd.DataFrame, bool]:
    """
    Check for existing MLflow runs with same name.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name.
    mlflow_ensemble_name : str
        Ensemble run name to search for.

    Returns
    -------
    pd.DataFrame
        Matching runs.
    bool
        True if matching runs exist.
    """
    mlflow_runs = mlflow.search_runs(experiment_names=[experiment_name])
    if mlflow_runs.shape[0] > 0:
        runs = mlflow_runs[mlflow_runs["tags.mlflow.runName"] == mlflow_ensemble_name]
        if runs.shape[0] > 0:
            old_exists = True
        else:
            old_exists = False
        return runs, old_exists


def check_for_old_run(
    experiment_name: str,
    mlflow_ensemble_name: str,
    cfg: DictConfig,
    delete_old_mlflow_run: bool = True,
) -> bool:
    """
    Check for and optionally delete existing ensemble runs.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name.
    mlflow_ensemble_name : str
        Ensemble run name.
    cfg : DictConfig
        Main Hydra configuration.
    delete_old_mlflow_run : bool, default True
        If True, delete existing runs with same name.

    Returns
    -------
    bool
        True if logging should continue.
    """
    runs, old_exists = get_existing_runs(experiment_name, mlflow_ensemble_name)
    if old_exists:
        if delete_old_mlflow_run:
            for run_id in runs["run_id"]:
                logger.warning(
                    'Delete old run = "{}", id = "{}"'.format(
                        mlflow_ensemble_name, run_id
                    )
                )
                mlflow.delete_run(run_id)
        else:
            logger.warning(
                f"Run {mlflow_ensemble_name} already exists and you chose not to delete it!"
            )

    if mlflow_ensemble_name is not None:
        continue_with_logging = True
    else:
        continue_with_logging = False
    logger.warning("continue_with_logging always now True!")

    return continue_with_logging


def log_ensembling_to_mlflow(
    experiment_name: str,
    ensemble_mlflow_runs: Dict[str, Union[Dict[str, Any], pd.DataFrame]],
    ensemble_name: str,
    cfg: DictConfig,
    task: str,
    metrics: Optional[Dict[str, Any]] = None,
    pred_masks: Optional[Dict[str, Any]] = None,
    output_dict: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log ensemble results to MLflow.

    Creates a new MLflow run for the ensemble, logs metrics, parameters,
    and artifacts based on task type.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name.
    ensemble_mlflow_runs : dict
        Dictionary of ensemble submodel runs.
    ensemble_name : str
        Source name for the ensemble.
    cfg : DictConfig
        Main Hydra configuration.
    task : str
        Task type ('anomaly_detection', 'imputation', or 'classification').
    metrics : dict, optional
        Pre-computed metrics (for anomaly detection).
    pred_masks : dict, optional
        Prediction masks (for anomaly detection).
    output_dict : dict, optional
        Full output dictionary (for imputation/classification).
    """
    # Log the ensembling results to MLflow
    run_ids = get_source_runs(
        ensemble_mlflow_runs_per_name=ensemble_mlflow_runs[ensemble_name]
    )

    if not ensemble_is_empty(ensemble_mlflow_runs, ensemble_name):
        mlflow_ensemble_name = get_mlflow_ensemble_name(
            task=task,
            ensemble_mlflow_runs=ensemble_mlflow_runs,
            ensemble_name=ensemble_name,
            cfg=cfg,
        )
        continue_with_logging = check_for_old_run(
            experiment_name, mlflow_ensemble_name, cfg
        )
        if continue_with_logging:
            with mlflow.start_run(run_name=mlflow_ensemble_name):
                mlflow.log_param("ensemble_run_ids", run_ids)
                mlflow.log_param("model", "ensemble")
                if task == "anomaly_detection":
                    dict_of_arrays_out = pred_masks
                elif task == "imputation":
                    dict_of_arrays_out = output_dict
                    metrics = dict_of_arrays_out["metrics"]
                elif task == "classification":
                    dict_of_arrays_out = output_dict
                    metrics = output_dict
                else:
                    logger.error(f"Unknown task: {task}")
                    raise ValueError(f"Unknown task: {task}")

                if task == "classification":
                    classifier_log_cls_evaluation_to_mlflow(
                        None,
                        None,
                        None,
                        metrics,
                        None,
                        None,
                        run_name=mlflow_ensemble_name,
                        model_name="ensemble",
                    )
                else:
                    log_ensemble_metrics(metrics, task)
                    log_ensemble_arrays(dict_of_arrays_out, task, mlflow_ensemble_name)
                mlflow.end_run()

    else:
        logger.warning(f"No runs found for ensemble {ensemble_name}")
        return None
