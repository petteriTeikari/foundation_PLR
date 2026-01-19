import os
from loguru import logger
import mlflow
import pandas as pd
from omegaconf import DictConfig

from src.classification.classifier_log_utils import (
    classifier_log_cls_evaluation_to_mlflow,
)
from src.log_helpers.local_artifacts import save_results_dict
from src.utils import get_artifacts_dir


def get_ensemble_pickle_name(ensemble_name: str) -> str:
    return f"ensemble_{ensemble_name}_results.pickle"


def get_source_runs(ensemble_mlflow_runs_per_name) -> list:
    run_ids = []
    if isinstance(ensemble_mlflow_runs_per_name, dict):
        for model_name, model_dict in ensemble_mlflow_runs_per_name.items():
            run_ids.append(model_dict["run_id"])
    elif isinstance(ensemble_mlflow_runs_per_name, pd.DataFrame):
        for idx, row in ensemble_mlflow_runs_per_name.iterrows():
            run_ids.append(row["run_id"])

    return run_ids


def get_ensemble_name(
    runs_per_name, ensemble_name_base: str, sort_name: str = "params.model"
) -> str:
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
    return "ensemble-" + ensemble_name + "__" + ensemble_name_base


def log_ensemble_metrics(metrics, task):
    for split in metrics.keys():
        if task == "anomaly_detection":
            for metric, value in metrics[split].items():
                if value is not None:
                    mlflow.log_metric(f"{split}/{metric}", value)
        elif task == "imputation":
            for metric, value in metrics[split]["global"].items():
                if value is not None:
                    mlflow.log_metric(f"{split}/{metric}", value)


def log_ensemble_arrays(pred_masks, task, ensemble_name):
    artifact_dir = get_artifacts_dir(service_name=task)
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir, exist_ok=True)
    results_path = os.path.join(artifact_dir, get_ensemble_pickle_name(ensemble_name))
    save_results_dict(results_dict=pred_masks, results_path=results_path)
    if task == "anomaly_detection":
        mlflow.log_artifact(
            results_path, "outlier_detection"
        )  # TODO! pick which to call this
    elif task == "imputation":
        mlflow.log_artifact(results_path, task)
    else:
        raise NotImplementedError("Classification metrics not yet implemented")


def get_mlflow_ensemble_name(task, ensemble_mlflow_runs, ensemble_name):
    """
    ensemble_name: str (source)
        e.g. "pupil_gt"
    mlflow_ensemble_name: str (model, source)
        e.g. "ensemble-CatBoost-LogisticRegression-TabM-XGBoost__pupil_gt"
    """
    if task == "classification":
        sort_name = "params.model_name"
    else:
        sort_name = "params.model"

    if not ensemble_mlflow_runs[ensemble_name].empty:
        mlflow_ensemble_name = get_ensemble_name(
            runs_per_name=ensemble_mlflow_runs[ensemble_name],
            ensemble_name_base=ensemble_name,
            sort_name=sort_name,
        )
    else:
        mlflow_ensemble_name = None

    return mlflow_ensemble_name


def get_existing_runs(experiment_name, mlflow_ensemble_name):
    mlflow_runs = mlflow.search_runs(experiment_names=[experiment_name])
    if mlflow_runs.shape[0] > 0:
        runs = mlflow_runs[mlflow_runs["tags.mlflow.runName"] == mlflow_ensemble_name]
        if runs.shape[0] > 0:
            old_exists = True
        else:
            old_exists = False
        return runs, old_exists


def check_for_old_run(
    experiment_name, mlflow_ensemble_name, cfg, delete_old_mlflow_run: bool = True
):
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
    ensemble_mlflow_runs: dict,
    ensemble_name: str,
    cfg: DictConfig,
    task: str,
    metrics: dict = None,
    pred_masks: dict = None,
    output_dict: dict = None,
):
    # Log the ensembling results to MLflow
    run_ids = get_source_runs(
        ensemble_mlflow_runs_per_name=ensemble_mlflow_runs[ensemble_name]
    )

    if not ensemble_mlflow_runs[ensemble_name].empty:
        mlflow_ensemble_name = get_mlflow_ensemble_name(
            task=task,
            ensemble_mlflow_runs=ensemble_mlflow_runs,
            ensemble_name=ensemble_name,
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
                        run_name=ensemble_name,
                        model_name="ensemble",
                    )
                else:
                    log_ensemble_metrics(metrics, task)
                    log_ensemble_arrays(dict_of_arrays_out, task, ensemble_name)
                mlflow.end_run()

    else:
        logger.warning(f"No runs found for ensemble {ensemble_name}")
        return None
