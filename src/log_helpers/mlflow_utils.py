import numpy as np
from loguru import logger
import os
from omegaconf import DictConfig

import mlflow
from mlflow.tracking import MlflowClient

from src.log_helpers.hydra_utils import (
    get_intermediate_hydra_log_path,
    get_hydra_output_dir,
    save_hydra_cfg_as_yaml,
    log_the_hydra_log_as_mlflow_artifact,
)
from src.log_helpers.local_artifacts import save_results_dict
from src.log_helpers.log_naming_uris_and_dirs import get_mlflow_metric_name
from src.log_helpers.mlflow_artifacts import (
    get_mlflow_info_from_model_dict,
    get_mlflow_params,
    get_best_previous_mlflow_logged_model,
    what_to_search_from_mlflow,
    get_best_metric_from_current_run,
)

from src.log_helpers.system_utils import get_system_param_dict

from src.imputation.pypots.pypots_utils import define_pypots_outputs
from src.utils import get_artifacts_dir
from tests.mlflow_tests import test_artifact_write


def init_mlflow(cfg: DictConfig):
    # Set the MLflow tracking URI (export MLFLOW_TRACKING_URI='file:////home/petteri/Dropbox/mlruns')
    if cfg["SERVICES"]["mlflow_tracking_uri"] is not None:
        mlflow.set_tracking_uri(cfg["SERVICES"]["mlflow_tracking_uri"])
    else:
        logger.warning(
            "You did not specify any MLflow tracking URI. Using the 'mlruns' dir inside 'src'"
        )
    logger.info(f"{mlflow.get_tracking_uri()}")


def init_mlflow_experiment(
    mlflow_cfg: DictConfig = None,
    experiment_name: str = "PLR_imputation",
    override_default_location: bool = False,
    permanent_delete: bool = True,
):
    # https://mlflow.org/docs/latest/getting-started/logging-first-model/step3-create-experiment.html
    if override_default_location:
        logger.info("Overriding default MLflow location")
        logger.warning(
            'Leads to permission denied error?! Set "override_default_location = True" in the code'
        )
        mlruns_dir = get_artifacts_dir("mlflow", "mlruns")
        mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    else:
        logger.debug("Using default MLflow location")

    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.error(f"Failed to set MLflow experiment: {e}")
        logger.error(
            "Failed to set MLflow experiment, but not auto-deleting the experiment. Solve this manually"
        )
        logger.error("See e.g. https://stackoverflow.com/a/60869104/6412152")
        logger.error("e.g. '' or 'mlflow gc [OPTIONS]'")
        raise e
    logger.info(
        f"MLflow | Initializing MLflow Experiment tracking (Server) at {mlflow.get_tracking_uri()}"
    )
    set_artifact_store_location()


def set_artifact_store_location():
    # https://mlflow.org/docs/latest/tracking/artifacts-stores.html
    # TODO! Some remote, e.g. S3
    return None


def init_mlflow_run(mlflow_cfg, run_name: str, cfg: DictConfig, experiment_name: str):
    try:
        mlflow.start_run(
            run_name=run_name, log_system_metrics=mlflow_cfg["log_system_metrics"]
        )
    except Exception as e:
        logger.error(f"Failed to start MLflow run: {e}")
        mlflow_info = get_mlflow_info()
        logger.error(mlflow_info)
        raise e

    logger.info(f"MLflow | Starting MLflow Run with name {run_name}")
    log_hydra_cfg_to_mlflow(cfg)

    if "OutlierDetection" in run_name:
        if mlflow_cfg["test_artifact_store"]:
            try:
                test_artifact_write()
                logger.debug("MLflow artifact store test passed")
            except Exception as e:
                logger.error(f"Failed to write MLflow artifact: {e}")
                raise e


def log_hydra_cfg_to_mlflow(cfg):
    # Log the Hydra config to MLflow
    logger.info("Logging Hydra config to MLflow")
    # TODO! save as YAML and log as an artifact?
    hydra_dir = get_hydra_output_dir()
    path_out = save_hydra_cfg_as_yaml(cfg, dir_output=hydra_dir)
    mlflow.log_artifact(path_out, artifact_path="config")


def get_mlflow_info():
    # ToOptimize, now we are running multiple times the same training module with different hyperparameters
    # and only do the "forward pass" evaluation to get the imputation results, and keep the performance metric
    # evaluation on a separate "Prefect task" allowing greater flexibility in the future so that you can implement
    # new metrics if desired without having to retrain the model
    # This means we need to know the experiment_name and run_name of the initial runs (assuming we want to still
    # log to MLflow)

    client = MlflowClient()
    mlflow_dict = {
        "run_tags": mlflow.active_run().data.tags,
        "run_info": dict(mlflow.active_run().info),
        "experiment": dict(
            client.get_experiment(mlflow.active_run().info.experiment_id)
        ),
    }

    return mlflow_dict


def log_metrics_as_mlflow_artifact(
    metrics_subjectwise, model_name, model_artifacts, cfg
):
    # Where are things saved locally, could be an ephemeral location, and the script logs
    # artifacts from here to MLflow that should be then in a non-ephemeral location
    output_dir, fname, artifacts_path = define_pypots_outputs(
        model_name=model_name, artifact_type="metrics"
    )

    # Save as a pickle
    save_results_dict(metrics_subjectwise, artifacts_path)

    # Save the subject-wise metrics as a pickled artifact
    mlflow_info = get_mlflow_info_from_model_dict(model_artifacts)
    experiment_id, run_id = get_mlflow_params(mlflow_info)
    with mlflow.start_run(run_id):
        logger.info("Logging metrics as a pickled artifact to MLflow")
        mlflow.log_artifact(artifacts_path, artifact_path="metrics")


def mlflow_imputation_metrics_logger(metrics_global, split):
    for metric_key in metrics_global:
        metric_out = get_mlflow_metric_name(split, metric_key)
        metric_value = metrics_global[metric_key]
        logger.debug(f"Logging metric {metric_out} to MLflow, value {metric_value}")
        if isinstance(metric_value, np.ndarray):
            mlflow.log_metric(metric_out + "_lo", metric_value[0])
            mlflow.log_metric(metric_out + "_hi", metric_value[1])
        else:
            mlflow.log_metric(metric_out, metric_value)


def log_mlflow_imputation_metrics(
    metrics_global: dict,
    model_name: str,
    split: str,
    model_artifacts: dict,
    cfg: DictConfig,
):
    mlflow_info = get_mlflow_info_from_model_dict(model_artifacts)
    experiment_id, run_id = get_mlflow_params(mlflow_info)

    # Log the metrics MLflow
    if mlflow.active_run() is not None:
        mlflow.end_run()

    with mlflow.start_run(run_id):
        mlflow_imputation_metrics_logger(metrics_global, split)

        # Intermediate Hydra log with the suffix
        hydra_log = get_intermediate_hydra_log_path()
        log_the_hydra_log_as_mlflow_artifact(
            hydra_log, suffix="_metrics", intermediate=True
        )


def log_system_params_to_mlflow(prefix: str = "sys/"):
    dict = get_system_param_dict()
    logger.info("Logging system parameters to MLflow")
    for key1, value1 in dict.items():
        for key2, value2 in dict[key1].items():
            logger.debug(f"Param type = {key1}, logging {prefix+key2} to MLflow")
            mlflow.log_param(prefix + key2, value2)


def log_mlflow_params(mlflow_params, model_name: str = None, run_name: str = None):
    logger.info("Logging MLflow parameters")
    try:
        mlflow.log_param("model", model_name)
    except Exception as e:
        logger.error(f"Failed to log model name to MLflow: {e}")

    for key, value in mlflow_params.items():
        mlflow.log_param(key, value)
    log_system_params_to_mlflow()


def save_pypots_model_to_mlflow(entry, model, cfg, as_artifact: bool = False):
    # Log the model to the models directory
    if as_artifact:
        mlflow.log_artifact(entry.path, artifact_path="models")
    else:
        mlflow_log_pytorch_model(model, path=entry.path, cfg=cfg)


def mlflow_log_pytorch_model(model, path, cfg):
    # https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html#mlflow.pytorch.log_model
    # TODO! impelment with signature and all when you are about the use this for inference,
    #  at this point for the paper, we only really need the results from the imputation, not the
    #  object model itself yet
    # TODO! PyPOTS model logging,
    #  TypeError: Argument 'pytorch_model' should be a torch.nn.Module
    mlflow.pytorch.log_model(
        model, path, conda_env=None, code_paths=None, registered_model_name=None
    )


def pytpots_artifact_wrapper(
    pypots_dir, model, cfg, model_ext=".pypots", as_artifact: bool = True
):
    logger.debug("Logging PyPOTS artifacts")
    obj = os.scandir(pypots_dir)
    try:
        for entry in obj:
            if entry.is_dir():
                logger.debug("dir ", entry.name)
                mlflow.log_artifacts(
                    entry.path, artifact_path="pypots/{}".format(entry.name)
                )
            elif entry.is_file():
                logger.debug("file ", entry.name)
                fname, ext = os.path.splitext(entry.name)
                if ext == model_ext:
                    save_pypots_model_to_mlflow(
                        entry=entry, as_artifact=as_artifact, model=model, cfg=cfg
                    )
                else:
                    mlflow.log_artifact(entry.path, artifact_path="pypots")
            else:
                logger.debug(
                    "Unknown entry type (not logging as PyPots artifact: ", entry.name
                )

    except Exception as e:
        logger.error(f"Failed to log results artifact: {e}")
        raise e


def log_mlflow_artifacts_after_pypots_model_train(results_path, pypots_dir, model, cfg):
    # The results .pickle
    try:
        mlflow.log_artifact(results_path, artifact_path="results")
    except Exception as e:
        logger.error(f"Failed to log results artifact: {e}")
        # https://www.restack.io/docs/mlflow-knowledge-mlflow-log-artifact-permission-denied
        # TODO! Inspect more why this happens? makedir fails
        #  PermissionError: [Errno 13] Permission denied: '/petteri'
        # https://github.com/mlflow/mlflow/issues/212#issuecomment-409260757
        # The artifact store (used for log_model or log_artifact) is used to persist the larger data such as models,
        # which is why we rely on an external persistent store. This is why the log_metric and log_param calls work
        # -- they only need to talk to the server -- while the log_model call is failing.

    # The pypots artifacts
    pytpots_artifact_wrapper(pypots_dir, model, cfg)


def log_imputation_db_to_mlflow(
    db_path: str, mlflow_cfg: dict, model: str, cfg: DictConfig
):
    with mlflow.start_run(run_id=mlflow_cfg["run_info"]["run_id"]):
        logger.info("Logging imputation database to MLflow as DuckDB")
        mlflow.log_artifact(db_path, artifact_path="imputation_db")


def post_imputation_model_training_mlflow_log(
    metrics_model: dict, model_artifacts: dict, cfg: DictConfig
):
    best_previous_run = get_best_previous_mlflow_logged_model(
        model_dict=model_artifacts, cfg=cfg
    )
    model_improved = is_current_better_than_previous(
        metrics_model=metrics_model,
        model_dict=model_artifacts,
        best_previous_run=best_previous_run,
        cfg=cfg,
    )

    if model_improved:
        # TODO! Implement actually the registering, and the model logging during previous MLflow logging
        logger.warning("Model improved, now possible to register MLflow Model Registry")
    else:
        logger.info(
            "Model did not improve, not registering to MLflow Model Registry "
            "as the best model (Staging) TO-BE-IMPLEMENTED!"
        )


def check_if_improved_with_direction(
    metric_string, metric_direction, current_metric_value, best_metric_value
):
    is_improved = False
    if metric_direction == "ASC":
        if current_metric_value < best_metric_value:
            logger.info(
                f"Current metric ({metric_string} = {current_metric_value:.5f}) is better than the previous best"
            )
            is_improved = True
        else:
            logger.info(
                f"Current metric ({metric_string} = {current_metric_value:.5f}) is worse (or equal) than the "
                f"previous best ({best_metric_value:.5f})"
            )
    elif metric_direction == "DESC":
        if current_metric_value > best_metric_value:
            logger.info(
                f"Current metric ({metric_string} = {current_metric_value:.5f}) is better than the previous best"
            )
            is_improved = True
        else:
            logger.info(
                f"Current metric ({metric_string} = {current_metric_value:.5f}) is worse (or equal) than the "
                f"previous best ({best_metric_value:.5f})"
            )
    else:
        logger.error(f"Unknown metric direction = {metric_direction}")
        raise ValueError(f"Unknown metric direction = {metric_direction}")

    return is_improved


def is_current_better_than_previous(
    metrics_model: dict, model_dict: dict, best_previous_run: dict, cfg: DictConfig
):
    mlflow_info = get_mlflow_info_from_model_dict(model_dict)
    current_experiment, metric_string, split_key, metric_direction = (
        what_to_search_from_mlflow(
            run_name=mlflow_info["run_info"]["run_name"], cfg=cfg
        )
    )

    best_metric_value = best_previous_run[f"metrics.{split_key}/{metric_string}"]
    logger.info(
        f"Best metric ({metric_string} = {best_metric_value}) from the logged MLflow runs"
    )
    current_metric_value = get_best_metric_from_current_run(
        metrics_model=metrics_model, split_key=split_key, metric_string=metric_string
    )

    return check_if_improved_with_direction(
        metric_string, metric_direction, current_metric_value, best_metric_value
    )
