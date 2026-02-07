import os
from typing import Any, Dict, Optional

import mlflow
import numpy as np
from loguru import logger
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig

from src.imputation.pypots.pypots_utils import define_pypots_outputs
from src.log_helpers.hydra_utils import (
    get_hydra_output_dir,
    get_intermediate_hydra_log_path,
    log_the_hydra_log_as_mlflow_artifact,
    save_hydra_cfg_as_yaml,
)
from src.log_helpers.local_artifacts import save_results_dict
from src.log_helpers.log_naming_uris_and_dirs import get_mlflow_metric_name
from src.log_helpers.mlflow_artifacts import (
    get_best_metric_from_current_run,
    get_best_previous_mlflow_logged_model,
    get_mlflow_info_from_model_dict,
    get_mlflow_params,
    what_to_search_from_mlflow,
)
from src.log_helpers.system_utils import get_system_param_dict
from src.utils import get_artifacts_dir
from tests.mlflow_tests import test_artifact_write


def init_mlflow(cfg: DictConfig) -> None:
    """
    Initialize MLflow tracking URI from configuration.

    Parameters
    ----------
    cfg : DictConfig
        Configuration containing SERVICES.mlflow_tracking_uri.

    Notes
    -----
    If no URI is specified, MLflow uses a local 'mlruns' directory.
    """
    # Set the MLflow tracking URI (export MLFLOW_TRACKING_URI='file:////home/petteri/Dropbox/mlruns')
    if cfg["SERVICES"]["mlflow_tracking_uri"] is not None:
        mlflow.set_tracking_uri(cfg["SERVICES"]["mlflow_tracking_uri"])
    else:
        logger.warning(
            "You did not specify any MLflow tracking URI. Using the 'mlruns' dir inside 'src'"
        )
    logger.info(f"{mlflow.get_tracking_uri()}")


def init_mlflow_experiment(
    mlflow_cfg: Optional[DictConfig] = None,
    experiment_name: str = "PLR_imputation",
    override_default_location: bool = False,
    _permanent_delete: bool = True,
) -> None:
    """
    Initialize or get an MLflow experiment.

    Parameters
    ----------
    mlflow_cfg : DictConfig, optional
        MLflow configuration (currently unused).
    experiment_name : str, default "PLR_imputation"
        Name of the experiment to create/get.
    override_default_location : bool, default False
        If True, use custom artifact location.
    permanent_delete : bool, default True
        Permanent deletion flag (currently unused).

    Raises
    ------
    Exception
        If experiment creation fails (e.g., permission issues).
    """
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


def set_artifact_store_location() -> None:
    """Set MLflow artifact store location.

    Currently a placeholder for future remote storage (e.g., S3) configuration.

    Returns
    -------
    None
        No artifact store location is set currently.
    """
    # https://mlflow.org/docs/latest/tracking/artifacts-stores.html
    # TODO! Some remote, e.g. S3
    return None


def init_mlflow_run(
    mlflow_cfg: DictConfig, run_name: str, cfg: DictConfig, experiment_name: str
) -> None:
    """
    Start a new MLflow run.

    Parameters
    ----------
    mlflow_cfg : DictConfig
        MLflow configuration with 'log_system_metrics' flag.
    run_name : str
        Name for the MLflow run.
    cfg : DictConfig
        Full Hydra configuration to log.
    experiment_name : str
        Name of the MLflow experiment.

    Raises
    ------
    Exception
        If run creation fails.
    """
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


def log_hydra_cfg_to_mlflow(cfg: DictConfig) -> None:
    """Log Hydra configuration to MLflow as a YAML artifact.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration to log.
    """
    # Log the Hydra config to MLflow
    logger.info("Logging Hydra config to MLflow")
    # TODO! save as YAML and log as an artifact?
    hydra_dir = get_hydra_output_dir()
    path_out = save_hydra_cfg_as_yaml(cfg, dir_output=hydra_dir)
    mlflow.log_artifact(path_out, artifact_path="config")


def get_mlflow_info() -> Dict[str, Any]:
    """Get current MLflow run information as a dictionary.

    Collects tags, run info, and experiment info from the active MLflow run.
    Useful for storing MLflow metadata alongside model artifacts for later
    reference when logging metrics or additional artifacts.

    Returns
    -------
    dict
        Dictionary with 'run_tags', 'run_info', and 'experiment' keys.
    """
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
    metrics_subjectwise: Dict[str, Any],
    model_name: str,
    model_artifacts: Dict[str, Any],
    cfg: DictConfig,
) -> None:
    """Log subject-wise metrics as a pickled MLflow artifact.

    Parameters
    ----------
    metrics_subjectwise : dict
        Dictionary containing per-subject metrics.
    model_name : str
        Name of the model for filename generation.
    model_artifacts : dict
        Model artifacts containing MLflow info.
    cfg : DictConfig
        Configuration object (currently unused).
    """
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


def mlflow_imputation_metrics_logger(
    metrics_global: Dict[str, Any], split: str
) -> None:
    """Log global imputation metrics to MLflow.

    Handles both scalar metrics and array metrics (e.g., confidence intervals).

    Parameters
    ----------
    metrics_global : dict
        Dictionary of metric names to values.
    split : str
        Data split name for metric naming.
    """
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
    metrics_global: Dict[str, Any],
    model_name: str,
    split: str,
    model_artifacts: Dict[str, Any],
    cfg: DictConfig,
) -> None:
    """Log imputation metrics and Hydra log to MLflow for an existing run.

    Parameters
    ----------
    metrics_global : dict
        Global metrics dictionary.
    model_name : str
        Name of the imputation model (currently unused).
    split : str
        Data split name.
    model_artifacts : dict
        Model artifacts with MLflow info.
    cfg : DictConfig
        Configuration object (currently unused).
    """
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


def log_system_params_to_mlflow(prefix: str = "sys/") -> None:
    """Log system parameters (hardware, library versions) to MLflow.

    Parameters
    ----------
    prefix : str, default "sys/"
        Prefix for parameter names in MLflow.
    """
    dict = get_system_param_dict()
    logger.info("Logging system parameters to MLflow")
    for key1, value1 in dict.items():
        for key2, value2 in dict[key1].items():
            logger.debug(f"Param type = {key1}, logging {prefix + key2} to MLflow")
            mlflow.log_param(prefix + key2, value2)


def log_mlflow_params(
    mlflow_params: Dict[str, Any],
    model_name: Optional[str] = None,
    run_name: Optional[str] = None,
) -> None:
    """Log model parameters and system info to MLflow.

    Parameters
    ----------
    mlflow_params : dict
        Dictionary of parameters to log.
    model_name : str, optional
        Model name to log as 'model' parameter.
    run_name : str, optional
        Run name (currently unused).
    """
    logger.info("Logging MLflow parameters")
    try:
        mlflow.log_param("model", model_name)
    except Exception as e:
        logger.error(f"Failed to log model name to MLflow: {e}")

    for key, value in mlflow_params.items():
        mlflow.log_param(key, value)
    log_system_params_to_mlflow()


def save_pypots_model_to_mlflow(
    entry: os.DirEntry, model: Any, cfg: DictConfig, as_artifact: bool = False
) -> None:
    """Save PyPOTS model to MLflow as artifact or registered model.

    Parameters
    ----------
    entry : os.DirEntry
        Directory entry for the model file.
    model : object
        PyPOTS model object.
    cfg : DictConfig
        Configuration object.
    as_artifact : bool, default False
        If True, log as simple artifact; if False, use MLflow model logging.
    """
    # Log the model to the models directory
    if as_artifact:
        mlflow.log_artifact(entry.path, artifact_path="models")
    else:
        mlflow_log_pytorch_model(model, path=entry.path, cfg=cfg)


def mlflow_log_pytorch_model(model: Any, path: str, cfg: DictConfig) -> None:
    """Log PyTorch model to MLflow.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to log.
    path : str
        Artifact path for the model.
    cfg : DictConfig
        Configuration object (currently unused).

    Notes
    -----
    This is a basic implementation without model signature. PyPOTS models
    may require special handling as they are not standard torch.nn.Module.
    """
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
    pypots_dir: str,
    model: Any,
    cfg: DictConfig,
    model_ext: str = ".pypots",
    as_artifact: bool = True,
) -> None:
    """Log all PyPOTS artifacts from a directory to MLflow.

    Iterates through the PyPOTS output directory and logs directories,
    model files, and other artifacts appropriately.

    Parameters
    ----------
    pypots_dir : str
        Path to PyPOTS output directory.
    model : object
        PyPOTS model object.
    cfg : DictConfig
        Configuration object.
    model_ext : str, default ".pypots"
        File extension for model files.
    as_artifact : bool, default True
        If True, log model as artifact; if False, use MLflow model logging.

    Raises
    ------
    Exception
        If artifact logging fails.
    """
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


def log_mlflow_artifacts_after_pypots_model_train(
    results_path: str, pypots_dir: str, model: Any, cfg: DictConfig
) -> None:
    """Log results and PyPOTS artifacts to MLflow after training.

    Parameters
    ----------
    results_path : str
        Path to results pickle file.
    pypots_dir : str
        Path to PyPOTS output directory.
    model : object
        PyPOTS model object.
    cfg : DictConfig
        Configuration object.
    """
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
    db_path: str, mlflow_cfg: Dict[str, Any], model: str, cfg: DictConfig
) -> None:
    """Log imputation DuckDB database to MLflow.

    Parameters
    ----------
    db_path : str
        Path to DuckDB file.
    mlflow_cfg : dict
        MLflow configuration with run_info.
    model : str
        Model name (currently unused).
    cfg : DictConfig
        Configuration object (currently unused).
    """
    with mlflow.start_run(run_id=mlflow_cfg["run_info"]["run_id"]):
        logger.info("Logging imputation database to MLflow as DuckDB")
        mlflow.log_artifact(db_path, artifact_path="imputation_db")


def post_imputation_model_training_mlflow_log(
    metrics_model: Dict[str, Any], model_artifacts: Dict[str, Any], cfg: DictConfig
) -> None:
    """Check if current model improved over previous best and log accordingly.

    Compares current model metrics against previously logged best model
    and logs to MLflow Model Registry if improved.

    Parameters
    ----------
    metrics_model : dict
        Current model metrics.
    model_artifacts : dict
        Model artifacts with MLflow info.
    cfg : DictConfig
        Configuration object.
    """
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
    metric_string: str,
    metric_direction: str,
    current_metric_value: float,
    best_metric_value: float,
) -> bool:
    """Check if current metric is better than previous best based on direction.

    Parameters
    ----------
    metric_string : str
        Name of the metric for logging.
    metric_direction : str
        'ASC' if lower is better, 'DESC' if higher is better.
    current_metric_value : float
        Current model's metric value.
    best_metric_value : float
        Previous best metric value.

    Returns
    -------
    bool
        True if current is better than previous best.

    Raises
    ------
    ValueError
        If metric_direction is not 'ASC' or 'DESC'.
    """
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
    metrics_model: Dict[str, Any],
    model_dict: Dict[str, Any],
    best_previous_run: Dict[str, Any],
    cfg: DictConfig,
) -> bool:
    """Determine if current model outperforms the previous best.

    Parameters
    ----------
    metrics_model : dict
        Current model metrics.
    model_dict : dict
        Model artifacts with MLflow info.
    best_previous_run : dict
        Previous best run data.
    cfg : DictConfig
        Configuration object.

    Returns
    -------
    bool
        True if current model is better.
    """
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
