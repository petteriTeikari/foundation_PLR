from pathlib import Path
from typing import Any

import mlflow
from loguru import logger
from omegaconf import DictConfig

from src.log_helpers.hydra_utils import log_hydra_artifacts_to_mlflow
from src.log_helpers.local_artifacts import save_object_to_pickle, save_results_dict
from src.log_helpers.log_naming_uris_and_dirs import get_imputation_pickle_name
from src.log_helpers.mlflow_artifacts import (
    get_mlflow_info_from_model_dict,
    get_mlflow_params,
)


def pypots_model_logger(
    model_obj: Any, model_name: str, model_info: dict[str, Any], artifacts_dir: str
) -> None:
    """Log a PyPOTS model and its training artifacts to MLflow.

    Copies the saved PyPOTS model file and training directory (including
    TensorBoard logs) to MLflow artifacts.

    Parameters
    ----------
    model_obj : PyPOTS model
        Trained PyPOTS model instance with saving_path attribute.
    model_name : str
        Name of the model for file naming.
    model_info : dict
        Model information dictionary containing 'num_params'.
    artifacts_dir : str
        Directory for artifacts (unused but kept for interface consistency).

    Raises
    ------
    FileNotFoundError
        If the model file is not found at the expected path.
    """
    model_path = Path(model_obj.saving_path) / f"{model_name}.pypots"
    if not model_path.exists():
        logger.error(f"Could not find the PyPOTS model from {model_path}")
        raise FileNotFoundError(f"Could not find the PyPOTS model from {model_path}")

    logger.debug(
        "Copying saved PyPOTS model to MLflow (from {})".format(model_obj.saving_path)
    )
    mlflow.log_artifact(str(model_path), artifact_path="model")
    logger.debug("Copying PyPOTS directory to MLflow (contains e.g. tensorboard logs)")
    mlflow.log_artifact(model_obj.saving_path, artifact_path="pyPOTS")
    # Log the number parameters of the model
    try:
        mlflow.log_param("num_params", model_info["num_params"])
    except Exception as e:
        logger.warning(f"Could not log the number of parameters to MLflow: {e}")


def generic_pickled_model_logger(
    model_obj: Any, model_name: str, artifacts_dir: str
) -> None:
    """Save a model as pickle and log to MLflow.

    Generic model logger for models that don't have specialized saving methods.

    Parameters
    ----------
    model_obj : object
        Trained model instance to pickle.
    model_name : str
        Name of the model for file naming.
    artifacts_dir : str
        Directory to save the pickle file.
    """
    logger.debug("Logging the MissForest model to local disk")
    fname = get_imputation_pickle_name(model_name)
    path = Path(artifacts_dir) / fname
    save_object_to_pickle(model_obj, str(path))
    logger.debug("Copying saved MissForest model to MLflow")
    mlflow.log_artifact(str(path), artifact_path="model")


def log_imputer_model(
    model_obj: Any, model_name: str, artifacts: dict[str, Any], artifacts_dir: str
) -> None:
    """Log an imputation model to MLflow using the appropriate method.

    Dispatches to the correct logging method based on model type
    (PyPOTS, MissForest, MOMENT, etc.).

    Parameters
    ----------
    model_obj : object
        Trained imputation model instance.
    model_name : str
        Name of the model.
    artifacts : dict
        Artifacts dictionary containing 'model_artifacts' with 'model_info'.
    artifacts_dir : str
        Directory for saving artifacts.

    Notes
    -----
    PyPOTS models use their specialized save format. MissForest uses pickle.
    MOMENT models are not currently logged (only results are logged).
    """
    # Log the model
    logger.debug("Logging the model to MLflow")

    if "model_info" in artifacts["model_artifacts"]:
        if "PyPOTS" in artifacts["model_artifacts"]["model_info"]:
            if artifacts["model_artifacts"]["model_info"]["PyPOTS"]:
                # This is now a PyPOTS model
                pypots_model_logger(
                    model_obj=model_obj,
                    model_name=model_name,
                    model_info=artifacts["model_artifacts"]["model_info"],
                    artifacts_dir=artifacts_dir,
                )
            else:
                logger.warning(
                    "Figure out how to log the new model to MLflow, where is the model located?"
                )
                logger.warning("Or is the model_obj still not saved to disk at all?")
                # raise NotImplementedError(
                #     "No non-PyPOTS evaluation/imputation implemented yet!"
                # )
        else:
            logger.warning(
                "Figure out how to log the new model to MLflow, where is the model located?"
            )
            logger.warning("Or is the model_obj still not saved to disk at all?")
            # raise NotImplementedError(
            #     "No non-PyPOTS evaluation/imputation implemented yet!"
            # )
    elif "MISSFOREST" in model_name:
        generic_pickled_model_logger(model_obj, model_name, artifacts_dir)

    elif "MOMENT" in model_name:
        # generic_pickled_model_logger(model_obj, model_name, artifacts_dir)
        logger.warning("Moment model is not logged now, only the results are logged")

    else:
        logger.warning(
            "Figure out how to log the new model to MLflow, where is the model located?"
        )
        logger.warning("Or is the model_obj still not saved to disk at all?")
        # raise NotImplementedError(
        #     "No non-PyPOTS evaluation/imputation implemented yet!"
        # )


def log_the_imputation_results(
    imputation_artifacts: dict[str, Any],
    model_name: str,
    artifacts_dir: str,
    cfg: DictConfig,
    run_name: str,
) -> None:
    """Save imputation results locally and log to MLflow.

    Parameters
    ----------
    imputation_artifacts : dict
        Dictionary containing imputation results to save.
    model_name : str
        Name of the model for file naming.
    artifacts_dir : str
        Directory to save the pickle file.
    cfg : DictConfig
        Configuration (unused but kept for interface consistency).
    run_name : str
        Run name (unused but kept for interface consistency).
    """
    # Log first to disk
    results_path = Path(artifacts_dir) / get_imputation_pickle_name(model_name)
    save_results_dict(
        results_dict=imputation_artifacts,
        results_path=str(results_path),
        name="imputation",
    )

    # And then copy this to MLflow
    logger.debug("Copying the imputation results to MLflow")
    mlflow.log_artifact(str(results_path), artifact_path="imputation")


def save_and_log_imputer_artifacts(
    model: Any,
    imputation_artifacts: dict[str, Any],
    artifacts_dir: str,
    cfg: DictConfig,
    model_name: str,
    run_name: str,
) -> None:
    """Save and log all imputation artifacts to MLflow.

    Orchestrates the logging of model, results, and Hydra configuration
    artifacts to the associated MLflow run.

    Parameters
    ----------
    model : object
        Trained imputation model instance.
    imputation_artifacts : dict
        Dictionary containing 'model_artifacts' with MLflow info and results.
    artifacts_dir : str
        Directory for saving artifacts locally.
    cfg : DictConfig
        Full Hydra configuration.
    model_name : str
        Name of the imputation model.
    run_name : str
        MLflow run name.

    Notes
    -----
    Ends any active MLflow run, then starts a new run context to log
    artifacts. The run is ended after all artifacts are logged.
    """
    logger.info("Logging the imputer artifacts to MLflow")

    # Log the metrics MLflow
    if mlflow.active_run() is not None:
        mlflow.end_run()

    mlflow_info = get_mlflow_info_from_model_dict(
        imputation_artifacts["model_artifacts"]
    )
    experiment_id, run_id = get_mlflow_params(mlflow_info)

    with mlflow.start_run(run_id):
        # Log the model
        log_imputer_model(
            model_obj=model,
            model_name=model_name,
            artifacts=imputation_artifacts,
            artifacts_dir=artifacts_dir,
        )

        # Log the imputation results (forward passes, and other data)
        log_the_imputation_results(
            imputation_artifacts, model_name, artifacts_dir, cfg, run_name
        )

        # Log the Hydra artifacts to MLflow
        log_hydra_artifacts_to_mlflow(artifacts_dir, model_name, cfg, run_name)

        # End the MLflow run, and you can still log to the same run later when evaluating inputation,
        # computing metrics, logging the artifacts with the run_id
        logger.debug(
            "MLflow | Ending MLflow run named: {}".format(
                mlflow.active_run().info.run_name
            )
        )
        mlflow.end_run()
