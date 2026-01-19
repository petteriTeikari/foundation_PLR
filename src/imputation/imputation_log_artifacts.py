import os
import mlflow
from loguru import logger

from src.log_helpers.hydra_utils import log_hydra_artifacts_to_mlflow
from src.log_helpers.local_artifacts import save_object_to_pickle, save_results_dict
from src.log_helpers.log_naming_uris_and_dirs import get_imputation_pickle_name
from src.log_helpers.mlflow_artifacts import (
    get_mlflow_info_from_model_dict,
    get_mlflow_params,
)


def pypots_model_logger(
    model_obj, model_name: str, model_info: dict, artifacts_dir: str
):
    model_path = os.path.join(model_obj.saving_path, f"{model_name}.pypots")
    if not os.path.exists(model_path):
        logger.error(f"Could not find the PyPOTS model from {model_path}")
        raise FileNotFoundError(f"Could not find the PyPOTS model from {model_path}")

    logger.debug(
        "Copying saved PyPOTS model to MLflow (from {})".format(model_obj.saving_path)
    )
    mlflow.log_artifact(model_path, artifact_path="model")
    logger.debug("Copying PyPOTS directory to MLflow (contains e.g. tensorboard logs)")
    mlflow.log_artifact(model_obj.saving_path, artifact_path="pyPOTS")
    # Log the number parameters of the model
    try:
        mlflow.log_param("num_params", model_info["num_params"])
    except Exception as e:
        logger.warning(f"Could not log the number of parameters to MLflow: {e}")


def generic_pickled_model_logger(model_obj, model_name: str, artifacts_dir: str):
    logger.debug("Logging the MissForest model to local disk")
    fname = get_imputation_pickle_name(model_name)
    path = os.path.join(artifacts_dir, fname)
    save_object_to_pickle(model_obj, path)
    logger.debug("Copying saved MissForest model to MLflow")
    mlflow.log_artifact(path, artifact_path="model")


def log_imputer_model(model_obj, model_name: str, artifacts: dict, artifacts_dir: str):
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
    imputation_artifacts, model_name, artifacts_dir, cfg, run_name
):
    # Log first to disk
    results_path = os.path.join(artifacts_dir, get_imputation_pickle_name(model_name))
    save_results_dict(
        results_dict=imputation_artifacts,
        results_path=results_path,
        name="imputation",
    )

    # And then copy this to MLflow
    logger.debug("Copying the imputation results to MLflow")
    mlflow.log_artifact(results_path, artifact_path="imputation")


def save_and_log_imputer_artifacts(
    model, imputation_artifacts, artifacts_dir, cfg, model_name, run_name
):
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
