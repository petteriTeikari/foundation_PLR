import pandas as pd
from omegaconf import DictConfig
from loguru import logger
import mlflow

from src.log_helpers.mlflow_artifacts import (
    what_to_search_from_mlflow,
    return_best_mlflow_run,
)


def check_if_imputation_model_trained_already_from_mlflow(
    cfg: DictConfig,
    run_name: str,
    model_type: str,
):
    current_experiment, metric_string, split_key, metric_direction = (
        what_to_search_from_mlflow(run_name=run_name, cfg=cfg, model_type=model_type)
    )

    if current_experiment is not None:
        logger.info(
            "MLflow | Searching for the best model (metric = {}, split_key = {}, "
            "direction = {})".format(metric_string, split_key, metric_direction)
        )

        best_run = return_best_mlflow_run(
            current_experiment,
            metric_string,
            split_key,
            metric_direction,
            run_name=run_name,
        )

    else:
        logger.debug(
            "No previous (best) runs found from MLflow, need to re-train the model"
        )
        best_run = None

    return best_run


def if_retrain_the_imputation_model(
    cfg: DictConfig,
    run_name: str = None,
    model_type: str = "imputation",
):
    if cfg["IMPUTATION_TRAINING"]["retrain_models"]:
        # No matter what, always retrain the model
        logger.debug("You had retraining model set to True, so retraining the model")
        return True, {}
    else:
        # check all the previous runs from MLflow, and see if you have already trained the model
        best_run = check_if_imputation_model_trained_already_from_mlflow(
            cfg=cfg,
            run_name=run_name,
            model_type=model_type,
        )
        if best_run is not None:
            logger.debug("Found previous runs from MLflow, so skipping the retraining")
            return False, best_run
        else:
            logger.debug("No previous runs found from MLflow, so training the model")
            return True, {}


def check_if_imputation_source_featurized_already_from_mlflow(
    cfg: DictConfig,
    experiment_name: str,
    run_name: str,
):
    current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
    df: pd.DataFrame = mlflow.search_runs([current_experiment["experiment_id"]])

    if df.shape[0] == 0:
        logger.debug("No previous runs found from MLflow, need to re-featurize")
        return False
    else:
        if run_name in df["tags.mlflow.runName"].values:
            logger.debug(
                f"Found previous runs (n={df.shape[0]}) from MLflow, "
                f"so skipping the refeaturization for '{run_name}'"
            )
            return True


def if_refeaturize_from_imputation(
    run_name: str, experiment_name: str, cfg: DictConfig
):
    if cfg["PLR_FEATURIZATION"]["re_featurize"]:
        # No matter what, always retrain the model
        logger.debug("You had re_featurize set to True, so re_featurizing the data")
        return True

    else:
        # check all the previous runs from MLflow, and see if you have already trained the model
        already_featurized = check_if_imputation_source_featurized_already_from_mlflow(
            cfg=cfg,
            experiment_name=experiment_name,
            run_name=run_name,
        )
        if already_featurized:
            logger.info("MLflow found -> Skipping the refeaturization for the sources")
            return False
        else:
            logger.info("MLflow not found -> Refeaturizing all the sources")
            return True


def if_recompute_and_viz_imputation_metrics(recompute: bool = True):
    true_out = True
    # TODO! implement this at some point, if you have this False, and you don't check
    #  for previously computed metrics, your downstream code will crash while you still have the imputation done,
    #  but not the metrics
    logger.warning(
        "Placeholder for metric recomputation decision, returning now = {}".format(
            true_out
        )
    )
    return true_out


def if_recreate_ensemble(ensemble_name, experiment_name, cfg):
    current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
    df: pd.DataFrame = mlflow.search_runs([current_experiment["experiment_id"]])

    if df.shape[0] == 0:
        logger.warning("No previous runs found from MLflow, need to re-ensemble")
        return True
    else:
        logger.warning(
            f"Found previous runs (n={df.shape[0]}) from MLflow, "
            f"so skipping the re-ensembling for '{ensemble_name}'"
        )
        return False
