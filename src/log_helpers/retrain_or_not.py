import mlflow
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from src.log_helpers.mlflow_artifacts import (
    return_best_mlflow_run,
    what_to_search_from_mlflow,
)


def check_if_imputation_model_trained_already_from_mlflow(
    cfg: DictConfig,
    run_name: str,
    model_type: str,
) -> dict | None:
    """Check if an imputation model with matching configuration exists in MLflow.

    Parameters
    ----------
    cfg : DictConfig
        Configuration for determining search parameters.
    run_name : str
        Name of the run to search for.
    model_type : str
        Type of model to search for.

    Returns
    -------
    dict or None
        Best matching run data if found, None otherwise.
    """
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
    run_name: str | None = None,
    model_type: str = "imputation",
) -> tuple[bool, dict]:
    """Determine whether to retrain an imputation model.

    Checks configuration flag and MLflow history to decide if retraining
    is needed.

    Parameters
    ----------
    cfg : DictConfig
        Configuration with IMPUTATION_TRAINING.retrain_models flag.
    run_name : str, optional
        Name of the run to check.
    model_type : str, default "imputation"
        Type of model.

    Returns
    -------
    tuple
        Tuple of (should_retrain: bool, best_run: dict).
    """
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
) -> bool:
    """Check if features have already been extracted for an imputation source.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object (currently unused).
    experiment_name : str
        MLflow experiment name.
    run_name : str
        Run name to search for.

    Returns
    -------
    bool
        True if featurization run exists, False otherwise.
    """
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
) -> bool:
    """Determine whether to re-extract features from imputation results.

    Parameters
    ----------
    run_name : str
        Run name to check.
    experiment_name : str
        MLflow experiment name.
    cfg : DictConfig
        Configuration with PLR_FEATURIZATION.re_featurize flag.

    Returns
    -------
    bool
        True if re-featurization is needed.
    """
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


def if_recompute_and_viz_imputation_metrics(_recompute: bool = True) -> bool:
    """Determine whether to recompute and visualize imputation metrics.

    Parameters
    ----------
    _recompute : bool, default True
        Input flag (currently unused — placeholder implementation).

    Returns
    -------
    bool
        Always returns True in current implementation.

    Notes
    -----
    This is a placeholder function. Future implementation should check
    for previously computed metrics to avoid redundant computation.
    """
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


def if_recreate_ensemble(
    ensemble_name: str, experiment_name: str, cfg: DictConfig
) -> bool:
    """Determine whether to recreate an ensemble model.

    Parameters
    ----------
    ensemble_name : str
        Name of the ensemble.
    experiment_name : str
        MLflow experiment name.
    cfg : DictConfig
        Configuration object (currently unused).

    Returns
    -------
    bool
        True if no previous runs found, False otherwise.
    """
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
