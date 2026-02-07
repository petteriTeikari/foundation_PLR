from typing import Any, Dict, Optional, Tuple

import mlflow
from loguru import logger
from omegaconf import DictConfig

from src.data_io.define_sources_for_flow import get_best_dict
from src.ensemble.ensemble_utils import get_best_imputation_col_name
from src.imputation.imputation_log_artifacts import save_and_log_imputer_artifacts
from src.imputation.missforest_main import missforest_main
from src.imputation.momentfm.moment_imputation_main import moment_imputation_main

# NOTE: nuwats archived - see archived/nuwats/ (not used in final paper)
# from src.imputation.nuwats.nuwats_main import nuwats_imputation_main
from src.imputation.pypots.pypots_wrapper import pypots_wrapper
from src.log_helpers.log_utils import define_run_name, update_run_name
from src.log_helpers.mlflow_utils import (
    get_mlflow_info,
    init_mlflow_run,
    log_mlflow_params,
)
from src.log_helpers.retrain_or_not import (
    if_retrain_the_imputation_model,
)
from src.metrics.evaluate_imputation_metrics import compute_metrics_by_model
from src.orchestration.debug_utils import (
    debug_train_only_for_one_epoch,
    fix_tree_learners_for_debug,
)
from src.utils import get_artifacts_dir


def setup_PLR_worklow(
    cfg: DictConfig, run_name: str
) -> Tuple[DictConfig, str, str, bool, Optional[Any], str]:
    """
    Set up the PLR imputation workflow.

    Configures the imputation pipeline:
    1. Extract model name from config
    2. Apply debug settings if enabled
    3. Check for existing model runs

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration.
    run_name : str
        MLflow run name.

    Returns
    -------
    tuple
        (cfg, model_name, updated_name, train_ON, best_run, artifacts_dir)
        - cfg: Updated configuration
        - model_name: Imputation model name
        - updated_name: Updated run name
        - train_ON: Whether to retrain
        - best_run: Best existing run (if any)
        - artifacts_dir: Directory for artifacts
    """
    # There should be only one model here atm, TO-OPTIMIZE how to reconcile this later TODO!
    assert len(cfg["MODELS"]) == 1, "Only one model should be trained at a time"

    # Refactor this later, as the model_name used to be looped here
    model_name = list(cfg["MODELS"].keys())[0]  # just pick the first here

    # Debug: set the epochs to 1
    if cfg["EXPERIMENT"]["debug"]:
        cfg = debug_train_only_for_one_epoch(cfg)
        cfg = fix_tree_learners_for_debug(cfg, model_name)

    # Check if you find older models, and if you want to retrain them
    updated_name = update_run_name(
        run_name=run_name, base_run_name=define_run_name(cfg=cfg)
    )
    train_ON, best_run = if_retrain_the_imputation_model(
        cfg=cfg, run_name=updated_name, model_type="imputation"
    )

    # get the artifacts dir
    artifacts_dir = get_artifacts_dir(service_name="imputation")

    return cfg, model_name, updated_name, train_ON, best_run, artifacts_dir


def mlflow_log_of_source_for_imputation(
    source_data: Dict[str, Any], cfg: DictConfig
) -> None:
    """Log source data information to MLflow for imputation tracking.

    Records the outlier detection run information that was used as input
    for the imputation step. This enables traceability of the preprocessing
    pipeline through MLflow.

    Parameters
    ----------
    source_data : dict
        Source data dictionary containing 'mlflow' key with run metadata
        and outlier detection results. If 'mlflow' is None, logs ground
        truth parameters instead.
    cfg : DictConfig
        Full Hydra configuration containing OUTLIER_DETECTION settings.

    Notes
    -----
    Logs either the outlier detection run ID and best metric, or None/0
    for ground truth data where no upstream outlier detection was used.
    """
    if source_data["mlflow"] is not None:
        mlflow.log_param("Outlier_run_id", source_data["mlflow"]["run_id"])
        best_outlier_dict = get_best_dict("outlier_detection", cfg)
        best_outlier_string = best_outlier_dict["string"].replace("metrics.", "")
        col_name = get_best_imputation_col_name(best_metric_cfg=best_outlier_dict)
        try:
            best_value = source_data["mlflow"][col_name]
        except Exception as e:
            logger.error(f"Could not find {best_outlier_string} in {source_data}")
            logger.error(e)
            raise e
        mlflow.log_param(f"OutlierBest_{best_outlier_string}", best_value)
    else:
        # This is now the manually annotated data, so no id, and loss is 0 to itself (as it's the ground truth)
        # MSE (anomaly score) in practice
        mlflow.log_param("Outlier_run_id", None)
        mlflow.log_param("OutlierBest", 0)


def imputation_model_selector(
    source_data: Dict[str, Any],
    cfg: DictConfig,
    model_name: str,
    run_name: str,
    artifacts_dir: str,
    experiment_name: str,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """
    Select and execute imputation method.

    Dispatches to the appropriate imputation implementation. Supports:
    - Deep learning: SAITS, CSDI, TimesNet (via PyPOTS)
    - Foundation models: MOMENT
    - Traditional: MissForest

    Note: NuwaTS was archived (see archived/nuwats/) - not used in final paper.

    Parameters
    ----------
    source_data : dict
        Data from outlier detection stage with signals and masks.
    cfg : DictConfig
        Full Hydra configuration.
    model_name : str
        Imputation method name. One of:
        'SAITS', 'CSDI', 'TimesNet', 'MISSFOREST', 'MOMENT'.
    run_name : str
        MLflow run name.
    artifacts_dir : str
        Directory for saving artifacts.
    experiment_name : str
        MLflow experiment name.

    Returns
    -------
    tuple
        (model, imputation_artifacts) where:
        - model: Trained imputation model
        - imputation_artifacts: dict with imputed data and metrics

    Raises
    ------
    NotImplementedError
        If model_name is not supported.
    """
    # MLflow run (init only when training again, no point when reading precomputed results)
    init_mlflow_run(
        mlflow_cfg=cfg["MLFLOW"],
        run_name=run_name,
        cfg=cfg,
        experiment_name=experiment_name,
    )

    # MLflow parameters
    log_mlflow_params(
        mlflow_params=cfg["MODELS"][model_name]["MODEL"], model_name=model_name
    )
    mlflow_log_of_source_for_imputation(source_data, cfg)

    logger.info("Imputation with model {}".format(model_name))
    if model_name == "SAITS" or model_name == "CSDI" or model_name == "TimesNet":
        model, model_artifacts = pypots_wrapper(
            source_data=source_data,
            model_cfg=cfg["MODELS"][model_name],
            cfg=cfg,
            model_name=model_name,
            run_name=run_name,
        )
    elif model_name == "MISSFOREST":
        model, model_artifacts = missforest_main(
            source_data=source_data,
            model_cfg=cfg["MODELS"][model_name],
            cfg=cfg,
            model_name=model_name,
            run_name=run_name,
        )
    elif model_name == "MOMENT":
        model, model_artifacts = moment_imputation_main(
            data_dict=source_data,
            model_cfg=cfg["MODELS"][model_name],
            cfg=cfg,
            model_name=model_name,
            run_name=run_name,
        )
    # NOTE: NuwaTS archived - see archived/nuwats/ (not used in final paper)
    # elif model_name == "NuwaTS":
    #     model, model_artifacts = nuwats_imputation_main(
    #         data_dict=source_data,
    #         model_cfg=cfg["MODELS"][model_name],
    #         cfg=cfg,
    #         model_name=model_name,
    #         run_name=run_name,
    #     )
    else:
        logger.error("Model {} not implemented! Typo?".format(model_name))
        raise NotImplementedError("Model {} not implemented!".format(model_name))

    # Save and log all the artifacts created during the training to MLflow
    if model_artifacts is not None:
        model_artifacts["mlflow"] = get_mlflow_info()
        imputation_artifacts = {
            "source_data": source_data,
            "model_artifacts": model_artifacts,
        }

        # with PyPOTS, you only do this here, harmonize later this!
        # Moment computes the metrics already inside the Moment code for example
        # NOTE: NuwaTS archived - special handling removed
        imputation_artifacts["model_artifacts"]["metrics"] = compute_metrics_by_model(
            model_name, imputation_artifacts, cfg
        )

        # Log to MLflow
        save_and_log_imputer_artifacts(
            model, imputation_artifacts, artifacts_dir, cfg, model_name, run_name
        )
        return model, imputation_artifacts
    else:
        # This is None, when you hit like all-NaNs in your predictions and you abort the training
        # e.g. with NuwaTS you might end up here with NaNs in preds
        mlflow.end_run()
        return None, None


# @task(
#     log_prints=True,
#     name="PLR Imputation",
#     description="PLR imputation models (train, evaluate and compute metrics)",
# )
def imputation_PLR_workflow(
    cfg: DictConfig,
    source_name: str,
    source_data: Dict[str, Any],
    run_name: str,
    experiment_name: str,
    _visualize: bool = False,
) -> Optional[Dict[str, Any]]:
    """Execute the PLR imputation workflow.

    Main entry point for training or loading imputation models on PLR data.
    Handles workflow setup, model training/loading, and artifact management.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration including MODELS, EXPERIMENT, and MLFLOW settings.
    source_name : str
        Name identifier for the data source (e.g., outlier detection method name).
    source_data : dict
        Data dictionary from outlier detection containing signals, masks, and metadata.
    run_name : str
        MLflow run name for this imputation experiment.
    experiment_name : str
        MLflow experiment name to log results under.
    _visualize : bool, optional
        Whether to generate visualizations (currently unused). Default is False.

    Returns
    -------
    dict
        Imputation artifacts containing imputed signals, metrics, and MLflow info.
        Returns None if training is skipped and no pre-computed results are loaded.

    Notes
    -----
    The workflow checks for existing trained models and can skip retraining
    if `train_ON` is False based on configuration and existing MLflow runs.
    """
    # Set-up the workflow
    cfg, model_name, run_name, train_ON, best_run, artifacts_dir = setup_PLR_worklow(
        cfg, run_name
    )

    # Task 1) Train the model and impute the missing data
    if train_ON:
        # This often can be time-consuming, so this also saves the results to MLflow
        _, imputation_artifacts = imputation_model_selector(
            source_data=source_data,
            cfg=cfg,
            run_name=run_name,
            artifacts_dir=artifacts_dir,
            model_name=model_name,
            experiment_name=experiment_name,
        )

    else:
        # The time-consuming imputation results can be imported here, so you
        # don't have to re-run the same experiments again
        # logging.info("Reading imputation results from MLflow")
        # imputation_artifacts, _ = retrieve_mlflow_artifacts_from_best_run(
        #     best_run, cfg, model_name
        # )
        logger.info("Skipping the re-computation of the imputation metrics")
        logger.debug("Nothing atm when you skip training, implement later here:")
        logger.debug(
            "If you want to compute new metrics or something without running the whole training"
        )
