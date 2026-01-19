from loguru import logger
from omegaconf import DictConfig

from src.ensemble.ensemble_anomaly_detection import ensemble_anomaly_detection
from src.ensemble.ensemble_classification import ensemble_classification
from src.ensemble.ensemble_imputation import ensemble_imputation
from src.ensemble.ensemble_logging import (
    log_ensembling_to_mlflow,
    get_mlflow_ensemble_name,
)
from src.ensemble.ensemble_utils import get_results_from_mlflow_for_ensembling
from src.log_helpers.mlflow_utils import init_mlflow_experiment
from src.log_helpers.log_naming_uris_and_dirs import experiment_name_wrapper


def check_if_for_reprocess(mlflow_ensemble_name, cfg):
    """
    Anomaly detection/Imputation ensembles are quite fast to compute, with classification ensemble it might be worth
    then to skip the processing if you really don't want it, e.g. when debugging, otherwise, it would be nicest to have
    the most fresh ensemble
    """
    # TODO! the logic
    reprocess = True
    logger.warning("Reprocess always now True!")
    return reprocess


def get_ensembled_prediction(
    ensemble_mlflow_runs: dict,
    experiment_name: str,
    cfg: DictConfig,
    task: str,
    sources: dict,
    recompute_metrics: bool = False,
):
    """
    ensemble_mlflow_runs: dict
        ensemble_name: dict (e.g. "pupil_gt")
            submodel_1: pd.Series
            ...
            submodel_n: pd.Series
    """
    if len(ensemble_mlflow_runs) == 0:
        logger.warning(
            "No models to ensemble, exiting the Ensembling Prefect task (task = {})".format(
                task
            )
        )
        return
    else:
        # Compute the metrics for the ensemble(s)
        logger.info("Computing metrics for the ensemble(s)")
        ensemble_output = {}
        for ensemble_name in ensemble_mlflow_runs.keys():
            mlflow_ensemble_name = get_mlflow_ensemble_name(
                task=task,
                ensemble_mlflow_runs=ensemble_mlflow_runs,
                ensemble_name=ensemble_name,
                cfg=cfg,
            )
            reprocess = check_if_for_reprocess(mlflow_ensemble_name, cfg)
            if reprocess:
                ensemble_output[ensemble_name] = {}
                if task == "anomaly_detection":
                    metrics, pred_masks = ensemble_anomaly_detection(
                        ensemble_mlflow_runs[ensemble_name],
                        cfg,
                        experiment_name=experiment_name,
                        ensemble_name=ensemble_name,
                        sources=sources,
                    )
                    ensemble_output[ensemble_name] = {
                        "metrics": metrics,
                        "pred_masks": pred_masks,
                    }
                elif task == "imputation":
                    ensemble_output[ensemble_name] = ensemble_imputation(
                        ensemble_model_runs=ensemble_mlflow_runs[ensemble_name],
                        cfg=cfg,
                        sources=sources,
                        ensemble_name=ensemble_name,
                        recompute_metrics=recompute_metrics,
                    )

                elif task == "classification":
                    ensemble_output[ensemble_name] = ensemble_classification(
                        ensemble_model_runs=ensemble_mlflow_runs[ensemble_name],
                        cfg=cfg,
                        sources=sources,
                        ensemble_name=ensemble_name,
                    )
                else:
                    logger.error(f"Unknown task: {task}")
                    raise ValueError(f"Unknown task: {task}")

    return ensemble_output


def log_ensemble_to_mlflow(
    ensemble_mlflow_runs: dict,
    experiment_name: str,
    cfg: DictConfig,
    ensemble_output: dict,
    task: str,
    recompute_metrics: bool = False,
):
    init_mlflow_experiment(mlflow_cfg=cfg["MLFLOW"], experiment_name=experiment_name)

    for ensemble_name, output_dict in ensemble_output.items():
        if task == "anomaly_detection":
            if "all_variants" not in ensemble_name:
                log_ensembling_to_mlflow(
                    experiment_name=experiment_name,
                    ensemble_mlflow_runs=ensemble_mlflow_runs,
                    ensemble_name=ensemble_name,
                    metrics=output_dict["metrics"],
                    pred_masks=output_dict["pred_masks"],
                    cfg=cfg,
                    task=task,
                )
        elif task == "imputation":
            log_ensembling_to_mlflow(
                experiment_name=experiment_name,
                ensemble_mlflow_runs=ensemble_mlflow_runs,
                ensemble_name=ensemble_name,
                output_dict=output_dict,
                cfg=cfg,
                task=task,
            )
        elif task == "classification":
            log_ensembling_to_mlflow(
                experiment_name=experiment_name,
                ensemble_mlflow_runs=ensemble_mlflow_runs,
                ensemble_name=ensemble_name,
                output_dict=output_dict,
                cfg=cfg,
                task=task,
            )
        else:
            logger.error(f"Unknown task: {task}")
            raise ValueError(f"Unknown task: {task}")


# @task(
#     log_prints=True,
#     name="Ensemble Anomaly Detection models",
#     description="Average anomaly detection outputs from different models",
# )
def task_ensemble(
    cfg: DictConfig, task: str, sources: dict, recompute_metrics: bool = False
):
    # Computationally independent so could be run as a flow, but seems more coherent on flow diagram
    # to be run inside the "imputation" flow
    # TODO! harmonize the naming, they could be the same without all this if/elif/else
    if task == "anomaly_detection":
        task_key = "OUTLIER_DETECTION"
    elif task == "imputation":
        task_key = "IMPUTATION"
    elif task == "classification":
        task_key = "CLASSIFICATION"
    else:
        logger.error(f"Unknown task: {task}")
        raise ValueError(f"Unknown task: {task}")

    prev_experiment_name = experiment_name_wrapper(
        experiment_name=cfg["PREFECT"]["FLOW_NAMES"][task_key], cfg=cfg
    )

    logger.info(
        "TASK (flow-like) | Name: {}".format(cfg["PREFECT"]["FLOW_NAMES"][task_key])
    )
    logger.info("=====================")

    # Ensemble from the imputation models
    ensemble_mlflow_runs = get_results_from_mlflow_for_ensembling(
        experiment_name=prev_experiment_name,
        cfg=cfg,
        task=task,
        recompute_metrics=recompute_metrics,
    )

    # Get ensemble predictions
    # 1) read artifacts from MLflow
    # 2) ensemble the predictions
    # 3) compute the metrics
    if ensemble_mlflow_runs is not None:
        ensemble_output = get_ensembled_prediction(
            ensemble_mlflow_runs=ensemble_mlflow_runs,
            experiment_name=prev_experiment_name,
            cfg=cfg,
            task=task,
            sources=sources,
            recompute_metrics=recompute_metrics,
        )

        if not recompute_metrics:
            # Finally log to MLflow
            log_ensemble_to_mlflow(
                ensemble_mlflow_runs,
                experiment_name=prev_experiment_name,
                cfg=cfg,
                ensemble_output=ensemble_output,
                task=task,
                recompute_metrics=recompute_metrics,
            )
    else:
        logger.warning('Skipping ensembled prediction as no submodel sources were found')
