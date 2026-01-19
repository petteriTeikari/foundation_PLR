import polars as pl
from omegaconf import DictConfig
from loguru import logger
from src.anomaly_detection.anomaly_utils import (
    if_remote_anomaly_detection,
    get_anomaly_detection_results_from_mlflow,
)
from src.anomaly_detection.outlier_sigllm import outlier_sigllm_wrapper
from src.anomaly_detection.outlier_sklearn import (
    outlier_sklearn_wrapper,
)
from src.anomaly_detection.outlier_tsb_ad import outlier_tsb_ad_wrapper
from src.anomaly_detection.timesnet_wrapper import timesnet_outlier_wrapper
from src.anomaly_detection.units.units_outlier import units_outlier_wrapper

from src.log_helpers.log_naming_uris_and_dirs import update_outlier_detection_run_name
from src.anomaly_detection.momentfm_outlier.momentfm_outlier import (
    momentfm_outlier_main,
)
from src.log_helpers.mlflow_utils import (
    init_mlflow_run,
    log_mlflow_params,
    init_mlflow_experiment,
)
from src.orchestration.debug_utils import debug_train_only_for_one_epoch
from src.anomaly_detection.outlier_prophet import outlier_prophet_wrapper


# from src.outlier_prophet import outlier_prophet_wrapper


def outlier_detection_selector(
    df: pl.DataFrame,
    cfg: DictConfig,
    experiment_name: str,
    run_name: str,
    model_name: str,
):
    # Init the MLflow experiment
    init_mlflow_experiment(experiment_name=experiment_name)

    # quick fix TODO! move somewhere before
    if run_name == "TimesNet":
        if cfg["OUTLIER_MODELS"][run_name]["MODEL"]["train_on"] == "pupil_orig_imputed":
            run_name += "-orig"

    # Init MLflow run
    init_mlflow_run(
        mlflow_cfg=cfg["MLFLOW"],
        run_name=run_name,
        cfg=cfg,
        experiment_name=experiment_name,
    )

    # Log MLflow parameters
    log_mlflow_params(
        mlflow_params=cfg["OUTLIER_MODELS"][model_name]["MODEL"],
        model_name=model_name,
        run_name=run_name,
    )

    logger.info("Outlier Detection with model {}".format(model_name))
    if model_name == "MOMENT":
        outlier_artifacts, model = momentfm_outlier_main(
            df=df,
            cfg=cfg,
            outlier_model_cfg=cfg["OUTLIER_MODELS"][model_name],
            experiment_name=experiment_name,
            run_name=run_name,
        )
    elif model_name == "TimesNet":
        outlier_artifacts, model = timesnet_outlier_wrapper(
            df=df,
            cfg=cfg,
            outlier_model_cfg=cfg["OUTLIER_MODELS"][model_name],
            experiment_name=experiment_name,
            run_name=run_name,
        )
    elif model_name == "UniTS":
        outlier_artifacts, model = units_outlier_wrapper(
            df=df,
            cfg=cfg,
            model_cfg=cfg["OUTLIER_MODELS"][model_name],
            experiment_name=experiment_name,
            run_name=run_name,
        )
    elif model_name == "LOF":
        outlier_artifacts, model = outlier_sklearn_wrapper(
            df=df,
            cfg=cfg,
            model_cfg=cfg["OUTLIER_MODELS"][model_name],
            experiment_name=experiment_name,
            run_name=run_name,
            model_name=model_name,
        )
    elif model_name == "OneClassSVM":
        outlier_artifacts, model = outlier_sklearn_wrapper(
            df=df,
            cfg=cfg,
            model_cfg=cfg["OUTLIER_MODELS"][model_name],
            experiment_name=experiment_name,
            run_name=run_name,
            model_name=model_name,
        )
    elif model_name == "PROPHET":
        outlier_artifacts, model = outlier_prophet_wrapper(
            df=df,
            cfg=cfg,
            model_cfg=cfg["OUTLIER_MODELS"][model_name],
            experiment_name=experiment_name,
            run_name=run_name,
        )
    elif model_name == "SigLLM":
        outlier_artifacts, model = outlier_sigllm_wrapper(
            df=df,
            cfg=cfg,
            model_cfg=cfg["OUTLIER_MODELS"][model_name],
            experiment_name=experiment_name,
            run_name=run_name,
        )
    elif model_name == "SubPCA" or model_name == "EIF":
        outlier_artifacts, model = outlier_tsb_ad_wrapper(
            df=df,
            cfg=cfg,
            model_cfg=cfg["OUTLIER_MODELS"][model_name],
            experiment_name=experiment_name,
            run_name=run_name,
            model_name=model_name,
        )
    else:
        logger.error(f"{model_name} Model not implemented yet")
        raise NotImplementedError(f"{model_name} Model not implemented yet")

    return outlier_artifacts, model


def outlier_detection_PLR_workflow(
    df: pl.DataFrame, cfg: DictConfig, experiment_name: str, run_name: str
) -> dict:
    # Set-up the workflow
    model_name = list(cfg["OUTLIER_MODELS"].keys())[0]

    # Debug: set the epochs to 1
    if cfg["EXPERIMENT"]["debug"]:
        logger.warning("Debug mode is on, training (finetuning) only for one epoch")
        cfg = debug_train_only_for_one_epoch(cfg)

    # If you wish to skip the recomputation, but no previous runs are found
    recompute_anomaly_detection = if_remote_anomaly_detection(
        try_to_recompute=cfg["OUTLIER_DETECTION"]["re_compute"],
        anomaly_cfg=cfg["OUTLIER_DETECTION"],
        experiment_name=experiment_name,
        cfg=cfg,
    )

    if recompute_anomaly_detection:
        outlier_artifacts, _ = outlier_detection_selector(
            df=df,
            cfg=cfg,
            experiment_name=experiment_name,
            run_name=run_name,
            model_name=model_name,
        )
    else:
        logger.info(f"{run_name} was found trained already")
        # Atm nothing happens after this, but when you have some viz or something, you could read these
        read_from_mlflow = False
        if read_from_mlflow:
            logger.info("Reading Anomaly Detection results from MLflow")
            run_name = update_outlier_detection_run_name(cfg)
            outlier_artifacts, _ = get_anomaly_detection_results_from_mlflow(
                experiment_name=experiment_name,
                cfg=cfg,
                run_name=run_name,
                model_name=model_name,
            )

    # Task) Recompute final metrics (destandardize signals for MSE?)

    # Task) You could visualize outlier detection results here (+mlflow)

    # Task) Create one long dataframe?
