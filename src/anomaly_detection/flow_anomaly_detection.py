import polars as pl
from loguru import logger
from omegaconf import DictConfig

from src.anomaly_detection.anomaly_detection import (
    outlier_detection_PLR_workflow,
)
from src.data_io.data_wrangler import convert_df_to_dict
from src.ensemble.tasks_ensembling import task_ensemble
from src.log_helpers.log_naming_uris_and_dirs import (
    get_outlier_detection_experiment_name,
)
from src.orchestration.hyperparameter_sweep_utils import define_hyperparam_group


# @flow(
#     log_prints=True,
#     name="PLR Anomaly Detection",
#     description="Detect anomalies, so you would actually get some data automatically to impute",
# )
def flow_anomaly_detection(cfg: DictConfig, df: pl.DataFrame) -> pl.DataFrame:
    """
    Run the complete anomaly detection flow for PLR data.

    Orchestrates outlier detection across multiple hyperparameter configurations
    and creates ensembles of the individual models.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing model and experiment parameters.
    df : pl.DataFrame
        Input PLR data with the following columns:
        - 'pupil_orig': Original recording from the pupillometer. The software
          has rejected some clear artifacts (null), but blink artifacts remain.
        - 'pupil_raw': Output from anomaly detection (ground truth for modeling).
          All outliers are set to null for subsequent imputation.
        - 'gt': Ground truth for imputation containing manually-supervised
          imputation (manually placed points + MissForest), denoised with CEEMD.
          This signal lacks the high-frequency noise present in raw signal.

    Returns
    -------
    pl.DataFrame
        The input DataFrame (currently unchanged; results logged to MLflow).

    Notes
    -----
    This function:
    1. Runs outlier detection for each hyperparameter configuration
    2. Logs results to MLflow
    3. Creates ensemble models from individual detectors
    """
    experiment_name = get_outlier_detection_experiment_name(cfg)
    logger.info("FLOW | Name: {}".format(experiment_name))
    logger.info("=====================")

    hyperparams_group = define_hyperparam_group(cfg, task="outlier_detection")
    for cfg_key, cfg_hyperparam in hyperparams_group.items():
        outlier_detection_PLR_workflow(
            df=df,
            cfg=cfg_hyperparam,
            experiment_name=experiment_name,
            run_name=cfg_key,
        )

    # The ensembling part will fetch the trained imputation models from MLflow
    data_dict = convert_df_to_dict(data_df=df, cfg=cfg)
    task_ensemble(
        cfg=cfg, task="anomaly_detection", sources=data_dict, recompute_metrics=True
    )
    task_ensemble(
        cfg=cfg, task="anomaly_detection", sources=data_dict, recompute_metrics=False
    )
