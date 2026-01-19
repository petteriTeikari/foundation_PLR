from loguru import logger
from omegaconf import DictConfig
import polars as pl

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
    Placeholder for anomaly detection
    - df: polars.DataFrame
        - 'pupil_orig': Original recording coming from the pupillometer, the pupillometry software has rejected already
                        some clear artifacts (null), but there are still a lot of blink artifacts (and others) left
        - 'pupil_raw': This is now the "output from anomaly detection", or the ground truth for this anomaly detection
                       flow that you would like model. Now all the outliers are set to null, and you can impute then
                       the missing values to this artifact-free PLR data
        - 'gt': The ground truth (for imputation) that contains manually-supervised imputation (manually placed points,
                combined with MissForest imputation), the imputation is then denoised with CEEMD. This signal lacks
                the "high-frequency noise" that is present in the raw signal
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
    task_ensemble(cfg=cfg, task="anomaly_detection", sources=data_dict, recompute_metrics=True)
    task_ensemble(
        cfg=cfg, task="anomaly_detection", sources=data_dict, recompute_metrics=False
    )
