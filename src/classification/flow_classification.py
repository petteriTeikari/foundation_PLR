from loguru import logger
from omegaconf import DictConfig

from src.classification.subflow_feature_classification import (
    flow_feature_classification,
)
from src.classification.subflow_ts_classification import flow_ts_classification  # noqa: F401
from src.log_helpers.log_naming_uris_and_dirs import (
    experiment_name_wrapper,
)
from src.log_helpers.mlflow_utils import init_mlflow_experiment


# @flow(
#     log_prints=True,
#     name="PLR Classification",
#     description="Classify Glaucoma from the PLR features",
# )
def flow_classification(cfg: DictConfig) -> None:
    """
    Main classification flow for glaucoma screening from PLR features.

    Orchestrates the classification pipeline including feature-based
    and time-series classification approaches. Initializes MLflow
    experiment and delegates to subflows.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration with PREFECT flow names and settings.

    Notes
    -----
    Time-series classification is currently disabled as it showed
    limited promise after refactoring.
    """
    experiment_name = experiment_name_wrapper(
        experiment_name=cfg["PREFECT"]["FLOW_NAMES"]["CLASSIFICATION"], cfg=cfg
    )
    logger.info("FLOW | Name: {}".format(experiment_name))
    logger.info("=====================")
    prev_experiment_name = experiment_name_wrapper(
        experiment_name=cfg["PREFECT"]["FLOW_NAMES"]["FEATURIZATION"], cfg=cfg
    )

    # Init the MLflow experiment
    init_mlflow_experiment(experiment_name=experiment_name)

    # Classify from hand-crafted features/embeddings
    flow_feature_classification(cfg, prev_experiment_name)

    # Classify from time series
    ts_cls = False
    if ts_cls:
        raise NotImplementedError(
            "Need to be finished, new bug with the refactoring, but did not seem promising"
        )
