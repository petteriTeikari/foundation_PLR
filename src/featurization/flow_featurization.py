from loguru import logger
from omegaconf import DictConfig

from src.data_io.define_sources_for_flow import (
    define_sources_for_flow,
)
from src.featurization.embedding.subflow_embedding import flow_embedding
from src.featurization.subflow_handcrafted_featurization import flow_handcrafted_featurization
from src.log_helpers.mlflow_utils import init_mlflow_experiment
from src.log_helpers.log_naming_uris_and_dirs import (
    experiment_name_wrapper,
)


# @flow(
#     log_prints=True,
#     name="PLR Featurization",
#     description="Featurize the data for PLR, from raw data, imputed single models and ensembled models",
# )
def flow_featurization(cfg: DictConfig):
    experiment_name = experiment_name_wrapper(
        experiment_name=cfg["PREFECT"]["FLOW_NAMES"]["FEATURIZATION"], cfg=cfg
    )
    logger.info("FLOW | Name: {}".format(experiment_name))
    logger.info("=====================")
    prev_experiment_name = experiment_name_wrapper(
        experiment_name=cfg["PREFECT"]["FLOW_NAMES"]["IMPUTATION"], cfg=cfg
    )

    # Initialize the MLflow experiment
    init_mlflow_experiment(mlflow_cfg=cfg["MLFLOW"], experiment_name=experiment_name)

    # Get the data sources (from imputation, and from original ground truth DuckDB database)
    sources = define_sources_for_flow(
        cfg=cfg, prev_experiment_name=prev_experiment_name, task="imputation"
    )

    # Get the handcrafed features
    flow_handcrafted_featurization(
        cfg=cfg,
        sources=sources,
        experiment_name=experiment_name,
        prev_experiment_name=prev_experiment_name,
    )

    # Get the "deep features" as in embeddings e.g. from foundation moodels
    compute_embeddings = False # not so useful, so quick'n'dirty skip
    if compute_embeddings:
        flow_embedding(
            cfg=cfg,
            sources=sources,
            experiment_name=experiment_name,
            prev_experiment_name=prev_experiment_name,
        )
