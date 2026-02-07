from loguru import logger
from omegaconf import DictConfig

from src.featurization.featurize_PLR import featurization_script
from src.log_helpers.log_naming_uris_and_dirs import (
    get_feature_name_from_cfg,
)


# Subflow to flow_featurization
# @flow(
#     log_prints=True,
#     name="PLR Featurization",
#     description="Featurize the data for PLR, from raw data, imputed single models and ensembled models",
# )
def flow_handcrafted_featurization(
    cfg: DictConfig, sources: dict, experiment_name: str, prev_experiment_name: str
) -> None:
    """Execute handcrafted featurization for all data sources.

    Iterates through all imputation sources and feature configurations,
    running the featurization script for each combination.

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary with PLR_FEATURIZATION settings.
    sources : dict
        Dictionary of data sources keyed by source name.
    experiment_name : str
        MLflow experiment name for featurization.
    prev_experiment_name : str
        Previous experiment name (imputation).
    """
    # Define the featurization methods
    # 1) You could use multiple .YAML files to define the hand-crafted features
    # 2) You could train MOMENT embeddings and use those for your classifier on next stage (e.g. XGBoost
    # placeholder atm
    feature_cfgs = {get_feature_name_from_cfg(cfg): cfg["PLR_FEATURIZATION"]}

    no_of_runs = len(sources) * len(feature_cfgs)
    run_idx = 0
    # Featurize (or skip if previous results found from MLflow)
    for source_idx, (source_name, source_data) in enumerate(sources.items()):
        for idx, (featurization_method, feature_cfg) in enumerate(feature_cfgs.items()):
            logger.info(f"Source #{source_idx + 1}/{len(sources)}: {source_name}")
            logger.info(
                f"Running pipeline for featurization method #{idx + 1}/{len(feature_cfgs)}: {featurization_method}"
            )
            run_name = f"{featurization_method}__{source_name}"
            logger.info(f"Run name #{run_idx + 1}/{no_of_runs}: {run_name}")
            run_idx += 1

            featurization_script(
                experiment_name=experiment_name,
                prev_experiment_name=prev_experiment_name,
                cfg=cfg,
                source_name=source_name,
                source_data=source_data,
                featurization_method=featurization_method,
                feature_cfg=feature_cfg,
                run_name=run_name,
            )
