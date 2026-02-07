from loguru import logger
from omegaconf import DictConfig

from src.data_io.define_sources_for_flow import define_sources_for_flow
from src.ensemble.tasks_ensembling import task_ensemble
from src.imputation.imputation_main import imputation_PLR_workflow
from src.log_helpers.log_naming_uris_and_dirs import experiment_name_wrapper
from src.log_helpers.mlflow_utils import init_mlflow_experiment
from src.orchestration.hyperparameter_sweep_utils import define_hyperparam_group


# @flow(
#     log_prints=True,
#     name="PLR Imputation",
#     description="Hyperparameter sweep of Impuation models",
# )
def flow_imputation(cfg: DictConfig) -> dict:
    """Execute the PLR imputation flow with hyperparameter sweep.

    Orchestrates the imputation pipeline by iterating over all combinations
    of data sources (from outlier detection) and hyperparameter configurations.
    Also handles ensembling of trained imputation models.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration containing PREFECT flow names, MLFLOW settings,
        and hyperparameter configurations for imputation models.

    Returns
    -------
    dict
        Results from the imputation pipeline (implicitly via MLflow logging).

    Notes
    -----
    The flow performs the following steps:
    1. Define hyperparameter groups from configuration
    2. Download outlier detection outputs from MLflow
    3. Define data sources (outlier detection + ground truth)
    4. Run imputation for each source x hyperparameter combination
    5. Recompute metrics for all submodels
    6. Create ensemble models from submodels
    """
    # Flatten the hyperparameter groups to a dict
    hyperparams_group = define_hyperparam_group(cfg, task="imputation")

    # Download data from MLflow (output from outlier detection)
    prev_experiment_name = experiment_name_wrapper(
        experiment_name=cfg["PREFECT"]["FLOW_NAMES"]["OUTLIER_DETECTION"], cfg=cfg
    )

    # Define the "sources" for the flow, as in the outlier detection output along with the
    # ground truth of manually annotated imputation masks
    sources = define_sources_for_flow(
        prev_experiment_name=prev_experiment_name, cfg=cfg
    )

    # Set the experiment name
    experiment_name = experiment_name_wrapper(
        experiment_name=cfg["PREFECT"]["FLOW_NAMES"]["IMPUTATION"], cfg=cfg
    )
    init_mlflow_experiment(mlflow_cfg=cfg["MLFLOW"], experiment_name=experiment_name)

    no_of_runs = len(sources) * len(hyperparams_group)
    run_idx = 0
    for source_idx, (source_name, source_data) in enumerate(sources.items()):
        for idx, (cfg_group_name, cfg_group) in enumerate(hyperparams_group.items()):
            logger.info(f"Source #{source_idx + 1}/{len(sources)}: {source_name}")
            logger.info(
                f"Running pipeline for hyperparameter group #{idx + 1}/{len(hyperparams_group)}: {cfg_group_name}"
            )
            run_name = f"{cfg_group_name}__{source_name}"
            logger.info(f"Run name #{run_idx + 1}/{no_of_runs}: {run_name}")
            run_idx += 1
            imputation_PLR_workflow(
                cfg=cfg_group,
                source_name=source_name,
                source_data=source_data,
                experiment_name=experiment_name,
                run_name=run_name,
            )

    # The ensembling part will fetch the trained imputation models from MLflow

    # First re-computing metrics for all the submodels, and making sure that they are correct
    task_ensemble(cfg=cfg, task="imputation", sources=sources, recompute_metrics=True)

    # Then ensembling the submodels
    task_ensemble(cfg=cfg, task="imputation", sources=sources, recompute_metrics=False)
