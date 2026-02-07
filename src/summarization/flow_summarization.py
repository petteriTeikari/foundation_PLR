import mlflow
from loguru import logger
from omegaconf import DictConfig

from src.data_io.flow_data import flow_import_data
from src.log_helpers.log_naming_uris_and_dirs import experiment_name_wrapper
from src.log_helpers.mlflow_utils import init_mlflow_experiment
from src.summarization.summarization_data_wrangling import get_summarization_flow_data
from src.summarization.summary_analysis_main import summary_analysis_main


# @task(
#     log_prints=True,
#     name="Get summarization data",
#     description="...",
# )
def get_summarization_data(
    cfg: DictConfig, experiment_name: str, summary_exp_name: str
):
    """Collect summarization data from all pipeline stages.

    Gathers results from outlier detection, imputation, featurization,
    and classification experiments into a unified dictionary.

    Parameters
    ----------
    cfg : DictConfig
        Configuration containing PREFECT.FLOW_NAMES for each stage.
    experiment_name : str
        Name of the summary experiment.
    summary_exp_name : str
        MLflow experiment name for summaries.

    Returns
    -------
    dict
        Dictionary with keys 'outlier_detection', 'imputation',
        'featurization', and 'classification', each containing
        that stage's summarization data.
    """
    flow_results = {}

    # Summarize outlier detection experiment
    flow_results["outlier_detection"] = get_summarization_flow_data(
        cfg,
        experiment_name=cfg["PREFECT"]["FLOW_NAMES"]["OUTLIER_DETECTION"],
        summary_exp_name=experiment_name,
    )

    # Summarize imputation experiment
    flow_results["imputation"] = get_summarization_flow_data(
        cfg,
        experiment_name=cfg["PREFECT"]["FLOW_NAMES"]["IMPUTATION"],
        summary_exp_name=experiment_name,
    )

    # Summarize featurization experiment
    flow_results["featurization"] = get_summarization_flow_data(
        cfg,
        experiment_name=cfg["PREFECT"]["FLOW_NAMES"]["FEATURIZATION"],
        summary_exp_name=experiment_name,
    )

    # Summarize classification experiment
    flow_results["classification"] = get_summarization_flow_data(
        cfg,
        experiment_name=cfg["PREFECT"]["FLOW_NAMES"]["CLASSIFICATION"],
        summary_exp_name=experiment_name,
    )

    return flow_results


# @flow(
#     log_prints=True,
#     name="PLR Summary",
#     description="Visualization, statistics and summary of the PLR pipeline",
# )
def flow_summarization(cfg: DictConfig):
    """Main summarization flow for the PLR pipeline.

    Orchestrates the collection, analysis, and export of results from
    all pipeline stages. Initializes MLflow tracking and coordinates
    data import and analysis tasks.

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary containing:
        - PREFECT.FLOW_NAMES: Experiment names for each stage
        - SUMMARIZATION: Import/export settings
    """
    experiment_name = experiment_name_wrapper(
        experiment_name=cfg["PREFECT"]["FLOW_NAMES"]["SUMMARY"], cfg=cfg
    )
    logger.info("FLOW | Name: {}".format(experiment_name))

    # Init the MLflow experiment
    init_mlflow_experiment(experiment_name=experiment_name)
    run_name = "summary_tmp"
    # duckdb now refers to disk, with both .db and .pickle, one day maybe, one large .db file?
    if not cfg["SUMMARIZATION"]["import_from_duckdb"]:
        mlflow.start_run(run_name=run_name)

    # Get summarization data (outlier detection, imputation, featurization)
    # classification, rather memory intensive when dumping into one file, see about it later
    flow_results = get_summarization_data(
        cfg,
        experiment_name,
        summary_exp_name=experiment_name,
    )

    # Get the input data
    flow_results["input_df"] = flow_import_data(cfg=cfg)

    # Analyse
    summary_analysis_main(flow_results=flow_results, cfg=cfg)

    # End the run
    mlflow.end_run()
