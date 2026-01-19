import mlflow
from omegaconf import DictConfig
from loguru import logger

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
        summary_exp_name=experiment_name
    )

    return flow_results


# @flow(
#     log_prints=True,
#     name="PLR Summary",
#     description="Visualization, statistics and summary of the PLR pipeline",
# )
def flow_summarization(cfg: DictConfig):
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
