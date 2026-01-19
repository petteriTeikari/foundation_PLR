import argparse
import os
import sys
import hydra

from loguru import logger
from omegaconf import DictConfig

# Import environment variables
from dotenv import load_dotenv

from src.anomaly_detection.flow_anomaly_detection import flow_anomaly_detection
from src.classification.flow_classification import (
    flow_classification,
)

from src.data_io.flow_data import flow_import_data
from src.deploy.flow_deployment import flow_deployment
from src.featurization.flow_featurization import flow_featurization
from src.imputation.flow_imputation import flow_imputation
from src.log_helpers.hydra_utils import add_hydra_cli_args
from src.log_helpers.log_utils import setup_loguru, log_loguru_log_to_prefect
from src.log_helpers.mlflow_utils import init_mlflow
from src.orchestration.prefect_utils import (
    pre_flow_prefect_checks,
)
from src.summarization.flow_summarization import flow_summarization
from src.utils import check_for_device

LOG_FILE_PATH = setup_loguru()
read_ok = load_dotenv()
if not read_ok:
    logger.warning("Could not read .env file!")
else:
    logger.info(".env file imported successfully.")

# Get the absolute local path for this project (in which 'src' is a subdirectory)
src_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.split(src_path)[0]
sys.path.insert(0, project_path)


def parse_args():
    parser = argparse.ArgumentParser(description="PLR Pipeline")
    parser.add_argument(
        "-c",
        "--config-name",
        type=str,
        required=False,
        default="debug_run",
        help="Name of your task-specific Hydra .yaml file (in 'configs' dir), e.g. 'hyperparam_sweep'",
    )
    parser.add_argument(
        "-cp",
        "--config-path",
        type=str,
        required=False,
        default="../configs",
        help="",
    )
    return parser.parse_args()


# @flow(
#     log_prints=True,
#     name="PLR EXPERIMENT",
#     description="Experiment Prefect Pipeline for analyzing the effects of PLR Imputation models to "
#     "ophthalmic classification performance from processed PLR time series",
# )
def flowMain_PLR_experiment(cfg: DictConfig) -> None:
    """
    So let's say we have 3 anomaly detection models, 3 imputation models, and 3 classification models.
    - Anomaly detection model Flow would run 3 times
    - Imputation model Flow would run 3*3 times = 9 times
    - Classification model Flow would run 3*3*3 times = 27 times (total of 3+9+27 = 39 flow runs)
    - After all the model evaluations, we could then deploy them to production?
    """
    # Init MLflow, as in set Tracking URI
    init_mlflow(cfg=cfg)
    prefect_flows = cfg["PREFECT"]["PROCESS_FLOWS"]

    if prefect_flows["OUTLIER_DETECTION"]:
        df = flow_import_data(cfg=cfg)
        flow_anomaly_detection(cfg=cfg, df=df)

    if prefect_flows["IMPUTATION"]:
        flow_imputation(cfg=cfg)

    if prefect_flows["FEATURIZATION"]:
        flow_featurization(cfg=cfg)

    if prefect_flows["CLASSIFICATION"]:
        flow_classification(cfg=cfg)

    if prefect_flows["SUMMARIZATION"]:
        flow_summarization(cfg=cfg)

    if prefect_flows["DEPLOYMENT"]:
        flow_deployment(cfg=cfg)

    # Log everything to Prefect
    # log_loguru_log_to_prefect(filepath=LOG_FILE_PATH, description="PLR Pipeline log")


@hydra.main(version_base=None)
def run_main_prefect_PLR_flow(cfg: DictConfig) -> None:
    # Prefect checks
    pre_flow_prefect_checks(prefect_cfg=cfg["PREFECT"])

    # Check for hardware / device (CUDA or CPU atm before anything fancier)
    cfg = check_for_device(cfg=cfg)

    # "PLR Experiment flow"
    # x anomaly detection models * y imputation models * z classification models
    flowMain_PLR_experiment(cfg=cfg)


if __name__ == "__main__":
    args = parse_args()
    add_hydra_cli_args(args)
    run_main_prefect_PLR_flow()
