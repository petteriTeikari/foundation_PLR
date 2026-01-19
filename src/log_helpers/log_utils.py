import os
import sys
from logging import exception
import datetime
from prefect.artifacts import create_markdown_artifact
from src.utils import get_artifacts_dir
from loguru import logger  # https://betterstack.com/community/guides/logging/loguru/


def define_run_name(cfg):
    return "{}_v{}".format(cfg["NAME"], cfg["VERSION"])


def define_suffix_to_run_name(model_name):
    # Placeholder atm
    return f"_{model_name}_ph1"


def update_run_name(run_name, base_run_name):
    return run_name + "_" + base_run_name


def setup_loguru():
    min_level = "INFO"

    def my_filter(record):
        return record["level"].no >= logger.level(min_level).no

    logger.remove()
    # https://stackoverflow.com/a/76583603/6412152
    log_dir = get_artifacts_dir(
        service_name="hydra"
    )  # harmonize naming maybe later? as this not Hydra log per se
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "pipeline_PLR.log")
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    logger.add(
        sys.stderr, filter=my_filter, colorize=True, backtrace=True, diagnose=True
    )
    logger.add(
        log_file_path, level=min_level, colorize=False, backtrace=True, diagnose=True
    )

    return log_file_path


def log_loguru_log_to_prefect(filepath: str, description: str):
    # https://docs.prefect.io/3.0/develop/artifacts#create-markdown-artifacts
    # Hacky solution to get the final log without any nice formatting
    try:
        with open(filepath, "r") as f:
            log_content = f.read()
        try:
            create_markdown_artifact(
                key="loguru-log",
                markdown=log_content,
                description=description,
            )
        except exception as e:
            logger.error(f"Failed to log the loguru-log as markdown to Prefect: {e}")
            return
    except exception as e:
        logger.error(f"Failed to read the log file: {e}")
        return


def get_datetime_as_string(use_gmt_time=False):
    if use_gmt_time:
        dt_now = datetime.datetime.now(datetime.timezone.utc)
    else:
        dt_now = datetime.datetime.now()
    date_string = dt_now.strftime("%Y%m%d-%H%M%S")
    return date_string
