import datetime
import sys
from logging import exception

from loguru import logger  # https://betterstack.com/community/guides/logging/loguru/
from prefect.artifacts import create_markdown_artifact

from src.utils import get_artifacts_dir


def define_run_name(cfg) -> str:
    """Define run name from configuration name and version.

    Parameters
    ----------
    cfg : DictConfig
        Configuration with 'NAME' and 'VERSION' keys.

    Returns
    -------
    str
        Run name in format '{name}_v{version}'.
    """
    return "{}_v{}".format(cfg["NAME"], cfg["VERSION"])


def define_suffix_to_run_name(model_name) -> str:
    """Generate suffix for run name based on model name.

    Parameters
    ----------
    model_name : str
        Name of the model.

    Returns
    -------
    str
        Suffix in format '_{model_name}_ph1'.

    Notes
    -----
    This is a placeholder implementation.
    """
    # Placeholder atm
    return f"_{model_name}_ph1"


def update_run_name(run_name, base_run_name) -> str:
    """Append base run name to existing run name.

    Parameters
    ----------
    run_name : str
        Existing run name.
    base_run_name : str
        Base name to append.

    Returns
    -------
    str
        Combined run name with underscore separator.
    """
    return run_name + "_" + base_run_name


def setup_loguru() -> str:
    """Configure loguru logger for console and file output.

    Sets up logging to stderr with color and to a file in the artifacts
    directory. Removes any existing log file before starting.

    Returns
    -------
    str
        Path to the log file.
    """
    min_level = "INFO"

    def my_filter(record):
        return record["level"].no >= logger.level(min_level).no

    logger.remove()
    # https://stackoverflow.com/a/76583603/6412152
    log_dir = get_artifacts_dir(
        service_name="hydra"
    )  # harmonize naming maybe later? as this not Hydra log per se
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / "pipeline_PLR.log"
    if log_file_path.exists():
        log_file_path.unlink()
    logger.add(
        sys.stderr, filter=my_filter, colorize=True, backtrace=True, diagnose=True
    )
    logger.add(
        str(log_file_path),
        level=min_level,
        colorize=False,
        backtrace=True,
        diagnose=True,
    )

    return str(log_file_path)


def log_loguru_log_to_prefect(filepath: str, description: str) -> None:
    """Log contents of loguru log file as Prefect markdown artifact.

    Parameters
    ----------
    filepath : str
        Path to the log file.
    description : str
        Description for the Prefect artifact.
    """
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


def get_datetime_as_string(use_gmt_time=False) -> str:
    """Get current datetime as formatted string.

    Parameters
    ----------
    use_gmt_time : bool, default False
        If True, use UTC time; otherwise use local time.

    Returns
    -------
    str
        Datetime string in format 'YYYYMMDD-HHMMSS'.
    """
    if use_gmt_time:
        dt_now = datetime.datetime.now(datetime.timezone.utc)
    else:
        dt_now = datetime.datetime.now()
    date_string = dt_now.strftime("%Y%m%d-%H%M%S")
    return date_string
