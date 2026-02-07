import hashlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from loguru import logger
from omegaconf import DictConfig, open_dict


def get_repo_root(base_name: str = "repo_PLR") -> Path:
    """
    Get the root directory of the repository.

    Determines the repository root based on the current working directory,
    handling cases where the script is run from src/ or subdirectories.

    Parameters
    ----------
    base_name : str, optional
        Expected name of the repository root directory, by default "repo_PLR".

    Returns
    -------
    Path
        Path to the repository root directory.
    """
    cwd = Path.cwd()
    if cwd.name == "src":
        repo_root = cwd.parent
    elif cwd.name != base_name:
        # subfolder in "src", make recursive later
        repo_root = cwd.parent.parent
    else:
        repo_root = cwd
    return repo_root


def get_data_dir(data_path: str = "data") -> Path:
    """
    Get the path to the data directory.

    Parameters
    ----------
    data_path : str, optional
        Relative path to the data directory from repo root, by default "data".

    Returns
    -------
    Path
        Absolute path to the data directory.
    """
    data_dir = get_repo_root() / data_path
    return data_dir


def get_artifacts_dir(
    service_name: str = "mlflow", run_name: str = None, subdir: str = None
) -> Path:
    """
    Get the path to the artifacts directory for a given service.

    Creates nested directory structure based on service name, run name, and
    optional subdirectory. Directories are created if they don't exist.

    Parameters
    ----------
    service_name : str, optional
        Name of the service (e.g., "mlflow", "hydra", "duckdb"), by default "mlflow".
    run_name : str, optional
        Name of the specific run to create a subdirectory for, by default None.
    subdir : str, optional
        Additional subdirectory level within the run directory, by default None.

    Returns
    -------
    Path
        Path to the artifacts directory for the specified service and run.
    """
    artifacts_dir = get_repo_root() / "artifacts"
    # TODO! add "tmp" to most of the dirs as they are meant to be mostly temporary
    #  and permanent artifacts are stored in MLflow (or Prefect)
    if service_name == "mlflow":
        logger.debug("MLflow Artifacts directory is used")
        artifacts_dir = artifacts_dir / service_name
    elif service_name == "hydra":
        logger.debug("Hydra Artifacts directory is used")
        artifacts_dir = artifacts_dir / service_name
    elif service_name == "best_models":
        # "Best" Pickles, DuckDBs, etc. go here atm
        logger.debug("Best models directory is used")
        # artifacts_dir stays as is (no subdirectory added)
    elif service_name == "features":
        # "Best" Pickles, DuckDBs, etc. go here atm
        logger.debug("Features directory is used")
        artifacts_dir = artifacts_dir / service_name
    elif service_name == "pypots":
        logger.debug("PyPOTS directory is used")
        artifacts_dir = artifacts_dir / service_name
    elif "figures" in service_name:
        logger.debug("{} is used".format(service_name))
        artifacts_dir = artifacts_dir / service_name
    elif service_name == "duckdb":
        logger.debug("duckdb is used")
        artifacts_dir = artifacts_dir / service_name
    elif service_name == "imputation":
        logger.debug("imputation")
        artifacts_dir = artifacts_dir / service_name
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    elif service_name == "outlier_detection":
        logger.debug("outlier_detection")
        artifacts_dir = artifacts_dir / service_name
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    elif service_name == "classification":
        logger.debug("imputation")
        artifacts_dir = artifacts_dir / service_name
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    elif service_name == "embeddings":
        artifacts_dir = artifacts_dir / service_name
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    elif service_name == "dataframes":
        artifacts_dir = artifacts_dir / service_name
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    elif service_name == "artifacts":
        artifacts_dir = artifacts_dir / service_name
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.debug("Unsorted Artifacts directory is used as no service specified")
        artifacts_dir = artifacts_dir / "Unsorted"

    if run_name is not None:
        logger.debug(
            "Run name ({}) is provided, creating a subdirectory".format(run_name)
        )
        artifacts_dir = artifacts_dir / run_name
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.debug(
            "No run name provided, artifacts will be saved to the root directory"
        )

    if subdir is not None:
        # More optional if there is an additional level of nesting that you need?
        logger.debug(
            "Subdirectory ({}) is provided, creating a subdirectory".format(subdir)
        )
        artifacts_dir = artifacts_dir / subdir
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    return artifacts_dir


def get_time_vector():
    """
    Generate a time vector for PLR signal analysis.

    Creates a linearly spaced time vector from 1 to 67 seconds with 1981 points,
    representing the standard PLR recording duration and sampling.

    Returns
    -------
    ndarray
        Array of 1981 time points from 1 to 67 seconds.
    """
    # start from 1 sec if some method struggles with 0? (TimeGPT?)
    time_vec = np.linspace(1, 67, 1981)
    return time_vec


def check_timegpt_token():
    """
    Verify that the TimeGPT API token is set in environment variables.

    Raises
    ------
    Exception
        If TIMEGPT_TOKEN environment variable is not set.

    Returns
    -------
    None
    """
    try:
        _ = os.environ["TIMEGPT_TOKEN"]
        logger.info("TimeGPT token found from the environment variables")
    except Exception as e:
        logger.error(
            "Please set TIMEGPT_TOKEN environment variable\n"
            "i.e. create .env file and store it there, or use Github Secrets for Github Actions"
        )
        raise e


def get_time_hash(n=10):
    """
    Generate a truncated SHA1 hash based on current time.

    Parameters
    ----------
    n : int, optional
        Number of characters to return from the hash, by default 10.

    Returns
    -------
    str
        First n characters of the SHA1 hash of the current timestamp.
    """
    hashlib.sha1().update(str(time.time()).encode("utf-8"))
    hash_out = hashlib.sha1().hexdigest()[:n]
    return hash_out


def pandas_concat(df1: pl.DataFrame, df2: pl.DataFrame, axis: int = 0):
    """
    Concatenate two Polars DataFrames using Pandas as intermediary.

    Workaround for Polars column length errors by converting to Pandas,
    performing concatenation, and converting back to Polars.

    Parameters
    ----------
    df1 : pl.DataFrame
        First Polars DataFrame.
    df2 : pl.DataFrame
        Second Polars DataFrame.
    axis : int, optional
        Axis along which to concatenate (0=rows, 1=columns), by default 0.

    Returns
    -------
    pl.DataFrame
        Concatenated Polars DataFrame.
    """
    # quickndirty to check for "The column lengths in the DataFrame are not equal." error
    # convert to Pandas, concatenate dataframes, and convert back to Polars
    df1_pd = df1.to_pandas()
    df2_pd = df2.to_pandas()
    df_out_pd = pd.concat([df1_pd, df2_pd], axis=axis)
    df_out = pl.from_pandas(df_out_pd)

    return df_out


def pandas_col_condition_filter(df, col_name, col_value):
    """
    Filter a Polars DataFrame by column value using Pandas as intermediary.

    Workaround for Polars filtering issues by converting to Pandas,
    performing the filter, and converting back to Polars.

    Parameters
    ----------
    df : pl.DataFrame
        Input Polars DataFrame to filter.
    col_name : str
        Name of the column to filter on.
    col_value : Any
        Value to match in the specified column.

    Returns
    -------
    pl.DataFrame
        Filtered Polars DataFrame containing only matching rows.
    """
    # "Pandas macro" as Polars went crazy?
    df_pd = df.to_pandas()
    df_out_pd = df_pd[df_pd[col_name] == col_value]
    df_out = pl.from_pandas(df_out_pd)
    return df_out


def check_for_device(cfg: DictConfig):
    """
    Check and configure the compute device (CPU/GPU) based on availability.

    Updates the configuration with the actual device to use and whether
    Automatic Mixed Precision (AMP) is enabled.

    Parameters
    ----------
    cfg : DictConfig
        Hydra/OmegaConf configuration containing DEVICE settings.

    Returns
    -------
    DictConfig
        Updated configuration with resolved device and AMP settings.
    """
    if cfg["DEVICE"]["device"] != "cpu":
        if torch.cuda.is_available():
            device = "cuda"  # torch.device("cuda")
            use_amp = cfg["DEVICE"]["use_amp"]
            logger.info("CUDA is available, using GPU")
            if use_amp:
                logger.info("Using Automatic Mixed Precision (AMP)")
        else:
            device = "cpu"  # torch.device("cpu")
            #
            use_amp = False
            logger.warning("CUDA is not available, using CPU")
    else:
        device = "cpu"  # torch.device("cpu")
        use_amp = False
        logger.info("Using CPU")

    with open_dict(cfg):
        cfg["DEVICE"]["device"] = device
        # https://pytorch.org/docs/stable/amp.html
        cfg["DEVICE"]["use_amp"] = use_amp

    return cfg
