import hashlib

import torch
from loguru import logger
import os
import time
import polars as pl
import pandas as pd
import numpy as np
from omegaconf import DictConfig, open_dict


def get_repo_root(base_name: str = "repo_PLR"):
    cwd = os.getcwd()
    if os.path.basename(cwd) == "src":
        repo_root = os.path.dirname(cwd)
    elif os.path.basename(cwd) != base_name:
        # subfolder in "src", make recursive later
        init = os.path.dirname(cwd)
        repo_root = os.path.dirname(init)
    else:
        repo_root = cwd
    return repo_root


def get_data_dir(data_path: str = "data"):
    data_dir = os.path.join(get_repo_root(), data_path)
    return data_dir


def get_artifacts_dir(
    service_name: str = "mlflow", run_name: str = None, subdir: str = None
):
    artifacts_dir = os.path.join(get_repo_root(), "artifacts")
    # TODO! add "tmp" to most of the dirs as they are meant to be mostly temporary
    #  and permanent artifacts are stored in MLflow (or Prefect)
    if service_name == "mlflow":
        logger.debug("MLflow Artifacts directory is used")
        artifacts_dir = os.path.join(artifacts_dir, service_name)
    elif service_name == "hydra":
        logger.debug("Hydra Artifacts directory is used")
        artifacts_dir = os.path.join(artifacts_dir, service_name)
    elif service_name == "best_models":
        # "Best" Pickles, DuckDBs, etc. go here atm
        logger.debug("Best models directory is used")
        artifacts_dir = os.path.join(artifacts_dir)
    elif service_name == "features":
        # "Best" Pickles, DuckDBs, etc. go here atm
        logger.debug("Features directory is used")
        artifacts_dir = os.path.join(artifacts_dir, service_name)
    elif service_name == "pypots":
        logger.debug("PyPOTS directory is used")
        artifacts_dir = os.path.join(artifacts_dir, service_name)
    elif "figures" in service_name:
        logger.debug("{} is used".format(service_name))
        artifacts_dir = os.path.join(artifacts_dir, service_name)
    elif service_name == "duckdb":
        logger.debug("duckdb is used")
        artifacts_dir = os.path.join(artifacts_dir, service_name)
    elif service_name == "imputation":
        logger.debug("imputation")
        artifacts_dir = os.path.join(artifacts_dir, service_name)
        os.makedirs(artifacts_dir, exist_ok=True)
    elif service_name == "outlier_detection":
        logger.debug("outlier_detection")
        artifacts_dir = os.path.join(artifacts_dir, service_name)
        os.makedirs(artifacts_dir, exist_ok=True)
    elif service_name == "classification":
        logger.debug("imputation")
        artifacts_dir = os.path.join(artifacts_dir, service_name)
        os.makedirs(artifacts_dir, exist_ok=True)
    elif service_name == "embeddings":
        artifacts_dir = os.path.join(artifacts_dir, service_name)
        os.makedirs(artifacts_dir, exist_ok=True)
    elif service_name == "dataframes":
        artifacts_dir = os.path.join(artifacts_dir, service_name)
        os.makedirs(artifacts_dir, exist_ok=True)
    elif service_name == "artifacts":
        artifacts_dir = os.path.join(artifacts_dir, service_name)
        os.makedirs(artifacts_dir, exist_ok=True)
    else:
        logger.debug("Unsorted Artifacts directory is used as no service specified")
        artifacts_dir = os.path.join(artifacts_dir, "Unsorted")

    if run_name is not None:
        logger.debug(
            "Run name ({}) is provided, creating a subdirectory".format(run_name)
        )
        artifacts_dir = os.path.join(artifacts_dir, run_name)
        os.makedirs(artifacts_dir, exist_ok=True)
    else:
        logger.debug(
            "No run name provided, artifacts will be saved to the root directory"
        )

    if subdir is not None:
        # More optional if there is an additional level of nesting that you need?
        logger.debug(
            "Subdirectory ({}) is provided, creating a subdirectory".format(subdir)
        )
        artifacts_dir = os.path.join(artifacts_dir, subdir)
        os.makedirs(artifacts_dir, exist_ok=True)

    return artifacts_dir


def get_time_vector():
    # start from 1 sec if some method struggles with 0? (TimeGPT?)
    time_vec = np.linspace(1, 67, 1981)
    return time_vec


def check_timegpt_token():
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
    hashlib.sha1().update(str(time.time()).encode("utf-8"))
    hash_out = hashlib.sha1().hexdigest()[:n]
    return hash_out


def pandas_concat(df1: pl.DataFrame, df2: pl.DataFrame, axis: int = 0):
    # quickndirty to check for "The column lengths in the DataFrame are not equal." error
    # convert to Pandas, concatenate dataframes, and convert back to Polars
    df1_pd = df1.to_pandas()
    df2_pd = df2.to_pandas()
    df_out_pd = pd.concat([df1_pd, df2_pd], axis=axis)
    df_out = pl.from_pandas(df_out_pd)

    return df_out


def pandas_col_condition_filter(df, col_name, col_value):
    # "Pandas macro" as Polars went crazy?
    df_pd = df.to_pandas()
    df_out_pd = df_pd[df_pd[col_name] == col_value]
    df_out = pl.from_pandas(df_out_pd)
    return df_out


def check_for_device(cfg: DictConfig):
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
