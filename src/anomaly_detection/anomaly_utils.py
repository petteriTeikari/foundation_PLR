import os
from typing import Any, Dict, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import polars as pl
import torch
from loguru import logger
from omegaconf import DictConfig

from src.anomaly_detection.momentfm_outlier.moment_io import (
    compare_state_dicts,
    load_model_from_disk,
)
from src.classification.classifier_log_utils import (
    get_cls_metrics_fname,
)
from src.data_io.data_utils import (
    export_dataframe_to_duckdb,
    get_unique_polars_rows,
    load_from_duckdb_as_dataframe,
)
from src.data_io.data_wrangler import convert_df_to_dict
from src.ensemble.ensemble_logging import get_ensemble_pickle_name
from src.imputation.imputation_log_artifacts import get_imputation_pickle_name
from src.log_helpers.local_artifacts import load_results_dict
from src.log_helpers.log_naming_uris_and_dirs import (
    get_outlier_pickle_name,
    get_torch_model_name,
    update_outlier_detection_run_name,
)
from src.log_helpers.mlflow_artifacts import (
    check_if_run_exists,
    get_col_for_for_best_anomaly_detection_metric,
    get_duckdb_from_mlflow,
)
from src.log_helpers.mlflow_utils import init_mlflow_experiment


def pick_just_one_light_vector(light: pd.Series) -> Dict[str, np.ndarray]:
    """
    Extract a single light vector from the dataset.

    Since all subjects share the same light stimulus timing, we only need
    one representative light vector for analysis.

    Parameters
    ----------
    light : pd.Series
        Series containing light stimulus arrays for each color channel.
        Each value is a 2D array where the first dimension is subjects.

    Returns
    -------
    dict
        Dictionary with the same keys as input, but with 1D arrays
        (single subject's light vector for each channel).
    """
    light_out: Dict[str, np.ndarray] = {}
    for key, array in light.items():
        light_out[key] = array[0, :]

    return light_out


def get_data_for_sklearn_anomaly_models(
    df: pl.DataFrame, cfg: DictConfig, train_on: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Prepare data for sklearn-based anomaly detection models.

    Extracts and formats training and test data from a Polars DataFrame
    for use with traditional machine learning outlier detection methods.

    Parameters
    ----------
    df : pl.DataFrame
        Input PLR data containing pupil signals and labels.
    cfg : DictConfig
        Hydra configuration containing data processing parameters.
    train_on : str
        Column name specifying which pupil signal to use for training
        (e.g., 'pupil_orig', 'pupil_raw').

    Returns
    -------
    tuple
        A tuple containing:
        - X : np.ndarray
            Training data array of shape (n_subjects, n_timepoints).
        - y : np.ndarray
            Training labels (outlier mask) of same shape as X.
        - X_test : np.ndarray
            Test data array.
        - y_test : np.ndarray
            Test labels (outlier mask).
        - light : dict
            Light stimulus timing vectors for each color channel.
    """
    data_dict = convert_df_to_dict(data_df=df, cfg=cfg)
    X = data_dict["df"]["train"]["data"][train_on]
    y = data_dict["df"]["train"]["labels"]["outlier_mask"]
    assert X.shape == y.shape, f"X.shape: {X.shape}, y.shape: {y.shape}"

    X_test = data_dict["df"]["test"]["data"][train_on]
    y_test = data_dict["df"]["test"]["labels"]["outlier_mask"]
    assert X_test.shape == y_test.shape, (
        f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}"
    )

    # the timing should be the same for both train and test, and for all the subjects
    light = data_dict["df"]["train"]["light"]
    light = pick_just_one_light_vector(light)

    return X, y, X_test, y_test, light


def sort_anomaly_detection_runs_ensemble(
    mlflow_runs: pd.DataFrame, best_metric_cfg: DictConfig, sort_by: str, task: str
) -> pd.Series:
    """
    Sort MLflow runs for ensemble anomaly detection by specified metric.

    Parameters
    ----------
    mlflow_runs : pd.DataFrame
        DataFrame containing MLflow run information.
    best_metric_cfg : DictConfig
        Configuration specifying which metric to use and sort direction.
    sort_by : str
        Sorting strategy. Currently only 'best_metric' is supported.
    task : str
        Task name for metric column lookup (e.g., 'outlier_detection').

    Returns
    -------
    pd.Series
        The best run according to the specified sorting criteria.

    Raises
    ------
    ValueError
        If sort_by is not 'best_metric' or direction is unknown.
    """
    if sort_by == "best_metric":
        col_name = get_col_for_for_best_anomaly_detection_metric(best_metric_cfg, task)
        if best_metric_cfg["direction"] == "DESC":
            mlflow_runs = mlflow_runs.sort_values(by=col_name, ascending=True)
        elif best_metric_cfg["direction"] == "ASC":
            mlflow_runs = mlflow_runs.sort_values(by=col_name, ascending=True)
        else:
            logger.error(f"Unknown direction: {best_metric_cfg['direction']}")
            raise ValueError(f"Unknown direction: {best_metric_cfg['direction']}")
    else:
        logger.error(f"Unknown sort_by: {sort_by}")
        raise ValueError(f"Unknown sort_by: {sort_by}")

    # get the first run, as in the latest/best run
    return mlflow_runs.iloc[0]


def sort_anomaly_detection_runs(
    mlflow_runs: pd.DataFrame, best_string: str, sort_by: str
) -> pd.Series:
    """
    Sort MLflow anomaly detection runs by time or loss metric.

    Parameters
    ----------
    mlflow_runs : pd.DataFrame
        DataFrame containing MLflow run information.
    best_string : str
        Column name for the loss metric when sorting by 'best_loss'.
    sort_by : str
        Sorting strategy: 'start_time' for most recent, 'best_loss' for lowest loss.

    Returns
    -------
    pd.Series
        The best run according to the specified sorting criteria.

    Raises
    ------
    ValueError
        If sort_by is not 'start_time' or 'best_loss'.

    Notes
    -----
    To be combined eventually with newer sort_anomaly_detection_runs_ensemble().
    """
    # sort based on the start time
    if sort_by == "start_time":
        mlflow_runs = mlflow_runs.sort_values(by="start_time", ascending=False)
    elif sort_by == "best_loss":
        mlflow_runs = mlflow_runs.sort_values(by=best_string, ascending=True)
    else:
        logger.error(f"Unknown sort_by: {sort_by}")
        raise ValueError(f"Unknown sort_by: {sort_by}")

    # get the first run, as in the latest/best run
    return mlflow_runs.iloc[0]


def get_anomaly_detection_run(
    experiment_name: str,
    cfg: DictConfig,
    sort_by: str = "start_time",
    best_string: str = "best_loss",
    best_metric_cfg: Optional[DictConfig] = None,
) -> Optional[pd.Series]:
    """
    Retrieve a previous anomaly detection run from MLflow.

    Searches for existing runs matching the current configuration and returns
    the best one according to the specified sorting criteria.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment to search.
    cfg : DictConfig
        Hydra configuration for determining run name.
    sort_by : str, optional
        Sorting strategy: 'start_time' or 'best_loss'. Default is 'start_time'.
    best_string : str, optional
        Column name for loss metric. Default is 'best_loss'.
    best_metric_cfg : DictConfig, optional
        Configuration for ensemble metric sorting. Default is None.

    Returns
    -------
    pd.Series or None
        The matching MLflow run as a Series, or None if no matching run found.
    """
    mlflow_runs: pd.DataFrame = mlflow.search_runs(experiment_names=[experiment_name])
    run_name = update_outlier_detection_run_name(cfg)
    if len(mlflow_runs) > 0:
        mlflow_runs_model = mlflow_runs[mlflow_runs["tags.mlflow.runName"] == run_name]
        if len(mlflow_runs_model) > 0:
            logger.info(
                "You wanted to skip anomaly detection, and previous run was found -> skipping"
            )
            if best_metric_cfg is not None:
                mlflow_run = sort_anomaly_detection_runs_ensemble(
                    mlflow_runs_model,
                    sort_by=sort_by,
                    best_metric_cfg=best_metric_cfg,
                    task="outlier_detection",
                )
            else:
                mlflow_run = sort_anomaly_detection_runs(
                    mlflow_runs_model, sort_by=sort_by, best_string=best_string
                )
            logger.debug(f"Previous run: {mlflow_run}")
            return mlflow_run
        else:
            return None
    else:
        return None


def if_remote_anomaly_detection(
    try_to_recompute: bool,
    _anomaly_cfg: DictConfig,
    experiment_name: str,
    cfg: DictConfig,
) -> bool:
    """
    Determine whether to recompute anomaly detection or use cached results.

    Parameters
    ----------
    try_to_recompute : bool
        If True, always recompute regardless of cached results.
    _anomaly_cfg : DictConfig
        Anomaly detection configuration (currently unused).
    experiment_name : str
        MLflow experiment name to check for existing runs.
    cfg : DictConfig
        Full Hydra configuration.

    Returns
    -------
    bool
        True if anomaly detection should be (re)computed, False if cached
        results should be used.
    """
    if try_to_recompute:
        logger.info("Recomputing the anomaly detection (as you explicitly want it)")
        return True
    else:
        mlflow_run = get_anomaly_detection_run(experiment_name, cfg)
        if mlflow_run is not None:
            return False
        else:
            logger.info(
                "You wanted to skip anomaly detection, but no previous run was found -> computing"
            )
            return True


def save_outlier_detection_dataframe_to_mlflow(
    df: pl.DataFrame,
    experiment_name: str,
    _previous_experiment_name: str,
    cfg: DictConfig,
    copy_orig_db: bool = False,
) -> None:
    """
    Save outlier detection results as a DuckDB database to MLflow.

    Exports the dataframe to DuckDB format and logs it as an MLflow artifact
    for later retrieval and analysis.

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame containing outlier detection results.
    experiment_name : str
        MLflow experiment name for logging.
    _previous_experiment_name : str
        Name of the previous experiment (currently unused, for future reference).
    cfg : DictConfig
        Hydra configuration.
    copy_orig_db : bool, optional
        Whether to copy the original database. Default is False.

    Notes
    -----
    TODO: Not needed for outlier detection as is, but could be useful for
    saving results as DuckDB for easy inspection without re-running.
    """
    db_name = f"{experiment_name}_modelDummy.db"
    db_path = export_dataframe_to_duckdb(
        df=df, db_name=db_name, cfg=cfg, name="anomaly", copy_orig_db=copy_orig_db
    )
    init_mlflow_experiment(experiment_name=experiment_name)
    run_name = "Original Data"
    if not check_if_run_exists(experiment_name, run_name):
        # A bit of a temp solution, no necessarily need to exist here at all?
        with mlflow.start_run(run_name=run_name):
            try:
                mlflow.log_artifact(db_path, "data")
            except Exception as e:
                logger.error(f"Could not log the artifact: {e}")
                raise e


def get_best_run(experiment_name: str) -> pd.Series:
    """
    Get the best (first) run from an MLflow experiment.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment to search.

    Returns
    -------
    pd.Series
        The first run found in the experiment.

    Raises
    ------
    ValueError
        If no runs are found in the experiment.

    Notes
    -----
    Currently picks the first run found (often the only one). Add filters
    if you have multiple dataset versions or different filter requirements.
    """
    # Pick the first run found (often the only). Add some filters if you start
    # actually doing outlier detection, and if you have multiple dataset versions,
    # different filters to retrieve data, etc.
    best_runs: pd.Series = mlflow.search_runs(experiment_names=[experiment_name])
    if len(best_runs) == 0:
        logger.error(
            f"No (outlier detection / data import) runs found for experiment: {experiment_name}"
        )
        raise ValueError(f"No runs found for experiment: {experiment_name}")
    else:
        logger.info(f"Found {len(best_runs)} runs for experiment: {experiment_name}")
        logger.info(f"Best (data/outlier detection) run: {best_runs.loc[0, :]}")
        return best_runs.loc[0, :]


def log_anomaly_model_as_mlflow_artifact(checkpoint_file: str, run_name: str) -> None:
    """
    Log a trained anomaly detection model to MLflow as an artifact.

    Parameters
    ----------
    checkpoint_file : str
        Path to the model checkpoint file.
    run_name : str
        Name of the MLflow run (for logging purposes).

    Raises
    ------
    Exception
        If the artifact cannot be logged to MLflow.

    Notes
    -----
    This can be slow for large models (e.g., 1.3GB).
    """
    logger.info("Logging Anomaly Detection model as an artifact to MLflow")
    # Note! this can be a bit slow if you need to upload 1.3G models
    try:
        mlflow.log_artifact(local_path=checkpoint_file, artifact_path="model")
    except Exception as e:
        logger.error(f"Could not log the artifact: {e}")
        raise e


def print_available_artifacts(path: str) -> None:
    """
    Print available artifacts at the given path.

    Parameters
    ----------
    path : str
        Full path to an artifact file.

    Notes
    -----
    Currently a stub function that extracts directory and filename.
    """
    dir, fname = os.path.split(path)


def get_artifact(
    run_id: str, run_name: str, model_name: str, subdir: str = "outlier_detection"
) -> Optional[str]:
    """
    Download an artifact from MLflow by run ID and subdirectory.

    Parameters
    ----------
    run_id : str
        MLflow run ID.
    run_name : str
        Name of the MLflow run.
    model_name : str
        Name of the model (used for filename generation).
    subdir : str, optional
        Artifact subdirectory: 'outlier_detection', 'model', 'imputation',
        'baseline_model', or 'metrics'. Default is 'outlier_detection'.

    Returns
    -------
    str or None
        Local path to the downloaded artifact, or None for baseline_model
        with ensembled input.

    Raises
    ------
    ValueError
        If subdir is unknown.
    Exception
        If artifact cannot be downloaded.
    """
    try:
        if subdir == "outlier_detection":
            fname = get_outlier_pickle_name(model_name)
            # HACK fir this, the pickle has been named after run_name and not model_name
            if fname == "outlierDetection_UniTS.pickle":
                logger.warning('Hacky fix for "outlierDetection_UniTS.pickle"')
                fname = get_outlier_pickle_name(run_name)
            if "ensemble" in model_name:
                fname = get_ensemble_pickle_name(ensemble_name=run_name)
        elif subdir == "model":
            fname = get_torch_model_name(run_name)
        elif subdir == "imputation":
            fname = get_imputation_pickle_name(model_name)
            if "ensemble" in model_name:
                # e.g. ensemble_ensemble-CSDI-SAITS-TimesNet__exclude_gt_results.pickle
                fname = get_ensemble_pickle_name(ensemble_name=run_name)
        elif subdir == "baseline_model":
            # if you want the Classifier object
            # fname = get_model_fname(run_name, prefix="baseline")
            fname = get_cls_metrics_fname(run_name, prefix="baseline")
        elif subdir == "metrics":
            # if you want the Classifier object
            # fname = get_model_fname(run_name, prefix="baseline")
            fname = get_cls_metrics_fname(run_name)
        else:
            logger.error(f"Unknown subdir: {subdir}")
            raise ValueError(f"Unknown subdir: {subdir}")

        path = f"runs:/{run_id}/{subdir}/{fname}"

        if subdir == "baseline_model" and "ensembled_input" in run_name:
            logger.warning(
                "No baseline available for the ensembled diverse classifiers"
            )
            return None
        else:
            try:
                artifact = mlflow.artifacts.download_artifacts(artifact_uri=path)
            except Exception as e:
                logger.error(f"Could not download the artifact: {e}")
                logger.error(
                    f"run_id: {run_id}, model_name: {model_name}, path: {path}"
                )
                raise e
            return artifact
    except Exception as e:
        logger.error(f"Problem getting the artifact: {e}")
        logger.error(f"run_id: {run_id}, model_name: {model_name}, path: {path}")
        raise e


def check_outlier_detection_artifact(outlier_artifacts: Dict[str, Any]) -> None:
    """
    Validate the structure of outlier detection artifacts.

    Parameters
    ----------
    outlier_artifacts : dict
        Dictionary containing outlier detection results with 'outlier_results' key.

    Raises
    ------
    AssertionError
        If the artifact structure is invalid.
    """
    first_epoch_key = list(outlier_artifacts["outlier_results"].keys())[0]
    outlier_results = outlier_artifacts["outlier_results"][first_epoch_key]
    check_outlier_results(outlier_results=outlier_results)


def check_outlier_results(outlier_results: Dict[str, Any]) -> None:
    """
    Validate the structure of outlier results dictionary.

    Parameters
    ----------
    outlier_results : dict
        Dictionary containing per-split outlier detection results.

    Raises
    ------
    AssertionError
        If the results structure is invalid.
    """
    first_split = list(outlier_results.keys())[0]
    split_results = outlier_results[first_split]["results_dict"]["split_results"]
    check_split_results(split_results)


def check_split_results(split_results: Dict[str, Any]) -> None:
    """
    Validate consistency between flat and array representations.

    Ensures that the number of samples in flattened arrays matches the
    total size of the original arrays.

    Parameters
    ----------
    split_results : dict
        Dictionary containing 'arrays_flat' and 'arrays' with prediction results.

    Raises
    ------
    AssertionError
        If sample counts do not match between representations.
    """
    no_samples_in_flat = split_results["arrays_flat"]["trues_valid"].shape[0]
    no_samples_in_array = split_results["arrays"]["trues"].size
    # A bit bizarre issue with samples being dropped somewhere?
    assert no_samples_in_flat == no_samples_in_array, (
        f"no_samples_in_flat: {no_samples_in_flat}, no_samples_in_array: {no_samples_in_array}, should be equal"
    )


def get_no_subjects_in_outlier_artifacts(outlier_artifacts: Dict[str, Any]) -> int:
    """
    Get the number of subjects from outlier detection artifacts.

    Parameters
    ----------
    outlier_artifacts : dict
        Dictionary containing outlier detection results.

    Returns
    -------
    int
        Number of subjects in the training split.
    """
    first_epoch_key = list(outlier_artifacts["outlier_results"].keys())[0]
    split_results = outlier_artifacts["outlier_results"][first_epoch_key]["train"][
        "results_dict"
    ]["split_results"]
    no_subjects = split_results["arrays"]["trues"].shape[0]
    return no_subjects


def outlier_detection_artifacts_dict(
    mlflow_run: pd.Series, model_name: str, task: str
) -> Dict[str, Any]:
    """
    Load outlier detection artifacts from an MLflow run.

    Parameters
    ----------
    mlflow_run : pd.Series
        MLflow run information containing run_id and run name.
    model_name : str
        Name of the outlier detection model.
    task : str
        Task subdirectory for artifacts.

    Returns
    -------
    dict
        Loaded outlier detection artifacts dictionary.

    Warnings
    --------
    Logs a warning if artifact file size exceeds 2GB.
    """
    run_id = mlflow_run["run_id"]
    run_name = mlflow_run["tags.mlflow.runName"]

    if model_name == "TimesNet-orig":
        model_name = "TimesNet"

    outlier_artifacts_path = get_artifact(
        run_id=run_id,
        run_name=run_name,
        model_name=model_name,
        subdir=task,
    )
    file_size_MB = os.path.getsize(outlier_artifacts_path) / (1024 * 1024)
    logger.debug("Artifact file size = {:.2f} MB".format(file_size_MB))
    if file_size_MB > 2048:
        # Obviously tune this threshold if you start using massive models
        logger.warning(
            f"File size is over 2GB ({file_size_MB / 1024:.2f} GB), is this correct? "
            f"something went wrong in the previous step?"
        )
        logger.warning(f"Artifact path: {outlier_artifacts_path}")
    outlier_artifacts = load_results_dict(outlier_artifacts_path)

    return outlier_artifacts


def get_moment_model_from_mlflow_artifacts(
    run_id: str,
    run_name: str,
    model: torch.nn.Module,
    device: str,
    cfg: DictConfig,
    task: str,
    model_name: str = "MOMENT",
) -> torch.nn.Module:
    """
    Load a MOMENT model from MLflow artifacts.

    Downloads the model checkpoint and loads it into the provided model object,
    verifying that the weights have changed from the initial state.

    Parameters
    ----------
    run_id : str
        MLflow run ID.
    run_name : str
        Name of the MLflow run.
    model : torch.nn.Module
        Model object to load weights into.
    device : str
        Device to load the model onto ('cpu' or 'cuda').
    cfg : DictConfig
        Hydra configuration.
    task : str
        Task name (e.g., 'outlier_detection', 'imputation').
    model_name : str, optional
        Name of the model. Default is 'MOMENT'.

    Returns
    -------
    torch.nn.Module
        Model with loaded weights.

    Raises
    ------
    AssertionError
        If loaded weights are identical to pretrained weights.
    """
    model_path = get_artifact(run_id, run_name, model_name, subdir="model")
    file_stats = os.stat(model_path)
    logger.info(
        f"Model size: {file_stats.st_size / (1024 * 1024):.0f} MB | {model_path}"
    )
    state_dict_in = model.state_dict().__str__()
    model = load_model_from_disk(model, model_path, device, cfg, task)
    state_dict_out = model.state_dict().__str__()
    compare_state_dicts(state_dict_out, state_dict_in, same_ok=False)
    return model


def get_anomaly_detection_results_from_mlflow(
    experiment_name: str,
    cfg: DictConfig,
    run_name: str,
    model_name: str,
    get_model: bool = False,
) -> Tuple[Dict[str, Any], Optional[torch.nn.Module]]:
    """
    Retrieve anomaly detection results and optionally the model from MLflow.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name.
    cfg : DictConfig
        Hydra configuration.
    run_name : str
        Name of the MLflow run.
    model_name : str
        Name of the outlier detection model.
    get_model : bool, optional
        Whether to also load the trained model. Default is False.

    Returns
    -------
    tuple
        A tuple containing:
        - outlier_artifacts : dict
            Loaded outlier detection artifacts.
        - model : torch.nn.Module or None
            The trained model if get_model=True, otherwise None.

    Raises
    ------
    ValueError
        If no matching run is found.
    NotImplementedError
        If get_model=True for finetuned models (not yet implemented).
    """
    sort_by = "best_loss"
    mlflow_run = get_anomaly_detection_run(
        experiment_name,
        cfg,
        sort_by=sort_by,
        best_string=cfg["OUTLIER_DETECTION"][sort_by],
    )

    if len(mlflow_run) == 0:
        logger.error(f"No run found for experiment: {experiment_name}")
        raise ValueError(f"No run found for experiment: {experiment_name}")

    run_id = mlflow_run["run_id"]
    run_name = mlflow_run["tags.mlflow.runName"]
    outlier_artifacts_path = get_artifact(
        run_id, run_name, model_name, subdir="outlier_detection"
    )
    outlier_artifacts = load_results_dict(outlier_artifacts_path)

    if "finetune" in run_name:
        # Now every different model have their different way to load them (possibly)
        if get_model:
            # You don't need the model until you are doing imputation (reconstruction) with these models?
            # model_path = get_artifact(run_id, model_name, subdir="model")
            # load_model_from_disk(model, model_path, device, cfg)
            raise NotImplementedError("Loading the model is not implemented yet")
        else:
            model = None
    else:
        # e.g. when you did zero-shot, no model was saved
        model = None

    return outlier_artifacts, model


# @task(
#     log_prints=True,
#     name="Import Anomaly Detection from MLflow",
#     description="Placeholder atm, as anomaly detection has been done manually",
# )
def get_source_dataframe_from_mlflow(
    experiment_name: str, cfg: DictConfig
) -> pl.DataFrame:
    """
    Import anomaly detection results from MLflow as a Polars DataFrame.

    Downloads the DuckDB artifact from MLflow and loads it as a DataFrame.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name to retrieve data from.
    cfg : DictConfig
        Hydra configuration for data loading.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame containing the outlier-detected data with
        train and test splits.

    Raises
    ------
    AssertionError
        If the DataFrame does not contain exactly 2 splits (train and val).

    Notes
    -----
    Placeholder task for importing anomaly detection results from MLflow.
    """
    # Get the best run
    best_run = get_best_run(experiment_name=experiment_name)

    # get the artifact path
    artifact_uri: str = mlflow.get_run(best_run["run_id"]).info.artifact_uri

    # Get the db path (refers to downloaded path)
    db_path = get_duckdb_from_mlflow(artifact_uri=artifact_uri)

    # Load the Polars dataframe from the DuckDB
    df = load_from_duckdb_as_dataframe(db_path=db_path, cfg=cfg)
    # df = load_both_splits_from_duckdb(db_path, cfg=cfg)
    logger.info(
        "Download Outlier Detected dataframe from MLflow, pl.DataFrame shape = {}".format(
            df.shape
        )
    )
    logger.info(f"db_path = {db_path}")
    splits = list(get_unique_polars_rows(df, unique_col="split")["split"])
    assert len(splits) == 2, "You should now have train and val, but you had {}".format(
        splits
    )

    return df
