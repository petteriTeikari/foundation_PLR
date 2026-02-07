from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from mlflow.entities import FileInfo
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig

from src.ensemble.ensemble_logging import get_ensemble_pickle_name
from src.log_helpers.local_artifacts import load_results_dict
from src.log_helpers.log_naming_uris_and_dirs import get_imputation_pickle_name
from src.utils import get_artifacts_dir


def get_mlflow_run_ids_from_imputation_artifacts(
    imputation_artifacts: Dict[str, Any],
) -> Dict[str, str]:
    """Extract MLflow run IDs from imputation artifacts dictionary.

    Parameters
    ----------
    imputation_artifacts : dict
        Dictionary containing 'artifacts' key with model-specific MLflow info.

    Returns
    -------
    dict
        Mapping of model names to their MLflow run IDs.
    """
    run_ids: Dict[str, str] = {}
    for model_name in imputation_artifacts["artifacts"].keys():
        mlflow_info = imputation_artifacts["artifacts"][model_name]["mlflow"]
        run_ids[model_name] = mlflow_info["run_info"]["run_id"]
    return run_ids


def get_mlflow_metric_params(
    metrics: Dict[str, Any],
    cfg: DictConfig,
    splitkey: str = "gt",
    metrictype: str = "global",
    metricname: str = "mae",
) -> Dict[str, Any]:
    """Extract specific metric parameters from nested metrics dictionary for MLflow logging.

    Filters metrics by split key, metric type, and metric name to keep the
    MLflow dashboard clean while still allowing programmatic access to all metrics.

    Parameters
    ----------
    metrics : dict
        Nested metrics dictionary with structure:
        {model_name: {split: {split_key: {metric_type: {metric: value}}}}}.
    cfg : DictConfig
        Configuration object (currently unused).
    splitkey : str, default "gt"
        Split key to filter (e.g., 'gt' for ground truth).
    metrictype : str, default "global"
        Metric type to filter (e.g., 'global', 'per_subject').
    metricname : str, default "mae"
        Specific metric name to extract.

    Returns
    -------
    dict
        Dictionary with model name and filtered metrics suitable for MLflow logging.

    Raises
    ------
    ValueError
        If more than one model is found in the metrics dictionary.
    """
    # You could obviously just get all, but taking the main metric to keep the Dashboard clean
    # you can always get all the metrics programatically from the MLflow API
    for i, model_name in enumerate(metrics.keys()):
        if i > 0:
            logger.error(
                "More than one model found, this should not happen now, as all the subflows should"
                "operate independently, and you should only have one model in the metrics dict"
            )
            raise ValueError("Too many models in the metrics dictionary")
        metric_params = {"model": model_name}
        for split in metrics[model_name].keys():
            for split_key in metrics[model_name][split].keys():
                for metric_type in metrics[model_name][split][split_key].keys():
                    for metric in metrics[model_name][split][split_key][
                        metric_type
                    ].keys():
                        if (
                            split_key == splitkey
                            and metric_type == metrictype
                            and metric == metricname
                        ):
                            key_out = f"imp_{split}/{metric}"
                            value_in = metrics[model_name][split][split_key][
                                metric_type
                            ][metric]
                            metric_params[key_out] = value_in

    return metric_params


def get_mlflow_params(mlflow_info: Dict[str, Any]) -> Tuple[str, str]:
    """Extract and set MLflow experiment and run ID from info dictionary.

    Parameters
    ----------
    mlflow_info : dict
        Dictionary containing 'experiment' and 'run_info' keys with MLflow metadata.

    Returns
    -------
    tuple of str
        Tuple of (experiment_id, run_id).

    Notes
    -----
    Also sets the MLflow experiment as a side effect.
    """
    # Get the MLflow experiment and run ID that was used during the training
    experiment_id = mlflow_info["experiment"]["name"]
    run_id = mlflow_info["run_info"]["run_id"]
    mlflow.set_experiment(experiment_id)
    return experiment_id, run_id


def get_mlflow_info_from_model_dict(model_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract MLflow info dictionary from model artifacts dictionary.

    Parameters
    ----------
    model_dict : dict
        Model artifacts dictionary containing 'mlflow' key with run/experiment info.

    Returns
    -------
    dict
        MLflow info dictionary with run_info, experiment, and artifact_uri.

    Raises
    ------
    Exception
        If 'mlflow' key is missing from model_dict.
    """
    # If everything went ok, you should have the MLflow run/experiment/artifact_uri/etc. info saved here
    try:
        mlflow_info = model_dict["mlflow"]
    except Exception as e:
        logger.error(f"Failed to get the MLflow info: {e}")
        logger.error(
            "How come did this happen, and you never saved the 'mlflow' key in the model_dict?"
        )
        raise e

    return mlflow_info


def get_duckdb_from_mlflow(
    artifact_uri: str, dir_name: str = "data", wildcard: str = ".db"
) -> str:
    """Download and locate DuckDB file from MLflow artifacts.

    Parameters
    ----------
    artifact_uri : str
        MLflow artifact URI to search.
    dir_name : str, default "data"
        Directory name within artifacts containing the database.
    wildcard : str, default ".db"
        File extension to match.

    Returns
    -------
    str
        Local path to downloaded DuckDB file.

    Raises
    ------
    FileNotFoundError
        If no DuckDB artifact is found.
    """
    db_path = None
    artifacts = mlflow.artifacts.list_artifacts(artifact_uri=artifact_uri)
    if len(artifacts) == 0:
        logger.error(
            "No DuckDB artifact found from the MLflow run, artifact_uri = {}".format(
                artifact_uri
            )
        )
        raise FileNotFoundError(
            "No DuckDB artifact found from the MLflow run, artifact_uri = {}".format(
                artifact_uri
            )
        )

    for artifact in artifacts:
        if dir_name in artifact.path:
            folder = mlflow.artifacts.download_artifacts(
                artifact_uri=f"{artifact_uri}/{dir_name}"
            )
            for root, dirs, files in Path(folder).walk():
                for file in files:
                    if wildcard in file:
                        db_path = str(root / file)
    if db_path is None:
        logger.error("Could not find the DuckDB file from the MLflow artifacts")
        raise FileNotFoundError(
            "Could not find the DuckDB file from the MLflow artifacts"
        )
    return db_path


def write_new_col_to_mlflow(
    model_best_runs: pd.DataFrame, col_name: str, col_name_init: str
) -> None:
    """Write a new metric column to MLflow runs.

    Used for harmonizing column names by writing values under a new metric name.

    Parameters
    ----------
    model_best_runs : pd.DataFrame
        DataFrame containing run_id and the column to write.
    col_name : str
        Source column name in the DataFrame.
    col_name_init : str
        Target metric name for MLflow (will have 'metrics.' prefix stripped).
    """
    no_of_runs = model_best_runs.shape[0]
    for i in range(no_of_runs):
        run_id = model_best_runs.iloc[i]["run_id"]
        with mlflow.start_run(run_id=run_id):
            value = model_best_runs.iloc[i][col_name]
            col_name_out = col_name_init.replace("metrics.", "")
            logger.info(f"Writing the new column {col_name_out} with value {value}")
            mlflow.log_metric(col_name_out, value)
            mlflow.end_run()


def get_col_for_for_best_anomaly_detection_metric(
    best_metric_cfg: DictConfig, task: str
) -> str:
    """Get DataFrame column name for best metric based on task type.

    Parameters
    ----------
    best_metric_cfg : DictConfig
        Configuration with 'string' (metric name) and 'split' keys.
    task : str
        Task type: 'anomaly_detection', 'outlier_detection', or 'imputation'.

    Returns
    -------
    str
        Column name in format 'metrics.{split}/{metric}' or direct string.

    Raises
    ------
    ValueError
        If task type is not recognized.
    """
    if task == "anomaly_detection" or task == "outlier_detection":
        # use only one name eventually
        best_metric_name = best_metric_cfg["string"]
        split = best_metric_cfg["split"]
        col_name = f"metrics.{split}/{best_metric_name}"
    elif task == "imputation":  # or task == "outlier_detection":
        # TODO! This is a bit hacky, but the best metric is always the same for imputation
        #  as not this is directly the col_name of anomaly detection
        col_name = best_metric_cfg["string"]
    else:
        logger.error("Unknon task = {}".format(task))
        raise ValueError("Unknon task = {}".format(task))
    return col_name


def harmonize_anomaly_col_name(
    col_name: str,
    model_best_runs: pd.DataFrame,
    best_metric_cfg: DictConfig,
    model: str,
) -> str:
    """Harmonize metric column name if not found in DataFrame.

    Falls back to 'test' split if the specified column is missing, and writes
    the harmonized values back to MLflow.

    Parameters
    ----------
    col_name : str
        Expected column name.
    model_best_runs : pd.DataFrame
        DataFrame with MLflow run data.
    best_metric_cfg : DictConfig
        Best metric configuration.
    model : str
        Model name for logging.

    Returns
    -------
    str
        Harmonized column name that exists in the DataFrame.

    Raises
    ------
    ValueError
        If harmonized column contains only NaN values.
    """
    if col_name not in model_best_runs.columns:
        col_name_init = col_name
        col_name = f"metrics.test/{best_metric_cfg['string']}"
        # best_series = model_best_runs.iloc[0]
        best_values = model_best_runs[col_name].to_numpy()
        if np.all(np.isnan(best_values)):
            logger.error(
                f"Could not find the column {col_name} in the model_best_runs dataframe"
            )
            raise ValueError(
                f"Could not find the column {col_name} in the model_best_runs dataframe"
            )
        else:
            # harmonize the column name and write this with the new column name
            logger.info("Harmonizing the column name to test")
            write_new_col_to_mlflow(model_best_runs, col_name, col_name_init)

    return col_name


def threshold_filter_run(
    best_run: Union[pd.Series, pd.DataFrame], col_name: str, best_metric_cfg: DictConfig
) -> Optional[Union[pd.Series, pd.DataFrame]]:
    """Filter run based on ensemble quality threshold.

    Returns None if the run's metric does not meet the threshold requirement.

    Parameters
    ----------
    best_run : pd.Series or pd.DataFrame
        Run data to filter.
    col_name : str
        Column name containing the metric to check.
    best_metric_cfg : DictConfig
        Configuration with 'ensemble_quality_threshold' and 'direction' keys.

    Returns
    -------
    pd.Series, pd.DataFrame, or None
        Original run data if threshold is met, None otherwise.
    """
    input_was_df = False
    if isinstance(best_run, pd.DataFrame):
        input_was_df = True
        best_run = pd.Series(best_run.iloc[0])

    if best_metric_cfg["ensemble_quality_threshold"] is not None:
        if best_metric_cfg["direction"] == "ASC":
            if best_run[col_name] > best_metric_cfg["ensemble_quality_threshold"]:
                # logger.warning(
                #     f"Model did not reach the ensemble quality threshold of "
                #     f"{best_metric_cfg['ensemble_quality_threshold']}"
                # )
                return None
        elif best_metric_cfg["direction"] == "DESC":
            if best_run[col_name] < best_metric_cfg["ensemble_quality_threshold"]:
                # logger.warning(
                #     f"Model did not reach the ensemble quality threshold of "
                #     f"{best_metric_cfg['ensemble_quality_threshold']}"
                # )
                return None
        else:
            logger.error("The direction of the best metric is not recognized")
            raise ValueError("The direction of the best metric is not recognized")

    if input_was_df:
        best_run = pd.DataFrame(best_run).T

    return best_run


def get_best_run_of_pd_dataframe(
    model_best_runs: pd.DataFrame,
    cfg: DictConfig,
    best_metric_cfg: DictConfig,
    task: str,
    model: str,
    include_all_variants: bool = False,
) -> Tuple[Optional[Union[pd.Series, pd.DataFrame]], Optional[float]]:
    """Find the best MLflow run from a DataFrame based on metric configuration.

    Parameters
    ----------
    model_best_runs : pd.DataFrame
        DataFrame containing MLflow runs for the model.
    cfg : DictConfig
        Full configuration object.
    best_metric_cfg : DictConfig
        Configuration specifying best metric, direction, and threshold.
    task : str
        Task type for determining column name format.
    model : str
        Model name for logging.
    include_all_variants : bool, default False
        If True, return all runs sorted; if False, return only the best run.

    Returns
    -------
    tuple
        Tuple of (best_run, best_metric) where best_run is a Series/DataFrame
        and best_metric is the metric value (or None if all variants returned).
    """
    col_name = get_col_for_for_best_anomaly_detection_metric(best_metric_cfg, task)
    col_name = harmonize_anomaly_col_name(
        col_name, model_best_runs, best_metric_cfg, model
    )

    try:
        if best_metric_cfg["direction"] == "ASC":
            sorted_runs = model_best_runs.sort_values(by=col_name, ascending=True)
        elif best_metric_cfg["direction"] == "DESC":
            sorted_runs = model_best_runs.sort_values(by=col_name, ascending=False)
        else:
            logger.error("The direction of the best metric is not recognized")
            raise ValueError("The direction of the best metric is not recognized")
    except Exception as e:
        logger.error(f"Failed to sort the runs based on the best metric: {e}")
        raise e

    if include_all_variants:
        # when you just want to recompute the metrics
        best_run = sorted_runs
        best_metric = None
    else:
        best_run = sorted_runs.iloc[0]
        logger.info(
            f"{model}: The best {best_metric_cfg['string']} is {best_run[col_name]:.3f}"
        )
        best_run = threshold_filter_run(best_run, col_name, best_metric_cfg)
        if best_run is not None:
            best_metric = best_run[col_name]
        else:
            best_metric = None

    return best_run, best_metric


def get_imputation_results_from_mlflow(
    mlflow_run: pd.Series,
    model_name: str,
    cfg: DictConfig,
    dir_name: str = "imputation",
) -> Dict[str, Any]:
    """Download imputation results from MLflow artifact store.

    Parameters
    ----------
    mlflow_run : pd.Series
        MLflow run data containing run_id and tags.
    model_name : str
        Name of the imputation model.
    cfg : DictConfig
        Configuration object (currently unused).
    dir_name : str, default "imputation"
        Artifact subdirectory name.

    Returns
    -------
    dict
        Loaded imputation results dictionary with 'mlflow_run' key added.

    Raises
    ------
    FileNotFoundError
        If imputation results cannot be found or downloaded.
    """
    if "ensemble" in mlflow_run["tags.mlflow.runName"]:
        fname = get_ensemble_pickle_name(ensemble_name=model_name)
        logger.debug(f"Ensemble model found, loading the ensemble pickle: {fname}")
    else:
        fname = get_imputation_pickle_name(model_name)

    artifact_uri = "runs:/{}/{}/{}".format(mlflow_run["run_id"], dir_name, fname)
    try:
        path_dir = mlflow.artifacts.download_artifacts(artifact_uri)
    except Exception as e:
        logger.error(f"Could not download the imputation results from MLflow: {e}")
        logger.info("mlflow_run: {}".format(mlflow_run))
        raise e

    if path_dir is not None:
        logger.info(
            f"Imputation results downloaded from MLflow, artifact_uri = {artifact_uri}"
        )
        dict_out = load_results_dict(path_dir)
    else:
        logger.error(
            f"Could not find imputation results for model = {model_name}, artifact_uri: {artifact_uri}"
        )
        raise FileNotFoundError(
            f"Could not find imputation results for model = {model_name}, artifact_uri: {artifact_uri}"
        )

    # Add the artifact_uri to the dictionary
    dict_out["mlflow_run"] = mlflow_run

    return dict_out


def get_mlflow_artifact_uri_from_run(best_run: Union[Dict[str, Any], pd.Series]) -> str:
    """Get artifact URI from MLflow run.

    Parameters
    ----------
    best_run : dict or pd.Series
        Run data containing 'run_id'.

    Returns
    -------
    str
        Artifact URI for the run.
    """
    artifact_uri: str = mlflow.get_run(best_run["run_id"]).info.artifact_uri
    return artifact_uri


def get_best_metric_from_current_run(
    metrics_model: dict, split_key: str, metric_string: str
) -> float:
    """Extract specific metric value from current run's metrics dictionary.

    Parameters
    ----------
    metrics_model : dict
        Metrics dictionary with structure {split_key: {global: {metric: value}}}.
    split_key : str
        Data split key (e.g., 'test', 'val').
    metric_string : str
        Name of the metric to extract.

    Returns
    -------
    float
        The metric value.
    """
    logger.info(
        f"Getting the best metric from the current run, metric = {metric_string}, "
        f"split = {split_key}"
    )
    return metrics_model[split_key]["global"][metric_string]


def get_best_previous_mlflow_logged_model(
    model_dict: Dict[str, Any], cfg: DictConfig
) -> Optional[Dict[str, Any]]:
    """Find the best previously logged MLflow model matching current configuration.

    Parameters
    ----------
    model_dict : dict
        Model artifacts dictionary containing MLflow info.
    cfg : DictConfig
        Configuration for determining search parameters.

    Returns
    -------
    dict
        Best previous run data, or None if no matching runs found.
    """
    mlflow_info = get_mlflow_info_from_model_dict(model_dict)
    experiment_id, run_id = get_mlflow_params(mlflow_info)
    current_experiment, metric_string, split_key, metric_direction = (
        what_to_search_from_mlflow(
            run_name=mlflow_info["run_info"]["run_name"], cfg=cfg
        )
    )

    best_previous_run = return_best_mlflow_run(
        current_experiment,
        metric_string,
        split_key,
        metric_direction,
        run_name=mlflow_info["run_info"]["run_name"],
    )

    return best_previous_run


def iterate_through_mlflow_run_artifacts(
    run_artifacts: List[FileInfo],
    fname: str,
    run_id: str,
    dir_download: str,
    artifacts_string: str = "imputation",
) -> Optional[Dict[str, Any]]:
    """Iterate through MLflow artifacts to find and download a specific file.

    Parameters
    ----------
    run_artifacts : list
        List of MLflow artifact objects.
    fname : str
        Filename to find and download.
    run_id : str
        MLflow run ID.
    dir_download : str
        Local directory for downloads (currently unused).
    artifacts_string : str, default "imputation"
        Artifact path to match.

    Returns
    -------
    dict or None
        Loaded results dictionary, or None if not found.

    Raises
    ------
    FileNotFoundError
        If the specified artifact cannot be found.
    """
    dict_out = None
    for artifact in run_artifacts:
        if artifact.path == artifacts_string:
            artifact_uri = "runs:/{}/{}/{}".format(run_id, artifact.path, fname)
            path_dir = mlflow.artifacts.download_artifacts(artifact_uri)
            if path_dir is not None:
                dict_out = load_results_dict(path_dir)
            else:
                logger.warning("MLFLOW | Could not find the artifact: {}".format(fname))
                raise FileNotFoundError(
                    "MLFLOW | Could not find the artifact: {}".format(fname)
                )

    return dict_out


def download_mlflow_artifacts(
    run_id: str, fname: str, run_artifacts: List[FileInfo]
) -> Optional[Dict[str, Any]]:
    """Download MLflow artifacts for a specific run.

    Parameters
    ----------
    run_id : str
        MLflow run ID.
    fname : str
        Filename to download.
    run_artifacts : list
        List of available artifacts.

    Returns
    -------
    dict
        Loaded artifacts dictionary.
    """
    dir_download = get_artifacts_dir("mlflow")
    dir_download.mkdir(parents=True, exist_ok=True)
    imputer_artifacts = iterate_through_mlflow_run_artifacts(
        run_artifacts, fname, run_id, str(dir_download)
    )

    return imputer_artifacts


def retrieve_mlflow_artifacts_from_best_run(
    best_run: Dict[str, Any], cfg: DictConfig, model_name: str
) -> Tuple[Dict[str, Any], List[FileInfo]]:
    """Retrieve imputation artifacts from the best MLflow run.

    Parameters
    ----------
    best_run : dict
        Best run data containing 'run_id'.
    cfg : DictConfig
        Configuration object (currently unused).
    model_name : str
        Name of the model for filename generation.

    Returns
    -------
    tuple
        Tuple of (imputer_artifacts, run_artifacts).

    Raises
    ------
    FileNotFoundError
        If no results are found in the best run.
    """
    fnames = {"imputation": get_imputation_pickle_name(model_name)}
    # NOT DONE ATM 'model': f"model_{model_name}.pickle"}

    run_id = best_run["run_id"]
    run_artifacts = mlflow.artifacts.list_artifacts(run_id=run_id)

    if run_artifacts is not None:
        imputer_artifacts = download_mlflow_artifacts(
            run_id, fname=fnames["imputation"], run_artifacts=run_artifacts
        )
        if imputer_artifacts is None:
            logger.error("MLflow | No imputation results found from the best run")
            raise FileNotFoundError("No imputation results found from the best run")
    else:
        # No we assume that you always saved "results", you may later wanna relax
        # this if you have some "mixed runs", or not?
        logger.error("MLflow | No results found from the best run")
        raise FileNotFoundError("No results found from the best run")

    return imputer_artifacts, run_artifacts


def get_mlflow_artifact_from_run_name(
    run_name: str, filter_for_finished: bool = True
) -> Optional[Dict[str, str]]:
    """Find MLflow artifact info by run name across all experiments.

    Parameters
    ----------
    run_name : str
        Name of the run to find.
    filter_for_finished : bool, default True
        If True, only search finished runs.

    Returns
    -------
    dict or None
        Dictionary with run_id, experiment_id, and artifact_uri if found.
    """
    all_runs = mlflow.search_runs(search_all_experiments=True)
    if filter_for_finished:
        # Filter for only "FINISHED" jobs
        all_runs: pd.DataFrame = all_runs[all_runs["status"] == "FINISHED"]

    # Check if the run_name exist (as if you have already run training with this name)
    if all_runs.shape[0] > 0:
        runs_remaining: pd.DataFrame = all_runs[
            all_runs["tags.mlflow.runName"] == run_name
        ]

        if runs_remaining.shape[0] > 0:
            mlflow_artifact = {
                "run_id": runs_remaining.iloc[0]["run_id"],
                "experiment_id": runs_remaining.iloc[0]["experiment_id"],
                "artifact_uri": runs_remaining.iloc[0]["artifact_uri"],
            }
            return mlflow_artifact
        else:
            logger.debug("No runs found with the run_name = {}".format(run_name))
            return None
    else:
        logger.debug("No runs found")
        return None


def return_best_mlflow_run(
    current_experiment: Dict[str, Any],
    metric_string: str,
    split_key: str,
    metric_direction: str,
    run_name: str,
) -> Optional[Dict[str, Any]]:
    """Find the best MLflow run matching the given criteria.

    Searches for runs with the specified name, filters out NaN metrics,
    and returns the best run based on metric direction.

    Parameters
    ----------
    current_experiment : dict
        Experiment dictionary with 'experiment_id'.
    metric_string : str
        Metric name to optimize.
    split_key : str
        Data split for the metric.
    metric_direction : str
        'ASC' for minimization, 'DESC' for maximization.
    run_name : str
        Exact run name to match.

    Returns
    -------
    dict or None
        Best run as dictionary, or None if no valid runs found.
    """

    def drop_nan_rows(df_runs: pd.DataFrame, metric_col: str) -> Optional[pd.DataFrame]:
        if metric_col in df_runs.columns:
            try:
                df_runs = df_runs.dropna(subset=[metric_col])
                return df_runs
            except Exception as e:
                logger.error("MLflow | Failed to drop NaN rows, e = {}".format(e))
                raise e
        else:
            logger.error(
                "MLflow | Could not find the metric column = {} in the dataframe".format(
                    metric_col
                )
            )
            logger.error(
                "Cannot pick the best model without the metric column, so returning an empty dictionary"
            )
            logger.error(
                "Handle better the runs that did not finish, so this metric easily might be missing!"
            )
            logger.error("Re-computing this part now!")
            logger.error(f"columns = {df_runs.columns}")
            return None

    def sort_runs_based_on_metric(
        df_runs: pd.DataFrame, metric_col: str, metric_direction: str
    ) -> Optional[Dict[str, Any]]:
        # Sort just to be sure (glitch while devving, should not be needed)
        if metric_direction == "ASC":
            df_runs = df_runs.sort_values(by=[best_metric_col], ascending=True)
        elif metric_direction == "DESC":
            df_runs = df_runs.sort_values(by=[best_metric_col], ascending=False)
        else:
            logger.error(
                "MLflow | Unknown metric direction = {}".format(metric_direction)
            )
            raise ValueError("Unknown metric direction = {}".format(metric_direction))

        if df_runs.shape[0] == 0:
            logger.warning(
                "MLflow | No runs found with the run_name = {}".format(run_name)
            )
            return None
        else:
            # first row is the best one, and we can convert it to a dictionary
            best_run_dict = df_runs.iloc[0].to_dict()
            logger.info(
                "MLflow | Found previous best run | Best run id = {}, best {} = {:.3f}".format(
                    best_run_dict["run_id"],
                    best_metric_col,
                    best_run_dict[best_metric_col],
                )
            )
            return best_run_dict

    # All runs in the experiment
    best_metric_col = f"metrics.{split_key}/{metric_string}"
    df: pd.DataFrame = mlflow.search_runs(
        [current_experiment["experiment_id"], f"{best_metric_col} {metric_direction}"]
    )
    logger.debug("MLflow | Found {} runs".format(len(df)))

    # Check for exact match of the run name
    df_runs = df[df["tags.mlflow.runName"] == run_name]
    logger.debug(
        "MLflow | Number of runs per this config version = {} (run_name = {})".format(
            df_runs.shape[0], run_name
        )
    )

    # Drop NaN rows (as in the best metric column, if you had unfinished runs)
    metric_col = f"metrics.{split_key}/{metric_string}"
    df_runs = drop_nan_rows(df_runs, metric_col)

    # Sort the runs based on the metric
    if df_runs is not None:
        df_runs = sort_runs_based_on_metric(df_runs, metric_col, metric_direction)

    return df_runs


def what_to_search_from_mlflow(
    run_name: str, cfg: DictConfig, model_type: Optional[str] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str], Optional[str]]:
    """Determine MLflow search parameters from run name and configuration.

    Parameters
    ----------
    run_name : str
        Name of the MLflow run.
    cfg : DictConfig
        Configuration containing IMPUTATION_METRICS settings.
    model_type : str, optional
        Model type (currently unused).

    Returns
    -------
    tuple
        Tuple of (current_experiment, metric_string, split_key, metric_direction),
        or (None, None, None, None) if run not found.
    """
    mlflow_artifacts = get_mlflow_artifact_from_run_name(run_name=run_name)

    if mlflow_artifacts is not None:
        client = MlflowClient()
        experiment_name = client.get_experiment(mlflow_artifacts["experiment_id"]).name
        current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
        best_metrics = cfg["IMPUTATION_METRICS"]["best_metric"]
        split = cfg["IMPUTATION_METRICS"]["best_metric"]["split"]
        split_key = f"{split}"

        # best_metric = list(best_metrics.keys())[0]
        metric_string = best_metrics["string"]
        metric_direction = best_metrics["direction"]

        return current_experiment, metric_string, split_key, metric_direction

    else:
        return None, None, None, None


def check_if_run_exists(experiment_name: str, run_name: str) -> bool:
    """Check if an MLflow run with the given name exists in the experiment.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment.
    run_name : str
        Run name to search for.

    Returns
    -------
    bool
        True if run exists, False otherwise.
    """
    runs = mlflow.search_runs(experiment_names=[experiment_name])
    if runs.shape[0] > 0:
        run_names = runs["tags.mlflow.runName"].values
        if run_name in run_names:
            logger.info(f"Run with the name {run_name} already exists")
            return True
        else:
            logger.info(f"Run with the name {run_name} does not exist")
            return False
    else:
        logger.info(f"No runs found for experiment: {experiment_name}")
        return False
