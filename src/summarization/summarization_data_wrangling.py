# import polars as pl
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import mlflow
import numpy as np
import pandas as pd
import psutil
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from src.anomaly_detection.anomaly_utils import get_artifact
from src.classification.subflow_feature_classification import get_the_features
from src.data_io.define_sources_for_flow import define_sources_for_flow
from src.ensemble.ensemble_logging import get_sort_name
from src.log_helpers.local_artifacts import load_results_dict, save_results_dict
from src.log_helpers.log_naming_uris_and_dirs import (
    get_summary_artifacts_fpath,
    get_summary_fname,
    get_summary_fpath,
    parse_task_from_exp_name,
)
from src.summarization.summarize_classification import get_classification_summary_data


def export_summary_db_to_mlflow(
    data: Dict[str, Any],
    db_path: str,
    artifact_path: str,
    summary_experiment_name: str,
    experiment_name: str,
    cfg: DictConfig,
) -> None:
    """Export summary database and artifacts to MLflow.

    Logs the DuckDB database file and artifacts pickle to MLflow,
    along with metadata about the number of unique runs.

    Parameters
    ----------
    data : dict
        Summary data containing 'data_df' and 'artifacts_dict_summary'.
    db_path : str
        Path to the DuckDB database file.
    artifact_path : str
        Path to the artifacts pickle file.
    summary_experiment_name : str
        Name of the MLflow experiment for summaries.
    experiment_name : str
        Name of the source experiment being summarized.
    cfg : DictConfig
        Configuration dictionary.
    """
    mlflow.set_experiment(summary_experiment_name)
    logger.info(f"Logging summarization data to MLflow: {summary_experiment_name}")

    # with mlflow.start_run(run_id=mlflow.active_run().info.run_id):
    mlflow.log_artifact(db_path, artifact_path="dataframes")
    if "source_name" in data["data_df"].columns:
        n_uniq_runs = data["data_df"]["source_name"].nunique()
    elif isinstance(data["artifacts_dict_summary"], dict):
        n_uniq_runs = len(data["artifacts_dict_summary"])
    else:
        n_uniq_runs = np.nan
    mlflow.log_param("no_unique_sources_{}".format(experiment_name), n_uniq_runs)
    mlflow.log_artifact(artifact_path, artifact_path="artifacts")


def import_summary_db_from_mlflow(
    experiment_name: str, summary_exp_name: str, cfg: DictConfig
) -> Dict[str, pd.DataFrame]:
    """Import summary database from MLflow artifacts.

    Downloads the most recent DuckDB database file from MLflow and
    reads it into a dictionary of DataFrames.

    Parameters
    ----------
    experiment_name : str
        Name of the source experiment to import summaries for.
    summary_exp_name : str
        Name of the MLflow summary experiment.
    cfg : DictConfig
        Configuration dictionary.

    Returns
    -------
    dict
        Dictionary containing 'data_df' and 'mlflow_runs' DataFrames.

    Raises
    ------
    ValueError
        If no runs are found in the MLflow experiment.
    """
    run_name = "summary_tmp"
    mlflow.set_experiment(experiment_name)
    logger.info(
        f"Reading {experiment_name} summarization data from MLflow: {summary_exp_name}"
    )
    runs = mlflow.search_runs(experiment_names=[summary_exp_name]).sort_values(
        by="start_time", ascending=False
    )
    # keep the last run of the run_name
    if len(runs) > 0:
        run_series = runs.iloc[0]
    else:
        logger.error(
            "No runs found in MLflow, set cfg['SUMMARIZATION']['import_from_duckdb'] = False? "
            "If you have never generated the summaries. Now the value is '{}'".format(
                cfg["SUMMARIZATION"]["import_from_duckdb"]
            )
        )
        raise ValueError(
            "No runs found in MLflow, set cfg['SUMMARIZATION']['import_from_duckdb'] = False"
        )

    logger.info("Get the latest run with name {}".format(run_name))

    subdir = "dataframes"
    fname = get_summary_fname(experiment_name)
    path = f"runs:/{run_series['run_id']}/{subdir}/{fname}"
    try:
        db_path = mlflow.artifacts.download_artifacts(artifact_uri=path)
        df = import_summary_dataframe_from_duckdb(db_path)
    except Exception as e:
        logger.error(f"Error downloading artifact: {e}")
        raise e

    return df


def import_summary_dataframe_from_duckdb(db_path: str) -> Dict[str, pd.DataFrame]:
    """Read summary DataFrames from a DuckDB database file.

    Parameters
    ----------
    db_path : str
        Path to the DuckDB database file.

    Returns
    -------
    dict
        Dictionary containing:
        - 'data_df': Main data DataFrame
        - 'mlflow_runs': MLflow run metadata DataFrame
    """
    filesize = Path(db_path).stat().st_size / 1024**2
    logger.info(
        f"Reading summarization dataframe from DuckDB Database ({filesize:.2f} MB): {db_path}"
    )
    with duckdb.connect(database=db_path, read_only=False) as con:
        data_df = con.query("SELECT * FROM data_df").df()
        mlflow_runs = con.query("SELECT * FROM mlflow_runs").df()
        # artifacts_dict_summary = con.query("SELECT * FROM artifacts_dict_summary").df()

    data = {
        "data_df": data_df,
        "mlflow_runs": mlflow_runs,
        # "artifacts_dict_summary": artifacts_dict_summary
    }
    return data


def export_summary_dataframe_to_duckdb(
    db_path: str,
    data: Dict[str, Any],
    debug_DuckDBWrite: bool = False,
) -> str:
    """Export summary DataFrames to a DuckDB database file.

    Creates tables for data_df and mlflow_runs in the database.
    Overwrites any existing database at the specified path.

    Parameters
    ----------
    db_path : str
        Path where the DuckDB database will be created.
    data : dict
        Dictionary containing 'data_df' and 'mlflow_runs' DataFrames.
    debug_DuckDBWrite : bool, optional
        If True, reads back the database to verify write success.
        Default is False.

    Returns
    -------
    str
        Path to the created database file.
    """
    logger.info("Writing dataframe to DuckDB Database: {}".format(db_path))
    db_path_obj = Path(db_path)
    if db_path_obj.exists():
        logger.warning("DuckDB Database already exists, removing the old one")
        db_path_obj.unlink()

    with duckdb.connect(database=db_path, read_only=False) as con:
        data_df = data["data_df"]  # noqa: F841
        con.execute("""
                    CREATE TABLE IF NOT EXISTS 'data_df' AS SELECT * FROM data_df;
                """)
        mlflow_runs = data["mlflow_runs"]  # noqa: F841
        con.execute("""
                            CREATE TABLE IF NOT EXISTS 'mlflow_runs' AS SELECT * FROM mlflow_runs;
                        """)
        # artifacts_dict_summary = data["artifacts_dict_summary"]  # noqa: F841
        # con.execute("""
        #                     CREATE TABLE IF NOT EXISTS 'artifacts_dict_summary' AS SELECT * FROM artifacts_dict_summary;
        #                 """)
    filesize = Path(db_path).stat().st_size / 1024**2
    logger.info(f"Filesize of DuckDB Database ({filesize:.2f} MB): {db_path}")
    if debug_DuckDBWrite:
        import_summary_dataframe_from_duckdb(db_path)

    return db_path


def export_summarization_flow_data(
    data: Dict[str, Any],
    experiment_name: str,
    summary_experiment_name: str,
    cfg: DictConfig,
) -> None:
    """Export complete summarization flow data to disk and MLflow.

    Saves data to DuckDB database and artifacts pickle file, then
    logs both to MLflow for reproducibility.

    Parameters
    ----------
    data : dict
        Summary data dictionary containing DataFrames and artifacts.
    experiment_name : str
        Name of the source experiment being summarized.
    summary_experiment_name : str
        Name of the MLflow summary experiment.
    cfg : DictConfig
        Configuration dictionary.
    """
    db_path = get_summary_fpath(experiment_name)
    export_summary_dataframe_to_duckdb(db_path=db_path, data=data)
    artifact_path = get_summary_artifacts_fpath(experiment_name)
    # TODO! eventually would be nicer to write a dataframe rather than this pickled dict, thus thw _df in key
    save_results_dict(
        results_dict=data["artifacts_dict_summary"],
        results_path=artifact_path,
        name="artifacts",
    )
    export_summary_db_to_mlflow(
        data, db_path, artifact_path, summary_experiment_name, experiment_name, cfg
    )


def flatten_data_per_split(
    split: str, split_data_dict: Dict[str, Dict[str, np.ndarray]]
) -> pd.DataFrame:
    """Flatten multi-dimensional data arrays into a single DataFrame.

    Converts nested data arrays from (subjects x timepoints) format
    into a flat format suitable for DataFrame storage.

    Parameters
    ----------
    split : str
        Data split identifier (unused, kept for API consistency).
    split_data_dict : dict
        Dictionary with 'data' key containing {variable: array} mappings.

    Returns
    -------
    pd.DataFrame
        DataFrame with flattened data, one column per variable.
    """
    dict_tmp = {}
    for category, variable_dict in split_data_dict.items():
        if (
            category == "data"
        ):  # all the metadata is the same across all the runs, just pick the stuff that is different
            for variable, data_array in variable_dict.items():
                # no_subjects, no_timepoints = data_array.shape
                flatten_array = data_array.flatten()
                # the names now should match the original DuckDB column names
                # metadata category data is not per time point, just per subject, so obviously you waste some RAM
                # here, but maybe sinmpler this way as we don't have a massive dataset anyway
                dict_tmp[variable] = flatten_array

    return pd.DataFrame(dict_tmp)


def create_dataframe_from_single_source(
    source_data: Dict[str, Any], source_name: str
) -> pd.DataFrame:
    """Create a combined DataFrame from a single data source.

    Flattens data across all splits and combines them into a single
    DataFrame with source identification.

    Parameters
    ----------
    source_data : dict
        Dictionary containing 'df' with nested split data.
    source_name : str
        Identifier for the data source (e.g., method name).

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all splits and source_name column.
    """
    dict_splits = source_data["df"]
    df_dict: dict[str, pd.DataFrame] = {}
    for split, split_data_dict in dict_splits.items():
        df_dict[split] = flatten_data_per_split(split, split_data_dict)

    # combine datafranes
    df_out = None
    for split, df in df_dict.items():
        if df_out is None:
            df_out = df
        else:
            df_out = pd.concat([df_out, df], axis=0)

    # add source data as a column
    df_out["source_name"] = source_name

    return df_out


def get_artifacts_dict(
    mlflow_run: Optional[pd.Series], experiment_name: str
) -> Optional[Dict[str, Any]]:
    """Load artifacts dictionary from an MLflow run.

    Downloads and loads the pickled artifacts (metrics, predictions)
    from the specified MLflow run.

    Parameters
    ----------
    mlflow_run : pd.Series or None
        MLflow run metadata as a pandas Series. None for ground truth sources.
    experiment_name : str
        Name of the experiment to determine artifact subdirectory.

    Returns
    -------
    dict or None
        Loaded artifacts dictionary, or None if mlflow_run is None.

    Notes
    -----
    For imputation tasks, 'source_data' is removed from artifacts to save RAM.
    """
    if mlflow_run is not None:
        task = parse_task_from_exp_name(experiment_name)
        run_id = mlflow_run["run_id"]
        run_name = mlflow_run["tags.mlflow.runName"]
        model_name = mlflow_run[get_sort_name(task)]
        artifact_path = get_artifact(run_id, run_name, model_name, subdir=task)
        artifacts = load_results_dict(artifact_path)
        if task == "imputation":
            # redundant, only eating up RAM
            artifacts.pop("source_data", None)
        return artifacts
    else:
        # is none when source is pupil_gt so no MLflow run was done to get it
        return None


def concatenate_dataframes_from_disk(df_sources_tmp_files: List[str]) -> pd.DataFrame:
    """Concatenate multiple CSV files into a single DataFrame.

    Reads temporary CSV files from disk and combines them into
    one DataFrame. Used to reduce memory usage during processing.

    Parameters
    ----------
    df_sources_tmp_files : list
        List of paths to temporary CSV files.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame from all source files.
    """
    logger.info("Concatenating dataframes from disk")
    for i, df_sources_tmp_file in enumerate(
        tqdm(df_sources_tmp_files, desc="Concatenating dataframes")
    ):
        if i == 0:
            df_sources = pd.read_csv(df_sources_tmp_file)
        else:
            df = pd.read_csv(df_sources_tmp_file)
            df_sources = pd.concat([df_sources, df], axis=0)

    return df_sources


def get_data_from_sources(
    sources: Dict[str, Dict[str, Any]], experiment_name: str, cfg: DictConfig
) -> Dict[str, Any]:
    """Extract and combine data from multiple experiment sources.

    Processes each source's data into a DataFrame, saves to temporary
    files to manage memory, and collects MLflow run metadata and artifacts.

    Parameters
    ----------
    sources : dict
        Dictionary mapping source names to their data dictionaries.
    experiment_name : str
        Name of the experiment being summarized.
    cfg : DictConfig
        Configuration dictionary.

    Returns
    -------
    dict
        Dictionary containing:
        - 'data_df': Combined DataFrame from all sources
        - 'mlflow_runs': DataFrame of MLflow run metadata
        - 'artifacts_dict_summary': Dictionary of artifacts per source
    """
    df_sources_tmp_files = []
    mlflow_runs = pd.DataFrame()
    artifacts_dict = {}
    # task = parse_task_from_exp_name(experiment_name)
    source_names = list(sources.keys())

    tmp_dir = tempfile.TemporaryDirectory()
    logger.info("Saving temporary dataframes to disk, {}".format(tmp_dir.name))

    for source_name in (pbar := tqdm(source_names)):
        # As in the imported data with the output of the task
        pbar.set_description(
            f"Import Sources | RAM use: {psutil.virtual_memory().percent} %: {source_name}"
        )

        df = create_dataframe_from_single_source(sources[source_name], source_name)
        df_sources_tmp_file = str(Path(tmp_dir.name) / f"{source_name}.csv")
        df.to_csv(df_sources_tmp_file, index=False)
        df_sources_tmp_files.append(df_sources_tmp_file)

        # The scalars logged to MLflow (if there are any)
        mlflow_run: pd.Series = sources[source_name]["mlflow"]
        if mlflow_run is not None:
            mlflow_run["source_name"] = source_name
            mlflow_runs = pd.concat([mlflow_runs, pd.DataFrame(mlflow_run).T], axis=0)

        # The artifacts saved during the run, i.e. metrics computed
        # TODO! Not the most memory efficient way to do this, as this maybe largish as is the dataframe
        artifacts_dict[source_name] = get_artifacts_dict(mlflow_run, experiment_name)

    # Delete the sources_dict and concanated dataframes saved to disk
    del sources
    df_sources = concatenate_dataframes_from_disk(df_sources_tmp_files)
    tmp_dir.cleanup()

    # You typically have one pupil_gt so you will have 1 less of mlflow_runs than no of sources
    data = {
        "data_df": df_sources,
        "mlflow_runs": mlflow_runs,
        "artifacts_dict_summary": artifacts_dict,  # convert_dict_of_metrics_to_df(artifacts_dict, task),
    }

    return data


def get_detaframe_from_features(features: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Create summary DataFrame from featurization results.

    Combines feature data from multiple sources and splits into
    a single DataFrame with featurization type annotations.

    Parameters
    ----------
    features : dict
        Dictionary mapping feature source names to their data and metadata.

    Returns
    -------
    dict
        Dictionary containing:
        - 'data_df': Combined features DataFrame
        - 'mlflow_runs': MLflow run metadata
        - 'artifacts_dict_summary': Empty dict (placeholder)
    """
    df_features = pd.DataFrame()
    mlflow_runs = pd.DataFrame()

    for feature_source, feature_dict in tqdm(
        features.items(), desc="Creating single dataframe from features"
    ):
        for split in feature_dict["data"].keys():
            df_per_split = feature_dict["data"][split].to_pandas()
            df_per_split["split"] = split
            df_per_split["source_name"] = feature_source
            if "embedding" in feature_source:
                df_per_split["featurization"] = "embedding"
            else:
                df_per_split["featurization"] = "handcrafted"
            df_features = pd.concat([df_features, df_per_split], axis=0)

        mlflow_run = pd.DataFrame(feature_dict["mlflow_run_featurization"])
        mlflow_runs = pd.concat([mlflow_runs, mlflow_run.T], axis=0)

    data = {
        "data_df": df_features,
        "mlflow_runs": mlflow_runs,
        "artifacts_dict_summary": {},  # convert_dict_of_metrics_to_df(artifacts_dict, task),
    }

    return data


def get_summarization_flow_data(
    cfg: DictConfig, experiment_name: str, summary_exp_name: str
) -> dict:
    """Get or generate summarization data for an experiment.

    Either imports existing summaries from DuckDB/MLflow or generates
    new summaries by processing all experiment sources.

    Parameters
    ----------
    cfg : DictConfig
        Configuration with SUMMARIZATION settings.
    experiment_name : str
        Name of the experiment to summarize.
    summary_exp_name : str
        Name of the summary experiment in MLflow.

    Returns
    -------
    dict
        Summary data dictionary containing DataFrames and artifacts.
    """
    if cfg["SUMMARIZATION"]["import_from_duckdb"]:
        data = import_summary_db_from_mlflow(experiment_name, summary_exp_name, cfg)
        data["artifacts_dict_summary"] = (
            1  # load_results_dict(get_summary_artifacts_fpath(experiment_name))
        )
    else:
        # get the data
        task = parse_task_from_exp_name(experiment_name)
        if task == "featurization":
            features = get_the_features(cfg=cfg, experiment_name=experiment_name)
            data = get_detaframe_from_features(features)
        elif task == "classification":
            data = get_classification_summary_data(cfg, experiment_name)
        else:
            sources = define_sources_for_flow(
                prev_experiment_name=experiment_name,
                cfg=cfg,
                task=parse_task_from_exp_name(experiment_name),
            )
            # create dataframe from the sources dictionary
            data = get_data_from_sources(sources, experiment_name, cfg)

        # save as DuckDB (this is easy single file to share on HuggingFace with the paper)
        export_summarization_flow_data(data, experiment_name, summary_exp_name, cfg)

    return data
