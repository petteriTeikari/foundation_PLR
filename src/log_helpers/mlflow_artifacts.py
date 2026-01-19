import numpy as np
from loguru import logger
import os
import pandas as pd
from omegaconf import DictConfig

import mlflow
from mlflow.tracking import MlflowClient

from src.ensemble.ensemble_logging import get_ensemble_pickle_name
from src.imputation.imputation_log_artifacts import get_imputation_pickle_name
from src.log_helpers.local_artifacts import load_results_dict
from src.utils import get_artifacts_dir


def get_mlflow_run_ids_from_imputation_artifacts(imputation_artifacts):
    run_ids = {}
    for model_name in imputation_artifacts["artifacts"].keys():
        mlflow_info = imputation_artifacts["artifacts"][model_name]["mlflow"]
        run_ids[model_name] = mlflow_info["run_info"]["run_id"]
    return run_ids


def get_mlflow_metric_params(
    metrics: dict, cfg: DictConfig, splitkey="gt", metrictype="global", metricname="mae"
):
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


def get_mlflow_params(mlflow_info):
    # Get the MLflow experiment and run ID that was used during the training
    experiment_id = mlflow_info["experiment"]["name"]
    run_id = mlflow_info["run_info"]["run_id"]
    mlflow.set_experiment(experiment_id)
    return experiment_id, run_id


def get_mlflow_info_from_model_dict(model_dict):
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
                artifact_uri=os.path.join(artifact_uri, dir_name)
            )
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if wildcard in file:
                        db_path = os.path.join(root, file)
    if db_path is None:
        logger.error("Could not find the DuckDB file from the MLflow artifacts")
        raise FileNotFoundError(
            "Could not find the DuckDB file from the MLflow artifacts"
        )
    return db_path


def write_new_col_to_mlflow(model_best_runs, col_name, col_name_init):
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
):
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


def harmonize_anomaly_col_name(col_name, model_best_runs, best_metric_cfg, model):
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
    best_run: pd.Series, col_name: str, best_metric_cfg: DictConfig
):
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
    model_best_runs,
    cfg,
    best_metric_cfg,
    task: str,
    model: str,
    include_all_variants: bool = False,
):
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
):
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


def get_mlflow_artifact_uri_from_run(best_run):
    artifact_uri: str = mlflow.get_run(best_run["run_id"]).info.artifact_uri
    return artifact_uri


def get_best_metric_from_current_run(
    metrics_model: dict, split_key: str, metric_string: str
) -> float:
    logger.info(
        f"Getting the best metric from the current run, metric = {metric_string}, "
        f"split = {split_key}"
    )
    return metrics_model[split_key]["global"][metric_string]


def get_best_previous_mlflow_logged_model(model_dict: dict, cfg: DictConfig) -> dict:
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
    run_artifacts: list,
    fname: str,
    run_id: str,
    dir_download: str,
    artifacts_string: str = "imputation",
):
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


def download_mlflow_artifacts(run_id: str, fname: str, run_artifacts: list) -> dict:
    dir_download = get_artifacts_dir("mlflow")
    os.makedirs(dir_download, exist_ok=True)
    imputer_artifacts = iterate_through_mlflow_run_artifacts(
        run_artifacts, fname, run_id, dir_download
    )

    return imputer_artifacts


def retrieve_mlflow_artifacts_from_best_run(
    best_run: dict, cfg: DictConfig, model_name: str
):
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


def get_mlflow_artifact_from_run_name(run_name: str, filter_for_finished: bool = True):
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
    current_experiment,
    metric_string,
    split_key,
    metric_direction,
    run_name: str,
) -> dict:
    def drop_nan_rows(df_runs: pd.DataFrame, metric_col: str) -> pd.DataFrame:
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
    ) -> dict:
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


def what_to_search_from_mlflow(run_name: str, cfg: DictConfig, model_type: str = None):
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


def check_if_run_exists(experiment_name, run_name):
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
