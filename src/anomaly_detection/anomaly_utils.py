import os


import pandas as pd
import polars as pl
from omegaconf import DictConfig
import mlflow
from loguru import logger

from src.anomaly_detection.momentfm_outlier.moment_io import (
    load_model_from_disk,
    compare_state_dicts,
)
from src.classification.classifier_log_utils import (
    get_cls_metrics_fname,
)
from src.data_io.data_wrangler import convert_df_to_dict
from src.ensemble.ensemble_logging import get_ensemble_pickle_name
from src.imputation.imputation_log_artifacts import get_imputation_pickle_name
from src.log_helpers.log_naming_uris_and_dirs import (
    update_outlier_detection_run_name,
    get_torch_model_name,
)
from src.data_io.data_utils import (
    export_dataframe_to_duckdb,
    load_from_duckdb_as_dataframe,
    get_unique_polars_rows,
)
from src.log_helpers.local_artifacts import load_results_dict
from src.log_helpers.log_naming_uris_and_dirs import get_outlier_pickle_name
from src.log_helpers.mlflow_artifacts import (
    get_duckdb_from_mlflow,
    check_if_run_exists,
    get_col_for_for_best_anomaly_detection_metric,
)
from src.log_helpers.mlflow_utils import init_mlflow_experiment


def pick_just_one_light_vector(light: pd.Series) -> pd.Series:
    """
    Just pick one light vector, as they should be the same for all the subjects
    """
    light_out = {}
    for key, array in light.items():
        light_out[key] = array[0, :]

    return light_out


def get_data_for_sklearn_anomaly_models(
    df: pl.DataFrame, cfg: DictConfig, train_on: str
):
    data_dict = convert_df_to_dict(data_df=df, cfg=cfg)
    X = data_dict["df"]["train"]["data"][train_on]
    y = data_dict["df"]["train"]["labels"]["outlier_mask"]
    assert X.shape == y.shape, f"X.shape: {X.shape}, y.shape: {y.shape}"

    X_test = data_dict["df"]["test"]["data"][train_on]
    y_test = data_dict["df"]["test"]["labels"]["outlier_mask"]
    assert (
        X_test.shape == y_test.shape
    ), f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}"

    # the timing should be the same for both train and test, and for all the subjects
    light = data_dict["df"]["train"]["light"]
    light = pick_just_one_light_vector(light)

    return X, y, X_test, y_test, light


def sort_anomaly_detection_runs_ensemble(
    mlflow_runs: pd.DataFrame, best_metric_cfg: DictConfig, sort_by: str, task: str
):
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
):
    """
    To be combined eventually with newer sort_anomaly_detection_runs_ensemble()
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
    best_metric_cfg: DictConfig = None,
) -> pd.Series:
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
    anomaly_cfg: DictConfig,
    experiment_name: str,
    cfg: DictConfig,
) -> bool:
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
    previous_experiment_name: str,
    cfg: DictConfig,
    copy_orig_db: bool = False,
):
    """
    TODO! not needed for the outlier detection as it is, but have some function end of each flow
     saving the results as duckDB so someone can check out the results easily without re-running everything?
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


def log_anomaly_model_as_mlflow_artifact(checkpoint_file, run_name):
    logger.info("Logging Anomaly Detection model as an artifact to MLflow")
    # Note! this can be a bit slow if you need to upload 1.3G models
    try:
        mlflow.log_artifact(local_path=checkpoint_file, artifact_path="model")
    except Exception as e:
        logger.error(f"Could not log the artifact: {e}")
        raise e


def print_available_artifacts(path):
    dir, fname = os.path.split(path)


def get_artifact(run_id, run_name, model_name, subdir="outlier_detection"):
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

        if subdir == "baseline_model" and 'ensembled_input' in run_name:
            logger.warning(f"No baseline available for the ensembled diverse classifiers")
            return None
        else:
            try:
                artifact = mlflow.artifacts.download_artifacts(artifact_uri=path)
            except Exception as e:
                logger.error(f"Could not download the artifact: {e}")
                logger.error(f"run_id: {run_id}, model_name: {model_name}, path: {path}")
                raise e
            return artifact
    except Exception as e:
        logger.error(f"Problem getting the artifact: {e}")
        logger.error(f"run_id: {run_id}, model_name: {model_name}, path: {path}")
        raise e


def check_outlier_detection_artifact(outlier_artifacts):
    first_epoch_key = list(outlier_artifacts["outlier_results"].keys())[0]
    outlier_results = outlier_artifacts["outlier_results"][first_epoch_key]
    check_outlier_results(outlier_results=outlier_results)


def check_outlier_results(outlier_results: dict):
    first_split = list(outlier_results.keys())[0]
    split_results = outlier_results[first_split]["results_dict"]["split_results"]
    check_split_results(split_results)


def check_split_results(split_results: dict):
    no_samples_in_flat = split_results["arrays_flat"]["trues_valid"].shape[0]
    no_samples_in_array = split_results["arrays"]["trues"].size
    # A bit bizarre issue with samples being dropped somewhere?
    assert (
        no_samples_in_flat == no_samples_in_array
    ), f"no_samples_in_flat: {no_samples_in_flat}, no_samples_in_array: {no_samples_in_array}, should be equal"


def get_no_subjects_in_outlier_artifacts(outlier_artifacts):
    first_epoch_key = list(outlier_artifacts["outlier_results"].keys())[0]
    split_results = outlier_artifacts["outlier_results"][first_epoch_key]["train"][
        "results_dict"
    ]["split_results"]
    no_subjects = split_results["arrays"]["trues"].shape[0]
    return no_subjects


def outlier_detection_artifacts_dict(mlflow_run, model_name, task):
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
            f"File size is over 2GB ({file_size_MB/1024:.2f} GB), is this correct? "
            f"something went wrong in the previous step?"
        )
        logger.warning(f"Artifact path: {outlier_artifacts_path}")
    outlier_artifacts = load_results_dict(outlier_artifacts_path)

    return outlier_artifacts


def get_moment_model_from_mlflow_artifacts(
    run_id, run_name, model, device, cfg, task: str, model_name="MOMENT"
):
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
    experiment_name, cfg, run_name, model_name, get_model: bool = False
) -> dict:
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
    Placeholder task for importing the anomaly detection results from MLflow, see if needs to stay or not
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
