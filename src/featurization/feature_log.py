import os.path

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
import mlflow

from src.data_io.define_sources_for_flow import (
    get_best_dict,
    get_best_run_dict,
    drop_ensemble_runs,
)
from src.ensemble.ensemble_utils import (
    get_best_imputation_col_name,
    parse_imputation_run_name_for_ensemble,
)
from src.featurization.feature_utils import (
    export_features_pickle_file,
)
from src.log_helpers.local_artifacts import load_results_dict
from src.log_helpers.log_naming_uris_and_dirs import (
    get_feature_pickle_artifact_uri,
    experiment_name_wrapper,
)


def get_best_outlier_detection_run(
    simple_outlier_name: str, cfg: DictConfig, id: str = None
):
    # hacky fix for TimesNet ambiguities
    if simple_outlier_name == "TimesNet-gt":
        # the -gt was not used when running the anomaly detection, was only added now later for better
        # name for downstream processing, need to harmonize the names and the codes later
        simple_outlier_name = "TimesNet"

    experiment_name = experiment_name_wrapper(
        experiment_name=cfg["PREFECT"]["FLOW_NAMES"]["OUTLIER_DETECTION"], cfg=cfg
    )
    mlflow_runs: pd.DataFrame = mlflow.search_runs(experiment_names=[experiment_name])
    best_dict = get_best_dict("outlier_detection", cfg)

    if id is None:
        # keep the ones that contain the simple_outlier_name
        fields = simple_outlier_name.split("-")
        if len(fields) == 2:
            model_name = fields[0]  # e.g. MOMENT
            model_type = fields[1]  # e.g. zeroshot
            runs = mlflow_runs[
                mlflow_runs["tags.mlflow.runName"].str.contains(model_type)
                & mlflow_runs["tags.mlflow.runName"].str.contains(model_name)
            ]
        elif len(fields) == 1:
            model_name = simple_outlier_name  # e.g. SAITS
            runs = mlflow_runs[
                mlflow_runs["tags.mlflow.runName"].str.contains(model_name)
            ]
        elif len(fields) == 3:
            model_name = fields[0]  # e.g. MOMENT
            data_type = fields[1]  # e.g. gt/orig
            model_type = fields[2]  # e.g. zeroshot
            runs = mlflow_runs[
                mlflow_runs["tags.mlflow.runName"].str.contains(model_type)
                & mlflow_runs["tags.mlflow.runName"].str.contains(data_type)
                & mlflow_runs["tags.mlflow.runName"].str.contains(model_name)
            ]
        else:
            if "ensembleThreshold" in simple_outlier_name:
                model_name = "-".join(fields[1:3])
                runs = mlflow_runs[
                    mlflow_runs["tags.mlflow.runName"].str.contains(model_name)
                ]
            elif "ensemble" in simple_outlier_name:
                # we removed the name from this making this more difficult here :S
                field0corr = fields[0] + "Thresholded"
                model_name = (
                    field0corr + "-" + "-".join(fields[1:3])
                )  # hacky as things have been renamed TODO!
                runs = mlflow_runs[
                    mlflow_runs["tags.mlflow.runName"].str.contains(model_name)
                ]
            else:
                logger.error(
                    f"Unsupported number ({len(fields)}) of name fields: {fields}"
                )
                raise ValueError(
                    f"Unsupported number ({len(fields)}) of name fields: {fields}"
                )

        if runs.shape[0] > 0:
            if "ensemble" not in simple_outlier_name:
                runs = drop_ensemble_runs(runs)

            best_run = get_best_run_dict(
                runs, best_dict, task="outlier_detection"
            ).iloc[0:1]
            assert best_run.shape[0] == 1, "You should have only one best run"
            return best_run
        else:
            logger.warning("Could not find any best run?")
            return None
    else:
        run = mlflow_runs[mlflow_runs["run_id"] == id]
        return run


def get_best_outlier_run(mlflow_run: pd.Series, source_name: str, cfg: DictConfig):
    simple_outlier_name = source_name.split("__")[1]
    if "pupil" in simple_outlier_name:
        logger.debug("Using human-annotated Pupil data, no outlier detection")
        return None, None

    else:
        if "Outlier_run_id" in mlflow_run.keys():
            outlier_run_id = mlflow_run["Outlier_run_id"]
            best_outlier_run = get_best_outlier_detection_run(
                simple_outlier_name=simple_outlier_name, cfg=cfg, id=outlier_run_id
            )
        else:
            # Older runs did not log this, and requires maybe some twiddling to get this
            # And ensemble runs obviously do not have this anymore
            try:
                best_outlier_run = get_best_outlier_detection_run(
                    simple_outlier_name=source_name.split("__")[1], cfg=cfg
                )
                if best_outlier_run is not None:
                    outlier_run_id = str(best_outlier_run["run_id"].values[0])
                else:
                    outlier_run_id = None
            except Exception as e:
                logger.error(f"Could not get the outlier run id: {e}")
                raise e
                # return None, None

        if "ensemble" in source_name:
            if best_outlier_run is None:
                logger.info(
                    "Ensembled imputation, thus no single anomaly detection can be identified"
                )
                logger.info("You can ignore the warning now")

        return best_outlier_run, outlier_run_id


def metrics_when_anomaly_detection_pupil_gt(best_outlier_string: str):
    mlflow.log_param("Outlier_run_id", None)
    mlflow.log_metric(f"Outlier_{best_outlier_string}", 1)
    mlflow.log_metric("Outlier_fp", 0)
    mlflow.log_metric("Outlier_f1__easy", 1)


def featurization_mlflow_metrics_and_params(
    mlflow_run: pd.Series, source_name: str, cfg: DictConfig
):
    best_imput_dict = get_best_dict("imputation", cfg)
    best_imput_string = best_imput_dict["string"].replace("metrics.", "")
    best_outlier_dict = get_best_dict("outlier_detection", cfg)
    best_outlier_string = best_outlier_dict["string"].replace("metrics.", "")

    if mlflow_run is None:
        mlflow.log_param("Data source", source_name.split("__")[0])
        mlflow.log_param("Imputation_run_id", None)
        mlflow.log_metric(f"Imputation_{best_imput_string}", 0)
        metrics_when_anomaly_detection_pupil_gt(best_outlier_string)

    else:
        mlflow.log_param("Imputation_run_id", mlflow_run["run_id"])
        col_name = get_best_imputation_col_name(best_metric_cfg=best_imput_dict)
        mlflow.log_metric(f"Imputation_{best_imput_string}", mlflow_run[col_name])
        best_outlier_run, outlier_run_id = get_best_outlier_run(
            mlflow_run, source_name, cfg
        )

        if best_outlier_run is not None:
            mlflow.log_param("Outlier_run_id", outlier_run_id)
            col_name = get_best_imputation_col_name(best_metric_cfg=best_outlier_dict)
            # best_outlier_value = best_outlier_run[best_outlier_dict["string"]].values[0]
            best_outlier_value = best_outlier_run[col_name].values[0]
            mlflow.log_metric(f"Outlier_{best_outlier_string}", best_outlier_value)

            # add some extra metrics here
            metric1 = "fp"
            col_name = col_name.replace(best_outlier_string, metric1)
            value = best_outlier_run[col_name].values[0]
            mlflow.log_metric(f"Outlier_{metric1}", value)

            metric2 = "f1__easy"
            col_name = col_name.replace(metric1, metric2)
            value = best_outlier_run[col_name].values[0]
            mlflow.log_metric(f"Outlier_{metric2}", value)

        else:
            model_name, anomaly_source = parse_imputation_run_name_for_ensemble(
                source_name
            )
            if anomaly_source == "pupil-gt":
                metrics_when_anomaly_detection_pupil_gt(best_outlier_string)
            else:
                logger.warning("Could not find any best outlier run?")
                mlflow.log_param("Outlier_run_id", None)
                mlflow.log_metric(f"Outlier_{best_outlier_string}", np.nan)


# @task(
#     log_prints=True,
#     name="Export PLR features to MLflow",
#     description=" ",
# )
def export_features_to_mlflow(features: dict, run_name: str, cfg: DictConfig):
    # Local pickle export
    output_path = export_features_pickle_file(features, run_name, cfg)

    # Log the same to MLflow
    log_features_to_mlflow(run_name, output_path, features["mlflow_run"], cfg)


def log_features_to_mlflow(run_name, output_path, mlflow_run, cfg):
    logger.info(
        "Logging features ({}) as a pickled artifact to MLflow".format(run_name)
    )
    mlflow.log_artifact(output_path, artifact_path="features")


def get_best_run_per_source(
    cfg: DictConfig,
    experiment_name: str = "PLR_Featurization",
    skip_embeddings: bool = True,
):
    mlflow_runs: pd.DataFrame = mlflow.search_runs(experiment_names=[experiment_name])
    if mlflow_runs.shape[0] == 0:
        logger.error('No MLflow featurization runs found from {}'.format(experiment_name))
        logger.error('Did you run the previous steps (anomaly/outlier detection, '
                     'imputation and featurization for this experiment?)?')
        logger.error('Check values in "PROCESS_FLOWS" of your defaults.yaml for example? Should be True')
        raise ValueError('No MLflow featurization runs found from {}'.format(experiment_name))

    unique_sources = mlflow_runs["tags.mlflow.runName"].unique()
    if skip_embeddings:
        logger.info(
            "Skipping classification from the embedding sources, only using the hand-crafted features"
        )
        unique_sources = [
            source for source in unique_sources if "embedding" not in source
        ]

    runs = {}
    for source in unique_sources:
        runs_per_source = mlflow_runs[mlflow_runs["tags.mlflow.runName"] == source]
        if runs_per_source.shape[0] == 0:
            logger.error(
                f"No runs found for source {source} (some glitch here as we already found runs with this?"
            )
            raise ValueError(f"No runs found for source {source}")
        # Get the latest run (TODO! you could propagate the MAE from imputation to featurization as a parameter)
        best_run = runs_per_source.sort_values("start_time", ascending=False).iloc[0]
        if best_run.shape[0] == 0:
            logger.error(f"No best run found for source {source}")
            raise ValueError(f"No best run found for source {source}")
        runs[source] = best_run

    return runs


def get_mlflow_run_by_id(
    run_id: str,
    source: str,
    data_source: str,
    model_name: str,
    cfg: DictConfig,
    task_key: str = "OUTLIER_DETECTION",
):
    experiment_name = experiment_name_wrapper(
        experiment_name=cfg["PREFECT"]["FLOW_NAMES"][task_key], cfg=cfg
    )
    mlflow_runs: pd.DataFrame = mlflow.search_runs(experiment_names=[experiment_name])
    run = mlflow_runs[mlflow_runs["run_id"] == run_id]
    # assert run.shape[0] > 1, "The run_id = {} was not found from runs?".format(run_id)
    # assert run.shape[0] == 1, "You should have only one run"
    if run.shape[0] != 1:
        # e.g. The run_id = simple1.0__TimesNet__MOMENT-finetune was not found from runs?
        # (source = None, data_source = None, model = 98aa458aaab94323883a8be9afe3a63f)
        # This might happen if you re-run anomaly detection without re-running the imputation, thus even though
        # the results would converge to the same loss/metric, the preceding tasks are not exactly the same
        # This is fine behavior when you are debugging and developing, but maybe you want to run all the imputations
        # and other downstream tasks for the exactly correct preceding tasks
        logger.error(
            "The run_id = {} was not found from runs? (source = {}, data_source = {}, model = {})".format(
                source, data_source, model_name, run_id
            )
        )
        return None

    else:
        return run.iloc[0]


def import_features_per_source(source, run, cfg, subdir: str = "features"):
    """
    Import the features from the best MLflow run
    features_per_source: dict
      - data: dict
        - test: pl.DataFrame
            - column_names=
              ['subject_code', 'Red_MAX_CONSTRICTION_value', 'Red_MAX_CONSTRICTION_std', ...,
               'Blue_PHASIC_value', 'Blue_PHASIC_std', ..., 'metadata_Age']
        - train: pl.DataFrame
      - mlflow_run_imputation: type?
    """
    # TODO! You could check here if there any artifacts saved (mlflow did not run until the end),
    #  or you simply have the name incorrect
    artifact_uri = get_feature_pickle_artifact_uri(run, source, cfg, subdir=subdir)
    try:
        feature_pickle = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
        features_per_source = load_results_dict(feature_pickle)
    except Exception as e:
        try:
            # quick'n'dirty fix, if you have the incorrect MLflow URI if you want to do non-debug work
            # on a laptop: (reminder to always use some remote server :P)
            file_path = e.args[0].split(": '/")[1].replace('"', "").replace("'", "")
            if not os.path.exists(file_path):
                exp_path = "/home/petteri/Dropbox/manuscriptDrafts/foundationPLR/repo_desktop_clone (Selective Sync Conflict)/foundation_PLR/src/mlruns/393094052990429979/"
                artifact_path = "1958771273844452a9aaf6fb26f6837f/artifacts/features/simple1.0__pupil-gt__pupil-gt.pickle"
                file_path = os.path.join(exp_path, artifact_path)
                if os.path.exists(file_path):
                    features_per_source = load_results_dict(file_path)
                else:
                    logger.error(f"File not found: {file_path}")
                    raise e
        except Exception as e:
            logger.error(f"Error downloading artifact: {e}")
            raise e
    return features_per_source


def import_features_from_best_runs(best_runs: dict, cfg: DictConfig):
    features = {}
    for source, run in best_runs.items():
        if "embedding" in source:
            # from embeddings
            features[source] = import_features_per_source(
                source, run, cfg, subdir="embeddings"
            )
        else:
            # From hand-crafted features
            features[source] = import_features_per_source(source, run, cfg)

        if features[source] is not None:
            # rename the "mlflow_run" key to "mlflow_run_imputation"
            features[source]["mlflow_run_imputation"] = features[source].pop(
                "mlflow_run"
            )
            # add also the featurization run
            features[source]["mlflow_run_featurization"] = run
            if "params.Outlier_run_id" in run:
                if run["params.Outlier_run_id"] != "None":
                    if "params.model_name" in run:
                        model_name = run["params.model_name"]
                    else:
                        model_name = None
                    features[source]["mlflow_run_outlier_detection"] = (
                        get_mlflow_run_by_id(
                            run_id=run["params.Outlier_run_id"],
                            source=source,
                            data_source=run["params.Data source"],
                            model_name=model_name,
                            cfg=cfg,
                        )
                    )
                else:
                    # None when using the manually annotated data
                    features[source]["mlflow_run_outlier_detection"] = None
            else:
                features[source]["mlflow_run_outlier_detection"] = None

    return features


# @task(
#     log_prints=True,
#     name="Import PLR features from MLflow",
#     description=" ",
# )
def import_features_from_mlflow(
    cfg: DictConfig, experiment_name: str = "PLR_Featurization"
):
    # Get the best (latest) MLflow run for each data source
    best_runs = get_best_run_per_source(
        cfg,
        experiment_name,
        skip_embeddings=cfg["CLASSIFICATION_SETTINGS"]["train_from_embeddings"],
    )

    # Import the features from the best runs
    features = import_features_from_best_runs(best_runs, cfg)

    return features
