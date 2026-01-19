import os

import mlflow
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
import xgboost as xgb

from src.log_helpers.hydra_utils import get_intermediate_hydra_log_path
from src.log_helpers.local_artifacts import save_results_dict, load_results_dict
from src.utils import get_artifacts_dir


def log_classifier_sources_as_params(
    features_per_source: dict, dict_arrays, run_name: str, cfg: DictConfig
):
    """
    Logs the run_id of the different tasks so you can fetch always some artifacts later if you need to
    or in general you need in the summarization
    """
    keys = [
        "mlflow_run_imputation",
        "mlflow_run_featurization",
        "mlflow_run_outlier_detection",
    ]
    for key in keys:
        if features_per_source[key] is not None:
            value = features_per_source[key]["run_id"]
        else:
            value = None
        mlflow.log_param(key, value)

    if "embedding" in run_name:
        mlflow.log_param("featurization_method", "embeddings")
        for key, value in cfg["CLASSIFICATION_SETTINGS"]["DIM_REDUCTION"].items():
            mlflow.log_param(key, value)
    else:
        mlflow.log_param("featurization_method", "handcrafted_features")

    # Log the codes used for this train (TODO! DVC for dataset, or/and MLflow dataset)
    mlflow.log_param("codes_train", dict_arrays["subject_codes_train"])
    mlflow.log_param("codes_test", dict_arrays["subject_codes_test"])

    # log the submodel sources
    parse_and_log_cls_run_name(run_name)

    # log the source metrics (imputation and anomaly detection)
    log_source_metrics_as_params(
        mlflow_run_feat=features_per_source["mlflow_run_featurization"]
    )


def log_source_metrics_as_params(mlflow_run_feat: pd.Series):
    mlflow.log_param("Imputation_mae", mlflow_run_feat["metrics.Imputation_mae"])
    mlflow.log_param("Outlier_f1", mlflow_run_feat["metrics.Outlier_f1"])
    mlflow.log_param("Outlier_fp", mlflow_run_feat["metrics.Outlier_fp"])
    mlflow.log_param("Outlier_f1__easy", mlflow_run_feat["metrics.Outlier_f1__easy"])


def parse_and_log_cls_run_name(run_name: str, delimiter: str = "__"):
    try:
        cls_model, feat_method, imput_source, anomaly_source = run_name.split(delimiter)
        mlflow.log_param("feature_param", feat_method)
        mlflow.log_param("imputation_source", imput_source)
        mlflow.log_param("anomaly_source", anomaly_source)
        if imput_source == "pupil_gt" and anomaly_source == "pupil_gt":
            mlflow.log_param("both_sources_gt", True)
        else:
            mlflow.log_param("both_sources_gt", False)

        if "ensemble" in imput_source and "ensemble" in anomaly_source:
            mlflow.log_param("both_sources_ensemble", True)
        else:
            mlflow.log_param("both_sources_ensemble", False)

    except Exception:
        logger.warning(f"Could not parse run name: {run_name}")


def get_artifact_fileinfo(best_run: pd.Series, artifact_subdir: str):
    artifact_list = mlflow.artifacts.list_artifacts(run_id=best_run.run_id)
    artifact = None
    for artifact in artifact_list:
        if artifact.path == artifact_subdir:
            return artifact

    if artifact is None:
        logger.error(f"Could not find the artifact with the path {artifact_subdir}")
        raise ValueError(f"Could not find the artifact with the path {artifact_subdir}")


def import_cls_model_from_mlflow(
    best_run: str,
    cfg: DictConfig,
    model_fname: str = "model.xgb",
    load_from_autolog: bool = False,
):
    def define_autolog_artifact_path(run_id, artifact_path, model_fname):
        return f"runs:/{run_id}/{artifact_path}/{model_fname}"

    logger.info(
        f"Importing the classifier model from MLflow: {best_run['tags.mlflow.runName']}"
    )
    artifact = get_artifact_fileinfo(best_run, artifact_subdir="model")
    if load_from_autolog:
        raise NotImplementedError
        # fpath = mlflow.artifacts.download_artifacts(artifact_uri=define_autolog_artifact_path(best_run.run_id,
        #                                                                                       artifact.path,
        #                                                                                       model_fname))
        # model = mlflow.xgboost.load_model(fpath)
    else:
        fname = get_model_fname(
            run_name=best_run["tags.mlflow.runName"], xgboost_cfg=None
        )
        artifact_uri = f"runs:/{best_run.run_id}/{artifact.path}/{fname}"
        fpath = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
        if os.path.exists(fpath):
            # https://forecastegy.com/posts/xgboost-save-load-model-python/
            model = xgb.XGBClassifier()
            model.load_model(fpath)
            # TODO! Check that this performs as it did when saving the model!
            return model
        else:
            logger.error(
                f"Could not find the model artifact with the path {artifact.path}"
            )
            raise ValueError(
                f"Could not find the model artifact with the path {artifact.path}"
            )


def import_cls_metrics_from_mlflow(best_run: str, cfg: DictConfig):
    logger.info(
        f"Importing the classifier metrics from MLflow: {best_run['tags.mlflow.runName']}"
    )
    artifact = get_artifact_fileinfo(best_run, artifact_subdir="metrics")
    fname = get_cls_metrics_fname(run_name=best_run["tags.mlflow.runName"])
    artifact_uri = f"runs:/{best_run.run_id}/{artifact.path}/{fname}"
    fpath = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
    metrics = load_results_dict(fpath)
    return metrics


def import_cls_dict_arrays_from_mlflow(best_run, cfg):
    logger.info(
        f"Importing the classifier dictionary arrays from MLflow: {best_run['tags.mlflow.runName']}"
    )
    artifact = get_artifact_fileinfo(best_run, artifact_subdir="dict_arrays")
    fname = get_cls_arrays_fname(run_name=best_run["tags.mlflow.runName"], cfg=cfg)
    artifact_uri = f"runs:/{best_run.run_id}/{artifact.path}/{fname}"
    fpath = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
    dict_arrays = load_results_dict(fpath)
    return dict_arrays


def get_previous_best_classifier_run(run_name, cfg):
    best_runs = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{run_name}'")
    if best_runs.shape[0] > 0:
        logger.info(f"Found previous best run for {run_name}")
        return True, best_runs
    else:
        logger.info(f"No previous best run found for {run_name}")
        return False, None


def get_best_cls_run(best_runs, cfg):
    def get_best_cls_metric_string(cfg):
        metric = cfg["CLASSIFICATION_SETTINGS"]["BEST_METRIC"]["name"]
        split = cfg["CLASSIFICATION_SETTINGS"]["BEST_METRIC"]["split"]
        string_metric = f"metrics.{split}/{metric}"
        return string_metric

    if len(best_runs) > 0:
        string_metric = get_best_cls_metric_string(cfg)
        if cfg["CLASSIFICATION_SETTINGS"]["BEST_METRIC"]["direction"] == "ASC":
            sorted_runs = best_runs.sort_values(by=string_metric, ascending=True)
            return sorted_runs.iloc[0]
        elif cfg["CLASSIFICATION_SETTINGS"]["BEST_METRIC"]["direction"] == "DESC":
            sorted_runs = best_runs.sort_values(by=string_metric, ascending=False)
            return sorted_runs.iloc[0]
        else:
            logger.error(
                f"Unknown direction: {cfg['CLASSIFICATION_SETTINGS']['BEST_METRIC']['direction']}"
            )
            raise ValueError(
                f"Unknown direction: {cfg['CLASSIFICATION_SETTINGS']['BEST_METRIC']['direction']}"
            )

    else:
        logger.error("No best runs found, cannot import data from MLflow")
        logger.error(
            "This is bizarre if this happens, "
            "and we just checked the existence of previous runs with retrain_classifier()"
        )
        raise ValueError("No best runs found, cannot import data from MLflow")


def import_cls_mlflow_wrapper(
    run_name: str, cfg: DictConfig, model_fname: str = "model.xgb"
):
    logger.info(f"Importing the classifier model and metrics from MLflow: {run_name}")
    _, best_runs = get_previous_best_classifier_run(run_name, cfg)
    best_run = get_best_cls_run(best_runs, cfg)
    model = import_cls_model_from_mlflow(best_run, cfg, model_fname=model_fname)
    metrics = import_cls_metrics_from_mlflow(best_run, cfg)
    dict_arrays = import_cls_dict_arrays_from_mlflow(best_run, cfg)
    # TODO! If you want the DMatrix back as well
    return model, metrics, dict_arrays


def retrain_classifier(
    run_name: str, cfg: DictConfig, cfg_key: str = "CLASSIFICATION_SETTINGS"
):
    if cfg[cfg_key]["retrain_classifiers"]:
        logger.info("Retraining the classifiers")
        return True
    else:
        if_prev_found, _ = get_previous_best_classifier_run(run_name, cfg)
        if if_prev_found:
            logger.info("Not retraining the classifiers, as previous run(s) were found")
            return False
        else:
            logger.info(
                "Retraining the classifiers despite your desire to skip this, "
                "as no previous run(s) were found"
            )
            return True


def get_cls_arrays_fname(run_name: str, cfg: DictConfig):
    return f"dictArrays_{run_name}.pickle"


def get_input_array_path(run_name: str, cfg: DictConfig):
    dir_out = get_artifacts_dir(service_name="classification")
    fname = get_cls_arrays_fname(run_name, cfg)
    return os.path.join(dir_out, fname)


def get_cls_metrics_fname(run_name: str, prefix: str = None):
    fname = f"metrics_{run_name}.pickle"
    if prefix is not None:
        fname = f"{prefix}_{fname}"
    return fname


def get_metrics_path(run_name: str, prefix: str = None):
    dir_out = get_artifacts_dir(service_name="classification")
    fname = get_cls_metrics_fname(run_name)
    if prefix is not None:
        fname = f"{prefix}_{fname}"
    return os.path.join(dir_out, fname)


def get_model_fname(run_name: str, prefix: str = None):
    fname = f"model_{run_name}.pickle"
    if prefix is not None:
        fname = f"{prefix}_{fname}"
    return fname


def get_model_fpath(run_name: str, xgboost_cfg: DictConfig = None, prefix: str = None):
    dir_out = get_artifacts_dir(service_name="classification")
    fname = get_model_fname(run_name)
    if prefix is not None:
        fname = f"{prefix}_{fname}"
    return os.path.join(dir_out, fname)


def classifier_log_cls_evaluation_to_mlflow(
    model,
    baseline_results,
    models: list,
    metrics: dict,
    dict_arrays: dict,
    cls_model_cfg: DictConfig,
    run_name: str,
    model_name: str,
    log_manual_model: bool = True,
):
    # TODO! Get this from cfrg, and not as hardcoded
    class_names = {0: "Control", 1: "Glaucoma"}

    def log_metrics_stats(metrics: dict):
        scalar_stats = metrics["metrics_stats"]
        for split in scalar_stats.keys():
            if scalar_stats[split]["metrics"]["scalars"] is not None:
                for key, stat_dict in scalar_stats[split]["metrics"]["scalars"].items():
                    mlflow.log_metric(f"{split}/{key}", stat_dict["mean"])
                    mlflow.log_metric(f"{split}/{key}_CI_lo", stat_dict["ci"][0])
                    mlflow.log_metric(f"{split}/{key}_CI_hi", stat_dict["ci"][1])

    def log_global_subject_stats(metrics):
        def log_per_stat_key(stats_per_split, split, stat_key, class_key, class_name):
            metric_string = f"{split}/probs_{class_name}_{stat_key}_"
            # Note, this is now a mean of mean, and you could do std of mean, std of std
            metric_value = stats_per_split["y_pred_proba"][stat_key][class_key]["mean"]
            mlflow.log_metric(metric_string, metric_value)

            metric_string = f"{split}/probs_{class_name}_{stat_key}_CI_lo"
            metric_value = stats_per_split["y_pred_proba"][stat_key][class_key]["ci"][0]
            mlflow.log_metric(metric_string, metric_value)

            metric_string = f"{split}/probs_{class_name}_{stat_key}_CI_hi"
            metric_value = stats_per_split["y_pred_proba"][stat_key][class_key]["ci"][1]
            mlflow.log_metric(metric_string, metric_value)

        global_subject_stats = metrics["subject_global_stats"]
        for split, stats_per_split in global_subject_stats.items():
            for stat_key in stats_per_split["y_pred_proba"].keys():
                for class_key in stats_per_split["y_pred_proba"][stat_key].keys():
                    try:
                        class_name = class_names[class_key]
                    except Exception:
                        class_name = class_key.capitalize()
                    log_per_stat_key(
                        stats_per_split, split, stat_key, class_key, class_name
                    )

    def log_subjectwise_uq(metrics):
        for split in metrics["subjectwise_stats"].keys():
            for key, value in metrics["subjectwise_stats"][split]["uq"][
                "scalars"
            ].items():
                metric_string = f"{split}/{key}"
                mlflow.log_metric(metric_string, value)

    def log_input_arrays(dict_arrays: dict):
        results_path = get_input_array_path(run_name, cls_model_cfg)
        save_results_dict(
            results_dict=dict_arrays, results_path=results_path, name="classification"
        )
        mlflow.log_artifact(results_path, "dict_arrays")

    def log_metrics_as_pickle(metrics: dict):
        results_path = get_metrics_path(run_name)
        save_results_dict(
            results_dict=metrics,
            results_path=results_path,
            name="classification_metrics",
        )
        mlflow.log_artifact(results_path, "metrics")

    def log_baseline_model_as_pickle(
        baseline_results, model, cls_model_cfg: DictConfig
    ):
        fpath = get_model_fpath(run_name, prefix="baseline")
        save_results_dict(model, fpath, name="classification_model")
        mlflow.log_artifact(fpath, "baseline_model")

        results_path = get_metrics_path(run_name, prefix="baseline")
        save_results_dict(baseline_results, results_path, name="classification_model")
        mlflow.log_artifact(results_path, "baseline_model")

    def log_model_manually(models: list, run_name: str, cls_model_cfg: DictConfig):
        # Less reproducible option in long-term, but easier now for training and quick debugging
        # autolog() saves already model.xbg
        fpath = get_model_fpath(run_name)
        # Saves as a pickle, if you want to serve the model as an ensemble, you should write a custom class
        #  for the ensemble (https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html)
        try:
            save_results_dict(models, fpath, name="classification_model")
        except Exception as e:
            logger.error(f"Could not save the model to a local path: {fpath}")
            raise e
        mlflow.log_artifact(fpath, "model")
        # TODO! log_model() is the more reproducible option, but requires a model signature
        #  or make the autolog() work for the model
        # TODO! CatBoost ensemble have the invidual models in models.ensemble

    logger.info("Logging extra metrics to MLflow")
    if model_name == "TabM":
        if cls_model_cfg["MODEL"]["SAVE"]["skip_model_export"]:
            logger.info("Not saving TabM model weights to disk!")
            log_manual_model = False

    if log_manual_model:
        try:
            log_model_manually(models, run_name, cls_model_cfg)
        except Exception as e:
            logger.error(f"Could not save the model to MLflow: {e}")
            raise
    else:
        logger.info("Skip manual model log")

    log_metrics_stats(metrics)
    log_subjectwise_uq(metrics)
    log_global_subject_stats(metrics)
    log_metrics_as_pickle(metrics)

    if dict_arrays is not None:
        log_input_arrays(dict_arrays)

    # log the baseline results (as in from a single model, or from an ensemble without the bootstrapping
    # or similar method
    if baseline_results is not None:
        log_baseline_model_as_pickle(baseline_results, model, cls_model_cfg)

    # Log the log
    hydra_log_path = get_intermediate_hydra_log_path()
    mlflow.log_artifact(hydra_log_path, "hydra_logs")


def log_classifier_params_to_mlflow(model_params, cls_model_cfg):
    logger.info("Logging classifier parameters to MLflow")
    for param in model_params.keys():
        mlflow.log_param(param, model_params[param])
    for param, value in cls_model_cfg["MODEL"]["WEIGHING"].items():
        mlflow.log_param(param, value)
    if "DATA" in cls_model_cfg:
        for param, value in cls_model_cfg["DATA"].items():
            mlflow.log_param(param, value)
