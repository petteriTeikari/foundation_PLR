import os

import mlflow
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score
import numpy as np
from loguru import logger
import polars as pl
from omegaconf import DictConfig
from tqdm import tqdm

from sklearn.svm import OneClassSVM

from src.anomaly_detection.extra_eval.eval_outlier_detection import (
    get_scalar_outlier_metrics,
)
from src.log_helpers.local_artifacts import save_results_dict
from src.log_helpers.log_naming_uris_and_dirs import get_outlier_pickle_name
from src.utils import get_artifacts_dir


def subjectwise_LOF(X_subj, clf):
    out = clf.fit_predict(X_subj)
    pred = np.where(out == -1, 1, 0)
    # anomaly_scores = clf.negative_outlier_factor_
    return pred


def subjectwise_HPO(X, clf, model_name):
    preds = None
    no_subjects = X.shape[0]
    for subj_idx in range(no_subjects):
        X_subj = X[subj_idx, :].reshape(-1, 1)
        if model_name == "LOF":
            pred_subj = subjectwise_LOF(X_subj, clf)
        else:
            raise ValueError(f"Model {model_name} not supported")
        if preds is None:
            preds = pred_subj
        else:
            preds = np.vstack((preds, pred_subj))
    return preds


def datasetwise_HPO(X, clf, model_name):
    X_flat = X.reshape(-1, 1)
    if model_name == "LOF":
        preds = clf.fit_predict(X_flat)
        preds = np.where(preds == -1, 1, 0)
        preds = preds.reshape(-1, X.shape[1])
    else:
        raise ValueError(f"Model {model_name} not supported")
    return preds


def get_LOF(X, params, subjectwise: bool = True):
    clf = LocalOutlierFactor(**params)
    if subjectwise:
        # More relevant for our needs as we would like to use this in real-life for new subjects
        preds = subjectwise_HPO(X, clf, model_name="LOF")
    else:
        preds = datasetwise_HPO(X, clf, model_name="LOF")
    return preds, clf


def get_outlier_y_from_data_dict(data_dict: dict, label: str, split: str):
    labels = data_dict["df"][split]["labels"]
    if label == "all":
        y = labels["outlier_mask"]
    elif label == "granular":
        y = labels["outlier_mask_easy"] | labels["outlier_mask_medium"]
    elif label == "easy":
        y = labels["outlier_mask_easy"]
    elif label == "medium":
        y = labels["outlier_mask_medium"]
    else:
        logger.error(f"unknown label = {label}")
        raise ValueError(f"unknown label = {label}")
    no_of_outliers = np.sum(y)
    outlier_percentage = 100 * (no_of_outliers / y.size)
    logger.debug(f"Outlier percentage in {label} set: {outlier_percentage:.2f}%")
    return y


def eval_on_all_outlier_difficulty_levels(data_dict, preds, split: str):
    # difficulties = ["all", "granular", "easy", "medium"]
    difficulties = ["all", "easy", "medium"]
    # difficulties = ["easy", "medium"]
    dict_out = {}
    for diff in difficulties:
        y = get_outlier_y_from_data_dict(data_dict, diff, split=split)
        dict_out[diff] = {
            "scalars": get_scalar_outlier_metrics(preds, y, score=None),
            "arrays": {"pred_mask": preds, "trues": y},
        }
    return dict_out


def get_outlier_metrics(
    score, preds, y, df: pl.DataFrame = None, cfg: DictConfig = None, split: str = None
):
    dict_all = {
        "scalars": get_scalar_outlier_metrics(preds, y, score, cfg),
        "arrays": {"pred_mask": preds, "trues": y},
    }
    # data_dict = convert_df_to_dict(data_df=df, cfg=cfg)
    # dict_granular = eval_on_all_outlier_difficulty_levels(data_dict, preds, split=split)
    return dict_all


def LOF_wrapper(X, y, X_test, y_test, best_params):
    def preds_per_split(X, y, best_params):
        preds, clf = get_LOF(X, params=best_params, subjectwise=True)
        metrics = get_outlier_metrics(None, preds, y)
        return metrics

    metrics = {}
    metrics["train"] = preds_per_split(X=X, y=y, best_params=best_params)
    metrics["test"] = preds_per_split(X=X_test, y=y_test, best_params=best_params)
    # to match the other reconstruction methods
    metrics["outlier_train"] = metrics["train"]
    metrics["outlier_test"] = metrics["test"]

    return metrics


def sklearn_outlier_hyperparameter_tuning(
    model_cfg, X, y, model_name, contamination: float, subjectwise: bool = True
):
    # Quick n dirty grid search as we have only now one hard-coded hyperparam
    n = model_cfg["SEARCH_SPACE"]["GRID"]["n_neighbors"]
    n_range = list(range(n[0], n[1] + n[2], n[2]))

    f1s = []
    if model_name == "LOF":
        for n_neigh in tqdm(n_range, "LOF Hyperparameter Tuning"):
            params = {"n_neighbors": n_neigh, "contamination": contamination}
            preds, _ = get_LOF(X, params, subjectwise)
            f1 = f1_score(y.flatten(), preds.flatten())
            f1s.append(f1)
    elif model_name == "OneClassSVM":
        raise NotImplementedError("OneClassSVM not implemented yet")
    else:
        logger.error(f"Model {model_name} not supported")
        raise ValueError(f"Model {model_name} not supported")

    best_f1 = np.max(f1s)

    if model_name == "LOF":
        best_n_neigh = n_range[np.argmax(f1s)]
        # LOF dataset-wise: Best F1: 0.09516351911561492, Best n_neigh: 145
        # LOF Best F1: 0.25406203840472674, Best n_neigh: 200
        logger.info(f"Best F1: {best_f1}, Best n_neigh: {best_n_neigh}")
        return {"n_neighbors": best_n_neigh}


def subjectwise_OneClassSVM(X, params):
    no_subjects = X.shape[0]
    preds = None
    for subj_idx in range(no_subjects):
        X_subj = X[subj_idx, :].reshape(-1, 1)
        clf = OneClassSVM(**params).fit(X_subj)
        pred = clf.predict(X_subj)
        pred = np.where(pred == -1, 1, 0)
        if preds is None:
            preds = pred
        else:
            preds = np.vstack((preds, pred))

    return preds


def OneClassSVM_wrapper(X, y, X_test, y_test, params):
    metrics = {}
    preds = subjectwise_OneClassSVM(X, params)
    metrics["train"] = get_outlier_metrics(None, preds, y)

    preds_test = subjectwise_OneClassSVM(X_test, params)
    metrics["test"] = get_outlier_metrics(None, preds_test, y_test)

    # to match the other reconstruction methods
    metrics["outlier_train"] = metrics["train"]
    metrics["outlier_test"] = metrics["test"]

    return metrics


def mlflow_log_params(params):
    for key, val in params.items():
        try:
            mlflow.log_param(key, val)
        except Exception as e:
            logger.warning(f"Error in logging param {key}: {e}")


def log_outlier_pickled_artifact(metrics, model_name):
    artifacts_dir = get_artifacts_dir("outlier_detection")
    fname = get_outlier_pickle_name(model_name)
    path_out = os.path.join(artifacts_dir, fname)
    save_results_dict(metrics, path_out)
    mlflow.log_artifact(path_out, "outlier_detection")


def log_prophet_model(model, model_name):
    if model is not None:
        logger.debug("if you have a model, log it here")


def log_outlier_mlflow_artifacts(metrics, model, model_name):
    for split in metrics:
        global_metrics = metrics[split]["scalars"]["global"]
        for key, value in global_metrics.items():
            if value is not None:
                if isinstance(value, np.ndarray):
                    mlflow.log_metric(f"{split}/{key}_lo", value[0])
                    mlflow.log_metric(f"{split}/{key}_hi", value[1])
                else:
                    mlflow.log_metric(f"{split}/{key}", value)

    log_outlier_pickled_artifact(metrics, model_name)
    log_prophet_model(model, model_name)


def outlier_sklearn_wrapper(
    df: pl.DataFrame,
    cfg: DictConfig,
    model_cfg: DictConfig,
    experiment_name: str,
    run_name: str,
    model_name: str,
):
    """
    Hyperparameter tuning, see e.g.
    https://github.com/vsatyakumar/automatic-local-outlier-factor-tuning / https://arxiv.org/abs/1902.00567
    """

    train_on = model_cfg["MODEL"]["train_on"]
    from src.anomaly_detection.anomaly_utils import get_data_for_sklearn_anomaly_models

    X, y, X_test, y_test, _ = get_data_for_sklearn_anomaly_models(
        df=df, cfg=cfg, train_on=train_on
    )
    contamination = np.sum(y.flatten()) / len(y.flatten())  # 0.077

    # HPO for LOF
    if model_name == "LOF":
        best_params = sklearn_outlier_hyperparameter_tuning(
            model_cfg, X, y, model_name, contamination=contamination
        )

    if model_name == "LOF":
        best_params = {**best_params, "contamination": contamination}
        mlflow_log_params(best_params)
        metrics = LOF_wrapper(X, y, X_test, y_test, best_params)

    elif model_name == "OneClassSVM":
        params = {
            "gamma": model_cfg["MODEL"]["gamma"],
            "kernel": model_cfg["MODEL"]["kernel"],
            "nu": contamination,
        }
        mlflow_log_params(params)
        metrics = OneClassSVM_wrapper(X, y, X_test, y_test, params)

    else:
        logger.error(f"Model {model_name} not supported")
        raise ValueError(f"Model {model_name} not supported")

    # Log the metrics and the results
    log_outlier_mlflow_artifacts(metrics, None, model_name)
    mlflow.end_run()

    return metrics, None
