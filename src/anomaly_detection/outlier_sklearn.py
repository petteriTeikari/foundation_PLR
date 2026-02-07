from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import numpy as np
import polars as pl
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import f1_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from tqdm import tqdm

from src.anomaly_detection.extra_eval.eval_outlier_detection import (
    get_scalar_outlier_metrics,
)
from src.log_helpers.local_artifacts import save_results_dict
from src.log_helpers.log_naming_uris_and_dirs import get_outlier_pickle_name
from src.utils import get_artifacts_dir


def subjectwise_LOF(X_subj: np.ndarray, clf: LocalOutlierFactor) -> np.ndarray:
    """
    Apply Local Outlier Factor to a single subject's data.

    Parameters
    ----------
    X_subj : np.ndarray
        Time series data for one subject, shape (n_timepoints, 1).
    clf : LocalOutlierFactor
        Configured LOF classifier instance.

    Returns
    -------
    np.ndarray
        Binary outlier predictions (1 = outlier, 0 = inlier).
    """
    out = clf.fit_predict(X_subj)
    pred = np.where(out == -1, 1, 0)
    # anomaly_scores = clf.negative_outlier_factor_
    return pred


def subjectwise_HPO(X: np.ndarray, clf: Any, model_name: str) -> np.ndarray:
    """
    Apply hyperparameter-optimized outlier detection subject by subject.

    Parameters
    ----------
    X : np.ndarray
        Input data array of shape (n_subjects, n_timepoints).
    clf : sklearn estimator
        Configured outlier detection classifier.
    model_name : str
        Name of the model ('LOF' currently supported).

    Returns
    -------
    np.ndarray
        Binary outlier predictions of shape (n_subjects, n_timepoints).

    Raises
    ------
    ValueError
        If model_name is not supported.
    """
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


def datasetwise_HPO(X: np.ndarray, clf: Any, model_name: str) -> np.ndarray:
    """
    Apply outlier detection on the entire dataset at once.

    Flattens all subjects' data and fits a single model, then reshapes
    predictions back to original shape.

    Parameters
    ----------
    X : np.ndarray
        Input data array of shape (n_subjects, n_timepoints).
    clf : sklearn estimator
        Configured outlier detection classifier.
    model_name : str
        Name of the model ('LOF' currently supported).

    Returns
    -------
    np.ndarray
        Binary outlier predictions of shape (n_subjects, n_timepoints).

    Raises
    ------
    ValueError
        If model_name is not supported.
    """
    X_flat = X.reshape(-1, 1)
    if model_name == "LOF":
        preds = clf.fit_predict(X_flat)
        preds = np.where(preds == -1, 1, 0)
        preds = preds.reshape(-1, X.shape[1])
    else:
        raise ValueError(f"Model {model_name} not supported")
    return preds


def get_LOF(
    X: np.ndarray, params: Dict[str, Any], subjectwise: bool = True
) -> Tuple[np.ndarray, LocalOutlierFactor]:
    """
    Run Local Outlier Factor outlier detection.

    Parameters
    ----------
    X : np.ndarray
        Input data array of shape (n_subjects, n_timepoints).
    params : dict
        Parameters for LocalOutlierFactor (n_neighbors, contamination, etc.).
    subjectwise : bool, optional
        If True, fit LOF independently for each subject. If False, fit on
        all data at once. Default is True (more relevant for deployment).

    Returns
    -------
    tuple
        A tuple containing:
        - preds : np.ndarray
            Binary outlier predictions of shape (n_subjects, n_timepoints).
        - clf : LocalOutlierFactor
            The configured LOF classifier instance.
    """
    clf = LocalOutlierFactor(**params)
    if subjectwise:
        # More relevant for our needs as we would like to use this in real-life for new subjects
        preds = subjectwise_HPO(X, clf, model_name="LOF")
    else:
        preds = datasetwise_HPO(X, clf, model_name="LOF")
    return preds, clf


def get_outlier_y_from_data_dict(
    data_dict: Dict[str, Any], label: str, split: str
) -> np.ndarray:
    """
    Extract outlier labels from data dictionary by difficulty level.

    Parameters
    ----------
    data_dict : dict
        Data dictionary containing labels for each split.
    label : str
        Difficulty level: 'all', 'granular', 'easy', or 'medium'.
    split : str
        Data split: 'train' or 'test'.

    Returns
    -------
    np.ndarray
        Boolean outlier mask array.

    Raises
    ------
    ValueError
        If label is not a recognized difficulty level.
    """
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


def eval_on_all_outlier_difficulty_levels(
    data_dict: Dict[str, Any], preds: np.ndarray, split: str
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate outlier detection across all difficulty levels.

    Parameters
    ----------
    data_dict : dict
        Data dictionary containing labels for each split.
    preds : np.ndarray
        Binary outlier predictions.
    split : str
        Data split: 'train' or 'test'.

    Returns
    -------
    dict
        Dictionary with metrics for each difficulty level ('all', 'easy', 'medium').
        Each entry contains 'scalars' (metrics) and 'arrays' (predictions, labels).
    """
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
    score: Optional[np.ndarray],
    preds: np.ndarray,
    y: np.ndarray,
    df: Optional[pl.DataFrame] = None,
    cfg: Optional[DictConfig] = None,
    split: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute outlier detection metrics.

    Parameters
    ----------
    score : np.ndarray or None
        Anomaly scores (if available).
    preds : np.ndarray
        Binary outlier predictions.
    y : np.ndarray
        Ground truth outlier labels.
    df : pl.DataFrame, optional
        Original DataFrame (for future use). Default is None.
    cfg : DictConfig, optional
        Configuration (for future use). Default is None.
    split : str, optional
        Data split name (for future use). Default is None.

    Returns
    -------
    dict
        Dictionary containing:
        - 'scalars': Scalar metrics (F1, precision, recall, etc.)
        - 'arrays': Prediction mask and true labels.
    """
    dict_all = {
        "scalars": get_scalar_outlier_metrics(preds, y, score, cfg),
        "arrays": {"pred_mask": preds, "trues": y},
    }
    # data_dict = convert_df_to_dict(data_df=df, cfg=cfg)
    # dict_granular = eval_on_all_outlier_difficulty_levels(data_dict, preds, split=split)
    return dict_all


def LOF_wrapper(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_params: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Run LOF outlier detection on train and test splits.

    Parameters
    ----------
    X : np.ndarray
        Training data of shape (n_subjects, n_timepoints).
    y : np.ndarray
        Training labels (outlier mask).
    X_test : np.ndarray
        Test data of shape (n_subjects, n_timepoints).
    y_test : np.ndarray
        Test labels (outlier mask).
    best_params : dict
        Optimized LOF parameters (n_neighbors, contamination).

    Returns
    -------
    dict
        Metrics dictionary with keys 'train', 'test', 'outlier_train', 'outlier_test'.
    """

    def preds_per_split(
        X: np.ndarray, y: np.ndarray, best_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute LOF predictions and metrics for a single data split.

        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_subjects, n_timepoints).
        y : np.ndarray
            Ground truth outlier labels.
        best_params : dict
            Optimized LOF parameters.

        Returns
        -------
        dict
            Outlier detection metrics.
        """
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
    model_cfg: DictConfig,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    contamination: float,
    subjectwise: bool = True,
) -> Dict[str, int]:
    """
    Perform grid search hyperparameter tuning for sklearn outlier detectors.

    Parameters
    ----------
    model_cfg : DictConfig
        Model configuration containing search space parameters.
    X : np.ndarray
        Training data of shape (n_subjects, n_timepoints).
    y : np.ndarray
        Ground truth outlier labels.
    model_name : str
        Name of the model ('LOF' currently supported).
    contamination : float
        Expected proportion of outliers in the data.
    subjectwise : bool, optional
        Whether to fit model per subject. Default is True.

    Returns
    -------
    dict
        Best hyperparameters found (e.g., {'n_neighbors': 200}).

    Raises
    ------
    NotImplementedError
        If model_name is 'OneClassSVM' (not yet implemented).
    ValueError
        If model_name is not supported.
    """
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


def subjectwise_OneClassSVM(X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Apply One-Class SVM outlier detection subject by subject.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_subjects, n_timepoints).
    params : dict
        Parameters for OneClassSVM (gamma, kernel, nu).

    Returns
    -------
    np.ndarray
        Binary outlier predictions of shape (n_subjects, n_timepoints).
    """
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


def OneClassSVM_wrapper(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Run One-Class SVM outlier detection on train and test splits.

    Parameters
    ----------
    X : np.ndarray
        Training data of shape (n_subjects, n_timepoints).
    y : np.ndarray
        Training labels (outlier mask).
    X_test : np.ndarray
        Test data of shape (n_subjects, n_timepoints).
    y_test : np.ndarray
        Test labels (outlier mask).
    params : dict
        OneClassSVM parameters (gamma, kernel, nu).

    Returns
    -------
    dict
        Metrics dictionary with keys 'train', 'test', 'outlier_train', 'outlier_test'.
    """
    metrics = {}
    preds = subjectwise_OneClassSVM(X, params)
    metrics["train"] = get_outlier_metrics(None, preds, y)

    preds_test = subjectwise_OneClassSVM(X_test, params)
    metrics["test"] = get_outlier_metrics(None, preds_test, y_test)

    # to match the other reconstruction methods
    metrics["outlier_train"] = metrics["train"]
    metrics["outlier_test"] = metrics["test"]

    return metrics


def mlflow_log_params(params: Dict[str, Any]) -> None:
    """
    Log parameters to MLflow.

    Parameters
    ----------
    params : dict
        Dictionary of parameter names and values to log.

    Notes
    -----
    Logs a warning if a parameter cannot be logged.
    """
    for key, val in params.items():
        try:
            mlflow.log_param(key, val)
        except Exception as e:
            logger.warning(f"Error in logging param {key}: {e}")


def log_outlier_pickled_artifact(metrics: Dict[str, Any], model_name: str) -> None:
    """
    Save and log outlier detection results as a pickled artifact to MLflow.

    Parameters
    ----------
    metrics : dict
        Outlier detection metrics and predictions to save.
    model_name : str
        Name of the model (used for filename).
    """
    artifacts_dir = Path(get_artifacts_dir("outlier_detection"))
    fname = get_outlier_pickle_name(model_name)
    path_out = artifacts_dir / fname
    save_results_dict(metrics, str(path_out))
    mlflow.log_artifact(str(path_out), "outlier_detection")


def log_prophet_model(model: Optional[Any], model_name: str) -> None:
    """
    Log a Prophet model to MLflow (placeholder).

    Parameters
    ----------
    model : Prophet or None
        The Prophet model to log.
    model_name : str
        Name of the model.

    Notes
    -----
    Currently a stub function; model logging not implemented.
    """
    if model is not None:
        logger.debug("if you have a model, log it here")


def log_outlier_mlflow_artifacts(
    metrics: Dict[str, Dict[str, Any]], model: Optional[Any], model_name: str
) -> None:
    """
    Log outlier detection metrics and artifacts to MLflow.

    Parameters
    ----------
    metrics : dict
        Dictionary containing metrics for each split with 'scalars' and 'arrays'.
    model : object or None
        The trained model (if applicable).
    model_name : str
        Name of the model.

    Notes
    -----
    Logs scalar metrics to MLflow. For array values (confidence intervals),
    logs separate _lo and _hi metrics.
    """
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
) -> Tuple[Dict[str, Dict[str, Any]], None]:
    """
    Run sklearn-based outlier detection with hyperparameter tuning.

    Supports Local Outlier Factor (LOF) and One-Class SVM methods.
    For LOF, performs grid search hyperparameter tuning on n_neighbors.

    Parameters
    ----------
    df : pl.DataFrame
        Input PLR data containing pupil signals.
    cfg : DictConfig
        Full Hydra configuration.
    model_cfg : DictConfig
        Model-specific configuration.
    experiment_name : str
        MLflow experiment name.
    run_name : str
        MLflow run name.
    model_name : str
        Name of the model: 'LOF' or 'OneClassSVM'.

    Returns
    -------
    tuple
        A tuple containing:
        - metrics : dict
            Outlier detection metrics for train and test splits.
        - model : None
            No model object returned for sklearn methods.

    Raises
    ------
    ValueError
        If model_name is not supported.

    References
    ----------
    Hyperparameter tuning approach:
    https://github.com/vsatyakumar/automatic-local-outlier-factor-tuning
    https://arxiv.org/abs/1902.00567
    """

    train_on = model_cfg["MODEL"]["train_on"]
    from src.anomaly_detection.anomaly_utils import get_data_for_sklearn_anomaly_models

    X, y, X_test, y_test, _ = get_data_for_sklearn_anomaly_models(
        df=df, cfg=cfg, train_on=train_on
    )
    contamination = np.sum(y.flatten()) / len(y.flatten())  # 0.077
    # sklearn LOF requires contamination in (0, 0.5], cap it if needed (e.g., synthetic data)
    if contamination > 0.5:
        logger.warning(
            f"Contamination {contamination:.4f} exceeds 0.5 limit, capping to 0.499"
        )
        contamination = 0.499

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
