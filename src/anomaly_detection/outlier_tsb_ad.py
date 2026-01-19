import mlflow
import polars as pl
from omegaconf import DictConfig


from src.anomaly_detection.TSB_AD.model_wrapper import run_Unsupervise_AD
from src.anomaly_detection.anomaly_utils import get_data_for_sklearn_anomaly_models
from src.anomaly_detection.outlier_sklearn import (
    log_outlier_mlflow_artifacts,
    get_outlier_metrics,
)


def metrics_wrapper(score, labels, use_tsb_as_metrics: bool = False):
    preds = (score > 0.5).astype(int)
    metrics = get_outlier_metrics(score, preds, labels)

    if use_tsb_as_metrics:
        metrics = metrics_wrapper(score, labels)

    return metrics


def outlier_tsb_ad_wrapper(
    df: pl.DataFrame,
    cfg: DictConfig,
    model_cfg: DictConfig,
    experiment_name: str,
    run_name: str,
    model_name: str,
):
    """
    Call the model as you would when installing via uv
    https://github.com/TheDatumOrg/TSB-AD?tab=readme-ov-file#-basic-usage
    """
    metrics = {}

    train_on = model_cfg["MODEL"]["train_on"]
    X, y, X_test, y_test, light = get_data_for_sklearn_anomaly_models(
        df=df, cfg=cfg, train_on=train_on
    )

    train_out = run_Unsupervise_AD(model_name=model_name, data=X)
    metrics["outlier_train"] = metrics_wrapper(train_out, y)
    metrics["train"] = metrics["outlier_train"]  # to match other methods

    test_out = run_Unsupervise_AD(model_name=model_name, data=X_test)
    metrics["outlier_test"] = metrics_wrapper(test_out, y_test)
    metrics["test"] = metrics["outlier_test"]  # to match other methods

    # Log the metrics and the results
    log_outlier_mlflow_artifacts(metrics, None, model_name)
    mlflow.end_run()

    return metrics, None
