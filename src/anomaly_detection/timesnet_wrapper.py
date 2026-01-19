import mlflow
import numpy as np
import polars as pl
from omegaconf import DictConfig

from src.anomaly_detection.log_anomaly_detection import log_outlier_artifacts_dict
from src.anomaly_detection.ts_library.TimesNet import Model
from src.anomaly_detection.ts_library.timesnet_train import timesnet_train
from src.data_io.data_wrangler import convert_df_to_dict
from src.imputation.momentfm.moment_utils import init_torch_training


def log_mlflow_params(model, outlier_model_cfg):
    # Get number of parameters
    mlflow.log_param(
        "num_params", sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
    for key, value in outlier_model_cfg["PARAMS"].items():
        mlflow.log_param(key, value)
    for key, value in outlier_model_cfg["MODEL"].items():
        mlflow.log_param(key, value)


def log_timesnet_mlflow_metrics(metrics: dict, results_best: dict, best_epoch: int):
    mlflow.log_metric("best_epoch", best_epoch)

    for split in metrics:
        for key, value in metrics[split]["global"].items():
            if value is not None:
                if isinstance(value, np.ndarray):
                    mlflow.log_metric(f"{split}/{key}_lo", value[0])
                    mlflow.log_metric(f"{split}/{key}_hi", value[1])
                else:
                    mlflow.log_metric(f"{split}/{key}", value)

    for split in results_best:
        if "losses" in results_best[split]:
            loss_values = results_best[split]["losses"]
            loss_value = loss_values[best_epoch]
            mlflow.log_metric(f"{split}/recon_loss", loss_value)


def timesnet_outlier_wrapper(
    df: pl.DataFrame,
    cfg: DictConfig,
    outlier_model_cfg: DictConfig,
    experiment_name: str,
    run_name: str,
    task="outlier_detection",
    model_name: str = "TimesNet",
):
    """
    See e.g.
    "4.3 Anomaly Detection"
    https://github.com/thuml/Time-Series-Library/blob/main/tutorial/TimesNet_tutorial.ipynb

    """
    data_dict = convert_df_to_dict(data_df=df, cfg=cfg)
    dataloaders = init_torch_training(
        data_dict,
        cfg,
        outlier_model_cfg,
        task=task,
        run_name=run_name,
        model_name=model_name,
        create_outlier_dataloaders=True,
    )

    # Model
    model = Model(configs=outlier_model_cfg["PARAMS"])
    model.to(cfg["DEVICE"]["device"])

    # Log params
    log_mlflow_params(model, outlier_model_cfg)

    # Train
    outlier_artifacts, model = timesnet_train(
        model=model,
        device=cfg["DEVICE"]["device"],
        outlier_model_cfg=outlier_model_cfg,
        cfg=cfg,
        run_name=run_name,
        experiment_name=experiment_name,
        train_loader=dataloaders["train"],
        test_loader=dataloaders["test"],
        outlier_train_loader=dataloaders["outlier_train"],
        outlier_test_loader=dataloaders["outlier_test"],
        recon_on_outliers=False,
    )

    # Log metrics to MLflow
    log_timesnet_mlflow_metrics(
        metrics=outlier_artifacts["metrics"],
        results_best=outlier_artifacts["results_best"],
        best_epoch=outlier_artifacts["metadata"]["best_epoch"],
    )

    # Rearrange these a bit so that the output would roughly match MOMENT output
    # outlier_results = {0: outlier_artifacts} # add a dummy epoch?

    # Log the artifacts as well (pickled)
    log_outlier_artifacts_dict(model_name, outlier_artifacts, cfg, checks_on=False)

    # Log the model (seems not so useful, big in size, and quick to re-train, so saving some disk space and
    # not saving) implement model saving here otherwise
    mlflow.end_run()

    return outlier_artifacts, model
