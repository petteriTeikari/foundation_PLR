import momentfm
import polars as pl
import torch
from loguru import logger
from omegaconf import DictConfig

from src.anomaly_detection.anomaly_utils import check_outlier_detection_artifact
from src.anomaly_detection.log_anomaly_detection import log_anomaly_detection_to_mlflow
from src.anomaly_detection.momentfm_outlier.moment_anomaly_utils import (
    rearrange_moment_outlier_finetune_output,
)
from src.anomaly_detection.momentfm_outlier.moment_outlier_zeroshot import (
    momentfm_outlier_zeroshot,
)
from src.anomaly_detection.momentfm_outlier.momentfm_outlier_finetune import (
    momentfm_outlier_finetune,
)
from src.data_io.data_wrangler import convert_df_to_dict
from src.imputation.momentfm.moment_utils import (
    import_moment_model,
    init_torch_training,
)


def momentfm_outlier_model_train_and_eval(
    model: momentfm.MOMENTPipeline,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    data_dict: dict,
    cfg: DictConfig,
    outlier_model_cfg: DictConfig,
    run_name: str,
    model_name: str,
):
    """
    Train and evaluate MOMENT model for outlier detection.

    Supports both zero-shot and fine-tuning approaches based on configuration.

    Parameters
    ----------
    model : momentfm.MOMENTPipeline
        Pre-loaded MOMENT model.
    dataloaders : dict[str, torch.utils.data.DataLoader]
        Dictionary of dataloaders for train, test, outlier_train, outlier_test.
    data_dict : dict
        Data dictionary with arrays and metadata.
    cfg : DictConfig
        Full Hydra configuration.
    outlier_model_cfg : DictConfig
        MOMENT model configuration.
    run_name : str
        MLflow run name.
    model_name : str
        Model name for logging.

    Returns
    -------
    dict
        Outlier detection artifacts containing:
        - 'outlier_results': per-epoch results
        - 'metrics': evaluation metrics
        - 'preds': predictions
        - 'metadata': with 'best_epoch'

    Raises
    ------
    ValueError
        If detection_type is not 'zero-shot' or 'fine-tune'.
    """
    if outlier_model_cfg["MODEL"]["detection_type"] == "zero-shot":
        logger.info("MOMENT | Outlier Detection | Zero-shot")
        logger.info(f"run_name: {run_name}")
        logger.info("======================================")
        outlier_results_tmp, metrics, preds = momentfm_outlier_zeroshot(
            model,
            dataloaders,
            data_dict,
            cfg,
            outlier_model_cfg,
            run_name,
            task_name="outlier_detection",
        )

        # Add another nesting level to match this with finetune
        outlier_results = {"0": outlier_results_tmp}
        best_epoch = None

    elif outlier_model_cfg["MODEL"]["detection_type"] == "fine-tune":
        logger.info("MOMENT | Outlier Detection | Fine-tune")
        logger.info(f"run_name: {run_name}")
        logger.info("======================================")
        model, eval_dicts, best_epoch = momentfm_outlier_finetune(
            model, dataloaders, data_dict, cfg, outlier_model_cfg, run_name
        )

        # Transform the output a bit so it matches the zero-shot output in terms of metrics and preds
        # and outlier_results contain then everything that you can dump as a pickle
        outlier_results, metrics, preds = rearrange_moment_outlier_finetune_output(
            eval_dicts, best_epoch
        )

    else:
        logger.error(
            "Detection type '{}' not implemented! Typo? '(either zero-shot or fine-tune)".format(
                outlier_model_cfg["MODEL"]["detection_type"]
            )
        )
        raise ValueError(
            "Detection type '{}' not implemented! Typo? '(either zero-shot or fine-tune)".format(
                outlier_model_cfg["MODEL"]["detection_type"]
            )
        )

    # Save and log all the artifacts created during the training to MLflow
    outlier_artifacts = {
        "outlier_results": outlier_results,
        "metrics": metrics,
        "preds": preds,
        "metadata": {"best_epoch": best_epoch},
    }
    check_outlier_detection_artifact(outlier_artifacts)

    # Log to MLflow
    log_anomaly_detection_to_mlflow(model_name, run_name, outlier_artifacts, cfg)

    return outlier_artifacts


def momentfm_outlier_main(
    df: pl.DataFrame,
    cfg: DictConfig,
    outlier_model_cfg: DictConfig,
    experiment_name: str,
    run_name: str,
    model_name: str = "MOMENT",
):
    """
    Main entry point for MOMENT-based outlier detection.

    Initializes the MOMENT model, prepares data, and runs training/evaluation.

    Parameters
    ----------
    df : pl.DataFrame
        Input PLR data.
    cfg : DictConfig
        Full Hydra configuration.
    outlier_model_cfg : DictConfig
        MOMENT model configuration.
    experiment_name : str
        MLflow experiment name.
    run_name : str
        MLflow run name.
    model_name : str, optional
        Model name for logging. Default is 'MOMENT'.

    Returns
    -------
    tuple
        A tuple containing:
        - outlier_artifacts : dict
            Complete outlier detection results.
        - model : momentfm.MOMENTPipeline
            The (potentially fine-tuned) MOMENT model.
    """
    # init stuff
    data_dict = convert_df_to_dict(data_df=df, cfg=cfg)
    dataloaders = init_torch_training(
        data_dict,
        cfg,
        outlier_model_cfg,
        task="outlier_detection",
        run_name=run_name,
        create_outlier_dataloaders=True,
    )

    # Import the pretrained model
    model = import_moment_model(outlier_model_cfg, task="outlier_detection", cfg=cfg)
    model = model.to(cfg["DEVICE"]["device"]).float()

    # (Train) and find the outliers
    outlier_artifacts = momentfm_outlier_model_train_and_eval(
        model, dataloaders, data_dict, cfg, outlier_model_cfg, run_name, model_name
    )

    return outlier_artifacts, model
