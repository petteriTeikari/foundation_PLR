import momentfm
import torch
from loguru import logger
from omegaconf import DictConfig

from src.anomaly_detection.anomaly_utils import (
    check_outlier_results,
    check_split_results,
)
from src.anomaly_detection.momentfm_outlier.moment_anomaly_utils import (
    rearrange_moment_outlier_zeroshot_output,
    select_criterion,
)
from src.anomaly_detection.momentfm_outlier.momentfm_outlier_finetune import (
    evaluate_model,
)


def momentfm_outlier_zeroshot(
    model: momentfm.MOMENTPipeline,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    data_dict: dict,
    cfg: DictConfig,
    outlier_model_cfg: DictConfig,
    run_name: str,
    task_name: str,
    tqdm_string="Zero-shot ",
):
    """
    Run zero-shot outlier detection with MOMENT.

    Uses the pretrained MOMENT model without fine-tuning to detect outliers
    based on reconstruction error.

    Parameters
    ----------
    model : momentfm.MOMENTPipeline
        Pretrained MOMENT model.
    dataloaders : dict[str, torch.utils.data.DataLoader]
        Dictionary of dataloaders for each split.
    data_dict : dict
        Data dictionary with arrays and metadata.
    cfg : DictConfig
        Full Hydra configuration.
    outlier_model_cfg : DictConfig
        Model configuration.
    run_name : str
        MLflow run name.
    task_name : str
        Task name for logging.
    tqdm_string : str, optional
        Progress bar prefix. Default is "Zero-shot ".

    Returns
    -------
    tuple
        A tuple containing:
        - outlier_results : dict
            Per-split outlier detection results.
        - metrics : dict
            Evaluation metrics per split.
        - preds : dict
            Predictions per split.
    """
    # We can compute the same loss as when finetuning, even though we are not optimizing any model parameters
    criterion = select_criterion(
        loss_type=outlier_model_cfg["LINEAR_PROBING"]["loss_type"], reduction="mean"
    )
    logger.info("Using loss: {}".format(criterion))
    device = cfg["DEVICE"]["device"]
    finetune_cfg = outlier_model_cfg["LINEAR_PROBING"]

    # Use the same evaluation function as when finetuning
    logger.info("MOMENT | {} | EVAL".format(task_name))
    eval_dicts = evaluate_model(
        dataloaders=dataloaders,
        data_dict=data_dict,
        model=model,
        device=device,
        criterion=criterion,
        finetune_cfg=finetune_cfg,
        outlier_model_cfg=outlier_model_cfg,
        cfg=cfg,
        cur_epoch=0,
        tqdm_string=tqdm_string,
        task_name=task_name,
    )
    logger.info("MOMENT | {} | EVAL | DONE!".format(task_name))
    check_split_results(
        split_results=eval_dicts["train"]["results_dict"]["split_results"]
    )

    outlier_results, metrics, preds = rearrange_moment_outlier_zeroshot_output(
        eval_dicts
    )
    check_outlier_results(outlier_results=outlier_results)

    return outlier_results, metrics, preds
