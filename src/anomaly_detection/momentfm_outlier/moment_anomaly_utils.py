import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from momentfm.utils.forecasting_metrics import sMAPELoss
from torch import optim

from src.anomaly_detection.anomaly_utils import (
    check_outlier_results,
)
from src.anomaly_detection.momentfm_outlier.moment_optims import (
    LinearWarmupCosineLRScheduler,
)


def select_criterion(loss_type: str = "mse", reduction: str = "none", **kwargs):
    """
    Select loss criterion for reconstruction.

    Parameters
    ----------
    loss_type : str, optional
        Type of loss: 'mse', 'mae', 'huber', or 'smape'. Default is 'mse'.
    reduction : str, optional
        Reduction mode: 'none', 'mean', or 'sum'. Default is 'none'.
    **kwargs : dict
        Additional arguments (e.g., delta for Huber loss).

    Returns
    -------
    torch.nn.Module
        The configured loss criterion.

    Raises
    ------
    NotImplementedError
        If loss_type is not supported.
    """
    if loss_type == "mse":
        criterion = nn.MSELoss(reduction=reduction)
    elif loss_type == "mae":
        criterion = nn.L1Loss(reduction=reduction)
    elif loss_type == "huber":
        criterion = nn.HuberLoss(reduction=reduction, delta=kwargs["delta"])
    elif loss_type == "smape":
        criterion = sMAPELoss(reduction=reduction)
    else:
        logger.error(f"Loss {loss_type} not implemented")
        raise NotImplementedError(f"Loss {loss_type} not implemented")
    return criterion


def select_optimizer(
    model, init_lr: float, weight_decay: float, optimizer_name: str = "AdamW"
):
    """
    Select optimizer for model training.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters to optimize.
    init_lr : float
        Initial learning rate.
    weight_decay : float
        Weight decay coefficient.
    optimizer_name : str, optional
        Optimizer type: 'AdamW' or 'Adam'. Default is 'AdamW'.

    Returns
    -------
    torch.optim.Optimizer
        Configured optimizer.

    Raises
    ------
    NotImplementedError
        If optimizer_name is not supported.
    """
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=init_lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=init_lr,
            weight_decay=weight_decay,
        )
    # elif optimizer_name == "SGD":
    #     if momentum is None:
    #         logger.error("Momentum is required for SGD optimizer")
    #         raise ValueError
    #     optimizer = optim.SGD(
    #         model.parameters(),
    #         lr=init_lr,
    #         weight_decay=weight_decay,
    #         momentum=momentum,
    #     )
    else:
        logger.error(f"Optimizer {optimizer_name} not implemented")
        raise NotImplementedError(f"Optimizer {optimizer_name} not implemented")
    return optimizer


def init_lr_scheduler(
    optimizer,
    max_epoch: int,
    min_lr: float,
    init_lr: float,
    decay_rate: float,
    warmup_start_lr: float,
    warmup_steps: int,
    train_dataloader,
    pct_start: float,
    type: str = "linearwarmupcosinelr",
):
    """
    Initialize learning rate scheduler.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to schedule.
    max_epoch : int
        Maximum number of training epochs.
    min_lr : float
        Minimum learning rate.
    init_lr : float
        Initial learning rate.
    decay_rate : float
        Learning rate decay rate.
    warmup_start_lr : float
        Starting learning rate for warmup.
    warmup_steps : int
        Number of warmup steps.
    train_dataloader : torch.utils.data.DataLoader
        Training dataloader (for steps_per_epoch).
    pct_start : float
        Percentage of cycle spent increasing LR (for OneCycleLR).
    type : str, optional
        Scheduler type: 'linearwarmupcosinelr', 'onecyclelr', or 'none'.
        Default is 'linearwarmupcosinelr'.

    Returns
    -------
    object or None
        The configured scheduler, or None if type='none'.
    """

    if type == "linearwarmupcosinelr":
        lr_scheduler = LinearWarmupCosineLRScheduler(
            optimizer=optimizer,
            max_epoch=max_epoch,
            min_lr=min_lr,
            init_lr=init_lr,
            decay_rate=decay_rate,
            warmup_start_lr=warmup_start_lr,
            warmup_steps=warmup_steps,
        )
    elif type == "onecyclelr":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=init_lr,
            epochs=max_epoch,
            steps_per_epoch=len(train_dataloader),
            pct_start=pct_start,
        )
    elif type == "none":
        logger.warning("No learning rate scheduler used")
        lr_scheduler = None

    return lr_scheduler


def dtype_map(dtype: str):
    """
    Map string dtype to PyTorch dtype.

    Parameters
    ----------
    dtype : str
        String representation of dtype (e.g., 'float16', 'float32').

    Returns
    -------
    torch.dtype
        Corresponding PyTorch dtype.
    """
    map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }
    return map[dtype]


def debug_model_outputs(model, loss, outputs, batch_x, **kwargs):
    """
    Debug model outputs and gradients for training issues.

    Checks for NaN/Inf in loss, outputs, and gradients across model components.

    Parameters
    ----------
    model : torch.nn.Module
        MOMENT model to debug.
    loss : torch.Tensor
        Current loss value.
    outputs : object
        Model outputs with 'illegal_output' attribute.
    batch_x : torch.Tensor
        Input batch (currently unused).
    **kwargs : dict
        Additional arguments (currently unused).

    Notes
    -----
    Triggers breakpoint if loss is NaN/Inf or outputs are illegal.
    Logs warning if gradients contain NaN/Inf values.
    """
    # Debugging code
    if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)) or (loss < 1e-3):
        logger.error(f"Loss is NaN or Inf or too small. Loss is {loss.item()}.")
        breakpoint()

    # Check model outputs
    if outputs.illegal_output:
        logger.error("Model outputs are NaN or Inf.")
        breakpoint()

    # Check model gradients
    illegal_encoder_grads = (
        torch.stack([~torch.isfinite(p).any() for p in model.encoder.parameters()])
        .any()
        .item()
    )

    illegal_head_grads = (
        torch.stack([~torch.isfinite(p).any() for p in model.head.parameters()])
        .any()
        .item()
    )

    illegal_patch_embedding_grads = (
        torch.stack(
            [~torch.isfinite(p).any() for p in model.patch_embedding.parameters()]
        )
        .any()
        .item()
    )

    illegal_grads = (
        illegal_encoder_grads or illegal_head_grads or illegal_patch_embedding_grads
    )

    if illegal_grads:
        # self.logger.alert(title="Model gradients are NaN or Inf",
        #                     text=f"Model gradients are NaN or Inf.",
        #                     level=AlertLevel.INFO)
        # breakpoint()
        # logger.error("Model gradients are NaN or Inf.")
        # raise ValueError("Model gradients are NaN or Inf.")
        logger.warning("Model gradients are NaN or Inf.")

    return


def rearrange_moment_outlier_finetune_output(eval_dicts, best_epoch):
    """
    Rearrange fine-tuning output to match expected format.

    Extracts metrics and predictions for the best epoch from evaluation results.

    Parameters
    ----------
    eval_dicts : dict
        Evaluation results per epoch and split.
    best_epoch : int
        Index of the best epoch.

    Returns
    -------
    tuple
        A tuple containing:
        - eval_dicts : dict
            Original evaluation dictionaries.
        - metrics : dict
            Metrics per split for best epoch.
        - preds : dict
            Predictions per split for best epoch.
    """
    metrics, preds = {}, {}
    for split in eval_dicts[best_epoch]:
        metrics[split] = eval_dicts[best_epoch][split]["results_dict"]["metrics"]
        preds[split] = eval_dicts[best_epoch][split]["results_dict"]["preds"]

    # for epoch in eval_dicts:
    #     for split in eval_dicts[epoch]:
    #         eval_dicts[epoch][split]["best_epoch"] = best_epoch

    return eval_dicts, metrics, preds


def rearrange_moment_outlier_zeroshot_output(outlier_results):
    """
    Rearrange zero-shot output to match expected format.

    Extracts metrics and predictions from zero-shot evaluation results.

    Parameters
    ----------
    outlier_results : dict
        Zero-shot outlier detection results per split.

    Returns
    -------
    tuple
        A tuple containing:
        - outlier_results : dict
            Original results.
        - metrics : dict
            Metrics per split.
        - preds : dict
            Predictions per split.
    """
    metrics, preds = {}, {}
    check_outlier_results(outlier_results=outlier_results)
    for split in outlier_results:
        metrics[split] = outlier_results[split]["results_dict"]["metrics"]
        preds[split] = outlier_results[split]["results_dict"]["preds"]

    return outlier_results, metrics, preds


def list_of_arrays_to_array(list_of_arrays: list) -> np.ndarray:
    """
    Concatenate list of arrays into single array.

    Parameters
    ----------
    list_of_arrays : list
        List of numpy arrays to concatenate.

    Returns
    -------
    np.ndarray
        Concatenated array. 3D arrays are squeezed to 2D.
    """
    for i in range(len(list_of_arrays)):
        if i == 0:
            array_out = list_of_arrays[i]
        else:
            array_out = np.concatenate((array_out, list_of_arrays[i]), axis=0)

    if len(array_out.shape) == 3:
        array_out = array_out.squeeze()

    return array_out


def check_if_improved(
    cur_epoch,
    loss,
    eval_dicts,
    best_validation_loss,
    best_validation_metric,
    cfg,
    finetune_cfg,
):
    """
    Check if model performance has improved.

    Compares current validation loss and metric against best values.

    Parameters
    ----------
    cur_epoch : int
        Current training epoch.
    loss : torch.Tensor
        Current training loss.
    eval_dicts : dict
        Evaluation results for all splits.
    best_validation_loss : float
        Best validation loss so far.
    best_validation_metric : float
        Best validation metric so far.
    cfg : DictConfig
        Full Hydra configuration.
    finetune_cfg : DictConfig
        Fine-tuning configuration.

    Returns
    -------
    tuple
        A tuple containing:
        - improved_loss : bool
            Whether validation loss improved.
        - best_validation_loss : float
            Updated best validation loss.
        - best_validation_metric : float
            Updated best validation metric.

    Notes
    -----
    Assumes larger metric values are better.
    """
    improved_loss = False

    # Log the training loss
    if cfg["EXPERIMENT"]["debug"]:
        logger.info(
            f"{cur_epoch + 1}/{finetune_cfg['max_epoch']}: Training Loss {loss.item():.3f}"
        )

    # check if loss has improved
    if eval_dicts["test"]["average_loss"] < best_validation_loss:
        if cfg["EXPERIMENT"]["debug"]:
            logger.info(
                f"Validation loss improved from {best_validation_loss:.3f} to "
                f"{eval_dicts['test']['average_loss']:.3f}"
            )
        best_validation_loss = eval_dicts["test"]["average_loss"]
        improved_loss = True

    # check if metric has improved TODO! assumes that larger is better
    if eval_dicts["outlier_test"]["best_metric"] > best_validation_metric:
        if cfg["EXPERIMENT"]["debug"]:
            logger.info(
                f"Validation metric improved from {best_validation_metric:.3f} to "
                f"{eval_dicts['outlier_test']['best_metric']:.3f}"
            )
        best_validation_metric = eval_dicts["outlier_test"]["best_metric"]

    return improved_loss, best_validation_loss, best_validation_metric
