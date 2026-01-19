import numpy as np
from loguru import logger
import torch
from torch import optim
import torch.nn as nn
from momentfm.utils.forecasting_metrics import sMAPELoss

from src.anomaly_detection.anomaly_utils import (
    check_outlier_results,
)
from src.anomaly_detection.momentfm_outlier.moment_optims import (
    LinearWarmupCosineLRScheduler,
)


def select_criterion(loss_type: str = "mse", reduction: str = "none", **kwargs):
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
    The Class implementation of the original paper was nicer
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
    metrics, preds = {}, {}
    for split in eval_dicts[best_epoch]:
        metrics[split] = eval_dicts[best_epoch][split]["results_dict"]["metrics"]
        preds[split] = eval_dicts[best_epoch][split]["results_dict"]["preds"]

    # for epoch in eval_dicts:
    #     for split in eval_dicts[epoch]:
    #         eval_dicts[epoch][split]["best_epoch"] = best_epoch

    return eval_dicts, metrics, preds


def rearrange_moment_outlier_zeroshot_output(outlier_results):
    metrics, preds = {}, {}
    check_outlier_results(outlier_results=outlier_results)
    for split in outlier_results:
        metrics[split] = outlier_results[split]["results_dict"]["metrics"]
        preds[split] = outlier_results[split]["results_dict"]["preds"]

    return outlier_results, metrics, preds


def list_of_arrays_to_array(list_of_arrays: list) -> np.ndarray:
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
    improved_loss = False

    # Log the training loss
    if cfg["EXPERIMENT"]["debug"]:
        logger.info(
            f"{cur_epoch+1}/{finetune_cfg['max_epoch']}: Training Loss {loss.item():.3f}"
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
