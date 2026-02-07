# See
# https://github.com/moment-timeseries-foundation-model/moment-research/blob/main/moment/tasks/anomaly_detection_finetune.py
# Simplified adaptation to this codebase

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from src.anomaly_detection.anomaly_detection_metrics_wrapper import (
    get_best_outlier_metric,
    metrics_per_split_v2,
)
from src.anomaly_detection.anomaly_utils import log_anomaly_model_as_mlflow_artifact
from src.anomaly_detection.momentfm_outlier.moment_anomaly_utils import (
    check_if_improved,
    debug_model_outputs,
    dtype_map,
    init_lr_scheduler,
    list_of_arrays_to_array,
    select_criterion,
    select_optimizer,
)
from src.anomaly_detection.momentfm_outlier.moment_forward import momentfm_forward_pass
from src.anomaly_detection.momentfm_outlier.moment_io import save_model_to_disk
from src.imputation.momentfm.moment_utils import (
    reshape_finetune_arrays,
)
from src.utils import get_artifacts_dir


def evaluate_model(
    dataloaders: dict[str, torch.utils.data.DataLoader],
    data_dict: dict,
    model,
    device,
    criterion,
    finetune_cfg: DictConfig,
    outlier_model_cfg: DictConfig,
    cfg: DictConfig,
    cur_epoch: int,
    tqdm_string: str = "",
    task_name: str = "outlier_detection",
):
    """
    Evaluate MOMENT model on all data splits.

    Parameters
    ----------
    dataloaders : dict[str, torch.utils.data.DataLoader]
        Dictionary of dataloaders for each split.
    data_dict : dict
        Data dictionary with arrays and metadata.
    model : torch.nn.Module
        MOMENT model to evaluate.
    device : str
        Device to run evaluation on ('cpu' or 'cuda').
    criterion : torch.nn.Module
        Loss function.
    finetune_cfg : DictConfig
        Fine-tuning configuration.
    outlier_model_cfg : DictConfig
        Model configuration.
    cfg : DictConfig
        Full Hydra configuration.
    cur_epoch : int
        Current training epoch.
    tqdm_string : str, optional
        Prefix for progress bar. Default is "".
    task_name : str, optional
        Task name. Default is "outlier_detection".

    Returns
    -------
    dict
        Evaluation results for each split containing average_loss,
        best_metric, and results_dict.
    """
    eval_dict = {}
    for split in dataloaders.keys():
        eval_dict[split] = eval_moment_outlier_finetune(
            model=model,
            dataloader=dataloaders[split],
            split=split,
            device=device,
            criterion=criterion,
            cfg=cfg,
            outlier_model_cfg=outlier_model_cfg,
            finetune_cfg=finetune_cfg,
            tqdm_string=tqdm_string,
            task_name=task_name,
        )

        array_dict = eval_dict[split]["results_dict"]["split_results"]["arrays"]
        array_names = list(array_dict.keys())
        shape_in = array_dict[array_names[0]].shape
        if shape_in[1] == cfg["DATA"]["PLR_length"]:
            logger.error("The numpy array(s) seem reshaped already!")
            raise ValueError("The numpy array(s) seem reshaped already!")

        eval_dict[split]["results_dict"] = reshape_finetune_arrays(
            results=eval_dict[split]["results_dict"],
            split=split,
            outlier_model_cfg=outlier_model_cfg,
            cfg=cfg,
        )

    # if you don't have outlier_xxx now in the dictionary, let's maintain downstream code compatibility and add them
    if "outlier_test" not in list(eval_dict.keys()):
        eval_dict["outlier_test"] = eval_dict["test"]
    if "outlier_train" not in list(eval_dict.keys()):
        eval_dict["outlier_train"] = eval_dict["train"]

    return eval_dict


def post_eval_checks(dataloader, results_dict, split, cfg, finetune_cfg):
    """
    Validate evaluation results consistency.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The dataloader used for evaluation.
    results_dict : dict
        Evaluation results to validate.
    split : str
        Data split name.
    cfg : DictConfig
        Full Hydra configuration.
    finetune_cfg : DictConfig
        Fine-tuning configuration.

    Raises
    ------
    AssertionError
        If number of predictions doesn't match input samples.
    """
    # Check if all the input samples got predicted (this is the outlier mask that came from the adjusted f1 score)
    if results_dict["preds"]["arrays"]["pred_mask"] is not None:
        no_of_input_samples = len(dataloader.dataset.tensors[0])
        no_of_pred_samples = results_dict["preds"]["arrays"]["pred_mask"].shape[0]
        assert (
            no_of_input_samples == no_of_pred_samples
        ), f"Input samples {no_of_input_samples} != Pred samples {no_of_pred_samples}"


def eval_moment_outlier_finetune(
    model,
    dataloader: torch.utils.data.DataLoader,
    split: str,
    device: str,
    criterion,
    cfg: DictConfig,
    outlier_model_cfg: DictConfig,
    finetune_cfg: DictConfig,
    tqdm_string: str = "",
    task_name: str = "outlier_detection",
):
    """
    Evaluate MOMENT model on a single data split.

    Parameters
    ----------
    model : torch.nn.Module
        MOMENT model to evaluate.
    dataloader : torch.utils.data.DataLoader
        DataLoader for the split.
    split : str
        Split name ('train', 'test', 'outlier_train', 'outlier_test').
    device : str
        Device to run on.
    criterion : torch.nn.Module
        Loss function.
    cfg : DictConfig
        Full Hydra configuration.
    outlier_model_cfg : DictConfig
        Model configuration.
    finetune_cfg : DictConfig
        Fine-tuning configuration.
    tqdm_string : str, optional
        Progress bar prefix. Default is "".
    task_name : str, optional
        Task name. Default is "outlier_detection".

    Returns
    -------
    dict
        Evaluation results containing:
        - 'average_loss': mean reconstruction loss
        - 'best_metric': best outlier detection metric
        - 'results_dict': detailed results with arrays and metrics
    """
    (
        trues,
        preds,
        losses,
        labels_out,
        trues_valid,
        preds_valid,
        labels_valid,
        input_masks_out,
    ) = [], [], [], [], [], [], [], []

    model.eval()
    with torch.set_grad_enabled(finetune_cfg["enable_val_grad"]):
        for i, (batch_x, labels, input_masks) in enumerate(
            (
                pbar := tqdm(
                    dataloader,
                    total=len(dataloader),
                    desc="MomentFM {}| Evaluate, split = {}".format(tqdm_string, split),
                )
            )
        ):
            with torch.autocast(
                device_type=device,
                dtype=dtype_map(finetune_cfg["torch_dtype"]),
                enabled=cfg["DEVICE"]["use_amp"],
            ):
                outputs, loss, valid_dict = momentfm_forward_pass(
                    model,
                    batch_x,
                    labels,
                    input_masks,
                    device,
                    criterion=criterion,
                    anomaly_criterion=finetune_cfg["anomaly_criterion"],
                    detect_anomalies=False,
                    task_name=task_name,
                )

            losses.append(loss.item())
            input_masks_out.append(input_masks.detach().cpu().numpy())
            trues.append(batch_x.detach().cpu().numpy())
            preds.append(outputs.reconstruction.detach().cpu().numpy())
            labels_out.append(labels.detach().cpu().numpy())
            trues_valid += list(valid_dict["valid_x"].detach().cpu().numpy())
            preds_valid += list(valid_dict["valid_recon"].detach().cpu().numpy())
            labels_valid += list(valid_dict["valid_labels"].detach().cpu().numpy())

            pbar.set_description(
                f"{tqdm_string}Eval {split} loss {np.mean(losses):.5f}"
            )

    losses = np.array(losses)
    average_loss = np.average(losses)

    split_results = {}
    split_results["arrays"] = {
        "trues": list_of_arrays_to_array(
            list_of_arrays=trues
        ),  # e.g. (1416, 512) (no_samples*windows, time_points)
        "preds": list_of_arrays_to_array(list_of_arrays=preds),
        "labels": list_of_arrays_to_array(list_of_arrays=labels_out),
    }
    split_results["arrays_flat"] = {
        "trues_valid": np.array(
            trues_valid
        ),  # e.g. (701274) flattened out with the padding NaNs removed
        "preds_valid": np.array(preds_valid),
        "labels_valid": np.array(labels_valid),
    }

    # Get the metrics and preds
    if "outlier" in split:
        # "pred_mask" "is boolean flagging the detected outliers (True) and inliers (False)
        metrics, preds = metrics_per_split_v2(
            split_results, split, cfg, outlier_model_cfg
        )
        # Get the best metric, i.e. the metric you can track to see if your model is improving
        best_metric = get_best_outlier_metric(
            scalar_metrics=metrics["scalars"], outlier_model_cfg=outlier_model_cfg
        )
    else:
        metrics, preds = {}, {}
        metrics["scalars"] = {}
        preds["arrays"] = {}
        preds["arrays"]["pred_mask"] = None
        metrics["arrays"] = {}
        metrics["arrays_flat"] = {}
        best_metric = None

    # Return losses per batch as well
    metrics["arrays_flat"]["losses"] = losses

    results_dict = {"split_results": split_results, "metrics": metrics, "preds": preds}
    model.train()
    post_eval_checks(dataloader, results_dict, split, cfg, finetune_cfg)

    return {
        "average_loss": average_loss,
        "best_metric": best_metric,
        "results_dict": results_dict,
    }


def train_moment_outlier_finetune(
    dataloaders: dict[str, torch.utils.data.DataLoader],
    data_dict: dict,
    model,
    criterion,
    optimizer,
    scaler,
    lr_scheduler,
    outlier_model_cfg: DictConfig,
    finetune_cfg: DictConfig,
    cfg: DictConfig,
    checkpoint_path: str,
    run_name: str,
):
    """
    Train MOMENT model for outlier detection via fine-tuning.

    Implements the training loop with gradient scaling, learning rate
    scheduling, and model checkpointing.

    Parameters
    ----------
    dataloaders : dict[str, torch.utils.data.DataLoader]
        Dictionary of dataloaders.
    data_dict : dict
        Data dictionary.
    model : torch.nn.Module
        MOMENT model to train.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer.
    scaler : torch.amp.GradScaler
        Gradient scaler for mixed precision.
    lr_scheduler : object
        Learning rate scheduler.
    outlier_model_cfg : DictConfig
        Model configuration.
    finetune_cfg : DictConfig
        Fine-tuning configuration.
    cfg : DictConfig
        Full Hydra configuration.
    checkpoint_path : str
        Directory for saving checkpoints.
    run_name : str
        MLflow run name.

    Returns
    -------
    tuple
        A tuple containing:
        - model : torch.nn.Module
            Trained model.
        - eval_dict_out : dict
            Evaluation results for best epoch.
        - best_epoch : int
            Index of the best epoch.

    References
    ----------
    https://github.com/moment-timeseries-foundation-model/moment-research/blob/
    3ab637e413f35f2c317573c0ace280d825c558de/moment/tasks/anomaly_detection_finetune.py#L67
    """
    # https://github.com/moment-timeseries-foundation-model/moment-research/blob/
    #    3ab637e413f35f2c317573c0ace280d825c558de/moment/tasks/anomaly_detection_finetune.py#L67

    # Unpack from dict
    train_dataloader = dataloaders["train"]
    device = cfg["DEVICE"]["device"]

    opt_steps = 0
    cur_epoch = 0
    best_validation_loss = np.inf
    best_validation_metric = -np.inf
    best_epoch = 0
    losses = []
    first_save = True
    checkpoint_file = None
    eval_dict_out = None

    # Evaluate the models before training
    _ = evaluate_model(
        dataloaders=dataloaders,
        data_dict=data_dict,
        model=model,
        device=device,
        criterion=criterion,
        finetune_cfg=finetune_cfg,
        outlier_model_cfg=outlier_model_cfg,
        cfg=cfg,
        cur_epoch=0,
        tqdm_string="Finetune ",
    )

    while cur_epoch < finetune_cfg["max_epoch"]:  # Epoch based learning only
        model.train()

        for i, (batch_x, labels, input_masks) in enumerate(
            (
                pbar := tqdm(
                    train_dataloader,
                    total=len(train_dataloader),
                    desc="MomentFM Finetuning | Train",
                )
            )
        ):
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device,
                dtype=dtype_map(finetune_cfg["torch_dtype"]),
                enabled=cfg["DEVICE"]["use_amp"],
            ):
                outputs, loss, valid_dict = momentfm_forward_pass(
                    model,
                    batch_x,
                    labels,  # same as batch_masks
                    input_masks,
                    device,
                    criterion=criterion,
                    anomaly_criterion=finetune_cfg["anomaly_criterion"],
                )

            if cfg["EXPERIMENT"]["debug"]:
                if opt_steps >= 1:
                    debug_model_outputs(model, loss, outputs, batch_x)

            losses.append(loss.item())
            pbar.set_description(
                f"{cur_epoch + 1}/{finetune_cfg['max_epoch']} TRAINING loss {np.mean(losses):.5f}"
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), finetune_cfg["max_norm"])
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()
            opt_steps += 1

            # Adjust learning rate
            if finetune_cfg["lr_scheduler_type"] == "linearwarmupcosinelr":
                lr_scheduler.step(cur_epoch=cur_epoch, cur_step=opt_steps)
            elif (
                finetune_cfg["lr_scheduler_type"] == "onecyclelr"
            ):  # Should be torch schedulers in general
                lr_scheduler.step()

        cur_epoch += 1

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
            tqdm_string="Finetune ",
        )

        # Check if the model has improved
        improved_loss, best_validation_loss, best_validation_metric = check_if_improved(
            eval_dicts=eval_dicts,
            cur_epoch=cur_epoch,
            loss=loss,
            best_validation_loss=best_validation_loss,
            best_validation_metric=best_validation_metric,
            cfg=cfg,
            finetune_cfg=finetune_cfg,
        )

        if improved_loss:
            best_epoch = cur_epoch
            checkpoint_file = save_model_to_disk(
                model,
                optimizer,
                scaler,
                checkpoint_path,
                cfg,
                best_epoch,
                device,
                first_save,
                run_name=run_name,
            )
            first_save = False  # you can do some debugging stuff on first save, but not on every model save
            eval_dict_out = {best_epoch: eval_dicts}

    # end of finetuning, log the best model to mlflow
    if checkpoint_file is not None:
        log_anomaly_model_as_mlflow_artifact(checkpoint_file, run_name)
    else:
        logger.error("No checkpoint file found, something went wrong!")
        raise ValueError("No checkpoint file found, something went wrong!")

    return model, eval_dict_out, best_epoch


def init_momentfm_finetune(
    model,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    cfg: DictConfig,
    outlier_model_cfg: DictConfig,
    finetune_cfg: DictConfig,
    run_name: str,
):
    """
    Initialize MOMENT fine-tuning components.

    Sets up criterion, optimizer, gradient scaler, and learning rate scheduler.

    Parameters
    ----------
    model : torch.nn.Module
        MOMENT model to prepare for training.
    dataloaders : dict[str, torch.utils.data.DataLoader]
        Dictionary of dataloaders.
    cfg : DictConfig
        Full Hydra configuration.
    outlier_model_cfg : DictConfig
        Model configuration.
    finetune_cfg : DictConfig
        Fine-tuning configuration.
    run_name : str
        MLflow run name.

    Returns
    -------
    tuple
        A tuple containing:
        - model : torch.nn.Module
            Prepared model.
        - criterion : torch.nn.Module
            Loss function.
        - optimizer : torch.optim.Optimizer
            Optimizer.
        - scaler : torch.amp.GradScaler
            Gradient scaler.
        - lr_scheduler : object
            Learning rate scheduler.
        - checkpoint_path : str
            Directory for checkpoints.

    Raises
    ------
    ValueError
        If model head is frozen or head parameters don't require grad.
    """
    # Where do artifacts go
    checkpoint_path = get_artifacts_dir(
        service_name="outlier_detection", run_name=run_name
    )
    logger.info(f"Checkpoint path: {checkpoint_path}")

    # Loss function, MSE, MAE, Huber, sMAPE
    criterion = select_criterion(
        loss_type=outlier_model_cfg["LINEAR_PROBING"]["loss_type"], reduction="mean"
    )
    logger.info("Using loss: {}".format(criterion))

    # Optimizer
    kwargs = {
        "init_lr": finetune_cfg["init_lr"],
        "weight_decay": finetune_cfg["weight_decay"],
        "optimizer_name": finetune_cfg["optimizer_name"],
    }
    optimizer = select_optimizer(model, **kwargs)
    logger.info("Using optimizer: {}".format(finetune_cfg["optimizer_name"]))
    logger.info(optimizer)

    # Gradient scaler (for CUDA)
    scaler = torch.amp.GradScaler(
        device=cfg["DEVICE"]["device"], enabled=cfg["DEVICE"]["use_amp"]
    )

    # Learning rate scheduler
    kwargs = {
        "type": finetune_cfg["lr_scheduler_type"],
        "max_epoch": finetune_cfg["max_epoch"],
        "min_lr": finetune_cfg["min_lr"],
        "init_lr": finetune_cfg["init_lr"],
        "decay_rate": finetune_cfg["decay_rate"],
        "warmup_start_lr": finetune_cfg["warmup_lr"],
        "warmup_steps": finetune_cfg["warmup_steps"],
        "train_dataloader": dataloaders["train"],
        "pct_start": finetune_cfg["pct_start"],
    }
    lr_scheduler = init_lr_scheduler(optimizer, **kwargs)
    logger.info("Using LR Scheduler: {}".format(finetune_cfg["lr_scheduler_type"]))
    logger.info(lr_scheduler)

    # Check the freezing staus
    logger.info(f"Encoder frozen: {model.freeze_encoder}")
    logger.info(f"Embedder frozen: {model.freeze_embedder}")
    logger.info(f"Head frozen: {model.freeze_head}")
    if model.freeze_head:
        logger.error(
            "Model head is frozen, no point in trying to train the model, fix something!"
        )
        raise ValueError(
            "Model head is frozen, no point in trying to train the model, fix something!"
        )

    for param in model.head.parameters():
        if not param.requires_grad:
            logger.error(
                "Model head parameters are not set to require grad, fix something!"
            )
            raise ValueError(
                "Model head parameters are not set to require grad, fix something!"
            )

    return model, criterion, optimizer, scaler, lr_scheduler, checkpoint_path


def momentfm_outlier_finetune(
    model,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    data_dict: dict,
    cfg: DictConfig,
    outlier_model_cfg: DictConfig,
    run_name: str,
):
    """
    Fine-tune MOMENT model for outlier detection.

    Main entry point for fine-tuning. Initializes training components
    and runs the training loop.

    Parameters
    ----------
    model : torch.nn.Module
        MOMENT model to fine-tune.
    dataloaders : dict[str, torch.utils.data.DataLoader]
        Dictionary of dataloaders.
    data_dict : dict
        Data dictionary.
    cfg : DictConfig
        Full Hydra configuration.
    outlier_model_cfg : DictConfig
        Model configuration.
    run_name : str
        MLflow run name.

    Returns
    -------
    tuple
        A tuple containing:
        - model : torch.nn.Module
            Fine-tuned model.
        - eval_dicts : dict
            Evaluation results.
        - best_epoch : int
            Index of the best epoch.
    """
    # Init training
    finetune_cfg = outlier_model_cfg["LINEAR_PROBING"]
    model, criterion, optimizer, scaler, lr_scheduler, checkpoint_path = (
        init_momentfm_finetune(
            model,
            dataloaders=dataloaders,
            cfg=cfg,
            outlier_model_cfg=outlier_model_cfg,
            finetune_cfg=finetune_cfg,
            run_name=run_name,
        )
    )

    # Train and evaluate
    model, eval_dicts, best_epoch = train_moment_outlier_finetune(
        dataloaders=dataloaders,
        data_dict=data_dict,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        lr_scheduler=lr_scheduler,
        outlier_model_cfg=outlier_model_cfg,
        finetune_cfg=finetune_cfg,
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        run_name=run_name,
    )

    return model, eval_dicts, best_epoch
