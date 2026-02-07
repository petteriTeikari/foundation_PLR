import math

import mlflow
import numpy as np
import polars as pl
import scipy.special
import sklearn.metrics
import sklearn.model_selection
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from torch import Tensor

from src.classification.classifier_log_utils import (
    classifier_log_cls_evaluation_to_mlflow,
    log_classifier_sources_as_params,
)
from src.classification.tabm.tabm_model import create_tabm_model
from src.classification.weighing_utils import return_weights_as_dict
from src.classification.xgboost_cls.xgboost_utils import data_transform_wrapper
from src.orchestration.tabm_hyperparams import (
    create_tabm_hyperparam_experiment,
    pick_the_best_hyperparam_metrics,
)


def tabm_train_script(
    data: dict,
    device,
    cls_model_cfg: DictConfig,
    hparam_cfg: DictConfig,
    cfg: DictConfig,
    compile_model: bool = False,
    debug_verbose: bool = True,
):
    """
    Train a TabM model for tabular classification.

    TabM is a modern deep learning architecture for tabular data that
    uses k predictions per object for improved uncertainty estimation.

    Parameters
    ----------
    data : dict
        Data dictionary with 'train', 'test', optionally 'val' splits,
        each containing 'x_cont', optionally 'x_cat', and 'y' tensors.
    device : torch.device
        Device to train on (cuda or cpu).
    cls_model_cfg : DictConfig
        Model configuration with arch_type, d_block, dropout, etc.
    hparam_cfg : DictConfig
        Hyperparameter configuration with metric_val.
    cfg : DictConfig
        Full Hydra configuration.
    compile_model : bool, default False
        Use torch.compile for faster training.
    debug_verbose : bool, default True
        Log training progress.

    Returns
    -------
    tuple
        (model, results) where model is trained TabM and results contains
        scores and predictions per split.

    References
    ----------
    TabM: https://github.com/yandex-research/tabm
    Paper: https://arxiv.org/abs/2410.24210
    """
    # TaskType = Literal["regression", "binclass", "multiclass"]
    # task_type: TaskType = "binclass"
    score_eval = hparam_cfg["HYPERPARAMS"]["metric_val"]

    # Automatic mixed precision (AMP)
    # torch.float16 is implemented for completeness,
    # but it was not tested in the project,
    # so torch.bfloat16 is used by default.
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
        if torch.cuda.is_available()
        else None
    )

    # Changing False to True will result in faster training on compatible hardware.
    amp_enabled = False and amp_dtype is not None
    grad_scaler = torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None  # type: ignore

    # Create the model
    model, optimizer, evaluation_mode = create_tabm_model(
        device,
        arch_type=cls_model_cfg["arch_type"],
        d_block=cls_model_cfg["d_block"],
        dropout=cls_model_cfg["dropout"],
        d_embedding=cls_model_cfg["d_embedding"],
        k=cls_model_cfg["k"],
        lr=cls_model_cfg["lr"],
        weight_decay=cls_model_cfg["weight_decay"],
        data=data,
        compile_model=compile_model,
    )

    @torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
    def apply_model(split: str, idx: Tensor) -> Tensor:
        return (
            model(
                data[split]["x_cont"][idx],
                data[split]["x_cat"][idx] if "x_cat" in data[split] else None,
            )
            .squeeze(-1)  # Remove the last dimension for regression tasks.
            .float()
        )

    base_loss_fn = F.cross_entropy

    def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
        # TabM produces k predictions per object. Each of them must be trained separately.
        # (classification) y_pred.shape == (batch_size, k, n_classes)
        k = y_pred.shape[-2]
        return base_loss_fn(y_pred.flatten(0, 1), y_true.repeat_interleave(k))

    @evaluation_mode()
    def evaluate(split: str) -> float:
        model.eval()

        # When using torch.compile, you may need to reduce the evaluation batch size.
        eval_batch_size = 8096
        y_pred: np.ndarray = (
            torch.cat(
                [
                    apply_model(split, idx)
                    for idx in torch.arange(len(data[split]["y"]), device=device).split(
                        eval_batch_size
                    )
                ]
            )
            .cpu()
            .numpy()
        )
        # For classification, the mean must be computed in the probability space.
        y_pred = scipy.special.softmax(
            y_pred, axis=-1
        )  # e.g. (145, 32, 2) (batch_size, k, n_classes)
        y_pred = y_pred.mean(1)

        y_true = data[split]["y"].cpu().numpy()
        if score_eval == "auc":
            y_probs_class1 = y_pred[:, 1]
            score = roc_auc_score(y_true, y_probs_class1)
        elif score_eval == "f1":
            y_pred_argmax = np.argmax(y_pred, axis=1)
            score = sklearn.metrics.f1_score(y_true, y_pred_argmax)
        else:
            logger.error(f"Unknown score evaluation type: {score_eval}")
            raise ValueError(f"Unknown score evaluation type: {score_eval}")

        return float(score), y_pred  # The higher -- the better.

    def when_tabm_model_improved(val_score, val_pred):
        # No need to do on every epoch, only when things improve
        train_score, train_pred = evaluate("train")
        test_score, test_pred = evaluate("test")
        results = {
            "test": {"score": test_score, "pred": test_pred},
            "train": {"score": train_score, "pred": train_pred},
            "val": {"score": val_score, "pred": val_pred},
        }
        return results

    # Training loop
    n_epochs = cls_model_cfg["n_epochs"]
    patience = cls_model_cfg["patience"]
    batch_size = cls_model_cfg["batch_size"]

    # epoch_size = math.ceil(len(data["train"]["y"]) / batch_size)  # How many batches per epoch
    results = {"val": {"score": -math.inf}}
    best_epoch = -1
    # Early stopping: the training stops when
    # there are more than `patience` consequtive bad updates.
    remaining_patience = patience

    for epoch in range(n_epochs):
        for batch_idx in torch.randperm(len(data["train"]["y"]), device=device).split(
            batch_size
        ):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(
                apply_model("train", batch_idx), data["train"]["y"][batch_idx]
            )
            if grad_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                grad_scaler.scale(loss).backward()  # type: ignore
                grad_scaler.step(optimizer)
                grad_scaler.update()

        if "val" in data:
            # when bootstrapping, we have "val" here
            val_score, val_pred = evaluate("val")
        else:
            # for example when getting the baseline model, we don't have "val"
            val_score, val_pred = evaluate("test")

        if val_score > results["val"]["score"]:
            results = when_tabm_model_improved(val_score, val_pred)
            best_epoch = epoch
            remaining_patience = patience
        else:
            remaining_patience -= 1

        if remaining_patience < 0:
            break

    if debug_verbose:
        logger.info(
            f"Best epoch: {best_epoch}, val score = {results['val']['score']:.4f}, "
            f"test score: {results['test']['score']:.4f}, "
            f"train score: {results['train']['score']:.4f}"
        )

    return model, results


def tabm_main(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    run_name: str,
    cfg: DictConfig,
    cls_model_cfg: DictConfig,
    hparam_cfg: DictConfig,
    features_per_source: dict,
):
    """
    Main entry point for TabM classifier training with MLflow tracking.

    Converts data, performs optional hyperparameter search, trains TabM
    with bootstrap evaluation, and logs results to MLflow.

    Parameters
    ----------
    train_df : pl.DataFrame
        Training data as Polars DataFrame.
    test_df : pl.DataFrame
        Test data as Polars DataFrame.
    run_name : str
        MLflow run name.
    cfg : DictConfig
        Full Hydra configuration.
    cls_model_cfg : DictConfig
        TabM model configuration.
    hparam_cfg : DictConfig
        Hyperparameter configuration.
    features_per_source : dict
        Feature source metadata for logging.

    References
    ----------
    TabM: https://github.com/yandex-research/tabm
    Paper: https://arxiv.org/abs/2410.24210
    """
    from src.classification.bootstrap_evaluation import bootstrap_evaluator

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_name", "TabM")
        for k, v in cls_model_cfg.items():
            if k != "MODEL":
                mlflow.log_param(k, v)

        _, _, dict_arrays = data_transform_wrapper(
            train_df, test_df, cls_model_cfg, None
        )
        log_classifier_sources_as_params(
            features_per_source, dict_arrays, run_name, cfg
        )

        # Define hyperparameter search space
        model_cfgs = create_tabm_hyperparam_experiment(
            run_name, hparam_cfg, cls_model_cfg
        )

        # Define the baseline model
        # Get the baseline model
        from src.classification.classifier_evaluation import get_the_baseline_model

        weights_dict = return_weights_as_dict(dict_arrays, cls_model_cfg)
        model, baseline_results = get_the_baseline_model(
            "TabM", cls_model_cfg, hparam_cfg, cfg, None, dict_arrays, weights_dict
        )

        metrics = {}
        for i, cls_model_cfg in enumerate(model_cfgs):
            logger.info(f"{i + 1}/{len(model_cfgs)}: Hyperparameter grid search")
            # e.g. Bootstrap iterations:  11%|█         | 110/1000 [01:19<07:53,
            models, metrics[i] = bootstrap_evaluator(
                model_name="TabM",
                run_name=run_name,
                dict_arrays=dict_arrays,
                best_params=None,
                cls_model_cfg=cls_model_cfg,
                method_cfg=cfg["CLS_EVALUATION"]["BOOTSTRAP"],
                hparam_cfg=hparam_cfg,
                cfg=cfg,
            )

        if len(metrics) == 1:
            metrics = metrics[0]
        else:
            logger.debug("Get the best set of hyperparameters")
            metrics, best_choice = pick_the_best_hyperparam_metrics(
                metrics, hparam_cfg, model_cfgs, cfg
            )

        # Log the extra metrics to MLflow
        classifier_log_cls_evaluation_to_mlflow(
            model,
            baseline_results,
            models,
            metrics,
            dict_arrays,
            cls_model_cfg,
            run_name=run_name,
            model_name="TabM",
        )
