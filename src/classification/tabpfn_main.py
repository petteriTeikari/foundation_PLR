import warnings
from typing import Any, Dict, List, Tuple

import mlflow
import numpy as np
import polars as pl
import torch
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import auc, roc_curve

from src.classification.classifier_log_utils import (
    classifier_log_cls_evaluation_to_mlflow,
    log_classifier_sources_as_params,
)
from src.classification.tabpfn import TabPFNClassifier
from src.classification.weighing_utils import return_weights_as_dict
from src.classification.xgboost_cls.xgboost_utils import data_transform_wrapper
from src.orchestration.tabm_hyperparams import (
    pick_the_best_hyperparam_metrics,
)


def eval_tabpfn_model(
    model: Any, dict_arrays: Dict[str, np.ndarray]
) -> Tuple[Dict[str, Dict[str, np.ndarray]], float]:
    """
    Evaluate TabPFN model on all available splits.

    Computes predictions and optionally AUROC for baseline evaluation.

    Parameters
    ----------
    model : TabPFNClassifier
        Fitted TabPFN model.
    dict_arrays : dict
        Data arrays with train/test/optionally val splits.

    Returns
    -------
    tuple
        (results, auroc) where results contains predictions per split and
        auroc is test AUROC (or NaN if validation split present).
    """
    if "x_val" in dict_arrays:
        splits = ["train", "val", "test"]
    else:
        splits = ["train", "test"]

    results = {}
    auroc = {}
    for split in splits:
        results[split] = {}
        results[split]["y_pred"] = model.predict(dict_arrays[f"x_{split}"])
        results[split]["y_pred_proba"] = model.predict_proba(dict_arrays[f"x_{split}"])[
            :, 1
        ]  # class 1 probs

        if "x_val" not in dict_arrays:
            # only for baseline model, saving some (milli)seconds when doing bootstrap
            fpr, tpr, thresholds = roc_curve(
                dict_arrays[f"y_{split}"], results[split]["y_pred"]
            )
            auroc[split] = auc(fpr, tpr)

    if "x_val" not in dict_arrays:
        logger.info(
            "TabPFN Baseline | Test AUROC: {:.3f}, Train AUROC: {:.3f}, GAP: {:.3f}".format(
                auroc["test"], auroc["train"], auroc["train"] - auroc["test"]
            )
        )
        return results, auroc["test"]
    else:
        return results, np.nan


def train_and_eval_tabpfn(
    dict_arrays: Dict[str, np.ndarray], hparams: Dict[str, Any]
) -> Tuple[None, Dict[str, Dict[str, np.ndarray]], float]:
    """
    Train and evaluate TabPFN classifier.

    TabPFN is a prior-fitted network that requires no training on the
    target dataset - it uses in-context learning.

    Parameters
    ----------
    dict_arrays : dict
        Data arrays with train/test splits.
    hparams : dict
        Hyperparameters for TabPFN (currently unused for v2).

    Returns
    -------
    tuple
        (model, results, metric) where model is None (to save RAM),
        results contains predictions, and metric is test AUROC.

    References
    ----------
    TabPFN: https://github.com/automl/TabPFN
    """
    # see https://github.com/automl/TabPFN?tab=readme-ov-file#getting-started
    # # When N_ensemble_configurations > #features * #classes, no further averaging is applied.
    # 17 > 8 x 2
    warnings.simplefilter("ignore")
    if torch.cuda.is_available():
        # around 6x times faster with 2070 Super than laptop CPU, so definitely use GPU if possible
        device = "cuda"
    else:
        device = "cpu"
    mlflow.log_param("device", device)

    classifier = TabPFNClassifier(
        device=device
    )  # , **hparams) # TODO! if you want to pass hyperparams for v2
    classifier.fit(dict_arrays["x_train"], dict_arrays["y_train"])
    results, metric = eval_tabpfn_model(classifier, dict_arrays)
    warnings.resetwarnings()
    # to save RAM, do not return the model, if you want these, write them on disk
    model = None

    return model, results, metric


def tabpfn_wrapper(
    dict_arrays: Dict[str, np.ndarray],
    cls_model_cfg: DictConfig,
    hparam_cfg: DictConfig,
    cfg: DictConfig,
    run_HPO: bool = False,
) -> Tuple[None, Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
    """
    Wrapper for TabPFN training with optional hyperparameter optimization.

    Parameters
    ----------
    dict_arrays : dict
        Data arrays with train/test splits.
    cls_model_cfg : DictConfig
        TabPFN model configuration.
    hparam_cfg : DictConfig
        Hyperparameter configuration.
    cfg : DictConfig
        Full Hydra configuration.
    run_HPO : bool, default False
        Run hyperparameter optimization (not implemented for v2).

    Returns
    -------
    tuple
        (model, results, best_hparams) from training.

    Raises
    ------
    NotImplementedError
        If run_HPO is True (was for TabPFN v1).
    """
    if run_HPO:
        raise NotImplementedError("HPO was for TabFPN v1")
        # quick'n'dirty one param HPO
        # n_vector = np.linspace(1, 64, 64).astype(int)
        # hparams = []
        # for n in n_vector:
        #     hparams.append({"N_ensemble_configurations": n})
        # best_hparams = None
    else:
        hparams = [dict(cls_model_cfg["MODEL"]["HYPERPARAMS"])]
        best_hparams = hparams[0]

    best_metric = 0
    for hparam_dict in hparams:
        # n=7 is the first value to reach test AUROC of 0.831 for the pupil-gt__pupil-gt
        # TabPFN Baseline | Test AUROC: 0.831, Train AUROC: 0.925, GAP: 0.094
        # Train AUROC fluctuates between 0.925 and 0.912 and converges on large ns, when > 30
        # you could later replicate this in the bootstrap_evaluator to have a better idea? TODO!
        # logger.info(f"Training TabPFN with hyperparameters: {hparam_dict}")
        try:
            model, results, metric = train_and_eval_tabpfn(dict_arrays, hparam_dict)
        except Exception as e:
            logger.error("Failed to train TabPFN due to error: {}".format(e))
            raise e
        if metric > best_metric:
            best_metric = metric
            best_hparams = hparam_dict

    # Log the hyperparams to MLflow
    # for key, value in best_hparams.items():
    #     mlflow.log_param('hparam'+key, value)

    return model, results, best_hparams


def tabpfn_main(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    run_name: str,
    cfg: DictConfig,
    cls_model_cfg: DictConfig,
    hparam_cfg: DictConfig,
    features_per_source: Dict[str, List[str]],
) -> None:
    """
    Main entry point for TabPFN classifier training with MLflow tracking.

    TabPFN uses in-context learning and doesn't require traditional training.
    This function handles data preparation, bootstrap evaluation, and
    MLflow logging.

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
        TabPFN model configuration.
    hparam_cfg : DictConfig
        Hyperparameter configuration.
    features_per_source : dict
        Feature source metadata for logging.
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_name", "TabPFN")
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
        model_cfgs = [cls_model_cfg]

        # Define the baseline model
        # Get the baseline model
        from src.classification.bootstrap_evaluation import bootstrap_evaluator
        from src.classification.classifier_evaluation import get_the_baseline_model

        weights_dict = return_weights_as_dict(dict_arrays, cls_model_cfg)
        model, baseline_results = get_the_baseline_model(
            "TabPFN", cls_model_cfg, hparam_cfg, cfg, None, dict_arrays, weights_dict
        )

        metrics = {}
        for i, cls_model_cfg in enumerate(model_cfgs):
            # logger.info(f"{i+1}/{len(model_cfgs)}: Hyperparameter grid search")
            # e.g. Bootstrap iterations:  11%|█         | 110/1000 [01:19<07:53,
            models, metrics[i] = bootstrap_evaluator(
                model_name="TabPFN",
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
            model_name="TabPFN",
        )
