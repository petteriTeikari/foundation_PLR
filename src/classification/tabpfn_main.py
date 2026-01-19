import warnings

import numpy as np
import polars as pl
import torch
from omegaconf import DictConfig
from loguru import logger
import mlflow
from sklearn.metrics import roc_curve, auc

from src.classification.tabpfn import TabPFNClassifier

from src.classification.classifier_log_utils import (
    classifier_log_cls_evaluation_to_mlflow,
    log_classifier_sources_as_params,
)

from src.classification.weighing_utils import return_weights_as_dict
from src.classification.xgboost_cls.xgboost_utils import data_transform_wrapper
from src.orchestration.tabm_hyperparams import (
    pick_the_best_hyperparam_metrics,
)


def eval_tabpfn_model(model, dict_arrays):
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


def train_and_eval_tabpfn(dict_arrays: dict, hparams: dict):
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
    dict_arrays: dict,
    cls_model_cfg: DictConfig,
    hparam_cfg: DictConfig,
    cfg: DictConfig,
    run_HPO: bool = False,
):
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
    features_per_source: dict,
):
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
        from src.classification.classifier_evaluation import get_the_baseline_model
        from src.classification.bootstrap_evaluation import bootstrap_evaluator

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
