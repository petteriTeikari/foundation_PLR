import mlflow
from omegaconf import DictConfig
from loguru import logger
from src.classification.bootstrap_evaluation import bootstrap_evaluator
from src.classification.classifier_log_utils import (
    classifier_log_cls_evaluation_to_mlflow,
)
from src.classification.cls_model_utils import bootstrap_model_selector
from src.classification.weighing_utils import return_weights_as_dict
from src.stats.calibration_metrics import get_calibration_metrics
from src.stats.classifier_metrics import get_classifier_metrics


def get_preds(model, dict_arrays):
    preds = {}
    for split in ["train", "test"]:
        X = dict_arrays[f"x_{split}"]
        predict_probs = model.predict_proba(X)  # (n_samples, n_classes), e.g. (72,2)
        preds[split] = {
            "y_pred_proba": predict_probs[
                :, 1
            ],  # (n_samples,) e.g. (72,) for the class 1 (e.g. glaucoma)
            "y_pred": model.predict(X),  # (n_samples,), e.g. (72,)
            "label": dict_arrays[f"y_{split}"],
        }

    return preds


def arrange_to_match_bootstrap_results(preds, metrics_dict):
    for split, preds_dict in preds.items():
        metrics_dict[split]["preds"] = {}
        metrics_dict[split]["preds"]["arrays"] = {}
        metrics_dict[split]["preds"]["arrays"]["predictions"] = preds_dict

    baseline_results = {"metrics": metrics_dict}
    return baseline_results


def eval_sklearn_baseline_results(model, dict_arrays, cfg):
    metrics_dict = {}
    preds = get_preds(model, dict_arrays)
    for split in preds.keys():
        metrics_dict[split] = get_classifier_metrics(
            y_true=dict_arrays[f"y_{split}"], preds=preds[split], cfg=cfg
        )
        metrics_dict[split] = get_calibration_metrics(
            model,
            metrics_dict[split],
            y_true=dict_arrays[f"y_{split}"],
            preds=preds[split],
        )

    baseline_results = arrange_to_match_bootstrap_results(preds, metrics_dict)
    return baseline_results


def get_the_baseline_model(
    model_name: str,
    cls_model_cfg: DictConfig,
    hparam_cfg: DictConfig,
    cfg: DictConfig,
    best_params: dict,
    dict_arrays: dict,
    weights_dict: dict,
):
    model, baseline_results = bootstrap_model_selector(
        model_name=model_name,
        cls_model_cfg=cls_model_cfg,
        hparam_cfg=hparam_cfg,
        cfg=cfg,
        best_params=best_params,
        dict_arrays=dict_arrays,
        weights_dict=weights_dict,
    )

    if baseline_results is None:
        # sklearn models do not return results, so you need to do the predict and compute the metrics
        baseline_results = eval_sklearn_baseline_results(model, dict_arrays, cfg)
        if model_name == "TabM":
            # due to how TabM was implemented, we got this twice if we did not have a validation split
            baseline_results.pop("val")

    return model, baseline_results


def evaluate_sklearn_classifier(
    model_name: str,
    dict_arrays: dict,
    best_params,
    cls_model_cfg: DictConfig,
    eval_cfg: DictConfig,
    cfg: DictConfig,
    run_name: str,
):
    eval_method = eval_cfg["method"]
    method_cfg = eval_cfg[eval_method]
    hparam_cfg = cfg["CLS_HYPERPARAMS"][model_name]

    # Get the baseline model
    weights_dict = return_weights_as_dict(dict_arrays, cls_model_cfg)
    model, baseline_results = get_the_baseline_model(
        model_name,
        cls_model_cfg,
        hparam_cfg,
        cfg,
        best_params,
        dict_arrays,
        weights_dict,
    )
    logger.info(
        f"Baseline Test AUROC = {baseline_results['metrics']['test']['metrics']['scalars']['AUROC']:.2f}"
    )

    # Bootstrap
    if eval_method == "BOOTSTRAP":
        models, metrics = bootstrap_evaluator(
            model_name,
            run_name,
            dict_arrays,
            best_params,
            cls_model_cfg,
            method_cfg=method_cfg,
            hparam_cfg=hparam_cfg,
            cfg=cfg,
        )  # ~30sec
    else:
        logger.error(f"Unknown evaluation method: {eval_cfg['method']}")
        raise ValueError(f"Unknown evaluation method: {eval_cfg['method']}")
        # - Conformal Prediction for classifier (e.g. https://github.com/donlnz/nonconformist)
        # - https://arxiv.org/abs/2404.19472v1

    # Log best params
    for best_param, best_value in best_params.items():
        if best_value is None:
            mlflow.log_param("hparam_" + best_param, "None")
        else:
            mlflow.log_param("hparam_" + best_param, best_value)

    # Log the extra metrics to MLflow
    classifier_log_cls_evaluation_to_mlflow(
        model,
        baseline_results,
        models,
        metrics,
        dict_arrays,
        cls_model_cfg,
        run_name=run_name,
        model_name=model_name,
    )

    return models, metrics
