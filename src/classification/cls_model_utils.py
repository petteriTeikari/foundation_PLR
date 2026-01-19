import torch
from loguru import logger
import numpy as np
from omegaconf import open_dict

from src.classification.tabm.tabm_main import tabm_train_script
from src.classification.tabm.tabm_utils import (
    transform_data_to_tabm_from_dict_arrays,
)
from src.classification.tabpfn_main import tabpfn_wrapper

from src.classification.xgboost_cls.xgboost_main import xgboost_train_script
from sklearn.linear_model import LogisticRegression
from src.log_helpers.log_naming_uris_and_dirs import (
    get_eval_metric_name,
    get_train_loss_name,
)


def logistic_regression_classifier(
    best_params: dict, X: np.ndarray, y: np.ndarray, weights_dict: dict = None
):
    model = LogisticRegression(**best_params, max_iter=500)
    model = model.fit(X, y)
    return model


def bootstrap_model_selector(
    model_name, cls_model_cfg, hparam_cfg, cfg, best_params, dict_arrays, weights_dict
):
    # You can use these in production if you like, define a class for an sklearn ensemble
    if model_name == "LogisticRegression":
        model = logistic_regression_classifier(
            best_params,
            X=dict_arrays["x_train"],
            y=dict_arrays["y_train"],
            weights_dict=weights_dict,
        )
        results = None
    elif model_name == "XGBOOST":
        model, _ = xgboost_train_script(
            best_params,
            dict_arrays,
            cls_model_cfg,
            use_RFE=False,
            verbose=False,
            weights_dict=weights_dict,
            eval_metric=get_eval_metric_name(model_name, cfg),
        )
        results = None
    elif model_name == "CATBOOST":
        from src.classification.catboost.catboost_main import (
            catboost_ensemble_wrapper,
            create_data_pools,
        )

        train, val, test, y_test = create_data_pools(dict_arrays, cls_model_cfg)
        eval_metric = get_eval_metric_name(cls_model_name="CATBOOST", cfg=cfg)
        loss_function = get_train_loss_name(cfg)
        with open_dict(cls_model_cfg):
            # Update the bootstrap esize (you could do 100 submodels for sure with bootstrap as well)
            cls_model_cfg["esize"] = cls_model_cfg["MODEL"]["CI"]["BOOTSTRAP"]["esize"]
        model, results = catboost_ensemble_wrapper(
            dict_arrays,
            train,
            test,
            best_params,
            cls_model_cfg,
            hparam_cfg,
            cfg,
            loss_function,
            eval_metric,
            val=val,
            rearrange_results=False,
            verbose=False,
        )
        # results = rearrange_the_split(results)
    elif model_name == "TabM":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data = transform_data_to_tabm_from_dict_arrays(dict_arrays, device)
        model, results = tabm_train_script(
            data=data,
            device=device,
            cls_model_cfg=cls_model_cfg,
            hparam_cfg=hparam_cfg,
            cfg=cfg,
        )
        # Note! We are returning now the results as TabM does not have sklearn-type predict() which the
        #  downstream code expects to come here for the returned model
    elif model_name == "TabPFN":
        model, results, best_hparams = tabpfn_wrapper(
            dict_arrays=dict_arrays,
            cls_model_cfg=cls_model_cfg,
            hparam_cfg=hparam_cfg,
            cfg=cfg,
        )

    else:
        logger.error(f"Unknown classifier model: {model_name}")
        raise ValueError(f"Unknown classifier model: {model_name}")

    return model, results
