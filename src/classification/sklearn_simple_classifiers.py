import time

import mlflow
import numpy as np
import polars as pl
from omegaconf import DictConfig
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


from src.classification.classifier_evaluation import (
    evaluate_sklearn_classifier,
)
from src.classification.classifier_log_utils import log_classifier_sources_as_params
from src.classification.weighing_utils import (
    weights_dict_wrapper,
)
from sklearn import preprocessing
from src.classification.xgboost_cls.xgboost_utils import (
    data_transform_wrapper,
    join_test_and_train_arrays,
)
from loguru import logger


def display_grid_search_results(grid_result: GridSearchCV, scoring: str):
    means = grid_result.cv_results_["mean_test_score"]
    stds = grid_result.cv_results_["std_test_score"]
    params = grid_result.cv_results_["params"]
    logger.debug("Grid search results:")
    for mean, stdev, param in zip(means, stds, params):
        logger.debug("%f (%f) with: %r" % (mean, stdev, param))

    logger.info(
        "Best %s: %f using %s"
        % (scoring, grid_result.best_score_, grid_result.best_params_)
    )

    logger.info("Log best params to MLflow")
    for key, value in grid_result.best_params_.items():
        key = f"hyperparam_{key}"
        mlflow.log_param(key, value)
        logger.debug(f"key {key}: {value}")


def standardize_features(X):
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    logger.info("Standardizing features:")
    logger.info("mean = {}".format(scaler.mean_))
    logger.info("std = {}".format(scaler.scale_))
    return X, scaler


def prepare_for_logistic_hpo(X, y, hparam_cfg):
    assert X.shape[0] == y.shape[0], "X and y must have the same number of rows"
    assert np.sum(np.isnan(X)) == 0, "X must not contain NaNs"

    # Standardize features
    X, scaler = standardize_features(X)

    # Not so many params to play with compared to XGBoost
    hparam_method = hparam_cfg["HYPERPARAMS"]["method"]
    if hparam_method is not None:
        hparams = hparam_cfg["SEARCH_SPACE"][hparam_method]
        grid = {}
        for key, value in dict(hparams).items():
            grid[key] = list(value)  # from ListConfig to list
    else:
        grid = None

    return X, y, grid


def logistic_regression_hpo_grid_search(
    X, y, weights_dict: dict, hparam_cfg: DictConfig, cls_model_cfg: DictConfig
):
    # https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
    X, y, grid = prepare_for_logistic_hpo(X, y, hparam_cfg)

    if grid is not None:
        logger.info("Grid search for Logistic Regression, params: {}".format(grid))
        model = LogisticRegression(max_iter=500)
        logger.info(
            "Cross-validation params: {}".format(hparam_cfg["HYPERPARAMS"]["cv_params"])
        )
        cv = RepeatedStratifiedKFold(**hparam_cfg["HYPERPARAMS"]["cv_params"])
        logger.info(
            "GridSearchCV params: {}".format(hparam_cfg["HYPERPARAMS"]["fit_params"])
        )
        start_time = time.time()
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid,
            cv=cv,
            verbose=2,
            **hparam_cfg["HYPERPARAMS"]["fit_params"],
        )
        grid_result = grid_search.fit(X, y)
        display_grid_search_results(
            grid_result, scoring=hparam_cfg["HYPERPARAMS"]["fit_params"]["scoring"]
        )
        logger.info("Grid search time: {:.2f} seconds".format(time.time() - start_time))
        best_params = grid_result.best_params_
    else:
        logger.info(
            "Logistic Regression without grid search, using default hyperparameters"
        )
        grid_result = None
        best_params = cls_model_cfg["MODEL"]["HYPERPARAMS_DEFAULT"]

    return grid_result, best_params


def logistic_regression(
    model_name,
    dict_arrays,
    weights_dict,
    cls_model_cfg,
    hparam_cfg,
    cfg,
    run_name: str,
    features_per_source: dict,
    join_test_and_train: bool = True,
):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(hparam_cfg)

        # Log the source params
        log_classifier_sources_as_params(
            features_per_source, dict_arrays, run_name, cfg
        )
        mlflow.log_param("model_name", "LogisticRegression")

        # https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html
        if join_test_and_train:
            X, y, X_weights = join_test_and_train_arrays(dict_arrays)
        else:
            X = dict_arrays["X_train"]
            y = dict_arrays["y_train"]
            # X_weights = weights_dict["X_train_w"]

        # Find best params with a grid search
        grid_result, best_params = logistic_regression_hpo_grid_search(
            X, y, weights_dict, hparam_cfg=hparam_cfg, cls_model_cfg=cls_model_cfg
        )

        # Evaluate the model performance
        models, metrics = evaluate_sklearn_classifier(
            model_name,
            dict_arrays,
            best_params=best_params,
            cls_model_cfg=cls_model_cfg,
            eval_cfg=cfg["CLS_EVALUATION"],
            cfg=cfg,
            run_name=run_name,
        )

        logger.debug("Logistic Regression model evaluation:", metrics, models)
        mlflow.end_run()


def sklearn_simple_cls_main(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    model_name: str,
    cfg: DictConfig,
    cls_model_cfg: DictConfig,
    hparam_cfg: DictConfig,
    run_name: str,
    features_per_source: dict,
):
    # Convert Polars DataFrames to arrays
    _, _, dict_arrays = data_transform_wrapper(train_df, test_df, cls_model_cfg, None)

    # Get weights for the sklearn classifiers if you want to use these
    weights_dict = weights_dict_wrapper(dict_arrays, cls_model_cfg)

    if model_name == "LogisticRegression":
        logistic_regression(
            model_name,
            dict_arrays,
            weights_dict,
            cls_model_cfg,
            hparam_cfg,
            cfg,
            run_name,
            features_per_source,
        )

    else:
        logger.error(f"Unknown classifier model: {model_name}")
        raise ValueError(f"Unknown classifier model: {model_name}")
