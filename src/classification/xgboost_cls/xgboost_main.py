import time
from copy import deepcopy

import mlflow
import polars as pl
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, tpe
from loguru import logger
from omegaconf import DictConfig

from src.classification.classifier_log_utils import (
    log_classifier_params_to_mlflow,
    log_classifier_sources_as_params,
)
from src.classification.classifier_utils import classifier_hpo_eval, preprocess_features
from src.classification.feature_selection import rfe_feature_selector
from src.classification.hyperopt_utils import parse_hyperopt_search_space
from src.classification.viz_classifiers import classifier_feature_importance
from src.classification.weighing_utils import get_weights_for_xgboost_fit
from src.classification.xgboost_cls.xgboost_grid import define_xgboost_grid_search_space
from src.classification.xgboost_cls.xgboost_utils import (
    data_transform_wrapper,
    find_best_metric,
    get_last_items_from_OrderedDicts,
)
from src.log_helpers.log_naming_uris_and_dirs import (
    get_eval_metric_name,
    get_train_loss_name,
)


def xgboost_hyperparameter_tuning(
    space: dict,
    dict_arrays: dict,
    loss_function: str,
    eval_metric: str,
    xgboost_cfg: DictConfig,
    hparam_cfg: DictConfig,
    cfg: DictConfig,
):
    """
    Perform Bayesian hyperparameter optimization for XGBoost using Hyperopt.

    Parameters
    ----------
    space : dict
        Hyperopt search space defining parameter distributions.
    dict_arrays : dict
        Dictionary containing training and test arrays.
    loss_function : str
        Loss function name for training.
    eval_metric : str
        Evaluation metric name for optimization.
    xgboost_cfg : DictConfig
        XGBoost configuration dictionary.
    hparam_cfg : DictConfig
        Hyperparameter configuration dictionary.
    cfg : DictConfig
        Main configuration dictionary.

    Returns
    -------
    dict
        Best hyperparameters found combined with static parameters.
    """

    # The function to be optimized by hyperopt
    def hyperparameter_tuning(space: dict):
        # https://medium.com/analytics-vidhya/hyperparameter-tuning-hyperopt-bayesian-optimization-for-xgboost-and-neural-network-8aedf278a1c9
        model = xgb.XGBClassifier(**space)

        # Evaluation sets
        evaluation = [
            (dict_arrays["x_train"], dict_arrays["y_train"]),
            (dict_arrays["x_test"], dict_arrays["y_test"]),
        ]

        # If you want to weigh something, features, samples or classes
        (
            sample_weight,
            sample_weight_eval_set,
            feature_weights,
            scale_pos_weight,
            norm_stats,
        ) = get_weights_for_xgboost_fit(dict_arrays, xgboost_cfg)

        # Fit the model
        model.fit(
            dict_arrays["x_train"],
            dict_arrays["y_train"],
            eval_set=evaluation,
            sample_weight=sample_weight,
            sample_weight_eval_set=sample_weight_eval_set,
            verbose=False,
        )

        # Visualize?
        # https://www.kaggle.com/code/ahmedalbaz/xgboost-hyperopt-in-credit-card-fraud-detection?scriptVersionId=29142951&cellId=21

        # What metric for evaluation?
        predict_proba = model.predict_proba(dict_arrays["x_test"])[:, 1]
        loss = classifier_hpo_eval(
            y_true=dict_arrays["y_test"],
            pred_proba=predict_proba,
            eval_metric=eval_metric,
            model="XGBOOST",
            hpo_method="hyperopt",
        )

        return {"loss": loss, "status": STATUS_OK, "model": model}

    # Run the optimization
    trials = Trials()
    logger.info("Starting hyperparameter tuning")
    start = time.time()
    best = fmin(
        fn=hyperparameter_tuning,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        show_progressbar=False,  # hparam_cfg["HYPERPARAMS"]["hyperopt_show_progressbar"],
        trials=trials,
        verbose=False,  # hparam_cfg["HYPERPARAMS"]["hyperopt_verbose"],
    )
    tuning_time = time.time() - start
    logger.info(f"Hyperparameter tuning Time: {tuning_time:.2f} seconds")
    logger.info(f"Best hyperparameters: {best}")

    # The "best" does not contain anymore the static parameters
    logger.info("Combine best hyperparameters with the static parameters")
    static_params = parse_hyperopt_search_space(
        hyperopt_cfg=hparam_cfg["SEARCH_SPACE"]["HYPEROPT"], return_only_static=True
    )
    model_params = {**best, **static_params}

    return model_params


def xgboost_train_script(
    model_params: dict,
    dict_arrays: dict,
    xgboost_cfg: DictConfig,
    weights_dict: dict = None,
    use_RFE: bool = True,
    verbose: bool = True,
    debug_mode: bool = False,
    eval_metric: str = None,
):
    """
    Train an XGBoost classifier with optional RFE feature selection.

    Parameters
    ----------
    model_params : dict
        Hyperparameters for the XGBoost model.
    dict_arrays : dict
        Dictionary containing training and test arrays.
    xgboost_cfg : DictConfig
        XGBoost configuration dictionary.
    weights_dict : dict, optional
        Dictionary of sample weights. Default is None.
    use_RFE : bool, optional
        Whether to use Recursive Feature Elimination. Default is True.
    verbose : bool, optional
        Whether to print training progress. Default is True.
    debug_mode : bool, optional
        Whether to print debug information. Default is False.
    eval_metric : str, optional
        Evaluation metric name. Default is None.

    Returns
    -------
    model : xgb.XGBClassifier
        Trained XGBoost classifier.
    dict_arrays : dict
        Updated dictionary with potentially reduced features after RFE.
    """
    # https://xgboosting.com/xgboost-feature-selection-with-rfe/
    if verbose:
        logger.info(
            "Training the model with the best hyperparameters: {}".format(model_params)
        )
    start = time.time()
    model = xgb.XGBClassifier(
        **model_params, objective="binary:logistic", eval_metric=eval_metric
    )
    if use_RFE:
        # Drops the features from numpy arrays that are not desired
        dict_arrays = rfe_feature_selector(model, dict_arrays, xgboost_cfg)

    # If you wish to use some weights, they come from here, defined in the config .yaml
    (
        sample_weight,
        sample_weight_eval_set,
        feature_weights,
        scale_pos_weight,
        norm_stats,
    ) = get_weights_for_xgboost_fit(dict_arrays, xgboost_cfg)

    # Define the evaluation sets
    if len(sample_weight_eval_set) == 2:
        evaluation = [
            (dict_arrays["x_train"], dict_arrays["y_train"]),
            (dict_arrays["x_test"], dict_arrays["y_test"]),
        ]
    elif len(sample_weight_eval_set) == 3:
        # bootstrapping
        evaluation = [
            (dict_arrays["x_train"], dict_arrays["y_train"]),
            (dict_arrays["x_test"], dict_arrays["y_test"]),
            (dict_arrays["x_val"], dict_arrays["y_val"]),
        ]
    else:
        logger.error(
            "Unknown number of sample weights = {}".format(len(sample_weight_eval_set))
        )
        raise ValueError(
            "Unknown number of sample weights = {}".format(len(sample_weight_eval_set))
        )

    # Fit the model
    model.fit(
        dict_arrays["x_train"],
        dict_arrays["y_train"],
        eval_set=evaluation,
        sample_weight=sample_weight,
        sample_weight_eval_set=sample_weight_eval_set,
        verbose=verbose,
    )

    if debug_mode:
        logger.info(f"Model feature importance: {model.feature_importances_}")

    train_time = time.time() - start
    if verbose:
        logger.info(f"XGBoost model training time: {train_time:.2f} seconds")

    return model, dict_arrays


def hpo_and_train_xgboost(
    dict_arrays: dict,
    xgboost_cfg: DictConfig,
    hparam_cfg: DictConfig,
    cfg: DictConfig,
    use_RFE: bool = True,
):
    """
    Run hyperparameter optimization and train XGBoost model.

    Parameters
    ----------
    dict_arrays : dict
        Dictionary containing training and test arrays.
    xgboost_cfg : DictConfig
        XGBoost configuration dictionary.
    hparam_cfg : DictConfig
        Hyperparameter configuration dictionary.
    cfg : DictConfig
        Main configuration dictionary.
    use_RFE : bool, optional
        Whether to use Recursive Feature Elimination. Default is True.

    Returns
    -------
    model : xgb.XGBClassifier
        Trained XGBoost classifier.
    space : dict
        Hyperopt search space used.
    model_params : dict
        Best hyperparameters found.
    dict_arrays : dict
        Updated dictionary with potentially reduced features.
    """
    loss_function = get_train_loss_name(cfg)
    eval_metric = get_eval_metric_name(cfg=cfg, cls_model_name="XGBOOST")

    # Get the hyperparameter search space for the hyperopt optimization
    # In theory, you could use different space for pre-RFE, and post-RFE training
    space = parse_hyperopt_search_space(
        hyperopt_cfg=hparam_cfg["SEARCH_SPACE"]["HYPEROPT"]
    )

    space["eval_metric"] = eval_metric
    if loss_function == "Logloss":
        # XGBoost uses different names than CatBoost
        # https://xgboost.readthedocs.io/en/latest/parameter.html
        space["objective"] = "binary:logistic"

    # Find the best hyperparameters (with hyperopt)
    model_params = xgboost_hyperparameter_tuning(
        space, dict_arrays, loss_function, eval_metric, xgboost_cfg, hparam_cfg, cfg
    )

    # Fit the model (with or without feature selection)
    model, dict_arrays = xgboost_train_script(
        model_params=model_params,
        dict_arrays=dict_arrays,
        xgboost_cfg=xgboost_cfg,
        use_RFE=use_RFE,
        eval_metric=get_eval_metric_name("XGBOOST", cfg),
    )

    return model, space, model_params, dict_arrays


def rfe_xgboost_train_wrapper(
    dict_arrays: dict, xgboost_cfg: DictConfig, hparam_cfg: DictConfig, cfg: DictConfig
):
    """
    Wrapper for XGBoost training with optional RFE and iterative HPO.

    Parameters
    ----------
    dict_arrays : dict
        Dictionary containing training and test arrays.
    xgboost_cfg : DictConfig
        XGBoost configuration dictionary.
    hparam_cfg : DictConfig
        Hyperparameter configuration dictionary.
    cfg : DictConfig
        Main configuration dictionary.

    Returns
    -------
    model : xgb.XGBClassifier
        Trained XGBoost classifier.
    space : dict
        Hyperopt search space used.
    model_params : dict
        Best hyperparameters found.
    dict_arrays : dict
        Updated dictionary with potentially reduced features.
    """
    start = time.time()
    if xgboost_cfg["FEATURE_SELECTION"]["RFE"]["use"]:
        model, space, model_params, dict_arrays = hpo_and_train_xgboost(
            dict_arrays,
            xgboost_cfg,
            hparam_cfg,
            cfg,
            use_RFE=xgboost_cfg["FEATURE_SELECTION"]["RFE"]["use"],
        )
        if xgboost_cfg["FEATURE_SELECTION"]["RFE"]["iterate_hyperopt"]:
            # Run again the hyperparam search and the model fit with reduced number of features
            model, space, model_params, _ = hpo_and_train_xgboost(
                dict_arrays,
                xgboost_cfg,
                hparam_cfg,
                cfg,
                use_RFE=False,
            )
    else:
        model, space, model_params, _ = hpo_and_train_xgboost(
            dict_arrays, xgboost_cfg, hparam_cfg, cfg, use_RFE=False
        )
    train_time = time.time() - start
    logger.info(f"XGBoost model HPO+training: {train_time:.2f} seconds")

    return model, space, model_params, dict_arrays


def pick_the_best_params_and_best_config(
    best_metrics,
    list_of_models,
    spaces,
    list_of_model_params,
    list_of_dict_arrays,
    grid_cfgs: dict,
    xgboost_cfg: DictConfig,
):
    """
    Select the best model and configuration from grid search results.

    Parameters
    ----------
    best_metrics : dict
        Dictionary mapping config names to their train/test metric values.
    list_of_models : list
        List of trained XGBoost models from each grid iteration.
    spaces : list
        List of hyperopt search spaces used.
    list_of_model_params : list
        List of best hyperparameters from each iteration.
    list_of_dict_arrays : list
        List of data dictionaries from each iteration.
    grid_cfgs : dict
        Dictionary of grid search configurations.
    xgboost_cfg : DictConfig
        XGBoost configuration dictionary.

    Returns
    -------
    model : xgb.XGBClassifier
        Best performing model.
    space : dict
        Search space used for best model.
    model_params : dict
        Hyperparameters of best model.
    dict_arrays : dict
        Data dictionary for best model.
    xgboost_cfg : DictConfig
        Configuration used for best model.
    """
    best_idx = find_best_metric(best_metrics, xgboost_cfg=xgboost_cfg)
    model = list_of_models[best_idx]
    space = spaces[best_idx]
    model_params = list_of_model_params[best_idx]
    dict_arrays = list_of_dict_arrays[best_idx]

    grid_names = list(grid_cfgs.keys())
    xgboost_cfg = deepcopy(grid_cfgs[grid_names[best_idx]])

    return model, space, model_params, dict_arrays, xgboost_cfg


def xgboost_grid_search(dict_arrays, xgboost_cfg, hparam_cfg, cfg):
    """
    Perform grid search over XGBoost configurations with nested HPO.

    Parameters
    ----------
    dict_arrays : dict
        Dictionary containing training and test arrays.
    xgboost_cfg : DictConfig
        XGBoost configuration dictionary.
    hparam_cfg : DictConfig
        Hyperparameter configuration dictionary.
    cfg : DictConfig
        Main configuration dictionary.

    Returns
    -------
    model : xgb.XGBClassifier
        Best performing model from grid search.
    space : dict
        Search space used for best model.
    model_params : dict
        Hyperparameters of best model.
    dict_arrays : dict
        Data dictionary for best model.
    xgboost_cfg : DictConfig
        Configuration used for best model.
    """
    # Define the grid search space (different weighing schemes at this point)
    time_start = time.time()
    grid_cfgs = define_xgboost_grid_search_space(hparam_cfg, xgboost_cfg, cfg)
    best_metrics = {}
    list_of_models, spaces, list_of_model_params, list_of_dict_arrays = [], [], [], []
    for i, (cfg_name, xgboost_cfg) in enumerate(grid_cfgs.items()):
        logger.info(
            f"XGBoost Grid search iteration {i + 1}/{len(grid_cfgs)}: {cfg_name}"
        )

        # Train the model with Hyperparameter Optimization (HPO) and possibly RFE feature selection
        model, space, model_params, dict_arrays = rfe_xgboost_train_wrapper(
            dict_arrays=dict_arrays,
            xgboost_cfg=xgboost_cfg,
            hparam_cfg=hparam_cfg,
            cfg=cfg,
        )

        # Best metric is now whatever you used in the first place to "define the bestness" in the .yaml config
        if hasattr(model, "evals_result_"):
            # TODO! How come this is missing from some grid search results?
            results = model.evals_result
            try:
                # as in trying to access that printed line
                # validation_0-auc:0.95143	validation_1-auc:0.96319
                best_metrics[cfg_name] = get_last_items_from_OrderedDicts(
                    results=results
                )
            except Exception as e:
                if len(grid_cfgs) > 1:
                    # now you actually have the grid search
                    logger.error(f"Could not get the best metric for {cfg_name}")
                    raise e
                else:
                    # You don't need to know the best model from the grid search as you only did one model here
                    # as in no need to pick the model with the best metric
                    logger.warning(
                        f"Could not get the best metric for {cfg_name}, "
                        f"but you are not doing luckily grid search"
                    )
        else:
            # this lack of "evals_result" in some cases is annoying, and not sure why this happens, but it is not
            # an issue if you don't do a grid search, it just makes hard to pick the best model from the
            # grid search if you don't know the metrics of each model. Either way main HPO is done by the hyperopt
            logger.warning(
                "Problem getting the best metric!, cfg_name = {}".format(cfg_name)
            )
            best_metrics[cfg_name] = None

        # Collect the other outputs as well and return the one corresponding the best grid run
        list_of_models.append(model)
        spaces.append(space)
        list_of_model_params.append(model_params)
        list_of_dict_arrays.append(dict_arrays)

    end_time = time.time() - time_start
    logger.info(f"XGBoost Grid search time: {end_time:.2f} seconds")
    mlflow.log_param("training_time", end_time)

    # Return the best configuration here
    if len(grid_cfgs) > 1:
        model, space, model_params, dict_arrays, xgboost_cfg = (
            pick_the_best_params_and_best_config(
                best_metrics,
                list_of_models,
                spaces,
                list_of_model_params,
                list_of_dict_arrays,
                grid_cfgs=grid_cfgs,
                xgboost_cfg=xgboost_cfg,
            )
        )
    else:
        # If you are not doing grid search, just return the first one
        model, space, model_params, dict_arrays = (
            list_of_models[0],
            spaces[0],
            list_of_model_params[0],
            list_of_dict_arrays[0],
        )

    return model, space, model_params, dict_arrays, xgboost_cfg


# @task(
#     log_prints=True,
#     name="PLR Classifier (XGBoost training)",
#     description="Hyperopt optimization for XGBoost classifier",
# )
def xgboost_train(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    run_name: str,
    cfg: DictConfig,
    xgboost_cfg: DictConfig,
    hparam_cfg: DictConfig,
    features_per_source: dict,
):
    """
    Train and evaluate an XGBoost classifier with MLflow logging.

    Parameters
    ----------
    train_df : pl.DataFrame
        Training data as a Polars DataFrame.
    test_df : pl.DataFrame
        Test data as a Polars DataFrame.
    run_name : str
        Name for the MLflow run.
    cfg : DictConfig
        Main configuration dictionary.
    xgboost_cfg : DictConfig
        XGBoost configuration dictionary.
    hparam_cfg : DictConfig
        Hyperparameter configuration dictionary.
    features_per_source : dict
        Dictionary mapping feature sources to feature counts.

    Returns
    -------
    model : xgb.XGBClassifier
        Trained XGBoost classifier.
    metrics : dict
        Evaluation metrics from the classifier.
    dict_arrays : dict
        Data dictionary used for training.
    """
    # circular import quick fix
    from src.classification.classifier_evaluation import evaluate_sklearn_classifier

    # Convert Polars DataFrames to arrats (or DMatrices)
    _, _, dict_arrays = data_transform_wrapper(
        train_df, test_df, xgboost_cfg, hparam_cfg
    )

    # mlflow.xgboost.autolog() # Does not really work as expected :(
    with mlflow.start_run(run_name=run_name):
        # Log the source params
        log_classifier_sources_as_params(
            features_per_source, dict_arrays, run_name, cfg
        )
        mlflow.log_param("model_name", "XGBoost")

        # This is a bit unorthodox now, we have nesting of multiple optimization "tricks"
        # 1) Manual grid search
        # 2) Feature selection with RFE (or not)
        # 3) (Bayesian) Hyperparameter optimization of XGBoost parameters with Hyperopt
        model, space, model_params, dict_arrays, xgboost_cfg = xgboost_grid_search(
            dict_arrays, xgboost_cfg, hparam_cfg, cfg
        )

        # Train model with these hyperparameters
        log_classifier_params_to_mlflow(model_params, xgboost_cfg)

        # Evaluate the model performance
        models, metrics = evaluate_sklearn_classifier(
            model_name="XGBOOST",
            dict_arrays=dict_arrays,
            best_params=model_params,
            cls_model_cfg=xgboost_cfg,
            eval_cfg=cfg["CLS_EVALUATION"],
            cfg=cfg,
            run_name=run_name,
        )

        mlflow.end_run()

    return model, metrics, dict_arrays


def xgboost_main(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    run_name: str,
    cfg: DictConfig,
    xgboost_cfg: DictConfig,
    hparam_cfg: DictConfig,
    features_per_source: dict,
):
    """
    Main entry point for XGBoost classification pipeline.

    Orchestrates feature preprocessing, model training with hyperparameter
    optimization, evaluation, and feature importance visualization.

    Parameters
    ----------
    train_df : pl.DataFrame
        Training data as a Polars DataFrame.
    test_df : pl.DataFrame
        Test data as a Polars DataFrame.
    run_name : str
        Name for the MLflow run.
    cfg : DictConfig
        Main configuration dictionary.
    xgboost_cfg : DictConfig
        XGBoost configuration dictionary.
    hparam_cfg : DictConfig
        Hyperparameter configuration dictionary.
    features_per_source : dict
        Dictionary mapping feature sources to feature counts.

    Returns
    -------
    None
        Results are logged to MLflow and visualizations are generated.
    """
    # Task) Preprocess the features (standardize, normalize, etc.)
    train_df, test_df = preprocess_features(
        train_df,
        test_df,
        _cls_preprocess_cfg=cfg["CLASSIFICATION_SETTINGS"]["PREPROCESS"],
    )

    # Task) Train the XGBoost model (with hyperparameter optimization) (and evaluate with metrics)
    model, metrics, dict_arrays = xgboost_train(
        train_df,
        test_df,
        run_name,
        cfg,
        xgboost_cfg,
        hparam_cfg=hparam_cfg,
        features_per_source=features_per_source,
    )

    # Task) Visualize the feature importance (SHAP, LIME, etc.)
    classifier_feature_importance(
        model, dict_arrays, metrics, xgboost_cfg, cfg, run_name
    )
