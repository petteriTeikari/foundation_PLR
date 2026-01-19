from copy import deepcopy

import numpy as np
from omegaconf import DictConfig
from loguru import logger
import polars as pl
import mlflow
from catboost import CatBoostClassifier
import optuna
from catboost import Pool

from src.classification.bootstrap_evaluation import (
    get_ensemble_stats,
    bootstrap_evaluator,
)
from src.classification.catboost.catboost_ensemble import (
    ClassificationEnsembleSGLB,
)
from src.classification.catboost.catboost_utils import rearrange_the_split
from src.classification.classifier_log_utils import (
    classifier_log_cls_evaluation_to_mlflow,
    log_classifier_params_to_mlflow,
    log_classifier_sources_as_params,
)
from src.classification.classifier_utils import preprocess_features, classifier_hpo_eval
from src.classification.stats_metric_utils import (
    bootstrap_metrics_per_split,
)
from src.classification.weighing_utils import return_weights_as_dict
from src.classification.xgboost_cls.xgboost_utils import data_transform_wrapper
from src.log_helpers.log_naming_uris_and_dirs import (
    get_train_loss_name,
    get_eval_metric_name,
)


def catboost_ensemble_fit(
    train: Pool,
    test: Pool,
    val: Pool,
    cls_model_cfg: DictConfig,
    cfg: DictConfig,
    param: dict,
    loss_function: str,
    verbose: bool = True,
):
    use_GPU = False
    if use_GPU:  # cls_model_cfg['MODEL']['use_GPU']:
        if cfg["DEVICE"]["device"] == "cuda":
            # https://catboost.ai/docs/en/features/training-on-gpu
            logger.info("CATBOOST training with GPU!")
            task_type = "GPU"
            devices = "0"
        else:
            task_type = None
            devices = None
    else:
        task_type = None
        devices = None

    ensemble_params = dict(deepcopy(cls_model_cfg))
    ensemble_params.pop("MODEL")
    param = {**param, **ensemble_params}
    param["loss_function"] = loss_function

    # Note! You are now calling this custom Ensemble Class that calls then the CatBoostClassifier
    # so not all default parameters are not available by default here, unless you explicitly add them
    ens = ClassificationEnsembleSGLB(
        verbose=verbose,
        **param,
        task_type=task_type,
        devices=devices,
    )

    # This itself trains n (esize) models, so this is slower than XGBoost HPO for example
    if val is not None:
        # https://github.com/catboost/catboost/issues/2705#issuecomment-2213766803 the last set is used as val
        ens.fit(train, eval_set=[test, val])
    else:
        ens.fit(train, eval_set=test)

    return ens


def catboost_ensemble_HPO(
    train: Pool,
    test: Pool,
    y_test: np.ndarray,
    cls_model_cfg: DictConfig,
    cfg: DictConfig,
    hparam_cfg: DictConfig,
    loss_function: str,
    eval_metric: str,
    verbose: bool = False,
):
    def objective(trial):
        """
        https://github.com/optuna/optuna-examples/blob/main/catboost/catboost_simple.py
        """
        optuna_params = hparam_cfg["SEARCH_SPACE"]["OPTUNA"]
        # https://github.com/yandex-research/GBDT-uncertainty/blob/339264ee82c1ec2b22d4200d3b9c18fcce56bb0d/gbdt_uncertainty/training.py#L22
        # TODO! You could unify (see XGBoost hyperparams)
        #  or think of a nicer way to get this param dict from the cfg .yaml file
        param = {
            # "loss_function": trial.suggest_categorical(
            #     "loss_function",
            #     list(optuna_params["loss_function"]),
            # ),
            "colsample_bylevel": trial.suggest_float(
                "colsample_bylevel",
                optuna_params["colsample_bylevel"][0],
                optuna_params["colsample_bylevel"][1],
            ),
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf",
                optuna_params["min_data_in_leaf"][0],
                optuna_params["min_data_in_leaf"][1],
            ),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg",
                optuna_params["l2_leaf_reg"][0],
                optuna_params["l2_leaf_reg"][1],
            ),
            "depth": trial.suggest_int(
                "depth", optuna_params["depth"][0], optuna_params["depth"][1]
            ),
            # "boosting_type": trial.suggest_categorical(
            #     "boosting_type", ["Ordered", "Plain"]
            # ),
            "lr": trial.suggest_categorical("lr", list(optuna_params["lr"])),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type",
                ["No"],  # , "Bayesian", "Bernoulli", "MVS"]
            ),
            "used_ram_limit": "36gb",
        }

        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 0, 10
            )
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

        # Fit the model
        ens = catboost_ensemble_fit(
            train, test, None, cls_model_cfg, cfg, param, loss_function, verbose=verbose
        )

        probs = ens.predict(test)  # (esize, no_subjects, no_classes)
        probs_mean = np.mean(probs, axis=0)  # (no_subjects, no_classes)
        probs_mean_class1 = probs_mean[:, 1]  # (no_subjects,)
        # preds = np.argmax(probs_mean, axis=1)  # (no_subjects,)

        # Compute from the pool in the future (and unify behind the same key for all the classifiers)
        score = classifier_hpo_eval(
            y_test,
            probs_mean_class1,
            eval_metric,
            model="CATBOOST",
            hpo_method="optuna",
        )

        return score

    study = optuna.create_study(direction="maximize")
    # Rather fast in relatively to HPO for the hand-crafted features, 1 hour not enough for embeddings on desktop CPU
    study.optimize(objective, n_trials=100, timeout=3600)

    logger.info("CatBoost HPO finished")
    logger.info(
        "Best value: {} = {}".format(
            hparam_cfg["HYPERPARAMS"]["metric_val"], study.best_value
        )
    )
    logger.info("Best params: {}".format(study.best_params))

    return study, study.best_params, study.best_value


def create_data_pools(dict_arrays: dict, cls_model_cfg: DictConfig):
    # Same weight dfrom other classifiers if you happen to need them
    weights_dict = return_weights_as_dict(dict_arrays, cls_model_cfg)

    if cls_model_cfg["MODEL"]["WEIGHING"]["weigh_the_samples"]:
        # https://github.com/catboost/catboost/issues/620#issuecomment-454434544
        logger.debug("Set the sample weights for the CatBoost model")
        train_weights = weights_dict["sample_weight_eval_set"][0]
        test_weights = weights_dict["sample_weight_eval_set"][1]
        if len(weights_dict["sample_weight_eval_set"]) > 2:
            val_weights = weights_dict["sample_weight_eval_set"][2]
            assert len(val_weights) == dict_arrays["y_val"].shape[0]
        assert len(train_weights) == dict_arrays["y_train"].shape[0]
        assert len(test_weights) == dict_arrays["y_test"].shape[0]
    else:
        train_weights = None
        test_weights = None

    train_dataset = Pool(
        data=dict_arrays["x_train"],
        label=dict_arrays["y_train"],
        weight=train_weights,
    )

    test_dataset = Pool(
        data=dict_arrays["x_test"],
        label=dict_arrays["y_test"],
        weight=test_weights,
    )

    if "x_val" in dict_arrays:
        # When bootstrapping, you will have a validation split here:
        val_dataset = Pool(
            data=dict_arrays["x_val"],
            label=dict_arrays["y_val"],
            weight=val_weights,
        )
    else:
        val_dataset = None

    return train_dataset, val_dataset, test_dataset, dict_arrays["y_test"]


def ensemble_eval_metrics(
    model: CatBoostClassifier,
    probs_per_model: np.ndarray,
    y: np.ndarray,
    metrics_iter: dict,
    cfg: DictConfig,
    split: str,
):
    """
    see bootstrap_metrics_per_split() in stats_metric_utils.py used by the bootstrap evaluation
    """

    preds = {
        "y_pred": np.argmax(probs_per_model, axis=1),
        "y_pred_proba": probs_per_model[:, 1],
        "labels": y,
    }

    pseudo_codes = np.linspace(0, y.shape[0] - 1, y.shape[0]).astype(int).astype(str)

    metrics_iter = bootstrap_metrics_per_split(
        X=probs_per_model,
        y_true=y,
        preds=preds,
        model=model,
        model_name="ensemble",
        metrics_per_split=metrics_iter,
        codes_per_split=pseudo_codes,
        method_cfg=cfg["CLS_EVALUATION"]["BOOTSTRAP"],
        cfg=cfg,
        split=split,
    )

    return metrics_iter


def combine_unks_with_subjectwise_stats(subjectwise_stats, unks, split: str):
    # these are now all subjectwise measures (from n models in the ensemble)
    for key, value in unks.items():
        nan_array = np.full_like(value, np.nan)
        subjectwise_stats[split]["preds"][key] = {
            "mean": value,
            "std": nan_array,
        }

    # Get the uncertainties (from the Catboost tutorial code)
    # https://github.com/yandex-research/GBDT-uncertainty/blob/main/synthetic_classification.ipynb
    # These are subject-wise uncertainties
    # Total Uncertainty - unks['entropy_of_expected']
    # Data Uncertainty - unks['expected_entropy']
    # Knowledge Uncertainty - unks['mutual_information']
    for metric, array in unks.items():
        if isinstance(array, np.ndarray):
            # Get the mean uncertainties for the whole split
            subjectwise_stats[split]["uq"]["scalars"][metric] = np.mean(array)
        else:
            logger.warning(f"Unknown type for {metric} in unks: {type(array)}")

    return subjectwise_stats


def eval_ensemble_split(
    ens,
    x: np.ndarray,
    y: np.ndarray,
    dict_arrays: dict,
    split: str,
    cfg: DictConfig,
    verbose: bool = False,
):
    """
    https://github.com/yandex-research/GBDT-uncertainty/blob/339264ee82c1ec2b22d4200d3b9c18fcce56bb0d/aggregate_results_classification.py#L190
    """
    probs = ens.predict(
        x
    )  # (esize, no_subjects, no_classes) # all_preds, e.g. (10, 145, 2)
    # preds_proba = np.mean(probs, axis=0) # (no_subjects, no_classes)
    # preds_var = np.var(probs, axis=0)[:, 1] # (no_subjects,)
    # preds_proba_class1 = preds_proba[:, 1] # (no_subjects,)
    # preds = np.argmax(preds_proba, axis=1) # (no_subjects,)

    method_cfg = cfg["CLS_EVALUATION"]["BOOTSTRAP"]

    metrics_iter = {}
    no_of_models = probs.shape[0]

    for idx in range(no_of_models):
        probs_per_model = probs[idx]  # (no_subjects, no_classes)
        model = ens.ensemble[idx]
        metrics_iter = ensemble_eval_metrics(
            model, probs_per_model, y, metrics_iter, cfg=cfg, split=split
        )

    no_of_scalar_metrics = len(metrics_iter["metrics"]["scalars"]["AUROC"])
    assert no_of_models == no_of_scalar_metrics, (
        f"Number of models ({no_of_models}( should match "
        f"with the number of scalar metrics ({no_of_scalar_metrics})!\n"
        f"Glitch in your aggregation loop?"
    )

    no_of_array_values = metrics_iter["metrics"]["arrays"]["AUROC"]["fpr"].shape[1]
    assert no_of_models == no_of_array_values, (
        f"Number of models ({no_of_models}( should match "
        f"with the number of array values ({no_of_array_values})!\n"
        f"Glitch in your aggregation loop?"
    )

    # Bootstrap evaluation code wants the split to be present in the metrics
    metrics = {split: metrics_iter}

    # Get ensemble stats
    metrics_stats, subjectwise_stats, subject_global_stats = get_ensemble_stats(
        metrics,
        dict_arrays,
        method_cfg,
        sort_list=False,
        call_from="CATBOOST",
        verbose=verbose,
    )

    # Same output as in the bootstrap evaluation, see bootstrap_evaluator()
    metrics_out = {
        "metrics_iter": metrics,
        "metrics_stats": metrics_stats,
        "subjectwise_stats": subjectwise_stats,
        "subject_global_stats": subject_global_stats,
    }

    return metrics_out


def catboost_ensemble_wrapper(
    dict_arrays: dict,
    train: Pool,
    test: Pool,
    best_params: dict,
    cls_model_cfg: DictConfig,
    hparam_cfg: DictConfig,
    cfg: DictConfig,
    loss_function: str,
    eval_metric: str,
    val: Pool = None,
    verbose: bool = False,
    rearrange_results: bool = True,
):
    # Fit the model
    loss_function = get_train_loss_name(cfg)
    ens = catboost_ensemble_fit(
        train,
        test,
        val,
        cls_model_cfg,
        cfg,
        best_params,
        verbose=verbose,
        loss_function=loss_function,
    )

    # Evaluate the model
    results = {}
    split = "train"
    results[split] = eval_ensemble_split(
        ens,
        x=dict_arrays[f"x_{split}"],
        y=dict_arrays[f"y_{split}"],
        dict_arrays=dict_arrays,
        split=split,
        cfg=cfg,
        verbose=verbose,
    )

    split = "test"
    results[split] = eval_ensemble_split(
        ens,
        x=dict_arrays[f"x_{split}"],
        y=dict_arrays[f"y_{split}"],
        dict_arrays=dict_arrays,
        split=split,
        cfg=cfg,
        verbose=verbose,
    )

    if val is not None:
        split = "val"
        results[split] = eval_ensemble_split(
            ens,
            x=dict_arrays[f"x_{split}"],
            y=dict_arrays[f"y_{split}"],
            dict_arrays=dict_arrays,
            split=split,
            cfg=cfg,
            verbose=verbose,
        )

    # Match the dict structure of the bootstrap evaluation
    if rearrange_results:
        results = rearrange_the_split(results)

    return ens, results


def catboost_train_script(
    dict_arrays: dict,
    cls_model_cfg: DictConfig,
    hparam_cfg: DictConfig,
    cfg: DictConfig,
    verbose: bool = True,
    run_name: str = None,
    skip_HPO: bool = True,
):
    """
    https://towardsdatascience.com/estimating-uncertainty-with-catboost-classifiers-2d0b2229ad6
    -> https://github.com/yandex-research/GBDT-uncertainty
    """
    # Define data
    train, _, test, y_test = create_data_pools(dict_arrays, cls_model_cfg)

    # Define the model + Fit the model
    eval_metric = get_eval_metric_name(cls_model_name="CATBOOST", cfg=cfg)
    loss_function = get_train_loss_name(cfg)
    logger.info(f"Loss function: {loss_function}, eval metric: {eval_metric}")
    if not skip_HPO:
        study, best_params, best_value = catboost_ensemble_HPO(
            train,
            test,
            y_test,
            cls_model_cfg,
            cfg,
            hparam_cfg,
            loss_function,
            eval_metric,
        )
    else:
        logger.info("Skipping HPO for CatBoost, and using the default hyperparameters")
        for key, value in hparam_cfg["HYPERPARAMS"]["defaults"].items():
            logger.info(f"{key}: {value}")
        best_params = hparam_cfg["HYPERPARAMS"]["defaults"]

    # Log the hyperparams (use the same function as for XGBoost)
    log_classifier_params_to_mlflow(best_params, cls_model_cfg)

    # Without bootstrapping, this is your baseline model then if you are doing
    # "Stability of clinical prediction models developed using statistical or machine learning methods"
    # see https://doi.org/10.1002/bimj.202200302 / https://stephenrho.github.io/pminternal/
    model, baseline_results = catboost_ensemble_wrapper(
        dict_arrays,
        train,
        test,
        best_params,
        cls_model_cfg,
        hparam_cfg,
        cfg,
        loss_function,
        eval_metric,
    )

    # Use the best params then to train the final ensemble model
    if cls_model_cfg["MODEL"]["CI"]["method_CI"] == "ENSEMBLE":
        logger.info("Compute Confidence Intervals with Ensemble")
        models, results = catboost_ensemble_wrapper(
            dict_arrays,
            train,
            test,
            best_params,
            cls_model_cfg,
            hparam_cfg,
            cfg,
            loss_function,
            eval_metric,
        )
    elif cls_model_cfg["MODEL"]["CI"]["method_CI"] == "BOOTSTRAP":
        logger.info("Compute Confidence Intervals with Bootstrap")
        models, results = bootstrap_evaluator(
            model_name="CATBOOST",
            run_name=run_name,
            dict_arrays=dict_arrays,
            best_params=best_params,
            cls_model_cfg=cls_model_cfg,
            method_cfg=cfg["CLS_EVALUATION"]["BOOTSTRAP"],
            hparam_cfg=hparam_cfg,
            cfg=cfg,
        )

    return model, baseline_results, models, results, dict_arrays


def catboost_main(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    run_name: str,
    cfg: DictConfig,
    cls_model_cfg: DictConfig,
    hparam_cfg: DictConfig,
    features_per_source: dict,
):
    # Task) Preprocess the features (standardize, normalize, etc.)
    train_df, test_df = preprocess_features(
        train_df,
        test_df,
        cls_preprocess_cfg=cfg["CLASSIFICATION_SETTINGS"]["PREPROCESS"],
    )

    # Convert Polars DataFrames to arrays
    _, _, dict_arrays = data_transform_wrapper(
        train_df, test_df, cls_model_cfg, hparam_cfg, run_name
    )

    with mlflow.start_run(run_name=run_name):
        log_classifier_sources_as_params(
            features_per_source, dict_arrays, run_name, cfg
        )
        mlflow.log_param("model_name", "CatBoost")
        model, baseline_results, models, metrics, dict_arrays = catboost_train_script(
            dict_arrays,
            cls_model_cfg,
            hparam_cfg,
            cfg,
            run_name=run_name,
            skip_HPO=hparam_cfg["HYPERPARAMS"]["skip_HPO"],
        )

        classifier_log_cls_evaluation_to_mlflow(
            model,
            baseline_results,
            models,
            metrics,
            dict_arrays,
            cls_model_cfg,
            run_name=run_name,
            model_name="CatBoost",
        )

        mlflow.end_run()
