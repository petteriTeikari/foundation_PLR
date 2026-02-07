import time
import warnings
from copy import deepcopy

import mlflow
import numpy as np
from loguru import logger
from omegaconf import DictConfig
from sklearn.utils import resample
from tqdm import tqdm

from src.classification.cls_model_utils import bootstrap_model_selector
from src.classification.stats_metric_utils import (
    bootstrap_compute_global_subject_stats,
    bootstrap_compute_stats,
    bootstrap_compute_subject_stats,
    bootstrap_metrics,
    compute_uq_for_subjectwise_stats,
)
from src.classification.weighing_utils import return_weights_as_dict
from src.stats.classifier_calibration import bootstrap_calibrate_classifier


def prepare_for_bootstrap(dict_arrays: dict, method_cfg: DictConfig):
    """
    Prepare data arrays for bootstrap evaluation.

    Sets up index arrays for stratified bootstrap resampling. The train split
    will be resampled into new train/val splits while test remains untouched.

    Parameters
    ----------
    dict_arrays : dict
        Dictionary containing:
        - x_train, y_train: Training features and labels
        - x_test, y_test: Test features and labels
        - subject_codes_train, subject_codes_test: Subject identifiers
    method_cfg : DictConfig
        Bootstrap configuration with 'join_test_and_train' option.

    Returns
    -------
    dict
        Updated dict_arrays with 'X_idxs' array for bootstrap sampling.

    References
    ----------
    - https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/
    """
    # Bootstrap resample is splitting the train split into -> new train and val
    # Test will be untouched
    dict_arrays["X_idxs"] = np.linspace(
        0, dict_arrays["x_train"].shape[0] - 1, dict_arrays["x_train"].shape[0]
    ).astype(int)

    if method_cfg["join_test_and_train"]:
        raise NotImplementedError
        # X = np.concatenate((X_train, X_test), axis=0)
        # y = np.concatenate((y_train, y_test), axis=0)
        # X_test, y_test, codes_test = None, None, None
    else:
        assert dict_arrays["x_test"].shape[0] == dict_arrays["y_test"].shape[0], (
            "X_test and y_test must have the same number of rows"
        )
        assert (
            dict_arrays["x_test"].shape[0] == dict_arrays["subject_codes_test"].shape[0]
        ), "X_test and subject_codes_test must have the same number of rows"

    assert dict_arrays["x_train"].shape[0] == dict_arrays["y_train"].shape[0], (
        "X and y must have the same number of rows"
    )
    assert dict_arrays["x_train"].shape[0] == dict_arrays["X_idxs"].shape[0], (
        "X and X_idxs must have the same number of rows"
    )
    assert (
        dict_arrays["x_train"].shape[0] == dict_arrays["subject_codes_train"].shape[0]
    ), "X and subject_codes_train must have the same number of rows"

    return dict_arrays


def select_bootstrap_samples(dict_arrays, n_samples, method_cfg) -> dict:
    """
    Select bootstrap samples for a single iteration.

    Performs stratified bootstrap resampling to create new train/val splits
    from the original training data.

    Parameters
    ----------
    dict_arrays : dict
        Data arrays including X_idxs for sampling.
    n_samples : int
        Number of samples to draw for training.
    method_cfg : DictConfig
        Bootstrap configuration.

    Returns
    -------
    dict
        Dictionary with resampled train/val data arrays.
    """

    def reample_split_indices(X_idxs: np.ndarray, n_samples: int, y: np.ndarray):
        train_idxs = resample(X_idxs, n_samples=n_samples, stratify=y)
        val_idxs = np.array(
            [x for x in X_idxs if x.tolist() not in train_idxs.tolist()]
        )
        return train_idxs, val_idxs

    dict_arrays_iter = dict_arrays.copy()

    # Get indices of the new split samples
    dict_arrays_iter["train_idxs"], dict_arrays_iter["val_idxs"] = (
        reample_split_indices(
            X_idxs=dict_arrays_iter["X_idxs"],
            n_samples=n_samples,
            y=dict_arrays_iter["y_train"],
        )
    )

    # TODO! You could obviously try to loop these and parametrize the split(s)
    #  and make this more compact
    # Tmp the original train split
    x_train = dict_arrays_iter["x_train"]
    y_train = dict_arrays_iter["y_train"]
    subject_codes_train = dict_arrays_iter["subject_codes_train"]
    x_train_w = dict_arrays_iter["x_train_w"]

    # Pick the corresponding samples
    dict_arrays_iter["x_train"] = x_train[dict_arrays_iter["train_idxs"]]
    dict_arrays_iter["y_train"] = y_train[dict_arrays_iter["train_idxs"]]
    dict_arrays_iter["x_val"] = x_train[dict_arrays_iter["val_idxs"]]
    dict_arrays_iter["y_val"] = y_train[dict_arrays_iter["val_idxs"]]

    dict_arrays_iter["subject_codes_train"] = subject_codes_train[
        dict_arrays_iter["train_idxs"]
    ]
    dict_arrays_iter["subject_codes_val"] = subject_codes_train[
        dict_arrays_iter["val_idxs"]
    ]
    dict_arrays_iter["x_train_w"] = x_train_w[dict_arrays_iter["train_idxs"]]
    dict_arrays_iter["x_val_w"] = x_train_w[dict_arrays_iter["val_idxs"]]

    assert dict_arrays_iter["x_train"].shape[0] == dict_arrays_iter["y_train"].shape[0]
    assert dict_arrays_iter["x_val"].shape[0] == dict_arrays_iter["y_val"].shape[0]
    assert (
        dict_arrays_iter["x_train_w"].shape[0] == dict_arrays_iter["y_train"].shape[0]
    )
    assert dict_arrays_iter["x_val_w"].shape[0] == dict_arrays_iter["y_val"].shape[0]
    assert (
        dict_arrays_iter["x_train"].shape[0]
        == dict_arrays_iter["subject_codes_train"].shape[0]
    )
    assert (
        dict_arrays_iter["x_val"].shape[0]
        == dict_arrays_iter["subject_codes_val"].shape[0]
    )

    return dict(sorted(dict_arrays_iter.items()))


def splits_as_dicts(dict_arrays_iter: dict):
    """
    Convert flat array dictionary to nested split-based structure.

    Parameters
    ----------
    dict_arrays_iter : dict
        Flat dictionary with keys like 'x_train', 'y_train', etc.

    Returns
    -------
    dict
        Nested dictionary with structure:
        {split: {'X': ..., 'y': ..., 'w': ..., 'codes': ...}}
    """
    splits = ["train", "val", "test"]
    dict_splits = {}
    for split in splits:
        dict_splits[split] = {
            "X": dict_arrays_iter[f"x_{split}"],
            "y": dict_arrays_iter[f"y_{split}"],
            "w": dict_arrays_iter[f"x_{split}_w"],
            "codes": dict_arrays_iter[f"subject_codes_{split}"],
        }
        assert dict_splits[split]["X"].shape[0] == dict_splits[split]["y"].shape[0]
        assert dict_splits[split]["X"].shape[0] == dict_splits[split]["w"].shape[0]
        assert dict_splits[split]["X"].shape[0] == dict_splits[split]["codes"].shape[0]

    return dict_splits


def check_bootstrap_iteration_quality(metrics_iter, dict_arrays_iter, dict_arrays):
    """
    Validate that bootstrap iteration used all expected samples.

    Checks that train/val splits used all subject codes from the original
    training data and that test samples match expected count.

    Parameters
    ----------
    metrics_iter : dict
        Metrics collected from all bootstrap iterations.
    dict_arrays_iter : dict
        Data arrays from the current iteration.
    dict_arrays : dict
        Original data arrays before bootstrap resampling.

    Raises
    ------
    AssertionError
        If train/val codes don't match original or test samples count is wrong.
    """
    train_codes_used = list(
        metrics_iter["train"]["preds_dict"]["arrays"]["y_pred_proba"].keys()
    )
    val_codes_used = list(
        metrics_iter["val"]["preds_dict"]["arrays"]["y_pred_proba"].keys()
    )
    # in the bootstrap scenario, after all the iterations, both splits should have had all the codes used
    # from the original train split
    assert len(train_codes_used) == len(val_codes_used), (
        "Train and val codes must have the same length"
    )
    assert len(train_codes_used) == dict_arrays["subject_codes_train"].shape[0], (
        "All codes must have been used"
    )

    # this has a different structure, as test samples are always the same across the bootstrapping
    # so we can just aggregate predictions to a np.ndarray
    no_test_samples_used = metrics_iter["test"]["preds"]["arrays"]["predictions"][
        "y_pred_proba"
    ].shape[0]
    assert no_test_samples_used == dict_arrays["x_test"].shape[0], (
        "All test samples must have been used"
    )


def get_ensemble_stats(
    metrics_iter,
    dict_arrays,
    method_cfg,
    call_from: str = None,
    sort_list: bool = True,
    verbose: bool = True,
):
    """
    Compute aggregate statistics from bootstrap iterations.

    Aggregates per-iteration metrics into final statistics including:
    - Mean and CI for AUROC, Brier, etc.
    - Per-subject prediction statistics
    - Global uncertainty metrics

    Parameters
    ----------
    metrics_iter : dict
        Per-iteration metrics from bootstrap.
    dict_arrays : dict
        Original data arrays.
    method_cfg : DictConfig
        Bootstrap configuration.
    call_from : str, optional
        Caller identifier for logging.
    sort_list : bool, default True
        Sort subject statistics.
    verbose : bool, default True
        Enable verbose logging.

    Returns
    -------
    tuple
        (metrics_stats, subjectwise_stats, subject_global_stats)
    """
    # Compute the final stats of the metrics (scalar AUROC, array ROC curves, etc.)
    try:
        metrics_stats = bootstrap_compute_stats(
            metrics_iter, method_cfg, call_from, verbose=verbose
        )
    except Exception as e:
        logger.error(f"Error in computing metrics stats: {e}")
        metrics_stats = None

    # Compute the stats of the predictions (class probabilities per subject)
    try:
        subjectwise_stats = bootstrap_compute_subject_stats(
            metrics_iter,
            dict_arrays,
            method_cfg,
            sort_list=sort_list,
            call_from=call_from,
            verbose=verbose,
        )

    except Exception as e:
        logger.error(f"Error in computing subjectwise stats: {e}")
        subjectwise_stats = None

    try:
        # Subjectwise Uncertainty metrics ("unks" from Catboost tutorial)
        subjectwise_stats = compute_uq_for_subjectwise_stats(
            metrics_iter, subjectwise_stats, verbose=verbose
        )
    except Exception as e:
        logger.error(f"Error in computing subjectwise uncertainty: {e}")

    # Compute "mean response" of the subjects, e.g. scalar mean UQ metric to describe the whole model uncertainty
    try:
        subject_global_stats = bootstrap_compute_global_subject_stats(
            subjectwise_stats, method_cfg, verbose=verbose
        )
    except Exception as e:
        logger.error(f"Error in computing global subject stats: {e}")
        subject_global_stats = None

    return metrics_stats, subjectwise_stats, subject_global_stats


def append_models_to_list_for_mlflow(models: list, model, model_name: str, i: int):
    """
    Add a trained model to the list for MLflow logging.

    Handles special cases like moving TabM models from GPU to CPU to avoid
    memory issues during serialization.

    Parameters
    ----------
    models : list
        List of trained models from previous iterations.
    model : object
        The trained model from current iteration.
    model_name : str
        Name of the classifier (e.g., 'TabM', 'CatBoost').
    i : int
        Current bootstrap iteration index.

    Returns
    -------
    list
        Updated list of models with the new model appended.
    """
    if model_name == "TabM":
        # classifier is now on CUDA and will cause possible memory issues if you don't detach it and use CPU
        model = model.to("cpu")
    models.append(deepcopy(model))
    return models


def bootstrap_evaluator(
    model_name: str,
    run_name: str,
    dict_arrays: dict,
    best_params,
    cls_model_cfg: DictConfig,
    method_cfg: DictConfig,
    hparam_cfg: DictConfig,
    cfg: DictConfig,
    debug_aggregation: bool = False,
):
    """
    Run bootstrap evaluation for classifier performance estimation.

    Performs n_iterations of bootstrap resampling to estimate:
    - STRATOS-compliant metrics (AUROC, calibration, clinical utility)
    - Confidence intervals via percentile bootstrap
    - Per-subject prediction uncertainty

    Parameters
    ----------
    model_name : str
        Classifier name (e.g., 'CatBoost', 'XGBoost').
    run_name : str
        MLflow run name.
    dict_arrays : dict
        Data arrays with train/test splits.
    best_params : dict
        Best hyperparameters from optimization.
    cls_model_cfg : DictConfig
        Classifier model configuration.
    method_cfg : DictConfig
        Bootstrap configuration with 'n_iterations', 'data_ratio'.
    hparam_cfg : DictConfig
        Hyperparameter configuration.
    cfg : DictConfig
        Full Hydra configuration.
    debug_aggregation : bool, default False
        Enable debug logging for metric aggregation.

    Returns
    -------
    tuple
        (models, results_dict) where:
        - models: List of trained models (one per iteration)
        - results_dict: Contains metrics_iter, metrics_stats,
          subjectwise_stats, subject_global_stats

    Notes
    -----
    Uses stratified resampling to maintain class balance across iterations.
    Test set remains fixed; only train is resampled into train/val.
    """
    warnings.simplefilter("ignore")
    start_time = time.time()
    dict_arrays = prepare_for_bootstrap(dict_arrays, method_cfg)
    n_samples = int(dict_arrays["X_idxs"].shape[0] * method_cfg["data_ratio"])
    metrics_iter = {}
    models = []

    for i in tqdm(
        range(method_cfg["n_iterations"]),
        total=method_cfg["n_iterations"],
        desc="Bootstrap iterations",
    ):
        # What samples to use per iteration (sample weights do not require re-computation)
        dict_arrays_iter = select_bootstrap_samples(dict_arrays, n_samples, method_cfg)

        # Update weights and other dataset dependent params as well
        weights_dict = return_weights_as_dict(dict_arrays_iter, cls_model_cfg)

        # Retrain the model with the bootstrapped samples
        model, results_per_iter = bootstrap_model_selector(
            model_name=model_name,
            cls_model_cfg=cls_model_cfg,
            hparam_cfg=hparam_cfg,
            cfg=cfg,
            best_params=best_params,
            dict_arrays=dict_arrays_iter,
            weights_dict=weights_dict,
        )

        # Calibrate classifier (if desired)
        model = bootstrap_calibrate_classifier(
            i, model, cls_model_cfg, dict_arrays_iter, weights_dict
        )

        # Append to a list to be saved to MLflow
        models = append_models_to_list_for_mlflow(models, model, model_name, i)

        # Easier to evaluate with nested dictionaries instead of the flat one above
        dict_splits = splits_as_dicts(dict_arrays_iter)

        # Compute your scalar metrics (AUROC, etc.), ROC Curve stats, and patient predictions with uncertainty
        metrics_iter = bootstrap_metrics(
            i,
            model,
            dict_splits,
            metrics_iter,
            results_per_iter,
            method_cfg,
            cfg,
            debug_aggregation=debug_aggregation,
            model_name=model_name,
        )

    # Check bootstrap iters
    del model
    check_bootstrap_iteration_quality(metrics_iter, dict_arrays_iter, dict_arrays)

    # Get ensemble stats
    metrics_stats, subjectwise_stats, subject_global_stats = get_ensemble_stats(
        metrics_iter, dict_arrays, method_cfg
    )

    end_time = time.time() - start_time
    logger.info("Bootstrap evaluation in {:.2f} seconds".format(end_time))
    try:
        mlflow.log_param("bootstrap_time", end_time)
    except Exception as e:
        # Might happen during grid search, you are trying to write over the same value
        logger.debug(f"MLFlow logging failed: {e}")
    warnings.resetwarnings()

    return models, {
        "metrics_iter": metrics_iter,
        "metrics_stats": metrics_stats,
        "subjectwise_stats": subjectwise_stats,
        "subject_global_stats": subject_global_stats,
    }
