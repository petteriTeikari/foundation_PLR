"""
Classification ensemble module.

Provides functionality to aggregate predictions from multiple classification
models and compute ensemble metrics with bootstrap confidence intervals.

Cross-references:
- src/ensemble/ensemble_utils.py for run retrieval
- src/classification/bootstrap_evaluation.py for metric computation
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from src.anomaly_detection.anomaly_utils import get_artifact
from src.classification.bootstrap_evaluation import get_ensemble_stats
from src.classification.stats_metric_utils import bootstrap_metrics_per_split
from src.classification.xgboost_cls.xgboost_utils import encode_labels_to_integers
from src.ensemble.ensemble_utils import are_codes_the_same
from src.log_helpers.local_artifacts import load_results_dict


def import_model_metrics(
    run_id: str, run_name: str, model_name: str, subdir: str = "baseline_model"
) -> dict[str, Any]:
    """
    Load model metrics from MLflow artifact storage.

    Parameters
    ----------
    run_id : str
        MLflow run ID.
    run_name : str
        MLflow run name.
    model_name : str
        Name of the model (for artifact path construction).
    subdir : str, default 'baseline_model'
        Subdirectory within artifacts to load from.

    Returns
    -------
    dict
        Dictionary containing model artifacts and metrics.
    """
    artifacts_path = get_artifact(run_id, run_name, model_name, subdir=subdir)
    artifacts = load_results_dict(artifacts_path)
    return artifacts


def get_preds_and_labels_from_artifacts(
    artifacts: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Extract predictions and labels from model artifacts.

    Retrieves mean predictions and variance from subjectwise statistics,
    which are averaged across bootstrap iterations.

    Parameters
    ----------
    artifacts : dict
        Dictionary containing model artifacts with 'subjectwise_stats'.

    Returns
    -------
    dict
        Mean predicted probabilities per split.
    dict
        Variance of predicted probabilities per split.
    dict
        Ground truth labels per split.
    """
    # NOTE! to make things simpler, we get the data from the stats subdict which are averaged from n bootstrap iters
    # this does not account the case in which you did not use same number of bootstrap iters (if you care), assuming
    # that all the models used the same number of iterations
    y_pred_proba, y_pred_proba_var, label = {}, {}, {}
    for split in artifacts["subjectwise_stats"].keys():
        y_pred_proba[split] = artifacts["subjectwise_stats"][split]["preds"][
            "y_pred_proba"
        ]["mean"]
        y_pred_proba_var[split] = (
            artifacts["subjectwise_stats"][split]["preds"]["y_pred_proba"]["std"] ** 2
        )
        label[split] = artifacts["subjectwise_stats"][split]["preds"]["label"]["mean"]

    return y_pred_proba, y_pred_proba_var, label


def import_model_preds_and_labels(
    run_id: str, run_name: str, model_name: str, subdir: str = "metrics"
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Load predictions and labels from MLflow run.

    Parameters
    ----------
    run_id : str
        MLflow run ID.
    run_name : str
        MLflow run name.
    model_name : str
        Name of the model.
    subdir : str, default 'metrics'
        Subdirectory within artifacts.

    Returns
    -------
    dict
        Mean predicted probabilities per split.
    dict
        Variance of predicted probabilities per split.
    dict
        Ground truth labels per split.
    """
    artifacts = import_model_metrics(run_id, run_name, model_name, subdir=subdir)
    y_pred_proba, y_pred_proba_var, label = get_preds_and_labels_from_artifacts(
        artifacts
    )
    return y_pred_proba, y_pred_proba_var, label


def import_metrics_iter(
    run_id: str, run_name: str, model_name: str, subdir: str = "metrics"
) -> dict[str, Any]:
    """
    Load per-iteration metrics from MLflow run.

    Parameters
    ----------
    run_id : str
        MLflow run ID.
    run_name : str
        MLflow run name.
    model_name : str
        Name of the model.
    subdir : str, default 'metrics'
        Subdirectory within artifacts.

    Returns
    -------
    dict
        Dictionary of metrics per bootstrap iteration per split.
    """
    artifacts = import_model_metrics(run_id, run_name, model_name, subdir=subdir)
    return artifacts["metrics_iter"]


def concentate_one_var(
    array_out: dict[str, np.ndarray] | None,
    array_per_submodel: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Concatenate arrays from submodel into accumulated output.

    Stacks submodel arrays along a new first axis.

    Parameters
    ----------
    array_out : dict or None
        Accumulated arrays (None for first submodel).
    array_per_submodel : dict
        Arrays from current submodel, keyed by split.

    Returns
    -------
    dict
        Updated accumulated arrays.
    """
    if array_out is None:
        array_out = {}
        for split in array_per_submodel.keys():
            array_out[split] = array_per_submodel[split][np.newaxis, :]
    else:
        for split in array_out.keys():
            array_out[split] = np.concatenate(
                [array_out[split], array_per_submodel[split][np.newaxis, :]], axis=0
            )

    return array_out


def concatenate_arrays(
    preds_out: dict[str, np.ndarray] | None,
    preds_var_out: dict[str, np.ndarray] | None,
    _labels_out: dict[str, np.ndarray] | None,
    y_pred_proba: dict[str, np.ndarray],
    y_pred_proba_var: dict[str, np.ndarray],
    label: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Concatenate prediction arrays from multiple submodels.

    Parameters
    ----------
    preds_out : dict or None
        Accumulated predictions.
    preds_var_out : dict or None
        Accumulated prediction variances.
    _labels_out : dict or None
        Accumulated labels (not modified, labels are same across models).
    y_pred_proba : dict
        Predictions from current submodel.
    y_pred_proba_var : dict
        Prediction variances from current submodel.
    label : dict
        Labels from current submodel.

    Returns
    -------
    dict
        Updated predictions.
    dict
        Updated variances.
    dict
        Labels (unchanged).
    """
    preds_out = concentate_one_var(preds_out, y_pred_proba)
    preds_var_out = concentate_one_var(preds_var_out, y_pred_proba_var)
    # should be the same, thus no need for this
    # labels_out = concentate_one_var(labels_out, label)

    return preds_out, preds_var_out, label


def check_dicts(
    preds_out: dict[str, np.ndarray],
    preds_var_out: dict[str, np.ndarray],
    _labels_out: dict[str, np.ndarray] | None,
    no_submodel_runs: int,
) -> None:
    """
    Verify concatenated arrays have expected dimensions.

    Parameters
    ----------
    preds_out : dict
        Accumulated predictions.
    preds_var_out : dict
        Accumulated variances.
    _labels_out : dict
        Labels (unused, kept for API compatibility).
    no_submodel_runs : int
        Expected number of submodels.

    Raises
    ------
    AssertionError
        If array dimensions don't match expected submodel count.
    """
    for split in preds_out.keys():
        assert preds_out[split].shape[0] == no_submodel_runs, (
            f"preds_out[split].shape[0]: "
            f"{preds_out[split].shape[0]}, "
            f"no_submodel_runs: {no_submodel_runs}"
        )
        assert preds_var_out[split].shape[0] == no_submodel_runs, (
            f"preds_var_out[split].shape[0]: "
            f"{preds_var_out[split].shape[0]}, "
            f"no_submodel_runs: {no_submodel_runs}"
        )


def compute_stats(
    preds_out: dict[str, np.ndarray], preds_var_out: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Compute ensemble statistics from stacked predictions.

    Parameters
    ----------
    preds_out : dict
        Stacked predictions (n_models x n_subjects).
    preds_var_out : dict
        Stacked variances (n_models x n_subjects).

    Returns
    -------
    dict
        Mean predictions across models.
    dict
        Standard deviation of predictions across models.
    dict
        Mean of within-model standard deviations.
    """
    preds = {}
    preds_std = {}
    preds_meanstd = {}
    for split in preds_out.keys():
        preds[split] = np.mean(preds_out[split], axis=0)
        preds_std[split] = np.std(preds_out[split], axis=0)
        preds_meanstd[split] = np.mean(preds_var_out[split], axis=0) ** 0.5

    return preds, preds_std, preds_meanstd


def aggregate_pred_dict(
    preds_out: dict[str, dict[str, list[Any]]],
    preds_per_submodel: dict[str, dict[str, list[Any]]],
    ensemble: bool = False,
) -> dict[str, dict[str, list[Any]]]:
    """
    Aggregate prediction dictionaries from submodels.

    Extends lists of predictions for each subject code.

    Parameters
    ----------
    preds_out : dict
        Accumulated predictions keyed by variable and subject code.
    preds_per_submodel : dict
        Predictions from current submodel.
    ensemble : bool, default False
        If True, performs additional consistency checks.

    Returns
    -------
    dict
        Updated accumulated predictions.
    """
    for var in preds_per_submodel:  # e.g. y_pred_proba, y_pred, label
        assert isinstance(preds_per_submodel[var], dict), (
            f"preds_per_submodel[var] is not a dict, "
            f"but {type(preds_per_submodel[var])}"
        )
        unique_codes_out = sorted(list(preds_out[var].keys()))
        unique_codes_in = sorted(list(preds_per_submodel[var].keys()))
        if ensemble:
            assert unique_codes_out == unique_codes_in, (
                "You do not have to have the same subjects in all splits? \n"
                "As in you ran some MLflow with runs with certain subjects,\n"
                "And later redefined the splits?"
            )
        for code in preds_per_submodel[var]:
            list_of_preds = preds_per_submodel[var][code]
            # no_of_bootstrap_iters = len(list_of_preds)
            preds_out[var][code] += list_of_preds

    return preds_out


def aggregate_preds(
    preds_out: dict[str, np.ndarray], preds_per_submodel: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """
    Concatenate prediction arrays along iteration axis.

    Parameters
    ----------
    preds_out : dict
        Accumulated predictions (n_subjects x n_accumulated_iters).
    preds_per_submodel : dict
        Predictions from current submodel.

    Returns
    -------
    dict
        Updated predictions with new iterations concatenated.
    """
    for split in preds_per_submodel:
        preds_out[split] = np.concatenate(
            [preds_out[split], preds_per_submodel[split]], axis=1
        )
    return preds_out


def check_metrics_iter_preds_dict(
    dict_arrays: dict[str, dict[str, list[Any]]],
) -> None:
    """
    Validate prediction dictionary has consistent dimensions.

    Checks that y_pred_proba, y_pred, and labels have same length.

    Parameters
    ----------
    dict_arrays : dict
        Dictionary with 'y_pred_proba', 'y_pred', and 'label'/'labels' keys.

    Raises
    ------
    AssertionError
        If lengths don't match.
    """
    # (no_subjects, no_of_bootstrap_iters)
    if "labels" in dict_arrays:
        # TODO! why with CATBOOST? fix this eventually so that only one key
        assert len(dict_arrays["y_pred_proba"]) == len(dict_arrays["labels"]), (
            f"you have {len(dict_arrays['y_pred_proba'])} y_pred_proba and {len(dict_arrays['labels'])} labels"
        )
    elif "label" in dict_arrays:
        assert len(dict_arrays["y_pred_proba"]) == len(dict_arrays["label"]), (
            f"you have {len(dict_arrays['y_pred_proba'])} y_pred_proba and {len(dict_arrays['label'])} labels"
        )
    assert len(dict_arrays["y_pred_proba"]) == len(dict_arrays["y_pred"])


def check_metrics_iter_preds(dict_arrays: dict[str, np.ndarray]) -> None:
    """
    Validate prediction arrays have consistent shapes.

    Checks that y_pred_proba, y_pred, and labels have same first dimension.

    Parameters
    ----------
    dict_arrays : dict
        Dictionary with numpy array values.

    Raises
    ------
    AssertionError
        If shapes don't match.
    """
    # (no_subjects, no_of_bootstrap_iters)
    if "label" in dict_arrays:
        assert dict_arrays["y_pred_proba"].shape[0] == dict_arrays["label"].shape[0], (
            f"you have {dict_arrays['y_pred_proba'].shape[0]} y_pred_proba and {dict_arrays['label'].shape[0]} labels"
        )
    elif "labels" in dict_arrays:
        # TODO! why with CATBOOST? fix this eventually so that only one key
        assert dict_arrays["y_pred_proba"].shape[0] == dict_arrays["labels"].shape[0], (
            f"you have {dict_arrays['y_pred_proba'].shape[0]} y_pred_proba and {dict_arrays['labels'].shape[0]} labels"
        )
    assert dict_arrays["y_pred_proba"].shape[0] == dict_arrays["y_pred"].shape[0]


def check_metrics_iter_shapes(iter_split: dict[str, Any]) -> None:
    """
    Dispatch shape checking based on data structure type.

    Parameters
    ----------
    iter_split : dict
        Split-level metrics iteration data.
    """
    if "preds_dict" in iter_split:
        check_metrics_iter_preds_dict(dict_arrays=iter_split["preds_dict"]["arrays"])
    else:
        check_metrics_iter_preds(
            dict_arrays=iter_split["preds"]["arrays"]["predictions"]
        )


def check_subjects_in_splits(
    metrics_iter: dict[str, Any] | None,
) -> dict[str, list[str]] | None:
    """
    Extract and validate subject codes from metrics iteration data.

    Parameters
    ----------
    metrics_iter : dict or None
        Metrics per iteration dictionary.

    Returns
    -------
    dict or None
        Dictionary mapping splits to sorted subject code lists.
    """
    if metrics_iter is not None:
        subjects = {}
        for split in metrics_iter.keys():
            if "preds_dict" in metrics_iter[split]:
                subjects[split] = sorted(
                    list(
                        metrics_iter[split]["preds_dict"]["arrays"][
                            "y_pred_proba"
                        ].keys()
                    )
                )
            elif "preds" in metrics_iter[split]:
                no_subjects_preds = len(
                    metrics_iter[split]["preds"]["arrays"]["predictions"][
                        "y_pred_proba"
                    ]
                )
                logger.debug(f"{no_subjects_preds} in test split")
            else:
                logger.error("Where are the preds?")
                raise ValueError("Where are the preds?")

        if "val" in subjects:
            assert subjects["train"] == subjects["val"], (
                "Your train and val codes do not match?"
            )

        return subjects
    else:
        return None


def check_compare_subjects_for_aggregation(
    subject_codes: dict[str, list[str]] | None,
    subject_codes_model: dict[str, list[str]] | None,
    run_name: str,
    i: int,
    split_to_check: str = "train",
) -> list[str]:
    """
    Compare subject codes between ensemble and current submodel.

    Verifies that the current submodel used same subjects as previous models.

    Parameters
    ----------
    subject_codes : dict or None
        Subject codes from ensemble (accumulated).
    subject_codes_model : dict or None
        Subject codes from current submodel.
    run_name : str
        Name of current run for error reporting.
    i : int
        Index of current submodel.
    split_to_check : str, default 'train'
        Which split to compare.

    Returns
    -------
    list
        List of run names with mismatched codes (empty if match).
    """
    error_run = []
    if subject_codes is not None and subject_codes_model is not None:
        for split in subject_codes.keys():
            # these come from whole bootstrap experiment, so train and val should have the same codes
            if split == split_to_check:
                if subject_codes[split] != subject_codes_model[split]:
                    logger.error(
                        f"Submodel #{i + 1}: {run_name} seem to have different subjects in splits than in previous submodels"
                    )
                    error_run = [run_name]
                    if len(subject_codes[split]) != len(subject_codes_model[split]):
                        logger.error(
                            "Lengths of splits do not even seem to match, ensemble n = {}, submodel n = {}".format(
                                len(subject_codes[split]),
                                len(subject_codes_model[split]),
                            )
                        )
                        # raise ValueError('Your ensemble seem to come from different splits, did you redefine the splits\n'
                        #                  'while running the experiments? Need to delete the runs with old split definitions\n'
                        #                  'and rerun those to get this ensembling working?')
                    else:
                        logger.debug("Ensemble codes | Model Codes")
                        for code_ens, code in zip(
                            subject_codes[split], subject_codes_model[split]
                        ):
                            logger.debug(f"{code_ens} | {code}")
                        # raise ValueError('Your ensemble seem to come from different splits, did you redefine the splits\n'
                        #                  'while running the experiments? Need to delete the runs with old split definitions\n'
                        #                  'and rerun those to get this ensembling working?')

    return error_run


def aggregate_metric_iter(
    metrics_iter: dict[str, Any] | None,
    metrics_iter_model: dict[str, Any],
    run_name: str,
    ensemble: bool = False,
) -> dict[str, Any]:
    """
    Aggregate metrics iteration data from submodel into ensemble.

    Combines predictions from multiple bootstrap models by extending
    the iteration arrays.

    Parameters
    ----------
    metrics_iter : dict or None
        Accumulated metrics (None for first submodel).
    metrics_iter_model : dict
        Metrics from current submodel.
    run_name : str
        Name of current run for logging.
    ensemble : bool, default False
        If True, performs additional consistency checks.

    Returns
    -------
    dict
        Updated accumulated metrics.
    """
    if metrics_iter is None:
        metrics_iter = {}
        for split in metrics_iter_model.keys():
            metrics_iter[split] = metrics_iter_model[split]
            metrics_iter[split].pop("metrics")

    else:
        for split in metrics_iter.keys():
            metrics_iter_model[split].pop("metrics")
            if "preds_dict" in metrics_iter_model[split]:
                # train/val
                preds: dict[str, dict] = metrics_iter_model[split]["preds_dict"][
                    "arrays"
                ]
                metrics_iter[split]["preds_dict"]["arrays"] = aggregate_pred_dict(
                    preds_out=metrics_iter[split]["preds_dict"]["arrays"],
                    preds_per_submodel=preds,
                    ensemble=ensemble,
                )
                check_metrics_iter_preds_dict(
                    dict_arrays=metrics_iter[split]["preds_dict"]["arrays"]
                )

            else:
                # test
                preds: dict[str, np.ndarray] = metrics_iter_model[split]["preds"][
                    "arrays"
                ]["predictions"]
                metrics_iter[split]["preds"]["arrays"]["predictions"] = aggregate_preds(
                    preds_out=metrics_iter[split]["preds"]["arrays"]["predictions"],
                    preds_per_submodel=preds,
                )
                check_metrics_iter_preds(
                    dict_arrays=metrics_iter[split]["preds"]["arrays"]["predictions"]
                )

    return metrics_iter


def get_label_array(label_dict: dict[str, np.ndarray]) -> np.ndarray:
    """
    Convert label dictionary to array.

    Parameters
    ----------
    label_dict : dict
        Dictionary mapping subject codes to label arrays.

    Returns
    -------
    np.ndarray
        Array of labels in same order as dictionary keys.
    """
    label_array = []
    for code in label_dict.keys():
        label_array.append(label_dict[code][0])
    label_array = np.array(label_array)
    assert label_array.shape[0] == len(label_dict), (
        f"label_array.shape[0]: {label_array.shape[0]}, "
        f"len(label_dict): {len(label_dict)}"
    )
    return label_array


def get_preds_array(preds_dict: dict[str, np.ndarray]) -> np.ndarray:
    """
    Convert prediction dictionary to 2D array.

    Parameters
    ----------
    preds_dict : dict
        Dictionary mapping subject codes to prediction arrays.

    Returns
    -------
    np.ndarray
        Array of shape (n_subjects, n_bootstrap_iters).
    """

    def get_min_bootstrap_iters_from_subjects(
        preds_dict: dict[str, np.ndarray],
    ) -> int:
        lengths = []
        for code in preds_dict.keys():
            lengths.append(len(preds_dict[code]))
        return min(lengths)

    array_iter_no = get_min_bootstrap_iters_from_subjects(preds_dict)
    preds_array = np.zeros((len(preds_dict), array_iter_no))
    for i, code in enumerate(preds_dict.keys()):
        preds_array[i] = preds_dict[code][:array_iter_no]

    return preds_array


def recompute_ensemble_metrics(
    metrics_iter: dict[str, Any], sources: dict[str, Any], cfg: DictConfig
) -> dict[str, Any]:
    """
    Recompute metrics for aggregated ensemble predictions.

    Takes combined predictions from all submodels and recomputes
    bootstrap metrics as if they were from a single model.

    Parameters
    ----------
    metrics_iter : dict
        Aggregated predictions from all submodels.
    sources : dict
        Source data containing subject information.
    cfg : DictConfig
        Main Hydra configuration.

    Returns
    -------
    dict
        Updated metrics_iter with recomputed metrics.
    """
    warnings.simplefilter("ignore")
    # skip "val" for now
    splits = ["train", "test"]
    for split in splits:  # metrics_iter.keys():
        if "preds_dict" in metrics_iter[split]:
            y_true = get_label_array(
                label_dict=metrics_iter[split]["preds_dict"]["arrays"]["label"]
            )
            preds_array = get_preds_array(
                preds_dict=metrics_iter[split]["preds_dict"]["arrays"]["y_pred_proba"]
            )

        elif "preds" in metrics_iter[split]:
            y_true = metrics_iter[split]["preds"]["arrays"]["predictions"]["label"][
                :, 0
            ]
            preds_array = metrics_iter[split]["preds"]["arrays"]["predictions"][
                "y_pred_proba"
            ]

        else:
            logger.error(
                "Where are the predictions? {}".format(metrics_iter[split].keys())
            )
            raise ValueError(
                "Where are the predictions? {}".format(metrics_iter[split].keys())
            )

        method_cfg = cfg["CLS_EVALUATION"]["BOOTSTRAP"]
        dict_arrays_compact = get_compact_dict_arrays(sources)
        codes_per_split = dict_arrays_compact[f"subject_codes_{split}"]

        for idx in tqdm(
            range(preds_array.shape[1]),
            desc=f"Recomputing ensemble metrics for {split}",
        ):
            preds = create_pred_dict(split_preds=preds_array[:, idx], y_true=y_true)
            metrics_iter[split] = bootstrap_metrics_per_split(
                None,
                y_true,
                preds,
                None,
                model_name="ensemble",
                metrics_per_split=metrics_iter[split],
                codes_per_split=codes_per_split,
                method_cfg=method_cfg,
                cfg=cfg,
                split=split,
                skip_mlflow=True,
                recompute_for_ensemble=True,
            )
            check_metrics_iter_shapes(iter_split=metrics_iter[split])

    warnings.resetwarnings()

    return metrics_iter


def get_cls_preds_from_artifact(
    run: pd.Series, i: int, no_submodel_runs: int, aggregate_preds: bool = False
) -> dict[str, Any]:
    """
    Load classification predictions from MLflow run artifact.

    Parameters
    ----------
    run : pd.Series
        MLflow run data.
    i : int
        Index of current submodel (for logging).
    no_submodel_runs : int
        Total number of submodels (for logging).
    aggregate_preds : bool, default False
        If True, log aggregation progress.

    Returns
    -------
    dict
        Metrics per iteration dictionary from the run.
    """
    run_id = run["run_id"]
    run_name = run["tags.mlflow.runName"]
    model_name = run["params.model_name"]
    if aggregate_preds:
        logger.info(
            f"{i + 1}/{no_submodel_runs}: Ensembling model: {model_name}, run_id: {run_id}, run_name: {run_name}"
        )
    # Baseline (as in no bootstrapping)
    # preds_baseline, labels_baseline = import_model_preds_and_labels(run_id, run_name, model_name, subdir='baseline_model')

    # Bootstrapped model
    # y_pred_proba, y_pred_proba_var, label = (
    #     import_model_preds_and_labels(run_id, run_name, model_name, subdir='metrics'))
    # preds_out, preds_var_out, labels_out = concatenate_arrays(
    #     preds_out, preds_var_out, labels_out, y_pred_proba, y_pred_proba_var, label
    # )
    metrics_iter_model = import_metrics_iter(
        run_id, run_name, model_name, subdir="metrics"
    )
    n = metrics_iter_model["test"]["preds"]["arrays"]["predictions"][
        "y_pred_proba"
    ].shape[1]
    if aggregate_preds:
        logger.info("Submodel consists of {} bootstrap iterations".format(n))

    return metrics_iter_model


def aggregate_submodels(
    ensemble_model_runs: pd.DataFrame,
    no_submodel_runs: int,
    aggregate_preds: bool = True,
    split_to_check: str = "train",
    ensemble_codes: pd.DataFrame | None = None,
) -> tuple[dict[str, Any] | None, pd.DataFrame, bool]:
    """
    Aggregate predictions from multiple classification submodels.

    Parameters
    ----------
    ensemble_model_runs : pd.DataFrame
        DataFrame of MLflow runs to aggregate.
    no_submodel_runs : int
        Number of submodels.
    aggregate_preds : bool, default True
        If True, actually aggregate predictions. If False, only check codes.
    split_to_check : str, default 'train'
        Split to use for code consistency checking.
    ensemble_codes : pd.DataFrame, optional
        Pre-computed ensemble codes for validation.

    Returns
    -------
    dict or None
        Aggregated metrics_iter (or None if aggregate_preds=False).
    pd.DataFrame
        DataFrame of subject codes per submodel.
    bool
        Whether all submodels have same subject codes.
    """
    metrics_iter = None
    subject_codes_out = {}
    error_runs = []

    for i, (idx, run) in enumerate(ensemble_model_runs.iterrows()):
        metrics_iter_model = get_cls_preds_from_artifact(
            run, i, no_submodel_runs, aggregate_preds=aggregate_preds
        )
        subject_codes = check_subjects_in_splits(metrics_iter)
        subject_codes_model = check_subjects_in_splits(metrics_iter_model)
        error_runs += check_compare_subjects_for_aggregation(
            subject_codes, subject_codes_model, run["tags.mlflow.runName"], i
        )
        subject_codes_out[run["tags.mlflow.runName"]] = subject_codes_model[
            split_to_check
        ]
        df_out = pd.DataFrame()

        if aggregate_preds:
            metrics_iter = aggregate_metric_iter(
                metrics_iter,
                metrics_iter_model,
                run_name=run["tags.mlflow.runName"],
                ensemble=True,
            )

    for run_name, list_of_codes in subject_codes_out.items():
        df_out[run_name] = list_of_codes
    all_submodels_have_same_codes = are_codes_the_same(df_out)

    if not aggregate_preds:
        if len(error_runs) > 0:
            # well assuming that the first submodel of the ensemble is correct
            logger.error(
                "These runs seem to be done with a different set of subjects than others?"
            )
            for run in error_runs:
                logger.error(run)
            raise ValueError(
                "Your ensemble seem to come from different splits, did you redefine the splits\n"
                "while running the experiments? Need to delete the runs with old split definitions\n"
                "and rerun those to get this ensembling working?"
            )

    return metrics_iter, df_out, all_submodels_have_same_codes


def get_classification_preds(
    ensemble_model_runs: pd.DataFrame, sources: dict[str, Any], cfg: DictConfig
) -> dict[str, Any] | None:
    """
    Get aggregated classification predictions from submodels.

    Coordinates the full aggregation process: code checking, prediction
    aggregation, and metric recomputation.

    Parameters
    ----------
    ensemble_model_runs : pd.DataFrame
        DataFrame of MLflow runs to ensemble.
    sources : dict
        Source data.
    cfg : DictConfig
        Main Hydra configuration.

    Returns
    -------
    dict or None
        Aggregated metrics_iter with recomputed metrics, or None if failed.
    """
    no_submodel_runs = ensemble_model_runs.shape[0]
    if no_submodel_runs > 0:
        _, ensemble_codes, same_codes = aggregate_submodels(
            ensemble_model_runs, no_submodel_runs, aggregate_preds=False
        )
        if same_codes:
            metrics_iter, _, _ = aggregate_submodels(
                ensemble_model_runs,
                no_submodel_runs,
                aggregate_preds=True,
                ensemble_codes=ensemble_codes,
            )
            # This metrics_iter is now the same that you would get from the "normal bootstrap" with just all the
            # iterations aggregated together
            n = metrics_iter["test"]["preds"]["arrays"]["predictions"][
                "y_pred_proba"
            ].shape[1]
            logger.info(
                "Ensemble consists a total of {} bootstrap iterations".format(n)
            )

            # compute the metrics for the ensemble
            metrics_iter = recompute_ensemble_metrics(metrics_iter, sources, cfg)
        else:
            logger.warning(
                "The codes used to train different submodels do not seem to be the same!"
            )
            metrics_iter = None
    else:
        metrics_iter = None

    return metrics_iter


def create_pred_dict(
    split_preds: np.ndarray, y_true: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Create prediction dictionary from arrays.

    Parameters
    ----------
    split_preds : np.ndarray
        Predicted probabilities.
    y_true : np.ndarray
        Ground truth labels.

    Returns
    -------
    dict
        Dictionary with y_pred, y_pred_proba, and labels.
    """
    preds = {
        "y_pred": (split_preds > 0.5).astype(int),
        "y_pred_proba": split_preds,
        "labels": y_true,
    }
    return preds


def get_compact_dict_arrays(sources: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Extract compact arrays of subject codes and labels from sources.

    Used by bootstrap metric computation functions.

    Parameters
    ----------
    sources : dict
        Source data dictionary.

    Returns
    -------
    dict
        Dictionary with keys like 'subject_codes_train', 'y_train', etc.
    """
    first_feature_source = list(sources.keys())[0]
    data = sources[first_feature_source]["data"]
    dict_arrays_compact = {}
    for split in data.keys():
        dict_arrays_compact[f"subject_codes_{split}"] = data[split][
            "subject_code"
        ].to_numpy()
        # this is now as a string, e.g. "control" vs. "glaucoma"
        dict_arrays_compact[f"y_{split}"] = data[split][
            "metadata_class_label"
        ].to_numpy()
        # to integers
        dict_arrays_compact[f"y_{split}"] = encode_labels_to_integers(
            dict_arrays_compact[f"y_{split}"]
        )

    return dict_arrays_compact


def compute_cls_ensemble_metrics(
    metrics_iter: dict[str, Any], sources: dict[str, Any], cfg: DictConfig
) -> dict[str, Any]:
    """
    Compute classification ensemble statistics.

    Parameters
    ----------
    metrics_iter : dict
        Aggregated predictions from ensemble.
    sources : dict
        Source data.
    cfg : DictConfig
        Main Hydra configuration.

    Returns
    -------
    dict
        Dictionary containing metrics_iter, metrics_stats, subjectwise_stats,
        and subject_global_stats.
    """
    method_cfg = cfg["CLS_EVALUATION"]["BOOTSTRAP"]
    dict_arrays_compact = get_compact_dict_arrays(sources)
    metrics_stats, subjectwise_stats, subject_global_stats = get_ensemble_stats(
        metrics_iter,
        dict_arrays_compact,
        method_cfg,
        call_from="classification_ensemble",
    )

    metrics = {
        "metrics_iter": metrics_iter,
        "metrics_stats": metrics_stats,
        "subjectwise_stats": subjectwise_stats,
        "subject_global_stats": subject_global_stats,
    }

    return metrics


def ensemble_classification(
    ensemble_model_runs: pd.DataFrame,
    cfg: DictConfig,
    sources: dict[str, Any],
    ensemble_name: str,
) -> dict[str, Any] | None:
    """
    Create classification ensemble from multiple models.

    Main entry point for classification ensembling. Aggregates predictions
    from submodels and computes ensemble metrics.

    Parameters
    ----------
    ensemble_model_runs : pd.DataFrame
        DataFrame of MLflow classification runs to ensemble.
    cfg : DictConfig
        Main Hydra configuration.
    sources : dict
        Source data.
    ensemble_name : str
        Name for the ensemble.

    Returns
    -------
    dict or None
        Ensemble metrics dictionary, or None if ensembling failed.
    """
    # Get imputation mask and labels for each model
    metrics_iter = get_classification_preds(ensemble_model_runs, sources, cfg=cfg)

    # Compute the metrics for the ensemble
    if metrics_iter is not None:
        metrics = compute_cls_ensemble_metrics(metrics_iter, sources, cfg)
    else:
        metrics = None

    return metrics
