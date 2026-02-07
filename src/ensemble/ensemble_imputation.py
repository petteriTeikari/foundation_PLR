"""
Imputation ensemble module.

Provides functionality to combine imputation outputs from multiple models
by averaging reconstructions and computing ensemble statistics.

Cross-references:
- src/metrics/evaluate_imputation_metrics.py for metric computation
- src/ensemble/ensemble_utils.py for run retrieval
"""

import warnings

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from src.anomaly_detection.anomaly_utils import get_artifact
from src.ensemble.ensemble_utils import (
    get_gt_imputation_labels,
    get_metadata_dict_from_sources,
)
from src.log_helpers.local_artifacts import load_results_dict
from src.metrics.evaluate_imputation_metrics import (
    compute_imputation_metrics,
    recompute_submodel_imputation_metrics,
)
from src.preprocess.preprocess_data import destandardize_for_imputation_metric
from src.preprocess.preprocess_PLR import standardize_recons_arrays


def get_imputation_results_per_model(model_name, outlier_artifacts):
    """
    Extract imputation reconstructions from model artifacts.

    Handles different artifact structures and destandardizes values
    for proper metric computation.

    Parameters
    ----------
    model_name : str
        Name of the imputation model.
    outlier_artifacts : dict
        Loaded artifacts from MLflow.

    Returns
    -------
    dict
        Reconstructions per split (destandardized).
    dict
        True pupil values per split (destandardized).
    dict
        Imputation indicating masks per split.
    """
    reconstructions = {"train": [], "test": []}
    true_pupil = {"train": [], "test": []}
    labels = {"train": [], "test": []}

    split = "train"
    imputation_dict = outlier_artifacts["model_artifacts"]["imputation"][split]
    # # (no_samples, no_timepoints, no_feats) -> (no_samples, no_timepoints) # e.g. (355,1981)
    if len(imputation_dict["imputation_dict"]["imputation"]["mean"].shape) == 3:
        reconstructions[split] = imputation_dict["imputation_dict"]["imputation"][
            "mean"
        ][:, :, 0]
    elif len(imputation_dict["imputation_dict"]["imputation"]["mean"].shape) == 3:
        reconstructions[split] = imputation_dict["imputation_dict"]["imputation"][
            "mean"
        ][:, :]

    try:
        true_pupil[split] = outlier_artifacts["source_data"]["df"][split]["data"]["X"]
    except Exception as e:
        logger.error(e)
        raise e

    if len(imputation_dict["imputation_dict"]["indicating_mask"].shape) == 3:
        labels[split] = imputation_dict["imputation_dict"]["indicating_mask"][
            :, :, 0
        ].astype(int)
    elif len(imputation_dict["imputation_dict"]["indicating_mask"].shape) == 2:
        labels[split] = imputation_dict["imputation_dict"]["indicating_mask"][
            :, :
        ].astype(int)
    else:
        logger.error(
            "Unknown shape of imputation indicating mask, shape = {}".format(
                true_pupil[split].shape
            )
        )
        raise ValueError(
            "Unknown shape of imputation indicating mask, shape = {}".format(
                true_pupil[split].shape
            )
        )

    assert true_pupil[split].shape == labels[split].shape, (
        f"true_pupil[split].shape: {true_pupil[split].shape}, "
        f"labels[split].shape: {labels[split].shape}"
    )

    split = "test"
    imputation_dict = outlier_artifacts["model_artifacts"]["imputation"][split]
    reconstructions[split] = imputation_dict["imputation_dict"]["imputation"]["mean"][
        :, :, 0
    ]
    true_pupil[split] = outlier_artifacts["source_data"]["df"][split]["data"]["X"]
    if len(imputation_dict["imputation_dict"]["indicating_mask"].shape) == 3:
        labels[split] = imputation_dict["imputation_dict"]["indicating_mask"][
            :, :, 0
        ].astype(int)
    elif len(imputation_dict["imputation_dict"]["indicating_mask"].shape) == 2:
        labels[split] = imputation_dict["imputation_dict"]["indicating_mask"][
            :, :
        ].astype(int)
    else:
        logger.error(
            "Unknown shape of imputation indicating mask, shape = {}".format(
                true_pupil[split].shape
            )
        )
        raise ValueError(
            "Unknown shape of imputation indicating mask, shape = {}".format(
                true_pupil[split].shape
            )
        )

    assert true_pupil[split].shape == labels[split].shape, (
        f"true_pupil[split].shape: {true_pupil[split].shape}, "
        f"labels[split].shape: {labels[split].shape}"
    )

    stdz_dict = outlier_artifacts["source_data"]["preprocess"]["standardization"]

    true_pupil["train"], reconstructions["train"] = destandardize_for_imputation_metric(
        true_pupil["train"], reconstructions["train"], stdz_dict
    )
    true_pupil["test"], reconstructions["test"] = destandardize_for_imputation_metric(
        true_pupil["test"], reconstructions["test"], stdz_dict
    )

    # if np.max(reconstructions["test"]) > 150:
    #     # check for stdz errors
    #     logger.error("Reconstruction values are too high, check for stdz errors")
    #     logger.error(
    #         f"Reconstruction values: min = {np.min(reconstructions)}, "
    #         f"max = {np.max(reconstructions)}"
    #     )
    #     raise ValueError("Reconstruction values are too high, check for stdz errors")

    return reconstructions, true_pupil, labels


def get_imputation_preds_and_labels(
    ensemble_model_runs: pd.DataFrame,
    gt_dict: dict,
    gt_preprocess: dict,
    cfg: DictConfig,
):
    """
    Load and stack imputation reconstructions from all submodels.

    Parameters
    ----------
    ensemble_model_runs : pd.DataFrame
        DataFrame of MLflow imputation runs.
    gt_dict : dict
        Ground truth data dictionary.
    gt_preprocess : dict
        Ground truth preprocessing parameters.
    cfg : DictConfig
        Main Hydra configuration.

    Returns
    -------
    dict
        Stacked reconstructions (n_models x n_subjects x n_timepoints).
    dict
        True pupil values (from last model).
    dict
        Imputation masks (from last model).
    """
    reconstructions = {"train": [], "test": []}
    no_submodel_runs = ensemble_model_runs.shape[0]
    no_submodel_ensembled = 0

    submodel_names = []
    for i, (idx, submodel_mlflow_run) in enumerate(ensemble_model_runs.iterrows()):
        run_id = submodel_mlflow_run["run_id"]
        run_name = submodel_mlflow_run["tags.mlflow.runName"]
        model_name = submodel_mlflow_run["params.model"]
        logger.info(
            f"{i + 1}/{no_submodel_runs}: Ensembling model: {model_name}, run_id: {run_id}, run_name: {run_name}"
        )
        try:
            outlier_artifacts_path = get_artifact(
                run_id, run_name, model_name, subdir="imputation"
            )
        except Exception:
            outlier_artifacts_path = None

        if outlier_artifacts_path is None:
            logger.warning(f"Could not load results for {model_name}")
            logger.warning(
                "Was there a glitch, and no results were saved? Or was this non-finished run?"
            )
            continue
        else:
            try:
                outlier_artifacts = load_results_dict(outlier_artifacts_path)
                try:
                    reconstructions_submodel, true_pupil_submodel, labels_submodel = (
                        get_imputation_results_per_model(model_name, outlier_artifacts)
                    )

                    _ = recompute_submodel_imputation_metrics(
                        run_id,
                        submodel_mlflow_run,
                        model_name,
                        gt_dict,
                        gt_preprocess,
                        reconstructions_submodel,
                        cfg,
                    )

                except Exception as e:
                    logger.error(
                        "Problem getting missingness mask and imputation, error = {}".format(
                            e
                        )
                    )
                    raise ValueError(
                        "Problem getting missingness mask and imputation, error = {}".format(
                            e
                        )
                    )

                no_submodel_ensembled += 1
                submodel_names.append(model_name)
                if len(reconstructions["train"]) == 0:
                    reconstructions["train"] = reconstructions_submodel["train"][
                        np.newaxis, ...
                    ]
                    reconstructions["test"] = reconstructions_submodel["test"][
                        np.newaxis, ...
                    ]
                else:
                    # stack numpy arrays to 3d arrays
                    reconstructions["train"] = np.vstack(
                        (
                            reconstructions["train"],
                            reconstructions_submodel["train"][np.newaxis, ...],
                        )
                    )
                    reconstructions["test"] = np.vstack(
                        (
                            reconstructions["test"],
                            reconstructions_submodel["test"][np.newaxis, ...],
                        )
                    )
            except Exception as e:
                logger.warning(f"Error: {e}")

    logger.info(
        "Out of {} submodels, {} were ensembled".format(
            no_submodel_runs, no_submodel_ensembled
        )
    )
    return reconstructions, true_pupil_submodel, labels_submodel


def compute_ensemble_imputation_metrics(recons, true_pupil, labels, cfg, metadata_dict):
    """
    Compute imputation metrics for ensemble (averaged) predictions.

    Parameters
    ----------
    recons : dict
        Stacked reconstructions per split.
    true_pupil : dict
        Ground truth values per split.
    labels : dict
        Imputation masks per split.
    cfg : DictConfig
        Main Hydra configuration.
    metadata_dict : dict
        Metadata per split.

    Returns
    -------
    dict
        Metrics per split.
    np.ndarray
        Ensemble predictions (mean of reconstructions).
    """
    metrics = {}
    for split in true_pupil.keys():
        targets = true_pupil[split]
        missingness_mask = labels[split]
        predictions = np.mean(recons[split], axis=0)
        metrics[split] = compute_imputation_metrics(
            targets,
            predictions,
            missingness_mask,
            cfg=cfg,
            metadata_dict=metadata_dict[split],
        )

    return metrics, predictions


def get_imputation_stats_dict(ensembled_recon, labels, p=0.05):
    """
    Compute ensemble statistics from stacked reconstructions.

    Calculates mean, std, and confidence intervals across submodels.

    Parameters
    ----------
    ensembled_recon : np.ndarray
        Stacked reconstructions (n_models x n_subjects x n_timepoints).
    labels : np.ndarray
        Imputation indicating mask.
    p : float, default 0.05
        Percentile for confidence interval bounds.

    Returns
    -------
    dict
        Dictionary with imputation statistics ready for downstream use.
    """
    warnings.simplefilter("ignore")
    ci = np.nanpercentile(ensembled_recon, [p, 100 - p], axis=0)
    stats_dict = {
        "imputation_dict": {
            "imputation": {
                "mean": np.mean(ensembled_recon, axis=0),
                "std": np.std(ensembled_recon, axis=0),
                "imputation_ci_pos": ci[1, :, :],
                "imputation_ci_neg": ci[0, :, :],
            },
            "indicating_mask": labels,
        }
    }
    warnings.resetwarnings()
    return stats_dict


def add_imputation_dict(recons, predictions, labels, stdz_dict):
    """
    Create imputation output dictionary for downstream processing.

    Standardizes reconstructions and computes ensemble statistics
    in format expected by featurization code.

    Parameters
    ----------
    recons : dict
        Stacked reconstructions per split.
    predictions : np.ndarray
        Ensemble mean predictions.
    labels : dict
        Imputation masks per split.
    stdz_dict : dict
        Standardization parameters.

    Returns
    -------
    dict
        Dictionary with 'model_artifacts' key containing imputation data.

    See Also
    --------
    src.featurization.get_arrays_for_splits_from_imputer_artifacts
    """
    logger.info("Compute Imputation ensemble stats")
    dict_out = {}
    dict_out["model_artifacts"] = {"imputation": {}}
    for split in recons.keys():
        logger.info(f"Split = {split}")
        ensembled_recon = recons[split]  # (no_submodels, no_samples, no_timepoints)
        ensembled_recon = standardize_recons_arrays(ensembled_recon, stdz_dict)
        dict_out["model_artifacts"]["imputation"][split] = get_imputation_stats_dict(
            ensembled_recon, labels[split]
        )

    return dict_out


def ensemble_imputation(
    ensemble_model_runs: pd.DataFrame,
    cfg: DictConfig,
    sources: dict,
    ensemble_name: str,
    recompute_metrics: bool = False,
):
    """
    Create imputation ensemble from multiple models.

    Main entry point for imputation ensembling. Loads reconstructions from
    submodels, averages them, and computes metrics.

    Parameters
    ----------
    ensemble_model_runs : pd.DataFrame
        DataFrame of MLflow imputation runs to ensemble.
    cfg : DictConfig
        Main Hydra configuration.
    sources : dict
        Source data including ground truth.
    ensemble_name : str
        Name for the ensemble.
    recompute_metrics : bool, default False
        If True, only recompute submodel metrics without creating ensemble.

    Returns
    -------
    dict or None
        Ensemble output dictionary with metrics, reconstructions, and
        model artifacts, or None if recompute_metrics=True.

    See Also
    --------
    ensemble_anomaly_detection.get_anomaly_masks_and_labels
    """

    # Get imputation mask and labels for each model
    gt_dict = sources["pupil_gt"]["df"]
    gt_preprocess = sources["pupil_gt"]["preprocess"]
    recons, true_pupil, labels_imputation = get_imputation_preds_and_labels(
        ensemble_model_runs, gt_dict, gt_preprocess, cfg=cfg
    )

    if not recompute_metrics:
        # Compute the metrics for the ensemble(s)
        metadata_dict = get_metadata_dict_from_sources(sources)
        labels = get_gt_imputation_labels(sources)
        metrics, predictions = compute_ensemble_imputation_metrics(
            recons, true_pupil, labels, cfg, metadata_dict
        )

        # Predictions (mean of recons), recons is the ensemble with all the submodel predictions
        stdz_dict = sources["pupil_gt"]["preprocess"]["standardization"]
        ensemble_output = add_imputation_dict(recons, predictions, labels, stdz_dict)
        ensemble_output["metrics"] = metrics
        ensemble_output["recons"] = recons

    else:
        ensemble_output = None
        logger.info(
            "Skipping ensemble imputation as we are now just re-computing metrics for submodels on this pass"
        )

    return ensemble_output
