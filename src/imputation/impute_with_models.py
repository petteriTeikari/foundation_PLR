import time

import mlflow
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from src.imputation.pypots.pypots_utils import define_pypots_outputs
from src.log_helpers.local_artifacts import save_results_dict


def evaluate_pypots_model(model, dataset_dict, split: str, cfg: DictConfig):
    """Evaluate a PyPOTS model by imputing missing values in the dataset.

    Runs the trained PyPOTS model on the provided dataset to impute
    missing values, handling both deterministic and probabilistic outputs.

    Parameters
    ----------
    model : PyPOTS model
        Trained PyPOTS imputation model (e.g., SAITS, CSDI, TimesNet).
    dataset_dict : dict
        Dataset dictionary containing 'X' array with NaN values to impute.
    split : str
        Data split name ('train' or 'test').
    cfg : DictConfig
        Configuration (unused but kept for interface consistency).

    Returns
    -------
    dict
        Imputation results containing:
        - 'imputation_dict': Dict with 'imputation' (mean, CI bounds) and
          'indicating_mask' (boolean mask of originally missing values)
        - 'timing': Elapsed time for imputation in seconds

    Raises
    ------
    ValueError
        If imputation output has unexpected shape (not 3D or 4D).

    Notes
    -----
    CSDI generates a 4D output with samples dimension, which is reduced
    to 3D by taking the first sample. Other models produce 3D output directly.
    """
    start_time = time.time()
    imputation = model.impute(dataset_dict)
    end_time = time.time() - start_time

    # indicating mask for imputation error calculation
    indicating_mask = np.isnan(dataset_dict["X"])

    # Save the imputation results
    if len(imputation.shape) == 3:
        # deterministic imputation
        imputation_mean = imputation
        imputation_ci_neg = None
        imputation_ci_pos = None
    elif len(imputation.shape) == 4:
        # CSDI generates a new dimension for the imputation
        # but by default it has only value? how to get the "spread"?
        imputation_mean = imputation[:, 0, :, :]
        imputation_ci_neg = None
        imputation_ci_pos = None
    else:
        logger.error("Unknown shape of the imputation results")
        raise ValueError("Unknown shape of the imputation results")

    # See "create_imputation_dict()" in src/imputation/train_utils.py
    # Combine these later
    imputation_dict = {
        "imputation_dict": {
            "imputation": {
                "mean": imputation_mean,
                "imputation_ci_neg": imputation_ci_neg,
                "imputation_ci_pos": imputation_ci_pos,
            },
            "indicating_mask": indicating_mask,
        },
        "timing": end_time,
    }

    return imputation_dict


def log_imputed_artifacts(
    imputation: dict, model_name: str, cfg: DictConfig, run_id: str
):
    """Save imputation results locally and log to MLflow.

    Parameters
    ----------
    imputation : dict
        Imputation results dictionary to save.
    model_name : str
        Name of the imputation model for file naming.
    cfg : DictConfig
        Configuration (unused but kept for interface consistency).
    run_id : str
        MLflow run ID to log artifacts to.

    Returns
    -------
    str
        Path to the saved artifacts file.
    """
    output_dir, fname, artifacts_path = define_pypots_outputs(
        model_name=model_name, artifact_type="imputation"
    )
    save_results_dict(imputation, artifacts_path, name="imputation")
    # with mlflow.start_run(run_id=run_id):
    mlflow.log_artifact(artifacts_path, artifact_path="results")

    return artifacts_path


# @task(
#     log_prints=True,
#     name="PyPOTS: Impute PLR data",
#     description="Impute the missing data with the trained models",
# )
def pypots_imputer_wrapper(model, model_name, dataset_dicts, source_data, cfg):
    """Wrapper to impute data across all splits using a PyPOTS model.

    Iterates over data splits and applies the trained PyPOTS model
    to impute missing values in each split.

    Parameters
    ----------
    model : PyPOTS model
        Trained PyPOTS imputation model.
    model_name : str
        Name of the model for logging.
    dataset_dicts : dict
        Dictionary of datasets keyed by split name (e.g., 'train', 'test').
    source_data : dict
        Source data dictionary (unused but kept for interface consistency).
    cfg : DictConfig
        Configuration for imputation settings.

    Returns
    -------
    dict
        Dictionary mapping split names to their imputation results.
    """
    logger.info("Evaluate (impute) the model on the data")
    imputed_dict = {}
    for i, split in enumerate(dataset_dicts.keys()):
        imputed_dict[split] = evaluate_pypots_model(
            model=model,
            dataset_dict=dataset_dicts[split],
            split=split,
            cfg=cfg,
        )

    return imputed_dict
