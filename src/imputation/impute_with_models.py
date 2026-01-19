import time

import numpy as np
from loguru import logger
import mlflow

from omegaconf import DictConfig

from src.imputation.pypots.pypots_utils import define_pypots_outputs
from src.log_helpers.local_artifacts import save_results_dict


def evaluate_pypots_model(model, dataset_dict, split: str, cfg: DictConfig):
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
