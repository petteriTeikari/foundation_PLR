import time

import numpy as np
from loguru import logger
import os

from src.preprocess.preprocess_data import destandardize_numpy


def if_results_file_found(results_path):
    if os.path.exists(results_path):
        return True
    else:
        dir, fname = os.path.split(results_path)
        logger.debug(
            "Could not find the results  file from the artifact_dir\n"
            "dir = {}, fname = {}".format(dir, fname)
        )
        return False


def create_imputation_dict(
    imputation_mean, preprocess, X_missing, cfg, end_time: float = None
):
    # Get boolean mask of missing (NaN) values
    indicating_mask = np.isnan(X_missing)

    if len(imputation_mean.shape) == 2:
        # Add the third dimension, as the downstream code expects 3D arrays
        imputation_mean = np.expand_dims(imputation_mean, axis=2)
        logger.debug("Adding the third dimension to the imputation_mean array")

    assert len(imputation_mean.shape) == 3

    # Destandardize the data (if needed)
    if cfg["PREPROCESS"]["standardize"]:
        logger.debug("Destandardizing the imputed data")
        imputation_mean = destandardize_numpy(
            imputation_mean,
            mean=preprocess["standardization"]["mean"],
            std=preprocess["standardization"]["stdev"],
        )

    imputation_dict = {
        "imputation_dict": {
            "imputation": {
                # (no_samples, no_timepoints, no_features)
                "mean": imputation_mean,
                "imputation_ci_neg": None,
                "imputation_ci_pos": None,
            },
            "indicating_mask": indicating_mask,
        },
        "timing": end_time,
    }

    return imputation_dict


def create_imputation_dict_from_moment(
    imputation_mean, indicating_mask, imputation_time
):
    imputation_dict = {
        "imputation_dict": {
            "imputation": {
                # (no_samples, no_timepoints, no_features)
                "mean": imputation_mean,
                "imputation_ci_neg": None,
                "imputation_ci_pos": None,
            },
            "indicating_mask": indicating_mask,
        },
        "timing": imputation_time,
    }

    return imputation_dict


def imputation_per_split_of_dict(data_dicts, df, preprocess, model, split, cfg):
    X_missing = df.to_numpy()
    logger.debug("Split = {}".format(split))
    start_time = time.time()
    imputation_mean = model.transform(x=df)
    dict_out = create_imputation_dict(
        imputation_mean=imputation_mean,
        preprocess=preprocess,
        X_missing=X_missing,
        end_time=time.time() - start_time,
        cfg=cfg,
    )

    return dict_out
