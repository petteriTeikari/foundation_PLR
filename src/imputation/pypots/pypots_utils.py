from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from src.data_io.data_conv_utils import long_df_to_long_numpy
from src.data_io.data_utils import get_unique_polars_rows, set_missing_in_data
from src.preprocess.preprocess_data import debug_triplet_stats, preprocess_PLR_data
from src.utils import get_artifacts_dir


def pypots_split_preprocess_wrapper(
    df_split: pl.DataFrame,
    model_cfg: DictConfig,
    cfg: dict[str, Any],
    preprocess_dict: Optional[dict[str, Any]] = None,
    split: str = "train",
) -> dict[str, Any]:
    # The denoised "gold standard", or pseudo-GT
    X_gt = long_df_to_long_numpy(df=df_split, size_col_name="gt")
    X_gt, preprocess_dict = preprocess_PLR_data(
        X=X_gt,
        preprocess_cfg=cfg["PREPROCESS"],
        preprocess_dict=preprocess_dict,
        data_filtering="gt",
        split=split,
    )

    # Same for the raw data
    X_raw = long_df_to_long_numpy(df=df_split, size_col_name="pupil_raw")
    X_raw, preprocess_dict = preprocess_PLR_data(
        X=X_raw,
        preprocess_cfg=cfg["PREPROCESS"],
        preprocess_dict=preprocess_dict,
        data_filtering="raw",
        split=split,
    )

    X_raw_imputed = long_df_to_long_numpy(
        df=df_split, size_col_name="pupil_raw_imputed"
    )
    X_raw_imputed, preprocess_dict = preprocess_PLR_data(
        X=X_raw_imputed,
        preprocess_cfg=cfg["PREPROCESS"],
        preprocess_dict=preprocess_dict,
        data_filtering="raw",
        split=split,
    )
    assert (
        np.isnan(X_raw_imputed).sum() == 0
    ), "There are still missing values in the imputed raw data"

    # Same for the orig data
    X_orig = long_df_to_long_numpy(df=df_split, size_col_name="pupil_orig")
    X_orig, preprocess_dict = preprocess_PLR_data(
        X=X_orig,
        preprocess_cfg=cfg["PREPROCESS"],
        preprocess_dict=preprocess_dict,
        data_filtering="orig",
        split=split,
    )

    X_orig_imputed = long_df_to_long_numpy(
        df=df_split, size_col_name="pupil_orig_imputed"
    )
    X_orig_imputed, preprocess_dict = preprocess_PLR_data(
        X=X_orig_imputed,
        preprocess_cfg=cfg["PREPROCESS"],
        preprocess_dict=preprocess_dict,
        data_filtering="orig",
        split=split,
    )
    assert (
        np.isnan(X_orig_imputed).sum() == 0
    ), "There are still missing values in the imputed raw data"

    # Set the same missing values missing in the denoised gold standard as were in the raw data
    X_gt_missing = set_missing_in_data(
        df_split, X=deepcopy(X_gt), missingness_cfg=cfg["MISSINGNESS"], split=split
    )

    # Outlier detection mask is the same as the missingness mask
    # Using 1 for outliers and 0 for non-outliers
    outlier_mask = long_df_to_long_numpy(
        df=df_split, size_col_name="imputation_mask"
    ).astype(int)
    assert outlier_mask.sum() > 0, "No outliers labeled in the mask"
    logger.info(
        "Number of samples marked as outlier in the outlier mask: {}".format(
            outlier_mask.sum()
        )
    )

    debug_triplet_stats(X_gt, X_gt_missing, X_raw, split)
    metadata_df = add_metadata_dicts(df_split, X_gt=X_gt, split=split, cfg=cfg)

    return {
        "data": {
            "ground_truth": {"gt": X_gt},
            "data_missing": {
                "gt": X_gt_missing,
                "raw": X_raw,
                "raw_imputed": X_raw_imputed,
            },
            "orig_with_outliers": {
                "mask": outlier_mask,
                "orig": X_orig,
                "orig_imputed": X_orig_imputed,
            },
        },
        "metadata": {
            "metadata_df": metadata_df,
            "preprocess": preprocess_dict,
        },
    }


def create_dataset_dicts_for_pypots(
    source_data: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    def dataset_per_split(
        split_data: dict[str, Any], split: str
    ) -> dict[str, np.ndarray]:
        # PyPOTS does not want mask per se, so X_ori is the ground truth (e.g. X for moment)
        # and we set the missing values to NaN and have that to be X
        X = split_data["X"].copy()
        assert np.isnan(X).sum() == 0, "Ground truth data has NaNs"
        X[split_data["mask"] == 1] = np.nan
        # Also PyPots wants a feature dimension as the third dimension, and as we only have one feature
        # (the pupil size), we have a singleton dimension there
        assert len(X.shape) == 2, "X has more than 2 dimensions"
        return {"X": X[:, :, np.newaxis], "X_ori": split_data["X"][..., np.newaxis]}

    logger.debug("Creating the dataset dictionaries for PyPOTS")
    dataset_dicts = {}
    for split in source_data["df"]:
        dataset_dicts[split] = dataset_per_split(
            split_data=source_data["df"][split]["data"], split=split
        )
        assert len(dataset_dicts[split]["X"].shape) == 3, "X has != 3 dimensions"

    return dataset_dicts


def define_pypots_outputs(
    model_name: str, artifact_type: str = "model"
) -> Tuple[str, str, str]:
    fname = f"{artifact_type}_{model_name}.pickle"
    # Define artifact location (use the mlflow directory for the PyPOTS output)
    output_dir = Path(get_artifacts_dir(service_name="pypots"))
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_path = output_dir / fname
    logger.info(
        "Saving PyPOTS ({}) {} artifacts to {}".format(
            model_name, artifact_type, output_dir
        )
    )

    return str(output_dir), fname, str(artifacts_path)


def add_metadata_dicts(
    df_split: pl.DataFrame,
    X_gt: np.ndarray,
    split: str,
    cfg: DictConfig,
) -> pl.DataFrame:
    unique_codes = get_unique_polars_rows(
        df_split,
        unique_col="subject_code",
        value_col="class_label",
        split=split,
        df_string="PLR",
    )

    assert (
        len(unique_codes) == X_gt.shape[0]
    ), "Unique codes ({}) in the metadata df and the number of rows in the data ({}) do not match".format(
        len(unique_codes), X_gt.shape[0]
    )

    return unique_codes
