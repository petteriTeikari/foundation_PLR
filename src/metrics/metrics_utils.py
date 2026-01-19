import numpy as np
from omegaconf import DictConfig
from loguru import logger


def get_array_triplet_for_pypots_metrics_from_imputer(
    split_imputation, split_data, split, cfg: DictConfig
):
    # Input data used to train the imputer with missing values
    X = split_data["data"]["X"].copy()[:, :, np.newaxis]
    indicating_mask = split_data["data"]["mask"][:, :, np.newaxis]
    # mask_ratio = np.sum(indicating_mask) / (indicating_mask.shape[0]* indicating_mask.shape[1])
    X[split_data["data"]["mask"] == 1] = np.nan

    # Ground truth, i.e. the manually supervised denoised smooth PLR data
    X_gt = split_data["data"]["X_GT"][:, :, np.newaxis]

    # Imputed data from the trained imputer model
    X_imputed = split_imputation["imputation_dict"]["imputation"]["mean"]

    # Check if the input arrays all have the same size (or some other check
    check_array_triplet(predictions=X_imputed, targets=X_gt, masks=indicating_mask)

    return X, X_gt, X_imputed, indicating_mask


def check_array_triplet(predictions, targets, masks):
    assert predictions.shape[0] == targets.shape[0] == masks.shape[0], (
        f"Predictions, targets, and masks should have the same number of subjects, "
        f"but got {predictions.shape[0]}, {targets.shape[0]}, and {masks.shape[0]}"
    )
    assert predictions.shape[1:] == targets.shape[1:] == masks.shape[1:], (
        f"Predictions, targets, and masks should have the same number of time points, "
        f"but got {predictions.shape[1:]}, {targets.shape[1:]}, and {masks.shape[1:]}"
    )
    if len(predictions.shape) == 2:
        logger.error("Predictions are now 2D and not 3D as expected")
        raise ValueError("Predictions are now 2D and not 3D as expected")


def get_subjectwise_arrays(predictions, targets, masks, i):
    def pick_subject_and_expand(X_in, i):
        return np.expand_dims(X_in[i], axis=0)

    return (
        pick_subject_and_expand(predictions, i),
        pick_subject_and_expand(targets, i),
        pick_subject_and_expand(masks, i),
    )


def init_metrics_dict():
    metrics = {}
    metrics["scalars"] = {}
    metrics["arrays"] = {}
    metrics["arrays_flat"] = []

    return metrics
