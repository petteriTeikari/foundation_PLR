import numpy as np
import torch

from src.classification.classifier_utils import get_dict_array_splits
from src.classification.xgboost_cls.xgboost_utils import data_transform_wrapper


def get_tabm_preds_from_results_for_bootstrap(split_results):
    """
    Should match now bootstrap_predict()
    predict_probs = model.predict_proba(X) # (n_samples, n_classes), e.g. (72,2)
        preds = {
            "y_pred_proba": predict_probs[:, 1], # (n_samples,), e.g. (72,) for the class 1 (e.g. glaucoma)
            "y_pred": model.predict(X), # (n_samples,), e.g. (72,)
        }
    """
    # TabM
    preds = {
        "y_pred_proba": split_results["pred"][:, 1],
        "y_pred": split_results["pred"].argmax(axis=1),
    }

    return preds


def transform_data_to_tabm_from_dict_arrays(dict_arrays, device):
    # Using this from the tutorial, remember that this can be now called from bootstrap,
    # so we have the "val" split here as well in addition to the original "train" and "test"
    splits = get_dict_array_splits(dict_arrays)
    data_numpy = {}
    for split in splits:
        data_numpy[split] = {
            "x_cont": dict_arrays[f"x_{split}"].astype(np.float32),
            "y": dict_arrays[f"y_{split}"].astype(np.int64),
        }
        assert (
            data_numpy[split]["x_cont"].shape[0] == data_numpy[split]["y"].shape[0]
        ), f"X and y shapes do not match for split: {split}"

    # Convert to tensors
    data = {
        part: {
            k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()
        }
        for part in data_numpy
    }

    return data


def transform_data_to_tabm_from_df(
    train_df, test_df, cls_model_cfg, hparam_cfg, device
):
    # Convert Polars DataFrames to arrays
    _, _, dict_arrays = data_transform_wrapper(
        train_df, test_df, cls_model_cfg, hparam_cfg
    )

    data = transform_data_to_tabm_from_dict_arrays(dict_arrays, device)

    return data
