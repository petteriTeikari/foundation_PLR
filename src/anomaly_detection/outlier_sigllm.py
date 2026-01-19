import os

import numpy as np
import polars as pl
from omegaconf import DictConfig

from src.anomaly_detection.anomaly_utils import get_data_for_sklearn_anomaly_models
from src.utils import get_data_dir


def write_numpy_to_disk(X, y, X_test, y_test, model_cfg):
    data_dir = get_data_dir()
    train_on = model_cfg["MODEL"]["train_on"]
    np.save(os.path.join(data_dir, f"{train_on}__X.npy"), X)
    np.save(os.path.join(data_dir, f"{train_on}__X_test.npy"), X_test)
    np.save(os.path.join(data_dir, f"{train_on}__y.npy"), y)
    np.save(os.path.join(data_dir, f"{train_on}__y_test.npy"), y_test)


def outlier_sigllm_wrapper(
    df: pl.DataFrame,
    cfg: DictConfig,
    model_cfg: DictConfig,
    experiment_name: str,
    run_name: str,
):
    train_on = model_cfg["MODEL"]["train_on"]
    X, y, X_test, y_test, _ = get_data_for_sklearn_anomaly_models(
        df=df, cfg=cfg, train_on=train_on
    )

    # The "SigLLM" at its 0.0.2 and 0.0.3dev0 have unsatisfiable dependencies (sklearn)
    # so let's save the data needed here and run the sigllm detection in a separate script
    write_numpy_to_disk(X, y, X_test, y_test, model_cfg)

    raise NotImplementedError(
        "SigLLM is not yet implemented in this version of the codebase"
    )
    # See "sigllm_anomaly_detection.py"
    #
    # pred_masks = {}
    # split = 'train'
    # pred_masks[split] = sigllm_splitwise_detection(X=X, y=y, split=split, model_cfg=model_cfg, cfg=cfg)
