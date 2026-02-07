from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import DictConfig

from src.anomaly_detection.anomaly_utils import get_data_for_sklearn_anomaly_models
from src.utils import get_data_dir


def write_numpy_to_disk(X, y, X_test, y_test, model_cfg):
    """
    Write training and test data arrays to disk as NumPy files.

    Saves the input arrays to the data directory with filenames prefixed by the
    training data source specified in model configuration.

    Parameters
    ----------
    X : np.ndarray
        Training feature array.
    y : np.ndarray
        Training target/label array.
    X_test : np.ndarray
        Test feature array.
    y_test : np.ndarray
        Test target/label array.
    model_cfg : DictConfig or dict
        Model configuration containing MODEL.train_on key that specifies
        the data source name used as filename prefix.

    Returns
    -------
    None
        Files are written to disk with naming pattern:
        {train_on}__X.npy, {train_on}__X_test.npy,
        {train_on}__y.npy, {train_on}__y_test.npy
    """
    data_dir = Path(get_data_dir())
    train_on = model_cfg["MODEL"]["train_on"]
    np.save(str(data_dir / f"{train_on}__X.npy"), X)
    np.save(str(data_dir / f"{train_on}__X_test.npy"), X_test)
    np.save(str(data_dir / f"{train_on}__y.npy"), y)
    np.save(str(data_dir / f"{train_on}__y_test.npy"), y_test)


def outlier_sigllm_wrapper(
    df: pl.DataFrame,
    cfg: DictConfig,
    model_cfg: DictConfig,
    experiment_name: str,
    run_name: str,
):
    """
    Wrapper for SigLLM-based outlier detection.

    Prepares data for SigLLM anomaly detection by extracting features and
    writing them to disk. Currently raises NotImplementedError as SigLLM
    has unsatisfiable dependencies in versions 0.0.2 and 0.0.3dev0.

    Parameters
    ----------
    df : pl.DataFrame
        Input Polars DataFrame containing the raw data for anomaly detection.
    cfg : DictConfig
        Main configuration object with data processing settings.
    model_cfg : DictConfig
        Model-specific configuration containing MODEL.train_on key.
    experiment_name : str
        Name of the MLflow experiment for logging (currently unused).
    run_name : str
        Name of the MLflow run for logging (currently unused).

    Returns
    -------
    None
        Currently raises NotImplementedError before returning.

    Raises
    ------
    NotImplementedError
        Always raised as SigLLM integration is not yet complete.

    Notes
    -----
    The data is saved to disk via `write_numpy_to_disk` for potential
    offline processing with a separate SigLLM script. See
    `sigllm_anomaly_detection.py` for the intended detection logic.
    """
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
