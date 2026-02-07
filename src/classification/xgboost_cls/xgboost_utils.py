import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, OrderedDict, Tuple

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder

from src.classification.classification_checks import pre_tree_based_classifier_checks


def encode_labels_to_integers(y_string: np.ndarray) -> np.ndarray:
    """
    Encode string class labels to integer values.

    Parameters
    ----------
    y_string : np.ndarray
        Array of string class labels (e.g., 'control', 'glaucoma').

    Returns
    -------
    np.ndarray
        Integer-encoded labels where control=0, glaucoma=1.
    """
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(y_string)
    # control - 0, glaucoma - 1
    y: np.ndarray = label_encoder.transform(y_string)
    logger.info(f"Class labels: {label_encoder.classes_}: {np.unique(y).tolist()}")
    return y


def get_x_y(
    df: pd.DataFrame,
    split: str,
    return_value: str,
    xgboost_cfg: DictConfig,
    hparam_cfg: DictConfig,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract feature matrix X and target vector y from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing features and metadata columns.
    split : str
        Data split identifier ('train' or 'val').
    return_value : str
        Type of values to return: 'mean' for feature values or 'weights' for
        inverse variance weights.
    xgboost_cfg : DictConfig
        XGBoost configuration dictionary.
    hparam_cfg : DictConfig
        Hyperparameter configuration dictionary.

    Returns
    -------
    x : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Integer-encoded target labels.
    feature_names : list
        List of feature column names.
    """
    try:
        df = df.drop("subject_code", axis=1)
    except Exception as e:
        logger.error(f"Could not drop subject_code: {e}")
        raise e
    df_out = pd.DataFrame()
    for col in df.columns:
        if "metadata" in col:
            if "class_label" in col:
                logger.debug("Picking column: {} as the target".format(col))
                y_string: np.ndarray = df[col].values
                y = encode_labels_to_integers(y_string)
            else:
                logger.debug("Dropping metadata column: {}".format(col))
        else:
            if return_value == "mean":
                if "_value" in col:
                    # TODO! DataFrame is highly fragmented.  This is usually the result of calling `frame.insert`
                    #  many times, which has poor performance.  Consider joining all columns at once
                    #  using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
                    warnings.simplefilter("ignore")
                    df_out[col] = df[col]
                    warnings.resetwarnings()
            elif return_value == "weights":
                if "_std" in col:
                    # TODO! DataFrame is highly fragmented. warning
                    warnings.simplefilter("ignore")
                    df_out[col] = df[col]
                    warnings.resetwarnings()
                    if (
                        xgboost_cfg["MODEL"]["WEIGHING"]["weights_creation_method"]
                        == "inverse_of_variance"
                    ):
                        df_out[col] = df[col] ** 2  # variance = std^2
                        df_out[col] = 1 / df_out[col]  # weights = 1 / variance
                        # you could rename the _std to _weight here if this is confusing
                        # but keeping now the original names so you see where the weight values come from
                    else:
                        logger.error("Unknown weighting scheme")
                        raise ValueError("Unknown weighting scheme")
            else:
                logger.error(f"Unknown return value: {return_value}")
                raise ValueError(f"Unknown return value: {return_value}")

    feature_names = list(df_out.columns)
    x: np.ndarray = df_out.values  # (no_subjects, no_features) (e.g. (4, 25) for debug)
    return x, y, feature_names


def transform_data_for_xgboost(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    xgboost_cfg: DictConfig,
    hparam_cfg: DictConfig,
    return_value: str = "mean",
    check_ratio: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Transform Polars DataFrames to numpy arrays for XGBoost training.

    Parameters
    ----------
    train_df : pl.DataFrame
        Training data as a Polars DataFrame.
    val_df : pl.DataFrame
        Validation data as a Polars DataFrame.
    xgboost_cfg : DictConfig
        XGBoost configuration dictionary.
    hparam_cfg : DictConfig
        Hyperparameter configuration dictionary.
    return_value : str, optional
        Type of values to extract: 'mean' or 'weights'. Default is 'mean'.
    check_ratio : bool, optional
        Whether to check train/test mean ratio for sanity. Default is False.

    Returns
    -------
    x_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    x_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Test labels.
    feature_names_train : list
        List of feature names.
    """
    x_train, y_train, feature_names_train = get_x_y(
        df=deepcopy(train_df.to_pandas()),
        split="train",
        return_value=return_value,
        xgboost_cfg=xgboost_cfg,
        hparam_cfg=hparam_cfg,
    )
    x_test, y_test, feature_names_test = get_x_y(
        df=deepcopy(val_df.to_pandas()),
        split="val",
        return_value=return_value,
        xgboost_cfg=xgboost_cfg,
        hparam_cfg=hparam_cfg,
    )

    assert feature_names_train == feature_names_test, (
        "Feature names do not match between train and test"
    )
    assert y_test.shape[0] == x_test.shape[0], (
        "Number of labels does not match the number of samples"
    )
    assert y_train.shape[0] == x_train.shape[0], (
        "Number of labels does not match the number of samples"
    )
    assert x_train.shape[1] == len(feature_names_train), (
        "Number of features does not match the number of feature names"
    )
    assert x_test.shape[1] == len(feature_names_test), (
        "Number of features does not match the number of feature names"
    )

    # quick test if you have for example the other split standardized
    # and the other not
    train_mean_1st_feature = np.mean(x_train[:, 0])
    test_mean_1st_feature = np.mean(x_test[:, 0])
    ratio = train_mean_1st_feature / test_mean_1st_feature

    if check_ratio:
        if not np.isnan(ratio):
            assert train_mean_1st_feature != test_mean_1st_feature, (
                "Test and Train seems to be the same data?"
            )
            assert ratio < 10 and ratio > 0.1, (
                "Train {} and test {} seem to have wildly different means".format(
                    train_mean_1st_feature, test_mean_1st_feature
                )
            )

    return x_train, y_train, x_test, y_test, feature_names_train


def convert_numpy_to_cupy(array: np.ndarray) -> Optional[Any]:
    """
    Convert a NumPy array to a CuPy array for GPU acceleration.

    Parameters
    ----------
    array : np.ndarray
        Input NumPy array to convert.

    Returns
    -------
    cupy.ndarray or None
        CuPy array if conversion succeeds, None if CuPy import fails.
    """
    try:
        # "uv add cupy" might give you "Exception: Your CUDA environment is invalid. Please check above error log."
        # and either way, you need to have CUDA installed at system level
        import cupy as cp
    except Exception as e:
        logger.error(f"Could not import CuPy: {e}")
        logger.error("Using Numpy instead of CuPy")
        return None
    return cp.asarray(array)


def polars_to_numpy_arrays(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    xgboost_cfg: DictConfig,
    hparam_cfg: DictConfig,
    run_name: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    np.ndarray,
    np.ndarray,
    List[str],
]:
    """
    Convert Polars DataFrames to numpy arrays with feature values and weights.

    Parameters
    ----------
    train_df : pl.DataFrame
        Training data as a Polars DataFrame.
    val_df : pl.DataFrame
        Validation data as a Polars DataFrame.
    xgboost_cfg : DictConfig
        XGBoost configuration dictionary.
    hparam_cfg : DictConfig
        Hyperparameter configuration dictionary.
    run_name : str
        Name of the current run, used to determine weighting scheme.

    Returns
    -------
    tuple
        Tuple containing (x_train, y_train, x_test, y_test, feature_names,
        x_train_w, x_test_w, feature_names_w) where _w suffix denotes weight arrays.
    """
    # Convert Polars DataFrames to numpy arrays
    logger.info("Transforming data for mean feature values")
    x_train, y_train, x_test, y_test, feature_names = transform_data_for_xgboost(
        train_df,
        val_df,
        return_value="mean",
        xgboost_cfg=xgboost_cfg,
        hparam_cfg=hparam_cfg,
    )
    logger.info("Transforming stdev of features to 1/variance = feature weights")
    if "embedding" in run_name:
        logger.debug("unity weighing for embeddings")
        ones_train = np.ones_like(x_train)
        ones_test = np.ones_like(x_test)
        x_train_w, x_test_w, feature_names_w = ones_train, ones_test, feature_names
    else:
        x_train_w, _, x_test_w, _, feature_names_w = transform_data_for_xgboost(
            train_df,
            val_df,
            return_value="weights",
            xgboost_cfg=xgboost_cfg,
            hparam_cfg=hparam_cfg,
        )

    assert len(feature_names) == len(feature_names_w), (
        "Number of feature names do not match between mean and variance data"
    )
    assert x_train.shape[0] == x_train_w.shape[0], (
        "Number of samples do not match between mean and variance data"
    )
    assert x_test.shape[0] == x_test_w.shape[0], (
        "Number of samples do not match between mean and variance data"
    )
    assert x_train.shape[1] == x_train_w.shape[1], (
        "Number of features do not match between mean and variance data"
    )
    assert x_test.shape[1] == x_test_w.shape[1], (
        "Number of features do not match between mean and variance data"
    )

    if "DATA" in xgboost_cfg:
        if xgboost_cfg["DATA"]["use_cupy_arrays"]:
            # https://xgboost.readthedocs.io/en/stable/python/gpu-examples/cover_type.html
            logger.info("Converting numpy arrays to CuPy arrays")
            raise NotImplementedError("CuPy arrays are not yet supported")
            # x_train = convert_numpy_to_cupy(x_train)
            # y_train = convert_numpy_to_cupy(y_train)
            # x_test = convert_numpy_to_cupy(x_test)
            # y_test = convert_numpy_to_cupy(y_test)
            # x_train_w = convert_numpy_to_cupy(x_train_w)
            # x_test_w = convert_numpy_to_cupy(x_test_w)
        else:
            logger.info("Numpy arrays are used (instead of CuPy arrays)")

    return (
        x_train,
        y_train,
        x_test,
        y_test,
        feature_names,
        x_train_w,
        x_test_w,
        feature_names_w,
    )


def create_dmatrices_and_dict_arrays(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    x_train_w: np.ndarray,
    x_test_w: np.ndarray,
    feature_names_w: List[str],
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    xgboost_cfg: DictConfig,
    check_ratio: bool = False,
) -> Tuple[None, None, Dict[str, Any]]:
    """
    Create XGBoost DMatrix objects and a dictionary of arrays for training.

    Parameters
    ----------
    x_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    x_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Test labels.
    feature_names : list
        List of feature names for mean values.
    x_train_w : np.ndarray
        Training feature weights.
    x_test_w : np.ndarray
        Test feature weights.
    feature_names_w : list
        List of feature names for weight values.
    train_df : pl.DataFrame
        Original training DataFrame (for subject codes).
    val_df : pl.DataFrame
        Original validation DataFrame (for subject codes).
    xgboost_cfg : DictConfig
        XGBoost configuration dictionary.
    check_ratio : bool, optional
        Whether to check train/test mean ratio. Default is False.

    Returns
    -------
    dtest : xgb.DMatrix or None
        XGBoost DMatrix for test data (currently None).
    dtrain : xgb.DMatrix or None
        XGBoost DMatrix for training data (currently None).
    dict_arrays : dict
        Dictionary containing all arrays and metadata for training.
    """

    def clean_feature_names(feature_names):
        # remove '_value' from a list of feature names
        feature_names = [name.replace("_value", "") for name in feature_names]
        # and remove '_std' from a list of feature names
        feature_names = [name.replace("_std", "") for name in feature_names]
        return feature_names

    dict_arrays = {
        "x_train": x_train,
        "x_train_w": x_train_w,
        "y_train": y_train,
        "x_test": x_test,
        "x_test_w": x_test_w,
        "y_test": y_test,
        "feature_names": feature_names,
        "subject_codes_train": train_df["subject_code"].to_numpy(),
        "subject_codes_test": val_df["subject_code"].to_numpy(),
    }

    train_mean_1st_feature = np.mean(dict_arrays["x_train"][:, 0])
    test_mean_1st_feature = np.mean(dict_arrays["x_test"][:, 0])
    ratio = train_mean_1st_feature / test_mean_1st_feature
    assert train_mean_1st_feature != test_mean_1st_feature, (
        "Test and Train seems to be the same data?"
    )
    assert ratio < 10 and ratio > 0.1, (
        "Train {} and test {} seem to have wildly different means".format(
            train_mean_1st_feature, test_mean_1st_feature
        )
    )

    train_w_mean_1st_feature = np.mean(dict_arrays["x_train_w"][:, 0])
    test_w_mean_1st_feature = np.mean(dict_arrays["x_test_w"][:, 0])
    if np.isinf(train_w_mean_1st_feature) and np.isinf(test_w_mean_1st_feature):
        ratio = np.nan
    else:
        ratio = train_w_mean_1st_feature / test_w_mean_1st_feature

    if check_ratio:
        if not np.isnan(ratio):
            if np.all(dict_arrays["x_train_w"] == 1):
                logger.debug(
                    "You have not computed weights as they are all 1 (e.g. with embeddings with no stdev)"
                )
            else:
                assert train_w_mean_1st_feature != test_w_mean_1st_feature, (
                    "Test and Train weights seems to be the same data?"
                )
                if ~np.isnan(ratio):
                    assert ratio < 10 and ratio > 0.1, (
                        "Train {} and test {} weights seem to have wildly different means".format(
                            train_w_mean_1st_feature, test_w_mean_1st_feature
                        )
                    )

    dtrain = None  # xgb.DMatrix(X_train, label=y_train, feature_names=feat_names)
    dtest = None  # xgb.DMatrix(X_test, label=y_test, feature_names=feat_names)

    return dtest, dtrain, dict_arrays


def join_test_and_train_arrays(
    dict_arrays: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenate train and test arrays for cross-validation.

    Parameters
    ----------
    dict_arrays : dict
        Dictionary containing x_train, x_test, y_train, y_test, x_train_w, x_test_w.

    Returns
    -------
    X : np.ndarray
        Combined feature matrix from train and test sets.
    y : np.ndarray
        Combined labels from train and test sets.
    X_w : np.ndarray
        Combined feature weights from train and test sets.
    """
    # e.g. when you are doing CV and you want to use the whole dataset
    X = np.concatenate((dict_arrays["x_train"], dict_arrays["x_test"]), axis=0)
    y = np.concatenate((dict_arrays["y_train"], dict_arrays["y_test"]), axis=0)
    X_w = np.concatenate((dict_arrays["x_train_w"], dict_arrays["x_test_w"]), axis=0)
    return X, y, X_w


def data_transform_wrapper(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    xgboost_cfg: DictConfig,
    hparam_cfg: DictConfig,
    run_name: str = "None",
) -> Tuple[None, None, Dict[str, Any]]:
    """
    Transform Polars DataFrames to XGBoost-ready format with all preprocessing.

    Parameters
    ----------
    train_df : pl.DataFrame
        Training data as a Polars DataFrame.
    test_df : pl.DataFrame
        Test data as a Polars DataFrame.
    xgboost_cfg : DictConfig
        XGBoost configuration dictionary.
    hparam_cfg : DictConfig
        Hyperparameter configuration dictionary.
    run_name : str, optional
        Name of the current run. Default is "None".

    Returns
    -------
    dtest : xgb.DMatrix or None
        XGBoost DMatrix for test data.
    dtrain : xgb.DMatrix or None
        XGBoost DMatrix for training data.
    dict_arrays : dict
        Dictionary containing all arrays and metadata for training.
    """
    logger.info("Transforming data for Classification Model")

    # Convert Polars DataFrames to numpy arrays
    (
        x_train,
        y_train,
        x_test,
        y_test,
        feature_names,
        x_train_w,
        x_test_w,
        feature_names_w,
    ) = polars_to_numpy_arrays(train_df, test_df, xgboost_cfg, hparam_cfg, run_name)

    # Pre-train checkups and filtering
    (
        x_train,
        y_train,
        x_test,
        y_test,
        feature_names,
        x_train_w,
        x_test_w,
        feature_names_w,
    ) = pre_tree_based_classifier_checks(
        xgboost_cfg,
        x_train,
        y_train,
        x_test,
        y_test,
        feature_names,
        x_train_w,
        x_test_w,
        feature_names_w,
    )

    # Create DMatrix from Numpy/CuPy arrays
    dtrain, dtest, dict_arrays = create_dmatrices_and_dict_arrays(
        x_train,
        y_train,
        x_test,
        y_test,
        feature_names,
        x_train_w,
        x_test_w,
        feature_names_w,
        train_df,
        test_df,
        xgboost_cfg,
    )

    return dtest, dtrain, dict_arrays


def get_last_items_from_OrderedDicts(
    results: Dict[str, OrderedDict[str, List[float]]],
) -> Dict[str, float]:
    """
    Extract the last metric values from XGBoost evaluation results.

    Parameters
    ----------
    results : dict
        XGBoost evals_result dictionary containing validation_0 and validation_1.

    Returns
    -------
    dict
        Dictionary with 'train' and 'test' keys containing the last metric value
        for each split.
    """

    # Inside XGBoost model, the "best metric" is a bit hidden so getting it out like this:
    def get_last_value_per_split(split: OrderedDict):
        last_key = next(reversed(split))
        last_value = split[last_key][-1]
        return last_value

    items_out = {
        "train": get_last_value_per_split(split=results["validation_0"]),
        "test": get_last_value_per_split(split=results["validation_1"]),
    }

    return items_out


def find_best_metric(
    best_metrics: Dict[str, Dict[str, float]],
    xgboost_cfg: DictConfig,
) -> int:
    """
    Find the index of the best performing configuration from grid search.

    Parameters
    ----------
    best_metrics : dict
        Dictionary mapping grid config names to their train/test metric values.
    xgboost_cfg : DictConfig
        XGBoost configuration dictionary.

    Returns
    -------
    int
        Index of the configuration with the best test metric value.
    """
    best_values = []
    for grid_name in best_metrics.keys():
        best_values.append(best_metrics[grid_name]["test"])

    best_metric_idx = np.argmax(best_values)
    best_metric_value = best_values[best_metric_idx]
    logger.info(
        f"Best (test) metric value: {best_metric_value:.3f} "
        f"(grid_name = {list(best_metrics.keys())[best_metric_idx]})"
    )

    return best_metric_idx
