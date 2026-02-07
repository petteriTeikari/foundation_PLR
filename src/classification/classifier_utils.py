from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import f1_score, log_loss, roc_auc_score


def cls_train_on_this_combo(run_name: str, cls_model_name: str) -> bool:
    """
    Determine if classifier should be trained on this preprocessing combination.

    Applies filtering logic to skip certain combinations (e.g., experimental
    models on non-ground-truth data, TabPFN on high-dimensional embeddings).

    Parameters
    ----------
    run_name : str
        MLflow run name encoding preprocessing pipeline.
    cls_model_name : str
        Classifier model name.

    Returns
    -------
    bool
        True if this combination should be trained, False to skip.
    """
    is_on_gt = cls_is_on_ground_truth(run_name=run_name)
    is_experimental = is_experimental_cls_model(cls_model_name=cls_model_name)
    is_on_embeddings = is_trained_on_embeddings(run_name=run_name)

    if is_on_gt:
        # train all the experimental and embeddings models always with the ground truth
        if is_on_embeddings and cls_model_name == "TabPFN":
            # maximum number of features is 100, will crash with the 1,024 dim embedding
            return False
        else:
            return True
    elif is_experimental:
        return False
    elif is_on_embeddings:
        # with 1024 features, this will take quite long
        return False
    else:
        return True


def cls_is_on_ground_truth(run_name: str) -> bool:
    """
    Check if run uses ground truth preprocessing.

    Parameters
    ----------
    run_name : str
        MLflow run name.

    Returns
    -------
    bool
        True if using ground truth outlier detection and imputation.
    """
    if "pupil-gt__pupil-gt" in run_name:
        return True
    else:
        return False


def get_cls_baseline_models() -> list[str]:
    """
    Get list of baseline classifier model names.

    Returns
    -------
    list of str
        Names of standard baseline classifiers used in the study.
    """
    return ["LogisticRegression", "XGBOOST", "CATBOOST", "TabPFN", "TabM"]


def is_experimental_cls_model(cls_model_name: str) -> bool:
    """
    Check if classifier is experimental (not a baseline model).

    Parameters
    ----------
    cls_model_name : str
        Classifier model name.

    Returns
    -------
    bool
        True if model is not in baseline models list.
    """
    for baseline_model in get_cls_baseline_models():
        if baseline_model in cls_model_name:
            return False
    return True


def is_trained_on_embeddings(run_name: str) -> bool:
    """
    Check if run uses embedding features instead of handcrafted.

    Parameters
    ----------
    run_name : str
        MLflow run name.

    Returns
    -------
    bool
        True if using foundation model embeddings as features.
    """
    if "embedding" in run_name:
        return True
    else:
        return False


def get_dict_array_splits(dict_arrays: dict[str, Any]) -> list[str]:
    """
    Extract split names from dict_arrays keys.

    Parameters
    ----------
    dict_arrays : dict
        Dictionary with keys like 'y_train', 'y_test', 'y_val'.

    Returns
    -------
    list
        Unique split names (e.g., ['train', 'test', 'val']).
    """
    keys = list(dict_arrays.keys())
    splits = []
    for key in keys:
        if key.startswith("y_"):
            key_fields = key.split("_")
            splits.append(key_fields[1])
    return list(set(splits))


def get_classifier_run_name(imputer_name: str) -> str:
    """
    Generate classifier run name from imputer name.

    Parameters
    ----------
    imputer_name : str
        Name of the imputation method.

    Returns
    -------
    str
        Run name for the classifier.
    """
    return f"{imputer_name}"


def get_cls_run_name(
    imputer_mlflow_run: dict[str, Any] | None,
    cls_model_name: str,
    cls_model_cfg: DictConfig,
    source: str,
) -> str:
    """
    Generate full classifier run name encoding the preprocessing pipeline.

    Parameters
    ----------
    imputer_mlflow_run : dict or None
        MLflow run info for the imputation model.
    cls_model_name : str
        Classifier model name.
    cls_model_cfg : DictConfig
        Classifier model configuration.
    source : str
        Data source identifier (e.g., 'GT', 'Raw').

    Returns
    -------
    str
        Full run name like 'CatBoost__pupil-gt__pupil-gt'.
    """
    if imputer_mlflow_run is not None:
        base_name = imputer_mlflow_run["tags.mlflow.runName"]
    else:
        # No MLflow run available for the non-imputed data sources (e.g. GT, Raw)
        base_name = source
    # TODO! Update the run name to include the classifier model configuration
    run_name = f"{cls_model_name}__{base_name}"
    return run_name


def preprocess_features(
    train_df: pl.DataFrame, val_df: pl.DataFrame, _cls_preprocess_cfg: DictConfig
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Trees do not really need standardization, but can benefit from something? See below

    See e.g. Hubert Ruczyński and Anna Kozak (2024) Do Tree-based Models Need Data Preprocessing?
    https://openreview.net/forum?id=08Y5sFtRhN
        Furthermore, we introduce the preprocessibility measure, based on tunability from (Probst et al.,
        2018). It describes how much performance can we gain or lose for a dataset 𝐷
        by using various preprocessing strategies.
    """
    logger.info("Placeholder for Preprocessing features")
    return train_df, val_df


def logger_remaining_samples(
    features: dict[str, Any], samples_in: dict[str, int], source: str
) -> None:
    """
    Log the number of remaining samples after filtering.

    Parameters
    ----------
    features : dict
        Features dictionary with data per source.
    samples_in : dict
        Original sample counts per split before filtering.
    source : str
        Data source name.
    """
    data: dict[str, pl.DataFrame] = features[source]["data"]
    for split in data.keys():
        df = data[split]
        logger.info(
            f"{split} | remaining samples = {df.shape[0]} / {samples_in[split]}"
        )


def drop_unlabeled_subjects(
    df: pl.DataFrame | pd.DataFrame,
    cfg: DictConfig,
    label_col_name: str = "metadata_class_label",
) -> pl.DataFrame:
    """
    Remove rows without classification labels from dataframe.

    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        Input dataframe with subject data.
    cfg : DictConfig
        Hydra configuration.
    label_col_name : str, default "metadata_class_label"
        Column name containing class labels.

    Returns
    -------
    pl.DataFrame
        Filtered dataframe with only labeled subjects.
    """
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    # Drop the rows from Polars dataframe with label_col_name being null or empty
    df = df.filter(pl.any_horizontal(pl.col(label_col_name).is_not_null()))
    # if you happen to have "None" string
    df = df.filter(pl.any_horizontal(pl.col(label_col_name) != "None"))
    return df


def drop_useless_cols(features: dict[str, Any], cfg: DictConfig) -> dict[str, Any]:
    """
    Remove metadata columns not needed for classification.

    Parameters
    ----------
    features : dict
        Features dictionary with data per source and split.
    cfg : DictConfig
        Hydra configuration.

    Returns
    -------
    dict
        Features dictionary with useless columns removed.
    """
    useless_cols = ["metadata_outlier_mask", "metadata_subject_code", "metadata_split"]
    logger.info("Dropping the 'useless columns': {}".format(useless_cols))
    for source in features.keys():
        for split in features[source]["data"].keys():
            for col in useless_cols:
                if col in features[source]["data"][split].columns:
                    features[source]["data"][split].drop_in_place(col)

    return features


def check_classification_labels(
    features: dict[str, Any], source: str, split: str, features_in: int
) -> None:
    """
    Validate that classification labels are present and binary.

    Parameters
    ----------
    features : dict
        Features dictionary.
    source : str
        Data source name.
    split : str
        Data split name ('train', 'test').
    features_in : int
        Expected number of features.

    Raises
    ------
    AssertionError
        If feature count changed or labels are not binary.
    """
    assert (
        features_in == features[source]["data"][split].shape[1]
    ), "Number of features changed"

    labels: pl.Series = features[source]["data"][split]["metadata_class_label"]
    unique_labels = set(labels)
    no_unique_labels = len(unique_labels)
    assert no_unique_labels == 2, (
        "We have != 2 unique labels (n={}), not good for a binary classification, something "
        "went wrong\n"
        "labels = {}\n"
        "n_samples = {}".format(no_unique_labels, unique_labels, len(labels))
    )


def keep_only_labeled_subjects(
    features: dict[str, Any], cfg: DictConfig, data_key: str = "data"
) -> dict[str, Any]:
    """
    Filter features to keep only subjects with classification labels.

    Parameters
    ----------
    features : dict
        Features dictionary with data per source and split.
    cfg : DictConfig
        Hydra configuration.
    data_key : str, default "data"
        Key in features dict containing the dataframes.

    Returns
    -------
    dict
        Filtered features dictionary.
    """
    logger.info("Dropping the subjects without a label")
    samples_in = {}
    for source in features.keys():
        for split in features[source][data_key].keys():
            df = features[source][data_key][split]
            samples_in[split], features_in = df.shape
            features[source][data_key][split] = drop_unlabeled_subjects(df, cfg)
            check_classification_labels(features, source, split, features_in)

    logger_remaining_samples(features, samples_in, source)
    return features


def get_numpy_boolean_index_for_class_labels(label_array: np.ndarray) -> np.ndarray:
    """
    Create boolean index for samples with valid classification labels.

    Parameters
    ----------
    label_array : np.ndarray
        2D array of labels (n_subjects, n_timepoints).

    Returns
    -------
    np.ndarray
        Boolean array where True indicates valid label.

    Raises
    ------
    AssertionError
        If input is not 2D numpy array or doesn't have exactly 2 classes.
    """
    assert isinstance(label_array, np.ndarray), "label_array must be a Numpy array"
    assert (
        len(label_array.shape) == 2
    ), "Must be a 2D array, (no_subjects, no_timepoints)"
    label_array = label_array[:, 0]  # (no_subjects)
    is_None = []
    for item in label_array:
        is_None.append(item is None)
    is_None = np.array(is_None, dtype=bool)
    is_str_None = label_array == "None"
    is_labeled = ~is_None & ~is_str_None
    labels = label_array[is_labeled]
    assert (
        len(np.unique(labels)) == 2
    ), "Label array must have 2 unique values (as we have a binary classification)"

    return is_labeled


def index_with_boolean_all_numpys_in_datadict(
    data_dict: dict[str, dict[str, np.ndarray]], labeled_boolean: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    """
    Apply boolean indexing to all numpy arrays in nested dictionary.

    Parameters
    ----------
    data_dict : dict
        Nested dictionary with numpy arrays as values.
    labeled_boolean : np.ndarray
        Boolean index array for filtering.

    Returns
    -------
    dict
        Copy of data_dict with all arrays filtered by boolean index.
    """
    dict_out = deepcopy(data_dict)
    for category in data_dict.keys():
        for variable in data_dict[category].keys():
            assert isinstance(
                data_dict[category][variable], np.ndarray
            ), "data_dict[category][variable] must be a numpy array"
            dict_out[category][variable] = data_dict[category][variable][
                labeled_boolean, :
            ]
    return dict_out


def keep_only_labeled_subjects_from_source(
    data_dicts: dict[str, dict[str, dict[str, np.ndarray]]], cfg: DictConfig
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """
    Filter data dictionaries to keep only labeled subjects.

    Parameters
    ----------
    data_dicts : dict
        Dictionary with splits as keys, each containing category/variable arrays.
    cfg : DictConfig
        Hydra configuration.

    Returns
    -------
    dict
        Filtered data dictionaries with only labeled subjects.
    """
    for split in data_dicts.keys():
        data_dict = data_dicts[split]
        labeled_boolean = get_numpy_boolean_index_for_class_labels(
            label_array=data_dict["labels"]["class_label"]
        )
        data_dict_cls = index_with_boolean_all_numpys_in_datadict(
            data_dict, labeled_boolean
        )
        data_dicts[split] = data_dict_cls

    return data_dicts


def pick_subset_of_features_for_classification(features: dict, cfg: DictConfig) -> dict:
    """
    Select specific feature subset for classification from full features.

    Parameters
    ----------
    features : dict
        Full features dictionary with all sources and splits.
    cfg : DictConfig
        Hydra configuration with DATA_SUBSET settings.

    Returns
    -------
    dict
        Features dictionary with only selected feature subset.

    Raises
    ------
    ValueError
        If unknown source type encountered.
    """
    features_out = {}
    for source, features_per_source in features.items():
        features_out[source] = {}
        features_out[source]["mlflow_run"] = features_per_source["mlflow_run"]
        features_out[source]["metadata"] = {
            "dummy": "placeholder"
        }  # features_per_source["metadata"]
        features_out[source]["data"] = {}
        for split, features_per_split in features_per_source["data"].items():
            if "BASELINE" in source:
                if "GT" in source:
                    split_key = "gt"
                elif "Raw" in source:
                    split_key = "raw"
                else:
                    logger.error(f"Unknown source: {source}")
                    raise ValueError(f"Unknown source: {source}")
            else:
                split_key = cfg["CLASSIFICATION_SETTINGS"]["DATA_SUBSET"]["split_key"]
            df: pl.DataFrame = features_per_source["data"][split][split_key]
            features_out[source]["data"][split] = df

    return features_out


def check_data_for_NaNs(source: str, features_per_source: dict[str, Any]) -> bool:
    """
    Check feature columns for NaN values.

    Parameters
    ----------
    source : str
        Data source name.
    features_per_source : dict
        Features for a single source with 'data' key.

    Returns
    -------
    bool
        True if no NaNs found in feature columns, False otherwise.
    """
    any_col_null_sums = False
    for split, df in features_per_source["data"].items():
        col_null_sums = df.select(pl.all().is_null().sum())
        for col in col_null_sums.columns:
            if col_null_sums[col][0] > 0:
                if "_value" in col:
                    logger.error(f"Found NaNs in {split} split, col_name: {col}")
                    any_col_null_sums = True

    if any_col_null_sums:
        logger.error(f"Found NaNs in feature columns, source = {source}")
        logger.error("This might easily happen for source 'BASELINE_OutlierRemovedRaw'")
        logger.error("As it has missing values")
        return False
    else:
        return True


def classifier_hpo_eval(
    y_true: np.ndarray,
    pred_proba: np.ndarray,
    eval_metric: str,
    model: str,
    hpo_method: str,
) -> float:
    """
    Evaluate classifier predictions for hyperparameter optimization.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    pred_proba : array-like
        Predicted class probabilities.
    eval_metric : str
        Metric to compute ('logloss', 'auc', 'f1').
    model : str
        Model name for logging.
    hpo_method : str
        HPO method ('hyperopt' negates loss for minimization).

    Returns
    -------
    float
        Computed metric value (negated for hyperopt).

    Raises
    ------
    ValueError
        If unknown eval_metric specified.
    """
    if eval_metric == "logloss":
        loss = log_loss(y_true, pred_proba)
    elif eval_metric == "auc":
        loss = roc_auc_score(y_true, pred_proba)
    elif eval_metric == "f1":
        pred = (pred_proba > 0.5).astype(int)
        loss = f1_score(y_true, pred, average="binary", zero_division=np.nan)
    else:
        logger.error("Unknown loss function {}".format(eval_metric))
        raise ValueError("Unknown loss function {}".format(eval_metric))

    if hpo_method == "hyperopt":
        loss = -1 * loss

    return loss
