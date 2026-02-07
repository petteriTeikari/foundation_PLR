from copy import deepcopy

import numpy as np
from loguru import logger
from omegaconf import DictConfig


def get_standardization_stats(split: str, col_name: str, data_dicts_df: dict):
    """Retrieve mean and standard deviation for standardization.

    Parameters
    ----------
    split : str
        Data split to compute statistics from (typically 'train').
    col_name : str
        Column name in the data dictionary to standardize.
    data_dicts_df : dict
        Nested dictionary containing data arrays organized by split and column.

    Returns
    -------
    tuple
        A tuple containing:
        - mean : float
            NaN-aware mean of the specified column.
        - std : float
            NaN-aware standard deviation of the specified column.
    """
    logger.debug("Standardizing on split = {}, dict_key = {}".format(split, col_name))
    mean = np.nanmean(data_dicts_df[split]["data"][col_name])
    std = np.nanstd(data_dicts_df[split]["data"][col_name])
    return mean, std


def standardize_the_data_dict(mean, stdev, data_dicts_df, cfg):
    """Apply standardization to all columns across all splits.

    Transforms each data column using z-score normalization:
    X_standardized = (X - mean) / stdev.

    Parameters
    ----------
    mean : float
        Mean value for standardization.
    stdev : float
        Standard deviation for standardization.
    data_dicts_df : dict
        Nested dictionary with structure {split: {'data': {col_name: array}}}.
    cfg : DictConfig
        Configuration dictionary (currently unused but kept for API consistency).

    Returns
    -------
    dict
        Updated data dictionary with standardized values.
    """
    for split in data_dicts_df.keys():
        for col_name in data_dicts_df[split]["data"].keys():
            logger.debug("Standardizing column: {}".format(col_name))
            array_tmp = data_dicts_df[split]["data"][col_name]
            array_tmp = (array_tmp - mean) / stdev
            data_dicts_df[split]["data"][col_name] = array_tmp

    return data_dicts_df


def destandardize_the_data_dict_for_featurization(
    split, split_dict, preprocess_dict, cfg
):
    """Destandardize data before feature extraction.

    Reverses standardization to restore original scale, which is required
    for computing physiologically meaningful handcrafted features.

    Parameters
    ----------
    split : str
        Data split identifier ('train', 'val', or 'test').
    split_dict : dict
        Dictionary containing data for a single split.
    preprocess_dict : dict
        Dictionary containing 'standardization' sub-dict with 'standardized',
        'mean', and 'stdev' keys.
    cfg : DictConfig
        Configuration dictionary.

    Returns
    -------
    dict
        Deep copy of split_dict with destandardized data values.
    """
    if preprocess_dict["standardization"]["standardized"]:
        logger.info(
            "Destandardizing the data for featurization, split = {}".format(split)
        )
        mean = preprocess_dict["standardization"]["mean"]
        stdev = preprocess_dict["standardization"]["stdev"]
        dicts_out = deepcopy(split_dict)
        dicts_out = destandardize_the_split_dict(dicts_out, split, stdev, mean, cfg)
    else:
        logger.info("No standardization applied, so no destandardization needed")
    return dicts_out


def destandardize_the_split_dict(data_dicts_df, split, stdev, mean, cfg):
    """Destandardize all non-mask columns in a split dictionary.

    Applies inverse z-score transformation: X_original = X_standardized * stdev + mean.

    Parameters
    ----------
    data_dicts_df : dict
        Dictionary containing 'data' sub-dict with column arrays.
    split : str
        Data split identifier (used for logging).
    stdev : float
        Standard deviation used in original standardization.
    mean : float
        Mean used in original standardization.
    cfg : DictConfig
        Configuration dictionary (currently unused).

    Returns
    -------
    dict
        Updated dictionary with destandardized values.

    Notes
    -----
    The 'mask' column is skipped as it contains boolean/integer flags,
    not continuous values that were standardized.
    """
    for col_name in data_dicts_df["data"].keys():
        if col_name != "mask":
            # or inverse transform as you wish to call this
            logger.debug("DeStandardizing column: {}".format(col_name))
            array_tmp = data_dicts_df["data"][col_name]
            array_tmp = (array_tmp * stdev) + mean
            data_dicts_df["data"][col_name] = array_tmp
    return data_dicts_df


def standardize_data_dicts(data_dicts: dict, cfg: DictConfig):
    """Standardize all data dictionaries using training set statistics.

    Computes mean and standard deviation from the training split and applies
    standardization across all splits. Stores computed statistics in the
    preprocess sub-dictionary.

    Parameters
    ----------
    data_dicts : dict
        Main data dictionary containing 'df' with nested split data.
    cfg : DictConfig
        Configuration with PREPROCESS.col_name specifying which column
        to use for computing statistics.

    Returns
    -------
    dict
        Updated data dictionary with standardized values and added
        'preprocess.standardization' metadata.
    """
    mean, stdev = get_standardization_stats(
        split="train",
        col_name=cfg["PREPROCESS"]["col_name"],
        data_dicts_df=data_dicts["df"],
    )

    logger.info("Standardizing, mean = {}, stdev = {}".format(mean, stdev))
    data_dicts["df"] = standardize_the_data_dict(
        mean=mean, stdev=stdev, data_dicts_df=data_dicts["df"], cfg=cfg
    )

    if "preprocess" not in data_dicts:
        data_dicts["preprocess"] = {}
        data_dicts["preprocess"]["standardization"] = {
            "standardized": True,
            "mean": mean,
            "stdev": stdev,
        }

    return data_dicts


def standardize_recons_arrays(array_in, stdz_dict: dict):
    """Standardize reconstruction arrays using stored statistics.

    Parameters
    ----------
    array_in : np.ndarray
        Input array to standardize.
    stdz_dict : dict
        Dictionary containing 'mean' and 'stdev' for standardization.

    Returns
    -------
    np.ndarray
        Standardized array (deep copy of input).
    """
    array_out = deepcopy(array_in)
    array_out = array_out - stdz_dict["mean"]
    array_out = array_out / stdz_dict["stdev"]
    return array_out


# NEW, move from other funcs eventually here and re-arrange
def preprocess_data_dicts(data_dicts: dict, cfg: DictConfig):
    """Main preprocessing entry point for data dictionaries.

    Applies configured preprocessing steps (currently only standardization)
    to the data dictionaries.

    Parameters
    ----------
    data_dicts : dict
        Main data dictionary containing 'df' with nested split data.
    cfg : DictConfig
        Configuration with PREPROCESS settings, including 'standardize' flag.

    Returns
    -------
    dict
        Preprocessed data dictionary.
    """
    if cfg["PREPROCESS"]["standardize"]:
        data_dicts = standardize_data_dicts(data_dicts=data_dicts, cfg=cfg)
    else:
        logger.info("No standardization applied")

    return data_dicts
