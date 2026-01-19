from copy import deepcopy

import numpy as np
from omegaconf import DictConfig
from loguru import logger


def get_standardization_stats(split: str, col_name: str, data_dicts_df: dict):
    logger.debug("Standardizing on split = {}, dict_key = {}".format(split, col_name))
    mean = np.nanmean(data_dicts_df[split]["data"][col_name])
    std = np.nanstd(data_dicts_df[split]["data"][col_name])
    return mean, std


def standardize_the_data_dict(mean, stdev, data_dicts_df, cfg):
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
    for col_name in data_dicts_df["data"].keys():
        if col_name != "mask":
            # or inverse transform as you wish to call this
            logger.debug("DeStandardizing column: {}".format(col_name))
            array_tmp = data_dicts_df["data"][col_name]
            array_tmp = (array_tmp * stdev) + mean
            data_dicts_df["data"][col_name] = array_tmp
    return data_dicts_df


def standardize_data_dicts(data_dicts: dict, cfg: DictConfig):
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
    array_out = deepcopy(array_in)
    array_out = array_out - stdz_dict["mean"]
    array_out = array_out / stdz_dict["stdev"]
    return array_out


# NEW, move from other funcs eventually here and re-arrange
def preprocess_data_dicts(data_dicts: dict, cfg: DictConfig):
    if cfg["PREPROCESS"]["standardize"]:
        data_dicts = standardize_data_dicts(data_dicts=data_dicts, cfg=cfg)
    else:
        logger.info("No standardization applied")

    return data_dicts
