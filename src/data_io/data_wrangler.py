import decimal
from copy import deepcopy

import numpy as np
from loguru import logger
import polars as pl
from omegaconf import DictConfig

from src.preprocess.preprocess_PLR import preprocess_data_dicts


def convert_datadict_to_dict_arrays(data_dict: dict, cls_model_cfg: DictConfig):
    """
    Needs to be this flat structure, used with some models
    see data_transform_wrapper()
     -> create_dmatrices_and_dict_arrays()
    """
    dict_arrays = {
        "x_train": data_dict["train"]["data"]["X"],
        "x_train_w": np.ones_like((data_dict["train"]["data"]["X"])),
        "y_train": data_dict["train"]["labels"]["class_label"][:, 0],
        "x_test": data_dict["test"]["data"]["X"],
        "x_test_w": np.ones_like((data_dict["test"]["data"]["X"])),
        "y_test": data_dict["test"]["labels"]["class_label"][:, 0],
        "feature_names": None,
        "subject_codes_train": data_dict["train"]["metadata"]["subject_code"][:, 0],
        "subject_codes_test": data_dict["test"]["metadata"]["subject_code"][:, 0],
    }

    return dict_arrays


def fix_pl_schema(df_metadata: pl.DataFrame):
    """
    Cast object types into something else
    See e.g. array = convert_object_type(array) # if possible Object types, causing downstream issues
    https://docs.pola.rs/user-guide/expressions/casting/#basic-example
    """

    def get_sample_value(sample_col: pl.Series):
        # get first non-None value from Polars Series
        for sample in sample_col:
            if sample is not None:
                return sample

    for col, dtype in zip(df_metadata.columns, df_metadata.dtypes):
        if dtype == pl.Object:
            sample_value = get_sample_value(sample_col=df_metadata[col])
            if isinstance(sample_value, decimal.Decimal):
                # e.g. Age type comes out like this
                # print(1, col)
                numpy_array = df_metadata[col].to_numpy().astype(float)
                df_metadata = df_metadata.with_columns(
                    pl.Series(name=col, values=numpy_array)
                )
            elif isinstance(sample_value, str):
                # e.g. "subject_code" type comes out like this
                # print(2, col)
                numpy_array = df_metadata[col].to_numpy().astype(str)
                df_metadata = df_metadata.with_columns(
                    pl.Series(name=col, values=numpy_array)
                )
            else:
                logger.warning("Casting issue with dtype = {}".format(dtype))

    return df_metadata


def convert_subject_dict_of_arrays_to_df(
    subject_dict: dict, wildcard_categories: list = None
) -> pl.DataFrame:
    df = pl.DataFrame()
    for category_name, category_dict in subject_dict.items():
        if wildcard_categories is None:
            for subkey, array in category_dict.items():
                assert len(array.shape) == 1, f"Array shape is not 1D: {array.shape}"
                array = convert_object_type(
                    array
                )  # if possible Object types, causing downstream issues
                df = df.with_columns(pl.Series(name=subkey, values=array))
        else:
            if category_name in wildcard_categories:
                for subkey, array in category_dict.items():
                    assert (
                        len(array.shape) == 1
                    ), f"Array shape is not 1D: {array.shape}"
                    array = convert_object_type(
                        array
                    )  # if possible Object types, causing downstream issues
                    df = df.with_columns(pl.Series(name=subkey, values=array))
    return df


def get_subject_dict_for_featurization(
    split_dict: dict, i: int, cfg: DictConfig, return_1st_value: bool = False
):
    subject_dict = deepcopy(split_dict)
    for category_name, category_dict in split_dict.items():
        for subkey, array in category_dict.items():
            if return_1st_value:
                subject_dict[category_name][subkey] = array[i, 0]
            else:
                subject_dict[category_name][subkey] = array[i, :]
    return subject_dict


def pick_correct_data_and_label_for_experiment(
    data_dict, cfg, task: str, task_cfg: DictConfig
):
    print("placeholder")


def get_dict_with_wildcard(df: pl.DataFrame, wildcard: str = "pupil"):
    # Get all columns that have the wildcard
    cols = [col for col in df.columns if wildcard in col]

    # Create a dictionary with the wildcard columns
    data_dict = {}
    for col in cols:
        data_dict[col] = df[col]

    return data_dict


def get_dict_with_list_of_cols(df: pl.DataFrame, cols: list):
    # Create a dictionary with the wildcard columns
    data_dict = {}
    for col in cols:
        data_dict[col] = df[col]

    return data_dict


def get_dict_with_remaining_cols(df_split, data_dict):
    # Get all the columns that are not in the data_dict
    used_cols = []
    for key1, dict in data_dict.items():
        for key2 in dict.keys():
            used_cols.append(key2)

    remaining_cols = [col for col in df_split.columns if col not in used_cols]

    # Create a dictionary with the wildcard columns
    data_dict_remaining = {}
    for col in remaining_cols:
        data_dict_remaining[col] = df_split[col]

    return data_dict_remaining


def convert_object_type(array_tmp):
    # these might cause weird unintuitive issues downstream
    if array_tmp.dtype == "object":
        first_value = array_tmp[0]
        if isinstance(first_value, decimal.Decimal):
            # e.g. Age type comes out like this
            array_tmp = array_tmp.astype(float)
        elif isinstance(first_value, str):
            # e.g. "subject_code" type comes out like this
            array_tmp = array_tmp.astype(str)

    return array_tmp


def reshape_flat_series_to_2d_arrays(dict_series: dict, length_PLR: int = 1981):
    dict_arrays = {}
    for key1, dict in dict_series.items():
        dict_arrays[key1] = {}
        for key2, series in dict.items():
            array_tmp = convert_object_type(array_tmp=series.to_numpy())
            dict_arrays[key1][key2] = array_tmp.reshape(-1, length_PLR)

    return dict_arrays


def split_df_to_dict(df_split: pl.DataFrame, cfg: DictConfig, split: str):
    """
    Try to keep the manual work only on this function, and have all the downstream code work on the dictionary
    """

    # Hierarchical dictionary, so you can easily add stuff later on
    # These will be pl.Series
    data_dict = {}
    data_dict["time"] = get_dict_with_wildcard(df_split, wildcard="time")
    data_dict["data"] = get_dict_with_wildcard(df_split, wildcard="pupil")
    data_dict["labels"] = get_dict_with_list_of_cols(
        df_split,
        cols=[
            "class_label",
            "outlier_mask",
            "imputation_mask",
            "outlier_mask_easy",
            "outlier_mask_medium",
        ],
    )
    data_dict["light"] = get_dict_with_list_of_cols(
        df_split, cols=["Red", "Blue", "light_stimuli"]
    )
    data_dict["metadata"] = get_dict_with_remaining_cols(df_split, data_dict)

    # Convert pl.Series to 2D numpy arrays (no_subjects, no_timepoints)
    data_dict = reshape_flat_series_to_2d_arrays(
        dict_series=data_dict, length_PLR=cfg["DATA"]["PLR_length"]
    )

    return data_dict


def convert_df_to_dict(data_df: pl.DataFrame, cfg: DictConfig):
    """
    All different models might want their data in different formats
    So convert from the "data_df" Polars dataframe ->
    - sklearn use (X_train, X_val, y_train, y_val)
    - Pytorch use (dataloader, dataset)

    Convert the Polars DataFrame to a dictionary
    Think of doing some PLRData class later?
    """

    data_dicts = {}
    data_dicts["df"] = {}
    for i, split in enumerate(data_df["split"].unique().to_list()):
        # Dataframe into a nested dictionary (categories) with 2D numpy arrays
        # (no_subjects, no_timepoints)
        data_dicts["df"][split] = split_df_to_dict(
            df_split=data_df.filter(pl.col("split") == split), cfg=cfg, split=split
        )

    # Preprocess if desired
    data_dicts = preprocess_data_dicts(data_dicts=data_dicts, cfg=cfg)

    return data_dicts
