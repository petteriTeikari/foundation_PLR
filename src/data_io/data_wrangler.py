import decimal
from copy import deepcopy
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from src.preprocess.preprocess_PLR import preprocess_data_dicts


def convert_datadict_to_dict_arrays(
    data_dict: dict[str, Any], cls_model_cfg: DictConfig
) -> dict[str, Any]:
    """Convert hierarchical data dictionary to flat arrays structure.

    Needs to be this flat structure, used with some models.
    See data_transform_wrapper() -> create_dmatrices_and_dict_arrays().

    Parameters
    ----------
    data_dict : dict
        Hierarchical data dictionary with train/test splits.
    cls_model_cfg : DictConfig
        Classification model configuration.

    Returns
    -------
    dict
        Flat dictionary with x_train, y_train, x_test, y_test, etc.
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


def fix_pl_schema(df_metadata: pl.DataFrame) -> pl.DataFrame:
    """Cast object types in a Polars dataframe to appropriate types.

    Handles conversion of decimal.Decimal to float and ensures string types
    are properly cast.

    Parameters
    ----------
    df_metadata : pl.DataFrame
        Polars dataframe with potential Object dtype columns.

    Returns
    -------
    pl.DataFrame
        Dataframe with Object types cast to appropriate types.

    See Also
    --------
    convert_object_type : Similar function for numpy arrays.

    References
    ----------
    https://docs.pola.rs/user-guide/expressions/casting/#basic-example
    """

    def get_sample_value(sample_col: pl.Series) -> Any:
        # get first non-None value from Polars Series
        for sample in sample_col:
            if sample is not None:
                return sample
        return None

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
    subject_dict: dict[str, dict[str, np.ndarray]],
    wildcard_categories: list[str] | None = None,
) -> pl.DataFrame:
    """Convert a subject dictionary of arrays to a Polars dataframe.

    Parameters
    ----------
    subject_dict : dict
        Dictionary with category names as keys containing sub-dictionaries
        with array data.
    wildcard_categories : list, optional
        If provided, only include categories in this list, by default None.

    Returns
    -------
    pl.DataFrame
        Polars dataframe with arrays as columns.

    Raises
    ------
    AssertionError
        If any array is not 1D.
    """
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
                    assert len(array.shape) == 1, (
                        f"Array shape is not 1D: {array.shape}"
                    )
                    array = convert_object_type(
                        array
                    )  # if possible Object types, causing downstream issues
                    df = df.with_columns(pl.Series(name=subkey, values=array))
    return df


def get_subject_dict_for_featurization(
    split_dict: dict[str, dict[str, np.ndarray]],
    i: int,
    cfg: DictConfig,
    return_1st_value: bool = False,
) -> dict[str, dict[str, Any]]:
    """Extract a single subject's data from a split dictionary.

    Parameters
    ----------
    split_dict : dict
        Dictionary containing data for all subjects in a split.
    i : int
        Subject index to extract.
    cfg : DictConfig
        Configuration dictionary.
    return_1st_value : bool, optional
        If True, return only the first value; otherwise return the full row,
        by default False.

    Returns
    -------
    dict
        Dictionary containing only the specified subject's data.
    """
    subject_dict = deepcopy(split_dict)
    for category_name, category_dict in split_dict.items():
        for subkey, array in category_dict.items():
            if return_1st_value:
                subject_dict[category_name][subkey] = array[i, 0]
            else:
                subject_dict[category_name][subkey] = array[i, :]
    return subject_dict


def pick_correct_data_and_label_for_experiment(
    data_dict: dict[str, Any], cfg: DictConfig, task: str, _task_cfg: DictConfig
) -> None:
    """Select appropriate data columns for a specific experiment task.

    Parameters
    ----------
    data_dict : dict
        Full data dictionary.
    cfg : DictConfig
        Configuration dictionary.
    task : str
        Task name.
    _task_cfg : DictConfig
        Task-specific configuration (currently unused).

    Notes
    -----
    Currently a placeholder function.
    """
    print("placeholder")


def get_dict_with_wildcard(
    df: pl.DataFrame, wildcard: str = "pupil"
) -> dict[str, pl.Series]:
    """Extract columns matching a wildcard pattern as a dictionary.

    Parameters
    ----------
    df : pl.DataFrame
        Input Polars dataframe.
    wildcard : str, optional
        Pattern to match in column names, by default "pupil".

    Returns
    -------
    dict
        Dictionary mapping column names to Polars Series.
    """
    # Get all columns that have the wildcard
    cols = [col for col in df.columns if wildcard in col]

    # Create a dictionary with the wildcard columns
    data_dict = {}
    for col in cols:
        data_dict[col] = df[col]

    return data_dict


def get_dict_with_list_of_cols(
    df: pl.DataFrame, cols: list[str]
) -> dict[str, pl.Series]:
    """Extract specific columns as a dictionary.

    Parameters
    ----------
    df : pl.DataFrame
        Input Polars dataframe.
    cols : list
        List of column names to extract.

    Returns
    -------
    dict
        Dictionary mapping column names to Polars Series.
    """
    # Create a dictionary with the wildcard columns
    data_dict = {}
    for col in cols:
        data_dict[col] = df[col]

    return data_dict


def get_dict_with_remaining_cols(
    df_split: pl.DataFrame, data_dict: dict[str, dict[str, Any]]
) -> dict[str, pl.Series]:
    """Extract columns not already present in data_dict as a dictionary.

    Parameters
    ----------
    df_split : pl.DataFrame
        Input Polars dataframe.
    data_dict : dict
        Existing data dictionary to check for used columns.

    Returns
    -------
    dict
        Dictionary mapping remaining column names to Polars Series.
    """
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


def convert_object_type(array_tmp: np.ndarray) -> np.ndarray:
    """Convert numpy object dtype arrays to appropriate types.

    Handles conversion of decimal.Decimal to float and str to string dtype.

    Parameters
    ----------
    array_tmp : np.ndarray
        Array potentially with object dtype.

    Returns
    -------
    np.ndarray
        Array with appropriate dtype (float or str).
    """
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


def reshape_flat_series_to_2d_arrays(
    dict_series: dict[str, dict[str, pl.Series]], length_PLR: int = 1981
) -> dict[str, dict[str, np.ndarray]]:
    """Reshape flat Polars Series to 2D numpy arrays.

    Converts a dictionary of Series to 2D arrays with shape
    (n_subjects, n_timepoints).

    Parameters
    ----------
    dict_series : dict
        Dictionary of dictionaries containing Polars Series.
    length_PLR : int, optional
        Number of timepoints per subject, by default 1981.

    Returns
    -------
    dict
        Dictionary with same structure but 2D numpy arrays.
    """
    dict_arrays = {}
    for key1, dict in dict_series.items():
        dict_arrays[key1] = {}
        for key2, series in dict.items():
            array_tmp = convert_object_type(array_tmp=series.to_numpy())
            dict_arrays[key1][key2] = array_tmp.reshape(-1, length_PLR)

    return dict_arrays


def split_df_to_dict(
    df_split: pl.DataFrame, cfg: DictConfig, split: str
) -> dict[str, dict[str, np.ndarray]]:
    """Convert a split dataframe to a hierarchical dictionary of 2D arrays.

    Creates a structured dictionary with categories: time, data, labels,
    light, and metadata. All values are reshaped to (n_subjects, n_timepoints).

    Parameters
    ----------
    df_split : pl.DataFrame
        Polars dataframe for a single split.
    cfg : DictConfig
        Configuration dictionary with DATA settings.
    split : str
        Name of the split (for logging).

    Returns
    -------
    dict
        Hierarchical dictionary with 2D numpy arrays.
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


def convert_df_to_dict(data_df: pl.DataFrame, cfg: DictConfig) -> dict[str, Any]:
    """Convert a Polars dataframe to a hierarchical dictionary for model input.

    Converts the combined dataframe into a structured dictionary that can
    be used with various ML frameworks:
    - sklearn: (X_train, X_val, y_train, y_val)
    - PyTorch: (dataloader, dataset)

    Parameters
    ----------
    data_df : pl.DataFrame
        Combined Polars dataframe with 'split' column.
    cfg : DictConfig
        Configuration dictionary with DATA settings.

    Returns
    -------
    dict
        Dictionary with 'df' key containing split dictionaries and
        'preprocess' key with preprocessing parameters.
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
