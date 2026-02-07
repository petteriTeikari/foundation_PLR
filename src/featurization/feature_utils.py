import numbers
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from src.data_io.data_utils import get_unique_polars_rows
from src.log_helpers.local_artifacts import load_results_dict, save_results_dict
from src.log_helpers.log_naming_uris_and_dirs import (
    define_featurization_run_name_from_base,
    get_baseline_names,
    get_features_pickle_fname,
)
from src.preprocess.preprocess_data import destandardize_dict, destandardize_numpy
from src.utils import get_artifacts_dir, pandas_col_condition_filter, pandas_concat


def if_refeaturize(cfg: DictConfig) -> bool:
    """Determine whether featurization should be re-run.

    Checks if re-featurization is forced by config or if no existing
    features are found on disk.

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary containing PLR_FEATURIZATION settings.

    Returns
    -------
    bool
        True if featurization should be performed, False if existing
        features should be loaded from disk.
    """
    re_featurize = cfg["PLR_FEATURIZATION"]["re_featurize"]
    file_path = get_features_fpath(cfg)

    features_found = False
    if Path(file_path).exists():
        features_found = True

    if not re_featurize and features_found:
        logger.info("Reading precomputed features from the disk: {}".format(file_path))
        return False
    else:
        if re_featurize:
            logger.info("Recomputing the features (re_featurize is set to True)")
            return True
        else:
            logger.info(
                "Re-featurization is set to False, but no features found from the disk -> re-featurizing"
            )
            return True


def combine_df_with_outputs(
    df: pl.DataFrame,
    data_dict: dict[str, np.ndarray],
    imputation: dict[str, Any],
    split: str,
    split_key: str,
    model_name: str,
) -> pl.DataFrame:
    """Combine dataframe with imputation outputs and standardized inputs.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe to combine with.
    data_dict : dict
        Dictionary containing standardized input data arrays.
    imputation : dict
        Dictionary containing imputation results.
    split : str
        Data split identifier (e.g., 'train', 'test').
    split_key : str
        Split key identifier (e.g., 'train_gt', 'train_raw').
    model_name : str
        Name of the imputation model.

    Returns
    -------
    pl.DataFrame
        Dataframe with imputation and standardized input columns added.
    """
    df = combine_inputation_with_df(df, imputation)
    df = combine_standardized_inputs_with_df(df, data_dict)

    return df


def combine_standardized_inputs_with_df(
    df: pl.DataFrame, data_dict: dict[str, np.ndarray]
) -> pl.DataFrame:
    """Add standardized input arrays as columns to a Polars dataframe.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe to add columns to.
    data_dict : dict
        Dictionary with keys as column names and values as numpy arrays.

    Returns
    -------
    pl.DataFrame
        Dataframe with new columns prefixed with 'standardized_'.

    Raises
    ------
    AssertionError
        If number of samples in array doesn't match dataframe rows.
    """
    logger.debug("Combining the standardized inputs with the dataframe")
    no_samples_in = df.shape[0]
    for data_key in data_dict.keys():
        array = data_dict[data_key]
        array_flattened = array.flatten()
        assert no_samples_in == array_flattened.shape[0], (
            "Number of samples in the imputation and the data should be the same"
        )
        df = df.with_columns(pl.lit(array_flattened).alias("standardized_" + data_key))

    return df


def combine_inputation_with_df(
    df: pl.DataFrame, imputation: dict[str, Any]
) -> pl.DataFrame:
    """Add imputation results as columns to a Polars dataframe.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe to add columns to.
    imputation : dict
        Dictionary containing 'imputation' sub-dict with arrays and
        'indicating_mask' for missingness.

    Returns
    -------
    pl.DataFrame
        Dataframe with imputation columns and missingness mask added.

    Raises
    ------
    AssertionError
        If number of samples in arrays doesn't match dataframe rows.
    """
    logger.debug("Combining the imputation input with the dataframe")
    no_samples_in = df.shape[0]
    for imputation_stats_key in imputation["imputation"].keys():
        array = imputation["imputation"][imputation_stats_key]
        if "imputation" not in imputation_stats_key:
            # quick fix to add the imputation prefix if it not there so it is easier
            # see what the columns are all about
            key_out = "imputation_" + imputation_stats_key
        else:
            key_out = imputation_stats_key
        if array is None:
            df = df.with_columns(pl.lit(None).alias(key_out))
        else:
            array_flattened = array.flatten()
            assert no_samples_in == array_flattened.shape[0], (
                "Number of samples in the imputation and the data should be the same"
            )
            df = df.with_columns(pl.lit(array_flattened).alias(key_out))

    missingness_mask = imputation["indicating_mask"]
    missingness_flattened = missingness_mask.flatten()
    assert no_samples_in == missingness_flattened.shape[0], (
        "Number of samples in the imputation and the data should be the same"
    )
    df = df.with_columns(
        pl.lit(missingness_flattened).alias("imputation_missingness_mask")
    )

    return df


def subjects_with_class_labels(df: pl.DataFrame, split: str) -> pl.DataFrame:
    """Get unique subject codes that have class labels (glaucoma/control).

    Parameters
    ----------
    df : pl.DataFrame
        Dataframe containing subject data with 'subject_code' and 'class_label'.
    split : str
        Data split identifier (e.g., 'train', 'test').

    Returns
    -------
    pl.DataFrame
        Sorted dataframe with unique subject codes that have non-null class labels.
    """
    unique_codes = get_unique_polars_rows(
        df,
        unique_col="subject_code",
        value_col="class_label",
        split=split,
        df_string="PLR",
    )

    unique_codes = unique_codes.sort("subject_code")

    # drop rows with no class_label from Polars dataframe
    unique_codes = unique_codes.filter(
        ~pl.all_horizontal(pl.col("class_label").is_null())
    )

    logger.info(
        "Number of subjects with a class label (glaucoma vs control) : {}".format(
            len(unique_codes)
        )
    )

    return unique_codes


def pick_correct_split(
    data_dict: dict[str, Any],
    split: str,
    split_key: str,
    eval_results: dict[str, Any],
    model_name: str,
    standardize_stats: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Select and destandardize evaluation results for the correct data split.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing preprocessed data.
    split : str
        Data split name (e.g., 'train', 'test').
    split_key : str
        Split key containing 'gt' or 'raw' suffix.
    eval_results : dict
        Dictionary containing evaluation results keyed by split_key.
    model_name : str
        Name of the model being processed.
    standardize_stats : dict
        Dictionary with 'gt' and 'raw' sub-dicts containing mean/std.

    Returns
    -------
    dict
        Destandardized results for the specified split key.

    Raises
    ------
    ValueError
        If split_key doesn't contain 'gt' or 'raw'.
    """
    # pick the split (train, val)
    logger.info(
        "Split key = {}, Featurizing the results from model {}".format(
            split_key, model_name
        )
    )
    if "gt" in split_key:
        logger.debug("Standardizing the results from model {} using the gt stats")
        stdz_dict = standardize_stats["gt"]
    elif "raw" in split_key:
        logger.debug("Standardizing the results from model {} using the raw stats")
        stdz_dict = standardize_stats["raw"]
    else:
        logger.error(
            'How come you have split_key = "{}" here? Should be either gt or raw'.format(
                split_key
            )
        )
        raise ValueError(
            'How come you have split_key = "{}" here? Should be either gt or raw'.format(
                split_key
            )
        )

    split_key_results = eval_results[split_key]
    split_key_results["imputation"] = destandardize_dict(
        imputation_dict=split_key_results["imputation"],
        mean=stdz_dict["mean"],
        std=stdz_dict["std"],
    )

    return split_key_results


def pick_input_data(
    input_data: dict[str, np.ndarray], split: str, split_key: str, model_name: str
) -> dict[str, np.ndarray]:
    """Select input data arrays for the correct split and data type.

    Parameters
    ----------
    input_data : dict
        Dictionary containing preprocessed data with keys like 'X_train_gt'.
    split : str
        Data split name, e.g., 'train'.
    split_key : str
        Split key, e.g., 'train_gt' or 'train_raw'.
    model_name : str
        Name of the model, e.g., 'CSDI'.

    Returns
    -------
    dict
        Dictionary with 'X' (selected data) and 'X_gt' (ground truth).

    Raises
    ------
    ValueError
        If split_key doesn't contain 'gt' or 'raw'.
    """
    if "raw" in split_key:
        X = input_data[f"X_{split}_raw"]
    elif "gt" in split_key:
        X = input_data[f"X_{split}_gt"]
    else:
        logger.error(
            'How come you have split_key = "{}" here? Should be either gt or raw'.format(
                split_key
            )
        )
        raise ValueError(
            'How come you have split_key = "{}" here? Should be either gt or raw'.format(
                split_key
            )
        )

    X_gt = input_data[f"X_{split}_gt"]

    return {"X": X, "X_gt": X_gt}


def get_light_stimuli_timings(
    df_subject: pl.DataFrame,
) -> dict[str, dict[str, float]]:
    """Extract light stimulus onset/offset timings for red and blue colors.

    Parameters
    ----------
    df_subject : pl.DataFrame
        Subject dataframe containing 'Red', 'Blue', and 'time' columns.

    Returns
    -------
    dict
        Dictionary with 'Red' and 'Blue' keys, each containing:
        - 'light_onset': float, time of light onset
        - 'light_offset': float, time of light offset
        - 'light_duration': float, duration of light stimulus
    """

    def set_null_to_zero(df: pl.DataFrame, col: str) -> pl.DataFrame:
        df = df.with_columns(
            (
                pl.when(pl.col(col).is_null()).then(pl.lit(0)).otherwise(pl.col(col))
            ).alias(col)
        )
        return df

    def individual_light_timing(
        df_subject: pl.DataFrame, color: str = "Red"
    ) -> dict[str, float]:
        df_subject = df_subject.fill_nan(0)
        light_offset_row = get_top1_of_col(df=df_subject, col=color, descending=True)
        light_onset_row = get_top1_of_col(df=df_subject, col=color, descending=False)
        assert light_offset_row.item(0, "time") > light_onset_row.item(0, "time"), (
            "Light offset should be after the light onset, but got "
            "light_onset = {} and light_offset = {}".format(
                light_onset_row.item(0, "time"), light_offset_row.item(0, "time")
            )
        )
        light_duration = light_offset_row.item(0, "time") - light_onset_row.item(
            0, "time"
        )

        return {
            "light_onset": light_onset_row.item(0, "time"),
            "light_offset": light_offset_row.item(0, "time"),
            "light_duration": light_duration,
        }

    timings = {}
    timings["Red"] = individual_light_timing(df_subject, color="Red")
    timings["Blue"] = individual_light_timing(df_subject, color="Blue")

    return timings


def get_top1_of_col(df: pl.DataFrame, col: str, descending: bool) -> pl.DataFrame:
    """Get the first or last row (by time) where a column has non-zero values.

    Used to find light onset (first timepoint where light=1) or light offset
    (last timepoint where light=1).

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe with 'time' column.
    col : str
        Column name to filter by (e.g., 'Red', 'Blue').
    descending : bool
        If True, get the LAST timepoint (light offset).
        If False, get the FIRST timepoint (light onset).

    Returns
    -------
    pl.DataFrame
        Single-row dataframe with the onset or offset timepoint.

    Raises
    ------
    AssertionError
        If no samples remain after filtering null values.
    """
    df = replace_zeros_with_null(df, col=col)
    df = df.filter(~pl.all_horizontal(pl.col(col).is_null()))
    assert df.shape[0] > 0, "No samples in the dataframe"
    # Sort by TIME to get first/last timepoint where light is on
    df = df.sort("time", descending=descending)
    df_row = df[0]
    return df_row


def replace_zeros_with_null(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Replace zero values with NaN in a specified column.

    Used to identify light onset/offset where zero indicates light-off periods.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe.
    col : str
        Column name to process.

    Returns
    -------
    pl.DataFrame
        Dataframe with zeros replaced by NaN in the specified column.
    """
    # as in when the light was not on, the value is zero
    # first and last non-zero value is the light onset and offset
    # df = df.with_columns(
    #     (pl.when(pl.col(col) == 0).then(pl.lit(None)).otherwise(pl.col(col))).alias(col)
    # )
    df_pd = df.to_pandas()
    df_pd[col] = df_pd[col].replace(0, np.nan)
    return pl.DataFrame(df_pd)


def convert_relative_timing_to_absolute_timing(
    light_timing: dict[str, float],
    feature_params: dict[str, Any],
    color: str,
    feature: str,
    feature_cfg: DictConfig,
) -> dict[str, Any]:
    """Convert relative feature timing to absolute timing based on light stimulus.

    Parameters
    ----------
    light_timing : dict
        Dictionary with 'light_onset' and 'light_offset' times.
    feature_params : dict
        Feature parameters with 'time_from', 'time_start', and 'time_end'.
    color : str
        Light color ('Red' or 'Blue').
    feature : str
        Feature name being computed.
    feature_cfg : DictConfig
        Feature configuration.

    Returns
    -------
    dict
        Updated feature_params with absolute 'time_start' and 'time_end'.

    Raises
    ------
    ValueError
        If 'time_from' is not 'onset' or 'offset'.
    """
    if feature_params["time_from"] == "onset":
        t0 = light_timing["light_onset"]
    elif feature_params["time_from"] == "offset":
        t0 = light_timing["light_offset"]
    else:
        logger.error("Unknown time_from = {}".format(feature_params["time_from"]))
        raise ValueError("Unknown time_from = {}".format(feature_params["time_from"]))

    feature_params["time_start"] = t0 + feature_params["time_start"]
    feature_params["time_end"] = t0 + feature_params["time_end"]

    return feature_params


def get_feature_samples(
    df_subject: pl.DataFrame,
    feature_params: dict[str, Any],
    col: str = "time",
    feature: Optional[str] = None,
) -> pl.DataFrame:
    """Filter dataframe to samples within a time window for feature extraction.

    Parameters
    ----------
    df_subject : pl.DataFrame
        Subject dataframe with time series data.
    feature_params : dict
        Dictionary with 'time_start' and 'time_end' defining the window.
    col : str, optional
        Column name for time values, by default 'time'.
    feature : str, optional
        Feature name for logging, by default None.

    Returns
    -------
    pl.DataFrame
        Filtered dataframe with samples within the time window.

    Notes
    -----
    Uses pandas conversion to avoid Polars Rust errors with Object arrays.
    See https://github.com/pola-rs/polars/issues/18399
    """
    logger.debug("f{feature}: {feature_params}")
    df_pd = df_subject.to_pandas()
    df_pd = df_pd[
        (df_pd[col] >= feature_params["time_start"])
        & (df_pd[col] <= feature_params["time_end"])
    ]
    feature_samples = pl.from_pandas(df_pd)

    return feature_samples


def flatten_dict_to_dataframe(
    features_nested: dict[str, dict[str, Any]],
    mlflow_series: Optional[pl.Series],
    cfg: DictConfig,
) -> dict[str, Any]:
    """Convert nested features dictionary to flat dataframe structure.

    Parameters
    ----------
    features_nested : dict
        Nested dictionary keyed by split, then by subject code, containing features.
    mlflow_series : pl.Series or None
        MLflow run information as a Polars series.
    cfg : DictConfig
        Configuration dictionary.

    Returns
    -------
    dict
        Dictionary with 'data' (containing flattened dataframes per split)
        and 'mlflow_run' information.
    """
    # Init the dict with the MLflow run the same for all the splits
    features_df = {
        "data": {},
        "mlflow_run": dict(mlflow_series) if mlflow_series is not None else None,
    }

    for j, split in enumerate(features_nested.keys()):
        features_df["data"][split] = {}
        subjects_as_dicts = features_nested[split]
        features_df["data"][split] = flatten_subject_dicts_to_df(subjects_as_dicts, cfg)

    return features_df


def flatten_subject_dicts_to_df(
    subjects_as_dicts: dict[str, dict[str, Any]], cfg: DictConfig
) -> pl.DataFrame:
    """Convert subject-wise feature dictionaries to a single dataframe.

    Parameters
    ----------
    subjects_as_dicts : dict
        Dictionary keyed by subject_code containing feature dictionaries.
    cfg : DictConfig
        Configuration dictionary.

    Returns
    -------
    pl.DataFrame
        Dataframe with one row per subject and feature columns.
    """
    # Each dictionary becomes a row in the dataframe, with the computed features as columns
    df_features = None
    for i, subject_code in enumerate(subjects_as_dicts.keys()):
        df_row = create_df_row(
            subject_dict=subjects_as_dicts[subject_code],
            subject_code=subject_code,
            cfg=cfg,
        )
        # TODO "direct concatenation"?
        if df_features is None:
            df_features = df_row.to_pandas()
        else:
            df_features = pd.concat(
                [df_features, df_row.to_pandas()], ignore_index=True, axis=0
            )

    return pl.from_pandas(df_features)


def create_df_row(
    subject_dict: dict[str, Any], subject_code: str, cfg: DictConfig
) -> pl.DataFrame:
    """Create a single dataframe row from a subject's feature dictionary.

    Parameters
    ----------
    subject_dict : dict
        Dictionary containing features keyed by color, then feature name.
    subject_code : str
        Unique identifier for the subject.
    cfg : DictConfig
        Configuration dictionary with stat_keys to extract.

    Returns
    -------
    pl.DataFrame
        Single-row dataframe with flattened feature columns.
    """
    statkeys_to_pick = cfg["PLR_FEATURIZATION"]["FEATURIZATION"]["stat_keys"]
    dict_features = {"subject_code": subject_code}
    for m, color in enumerate(subject_dict.keys()):
        color_features = subject_dict[color]
        if isinstance(color_features, dict):
            for n, feature_name in enumerate(color_features.keys()):
                feature_name_flat = f"{color}_{feature_name}"
                # This is the feature dictionary with the stat keys (value, std, CI, etc.)
                feature_dict = color_features[feature_name]
                for o, dict_key in enumerate(feature_dict.keys()):
                    if dict_key in statkeys_to_pick:
                        feature_name_out = f"{feature_name_flat}_{dict_key}"
                        dict_features[feature_name_out] = feature_dict[dict_key]
        elif isinstance(color_features, pl.DataFrame):
            # Metadata (e.g. glaucoma or not, age, or whatever)
            for col in color_features.columns:
                col_out = f"metadata_{col}"
                dict_features[col_out] = color_features[col][0]
    return pl.DataFrame(dict_features)


def get_features_fpath(cfg: DictConfig, service_name: str = "best_models") -> str:
    """Construct the file path for saving/loading features.

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary with DATA and ARTIFACTS settings.
    service_name : str, optional
        Service name for artifacts directory, by default 'best_models'.

    Returns
    -------
    str
        Full file path for the features file.
    """
    duckdb_path = Path(cfg["DATA"]["filename_DuckDB"])
    filename = (
        duckdb_path.stem
        + cfg["PLR_FEATURIZATION"]["feature_file_suffix"]
        + "."
        + cfg["ARTIFACTS"]["results_format"]
    )
    artifacts_dir = Path(get_artifacts_dir(service_name=service_name))
    return str(artifacts_dir / filename)


def export_features_to_disk(dict_out: dict[str, Any], cfg: DictConfig) -> None:
    """Save features dictionary to disk as a pickle file.

    Parameters
    ----------
    dict_out : dict
        Features dictionary to save.
    cfg : DictConfig
        Configuration dictionary for determining file path.
    """
    features_filepath = Path(get_features_fpath(cfg))
    features_filepath.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting the features to the disk: {}".format(features_filepath))
    save_results_dict(dict_out, str(features_filepath), debug_load=True)


def load_features_from_disk(cfg: DictConfig) -> dict[str, Any]:
    """Load features dictionary from disk.

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary for determining file path.

    Returns
    -------
    dict
        Loaded features dictionary.
    """
    features_filepath = get_features_fpath(cfg)
    features = load_results_dict(features_filepath)
    return features


def get_feature_names(
    features: dict[str, Any],
    cols_exclude: tuple[str, ...] = ("subject_code",),
    name_substring: str = "_value",
) -> list[str]:
    """Extract feature names from a nested features dictionary.

    Parameters
    ----------
    features : dict
        Nested features dictionary with source -> data -> split structure.
    cols_exclude : tuple, optional
        Column names to exclude, by default ('subject_code',).
    name_substring : str, optional
        Substring to filter column names, by default '_value'.

    Returns
    -------
    list
        List of feature names with the substring removed.
    """

    def filter_for_col_substring(col_names, name_substring):
        # we now have Blue_PIPR_value, and Blue_PIPR_std so we do not duplicate names
        col_names = [col for col in col_names if name_substring in col]
        # get rid of the substring
        return [col.replace(name_substring, "") for col in col_names]

    def exclude_col_names(col_names, cols_exclude):
        return [col for col in col_names if col not in cols_exclude]

    for i, source in enumerate(features):
        if i == 0:
            for j, split in enumerate(features[source]["data"]):
                if j == 0:
                    for k, split_key in enumerate(features[source]["data"][split]):
                        if k == 0:
                            col_names = features[source]["data"][split][
                                split_key
                            ].columns

    col_names = exclude_col_names(col_names, cols_exclude)
    return filter_for_col_substring(col_names, name_substring)


def get_split_keys(
    features: dict[str, Any],
    model_exclude: str = "BASELINE_GT",
    return_suffix: bool = True,
) -> Optional[list[str]]:
    """Get split key suffixes from features dictionary.

    Parameters
    ----------
    features : dict
        Features dictionary keyed by split.
    model_exclude : str, optional
        Model name to exclude from search, by default 'BASELINE_GT'.
    return_suffix : bool, optional
        If True, return only the suffix; if False, return full keys.

    Returns
    -------
    list
        List of split key suffixes (e.g., ['_gt', '_raw']).
    """
    for i, split in enumerate(features):
        if i == 0:
            for j, model in enumerate(features[split]):
                if model is not model_exclude:
                    keys = list(features[split][model].keys())
                    # replace split in the keys with ''
                    if return_suffix:
                        keys = [key.replace(f"{split}", "") for key in keys]
                    return keys


# def check_standardization(
#     data_dict: dict, split_key: str, cfg: DictConfig, stdz_threshold: float = 0.001
# ):
#     # If you only standardized the data using the "gt" stats, the "raw" key just contains the same stats
#     # if you wanted to separately standardize the "raw" data, you have that supported here as well
#     if cfg["PREPROCESS"]["use_gt_stats_for_raw"]:
#         standardize_stats = data_dict["metadata"]["preprocess"]["standardize"]["gt"]
#     else:
#         raise NotImplementedError("Not implemented yet")
#         # standardize_stats = data_dict["metadata"]["preprocess"]["standardize"][split_key]
#
#     if np.nanmean(data_dict['imputation_dict']["imputation"]["mean"]) < stdz_threshold:
#         data_standardized = True
#         logger.debug("Input data seems standardized")
#     else:
#         data_standardized = False
#         logger.debug("Input data seems not standardized")
#
#     # TODO! add from cfg option to control this better
#     if data_standardized:
#         logger.debug(
#             "Destandardizing the data, mean = {}, std = {}".format(
#                 standardize_stats["mean"], standardize_stats["std"]
#             )
#         )
#         data_dict["data"] = destandardize_dict(
#             data_dict["data"],
#             mean=standardize_stats["mean"],
#             std=standardize_stats["std"],
#         )
#
#     return data_dict


def data_for_featurization_wrapper(
    artifacts: dict[str, Any], cfg: DictConfig
) -> dict[str, Any]:
    """Prepare data dictionaries for featurization from artifacts.

    Combines imputed data from models with baseline input data.

    Parameters
    ----------
    artifacts : dict
        Dictionary containing model artifacts with imputation results.
    cfg : DictConfig
        Configuration dictionary.

    Returns
    -------
    dict
        Combined dictionary of imputed and baseline data ready for featurization.
    """
    logger.info("Get the data for featurization from artifacts")

    # Imputed data from various imputer models
    imputed_data, model_name, split_names = get_imputed_data_for_featurization(
        artifacts, cfg
    )

    # Baseline data for featurization
    input_data = get_baseline_input_data_for_featurization(
        artifacts, model_name, split_names
    )

    # combine dictionaries for output
    return {**input_data, **imputed_data}


def get_baseline_input_data_for_featurization(
    artifacts: dict[str, Any], model_name: str, split_names: list[str]
) -> dict[str, Any]:
    """Extract baseline input data (GT and raw) formatted for featurization.

    Creates pseudo-imputation dictionaries from original input data to enable
    consistent processing with actual imputation outputs.

    Parameters
    ----------
    artifacts : dict
        Dictionary containing model artifacts.
    model_name : str
        Name of a model to extract metadata from.
    split_names : list
        List of split names (e.g., ['train', 'test']).

    Returns
    -------
    dict
        Dictionary with 'BASELINE_GT' and 'BASELINE_RAW' data structures.
    """
    # Get the data for featurization (for baseline features)
    # You will get 2 new "models" here as it is easier to compute stuff and visualize when input data
    # is treated as the imputed model outputs
    model_artifacts = artifacts[model_name]
    data_names = ["BASELINE_GT", "BASELINE_RAW"]
    split_key_names = ["gt", "raw"]
    input_data = {}
    for i, dataname in enumerate(data_names):
        input_data[dataname] = {}
        for j, split in enumerate(split_names):
            input_data[dataname][split] = {}
            data = get_pseudoimputation_dicts_from_input_data(model_artifacts, split)
            split_key = split_key_names[i]
            data_out = data[split_key]
            input_data[dataname][split][split_key] = {}
            input_data[dataname][split][split_key]["data"] = data_out
            input_data[dataname][split][split_key]["metadata"] = model_artifacts[
                "data_input"
            ][split]["metadata"]

    return input_data


def get_imputed_data_for_featurization(
    artifacts: dict[str, Any], cfg: DictConfig
) -> tuple[dict[str, Any], str, Any]:
    """Extract imputed data from all models for featurization.

    Parameters
    ----------
    artifacts : dict
        Dictionary keyed by model name containing imputation results.
    cfg : DictConfig
        Configuration dictionary.

    Returns
    -------
    tuple
        (imputed_data, model_name, split_names) where imputed_data is
        the nested dictionary of imputation results.
    """
    # Get the dictinaries outputted from the imputation models
    imputed_data = {}
    for i, (model_name, model_artifacts) in enumerate(artifacts.items()):
        imputed_data[model_name] = {}
        split_names = model_artifacts["imputation"].keys()
        for split in split_names:
            imputed_data[model_name][split] = {}
            split_keys = model_artifacts["imputation"][split].keys()
            for split_key in split_keys:
                imputed_data[model_name][split][split_key] = imputed_data_by_split_key(
                    imputation=model_artifacts["imputation"][split][split_key][
                        "imputation_dict"
                    ]["imputation"],
                    metadata=model_artifacts["data_input"][split]["metadata"],
                    split=split,
                    split_key=split_key,
                )

    return imputed_data, model_name, split_names


def imputed_data_by_split_key(
    imputation: dict[str, Any], metadata: dict[str, Any], split: str, split_key: str
) -> dict[str, Any]:
    """Package imputation data with metadata for a specific split key.

    Parameters
    ----------
    imputation : dict
        Imputation results dictionary.
    metadata : dict
        Metadata dictionary for the split.
    split : str
        Split name (e.g., 'train', 'test').
    split_key : str
        Split key (e.g., 'gt', 'raw').

    Returns
    -------
    dict
        Dictionary with 'data' and 'metadata' keys.
    """
    imputed_data = {"data": imputation, "metadata": metadata}
    return imputed_data


def get_pseudoimputation_dicts_from_input_data(
    model_artifacts: dict[str, Any], split: str
) -> dict[str, dict[str, Any]]:
    """Create imputation-like dictionaries from raw input data.

    Converts ground truth and raw data arrays into the same format
    as imputation model outputs for consistent downstream processing.

    Parameters
    ----------
    model_artifacts : dict
        Model artifacts containing data_input with ground truth and raw data.
    split : str
        Split name (e.g., 'train', 'test').

    Returns
    -------
    dict
        Dictionary with 'gt' and 'raw' keys containing pseudo-imputation dicts.
    """

    def convert_np_array_to_dict(X):
        return {"mean": X, "ci_pos": None, "ci_neg": None}

    # Ground truth (denoised), no missing data
    X_gt = model_artifacts["data_input"][split]["data"]["ground_truth"]["gt"]

    # Raw data (noisy), missing data
    X_raw = model_artifacts["data_input"][split]["data"]["data_missing"]["raw"]

    return {
        "gt": convert_np_array_to_dict(X_gt),
        "raw": convert_np_array_to_dict(X_raw),
    }


def get_dict_PLR_per_code(
    data_dict: dict[str, Any], i: int
) -> dict[str, Optional[np.ndarray]]:
    """Extract PLR data for a single subject from a data dictionary.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing numpy arrays with shape (n_subjects, n_timepoints, 1).
    i : int
        Subject index to extract.

    Returns
    -------
    dict
        Dictionary with PLR data arrays for the specified subject.

    Raises
    ------
    ValueError
        If data type is not a number or numpy array.
    """
    dict_PLR = {}
    dict_tmp = data_dict
    for key_in in dict_tmp.keys():
        if dict_tmp[key_in] is not None:
            if isinstance(dict_tmp[key_in], numbers.Number):
                # e.g. you might have n for ensemble models, integer indicating the number of models
                dict_PLR[key_in] = dict_tmp[key_in]
            elif isinstance(dict_tmp[key_in], np.ndarray):
                dict_PLR[key_in] = dict_tmp[key_in][i, :, 0]
            else:
                logger.error(
                    "Unknown type for the PLR data: {}".format(type(dict_tmp[key_in]))
                )
                raise ValueError(
                    "Unknown type for the PLR data: {}".format(type(dict_tmp[key_in]))
                )

        else:
            # when you have undefined CI for example, this is still None
            dict_PLR[key_in] = None

    return dict_PLR


def subjectwise_df_for_featurization(
    data_dict_subj: dict[str, Any],
    metadata_subject: pl.DataFrame,
    subject_code: str,
    cfg: DictConfig,
    i: Optional[int] = None,
) -> pl.DataFrame:
    """Create a subject-specific dataframe for featurization.

    Combines PLR time series data with subject metadata into a single dataframe.

    Parameters
    ----------
    data_dict_subj : dict
        Dictionary containing PLR data arrays for the subject.
    metadata_subject : pl.DataFrame
        Subject metadata as a Polars dataframe.
    subject_code : str
        Unique subject identifier.
    cfg : DictConfig
        Configuration dictionary with PLR_length.
    i : int, optional
        Subject index for extraction, by default None.

    Returns
    -------
    pl.DataFrame
        Combined dataframe with PLR data and metadata.

    Raises
    ------
    AssertionError
        If dataframe length doesn't match expected PLR length.
    """
    # PLR recording from dict to Polars dataframe
    dict_PLR = get_dict_PLR_per_code(data_dict_subj, i)
    df_PLR = subjectdict_to_df(dict_PLR)
    assert df_PLR.shape[0] == cfg["DATA"]["PLR_length"], (
        f"{df_PLR.shape[0]} should be the same as "
        f"PLR length {cfg['DATA']['PLR_length']}"
    )

    # Combine these two dataframes
    df_subject = pandas_concat(df_PLR, metadata_subject, axis=1)
    # df_subject = pl.concat([df_PLR, metadata_subject], how="horizontal")
    assert df_subject.shape[0] == cfg["DATA"]["PLR_length"], (
        f"{df_subject.shape[0]} should be the same as "
        f"PLR length {cfg['DATA']['PLR_length']}"
    )

    # Check and fix the schema of the dataframe
    df_subject = check_and_fix_df_schema(df_subject)

    return df_subject


def drop_useless_metadata_cols(
    metadata_subject: pl.DataFrame, i: int, cfg: DictConfig
) -> pl.DataFrame:
    """Remove unnecessary metadata columns from subject dataframe.

    Parameters
    ----------
    metadata_subject : pl.DataFrame
        Subject metadata dataframe.
    i : int
        Subject index (used for logging only on first subject).
    cfg : DictConfig
        Configuration with DROP_COLS and DROP_COLS_EXTRA lists.

    Returns
    -------
    pl.DataFrame
        Dataframe with specified columns removed.
    """
    # Note! the df_PLR now contains also "useless" columns, but it is easier to keep them
    #  than to figure out a flexible scheme that would handle new added metadata columns
    try:
        metadata_subject = metadata_subject.drop(cfg["PLR_FEATURIZATION"]["DROP_COLS"])
    except Exception as e:
        logger.warning(
            "Failed to drop the 'useless columns' from the metadata: {}".format(e)
        )
        logger.warning(
            "You will now have some extra columns that might confuse you? "
            "But no problems caused for computation of features"
        )

    # Now there are also the non-harmonized naming
    for set_keys in cfg["PLR_FEATURIZATION"]["DROP_COLS_EXTRA"]:
        drop_cols = cfg["PLR_FEATURIZATION"]["DROP_COLS_EXTRA"][set_keys]
        try:
            metadata_subject = metadata_subject.drop(drop_cols)
        except Exception:
            pass

    if i == 0:
        # Display only on the first subject, no need to clutter the logs
        logger.debug(
            'Dropping the "useless columns" from the metadata: {}'.format(
                cfg["PLR_FEATURIZATION"]["DROP_COLS"]
            )
        )
        logger.debug(
            "Remaining columns in the metadata: {}".format(metadata_subject.columns)
        )

    return metadata_subject


def subjectdict_to_df(
    dict_PLR: dict[str, Optional[np.ndarray]],
) -> Optional[pl.DataFrame]:
    """Convert a subject's PLR dictionary to a Polars dataframe.

    Parameters
    ----------
    dict_PLR : dict
        Dictionary with PLR data arrays keyed by data type.

    Returns
    -------
    pl.DataFrame
        Dataframe with one column per data type.
    """
    df = None  # pl.DataFrame
    for data_key in dict_PLR.keys():
        if dict_PLR[data_key] is not None:
            array = dict_PLR[data_key]
            if df is not None:
                df = df.with_columns(pl.lit(array).alias(data_key))
            else:
                df = pl.DataFrame({data_key: array})
        else:
            if df is not None:
                df = df.with_columns(pl.lit(None).alias(data_key))
            else:
                df = pl.DataFrame({data_key: None})
    return df


def get_df_subject_per_code(
    data_df: pl.DataFrame, subject_code: str, cfg: DictConfig
) -> pl.DataFrame:
    """Filter dataframe to get data for a specific subject.

    Parameters
    ----------
    data_df : pl.DataFrame
        Full dataframe containing all subjects.
    subject_code : str
        Subject code to filter for.
    cfg : DictConfig
        Configuration with PLR_length for validation.

    Returns
    -------
    pl.DataFrame
        Filtered dataframe for the specified subject.

    Raises
    ------
    AssertionError
        If filtered dataframe length doesn't match expected PLR length.
    """
    # df_subject = data_df.filter(pl.col("subject_code") == subject_code)
    df_subject = pandas_col_condition_filter(
        df=data_df, col_name="subject_code", col_value=subject_code
    )

    assert df_subject.shape[0] == cfg["DATA"]["PLR_length"], (
        f"df length {df_subject.shape[0]} should be the same as "
        f"PLR length {cfg['DATA']['PLR_length']}"
    )
    return df_subject


def get_metadata_row(df_subject: pl.DataFrame, cfg: DictConfig):
    """Extract scalar metadata from the first row of subject dataframe.

    Parameters
    ----------
    df_subject : pl.DataFrame
        Subject dataframe with repeated metadata across all timepoints.
    cfg : DictConfig
        Configuration dictionary (unused but kept for API consistency).

    Returns
    -------
    pl.DataFrame
        First row containing scalar metadata values.
    """
    # You have 1981 (or n number) of datapoints, so just take the first row to get the "scalar metadata"
    first_row = df_subject[0]
    # This contains "useless cols" but easier to just use all the columns
    return first_row


def get_feature_cfg_hash(subcfg: DictConfig) -> str:
    """Generate a hash string for feature configuration.

    Parameters
    ----------
    subcfg : DictConfig
        Feature configuration dictionary.

    Returns
    -------
    str
        Hash string for the configuration (currently returns placeholder).

    Notes
    -----
    Not fully implemented - returns 'dummyHash'.
    """
    # see e.g. https://stackoverflow.com/questions/5884066/hashing-a-dictionary
    return "dummyHash"


def export_features_pickle_file(features: dict, data_source: str, cfg: DictConfig):
    """Export features dictionary to a pickle file.

    Parameters
    ----------
    features : dict
        Features dictionary with structure:
        - data: dict with 'train' and 'test' pl.DataFrames
        - mlflow_run: MLflow run information
    data_source : str
        Data source name used for filename.
    cfg : DictConfig
        Configuration dictionary.

    Returns
    -------
    str
        Path to the exported pickle file.

    Raises
    ------
    Exception
        If saving fails.
    """
    output_dir = Path(get_artifacts_dir(service_name="features"))
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = get_features_pickle_fname(data_source)
    output_path = output_dir / fname
    try:
        save_results_dict(features, str(output_path), name="features")
    except Exception as e:
        logger.error(f"Failed to save features as a pickle: {e}")
        raise e

    return str(output_path)


def add_feature_metadata_suffix_to_run_name(run_name: str, subcfg: DictConfig) -> str:
    """Append feature metadata suffix to run name.

    Parameters
    ----------
    run_name : str
        Base run name.
    subcfg : DictConfig
        Configuration with 'name' and 'version' keys.

    Returns
    -------
    str
        Run name with appended metadata suffix.
    """
    return "{}_{}_v{}".format(run_name, subcfg["name"], subcfg["version"])


def harmonize_to_imputation_dict(
    data_array: np.ndarray,
    metadata: dict[str, Any],
    split_key_fixed: str,
    cfg: DictConfig,
    destandardize: bool = True,
) -> dict[str, Any]:
    """Convert raw data array to imputation dictionary format.

    Optionally destandardizes the data and packages it in the same
    format as imputation model outputs.

    Parameters
    ----------
    data_array : np.ndarray
        Input data array.
    metadata : dict
        Metadata dictionary with preprocessing stats.
    split_key_fixed : str
        Split key ('gt' or 'raw').
    cfg : DictConfig
        Configuration dictionary.
    destandardize : bool, optional
        Whether to destandardize the data, by default True.

    Returns
    -------
    dict
        Dictionary conforming to imputation output format with
        'imputation_dict' and 'metadata' keys.
    """

    dict_out = {}

    if destandardize:
        mean_before, std_before = np.nanmean(data_array), np.nanstd(data_array)
        if cfg["PREPROCESS"]["standardize"]:
            standardize_stats = metadata["preprocess"]["standardize"]["gt"]
            data_array = destandardize_numpy(
                X=data_array,
                mean=standardize_stats["mean"],
                std=standardize_stats["std"],
            )
        logger.info(
            "Destandardizing the input data | Mean {:.2f} -> {:.2f}, std {:.2f} -> {:.2f}".format(
                mean_before, np.nanmean(data_array), std_before, np.nanstd(data_array)
            )
        )

    # TODO! If you actually start doing anomaly detection, you may want to rethink this a bit
    dict_out[split_key_fixed] = {}
    dict_out[split_key_fixed]["imputation_dict"] = {
        "imputation": {"mean": data_array, "ci_pos": None, "ci_neg": None}
    }
    dict_out[split_key_fixed]["metadata"] = metadata

    return dict_out


def get_original_data_per_split_key(
    model_dict: dict[str, Any], cfg: DictConfig, split_key: str
) -> dict[str, Any]:
    """Extract and format original data for a specific baseline type.

    Parameters
    ----------
    model_dict : dict
        Model dictionary containing data_input.
    cfg : DictConfig
        Configuration dictionary.
    split_key : str
        Baseline type ('BASELINE_DenoisedGT' or 'BASELINE_OutlierRemovedRaw').

    Returns
    -------
    dict
        Dictionary keyed by split with harmonized imputation format.

    Raises
    ------
    ValueError
        If split_key is not recognized.
    """
    results_out = {}
    for split in model_dict["data_input"].keys():
        if split_key == "BASELINE_DenoisedGT":
            data_array = model_dict["data_input"][split]["data"]["ground_truth"]["gt"]
            metadata = model_dict["data_input"][split]["metadata"]
            split_key_fixed = "gt"
        elif split_key == "BASELINE_OutlierRemovedRaw":
            data_array = model_dict["data_input"][split]["data"]["data_missing"]["raw"]
            metadata = model_dict["data_input"][split]["metadata"]
            split_key_fixed = "raw"
        else:
            logger.error(f"Unknown split_key: {split_key}")
            raise ValueError(f"Unknown split_key: {split_key}")

        results_out[split] = harmonize_to_imputation_dict(
            data_array, metadata, split_key_fixed, cfg
        )

    return results_out


def name_imputation_sources_for_featurization(
    sources: list[str], cfg: DictConfig
) -> list[str]:
    """Generate featurization run names from imputation source names.

    Parameters
    ----------
    sources : list
        List of imputation source names.
    cfg : DictConfig
        Configuration dictionary.

    Returns
    -------
    list
        List of featurization run names.
    """
    sources_features = []
    for i, source in enumerate(sources):
        sources_features.append(
            define_featurization_run_name_from_base(base_name=source, cfg=cfg)
        )
    return sources_features


# TODO! TO BE MOVED to "define_sources_for_flow":
def get_original_data_to_results(
    model_dict: dict[str, Any], cfg: DictConfig
) -> dict[str, Any]:
    """Get baseline data (GT and raw) formatted as results dictionaries.

    Parameters
    ----------
    model_dict : dict
        Model dictionary containing data_input.
    cfg : DictConfig
        Configuration dictionary.

    Returns
    -------
    dict
        Dictionary keyed by baseline split keys with data and mlflow_run.
    """
    results = {}
    split_keys = get_baseline_names()

    for split_key in split_keys:
        results[split_key] = {
            "data": get_original_data_per_split_key(model_dict, cfg, split_key),
            "mlflow_run": None,
        }

    return results


def get_imputed_results(model_dict: dict[str, Any], cfg: DictConfig) -> dict[str, Any]:
    """Extract imputation results with metadata from model dictionary.

    Parameters
    ----------
    model_dict : dict
        Model dictionary with 'imputation', 'mlflow_run', and 'data_input'.
    cfg : DictConfig
        Configuration dictionary.

    Returns
    -------
    dict
        Dictionary with 'data', 'mlflow_run', and metadata per split.
    """
    results_out = {
        "data": model_dict["imputation"],
        "mlflow_run": model_dict["mlflow_run"],
    }

    for split in model_dict["data_input"].keys():
        for split_key in results_out["data"][split].keys():
            results_out["data"][split][split_key]["metadata"] = model_dict[
                "data_input"
            ][split]["metadata"]

    return results_out


def create_dict_for_featurization_from_imputation_results_and_original_data(
    imputation_results: dict[str, Any], cfg: DictConfig
) -> dict[str, Any]:
    """Create unified dictionary for featurization from imputation and baseline data.

    Combines imputation model results with original baseline data (GT and raw)
    into a single dictionary structure for featurization.

    Parameters
    ----------
    imputation_results : dict
        Dictionary keyed by model name containing imputation results.
    cfg : DictConfig
        Configuration dictionary.

    Returns
    -------
    dict
        Unified dictionary with all sources ready for featurization.
    """
    for i, (model_name, model_dict) in enumerate(imputation_results.items()):
        if i == 0:
            print(f"Model name: {model_name}")
            # get the original data (i.e. "raw" without the outliers, and 'gt' that is denoised and imputed)
            results = get_original_data_to_results(model_dict, cfg)

        # Now process normally the imputed data
        results[model_name] = get_imputed_results(model_dict, cfg)

    return results


def check_and_fix_df_schema(df_subject: pl.DataFrame) -> pl.DataFrame:
    """Validate dataframe schema and raise error for Object type columns.

    Polars Object type columns cause issues with filtering operations.

    Parameters
    ----------
    df_subject : pl.DataFrame
        Subject dataframe to validate.

    Returns
    -------
    pl.DataFrame
        Validated dataframe (unchanged if no issues).

    Raises
    ------
    ValueError
        If any column has Object dtype.

    See Also
    --------
    https://github.com/pola-rs/polars/issues/18399
    """
    # Check the schema of the dataframe
    # https://github.com/pola-rs/polars/issues/18399
    # Problem with some column being an Object class?
    logger.debug("Checking the column schema of the Polars dataframe:")
    for col in df_subject.columns:
        if df_subject[col].dtype == pl.datatypes.Object:
            # Cannot fix? https://stackoverflow.com/q/76829116/6412152
            logger.error(
                "Column {} is of Object type, cannot be used for featurization".format(
                    col
                )
            )
            logger.error(
                "You are creating your DataFrame incorrectly, check your code! "
                "see e.g. https://stackoverflow.com/a/76720675/6412152"
            )
            raise ValueError(
                "Column {} is of Object type, cannot be used for featurization".format(
                    col
                )
            )

    return df_subject
