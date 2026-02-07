import polars as pl
from loguru import logger
from omegaconf import DictConfig

from src.data_io.data_utils import (
    export_dataframe_to_duckdb,
)
from src.ensemble.ensemble_utils import (
    get_used_imputation_models_from_mlflow,
)
from src.featurization.feature_utils import data_for_featurization_wrapper
from src.log_helpers.mlflow_artifacts import get_imputation_results_from_mlflow
from src.log_helpers.mlflow_utils import (
    log_imputation_db_to_mlflow,
)
from src.log_helpers.polars_utils import cast_numeric_polars_cols
from src.utils import pandas_concat


# @task(
#     log_prints=True,
#     name="Create Imputation Dataframe",
#     description="Useful for easy visualization and sharing of the imputed data",
# )
def create_imputation_df(
    imputer_artifacts: dict, data_df: pl.DataFrame, cfg: DictConfig
):
    """Create a Polars DataFrame from imputation artifacts for visualization.

    Combines baseline PLR data with imputed data and exports to DuckDB
    for downstream analysis and visualization.

    Parameters
    ----------
    imputer_artifacts : dict
        Dictionary containing imputation results per model, with 'mlflow'
        metadata for each model.
    data_df : pl.DataFrame
        Original PLR data DataFrame with subject codes and time series.
    cfg : DictConfig
        Full Hydra configuration including DATA and MLFLOW settings.

    Returns
    -------
    pl.DataFrame
        Combined DataFrame with imputed values, model identifiers, and
        original data columns.
    """
    # Combine the baseline PLR data with the imputed data
    data_for_features = data_for_featurization_wrapper(
        artifacts=imputer_artifacts, cfg=cfg
    )

    # Create Dataframes per subplot
    df = create_imputation_plot_df(data_for_features, data_df, cfg)

    # Export imputation dataframe as DuckDB, and log as artifact to MLflow as well
    mlflow_cfgs = get_mlflow_cfgs_from_imputation_artifacts(imputer_artifacts, cfg)
    export_imputation_df(df, mlflow_cfgs, cfg)

    return df


def create_imputation_plot_df(
    data_for_features: dict, data_df: pl.DataFrame, cfg: DictConfig
):
    """Create DataFrame containing imputation results per model, split, and subject.

    Iterates through all combinations of models, splits, and split keys to
    construct a unified DataFrame suitable for plotting and analysis.

    Parameters
    ----------
    data_for_features : dict
        Nested dictionary with structure {model: {split: {split_key: data}}},
        containing imputed values and metadata.
    data_df : pl.DataFrame
        Original PLR data DataFrame with subject information.
    cfg : DictConfig
        Configuration containing DATA.PLR_length for validation.

    Returns
    -------
    pl.DataFrame
        Combined DataFrame with all imputation results, reordered to match
        initial column order.

    Raises
    ------
    AssertionError
        If row count is not a multiple of PLR_length.
    """
    logger.info("Creating the imputation dataframe")
    init_cols_saved = False
    df_imputation = pl.DataFrame()
    debug_dfs = {}
    for model in data_for_features.keys():
        for split in data_for_features[model].keys():
            for split_key in data_for_features[model][split].keys():
                logger.debug(f"Creating dataframe for {model} {split} {split_key}")
                df_tmp, size_debug = create_subjects_df(
                    subplot_dict=data_for_features[model][split][split_key],
                    data_df=data_df,
                    cfg=cfg,
                )
                debug_dfs[f"{model}_{split}_{split_key}"] = size_debug
                df_tmp = add_loop_keys(model, split, split_key, df_tmp)
                if not init_cols_saved:
                    init_cols_saved = True
                    colnames_init = df_tmp.columns
                df_imputation = concatenate_imputation_dfs(
                    df_list=[df_imputation, df_tmp]
                )

                assert df_imputation.shape[0] % cfg["DATA"]["PLR_length"] == 0, (
                    "The number of rows in the output DataFrame "
                    "is not correct, shoud be a multiple of the "
                    "length of the time vector (PLR_length={})"
                ).format(cfg["DATA"]["PLR_length"])

    # reorder the columns based on the first subject
    df_imputation = df_imputation.select(colnames_init)
    no_of_PLRs = df_imputation.shape[0] / cfg["DATA"]["PLR_length"]
    assert df_imputation.shape[0] % cfg["DATA"]["PLR_length"] == 0, (
        "The number of rows in the output DataFrame "
        "is not correct, shoud be a multiple of the "
        "length of the time vector (PLR_length={})"
    ).format(cfg["DATA"]["PLR_length"])

    logger.info(
        "Imputation dataframe created, shape: {} ({} of PLRs in total, from {} options)".format(
            df_imputation.shape, int(no_of_PLRs), len(debug_dfs)
        )
    )

    return df_imputation


def concatenate_imputation_dfs(df_list: list):
    """Concatenate imputation DataFrames with proper type casting.

    Processes and combines DataFrames for imputation results, ensuring
    consistent column types and naming conventions.

    Parameters
    ----------
    df_list : list
        List of two Polars DataFrames [existing_df, new_df] to concatenate.
        The first may be empty, the second is processed before concatenation.

    Returns
    -------
    pl.DataFrame
        Vertically concatenated DataFrame with consistent column types.

    Raises
    ------
    Exception
        If concatenation fails due to schema mismatches.
    """
    # Now operating only on the df to be added, and we don't need to touch the 1st as
    # it's empty in the beginning, and all the 2nd df's will be added through these operations
    df_list[1] = cast_numeric_polars_cols(df=df_list[1], cast_to="Float64")
    df_list[1] = rename_ci_cols(df=df_list[1])
    df_list[1] = df_list[1].select(sorted(df_list[1].columns))

    try:
        df_imputation = pl.concat(df_list, how="vertical")
    except Exception as e:
        logger.error("Error in concatenating the imputation dataframes: {}".format(e))
        logger.error("Trying to convert the dataframes to Float32")
        raise e

    return df_imputation


def create_subjects_df(subplot_dict: dict, data_df: pl.DataFrame, cfg: DictConfig):
    """Create DataFrame for all subjects from imputation subplot data.

    Combines time series data with subject metadata for all subjects in
    the subplot dictionary.

    Parameters
    ----------
    subplot_dict : dict
        Dictionary containing 'data' with imputation arrays (mean, CI, etc.)
        and 'metadata' with subject codes.
    data_df : pl.DataFrame
        Original PLR data DataFrame for looking up subject information.
    cfg : DictConfig
        Configuration (unused but kept for interface consistency).

    Returns
    -------
    tuple
        (df_out, size_debug) where df_out is the combined DataFrame and
        size_debug is a dict with 'no_subjects' and 'no_timepoints'.

    Raises
    ------
    AssertionError
        If timepoint counts don't match expected values.
    """
    # see e.g. compute_features_from_dict() and combine these eventually
    no_subjects, no_timepoints, no_features = subplot_dict["data"]["mean"].shape
    df_out = pl.DataFrame()
    for idx in range(no_subjects):
        df_subject = pl.DataFrame()
        df_subject = add_ts_cols(subplot_dict, df_subject, idx, no_timepoints)
        assert (
            df_subject.shape[0] == no_timepoints
        ), f"df_subject: {df_subject.shape[0]} time points for {idx}th subject "
        subject_code = subplot_dict["metadata"]["metadata_df"]["subject_code"][idx]
        data_subject = get_subject_datadf(data_df, subject_code, no_timepoints)
        df_subject = pl.concat([df_subject, data_subject], how="horizontal")
        assert (
            df_subject.shape[0] == no_timepoints
        ), f"{df_subject.shape[0]} time points for {idx} subject "
        df_out = pandas_concat(df1=df_out, df2=df_subject)
        assert (
            df_out.shape[0] == (idx + 1) * no_timepoints
        ), f"df_out: {df_out.shape[0]} time points for {idx} subject "
        # The column lengths in the DataFrame are not equal.

    assert (
        df_out.shape[0] == no_subjects * no_timepoints
    ), "The number of rows in the output DataFrame is not correct"

    return df_out, {"no_subjects": no_subjects, "no_timepoints": no_timepoints}


def get_subject_datadf(data_df: pl.DataFrame, subject_code: str, no_timepoints: int):
    """Extract time series data for a specific subject from the DataFrame.

    Parameters
    ----------
    data_df : pl.DataFrame
        Full PLR data DataFrame containing all subjects.
    subject_code : str
        Unique identifier for the subject to extract.
    no_timepoints : int
        Expected number of timepoints for validation.

    Returns
    -------
    pl.DataFrame
        DataFrame containing only the specified subject's time series.

    Raises
    ------
    AssertionError
        If the number of rows doesn't match expected timepoints.
    """
    # Pick the time series from Polars DataFrame matching the subject code
    data_subject = data_df.filter(data_df["subject_code"] == subject_code)
    assert (
        data_subject.shape[0] == no_timepoints
    ), f"data_subject: {data_subject.shape[0]} time points for {subject_code} subject "
    # Polars->Pandas->Polars to maybe catch the "The column lengths in the DataFrame are not equal."
    return pl.from_pandas(data_subject.to_pandas())


def add_ts_cols(
    subplot_dict: dict,
    df_out: pl.DataFrame,
    idx: int,
    no_timepoints: int,
    add_as_list: str = True,
):
    """Add time series columns from subplot data to a DataFrame.

    Extracts and adds imputation data (mean, CI bounds, etc.) for a
    specific subject index to the output DataFrame.

    Parameters
    ----------
    subplot_dict : dict
        Dictionary containing 'data' with arrays keyed by time series type.
    df_out : pl.DataFrame
        Output DataFrame to add columns to.
    idx : int
        Subject index to extract data for.
    no_timepoints : int
        Expected number of timepoints for validation.
    add_as_list : str, optional
        If True, convert arrays to lists before adding (helps with Polars
        compatibility issues). Default is True.

    Returns
    -------
    pl.DataFrame
        DataFrame with added time series columns.

    Raises
    ------
    AssertionError
        If the number of timepoints doesn't match expected value.
    """
    for ts_key in subplot_dict["data"].keys():
        if subplot_dict["data"][ts_key] is not None:
            # add the array to Polars DataFrame
            array_tmp = subplot_dict["data"][ts_key][idx, :, :].flatten()
            assert (
                len(array_tmp) == no_timepoints
            ), f"array tmp: {len(array_tmp)} time points for {idx} subject "
            if add_as_list:
                # some weird Polars glitch after Numpy 1.25.2 downgrade, getting
                # "The column lengths in the DataFrame are not equal." Maybe list is better?
                list_tmp = list(array_tmp)
                df_out = df_out.with_columns(pl.Series(name=ts_key, values=list_tmp))
            else:
                df_out = df_out.with_columns(pl.lit(array_tmp).alias(ts_key))
        else:
            df_out = df_out.with_columns(pl.lit(None).alias(ts_key))
    assert (
        df_out.shape[0] == no_timepoints
    ), f"df_out: {df_out.shape[0]} time points for {idx} subject "

    return df_out


def add_loop_keys(model, split, split_key, df_tmp):
    """Add model, split, and split_key identifiers as columns to DataFrame.

    Parameters
    ----------
    model : str
        Name of the imputation model.
    split : str
        Data split identifier (e.g., 'train', 'test').
    split_key : str
        Additional split key identifier.
    df_tmp : pl.DataFrame
        DataFrame to add identifier columns to.

    Returns
    -------
    pl.DataFrame
        DataFrame with added 'model', 'split', and 'split_key' columns.
    """
    df_tmp = df_tmp.with_columns(pl.lit(model).alias("model"))
    df_tmp = df_tmp.with_columns(pl.lit(split).alias("split"))
    df_tmp = df_tmp.with_columns(pl.lit(split_key).alias("split_key"))

    return df_tmp


def rename_ci_cols(df):
    """Rename confidence interval columns to standard names.

    Normalizes column names for imputation confidence intervals to
    consistent 'ci_pos' and 'ci_neg' names.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with potentially inconsistent CI column names.

    Returns
    -------
    pl.DataFrame
        DataFrame with standardized CI column names.

    Notes
    -----
    This is a temporary fix for column naming inconsistencies that should
    be harmonized upstream.
    """
    # TODO! Hacky way to handle columns, harmonize these so you don't get mixed? or how is this actually happening?
    for col in df.columns:
        if "imputation_ci_pos" in col:
            df = df.rename({"imputation_ci_pos": "ci_pos"})
        elif "imputation_ci_neg" in col:
            df = df.rename({"imputation_ci_neg": "ci_neg"})
    return df


def get_mlflow_cfgs_from_imputation_artifacts(imputer_artifacts: dict, cfg: DictConfig):
    """Extract MLflow configurations from imputation artifacts.

    Parameters
    ----------
    imputer_artifacts : dict
        Dictionary of imputation results keyed by model name, each containing
        'mlflow' metadata.
    cfg : DictConfig
        Configuration (unused but kept for interface consistency).

    Returns
    -------
    dict
        Dictionary mapping model names to their MLflow configurations.
    """
    mlflow_cfgs = {}
    for model in imputer_artifacts.keys():
        mlflow_cfgs[model] = imputer_artifacts[model]["mlflow"]
    return mlflow_cfgs


def export_imputation_df(df: pl.DataFrame, mlflow_cfgs: dict, cfg: DictConfig):
    """Export imputation DataFrame to DuckDB and log to MLflow.

    Creates a DuckDB database for each model's imputation results and
    logs it as an artifact to the corresponding MLflow run.

    Parameters
    ----------
    df : pl.DataFrame
        Combined imputation DataFrame with model column for filtering.
    mlflow_cfgs : dict
        Dictionary mapping model names to MLflow configurations.
    cfg : DictConfig
        Configuration for export settings.
    """
    logger.info("Exporting the imputation dataframe (per model output) to DuckDB")
    for model in mlflow_cfgs.keys():
        db_name = f"imputation_{model}.db"
        # Pick the samples from the given model
        df_subset = df.filter(df["model"] == model)
        # Save as DuckDB database
        db_path = export_dataframe_to_duckdb(
            df=df_subset, db_name=db_name, cfg=cfg, name="imputation"
        )
        # Log as artifact to MLflow
        log_imputation_db_to_mlflow(
            db_path=db_path, mlflow_cfg=mlflow_cfgs[model], model=model, cfg=cfg
        )


# @task(
#     log_prints=True,
#     name="Get Imputation Results from MLflow for Features",
#     description="Get the imputation results from MLflow for the features",
# )
def get_imputation_results_from_mlflow_for_features(
    experiment_name: str, cfg: DictConfig
):
    """Retrieve imputation results from MLflow for feature computation.

    Fetches the best hyperparameter configurations and their corresponding
    imputation results from MLflow for use in downstream featurization.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name to search for imputation runs.
    cfg : DictConfig
        Configuration for MLflow and model settings.

    Returns
    -------
    dict
        Dictionary mapping model names to their imputation results from MLflow.
    """
    # Gets the best hyperparam
    best_unique_models = get_used_imputation_models_from_mlflow(
        experiment_name, cfg, exclude_ensemble=False
    )

    results_per_model = {}
    for i, model in enumerate(best_unique_models.keys()):
        logger.info(
            f"Getting the results of the model: {model} (#{i + 1}/{len(best_unique_models.keys())})"
        )
        results_per_model[model] = get_imputation_results_from_mlflow(
            mlflow_run=best_unique_models[model], model_name=model, cfg=cfg
        )

    return results_per_model
