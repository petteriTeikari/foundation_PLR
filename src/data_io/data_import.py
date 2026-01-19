import glob

from loguru import logger
import os

from tqdm import tqdm
import numpy as np
import pandas as pd
import polars as pl
from omegaconf import DictConfig

from src.data_io.data_imputation import (
    impute_orig_for_training,
    update_number_of_outliers,
)
from src.data_io.data_outliers import granularize_outlier_labels
from src.data_io.data_utils import (
    define_split_csv_paths,
    import_duckdb_as_dataframes,
    check_data_import,
    export_dataframes_to_duckdb,
    combine_metadata_with_df_splits,
    prepare_dataframe_for_imputation,
    check_time_vector_quality,
    check_for_unique_timepoints,
    fix_for_orphaned_nans,
)
from src.data_io.metadata_from_xlsx import metadata_wrapper
from src.data_io.stratification_utils import stratify_splits
from src.log_helpers.log_naming_uris_and_dirs import get_duckdb_file
from src.utils import get_repo_root


# @task(
#     log_prints=True,
#     name="Import from DuckDB or CSV",
#     description="Import the data from disk (single .db or multiple multi-column CSV files), "
#     "or from some remote storage (S3, Huggung Face, etc.)",
# )
def import_PLR_data_wrapper(cfg: DictConfig, data_dir: str = None):
    # Import data
    df_raw = import_data(cfg, data_dir)

    # import the metadata
    df_metadata = metadata_wrapper(metadata_cfg=cfg["METADATA"])

    # Combine with the time series data
    df_raw, code_stats = combine_metadata_with_df_splits(df_raw, df_metadata)

    # Pick the relevant columns for the task at hand
    df_raw = prepare_dataframe_for_imputation(df_raw, cfg)

    # Check that nothing funky happened to the time vector
    check_for_unique_timepoints(df_raw, cfg, col="time", assert_on_error=True)
    check_for_unique_timepoints(df_raw, cfg, col="time_orig", assert_on_error=False)

    # "pupil_orig" is a bit tricky probably as it is the raw data with outliers _and_ NaNs
    # with the NaN possibly causing some algorithms problems
    # Let's create a novel column that has the missing data imputed
    df_raw = impute_orig_for_training(df_raw, cfg)

    # Automatic split of outliers to easy (blinks) and hard (mostly pupil segmentation algorithm noise)
    # that is closer to true trend
    df_raw = granularize_outlier_labels(df_raw, cfg)

    # Update no_outliers per subject
    df_raw = update_number_of_outliers(df_raw, cfg)

    # Stratify data to train and validation sets
    df_train, df_test = stratify_splits(df_raw, cfg)

    # Export the subset of data to DuckDB
    _ = export_dataframes_to_duckdb(
        df_train, df_test, db_name=cfg["DATA"]["filename_DuckDB"], data_dir=data_dir
    )

    return df_train, df_test


def import_data(cfg: DictConfig, data_dir: str = None) -> pl.DataFrame:
    # if you have access to the raw files, you can import them here (otherwise import the DuckDB)
    dir_in = os.path.join(get_repo_root(), cfg["DATA"]["individual_subjects_path"])
    df_raw = create_csvs_from_individual_subjects(
        individual_subjects_dir=dir_in, data_dir=data_dir, cfg=cfg
    )
    return pl.from_pandas(df_raw)


def create_csvs_from_individual_subjects(
    individual_subjects_dir: str,
    data_dir: str,
    no_of_timepoints: int = 1981,
    cfg: DictConfig = None,
) -> pd.DataFrame:
    if not os.path.exists(individual_subjects_dir):
        logger.error(
            "Individual subjects directory does not exist: {}".format(
                individual_subjects_dir
            )
        )
        raise FileNotFoundError
    else:
        logger.info(
            'Import individual subjects to CSV(s) from "{}"'.format(
                individual_subjects_dir
            )
        )

    files = glob.glob(os.path.join(individual_subjects_dir, "*.csv"))
    logger.info("Found {} files".format(len(files)))  # Found 507 files

    list_of_dfs, outliers, subject_codes = [], [], []
    for i, file in enumerate(tqdm(files, desc="Importing individual subjects")):
        no_outliers, csv_data, subject_code = import_master_csv(
            i=i, csv_path=file, cfg=cfg
        )
        if csv_data.shape[0] == no_of_timepoints:
            list_of_dfs.append(csv_data)
            outliers.append(no_outliers)
            subject_codes.append(subject_code)
        else:
            logger.warning(
                "Subject {} has {} timepoints instead of {}".format(
                    subject_code, csv_data.shape[0], no_of_timepoints
                )
            )

    # Create dataframe from the list of dataframes
    df_raw = convert_list_of_dfs_to_df(list_of_dfs, outliers, subject_codes)
    max_no_outliers = max(outliers)  # 749 (out of 1981)
    logger.info("Max number of outliers per subject: {}".format(max_no_outliers))
    return df_raw


def convert_list_of_dfs_to_df(list_of_dfs, outliers, subject_codes):
    df_out = pd.DataFrame()
    for i, (df, no_outliers, code) in enumerate(
        tqdm(
            zip(list_of_dfs, outliers, subject_codes),
            desc="Converting to single dataframe",
            total=len(list_of_dfs),
        )
    ):
        # add scalars to the dataframe
        df["subject_code"] = code
        df["no_outliers"] = no_outliers
        if i == 0:
            df_out = df
        else:
            df_out = pd.concat([df_out, df])

    return df_out


def export_split_dataframes(
    df_train: pd.DataFrame, df_val: pd.DataFrame, data_dir: str
):
    if not os.path.exists(data_dir):
        logger.info("Create output directory: {}".format(data_dir))
        os.makedirs(data_dir)

    train_path, val_path = define_split_csv_paths(data_dir=data_dir)

    logger.info("Export train split to {}".format(train_path))
    df_train.to_csv(train_path, index=False)

    logger.info("Export val split to {}".format(val_path))
    df_val.to_csv(val_path, index=False)


def define_split(
    subject_codes: list,
    csv_subsets: list,
    indices: list,
    split: str,
    drop_raw_pupil_values: bool = False,
):
    codes = np.array(subject_codes)[indices]
    list_of_df = [csv_subsets[i] for i in indices]
    assert len(codes) == len(list_of_df)
    logger.info('Split "{}" contains {} subjects'.format(split, len(codes)))

    for i, code in enumerate(codes):
        df_with_code = list_of_df[i]
        no_of_timepoints1 = df_with_code.shape[0]
        df_with_code["subject_code"] = code
        if drop_raw_pupil_values:
            df_with_code = df_with_code.dropna()
        no_of_timepoints2 = df_with_code.shape[0]
        if no_of_timepoints1 != no_of_timepoints2:
            logger.info(
                "Subject {} had {} timepoints dropped due to pupil raw NaNs".format(
                    code, no_of_timepoints1 - no_of_timepoints2
                )
            )

        if i == 0:
            df_out = df_with_code
        else:
            df_out = pd.concat([df_out, df_with_code])

    logger.info(
        'Total of {} timepoints ({}x{}) in split "{}"'.format(
            df_out.shape[0], len(codes), no_of_timepoints1, split
        )
    )

    return df_out


def import_master_csv(i: int, csv_path: str, cfg: DictConfig):
    subject_code = os.path.basename(csv_path).split(".")[0].split("_")[0]
    csv_raw = pd.read_csv(csv_path)  # e.g. (1981, 94) # TODO! direct Polars import
    csv_subset = csv_raw[cfg["DATA"]["COLUMNS_TO_KEEP"]]
    if i == 0:
        logger.info(
            "Keeping the following column subset: {}".format(
                cfg["DATA"]["COLUMNS_TO_KEEP"]
            )
        )
        logger.info(
            f"{csv_subset.shape[1]} columns out of total of {csv_raw.shape[1]} columns"
        )

    # Check the time vector
    # Just to be safe, use the same time vector for all subjects ("time"), but store the original time vector
    # to a new column ("time_orig") in case you want to have a look at it later, or they are actually really irregular,
    # and your modeling approach could exploit the multiscale nature of the data?
    time_orig, time_ideal, time_checks = check_time_vector_quality(
        subject_code, csv_subset, cfg
    )
    csv_subset = csv_subset.assign(time=pd.Series(time_ideal))
    csv_subset = csv_subset.assign(time_orig=pd.Series(time_orig))
    if not time_checks["OK"]:
        logger.warning(
            "Time vector quality checks failed for subject {}".format(subject_code)
        )
        for key, value in time_checks.items():
            logger.warning(f"{key}: {value}")

    # rename color columns for something less ambiguous
    csv_subset = csv_subset.rename(columns={"R": "Red", "B": "Blue"})
    csv_subset = csv_subset.rename(
        columns={
            "denoised": "pupil_gt",
            "pupil_raw": "pupil_orig",
            "pupil_toBeImputed": "pupil_raw",
        }
    )

    if i == 0:
        logger.info("Renamed the columns to: {}".format(list(csv_subset.columns)))

    # The color columns have NaNs for outliers?
    # the light was on obviously even during the blinks and there is no ambiguity there
    csv_subset["Red"] = linear_interpolation_of_col(column=csv_subset["Red"])
    csv_subset["Blue"] = linear_interpolation_of_col(column=csv_subset["Blue"])

    # if first or last value is NaN, the interpolation will not work
    csv_subset = fix_for_orphaned_nans(
        subject_code, csv_subset, cfg, cols=("Red", "Blue")
    )

    no_outliers = csv_subset["outlier_labels"].sum()

    return no_outliers, csv_subset, subject_code


def linear_interpolation_of_col(column: pd.Series):
    return column.interpolate()


def import_data_from_duckdb(data_cfg: DictConfig,
                            data_dir: str,
                            use_demo_data: bool = False):
    db_path = get_duckdb_file(data_cfg, use_demo_data)
    if not os.path.exists(db_path):
        logger.error("DuckDB file not found: {}".format(db_path))
        logger.error("Typo with the filename, or you simply do not have this .db file?")
        logger.error(
            'Set cfg["DATA"]["import_from_DuckDB"] = False so you can import from .CSV files'
        )
        raise FileNotFoundError
    else:
        logger.info(
            "Importing the PLR data from DuckDB database as Polars dataframes ({})".format(
                data_cfg["filename_DuckDB"]
            )
        )

    df_train, df_val = import_duckdb_as_dataframes(db_path)
    check_data_import(df_train, df_val)

    return df_train, df_val
