import shutil
import random
from loguru import logger
import os
from omegaconf import DictConfig
import numpy as np

from src.log_helpers.log_naming_uris_and_dirs import get_duckdb_file, get_demo_string_to_add
from src.utils import get_artifacts_dir, pandas_concat
from tqdm import tqdm
from scipy import interpolate


import duckdb
import pandas as pd
import polars as pl


def convert_sec_to_date(df, time_col: str = "ds", seconds_offset: float = 1):
    df[time_col] = pd.to_datetime(
        1000 * (df[time_col] + seconds_offset), unit="ms", errors="coerce"
    )
    # date = pd.date_range('2004-01-01', '2018-01-01', freq="AS")
    return df


def convert_sec_to_millisec(df, time_col: str = "ds", seconds_offset: float = 1):
    df[time_col] += seconds_offset
    df[time_col] *= 1000
    return df


def split_df_to_samples(
    df: pd.DataFrame, split: str = "train", subject_col_name: str = "unique_id"
) -> dict:
    subject_codes = df[subject_col_name].unique()
    no_of_unique_subjects = len(subject_codes)

    logger.info(
        "Splitting Pandas Dataframe data into {} single-subject Dataframes".format(
            no_of_unique_subjects
        )
    )

    dict_of_dfs = {}
    # TOOPTIMIZE: There is probably some more efficient way to do this
    for i, code in enumerate(
        tqdm(subject_codes, desc='Split "{}", df to dict of dfs'.format(split))
    ):
        df_sample = df.loc[df[subject_col_name] == code]
        df_sample = df_sample.drop(subject_col_name, axis=1)
        dict_of_dfs[code] = df_sample

    assert (
        len(dict_of_dfs) == no_of_unique_subjects
    ), "Number of subjects does not match!"

    return dict_of_dfs


def get_subset_of_data(df_subset: pd.DataFrame, t0: float = 18.0, t1: float = 19.05):
    df_subset.drop(df_subset[df_subset.ds > t1].index, inplace=True)
    df_subset.drop(df_subset[df_subset.ds < t0].index, inplace=True)
    df_subset.reset_index(drop=True, inplace=True)

    # Manually get 3 first subjects
    df_subset = df_subset.iloc[:96]

    return df_subset


def define_split_csv_paths(data_dir: str, suffix: str = ""):
    train_path = os.path.join(data_dir, "train_PLR{}.csv".format(suffix))
    val_path = os.path.join(data_dir, "val_PLR{}.csv".format(suffix))

    return train_path, val_path


def import_nonnan_data_from_csv(data_dir: str, suffix: str = "_nonNan"):
    logger.info("Import PLR data with the NaN/outlier rows removed")
    train_path, val_path = define_split_csv_paths(data_dir=data_dir, suffix="_nonNan")
    logger.info("TRAIN split path = {}".format(train_path))
    df_train = pl.read_csv(train_path)
    logger.info("VAL split path = {}".format(val_path))
    df_val = pl.read_csv(val_path)

    return df_train, df_val


def import_PLR_data_from_CSV(data_dir: str):
    logger.info('Import data from CSVs in "{}"'.format(data_dir))
    train_path, val_path = define_split_csv_paths(data_dir=data_dir)

    logger.info("Import train split from {}".format(train_path))
    df_train = pl.read_csv(train_path)

    logger.info("Import val split from {}".format(val_path))
    df_val = pl.read_csv(val_path)

    return df_train, df_val


def export_dataframe_to_duckdb(
    df: pl.DataFrame,
    db_name: str,
    cfg: DictConfig,
    name: str = None,
    service_name: str = "duckdb",
    debug_DuckDBWrite: bool = True,
    copy_orig_db: bool = False,
):
    dir_out = get_artifacts_dir(service_name=service_name)
    os.makedirs(dir_out, exist_ok=True)
    db_path_out = os.path.join(dir_out, db_name)
    if os.path.exists(db_path_out):
        os.remove(db_path_out)

    if copy_orig_db:
        # TODO! debug code, eliminate when you figure out what to do with the anomaly detection
        # the one that flow_import_data() uses
        db_path_in = get_duckdb_file(data_cfg=cfg["DATA"])

        shutil.copyfile(db_path_in, db_path_out)
        logger.info("Copying the DuckDB Database, from {}".format(db_path_in))
        logger.info("to: {}".format(db_path_out))
    else:
        logger.info("Writing dataframe to DuckDB Database: {}".format(db_path_out))
        logger.info(
            "Shape of the dataframe written to disk as DuckDB {}".format(df.shape)
        )
        if os.path.exists(db_path_out):
            logger.warning("DuckDB Database already exists, removing the old one")
            os.remove(db_path_out)
        with duckdb.connect(database=db_path_out, read_only=False) as con:
            con.execute("""
                        CREATE TABLE IF NOT EXISTS 'train' AS SELECT * FROM df;
                    """)

    if debug_DuckDBWrite:
        logger.debug("Reading back from DuckDB (to test that stuff was written)")
        if copy_orig_db:
            df_train = load_from_duckdb_as_dataframe(
                db_path=db_path_out, cfg=cfg, split="train"
            )
            df_val = load_from_duckdb_as_dataframe(
                db_path=db_path_out, cfg=cfg, split="val"
            )
            df_back = pl.concat([df_train, df_val])
        else:
            df_back = load_from_duckdb_as_dataframe(db_path=db_path_out, cfg=cfg)
        # TODO! figure out why this happens? :o
        #  Saved dataframe shape (1004367, 15) does not match the shape read back: (1004367, 13)
        #  does not write the "new cols" for some reason, "_imputed" suffix ones
        assert df.shape == df_back.shape, (
            f"Saved dataframe shape {df.shape} does not match the "
            f"shape read back: {df_back.shape} (Samples (time points), Features)"
        )
        logger.debug("Read successful!")

    return db_path_out


def load_both_splits_from_duckdb(db_path: str, cfg: DictConfig):
    df_train = load_from_duckdb_as_dataframe(db_path=db_path, cfg=cfg, split="train")
    df_val = load_from_duckdb_as_dataframe(db_path=db_path, cfg=cfg, split="val")
    return pl.concat([df_train, df_val])


def load_from_duckdb_as_dataframe(db_path: str, cfg: DictConfig, split: str = "train"):
    try:
        with duckdb.connect(database=db_path, read_only=False) as con:
            df_load = con.query(f"SELECT * FROM {split}").pl()
    except Exception as e:
        logger.error("Error in reading DuckDB: {}".format(e))
        raise e
    return df_load


def export_dataframes_to_duckdb(
    df_train,
    df_test,
    db_name: str = "SERI_PLR_GLAUCOMA.db",
    data_dir: str = None,
    debug_DuckDBWrite: bool = True,
):
    # https://duckdb.org/docs/api/python/overview.html#persistent-storage
    db_path = os.path.join(data_dir, db_name)
    logger.info("Writing dataframes to DuckDB Database: {}".format(db_path))

    if os.path.exists(db_path):
        logger.warning("DuckDB Database already exists, removing the old one")
        os.remove(db_path)

    with duckdb.connect(database=db_path, read_only=False) as con:
        # TOOPTIMIZE! Blue and Red are now written as double (could be just uint8)
        con.execute("""
                    CREATE TABLE IF NOT EXISTS 'train' AS SELECT * FROM df_train;
                """)
        con.execute("""
                                CREATE TABLE IF NOT EXISTS 'test' AS SELECT * FROM df_test;
                            """)
    logger.info("Write finished".format())

    if debug_DuckDBWrite:
        logger.info("Reading back from DuckDB")
        logger.info("TRAIN split")
        with duckdb.connect(database=db_path, read_only=False) as con:
            con.query("SELECT * FROM train").show()
        logger.info("TEST split")
        with duckdb.connect(database=db_path, read_only=False) as con:
            con.query("SELECT * FROM test").show()
        logger.info("Read successful!")

    return db_path


def import_duckdb_as_dataframes(db_path: str):

    logger.info("Reading data from DuckDB Database: {}".format(db_path))
    try:
        with duckdb.connect(database=db_path, read_only=False) as con:
            train = con.query("SELECT * FROM train")
            df_train = train.pl()
            test = con.query("SELECT * FROM test")
            df_test = test.pl()
    except Exception as e:
        logger.error("Error in reading DuckDB: {}".format(e))
        raise e
    logger.info("Done with the read from DuckDb to Polars dataframes".format())

    return df_train, df_test


def check_data_import(df_train, df_val, display_outliers: bool = True):
    unique_subjects_train = get_unique_polars_rows(
        df_train,
        unique_col="subject_code",
        value_col="pupil_raw",
        split="train",
        df_string="PLR",
    )
    unique_subjects_val = get_unique_polars_rows(
        df_val,
        unique_col="subject_code",
        value_col="pupil_raw",
        split="val",
        df_string="PLR",
    )

    # Check for leakage, you cannot have the same subject both in train and validation
    matching_codes = unique_subjects_train.join(
        unique_subjects_val, on="subject_code", how="inner"
    )
    if len(matching_codes) > 0:
        logger.error(
            "Data leakage detected! The same subject code is found in both train and validation splits. Redefine the splits!"
        )
        raise ValueError(
            "Data leakage detected! The same subject code is found in both train and validation"
        )

    if display_outliers:
        # When you are importing from CSVs, this can be confusing as the outliers are found from both "pupil_faw"
        # and "outlier_labels" so as a quick fix, skip the confusing display, or define the outlier percentage
        # correctly for CSV import, see e.g. prepare_dataframe_for_imputation()
        # TODO! Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future.
        #  Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
        #  data_utils.py:266: DeprecationWarning:
        no_outliers = int(df_train.select(pl.col("pupil_raw").null_count()).to_numpy())
        train_outlier_percentage = no_outliers / df_train.shape[0] * 100

        no_outliers = int(df_val.select(pl.col("pupil_raw").null_count()).to_numpy())
        val_outlier_percentage = no_outliers / df_val.shape[0] * 100

        # TODO! % of outliers
        logger.info(
            "Train split shape: {}, unique subjects = {} ({:.2f}% missing)".format(
                df_train.shape, len(unique_subjects_train), train_outlier_percentage
            )
        )
        logger.info(
            "Val split shape: {}, unique subjects = {} ({:.2f}% missing)".format(
                df_val.shape, len(unique_subjects_val), val_outlier_percentage
            )
        )


def prepare_dataframe_for_imputation(df, cfg):
    # Fix light stimuli vector
    df = fix_light_stimuli_vector(df, cfg)

    # Set pupil_raw outliers to null
    df = set_outliers_to_null(df, cfg)

    return df


def fix_light_stimuli_vector(df, cfg, drop_colors: bool = False):
    logger.info("Combining Red and Blue into a single light stimuli column")
    df = df.with_columns(light_stimuli=pl.Series(df["Blue"] + df["Red"]))
    df = interpolate_missing_light_stimuli_values(df, cfg, col_name="light_stimuli")
    # remove the original columns
    if drop_colors:
        logger.debug("Dropping the original Red and Blue columns")
        df = df.drop(["Blue", "Red"])
    else:
        logger.debug("Keeping the original Red and Blue columns")
        df = interpolate_missing_light_stimuli_values(df, cfg, col_name="Blue")
        df = interpolate_missing_light_stimuli_values(df, cfg, col_name="Red")

    return df


def interpolate_missing_light_stimuli_values(df, cfg, col_name: str = "light_stimuli"):
    no_nulls = int(df.select(pl.col(col_name).null_count()).to_numpy())
    if no_nulls > 0:
        logger.debug(
            "Interpolating missing {} values (n = {})".format(col_name, no_nulls)
        )
        df = df.with_columns(
            light_stimuli=pl.when(pl.col(col_name).is_null())
            .then(pl.col(col_name).interpolate())
            .otherwise(pl.col(col_name))
        )

    return df


def set_outliers_to_null(df, cfg):
    # Raw should have this correct already, whereas "_orig" not so much necessarily

    # Set 'null' values to the outliers in the pupil_raw (input data
    df = df.with_columns(
        pupil_raw=pl.when(pl.col("outlier_labels") == 1)
        .then(pl.lit(None))
        .otherwise(pl.col("pupil_raw"))
    )

    # Drop again unnecessary column as this is encoded to the pupil_raw column
    df = df.drop(["outlier_labels"])

    # Count the amount of outliers (null in Polars)
    no_outliers = int(df.select(pl.col("pupil_raw").null_count()).to_numpy())
    outlier_percentage = no_outliers / df.shape[0] * 100
    logger.info(
        "Number of outliers set to null = {} ({:.2f}% of all samples)".format(
            no_outliers, outlier_percentage
        )
    )

    return df


def set_missing_in_data(
    df, X, missingness_cfg, col_name: str = "pupil_raw", split: str = "train"
):
    raw = df.select(col_name)
    raw = raw.to_numpy().reshape(X.shape[0], X.shape[1], X.shape[2])
    X[np.isnan(raw)] = np.nan

    mask_no_missing = np.isnan(raw).sum()
    out_no_missing = np.isnan(X).sum()
    assert mask_no_missing == out_no_missing, (
        "Number of missing values "
        "in the mask and the output data do not match! (for some weird reason)"
    )

    logger.info(
        "Percentage of missing values in the data ({}) = {:.2f}%".format(
            split, mask_no_missing / raw.size * 100
        )
    )

    return X


def combine_metadata_with_df_splits(df_raw, df_metadata):
    logger.info("Combining metadata with the data splits")
    df, code_stats = combine_metadata_with_df(
        df=df_raw, df_metadata=df_metadata, split="all data"
    )
    return df, code_stats


def combine_metadata_with_df(df, df_metadata, split):
    unique_PLR = get_unique_polars_rows(
        df, unique_col="subject_code", value_col="time", split=split, df_string="PLR"
    )

    unique_metadata = get_unique_polars_rows(
        df_metadata,
        unique_col="subject_code",
        value_col="class_label",
        split=split,
        df_string="metadata",
    )

    code_stats = get_missing_labels(unique_PLR, unique_metadata, split)
    # Note! When training the imputation we don't need the class labels (in metadata), but when analyzing the
    # downstream effects of the imputation we do need the class labels

    # Loop through the values in metadata (maybe there is a more efficient way to do this)
    df = add_labels_for_matching_codes(
        df, df_metadata, matching_codes=code_stats["matching"]
    )

    return df, code_stats


def get_missing_labels(unique_PLR, unique_metadata, split):
    def check_code_segment(df, string):
        list_of_codes = []
        for row in df.rows(named=True):
            list_of_codes.append(row["subject_code"])
        logger.info(f"{string} ({split} split), number of labels: {len(list_of_codes)}")
        for code in list_of_codes:
            logger.debug(code)
        return sorted(list_of_codes)

    codes_PLR = unique_PLR.select(pl.col("subject_code"))
    # codes_metadata = unique_metadata.select(pl.col("subject_code"))

    # That you could actually do some classification
    matching_codes = unique_PLR.join(unique_metadata, on="subject_code", how="inner")
    # What codes you have in metadata XLSX but are not found as PLR recordings:
    extra_metadata = unique_metadata.join(matching_codes, on="subject_code", how="anti")
    # What codes should have class_labels added to the XLSX
    missing_from_PLR = unique_PLR.join(matching_codes, on="subject_code", how="anti")

    assert len(matching_codes) + len(extra_metadata) == len(unique_metadata)
    assert len(matching_codes) + len(missing_from_PLR) == len(codes_PLR)

    codes = {}
    codes["matching"] = check_code_segment(df=matching_codes, string="Matching codes")
    codes["extra_metadata"] = check_code_segment(
        df=extra_metadata, string="Extra metadata"
    )
    codes["missing_from_PLR"] = check_code_segment(
        df=missing_from_PLR, string="Missing from PLR"
    )

    return codes


def add_labels_for_matching_codes(df, df_metadata, matching_codes):
    def add_empty_cols(df, df_metadata):
        for col in df_metadata.columns:
            if col not in df.columns:
                logger.debug(f"Adding empty column: {col}")
                df = df.with_columns(pl.lit(None).alias(col))
        return df

    df = add_empty_cols(df, df_metadata)

    for i, code in enumerate(matching_codes):
        logger.debug(f"Adding metadata for code {code}")
        df = add_label_per_code(code, df, df_metadata)

    check_post_metadata_add(df)

    return df


def add_label_per_code(code, df, df_metadata, code_col="subject_code", length_PLR=1981):
    df_per_code = df.filter(pl.col(code_col) == code)
    # Obviosuly this will break if you start doing some custom recordings
    assert (
        len(df_per_code) == length_PLR
    ), f"Length of the PLR data for {code} is not {length_PLR}"

    # Loop through all the metadata columns that you have and them based on the subject code
    for col in df_metadata.columns:
        # Don't add the subject_code column to the subject column (replace it basically
        if col is not code_col:
            value = df_metadata.filter(pl.col(code_col) == code)[col].to_numpy()[0]

            df = df.with_columns(
                (
                    pl.when(pl.col(code_col) == code)
                    .then(pl.lit(value))
                    .otherwise(pl.col(col))
                ).alias(col)
            )

    return df


def get_unique_labels(
    df, unique_col: str = "class_label", value_col: str = "time"
) -> list:
    unique_labels = get_unique_polars_rows(
        df, unique_col=unique_col, value_col=value_col
    )
    unique_labels = unique_labels.filter(
        pl.col(unique_col).is_not_null()
    )  # drop null rows
    return list(unique_labels[:, unique_col].to_numpy())


def pick_per_label(df, label, cfg):
    # Polars alternative due to a weird Polars issue
    df_pd = df.to_pandas()
    df_label = df_pd[df_pd["class_label"] == label]
    df = pl.DataFrame(df_label)
    check_for_data_lengths(df, cfg)
    return pl.DataFrame(df_label)


def get_outlier_count_per_code(unique_value, df_pd):

    no_outliers = np.zeros_like(unique_value)
    for i in range(len(unique_value)):
        unique_code = unique_value[i]
        df_code = df_pd[df_pd["subject_code"] == unique_code]
        no_outliers[i] = df_code['no_outliers'].iloc[0]

    sorted_indices = no_outliers.argsort()
    outliers_count_sorted = no_outliers[sorted_indices]
    codes_sorted = unique_value[sorted_indices]

    median_outlier = np.median(no_outliers)
    over_median = outliers_count_sorted > median_outlier
    counts_left = outliers_count_sorted[over_median]
    codes_left = codes_sorted[over_median]

    return counts_left, codes_left


def pick_random_subjects_with_outlier_no_cutoff(unique_value, df_pd, n):

    outlier_counts, codes_left = get_outlier_count_per_code(unique_value, df_pd)
    random_idx = random.sample(range(0, len(codes_left)-1), n)
    random_codes = codes_left[random_idx]

    return random_codes


def pick_n_subjects_per_label_pandas(
    df: pl.DataFrame,
    n: int,
    PLR_length=1981,
    col_select: str = "subject_code",
    pick_random: bool = False,
) -> pl.DataFrame:
    # Pandas alternative due to a weird Polars issue
    df_pd = df.to_pandas()
    unique_value = df_pd[col_select].unique()
    if pick_random:
        first_n_subjects = unique_value[:n]
    else:
        first_n_subjects = pick_random_subjects_with_outlier_no_cutoff(unique_value, df_pd, n)

    df_out = pd.DataFrame()
    for i, code in enumerate(first_n_subjects):
        df_code = df_pd[df_pd[col_select] == code]
        assert df_code.shape[0] == PLR_length
        df_out = pd.concat([df_out, df_code])

    assert (
        df_out.shape[0] == n * PLR_length
    )  # TODO! check with other n values than 8 for debug
    return pl.DataFrame(df_out)


def pick_n_subjects_per_label(df: pl.DataFrame, label: str, n: int, PLR_length=1981):
    df_label = df.filter(pl.col("class_label") == label)
    unique_subjects = get_list_of_unique_subjects(df_label)
    first_n_subjects = unique_subjects[:n]
    df_out = pl.DataFrame()
    for i, code in enumerate(first_n_subjects):
        df_code = df_label.filter(pl.col("subject_code") == code)
        assert (
            len(df_code) == PLR_length
        ), "Length of the PLR data for {} is not {}".format(code, PLR_length)
        df_out = pl.concat([df_out, df_code])
        assert df_out.shape[0] == (i + 1) * PLR_length, (
            "Seems like subject was not added with pl.concat?\n"
            "{}, {}, {} (i, df_code, df_out)"
        ).format(i, df_code.shape, df_out.shape)
        # due to Polars issue?
        # note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
        # thread '<unnamed>' panicked at crates/polars-core/src/fmt.rs:567:13:
        # The column lengths in the DataFrame are not equal.

    df_n_subjects = int(df_out.shape[0] / PLR_length)
    assert df_out.shape[0] == n * PLR_length, (
        f"Number of rows ({df_n_subjects}) in the output dataframe "
        f"is not equal to the number of subjects ({n}) requested"
    )
    unique_subjects_out = get_list_of_unique_subjects(df_out)
    assert (
        len(unique_subjects_out) == n
    ), "Number of subjects in the output dataframe is not equal to the number of subjects requested"

    return df_out


def get_list_of_unique_subjects(df_label, unique_col: str = "subject_code"):
    unique_rows = get_unique_polars_rows(df_label, unique_col=unique_col)
    return list(unique_rows[:, unique_col].to_numpy())


def get_unique_polars_rows(
    df,
    unique_col: str = "subject_code",
    value_col: str = "time",
    split: str = None,
    df_string: str = None,
    pandas_fix: bool = True,
):
    # Get one value per subject code to check if the metadata is there
    try:
        unique_values = list(df[unique_col].unique())
        if pandas_fix:
            df = df.to_pandas()
            df = df.drop_duplicates(subset=[unique_col])
            df = pl.DataFrame(df)
        else:
            # polars.exceptions.ShapeError: series used as keys should have the same length as the DataFrame
            df = df.select(
                pl.all()
                .top_k_by(value_col, k=1)
                .over(unique_col, mapping_strategy="explode")
            )
    except KeyError as e:
        logger.error(f"Unique values = {unique_values}")
        logger.error(f"Number of subjects (in df): {int(df.shape[0]/1981)}")
        logger.error("Error in getting unique rows from the dataframe: {}".format(e))
        raise e

    logger.debug(
        f"{split} split: number of unique subjects ({df_string}) = {df.shape[0]}"
    )

    return df


def check_post_metadata_add(df, length_PLR: int = 1981):
    no_nonnull_rows = df.select(pl.count("class_label")).to_numpy()[0]
    no_nonnull_subjects = float(no_nonnull_rows / length_PLR)
    assert no_nonnull_subjects.is_integer(), (
        "Number of non-null rows is not a multiple of the PLR length, "
        "no_nonnull_subjects = {}"
    ).format(no_nonnull_subjects)

    # would be bizarre if this failed if the one above was ok
    no_null_rows = df.select(pl.col("class_label").is_null().sum()).to_numpy()[0]
    no_null_subjects = float(no_null_rows / length_PLR)
    assert no_null_subjects.is_integer(), (
        "Number of null rows is not a multiple of the PLR length, "
        "no_null_subjects = {}"
    ).format(no_null_subjects)

    logger.info(
        "After adding metadata to the PLR data, "
        "we have {} subjects with a class label (control vs glaucoma), and {} with no class labels".format(
            int(no_nonnull_subjects), int(no_null_subjects)
        )
    )


def pick_debug_data(df, string, cfg, n: int = 4, pick_random: bool = False):
    logger.warning(
        'You have a debug mode on, picking a subset of the "{}" data!'.format(string)
    )
    logger.warning("Number of subjects to pick per label = {}".format(n))
    # Use a smaller data subset so things run faster (this is useful for debugging and testing)
    unique_labels = get_unique_labels(df)
    check_for_data_lengths(df, cfg)
    df_out = pl.DataFrame()
    for idx, label in enumerate(unique_labels):
        # df_label = pick_n_subjects_per_label(df, label, n)
        df_label = pick_per_label(df, label, cfg)
        df_label = pick_n_subjects_per_label_pandas(df_label, n, pick_random=pick_random)
        get_list_of_unique_subjects(df_label)
        df_out = pandas_concat(df_out, df_label)
        logger.info(
            f'{idx} ({label}): {int(df_out.shape[0]/cfg["DATA"]["PLR_length"])} subjects for the "{label}" label'
        )
        logger.info(get_list_of_unique_subjects(df_label))

    check_for_data_lengths(df_out, cfg)
    return df_out


def combine_split_dataframes(
    df_train, df_val, cfg: DictConfig, debug_mode: bool = False, debug_n: int = 4, pick_random: bool = False,
        demo_mode: bool = False,
):
    def add_split_column(df, string, debug_mode=False, debug_n=debug_n,
                         pick_random :bool = False, demo_mode: bool = False):
        check_for_data_lengths(df, cfg)
        df_pd = df.to_pandas()
        df_pd["split"] = string
        df = pl.DataFrame(df_pd)
        check_for_data_lengths(df, cfg)
        if not demo_mode:
            if debug_mode:
                df = pick_debug_data(df, string, cfg, n=debug_n, pick_random=pick_random)
        else:
            logger.info('Demo mode is on, not picking any debug subjects here')
        return df

    df_train = add_split_column(
        df=df_train, string="train", debug_mode=debug_mode, debug_n=debug_n,
        pick_random=pick_random, demo_mode=demo_mode
    )
    check_for_data_lengths(df_train, cfg)

    df_test = add_split_column(df=df_val, string="test", debug_mode=debug_mode, debug_n=debug_n,
                               pick_random=pick_random, demo_mode=demo_mode
    )
    logger.info(
        "Combining the train ({}) and test ({}) splits".format(
            df_train.shape, df_test.shape
        )
    )
    check_for_data_lengths(df_test, cfg)

    df = pl.concat([df_train, df_test])
    check_for_data_lengths(df, cfg)
    logger.info("Combined dataframe shape = {}".format(df.shape))
    logger.info(f"Number of time points = {df.shape[0]:,}")

    return df


def define_desired_timevector(PLR_length: int = 1981, fps: int = 30):
    time_vector = np.linspace(0, (PLR_length - 1) / fps, PLR_length)
    return time_vector


def check_time_similarity(time_vec_in, time_vec_ideal):
    time_checks = {}
    # check if the time vectors are similar (within a tolerance), picks up rounding off issues
    # and if there is some jitter in the original recording?
    time_checks["allclose"] = np.allclose(time_vec_in, time_vec_ideal, atol=0)
    # check that min and max are the same
    time_checks["min_in"] = np.min(time_vec_in)
    time_checks["min_same"] = np.min(time_vec_in) == np.min(time_vec_ideal)
    time_checks["max_in"] = np.max(time_vec_in)
    time_checks["max_same"] = np.max(time_vec_in) == np.max(time_vec_ideal)
    time_checks["OK"] = (
        time_checks["allclose"] and time_checks["min_same"] and time_checks["max_same"]
    )
    return time_checks


def check_time_vector_quality(
    subject_code: str, csv_subset: pd.DataFrame, cfg: DictConfig
):
    time_vec_in = csv_subset["time"].to_numpy()
    time_vec_ideal = define_desired_timevector(PLR_length=cfg["DATA"]["PLR_length"])
    assert (
        time_vec_in.shape[0] == time_vec_ideal.shape[0]
    ), "Time vector length does not match"
    time_checks = check_time_similarity(time_vec_in, time_vec_ideal)

    return time_vec_in, time_vec_ideal, time_checks


def check_for_unique_timepoints(
    df, cfg: DictConfig, col: str = "time", assert_on_error: bool = True
):
    time_df = get_unique_polars_rows(df, unique_col=col)
    time_col = time_df["time"].to_numpy()
    if assert_on_error:
        assert len(time_col) == cfg["DATA"]["PLR_length"], (
            f"Number of unique time points {len(time_col)} is "
            f"not {cfg['DATA']['PLR_length']} in the imported dataframe "
            f"(all the subjects)\nWhich means that subjects had different"
            f"timevectors"
        )
    else:
        logger.warning(
            f"Number of unique time points {len(time_col)} is not {cfg['DATA']['PLR_length']} in the"
        )
        logger.warning(
            "imported dataframe (all the subjects)\nWhich means that subjects had different timevectors"
        )
        logger.warning(
            'This is still ok for "time_raw" col as we are using "ideal time vector" for the modeling and'
        )
        logger.warning("visualization to account for small rounding off errors")


def fix_for_orphaned_nans(
    subject_code: str,
    csv_subset: pd.DataFrame,
    cfg: DictConfig,
    cols: tuple = ("Red", "Blue"),
):
    # Check for orphaned NaNs
    for col in cols:
        no_orphaned_nans = csv_subset[col].isnull().sum()
        if no_orphaned_nans > 0:
            # e.g. PLR4199, PLR4195, PLR4194, PLR1127, PLR4204, PLR1081, PLR4140, PLR4164
            logger.warning(
                f"Subject {subject_code} has {no_orphaned_nans} orphaned NaNs in the {col} column after "
                f"interpolation (replacing with 0)"
            )
            csv_subset[col] = csv_subset[col].fillna(0)
            no_orphaned_nans_after = csv_subset[col].isnull().sum()
            if no_orphaned_nans_after > 0:
                logger.error(
                    f"Subject {subject_code} still has {no_orphaned_nans_after} orphaned NaNs in the {col} column"
                )
                raise ValueError(
                    f"Subject {subject_code} still has {no_orphaned_nans_after} orphaned NaNs in the {col} column"
                )

    assert (
        csv_subset.shape[0] == cfg["DATA"]["PLR_length"]
    ), 'Length of the PLR data for "{}" is not {}'.format(
        subject_code, cfg["DATA"]["PLR_length"]
    )

    return csv_subset


def check_for_data_lengths(df, cfg):
    unique_codes = get_unique_polars_rows(df)
    unique_codes = list(unique_codes["subject_code"].to_numpy())

    for code in unique_codes:
        df_code = df.filter(pl.col("subject_code") == code)
        assert (
            len(df_code) == cfg["DATA"]["PLR_length"]
        ), 'Length ({}) of the PLR data for "{}" is not {}'.format(
            df_code.shape[0], code, cfg["DATA"]["PLR_length"]
        )


def transform_data_for_momentfm(X, mask, dataset_cfg, model_name):
    logger.debug("Trimming the data for MomentFM")
    logger.debug(f"Trimming to size = {dataset_cfg.trim_to_size}")
    logger.debug(f"Downsample factor = {dataset_cfg.downsample_factor}")

    # Input data, e.g. standardized pupil size (PLR)
    X = transform_for_moment_fm_length(
        data_array=X,
        trim_to_size=dataset_cfg.trim_to_size,
        pad_ts=dataset_cfg.pad_ts,
        downsample_factor=dataset_cfg.downsample_factor,
        resample_method=dataset_cfg.resample_method,
        split_subjects_to_windows=dataset_cfg.split_subjects_to_windows,
        fill_na=dataset_cfg.fill_na,
        model_name=model_name,
    )

    # Mask data, e.g. what you have labeled as being outliers
    # no_of_outliers = mask.sum()
    mask = transform_for_moment_fm_length(
        data_array=mask,
        trim_to_size=dataset_cfg.trim_to_size,
        pad_ts=dataset_cfg.pad_ts,
        downsample_factor=dataset_cfg.downsample_factor,
        resample_method=dataset_cfg.resample_method,
        split_subjects_to_windows=dataset_cfg.split_subjects_to_windows,
        fill_na="0",
        binarize_output=True,
        model_name=model_name,
    )
    assert mask.shape == X.shape, "Mask and data shapes do not match"

    # Input mask data, e.g. as we have some NaNs padded, we can tell MomentFM to ignore these
    # and attend to the parts where mask is 1
    # "The input mask is utilized to regulate the time steps or patches that the model should attend to.
    #  For instance, in the case of shorter time series, you may opt not to attend to padding. To implement this,
    #  you can provide an input mask with zeros in the padded locations."
    input_mask = np.zeros((X.shape[0], X.shape[1]))
    input_mask[~np.isnan(X)] = 1

    return X.copy(), mask.copy(), input_mask.copy()


def fill_na_in_array_before_windowing(array, fill_na, trim_to_size, model_name):
    """
    array: np.ndarray
        shape (batch_size, time_points)
    """
    logger.debug(f"Filling NaNs in the array before windowing with {fill_na}")

    def fill_na_per_subject(
        array_subj: np.ndarray,
        fill_na="median",
        model_name: str = "UniTS",
        start_idxs=(9, 12),
        end_idxs=(1987, 1990),
        trim_to_size=trim_to_size,
    ):
        if fill_na == "median":
            fillna_start = np.nanmedian(array_subj[start_idxs[0] : start_idxs[1]])
            fillna_end = np.nanmedian(array_subj[end_idxs[0] : end_idxs[1]])
        elif fill_na == "0":
            fillna_start = 0
            fillna_end = 0
        else:
            logger.error(f"fill_na = {fill_na} not implemented")
            raise NotImplementedError(f"fill_na = {fill_na} not implemented")
        array_subj[: start_idxs[0]] = fillna_start
        array_subj[end_idxs[1] :] = fillna_end
        return array_subj

    if model_name is not None:
        if fill_na is not None:
            if model_name == "TimesNet" or model_name == "UniTS":
                # TimesNet does not like NaNs in the input data so you can have this quick hacky fix
                # Assumed that the padding is now a multiple of 100 (or 500) -> giving 2,000 as PLR length
                # Momemnt used 512*4=2048 in contrast
                no_subjects = array.shape[0]
                for subj_idx in range(no_subjects):
                    array[subj_idx, :] = fill_na_per_subject(
                        array_subj=array[subj_idx, :],
                        fill_na=fill_na,
                        model_name=model_name,
                    )
            elif model_name == "NuwaTS":
                no_subjects = array.shape[0]
                # TODO! make this automagic later, these are the indices
                #  that you get the median from, not the Nan Indices per se
                start_idxs = (17, 20)
                end_idxs = (1994, 1997)
                for subj_idx in range(no_subjects):
                    array[subj_idx, :] = fill_na_per_subject(
                        array_subj=array[subj_idx, :],
                        fill_na="median",  # fill_na,
                        model_name=model_name,
                        start_idxs=start_idxs,
                        end_idxs=end_idxs,
                    )
            else:
                # e.g. Moment is okay with NaNs in the data as you can use the input_mask to mask out invalid
                # (e.g. padded or missing points) in the input data
                logger.warning("Unknown model_name = {}".format(model_name))
                raise ValueError(f"Unknown model_name = {model_name}")

            no_of_nans = np.sum(np.isnan(array))
            assert no_of_nans == 0, "No NaNs detected, after padding and filling"

    return array


def transform_for_moment_fm_length(
    data_array: np.ndarray,
    trim_to_size: int = 512,
    pad_ts: bool = True,
    downsample_factor: int = 4,
    resample_method: str = "cubic",
    split_subjects_to_windows: bool = True,
    binarize_output: bool = False,
    fill_na: str = None,
    model_name: str = None,
):
    if pad_ts:
        # Pad to the next multiple of 512, e.g. (1981,) -> (2048,) with NaNs for the padding
        array = pad_glaucoma_PLR(data_array=data_array, trim_to_size=trim_to_size)
    else:
        # Trim to the multiple of trim_to_size (e.g. 96): (1981,) -> (1920) = 20*96
        array = trim_to_multiple_of(data_array=data_array, window_size=trim_to_size)
        assert array.shape[1] % trim_to_size == 0, (
            "Something funky happened with the trim? "
            "Length ({}) should be a multiple of trim_to_size ({})".format(
                array.shape[1], trim_to_size
            )
        )

    # Make new pseudosubjects
    if split_subjects_to_windows:
        # e.g. (355,1981) -> (7100,100) for TimesNet
        array = fill_na_in_array_before_windowing(
            array, fill_na, trim_to_size, model_name
        )
        array = split_subjects_to_windows_PLR(array=array, window_size=trim_to_size)

    else:
        if downsample_factor is not None:
            array = downsample_PLR(
                array=array,
                downsample_factor=downsample_factor,
                resample_method=resample_method,
            )

    return array


def trim_to_multiple_of(data_array: np.ndarray, window_size: int = 96):
    length_in = data_array.shape[1]
    no_of_windows = np.floor(length_in / window_size).astype(int)
    new_length = no_of_windows * window_size
    to_trim = length_in - new_length
    if (to_trim % 2) == 0:  # for odd trim
        i1 = to_trim // 2
        i2 = length_in - i1
    else:  # for even, trim 1 point more from beginning with less important data
        i1 = to_trim // 2
        i2 = length_in - i1
        i1 += 1

    return data_array[:, i1:i2]


def get_no_of_windows(length_PLR: int = 1981, window_size: int = 512):
    return np.ceil(length_PLR / window_size).astype(int)


def split_subjects_to_windows_PLR(array: np.ndarray, window_size: int = 512):
    windows_per_subject = array.shape[1] // window_size
    no_subjects = array.shape[0]
    array_out = np.reshape(array, (no_subjects * windows_per_subject, window_size))
    return array_out


def downsample_PLR(
    array: np.ndarray, downsample_factor: int = 4, resample_method: str = "cubic"
):
    def downsample_subject(x, y, downsample_factor, resample_method):
        nan_ratio = np.isnan(y).sum() / len(y)
        x_new = np.linspace(x[0], x[-1], len(x) // downsample_factor)
        f = interpolate.interp1d(x, y, kind=resample_method)
        y_resampled = f(x_new)
        nan_ratio_resampled = np.isnan(y_resampled).sum() / len(y_resampled)
        # we know assume that you only have NaN padding and no NaNs in the signal, so if you get some new
        # NaNs, it is an issue. And this might happen also with non-nice downsample factors?
        safety_factor = 1.5
        assert (
            nan_ratio_resampled < nan_ratio * safety_factor
        ), f"NaN ratio before ({nan_ratio}) and after ({nan_ratio_resampled}) resampling do not match"
        # import matplotlib.pyplot as plt
        # plt.plot(x, y)
        # plt.show()

        return y_resampled

    samples_out: int = int(array.shape[1] // downsample_factor)
    no_subjects, no_timepoints = array.shape
    for i in range(no_subjects):
        x = np.linspace(0, no_timepoints, no_timepoints)
        y = array[i, :]
        y_resampled = downsample_subject(x, y, downsample_factor, resample_method)
        if i == 0:
            y_out = y_resampled
        else:
            y_out = np.vstack((y_out, y_resampled))

    assert (
        y_out.shape[1] == samples_out
    ), f"Downsampled array length is not {samples_out}"

    nan_ratio = np.isnan(y_out).sum() / y_out.size
    assert nan_ratio != 1, "All your values seem NaN now"

    return y_out


def unpad_glaucoma_PLR(array: np.ndarray, length_PLR: int = 1981):
    start_idx, end_idx = get_padding_indices(
        length_orig=length_PLR, length_padded=array.shape[1]
    )
    array_out = array[:, start_idx:end_idx]

    return array_out


def get_padding_indices(length_orig: int = 1981, length_padded: int = 2048):
    no_points_pad = length_padded - length_orig  # 67
    start_idx = no_points_pad // 2  # 33
    end_idx = start_idx + length_orig  # 2014
    return start_idx, end_idx


def pad_glaucoma_PLR(data_array: np.ndarray, trim_to_size: int = 512):
    new_length = int(np.ceil(data_array.shape[1] / trim_to_size)) * trim_to_size  # 2048
    length_in = data_array.shape[1]  # 1981
    start_idx, end_idx = get_padding_indices(length_in, new_length)

    # Pad the input array with NaNs
    array_out = np.zeros((data_array.shape[0], new_length))
    array_out[:] = np.nan
    nan_sum = np.isnan(array_out).sum()
    array_out[:, start_idx:end_idx] = data_array

    assert nan_sum != np.isnan(array_out).sum(), "it seems that all you have is NaNs?"
    assert array_out.shape[1] == new_length, f"Padded array length is not {new_length}"
    return array_out
