from loguru import logger
import polars as pl
from omegaconf import DictConfig

from src.data_io.data_utils import get_unique_polars_rows


def add_task_target_masks(df_raw, cfg, null_mask: pl.Series, mask_name: str):
    # You can obviosuly same some disk space without adding these, but maybe more intuitive
    # if you want to re-analyze the dataframe (duckDB) without the code years or months from now,
    # most likely by a person who is not familiar with the code at all, and don't necessarily want/need to get familiar
    # no_trues = null_mask.sum()
    null_mask = null_mask.rename(mask_name)
    df_raw.insert_column(df_raw.shape[1], null_mask)

    return df_raw


def impute_from_gt(
    df_raw: pl.DataFrame,
    col_with_missing: str,
    col_gt: str,
    new_col_name: str,
    cfg: DictConfig,
):
    # Impute the missing values in the col_with_missing with the col_gt
    # This is a simple imputation method, where the missing values are replaced with the ground truth
    # This is useful for some outlier detection algorithms that might need non-NaN vectors
    no_nans_in = df_raw[col_with_missing].null_count()
    nan_percentage = no_nans_in / len(df_raw) * 100
    logger.info(
        f"Imputing {no_nans_in} NaNs ({nan_percentage:.2f}%) in {col_with_missing} with {col_gt}"
    )

    no_nans_gt = df_raw[col_gt].null_count()
    assert no_nans_gt == 0, (
        f"You are trying to impute from column ({col_gt}) "
        f"that has also nulls! no of nulls = {no_nans_gt}"
    )

    # Create a duplicate of the column with missing values
    df_raw = df_raw.with_columns([pl.col(col_with_missing).alias(new_col_name)])

    # And then impute on this new column so that you still have the non-imputed column
    null_mask = df_raw[col_with_missing].is_null()
    df_raw = df_raw.with_columns(pl.col(new_col_name).fill_null(pl.col(col_gt)))
    no_nans_out = df_raw[new_col_name].null_count()
    assert (
        no_nans_out == 0
    ), f"Imputation failed, {no_nans_out} NaNs left in {col_with_missing}"

    if "_orig" in new_col_name:
        # the _orig signal has some missing values "rejected" by the pupillometer software, and
        # outliers still in the signal. These remaining outliers have been manually removed and are
        # the missing values in "raw" data, i.e. the imputation mask below, so you can use the same
        # ground truth ("outlier_mask") for both outlier detection and imputation
        logger.debug('Outlier/imputation mask added with "_raw", not with "_orig"')
        df_raw = add_task_target_masks(df_raw, cfg, null_mask, mask_name="outlier_mask")
    elif "_raw" in new_col_name:
        df_raw = add_task_target_masks(
            df_raw, cfg, null_mask, mask_name="imputation_mask"
        )
    else:
        logger.error(f"Unknown column name {new_col_name}")
        raise NotImplementedError

    return df_raw


def update_number_of_outliers(df_raw, cfg):
    # Get the number of outliers per subject, from "outlier_mask" column
    unique_codes = list(get_unique_polars_rows(df_raw, "subject_code")["subject_code"])
    outlier_sums = []

    for code in unique_codes:
        no_outliers = df_raw.filter(pl.col("subject_code") == code)[
            "outlier_mask"
        ].sum()
        outlier_sums.append(no_outliers)
        df_raw = df_raw.with_columns(
            no_outliers=pl.when(pl.col("subject_code") == code)
            .then(pl.lit(no_outliers))
            .otherwise(pl.col("no_outliers"))
        )

    outliers_out = list(get_unique_polars_rows(df_raw, "subject_code")["no_outliers"])
    assert outlier_sums == outliers_out, "Outlier sums not updated correctly"

    return df_raw


def fix_outlier_mask(df_raw, cfg):
    outlier_mask_sum = df_raw["outlier_mask"].sum()
    outlier_perc_in = outlier_mask_sum / len(df_raw) * 100
    # imputation_mask_sum = df_raw["imputation_mask"].sum()

    df_raw = df_raw.with_columns(
        pl.when(
            pl.col("outlier_mask") == 0,
            pl.col("imputation_mask") == 1,
        )
        .then(1)
        .otherwise(0)
        .alias("outlier_mask")
    )
    # e.g. 2.08% -> 7.66%%
    outlier_perc_out = df_raw["outlier_mask"].sum() / len(df_raw) * 100
    logger.info(
        "Fixed the outlier mask, {:.2f}% -> {:.2f}%".format(
            outlier_perc_in, outlier_perc_out
        )
    )
    # e.g. 9.74% this contains all the missing values
    logger.info(
        "Imputation mask percentage: {:.2f}%".format(
            df_raw["imputation_mask"].sum() / len(df_raw) * 100
        )
    )

    return df_raw


def impute_orig_for_training(df_raw: pl.DataFrame, cfg: DictConfig):
    # Some outlier detection algorithms might need non-NaN vectors,
    # https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/anomaly_detection.ipynb
    # Featurization (e.g. AUC) from raw data with missing value might benefit from imputed gt
    if cfg["DATA"]["impute_missing_data_for_orig_and_raw"]:
        if cfg["DATA"]["imputation_method"] == "gt":
            new_col_name = "pupil_orig_imputed"
            df_raw = impute_from_gt(
                df_raw,
                col_with_missing="pupil_orig",
                col_gt="pupil_gt",
                new_col_name=new_col_name,
                cfg=cfg,
            )
            new_col_name = "pupil_raw_imputed"
            df_raw = impute_from_gt(
                df_raw,
                col_with_missing="pupil_raw",
                col_gt="pupil_gt",
                new_col_name=new_col_name,
                cfg=cfg,
            )

            # Fix the outlier mask, as these were actually filled with the "pupil_gt" values and were hardly outliers
            # and were rejected by pupillometer. If you flag these as outliers, the outlier detection algorithms
            # might get confused
            df_raw = fix_outlier_mask(df_raw, cfg)

        else:
            logger.error(
                f"Imputation method {cfg['DATA']['imputation_method']} not implemented"
            )
            raise NotImplementedError
    else:
        logger.warning("Not imputing missing data for original/raw data")

    return df_raw
