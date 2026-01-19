from loguru import logger
import os

import polars as pl

from src.utils import get_data_dir

FNAME_CTRL = "Master_File_De-Identified_Copy_For_Petteri_control.xlsx"
FNAME_GLAUCOMA = "Master_File_De-Identified_Copy_For_Petteri_glaucoma.xlsx"


def pick_subset_of_cols(df: pl.DataFrame) -> pl.DataFrame:
    cols_to_keep = ["SubjectChar", "Age"]
    df_subset = df.select(pl.col(cols_to_keep))

    # cast age to float (decimal)
    df_subset = df_subset.with_columns(
        pl.col("Age").cast(pl.Decimal(precision=2, scale=0))
    )
    df_subset = df_subset.rename(
        {"SubjectChar": "subject_code"}
    )  # rename the column to match the other dataframes

    no_rows = df_subset.shape[0]
    df_subset = df_subset.drop_nulls()
    no_rows_after_drop = df_subset.shape[0]
    logger.debug(
        f"Dropped {no_rows - no_rows_after_drop} rows with missing values in the subset"
    )

    return df_subset


def import_excel_file(path: str, class_label: str):
    if not os.path.exists(path):
        logger.error(f"XLSX File not found: {path}")

    logger.info(f"Reading the metadata from the XLSX files: {path}")
    df = pl.read_excel(source=path)

    df = pick_subset_of_cols(df)
    df = df.with_columns(pl.lit(class_label).alias("class_label"))
    df = check_for_duplicate_codes(df)

    logger.info(
        f"XLSX IMPORT: Number of PLR recordings in the {class_label} group: {df.shape[0]}"
    )

    return df


def check_for_duplicate_codes(df):
    codes = df.select(pl.col("subject_code"))
    code_duplicates = codes.filter(codes.is_duplicated())
    if code_duplicates.shape[0] > 0:
        logger.warning(
            f"Duplicate subject_codes codes found in the metadata: {code_duplicates}"
        )
    # implement some autodropping? Will be done later when doing top-1 values per code

    return df


def get_metadata_from_xlsx(path_in_ctrl: str = None, path_in_glaucoma: str = None):
    df_ctrl = import_excel_file(path=path_in_ctrl, class_label="control")
    df_glaucoma = import_excel_file(path=path_in_glaucoma, class_label="glaucoma")

    # combine the dataframes
    df = pl.concat([df_ctrl, df_glaucoma])

    return df


def metadata_wrapper(metadata_cfg):
    data_dir = get_data_dir()
    logger.info(
        f"Reading the metadata from the XLSX files in the directory: {data_dir}"
    )
    df = get_metadata_from_xlsx(
        path_in_ctrl=os.path.join(data_dir, FNAME_CTRL),
        path_in_glaucoma=os.path.join(data_dir, FNAME_GLAUCOMA),
    )

    return df
