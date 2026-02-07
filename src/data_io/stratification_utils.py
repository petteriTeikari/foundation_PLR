import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from omegaconf import DictConfig, ListConfig
from skmultilearn.model_selection import IterativeStratification

from src.data_io.data_utils import check_data_import, get_unique_polars_rows


def add_split_col_to_dataframe(df_raw: pl.DataFrame, split_codes: dict):
    """Add a 'split' column to dataframe based on subject code assignments.

    Parameters
    ----------
    df_raw : pl.DataFrame
        Raw data dataframe.
    split_codes : dict
        Dictionary mapping split names to lists of subject codes.

    Returns
    -------
    pl.DataFrame
        Dataframe with added 'split' column.
    """
    if isinstance(df_raw, pd.DataFrame):
        df_raw = pl.DataFrame(df_raw)  # if a Pandas, convert to Polars

    # Add the split column to the raw data
    df_raw = df_raw.with_columns(pl.lit(None).alias("split"))

    for split, codes in split_codes.items():
        for i, code in enumerate(codes):
            df_raw = df_raw.with_columns(
                pl.when(pl.col("subject_code") == code)
                .then(pl.lit(split))
                .otherwise("split")
                .alias("split")
            )

    return df_raw


def multicol_stratification(
    df_tmp: pd.DataFrame,
    test_size: float,
    stratify_columns: list,
    cfg: DictConfig,
    col_to_return: str = "subject_code",
) -> dict:
    """Perform multi-column iterative stratification for train/test split.

    Custom iterative train test split which 'maintains balanced representation
    with respect to order-th label combinations.'

    Parameters
    ----------
    df_tmp : pd.DataFrame
        Temporary dataframe with columns to stratify on.
    test_size : float
        Proportion of data to use for test set (0-1).
    stratify_columns : list
        List of column names to use for stratification.
    cfg : DictConfig
        Configuration dictionary.
    col_to_return : str, optional
        Column name to return for each split, by default "subject_code".

    Returns
    -------
    dict
        Dictionary with 'train' and 'test' keys mapping to lists of values.

    References
    ----------
    - https://www.abzu.ai/data-science/stratified-data-splitting-part-2/
    - https://madewithml.com/courses/mlops/splitting/#stratified-split
    """
    # One-hot encode the stratify columns and concatenate them
    if isinstance(df_tmp, pl.DataFrame):
        df_tmp = df_tmp.to_pandas()
    one_hot_cols = [pd.get_dummies(df_tmp[col]) for col in stratify_columns]
    one_hot_cols = pd.concat(one_hot_cols, axis=1).to_numpy()
    stratifier = IterativeStratification(
        n_splits=2,
        order=len(stratify_columns),
        sample_distribution_per_fold=[test_size, 1 - test_size],
    )
    train_indices, test_indices = next(
        stratifier.split(df_tmp.to_numpy(), one_hot_cols)
    )
    # Return the train and test set dataframes
    train, test = (
        df_tmp.iloc[train_indices][col_to_return],
        df_tmp.iloc[test_indices][col_to_return],
    )
    return {"train": list(train), "test": list(test)}


def create_tmp_stratification_df(df_raw: pl.DataFrame, stratify_columns: ListConfig):
    """Create a temporary dataframe for stratification with binned features.

    Parameters
    ----------
    df_raw : pl.DataFrame
        Raw data dataframe.
    stratify_columns : ListConfig
        List of columns to use for stratification.

    Returns
    -------
    pd.DataFrame
        Temporary pandas dataframe with subject_code, no_outliers_bins,
        and class_label columns.
    """
    # Stratify the data based on both the class_label and the missingness_ratio
    codes = get_unique_polars_rows(df_raw, "subject_code")
    subject_codes = codes["subject_code"].to_numpy()
    no_outliers = codes["no_outliers"].to_numpy()
    no_outliers_bins = bin_outlier_counts(
        no_outliers
    )  # Bin the missingness (or no of outliers) into n bins
    class_labels = codes["class_label"].to_numpy()
    df_tmp = pd.DataFrame(
        {
            "subject_code": subject_codes,
            "no_outliers_bins": no_outliers_bins,
            "class_label": class_labels,
        }
    )

    return df_tmp


def bin_outlier_counts(outliers: list, no_of_bins: int = 5):
    """Bin outlier counts into quantile-based categories.

    Parameters
    ----------
    outliers : list
        List of outlier counts per subject.
    no_of_bins : int, optional
        Number of bins to create, by default 5.

    Returns
    -------
    np.ndarray
        Array of bin labels for each subject.
    """
    labels = np.linspace(0, no_of_bins - 1, no_of_bins)
    bins = pd.qcut(outliers, no_of_bins, labels=labels)
    return bins.to_numpy()


def create_splits_to_df(df_raw, cfg: DictConfig):
    """Create stratified splits and add split column to dataframe.

    Parameters
    ----------
    df_raw : pl.DataFrame
        Raw data dataframe.
    cfg : DictConfig
        Configuration with STRATIFICATION settings.

    Returns
    -------
    pl.DataFrame
        Dataframe with added 'split' column.

    Raises
    ------
    ValueError
        If any data points have missing split assignments.
    """
    # Create three-column df for two-column stratification to get bac the subject codes per split
    df_tmp = create_tmp_stratification_df(
        df_raw, stratify_columns=cfg["STRATIFICATION"]["test_size"]
    )
    # Get subject codes belonging to the training and validation sets
    split_codes = multicol_stratification(
        df_tmp=df_tmp,
        test_size=cfg["STRATIFICATION"]["test_size"],
        stratify_columns=list(cfg["STRATIFICATION"]["stratify_columns"]),
        cfg=cfg,
    )

    # Add the split column to the raw data
    df_raw = add_split_col_to_dataframe(df_raw, split_codes)

    # Check that all the data has a split
    if df_raw["split"].null_count() > 0:
        logger.error(f"Data has {df_raw['split'].null_count()} missing splits")
        raise ValueError(f"Data has {df_raw['split'].null_count()} missing splits")

    return df_raw


def stratify_splits(
    df_raw: pl.DataFrame,
    cfg: DictConfig,
):
    """Main function to stratify data into train and test splits.

    Performs multi-column stratification based on class labels and outlier
    counts, ensuring balanced representation in both splits.

    Parameters
    ----------
    df_raw : pl.DataFrame
        Raw data dataframe with all subjects.
    cfg : DictConfig
        Configuration with STRATIFICATION settings.

    Returns
    -------
    tuple
        Tuple containing (df_train, df_test) as Polars dataframes.
    """
    logger.info("Create splits (train/test)")
    df = create_splits_to_df(df_raw, cfg)

    # Split the data into training and validation Polars dataframes
    df_train = df.filter(pl.col("split") == "train")
    df_test = df.filter(pl.col("split") == "test")

    # Check against data leakage
    check_data_import(df_train, df_test, display_outliers=False)

    return df_train, df_test
