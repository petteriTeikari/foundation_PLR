import time
from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from missforest import MissForest
from omegaconf import DictConfig

from src.imputation.train_utils import imputation_per_split_of_dict


def missforest_create_imputation_dicts(model, df_dict, source_data, cfg):
    """Create imputation dictionaries from MissForest model outputs.

    Transforms MissForest imputation results into the standardized format
    used by PyPOTS models for downstream processing compatibility.

    Parameters
    ----------
    model : MissForest
        Trained MissForest model.
    df_dict : dict
        Dictionary of DataFrames keyed by split name ('train', 'test').
    source_data : dict
        Source data containing 'df' with data dictionaries per split
        and 'preprocess' with standardization statistics.
    cfg : DictConfig
        Configuration for imputation settings.

    Returns
    -------
    dict
        Dictionary mapping split names to imputation results in
        PyPOTS-compatible format.
    """
    # Harmonize the output with the PyPOTS outputs, so that the downstream code works well
    # see pypots_imputer_wrapper()
    imputed_dict = {}
    for i, split in enumerate(source_data["df"]):
        imputed_dict[split] = {}
        data_dicts = source_data["df"][split]["data"]
        # metadata = source_data["df"][split]["metadata"]
        imputed_dict[split] = imputation_per_split_of_dict(
            data_dicts=data_dicts,
            df=df_dict[split],
            preprocess=source_data[
                "preprocess"
            ],  # needs only the standardization stats
            model=model,
            split=split,
            cfg=cfg,
        )

    return imputed_dict


def check_df(df: pd.DataFrame):
    """Validate and convert DataFrame for MissForest compatibility.

    Logs the number of NaN values and ensures all columns are float type
    to prevent type errors during MissForest fitting.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with potential NaN values.

    Returns
    -------
    pd.DataFrame
        DataFrame with all columns cast to float type.
    """
    no_of_nans_cols = df.isnull().sum()
    no_of_nans = no_of_nans_cols.sum()
    logger.info("Number of NaNs in train_df: {}".format(no_of_nans))

    # model.fit(
    #   python3.11/site-packages/missforest/missforest.py", line 438, in fit
    #     nrmse_score.append(
    #   python3.11/site-packages/missforest/_array.py", line 47, in append
    #     raise ValueError(f"Datatype of new item must {self.dtype}.")
    # ValueError: Datatype of new item must <class 'float'>.
    # python-BaseException

    # if not isinstance(item, self.dtype):
    #     raise ValueError(f"Datatype of new item must {self.dtype}.")
    # when item: float32 instead of <class 'float'> in self.dtype

    # dtypes = df.dtypes
    df = df.astype(float)
    # logger.info(df.dtypes)

    return df


def missforest_fit_script(train_df: pd.DataFrame, cfg: DictConfig):
    """Fit a MissForest model on training data.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame with NaN values to learn imputation patterns from.
    cfg : DictConfig
        Configuration containing MODELS.MISSFOREST.MODEL parameters.

    Returns
    -------
    tuple
        (model, results) where model is the fitted MissForest instance and
        results is a dict with 'train' timing in seconds.
    """
    train_df = check_df(df=train_df)
    logger.info("Fitting the MissForest model")
    start_time = time.time()
    # Default estimators are lgbm classifier and regressor
    params = cfg["MODELS"]["MISSFOREST"]["MODEL"]
    model = MissForest(**params)
    logger.info("MissForest | Model parameters: {}".format(params))
    model.fit(
        x=train_df,
    )
    results = {"train": time.time() - start_time}
    logger.info("Fitting done in {:.2f} seconds".format(results["train"]))

    return model, results


def get_dataframes_from_dict_for_missforest(source_data: dict):
    """Convert source data dictionaries to DataFrames for MissForest.

    Extracts arrays from source data, applies masks by setting masked
    values to NaN, and converts to pandas DataFrames.

    Parameters
    ----------
    source_data : dict
        Source data containing 'df' with 'train' and 'test' splits,
        each having 'data' with 'X' arrays and 'mask' arrays.

    Returns
    -------
    tuple
        (df_train, df_test) as pandas DataFrames with NaN values where
        mask indicates missing data.

    Raises
    ------
    AssertionError
        If input arrays contain unexpected NaN values or masking fails.
    """

    def get_df_per_split(split_dict):
        """Extract array from split dict, apply mask, and convert to DataFrame."""
        array = split_dict["data"]["X"]
        no_of_nans = np.sum(np.isnan(array))
        assert no_of_nans == 0, "There are NaNs in the data"
        mask = split_dict["data"]["mask"]
        mask_sum = np.sum(mask == 1)
        array[mask == 1] = np.nan
        masked_sum = np.sum(np.isnan(array))
        assert masked_sum == mask_sum, "Masking issue"
        df = pl.DataFrame(array)
        return df

    df_train = get_df_per_split(split_dict=deepcopy(source_data["df"]["train"]))
    df_test = get_df_per_split(split_dict=deepcopy(source_data["df"]["test"]))

    return df_train.to_pandas(), df_test.to_pandas()


def missforest_main(
    source_data: dict,
    model_cfg: DictConfig,
    cfg: DictConfig,
    model_name: str = None,
    run_name: str = None,
):
    """
    See e.g. El Badisy et al. (2024) https://doi.org/10.1186/s12874-024-02305-3
    Albu et al. (2024) https://arxiv.org/abs/2407.03379 for "missForestPredict"
    Original paper by Stekhoven and Bühlmann (2012) https://doi.org/10.1093/bioinformatics/btr597
    Python https://github.com/yuenshingyan/MissForest / https://pypi.org/project/MissForest/
    """
    # MissForest is not learning from data, rather working on dataset-wise, so not even a good algorithm
    # to be used in production with new patients
    # MissForest not implemented - placeholder for future work.
    # See function docstring for references on implementation approach.
    raise NotImplementedError(
        "MissForest imputation not implemented. "
        "Check for the output scaling, test/train seems differently scaled/standardized"
    )
