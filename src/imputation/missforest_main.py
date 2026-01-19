from copy import deepcopy

import numpy as np
import pandas as pd
from loguru import logger
import time
import polars as pl
from omegaconf import DictConfig
from missforest import MissForest

from src.imputation.train_utils import imputation_per_split_of_dict


def missforest_create_imputation_dicts(model, df_dict, source_data, cfg):
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
    #logger.info(df.dtypes)

    return df


def missforest_fit_script(train_df: pd.DataFrame, cfg: DictConfig):
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
    def get_df_per_split(split_dict):
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
    raise NotImplementedError('Check for the output scaling, test/train seems differently scaled/standardized')
    model_artifacts = {}

    # Transform helper from data_df to easier to use data formats
    df_train, df_test = get_dataframes_from_dict_for_missforest(source_data)

    # Fit the model
    model, model_artifacts["timing"] = missforest_fit_script(train_df=df_train, cfg=cfg)

    # Impute the data
    start_time = time.time()  # This is very slow then
    df_dict = {"train": df_train, "test": df_test}
    model_artifacts["imputation"] = missforest_create_imputation_dicts(
        model, df_dict, source_data, cfg
    )
    model_artifacts["timing"]["imputation"] = time.time() - start_time
    logger.info(
        "Imputation done in {:.2f} seconds".format(
            model_artifacts["timing"]["imputation"]
        )
    )

    return model, model_artifacts
