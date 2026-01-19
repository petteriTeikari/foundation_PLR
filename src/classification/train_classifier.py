import warnings

import polars as pl
from loguru import logger
from omegaconf import DictConfig

from src.classification.catboost.catboost_main import catboost_main
from src.classification.sklearn_simple_classifiers import sklearn_simple_cls_main

from src.classification.tabm.tabm_main import tabm_main
from src.classification.tabpfn_main import tabpfn_main
from src.classification.xgboost_cls.xgboost_main import xgboost_main


def drop_metadata_cols_from_df(
    df: pl.DataFrame, cfg: DictConfig, drop_subject_code: bool = False
) -> (pl.DataFrame, dict):
    # drop all the columns that have the metadata_ prefix
    metadata = {}
    for col in df.columns:
        if col.startswith("metadata_"):
            metadata[col] = df[col]
            df = df.drop(col)

    if drop_subject_code:
        # Depending on your downstream classifier, default is to keep it with the features
        # drop the subject code column, and now you are left only with the PLR features
        # e.g. blue, red (both the "value" (e.g. AUC, mean, median) and the CI (from stdev of a time window for example)
        if "subject_code" in df.columns:
            df = df.drop("subject_code")

    return df, metadata


def what_to_train_the_classifier_on(features_per_source: dict, cfg: DictConfig):
    """
    You could just keep the PLR features, or add some metadata fields to the features,
    e.g. age or some other field that you imported from the Excel file.
    """
    metadata = {}
    if cfg["CLASSIFICATION_SETTINGS"]["DATA_TO_TRAIN"]["scheme"] == "features":
        train_df: pl.DataFrame = features_per_source["data"]["train"]
        test_df: pl.DataFrame = features_per_source["data"]["test"]
        drop_metata_cols = False
        if drop_metata_cols:
            # at the moment, downstream code work with the metadata columns,
            # and you could have "class_label" in the metadata as they are the targets
            train_df, metadata["train"] = drop_metadata_cols_from_df(
                df=train_df, cfg=cfg
            )
            test_df, metadata["test"] = drop_metadata_cols_from_df(df=test_df, cfg=cfg)
    else:
        logger.error(
            "Unknown data scheme '{}' for classification".format(
                cfg["CLASSIFICATION_SETTINGS"]["DATA_TO_TRAIN"]["scheme"]
            )
        )
        raise ValueError(
            "Unknown data scheme '{}' for classification".format(
                cfg["CLASSIFICATION_SETTINGS"]["DATA_TO_TRAIN"]["scheme"]
            )
        )

    return train_df, test_df, metadata


# @task(
#     log_prints=True,
#     name="PLR Classifiers (One Source)",
#     description="Train classifiers for one source",
# )
def train_classifier(
    source_name: str,
    features_per_source: dict,
    cls_model_name: str,
    run_name: str,
    cfg: DictConfig,
):
    try:
        cls_model_cfg = cfg["CLS_MODELS"][cls_model_name]
    except KeyError:
        logger.error(f"Unknown classifier model: {cls_model_name}")
        raise ValueError(f"Unknown classifier model: {cls_model_name}")

    # What to train the classifier on
    train_df, test_df, _ = what_to_train_the_classifier_on(features_per_source, cfg)

    # MLflow run init here?
    #  TODO! Log also the MLflow run ids here (as in outlier detection, imputation, featurization)
    #   as this is the final task before summarization so it is nice to know the full trace of the "blocks"
    #   in the pipeline

    # Model selector
    if "XGBOOST" in cls_model_name:
        xgboost_main(
            train_df,
            test_df,
            run_name=run_name,
            cfg=cfg,
            xgboost_cfg=cls_model_cfg,
            # Now the variants share the hyperparam config
            hparam_cfg=cfg["CLS_HYPERPARAMS"][cls_model_name],
            features_per_source=features_per_source,
        )
    elif "CATBOOST" in cls_model_name:
        catboost_main(
            train_df,
            test_df,
            run_name=run_name,
            cfg=cfg,
            cls_model_cfg=cls_model_cfg,
            # Now the variants share the hyperparam config
            hparam_cfg=cfg["CLS_HYPERPARAMS"][cls_model_name],
            features_per_source=features_per_source,
        )
    elif "TabM" in cls_model_name:
        tabm_main(
            train_df,
            test_df,
            run_name=run_name,
            cfg=cfg,
            cls_model_cfg=cls_model_cfg,
            # Now the variants share the hyperparam config
            hparam_cfg=cfg["CLS_HYPERPARAMS"][cls_model_name],
            features_per_source=features_per_source,
        )
    elif "TabPFN" in cls_model_name:
        warnings.simplefilter("ignore")
        tabpfn_main(
            train_df,
            test_df,
            run_name=run_name,
            cfg=cfg,
            cls_model_cfg=cls_model_cfg,
            # Now the variants share the hyperparam config
            hparam_cfg=cfg["CLS_HYPERPARAMS"][cls_model_name],
            features_per_source=features_per_source,
        )
        warnings.resetwarnings()
    elif "LogisticRegression" in cls_model_name:
        sklearn_simple_cls_main(
            train_df,
            test_df,
            model_name=cls_model_name,
            cfg=cfg,
            cls_model_cfg=cls_model_cfg,
            hparam_cfg=cfg["CLS_HYPERPARAMS"][cls_model_name],
            run_name=run_name,
            features_per_source=features_per_source,
        )
    else:
        logger.error(f"Unknown classifier model: {cls_model_name}")
        raise ValueError(f"Unknown classifier model: {cls_model_name}")
