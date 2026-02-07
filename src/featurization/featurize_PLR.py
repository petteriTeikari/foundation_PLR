from copy import deepcopy

import mlflow
import polars as pl
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from src.data_io.data_wrangler import (
    convert_subject_dict_of_arrays_to_df,
    get_subject_dict_for_featurization,
)
from src.featurization.feature_log import (
    export_features_to_mlflow,
    featurization_mlflow_metrics_and_params,
)
from src.featurization.feature_utils import (
    flatten_dict_to_dataframe,
    get_light_stimuli_timings,
)
from src.featurization.featurizer_PLR_subject import (
    check_that_features_are_not_the_same_for_colors,
    get_features_per_color,
)
from src.featurization.visualize_features import visualize_features_of_all_sources
from src.log_helpers.mlflow_utils import get_mlflow_info
from src.log_helpers.retrain_or_not import if_refeaturize_from_imputation
from src.preprocess.preprocess_PLR import (
    destandardize_the_data_dict_for_featurization,
)


def featurize_subject(
    subject_dict: dict,
    subject_code: str,
    cfg: DictConfig,
    feature_cfg: DictConfig,
    i: int,
    feature_col: str = "X",
):
    """Compute all features for a single subject.

    Extracts features for each light color and combines with metadata.

    Parameters
    ----------
    subject_dict : dict
        Dictionary containing subject data arrays.
    subject_code : str
        Unique subject identifier.
    cfg : DictConfig
        Main configuration dictionary.
    feature_cfg : DictConfig
        Feature-specific configuration.
    i : int
        Subject index in the dataset.
    feature_col : str, optional
        Column name for feature values, by default 'X'.

    Returns
    -------
    dict
        Dictionary with features per color and metadata.
    """
    features = {}

    # Convert to Polars dataframe
    df_subject: pl.DataFrame = convert_subject_dict_of_arrays_to_df(subject_dict)
    light_timings = get_light_stimuli_timings(df_subject)
    for color in light_timings.keys():
        features[color] = get_features_per_color(
            df_subject,
            light_timing=light_timings[color],
            bin_cfg=feature_cfg["FEATURES"],
            color=color,
            feature_col=feature_col,
        )

    # check that colors are different
    check_that_features_are_not_the_same_for_colors(features)

    # add metadata to the dataframe
    df_subject_metadata = convert_subject_dict_of_arrays_to_df(
        subject_dict, wildcard_categories=["metadata", "labels"]
    )
    features["metadata"] = df_subject_metadata

    return features


def compute_features_from_dict(
    split_dict: dict,
    split: str,
    preprocess_dict: dict,
    feature_cfg: DictConfig,
    cfg: DictConfig,
):
    """Compute features for all subjects in a data split.

    Destandardizes data if needed, then iterates through subjects to
    compute hand-crafted PLR features.

    Parameters
    ----------
    split_dict : dict
        Dictionary containing split data with 'data' and 'X' arrays.
    split : str
        Split name (e.g., 'train', 'test').
    preprocess_dict : dict
        Preprocessing statistics for destandardization.
    feature_cfg : DictConfig
        Feature configuration.
    cfg : DictConfig
        Main configuration dictionary.

    Returns
    -------
    dict
        Dictionary keyed by subject_code containing computed features.
    """
    # Destandardize the data (if needed)
    split_dict = destandardize_the_data_dict_for_featurization(
        split, split_dict, preprocess_dict, cfg
    )

    no_of_subjects = split_dict["data"]["X"].shape[0]
    features_per_code = {}
    for i in tqdm(
        range(no_of_subjects), total=no_of_subjects, desc="Featurizing the PLR subjects"
    ):
        # server: Featurizing the PLR subjects: 100%|██████████| 152/152 [02:03<00:00,  1.23it/s] why so slow?
        # laptop: Featurizing the PLR subjects: 100%|██████████| 16/16 [00:01<00:00, 11.18it/s]
        # Keeps just the one subject in each of the arrays in the "data_dict"
        # TODO! Make this faster, the pl.DataFrame creation per subject slowing thins down
        subject_dict = get_subject_dict_for_featurization(split_dict, i, cfg)
        subject_code = subject_dict["metadata"]["subject_code"][0]

        # Compute the features for the subject
        features_per_code[subject_code] = featurize_subject(
            subject_dict,
            subject_code=subject_code,
            cfg=cfg,
            feature_cfg=feature_cfg,
            i=i,
        )

    return features_per_code


# @task(
#     log_prints=True,
#     name="Featurize PLR Time Series (Handcrafted Features)",
#     description="Derive binned features from time series",
# )
def get_handcrafted_PLR_features(
    source_data: dict, cfg: DictConfig, feature_cfg: DictConfig
):
    """Extract handcrafted PLR features from source data.

    Processes all splits, computes features per subject, and flattens
    the nested structure into dataframes.

    Parameters
    ----------
    source_data : dict
        Source data dictionary with 'df', 'preprocess', and 'mlflow' keys.
    cfg : DictConfig
        Main configuration dictionary.
    feature_cfg : DictConfig
        Feature-specific configuration.

    Returns
    -------
    dict
        Dictionary with 'data' (dataframes per split) and 'mlflow_run'.
    """
    features_nested = {}
    preprocess_dict = source_data["preprocess"]
    for split, split_dict in deepcopy(source_data["df"]).items():
        features_nested[split] = compute_features_from_dict(
            split_dict, split, preprocess_dict, feature_cfg, cfg
        )

    # Subject-wise dicts flattened to a dataframe
    features = flatten_dict_to_dataframe(
        features_nested=features_nested, mlflow_series=source_data["mlflow"], cfg=cfg
    )

    return features


def featurization_script(
    experiment_name: str,
    prev_experiment_name: str,
    cfg: DictConfig,
    source_name: str,
    source_data: dict,
    featurization_method: str,
    feature_cfg: DictConfig,
    run_name: str,
):
    """Execute the featurization pipeline for a single source.

    Runs featurization with MLflow tracking, logging parameters, metrics,
    and artifacts. Supports handcrafted features and embeddings.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name for featurization.
    prev_experiment_name : str
        Previous experiment name (imputation).
    cfg : DictConfig
        Main configuration dictionary.
    source_name : str
        Name of the data source being featurized.
    source_data : dict
        Source data dictionary.
    featurization_method : str
        Method name ('handcrafted_features' or 'embeddings').
    feature_cfg : DictConfig
        Feature configuration.
    run_name : str
        MLflow run name.
    """
    if if_refeaturize_from_imputation(
        run_name=run_name, experiment_name=experiment_name, cfg=cfg
    ):
        with mlflow.start_run(run_name=run_name):
            # Log params and metrics to MLflow
            featurization_mlflow_metrics_and_params(
                mlflow_run=source_data["mlflow"], source_name=source_name, cfg=cfg
            )

            # Task) Featurize the data
            if (
                feature_cfg["FEATURES_METADATA"]["feature_method"]
                == "handcrafted_features"
            ):
                features = get_handcrafted_PLR_features(
                    source_data=source_data, cfg=cfg, feature_cfg=feature_cfg
                )
            elif feature_cfg["FEATURES_METADATA"]["feature_method"] == "embeddings":
                logger.error("Embeddings not implemented yet")
                raise NotImplementedError("Embeddings not implemented yet")
                # features = get_PLR_embeddings(
                #     source_data=source_data, cfg=cfg, feature_cfg=feature_cfg
                # )
            else:
                logger.error("Unknown feature method")
                raise NotImplementedError("Unknown feature method")

            # Task) Log to MLflow
            export_features_to_mlflow(
                features=features,
                run_name=run_name,
                cfg=cfg,
            )

            # TODO! If you like to use a Feature Store, you could implement it here:
            #  https://www.snowflake.com/guides/what-feature-store-machine-learning/
            #  https://github.com/awesome-mlops/awesome-feature-store

            # Task) Visualize the features
            mlflow_info = get_mlflow_info()
            visualize_features_of_all_sources(
                features=features, mlflow_infos=mlflow_info, cfg=cfg
            )

            mlflow.end_run()

    else:
        # You could put the visualization here, and read the results from MLflow
        logger.info(
            "The imputation results have been already featurized, skipping the featurization step"
        )
