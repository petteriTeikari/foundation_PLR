from copy import deepcopy
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import psutil
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from src.anomaly_detection.anomaly_utils import outlier_detection_artifacts_dict
from src.data_io.data_wrangler import convert_df_to_dict
from src.data_io.flow_data import flow_import_data
from src.ensemble.ensemble_anomaly_detection import ensemble_masks
from src.ensemble.ensemble_utils import (
    get_best_imputation_col_name,
    get_grouped_classification_runs,
)
from src.log_helpers.log_naming_uris_and_dirs import (
    get_foundation_model_names,
    get_model_name_from_run_name,
    get_simple_outlier_detectors,
)
from src.log_helpers.mlflow_artifacts import (
    get_col_for_for_best_anomaly_detection_metric,
)

# def create_dict_for_featurization_from_imputation_results_and_original_data(
#     imputation_results, cfg
# ):
#     for i, (model_name, model_dict) in enumerate(imputation_results.items()):
#         if i == 0:
#             print(f"Model name: {model_name}")
#             # get the original data (i.e. "raw" without the outliers, and 'gt' that is denoised and imputed)
#             results = get_original_data_to_results(model_dict, cfg)
#
#         # Now process normally the imputed data
#         results[model_name] = get_imputed_results(model_dict, cfg)
#
#     return results


def get_best_mlflow_col_for_imputation(cfg: DictConfig, string: str = "MAE") -> str:
    """Get the MLflow column name for the best imputation metric.

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary containing IMPUTATION_METRICS settings.
    string : str, optional
        Metric identifier (e.g., "MAE"), by default "MAE".

    Returns
    -------
    str
        MLflow column name for the specified metric.
    """
    best_metric: dict = cfg["IMPUTATION_METRICS"]["best_metric"]
    return best_metric[string]


def get_best_string_for_imputation(cfg: DictConfig, split: str = "test") -> dict:
    """Get the best metric configuration dictionary for imputation.

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary containing IMPUTATION_METRICS settings.
    split : str, optional
        Data split name, by default "test".

    Returns
    -------
    dict
        Best metric configuration dictionary.
    """
    best_metric = cfg["IMPUTATION_METRICS"]["best_metric"]
    return best_metric


def get_best_string_for_outlier_detection(cfg: DictConfig) -> dict:
    """Get the best metric configuration for outlier detection.

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary containing OUTLIER_DETECTION settings.

    Returns
    -------
    dict
        Best metric configuration for outlier detection.
    """
    what_is_best = cfg["OUTLIER_DETECTION"]["what_is_best"]
    return cfg["OUTLIER_DETECTION"][what_is_best]


def get_best_string_for_classification(cfg: DictConfig) -> dict:
    """Get the best metric configuration for classification.

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary containing CLASSIFICATION_SETTINGS.

    Returns
    -------
    dict
        Best metric configuration for classification.
    """
    return cfg["CLASSIFICATION_SETTINGS"]["BEST_METRIC"]


def get_best_dict(task: str, cfg: DictConfig) -> Optional[dict]:
    """Get the best metric dictionary for a given task.

    Parameters
    ----------
    task : str
        Task name ("outlier_detection", "imputation", "featurization", or "classification").
    cfg : DictConfig
        Configuration dictionary.

    Returns
    -------
    dict or None
        Best metric configuration dictionary, or None for featurization.

    Raises
    ------
    NotImplementedError
        If the task is unknown.
    """
    if task == "outlier_detection":
        best_dict = get_best_string_for_outlier_detection(cfg)
    elif task == "imputation":
        best_dict = get_best_string_for_imputation(cfg)
    elif task == "featurization":
        best_dict = None
    elif task == "classification":
        best_dict = get_best_string_for_classification(cfg)
    else:
        logger.error(f"Unknown task: {task}")
        raise NotImplementedError(f"Unknown task: {task}")
    return best_dict


def get_best_run_dict(run_df: pd.DataFrame, best_dict: dict, task: str) -> pd.DataFrame:
    """Sort MLflow runs by the best metric and return the sorted dataframe.

    Parameters
    ----------
    run_df : pd.DataFrame
        DataFrame of MLflow runs.
    best_dict : dict
        Configuration specifying the metric column and sort direction.
    task : str
        Task name for metric column selection.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame with best runs first.

    Raises
    ------
    ValueError
        If task is unknown or metric column not found.
    """
    if task == "outlier_detection":
        col_name = get_col_for_for_best_anomaly_detection_metric(best_dict, task)
    elif task == "imputation":
        col_name = get_best_imputation_col_name(best_dict)
    elif task == "classification":
        col_name = get_best_imputation_col_name(best_dict)
    else:
        logger.error(f"Unknown task: {task}")
        raise ValueError(f"Unknown task: {task}")

    if col_name not in run_df.columns:
        logger.error(f"Unknown string: {best_dict['string']}")
        logger.error(f"Available columns: {run_df.columns}")
        raise ValueError(f"Unknown string: {best_dict['string']}")

    if best_dict["direction"] == "ASC":
        run_df = run_df.sort_values(by=col_name, ascending=True)
    elif best_dict["direction"] == "DESC":
        run_df = run_df.sort_values(by=col_name, ascending=False)
    else:
        logger.error(f"Unknown direction: {best_dict['direction']}")
        raise ValueError(f"Unknown direction: {best_dict['direction']}")

    return run_df


def drop_ensemble_runs(runs_model: pd.DataFrame) -> pd.DataFrame:
    """Remove ensemble runs from a MLflow runs dataframe.

    Parameters
    ----------
    runs_model : pd.DataFrame
        DataFrame of MLflow runs.

    Returns
    -------
    pd.DataFrame
        DataFrame with ensemble runs removed.
    """
    runs_model_out = pd.DataFrame()
    logger.info("Dropping ensemble runs")
    for idx, row in runs_model.iterrows():
        if "ensemble" not in row["tags.mlflow.runName"]:
            runs_model_out = pd.concat([runs_model_out, pd.DataFrame(row).T])
    return runs_model_out


def foundation_model_filter(
    mlflow_runs: pd.DataFrame, best_dict: dict, model_name: str, task: str
) -> Optional[pd.DataFrame]:
    """Filter foundation model runs to get best zeroshot and finetuned variants.

    The idea is to get both the zeroshot and finetuned model with the foundation
    models (if available) whereas with the more traditional models, the zeroshot
    option is not there typically (or does not perform so well at all).

    Parameters
    ----------
    mlflow_runs : pd.DataFrame
        DataFrame of all MLflow runs.
    best_dict : dict
        Configuration specifying the best metric and direction.
    model_name : str
        Name of the foundation model to filter for.
    task : str
        Task name for metric selection.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with filtered runs, or None if no runs found.
    """
    df = pd.DataFrame()
    runs_model = mlflow_runs[
        mlflow_runs["tags.mlflow.runName"].str.contains(model_name)
    ]
    runs_model = drop_ensemble_runs(runs_model)

    if df.shape[0] > 0:
        criteria = ["zeroshot", "finetune"]
        data_sources = ["gt", "orig"]
        # You should get 3 (or 4 runs) as the zeroshot would be evaluate always on the "orig", split
        # so it does not matter if the source is "gt" or "orig"
        for criterion in criteria:
            for data_source in data_sources:
                try:
                    runs_criterion = runs_model[
                        runs_model["tags.mlflow.runName"].str.contains(criterion)
                    ]
                except Exception as e:
                    logger.error(
                        f"Could not filter the runs with criterion: {criterion}, model: {model_name}"
                    )
                    raise e
                runs_criterion_source = runs_criterion[
                    runs_model["tags.mlflow.runName"].str.contains(data_source)
                ]
                if runs_criterion_source.shape[0] > 0:
                    run_df = get_best_run_dict(runs_criterion_source, best_dict, task)
                    df = pd.concat([df, run_df.iloc[0:1]])

        return df

    else:
        logger.warning(f"No runs found for foundation model: {model_name}")
        return None


def get_best_imputation_runs(
    mlflow_runs: pd.DataFrame, task: str, cfg: DictConfig
) -> pd.DataFrame:
    """Get the best run for each unique imputer+outlier combination.

    Unlike best outlier_runs, we now have added 3 new fields to the mlflow_runs:
    1. imputer_model
    2. outlier_source
    3. unique_combo

    And the unique_combo defines how many imputer+outlier_source combos we have
    for featurization.

    Parameters
    ----------
    mlflow_runs : pd.DataFrame
        DataFrame of MLflow runs with unique_combo column.
    task : str
        Task name for metric selection.
    cfg : DictConfig
        Configuration dictionary.

    Returns
    -------
    pd.DataFrame
        DataFrame with one best run per unique combination.
    """
    unique_combos = mlflow_runs["unique_combo"].unique()
    best_dict = get_best_dict(task, cfg)
    runs_out = pd.DataFrame()
    for i, unique_combo in enumerate(unique_combos):
        run_df = mlflow_runs[mlflow_runs["unique_combo"] == unique_combo]
        best_run = get_best_run_dict(run_df, best_dict, task)[0:1]
        assert best_run.shape[0] == 1, f"Expected 1 run, got {best_run.shape[0]}"
        logger.debug(f"Unique combo: {unique_combo}")
        runs_out = pd.concat([runs_out, best_run])

    assert len(runs_out) == len(unique_combos), (
        f"Expected {len(unique_combos)} runs, got {len(runs_out)}"
    )

    return runs_out


def drop_foundational_models(
    mlflow_runs: pd.DataFrame, foundation_model_names: list[str]
) -> pd.DataFrame:
    """Remove foundation model runs from MLflow runs dataframe.

    Parameters
    ----------
    mlflow_runs : pd.DataFrame
        DataFrame of MLflow runs.
    foundation_model_names : list
        List of foundation model name strings to filter out.

    Returns
    -------
    pd.DataFrame
        DataFrame with foundation model runs removed.
    """

    def check_name(foundation_model_names: list[str], run_name: str) -> bool:
        is_foundational_run = False
        for name in foundation_model_names:
            if name in run_name:
                is_foundational_run = True
        return is_foundational_run

    runs_out = pd.DataFrame()
    for i, row in mlflow_runs.iterrows():
        if not check_name(foundation_model_names, run_name=row["tags.mlflow.runName"]):
            runs_out = pd.concat([runs_out, pd.DataFrame(row).T])

    return runs_out


def get_best_model_runs(
    mlflow_runs: pd.DataFrame, task: str, cfg: DictConfig
) -> pd.DataFrame:
    """Get the best runs for each model type (foundation and traditional).

    Manual definition of what you want to compare. You could also simply use
    all MLflow runs as source, and put the return_subset to False. This
    artisanal selection is here just to reduce the number of combos to return.

    Parameters
    ----------
    mlflow_runs : pd.DataFrame
        DataFrame of all MLflow runs.
    task : str
        Task name for metric selection.
    cfg : DictConfig
        Configuration dictionary.

    Returns
    -------
    pd.DataFrame
        DataFrame with best runs for each model type.
    """
    best_dict = get_best_dict(task, cfg)

    # Get the best foundational model runs
    foundation_model_names = get_foundation_model_names()
    runs_out = pd.DataFrame()
    for model_name in foundation_model_names:
        mlflow_runs_foundational = foundation_model_filter(
            mlflow_runs, best_dict, model_name=model_name, task=task
        )
        if mlflow_runs_foundational is not None:
            runs_out = pd.concat([runs_out, mlflow_runs_foundational])

    # Drop foundational models
    mlflow_others = drop_foundational_models(mlflow_runs, foundation_model_names)
    runs_out = pd.concat([runs_out, mlflow_others])

    # Get ensemble runs
    mlflow_ensemble = mlflow_runs[
        mlflow_runs["tags.mlflow.runName"].str.contains("ensemble")
    ]
    runs_out = pd.concat([runs_out, mlflow_ensemble])

    logger.info(f"Found {len(runs_out)} runs")

    return runs_out


def get_unique_outlier_runs(
    mlflow_runs: pd.DataFrame, cfg: DictConfig, task: str
) -> pd.DataFrame:
    """Get one best run per unique run name from MLflow runs.

    Parameters
    ----------
    mlflow_runs : pd.DataFrame
        DataFrame of MLflow runs.
    cfg : DictConfig
        Configuration dictionary.
    task : str
        Task name for metric selection.

    Returns
    -------
    pd.DataFrame
        DataFrame with one best run per unique run name.
    """
    best_dict = get_best_dict(task, cfg)
    best_runs = pd.DataFrame()
    unique_run_names = mlflow_runs["tags.mlflow.runName"].unique()
    for unique_run in unique_run_names:
        run_df = mlflow_runs[mlflow_runs["tags.mlflow.runName"] == unique_run]
        if best_dict is not None:
            run_df = get_best_run_dict(run_df, best_dict, task)
        else:
            # e.g. featurization does not have any metrics, so sort by latest
            run_df = run_df.sort_values(by="start_time", ascending=False)
        best_runs = pd.concat([best_runs, run_df.iloc[0:1]])

    assert len(best_runs) == len(unique_run_names), (
        f"Expected {len(unique_run_names)} runs, got {len(best_runs)}"
    )

    # Drop rows with NaN values in the best metric
    # best_runs = best_runs.dropna(subset=[best_string])

    return best_runs


def parse_run_name_for_two_model_names(
    run_name: str, delimiter: str = "__"
) -> tuple[str, str]:
    """Parse a run name to extract imputer model and outlier source names.

    Parameters
    ----------
    run_name : str
        MLflow run name in format "imputer__outlier_source".
    delimiter : str, optional
        Delimiter separating model names, by default "__".

    Returns
    -------
    tuple
        Tuple containing (imputer_model, outlier_source) as simplified names.

    Raises
    ------
    Exception
        If the run name cannot be parsed.
    """
    try:
        imputer_model, outlier_source = run_name.split(delimiter)
    except Exception:
        try:
            # how did the extra delimiter appear here?
            imputer_model, outlier_source, extra = run_name.split(delimiter)
            outlier_source = outlier_source + "_" + extra
        except Exception as e:
            logger.error('Could not parse run name "{}"'.format(run_name))
            raise e
    imputer_model = simplify_model_name(imputer_model)
    outlier_source = simplify_model_name(outlier_source)
    if outlier_source == "TimesNet":
        outlier_source = "TimesNet-gt"
    return imputer_model, outlier_source


def simplify_model_name(model_name: str, delimiter: str = "_") -> str:
    """Simplify a model name by extracting the core identifier.

    Handles special cases for ensemble and MOMENT model naming conventions.

    Parameters
    ----------
    model_name : str
        Full model name from MLflow run.
    delimiter : str, optional
        Delimiter in the model name, by default "_".

    Returns
    -------
    str
        Simplified model name.
    """
    model_name = model_name.replace("pupil_", "pupil-")
    model_name_out = model_name.split(delimiter)[0]
    # if "zeroshot" in model_name:
    #     model_name_out = model_name_out + "-zeroshot"
    # elif "finetune" in model_name:
    #     model_name_out = model_name_out + "-finetune"
    if "ensemble" in model_name:
        if "gt_thresholded" not in model_name:
            model_name_out = model_name_out.replace("ensembleThresholded", "ensemble")
    else:
        if "MOMENT" in model_name:
            model_name_fields = model_name.split("_")
            if model_name_fields[3] == "gt" or model_name_fields[3] == "orig":
                # Outlier naming
                model_name_out = model_name_out = (
                    model_name_fields[0]
                    + "-"
                    + model_name_fields[3]
                    + "-"
                    + model_name_fields[1]
                )
            else:
                # Imputation naming
                model_name_out = model_name_fields[0] + "-" + model_name_fields[1]

    model_name_out = model_name_out.replace("UniTS-Outlier", "UniTS")

    return model_name_out


def get_unique_combo_runs(
    mlflow_runs_in: pd.DataFrame, cfg: DictConfig, task: str, delimiter: str = "__"
) -> pd.DataFrame:
    """Add unique combo columns to MLflow runs for imputer+outlier tracking.

    Parses run names to extract imputer_model and outlier_source, creating
    a unique_combo identifier for each combination.

    Parameters
    ----------
    mlflow_runs_in : pd.DataFrame
        Input MLflow runs DataFrame.
    cfg : DictConfig
        Configuration dictionary.
    task : str
        Task name.
    delimiter : str, optional
        Delimiter separating model names, by default "__".

    Returns
    -------
    pd.DataFrame
        DataFrame with added imputer_model, outlier_source, and unique_combo columns.

    Raises
    ------
    ValueError
        If any run has a NaN run_id.
    """
    # Add empty columns to the Pandas DataFrame
    mlflow_runs = deepcopy(mlflow_runs_in)
    mlflow_runs = mlflow_runs.reset_index(drop=True)
    mlflow_runs["imputer_model"] = ""
    mlflow_runs["outlier_source"] = ""
    mlflow_runs["unique_combo"] = ""

    for i, run_df in enumerate(mlflow_runs.iterrows()):
        run_name = run_df[1]["tags.mlflow.runName"]
        imputer_model, outlier_source = parse_run_name_for_two_model_names(
            run_name, delimiter=delimiter
        )
        mlflow_runs.at[i, "imputer_model"] = imputer_model
        mlflow_runs.at[i, "outlier_source"] = outlier_source
        unique_combo_string = f"{imputer_model}{delimiter}{outlier_source}"
        mlflow_runs.at[i, "unique_combo"] = unique_combo_string
        logger.debug(
            f"{i + 1}/{mlflow_runs.shape[0]}: {imputer_model}, {outlier_source}, {run_name}"
        )
        logger.info(f"{i + 1}/{mlflow_runs.shape[0]}: {unique_combo_string}")

    assert len(mlflow_runs) == len(mlflow_runs_in), (
        "input mlflow_run has different number of runs ({}) "
        "than the output mlflow_run ({})".format(len(mlflow_runs_in), len(mlflow_runs))
    )

    run_ids = mlflow_runs["run_id"].tolist()
    isnan = False
    for run_id in run_ids:
        if isinstance(run_id, float):
            isnan = np.isnan(run_id)

    if isnan:
        logger.error("You have NaN runs?")
        logger.error(run_ids)
        logger.error(mlflow_runs)
        raise ValueError("You have NaN runs?")

    # unique_combos = sorted(list(set(mlflow_runs["unique_combo"])))
    return mlflow_runs


def get_previous_best_mlflow_runs(
    experiment_name: str,
    cfg: DictConfig,
    task: str = "outlier_detection",
    return_subset: bool = True,
) -> Optional[pd.DataFrame]:
    """Get the best MLflow runs from a previous experiment for use as data sources.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment.
    cfg : DictConfig
        Configuration dictionary.
    task : str, optional
        Task name to determine filtering strategy, by default "outlier_detection".
    return_subset : bool, optional
        Whether to return a curated subset or all runs, by default True.

    Returns
    -------
    pd.DataFrame or None
        DataFrame of best MLflow runs, or None if no runs found.

    Raises
    ------
    ValueError
        If the task is unknown.
    NotImplementedError
        If featurization task is requested.
    """
    # Gives all the runs per experiment
    mlflow_runs = mlflow.search_runs(experiment_names=[experiment_name])

    # Gives one run per unique run_name (with the best metric/loss returned)
    if mlflow_runs.shape[0] == 0:
        return None
    else:
        mlflow_runs = get_unique_outlier_runs(mlflow_runs, cfg, task=task)

        # Manual selection
        # Note! The task names refer to previous tasks
        if task == "outlier_detection":
            # Outlier detection results -> imputation source
            if return_subset:
                # i.e. best zero-shot and best finetuned per model, and not all the
                # different hyperparam combos you tried (and were reflected in the run_name)
                mlflow_runs = get_best_model_runs(mlflow_runs, task, cfg)
                logger.info(
                    "Returning subset of the outlier detection runs, {} runs".format(
                        len(mlflow_runs)
                    )
                )
        elif task == "imputation":
            # Imputation results -> featurization source
            mlflow_runs_combo = get_unique_combo_runs(mlflow_runs, cfg, task=task)
            if return_subset:
                # Note! with the MOMENT models, the logic is the same as for outlier detection,
                # i.e. keep best zeroshot and finetuned model
                mlflow_runs = get_best_imputation_runs(mlflow_runs_combo, task, cfg)
                logger.info(
                    "Returning subset of the imputation runs, {} runs".format(
                        len(mlflow_runs)
                    )
                )

        elif task == "featurization":
            # PLR features (handcrafted or embeddings) -> classification source
            logger.info("Returning all the featurization runs")
            raise NotImplementedError("Featurization not implemented")

        elif task == "classification":
            best_unique_models = get_grouped_classification_runs(
                {}, experiment_name, cfg, task
            )
            # TODO! you could use this as a sanity check
            no_of_mlflow_runs = mlflow_runs.shape[0]
            no_of_unique_submodels_in_ensemble = 0
            for ensemble_name, ensemble_dict in best_unique_models.items():
                no_of_unique_submodels_in_ensemble += len(ensemble_dict)
            if no_of_mlflow_runs != no_of_unique_submodels_in_ensemble:
                logger.error(
                    "The number of unique submodels in the ensemble does not match the number of MLflow runs"
                )
                logger.error(
                    f"No of MLflow runs: {no_of_mlflow_runs}, "
                    f"no of unique submodels in ensemble: {no_of_unique_submodels_in_ensemble}"
                )

        else:
            logger.error(f"Unknown task: {task}")
            raise ValueError(f"Unknown task: {task}")

        return mlflow_runs


def get_arrays_for_splits_from_imputer_artifacts(
    artifacts: dict, run_name: str
) -> dict[str, dict[str, np.ndarray]]:
    """Extract imputation arrays from MLflow imputer artifacts.

    Parameters
    ----------
    artifacts : dict
        MLflow artifacts containing imputation results.
    run_name : str
        MLflow run name for logging.

    Returns
    -------
    dict
        Dictionary with train/test splits containing X, CI_pos, CI_neg, and mask arrays.
    """
    dict_out = {}
    imputation = artifacts["model_artifacts"]["imputation"]
    for split in imputation.keys():
        imputation_dict = imputation[split]["imputation_dict"]
        imputation_mean = imputation_dict["imputation"]["mean"]
        mask = imputation_dict["indicating_mask"]
        ci_pos = imputation_dict["imputation"]["imputation_ci_pos"]
        ci_neg = imputation_dict["imputation"]["imputation_ci_neg"]
        no_dims = len(imputation_mean.shape)
        if no_dims == 3:
            # this is in "PyPOTS space", make it 2D
            imputation_mean = imputation_mean[:, :, 0]
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            if ci_pos is not None:
                if len(ci_pos.shape) == 3:
                    ci_pos = ci_pos[:, :, 0]
                    ci_neg = ci_neg[:, :, 0]

        if ci_pos is None:
            ci_pos = np.ones_like(imputation_mean)
            ci_pos[:] = np.nan
            ci_neg = np.ones_like(imputation_mean)
            ci_neg[:] = np.nan

        dict_out[split] = {
            # Same as reconstruction
            "X": imputation_mean,
            "CI_pos": ci_pos,
            "CI_neg": ci_neg,
            "mask": mask,
        }

    return dict_out


def check_arrays(splits_dicts: dict[str, dict[str, np.ndarray]], task: str) -> None:
    """Validate that X and mask arrays have matching shapes.

    Parameters
    ----------
    splits_dicts : dict
        Dictionary of split dictionaries containing X and mask arrays.
    task : str
        Task name for error messages.

    Raises
    ------
    AssertionError
        If X and mask arrays have different shapes.
    """
    for split, split_dict in splits_dicts.items():
        assert split_dict["X"].shape == split_dict["mask"].shape, (
            "{} X and mask have different sizes, X: {}, mask: {}".format(
                split, split_dict["X"].shape, split_dict["mask"].shape
            )
        )


def get_best_epoch(outlier_artifacts: dict) -> tuple[dict, bool]:
    """Extract the best epoch results from outlier detection artifacts.

    Handles multiple artifact formats from different outlier detection methods.

    Parameters
    ----------
    outlier_artifacts : dict
        Dictionary of outlier detection artifacts from MLflow.

    Returns
    -------
    tuple
        Tuple containing (results_best, simple_format) where simple_format
        indicates the artifact structure type.
    """
    # TODO! There is no need to have all these options, have all the outlier detection method output the same format
    simple_format = True
    if "outlier_results" in outlier_artifacts:
        # if you logged results at each epoch
        if outlier_artifacts["metadata"]["best_epoch"] is not None:
            results_best = outlier_artifacts["outlier_results"][
                outlier_artifacts["metadata"]["best_epoch"]
            ]
        else:
            last_key = list(outlier_artifacts["outlier_results"].keys())[-1]
            results_best = outlier_artifacts["outlier_results"][last_key]
        simple_format = False
    elif "best_arrays" in outlier_artifacts:
        results_best = outlier_artifacts["best_arrays"]
    elif "results" in outlier_artifacts:
        results_best = outlier_artifacts["results"]
    else:
        # e.g. sklearn, Prophet
        results_best = outlier_artifacts
    return results_best, simple_format


def if_pick_the_split(run_name: str, split: str) -> bool:
    """Determine if a given split should be processed based on run name.

    Parameters
    ----------
    run_name : str
        MLflow run name.
    split : str
        Split name to check.

    Returns
    -------
    bool
        True if the split should be picked, False otherwise.
    """
    pick_split = False
    if "outlier" in split:
        return True

    # these did not have any outlier split
    simple_detectors = get_simple_outlier_detectors()
    for name in simple_detectors:
        if name in run_name:
            return True

    return pick_split


def get_arrays_for_splits_from_outlier_artifacts(
    outlier_artifacts: dict, run_name: str
) -> dict[str, dict[str, np.ndarray]]:
    """Extract reconstruction and mask arrays from outlier detection artifacts.

    Handles multiple artifact formats from different outlier detection methods
    (MOMENT, TimesNet, LOF, Prophet, etc.).

    Parameters
    ----------
    outlier_artifacts : dict
        Dictionary of outlier detection artifacts from MLflow.
    run_name : str
        MLflow run name for logging and method detection.

    Returns
    -------
    dict
        Dictionary with train/test splits containing X (reconstruction) and mask arrays.
        Format: {split: {'X': np.array, 'mask': np.array}}

    Raises
    ------
    ValueError
        If arrays cannot be extracted from the artifacts.
    AssertionError
        If no splits were selected for analysis.
    """
    # best_arrays_format = "best_arrays" in outlier_artifacts
    results_best, simple_format = get_best_epoch(outlier_artifacts)
    dict_out = {}
    for split in results_best.keys():
        if if_pick_the_split(run_name, split):
            # Remember that the "vanilla train and test" were used for reconstruction learning, and did not
            # contain any outlier labels (unsupervised learning), and the outlier detection capability was
            # evaluated using the "pupil_orig" data (from both "train" and "test")
            split_fields = split.split("_")
            if len(split_fields) == 2:
                split_out = split_fields[1]
            elif len(split_fields) == 1:
                split_out = split_fields[0]
            else:
                logger.error("What is this split = {}".format(split))
                raise ValueError(f"Unknown split {split}")

            try:
                if simple_format:
                    # TimesNet (well not specific to TimesNet, simpler structure, move on to this?)
                    try:
                        dict_out[split_out] = {
                            "X": results_best[split]["preds"],
                            "mask": results_best[split]["pred_mask"],
                        }
                    except Exception:
                        try:
                            dict_out[split_out] = {
                                "X": results_best[split]["arrays"]["preds"],
                                "mask": results_best[split]["arrays"]["pred_mask"],
                            }
                        except Exception:
                            try:
                                # e.g. LOF, Prophet, etc. do not reconstruct
                                X_nan = np.zeros_like(
                                    results_best[split]["arrays"]["pred_mask"]
                                ).astype(float)
                                X_nan[:] = np.nan
                                dict_out[split_out] = {
                                    "X": X_nan,
                                    "mask": results_best[split]["arrays"]["pred_mask"],
                                }
                            except Exception as e:
                                logger.error(
                                    "Could not get best results!, error = {}".format(e)
                                )
                                raise e

                    assert len(dict_out[split_out]["X"].shape) == 2, (
                        "reconstructed signal array needs to be 2D"
                    )
                    assert len(dict_out[split_out]["mask"].shape) == 2, (
                        "mask needs to be 2D"
                    )

                else:
                    # MOMENT
                    # Finetuned
                    dict_out[split_out] = {
                        # Same as reconstruction
                        "X": results_best[split]["results_dict"]["split_results"][
                            "arrays"
                        ]["preds"],
                        # When anomaly score is used with the adaptive f1 to get timepoint-wise labels
                        "mask": results_best[split]["results_dict"]["preds"]["arrays"][
                            "pred_mask"
                        ],
                    }
            except Exception as e:
                logger.error(f"Error: {e}")
                logger.error(
                    f"split: {split}, split_out: {split_out}, run_name: {run_name}"
                )
                raise ValueError(f"Error: {e}")

    assert len(dict_out) > 0, (
        "You did not pick anything from this model to analyze! "
        "Glitch in if_pick_the_split(run_name, split)?"
    )

    return dict_out


def get_ensembled_anomaly_masks(artifacts: dict) -> dict[str, dict[str, np.ndarray]]:
    """Create ensembled anomaly masks from individual detector masks.

    Parameters
    ----------
    artifacts : dict
        Dictionary mapping splits to 3D arrays of individual detector masks.

    Returns
    -------
    dict
        Dictionary with ensembled masks for each split.
    """
    dict_out = {}
    for split, array_3D in artifacts.items():
        dict_out[split] = {"mask": ensemble_masks(array_3D)}
    return dict_out


def get_source_data(
    mlflow_runs: pd.DataFrame, cfg: DictConfig, task: str
) -> tuple[dict, dict]:
    """Load source data arrays from MLflow artifacts for each run.

    Parameters
    ----------
    mlflow_runs : pd.DataFrame
        DataFrame of MLflow runs to process.
    cfg : DictConfig
        Configuration dictionary.
    task : str
        Task name ("outlier_detection" or "imputation").

    Returns
    -------
    tuple
        Tuple containing (dicts_out, mlflow_dict) where dicts_out contains
        the data arrays and mlflow_dict maps source names to MLflow run info.

    Raises
    ------
    ValueError
        If the task is unknown.
    """
    dicts_out = {}
    mlflow_dict = {}
    for mlflow_row in (
        pbar := tqdm(
            mlflow_runs.iterrows(),
            desc="Getting source data",
            total=mlflow_runs.shape[0],
        )
    ):
        mlflow_run = mlflow_row[1]
        run_name = mlflow_run["tags.mlflow.runName"]
        model_name, model_key = get_model_name_from_run_name(run_name, task)
        logger.info(f"Model name: {model_name}, run_name: {run_name}")
        artifacts = outlier_detection_artifacts_dict(mlflow_run, model_name, task)
        if task == "outlier_detection":
            if "ensemble" in model_name:
                dicts_out[model_key] = get_ensembled_anomaly_masks(artifacts)
            else:
                dicts_out[model_key] = get_arrays_for_splits_from_outlier_artifacts(
                    outlier_artifacts=artifacts, run_name=run_name
                )
                check_arrays(splits_dicts=dicts_out[model_key], task=task)
            mlflow_dict[model_key] = mlflow_run
            del artifacts
            pbar.set_description(
                f"Import Sources | RAM use: {psutil.virtual_memory().percent} %: {model_name}"
            )
        elif task == "imputation":
            # Use the a bit shorter unique combo name, the exact names and run_id will still be
            # returned in the mlflow_dict
            dict_key = mlflow_row[1]["unique_combo"]
            dicts_out[dict_key] = get_arrays_for_splits_from_imputer_artifacts(
                artifacts, run_name
            )
            check_arrays(splits_dicts=dicts_out[dict_key], task=task)
            mlflow_dict[dict_key] = mlflow_run
        else:
            logger.error(f"Unknown task: {task}")
            raise ValueError(f"Unknown task: {task}")

    logger.info(f"Found {len(list(dicts_out.keys()))} sources")

    return dicts_out, mlflow_dict


def add_array_to_dict(
    array_to_add: np.ndarray, key: str, astype: Optional[str] = None
) -> np.ndarray:
    """Validate and optionally cast a numpy array before adding to a dictionary.

    Parameters
    ----------
    array_to_add : np.ndarray
        Array to validate and add.
    key : str
        Dictionary key name for error messages.
    astype : str, optional
        Data type to cast the array to, by default None.

    Returns
    -------
    np.ndarray
        Validated (and optionally cast) array.

    Raises
    ------
    ValueError
        If the input is not a numpy array.
    """
    if isinstance(array_to_add, np.ndarray):
        if astype is not None:
            return array_to_add.astype(astype)
        else:
            return array_to_add
    else:
        logger.error(
            "You are trying to add (key={}) a non-Numpy array, type = {}".format(
                key, type(array_to_add)
            )
        )
        if isinstance(array_to_add, dict):
            logger.error(f"dict keys = {list(array_to_add.keys())}")
        raise ValueError(
            "You are trying to add a non-Numpy array, type = {}".format(
                type(array_to_add)
            )
        )


def get_dict_per_col_name(
    col_names: list[str], data_dict: dict, col_name: str, mask_col: str
) -> dict:
    """Create data dictionary structure for a specific column name.

    Parameters
    ----------
    col_names : list
        List of column names to process (typically pupil signal columns).
    data_dict : dict
        Source data dictionary with df and preprocess keys.
    col_name : str
        Target column name for the data.
    mask_col : str
        Column name for the mask data.

    Returns
    -------
    dict
        Structured data dictionary with X, X_GT, and mask arrays.
    """
    data_dicts = {}
    for gt_source in col_names:  # see if this for loop is really necessary
        data_dicts = {}
        data_dicts["df"] = {}
        data_dicts["preprocess"] = data_dict["preprocess"]
        # TODO! if you add a new key upstream, this will lose it

        for split in data_dict["df"].keys():
            data_dicts["df"][split] = {}
            for key, split_dict in data_dict["df"][split].items():
                if key != "data":
                    # copy as it is, the other dicts
                    assert isinstance(split_dict, dict), (
                        "This should be a dict, not {}".format(type(split_dict))
                    )
                    data_dicts["df"][split][key] = split_dict
                if key == "labels":
                    if "data" not in data_dicts["df"][split].keys():
                        data_dicts["df"][split]["data"] = {}
                    # Human annotated ground truth for missing values, you are evaluating
                    # on how well the model(s) can impute these
                    data_dicts["df"][split]["data"]["mask"] = add_array_to_dict(
                        array_to_add=data_dict["df"][split][key][mask_col],
                        astype="int",
                        key=key,
                    )
                if key == "data":
                    if "data" not in data_dicts["df"][split].keys():
                        data_dicts["df"][split]["data"] = {}
                    # Set the "col_name" to the column name of the pupil signal, e.g.
                    # "pupil_gt" or "pupil_raw", "pupil_orig that you wish to train the model with
                    data_dicts["df"][split]["data"]["X"] = add_array_to_dict(
                        array_to_add=data_dict["df"][split][key][gt_source], key=key
                    )

                    # This is the denoised ground truth, if you trained on the "pupil_gt" data,
                    # this will be exactly the same then
                    data_dicts["df"][split]["data"]["X_GT"] = add_array_to_dict(
                        array_to_add=data_dict["df"][split][key][gt_source], key=key
                    )

    return data_dicts


def print_mask_stats(data_dict: dict, mask_col: str) -> None:
    """Log statistics about the mask coverage for each split.

    Parameters
    ----------
    data_dict : dict
        Data dictionary containing df with mask arrays.
    mask_col : str
        Name of the mask column for logging.
    """
    for split, dict_data in data_dict["df"].items():
        mask = dict_data["data"]["mask"]
        mask_sum = np.sum(mask)
        mask_percentage = 100 * (mask_sum / mask.size)
        logger.info(
            f"Split = {split}: {mask_percentage:.2f}% of mask is True ({mask_col})"
        )


def import_data_for_flow(cfg: DictConfig, task: str) -> tuple[dict, str, dict]:
    """Import data from DuckDB and prepare it for the processing flow.

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary.
    task : str
        Task name ("outlier_detection" or "imputation").

    Returns
    -------
    tuple
        Tuple containing (data_dicts, input_signal, data_dict) where
        data_dicts is the structured source data, input_signal is the
        column name for input data, and data_dict is the raw imported data.

    Raises
    ------
    ValueError
        If the task is unknown.
    """
    data_df = flow_import_data(cfg=cfg)
    data_dict = convert_df_to_dict(data_df=data_df, cfg=cfg)

    if task == "outlier_detection":
        # Note! outlier_detection refers to the previous task before imputation training, as in where is the data
        # that you are going to use for training the imputation models
        gt_signal = cfg["IMPUTATION_TRAINING"]["gt_col_name"]
        # for non-reconstructing methods, you need some pupil signal to mask with the pred_mask then
        input_signal = "pupil_orig_imputed"
        gt_signal_out = gt_signal
        # Remember: this task above now refers for the MLflow task, i.e.
        # for imputation training, you are getting the data from the previous outlier detection task
        mask_col = "imputation_mask"
    elif task == "imputation":
        # You could use some other key for the featurization, but atm just using the same as used
        # before in the imputation training
        gt_signal = cfg["IMPUTATION_TRAINING"]["gt_col_name"]
        gt_signal_out = gt_signal.replace("_", "-")
        # we have now two sources/models encoded in the name, i.e. imputation_model__outlier_source
        gt_signal_out = f"{gt_signal_out}__{gt_signal_out}"
        input_signal = "pupil_raw_imputed"
        # Now the imputation mask might or might not be needed when featurising PLR or getting the embeddings
        mask_col = "imputation_mask"
    else:
        logger.error(f"Unknown task: {task}")
        raise ValueError(f"Unknown task: {task}")

    data_dicts = {}
    data_dicts[gt_signal_out] = get_dict_per_col_name(
        col_names=cfg["IMPUTATION_TRAINING"]["col_name"],
        data_dict=data_dict,
        col_name=gt_signal_out,
        mask_col=mask_col,
    )

    print_mask_stats(data_dict=data_dicts[gt_signal_out], mask_col=mask_col)

    if task == "imputation":
        # rename key to match the "2-field encoding" of the sources (if you start parsing these or something)
        data_dicts[gt_signal_out] = data_dicts.pop(gt_signal_out)

    return data_dicts, input_signal, data_dict


def check_combination(source_data: dict, source_name: str, split: str) -> None:
    """Validate that X, mask, and time arrays have consistent dimensions.

    Parameters
    ----------
    source_data : dict
        Dictionary of source data.
    source_name : str
        Name of the source to check.
    split : str
        Split name to check.

    Raises
    ------
    AssertionError
        If array dimensions are inconsistent.
    """
    assert (
        source_data[source_name]["df"][split]["data"]["X"].shape[0]
        == source_data[source_name]["df"][split]["data"]["mask"].shape[0]
    ), "X and mask have different number of samples"
    assert (
        source_data[source_name]["df"][split]["data"]["X"].shape[0]
        == source_data[source_name]["df"][split]["time"]["time"].shape[0]
    ), (
        "X and time have different number of time points, "
        "outlier detecion data had {} samples, and {} time points"
    ).format(
        source_data[source_name]["df"][split]["data"]["X"].shape[0],
        source_data[source_name]["df"][split]["time"]["time"].shape[0],
    )


def check_gt_and_X(source_data: dict, source_name: str, split: str) -> None:
    """Check if X and X_GT arrays are identical and warn if so.

    Parameters
    ----------
    source_data : dict
        Dictionary of source data.
    source_name : str
        Name of the source to check.
    split : str
        Split name for logging.
    """
    model_data = source_data[source_name]
    for split, split_dict in model_data["df"].items():
        data_dict = split_dict["data"]
        X = data_dict[
            "X"
        ]  # could be either "pupil_gt" (not recommended), or coming from the previous outlier detection
        # i.e. reconstructed "pupil_orig" possibly with a lot of glitch still around
        X_GT = data_dict["X_GT"]  # from "pupil_gt"
        assert isinstance(X, np.ndarray), "X should be a numpy array"
        assert isinstance(X_GT, np.ndarray), "X_GT should be a numpy array"
        if np.all(X == X_GT):
            logger.warning(
                "Your X and Ground truth seem to be same, is this how you want things to be?"
            )
            logger.warning(f"source_name={source_name}, split={split}")


def add_CI_to_data_dicts(data_dicts: dict) -> dict:
    """Add placeholder confidence interval arrays to data dictionaries.

    The featurization script assumes CI arrays exist, so this adds NaN-filled
    arrays where they are missing.

    Parameters
    ----------
    data_dicts : dict
        Data dictionaries to add CI arrays to.

    Returns
    -------
    dict
        Updated data dictionaries with CI_pos and CI_neg arrays.
    """
    # Featurization script assumes that you have something here
    logger.info("Adding CI to data dicts")
    for pupil_col in data_dicts.keys():
        for split in data_dicts[pupil_col]["df"].keys():
            array_tmp = np.ones_like(data_dicts[pupil_col]["df"][split]["data"]["X"])
            array_tmp[:] = np.nan
            if "CI_pos" not in data_dicts[pupil_col]["df"][split]["data"].keys():
                data_dicts[pupil_col]["df"][split]["data"]["CI_pos"] = array_tmp
            if "CI_neg" not in data_dicts[pupil_col]["df"][split]["data"].keys():
                data_dicts[pupil_col]["df"][split]["data"]["CI_neg"] = array_tmp

    return data_dicts


def add_mlflow_dict_to_sources(sources: dict, mlflow_dict: Optional[dict]) -> dict:
    """Add MLflow run information to each source dictionary.

    Parameters
    ----------
    sources : dict
        Dictionary of source data.
    mlflow_dict : dict or None
        Dictionary mapping source names to MLflow run info.

    Returns
    -------
    dict
        Updated sources dictionary with mlflow key added to each source.
    """
    for source_name in sources.keys():
        if mlflow_dict is not None:
            if source_name in mlflow_dict.keys():
                sources[source_name]["mlflow"] = mlflow_dict[source_name]
            else:
                # e.g. the "pupil_gt" do not come from the MLflow, but from the DuckDB
                sources[source_name]["mlflow"] = None
        else:
            sources[source_name]["mlflow"] = None
    return sources


def check_sources(sources: dict) -> None:
    """Quality check all source data for NaN values.

    Parameters
    ----------
    sources : dict
        Dictionary of source data to check.

    Raises
    ------
    ValueError
        If any source contains NaN values in its data arrays.
    """
    logger.info("Checking quality (QA) of the sources")
    for source in sources.keys():
        for split in sources[source]["df"].keys():
            for name, data_array in sources[source]["df"][split]["data"].items():
                no_nan = np.isnan(data_array).sum()
                if no_nan > 0:
                    logger.error(
                        f"source_name={source}, split={split}, var_name={name}, no_nan={no_nan}"
                    )
                    raise ValueError(
                        f"source_name={source}, split={split}, var_name={name}, no_nan={no_nan}"
                    )


def combine_source_with_data_dicts(
    source_data: Optional[dict],
    data_dicts_for_source: dict,
    mlflow_dict: Optional[dict],
    cfg: DictConfig,
    task: str,
    input_signal: str,
    data_dict: dict,
) -> dict:
    """Combine MLflow source data with the base data dictionary template.

    Merges reconstruction/mask arrays from MLflow runs with the full data
    structure (time, metadata, etc.) from the DuckDB import.

    Parameters
    ----------
    source_data : dict or None
        Dictionary of source data from MLflow runs.
    data_dicts_for_source : dict
        Base data dictionary template from DuckDB import.
    mlflow_dict : dict
        Dictionary mapping source names to MLflow run info.
    cfg : DictConfig
        Configuration dictionary.
    task : str
        Task name ("outlier_detection" or "imputation").
    input_signal : str
        Column name for input data when no reconstruction available.
    data_dict : dict
        Raw imported data dictionary for fallback values.

    Returns
    -------
    dict
        Combined sources dictionary with full data structure for each source.

    Notes
    -----
    Expected data_dict_template structure:
        df: dict
            train: dict
                time: dict
                data: dict
                labels: dict
                light: dict
                metadata: dict
            test: dict
                same as train
        preprocess: dict
            standardization: dict
    """
    # e.g. "pupil_gt" or "pupil_gt" and "pupil_raw"
    no_of_data_dicts_for_source = len(data_dicts_for_source)
    assert no_of_data_dicts_for_source == 1, (
        "Implement more input data sources if you feel like"
    )
    pupil_col = list(data_dicts_for_source.keys())[0]
    data_dict_template = data_dicts_for_source[pupil_col]

    # Change the 'data' of the template based on each source (outlier detection model)
    if source_data is not None:
        for i, (source_name, source_dict) in enumerate(
            tqdm(source_data.items(), desc="Getting data sources")
        ):
            logger.debug(
                "Picking data for source = {} (#{}/{})".format(
                    source_name, i + 1, len(source_data)
                )
            )
            dict_tmp = deepcopy(data_dict_template)
            for split, split_dict in source_dict.items():
                if "X" in split_dict:
                    # with non-ensemble methods, you typically have the input here as well
                    if np.all(np.isnan(split_dict["X"])):
                        # for sure you don't have any reconstruction done and you need to pick "orig_data
                        # This is true from non-reconstructing methods like LOF, OneClassSVM, Prophet, etc.
                        # We don't have any prediction so we just use the original data, that you can then mask
                        # with the pred_mask returned by these "simple methods"
                        dict_tmp["df"][split]["data"]["X"] = data_dict["df"][split][
                            "data"
                        ][input_signal]
                        if split == list(source_dict.keys())[0]:
                            logger.debug(
                                "No reconstruction from {}, "
                                'using "{}" pupil column as the original data'.format(
                                    source_name, input_signal
                                )
                            )
                        # import matplotlib.pyplot as plt
                        # plt.plot(dict_tmp["df"][split]["data"]["X"][0,:])
                        # plt.show()
                    else:
                        if not cfg["IMPUTATION_TRAINING"]["use_orig_data"]:
                            # This would come from the outlier detection model, e.g. MOMENT/TimesNet reconstructs
                            # the data (reconstructs from the "outlier_test/train" which then are the "pupil_raw")
                            # So setting "use_orig_data = False" you are having realistic evaluation of processing steps
                            dict_tmp["df"][split]["data"]["X"] = split_dict["X"]
                        else:
                            # Otherwise, use the data vector from the original data (DuckDB)
                            # i.e. "pupil_gt" that you mask with the outlier detection results
                            # You could use this in isolation for testing, but the "pupil_gt" is human-annotated ground truth
                            # that you won't be having access in "real-world" where you want an automatic algorithm and don't
                            # necessarily have the time for proofreading
                            if not np.all(np.isnan(split_dict["X"])):
                                logger.warning(
                                    "You are using the original non-reconstructed data "
                                    "even though this is not coming from a simple method (i.e. all values NaN)"
                                )

                    dict_tmp["df"][split]["data"]["mask"] = split_dict["mask"].astype(
                        int
                    )
                    if task == "imputation":
                        dict_tmp["df"][split]["data"]["CI_pos"] = split_dict["CI_pos"]
                        dict_tmp["df"][split]["data"]["CI_neg"] = split_dict["CI_neg"]
                else:
                    # only update the mask with the ensemble methods
                    dict_tmp["df"][split]["data"]["mask"] = split_dict["mask"].astype(
                        int
                    )

            source_data[source_name] = dict_tmp
            check_combination(source_data, source_name, split)
            check_gt_and_X(source_data, source_name, split)

    else:
        logger.debug("No source data, using only the original data as the source")

    if task == "imputation":
        data_dicts_for_source = add_CI_to_data_dicts(data_dicts=data_dicts_for_source)

    if source_data is not None:
        # Sort the sources
        source_data = dict(sorted(source_data.items()))
        sources = {**data_dicts_for_source, **source_data}
    else:
        sources = data_dicts_for_source

    # add the MLflow dict to the sources, if you need to trace later where the source was from
    sources = add_mlflow_dict_to_sources(sources, mlflow_dict)

    # quality checking
    # check_sources(sources)

    return sources


def define_sources_for_flow(
    prev_experiment_name: str, cfg: DictConfig, task: str = "outlier_detection"
) -> dict:
    """Define all data sources for a processing flow from previous MLflow experiments.

    Main entry point for loading source data. Combines:
    1. Best runs from the previous MLflow experiment
    2. Original ground truth data from DuckDB

    Parameters
    ----------
    prev_experiment_name : str
        Name of the previous MLflow experiment to get runs from.
    cfg : DictConfig
        Configuration dictionary.
    task : str, optional
        Task name for determining data source type, by default "outlier_detection".

    Returns
    -------
    dict
        Dictionary of all source data, including both MLflow and ground truth sources.
    """
    logger.debug("Defining sources for the flow = {}".format(task))

    # Get the best runs from the previous experiment
    mlflow_runs = get_previous_best_mlflow_runs(
        experiment_name=prev_experiment_name, cfg=cfg, task=task
    )

    # Get the data from the mlflow runs to be used in the flow (as in the imputation)
    if mlflow_runs is not None:
        source_data, mlflow_dict = get_source_data(mlflow_runs, cfg, task=task)
    else:
        source_data, mlflow_dict = None, None

    # Get the original ("ground truth") data annotated by the human
    # That you will use as the BASELINE against the various outlier detection
    # and imputation methods
    data_dicts_for_source, input_signal, data_dict = import_data_for_flow(cfg, task)

    # Now the MLflow source data (outputs from the outlier detection does not have the extra keys from
    # the imported data, so we copy those to source daata so that all the dictionaries have similar contents
    logger.info("Combine sources with data dicts")
    sources = combine_source_with_data_dicts(
        source_data,
        data_dicts_for_source,
        mlflow_dict,
        cfg,
        task,
        input_signal,
        data_dict,
    )

    logger.info("Total of {} sources for processing".format(len(sources)))

    return sources
