import copy

import numpy as np
import pandas as pd
from omegaconf import DictConfig, open_dict
import mlflow
from loguru import logger

from src.log_helpers.mlflow_artifacts import (
    get_best_run_of_pd_dataframe,
    get_imputation_results_from_mlflow,
    threshold_filter_run,
)
from src.log_helpers.retrain_or_not import if_recreate_ensemble


def get_unique_models_from_best_runs(best_runs: pd.DataFrame) -> list:
    """
    Get the unique models from the best runs
    """
    models = []
    model_cfg_names = best_runs["tags.mlflow.runName"].unique()
    for model in model_cfg_names:
        logger.debug(f"Model cfg: {model}")
        model_architecture = model.split("_")[0]
        models.append(model_architecture)

    return list(set(models))


def get_best_run_of_the_model(
    best_runs: pd.DataFrame,
    model: str,
    cfg: DictConfig,
    best_metric_cfg: DictConfig,
    task: str,
    include_all_variants: bool = False,
) -> pd.Series:
    def parse_run_namme_for_model_name(run_col: pd.Series) -> pd.Series:
        model_names = []
        for run_name in run_col:
            model_name = run_name.split("_")[0]
            model_names.append(model_name)
        return model_names

    # parse run name to get the model name -> easier boolean indexing
    def add_parsed_run_name_to_df(
        run_col: pd.Series, best_runs: pd.DataFrame
    ) -> pd.DataFrame:
        model_names = parse_run_namme_for_model_name(run_col)
        best_runs["model_name"] = model_names
        return best_runs

    best_runs = add_parsed_run_name_to_df(
        run_col=best_runs["tags.mlflow.runName"], best_runs=best_runs
    )

    model_best_runs = best_runs[best_runs["model_name"] == model]
    try:
        model_best_run, best_metric = get_best_run_of_pd_dataframe(
            model_best_runs,
            cfg,
            best_metric_cfg,
            task,
            model,
            include_all_variants=include_all_variants,
        )
        logger.debug(
            "Best run_name for model {}: {}".format(
                model, best_runs["tags.mlflow.runName"]
            )
        )
    except Exception as e:
        logger.error(f"Could not get best run for model {model} with error: {e}")
        raise e

    return model_best_run, best_metric


def exclude_ensembles_from_mlflow_runs(best_runs: pd.DataFrame):
    # you do not want to get already existing ensembled models, but only the submodels
    logger.info('Excluding runs with "ensemble" in the name')
    if best_runs.shape[0] > 0:
        best_runs = best_runs[
            ~best_runs["tags.mlflow.runName"].str.contains("ensemble")
        ]
    else:
        best_runs = None
    return best_runs


def exclude_imputation_ensembles_from_mlflow_runs(best_runs: pd.DataFrame):
    runs_out = pd.DataFrame()
    for idx, row in best_runs.iterrows():
        run_name = row["tags.mlflow.runName"]
        model_name, anomaly_source = parse_imputation_run_name_for_ensemble(run_name)
        if "ensemble" not in model_name:
            runs_out = pd.concat([runs_out, pd.DataFrame(row).T])
        else:
            logger.info("Not recomputing for model_name = {}".format(model_name))

    return runs_out


def keep_only_imputations_from_anomaly_ensembles(best_runs):
    runs_out = pd.DataFrame()
    for idx, row in best_runs.iterrows():
        run_name = row["tags.mlflow.runName"]
        model_name, anomaly_source = parse_imputation_run_name_for_ensemble(run_name)
        if "ensemble" in anomaly_source:
            runs_out = pd.concat([runs_out, pd.DataFrame(row).T])

    return runs_out


def remove_worst_model(best_unique_models, best_metrics, best_metric_cfg: DictConfig):
    if best_metric_cfg["direction"] == "DESC":
        # remove the lowest value when largest value is the best
        idx = np.nanargmin(best_metrics)
    elif best_metric_cfg["direction"] == "ASC":
        # remove the highest value when lowest value is the best
        idx = np.nanargmax(best_metrics)
    else:
        logger.error(f"Direction {best_metric_cfg['direction']} not implemented")
        raise NotImplementedError(
            f"Direction {best_metric_cfg['direction']} not implemented"
        )

    model_keys = list(best_unique_models.keys())
    model_to_remove = model_keys[idx]
    logger.info(f"Removing the worst model: {model_to_remove}")
    best_unique_models.pop(model_to_remove)

    return best_unique_models


def exclude_pupil_orig_imputed(best_unique_models: dict, best_metrics: list):
    # Don't include the models trained on "pupil_orig_imputed" data, use just the 'gt' ones
    model_names = list(best_unique_models.keys())
    metrics_out = []
    for metric, model in zip(best_metrics, model_names):
        if "orig" in model:
            logger.info(
                'Removing "pupil_orig_imputed" run from ensemble, model: {}'.format(
                    model
                )
            )
            best_unique_models.pop(model)
        else:
            metrics_out.append(metric)
    return best_unique_models, metrics_out


def get_anomaly_runs(
    best_runs: pd.Series,
    best_metric_cfg: DictConfig,
    cfg: DictConfig,
    task: str,
    return_odd_number_of_models: bool = False,
    exclude_orig_data: bool = True,
    include_all_variants: bool = False,
):
    unique_models: list = get_unique_models_from_best_runs(best_runs)
    best_unique_models = {}
    best_metrics = []
    for model in unique_models:
        logger.debug("Getting the best run for model: {}".format(model))
        best_run, best_metric = get_best_run_of_the_model(
            best_runs,
            model,
            cfg,
            best_metric_cfg=best_metric_cfg,
            task=task,
            include_all_variants=include_all_variants,
        )
        if best_run is not None:
            best_unique_models[model] = best_run
            best_metrics.append(best_metric)

    if len(best_metrics) == 0:
        logger.error('No best runs were added? glitch somewhere? Ensemble thresholding too high?')
        # e.g. cfg['OUTLIER_DETECTION']['best_metric']['ensemble_quality_threshold']
        raise RuntimeError('No best runs were added? glitch somewhere?')

    if not include_all_variants:
        if exclude_orig_data:
            best_unique_models, best_metrics = exclude_pupil_orig_imputed(
                best_unique_models, best_metrics
            )

    if return_odd_number_of_models:
        if len(best_unique_models) % 2 == 0:
            logger.info("Returning odd number of models for anomaly detection")
            best_unique_models = remove_worst_model(
                best_unique_models, best_metrics, best_metric_cfg=best_metric_cfg
            )

    # When you include all the variants, the dataframe in each submodel might contain multiple rows,
    # create new key for each row so that downstream code works
    if include_all_variants:
        best_unique_models = create_new_keys_for_all_variants(best_unique_models)

    return best_unique_models


def create_new_keys_for_all_variants(best_unique_models):
    logger.info("Creating new keys for all models (all variants)")
    best_unique_models_out = {}
    for submodel, model_df in best_unique_models.items():
        no_rows = model_df.shape[0]
        if no_rows > 1:
            for idx, row in model_df.iterrows():
                submodel_name = row["tags.mlflow.runName"]
                best_unique_models_out[submodel_name] = pd.DataFrame(row).T
        else:
            best_unique_models_out[submodel] = model_df
    logger.info("-> {} of model variants in total".format(len(best_unique_models_out)))
    logger.info(list(best_unique_models_out.keys()))

    return best_unique_models_out


def get_best_imputation_col_name(best_metric_cfg: DictConfig) -> str:
    split = best_metric_cfg["split"]
    metric_name = best_metric_cfg["string"]
    return f"metrics.{split}/{metric_name}"


def get_best_imputation_model_per_run_name(runs, best_metric_cfg):
    if runs.shape[0] > 1:
        # e.g. 'metrics.test/mae'
        col_name = get_best_imputation_col_name(best_metric_cfg)
        if best_metric_cfg["direction"] == "DESC":
            runs = runs.sort_values(by=col_name, ascending=False)
        elif best_metric_cfg["direction"] == "ASC":
            runs = runs.sort_values(by=col_name, ascending=True)
        else:
            logger.error(f"Direction {best_metric_cfg['direction']} not implemented")
            raise NotImplementedError(
                f"Direction {best_metric_cfg['direction']} not implemented"
            )
        return runs.iloc[[0]]
    else:
        return runs


def get_best_unique_imputation_models(
    best_runs: pd.Series, best_metric_cfg: DictConfig, cfg: DictConfig, task: str
):
    best_unique_runs = pd.DataFrame()
    unique_run_names = best_runs["tags.mlflow.runName"].unique()
    for run_name in unique_run_names:
        # if you have ran the same config multiple times, pick the best run, most likely you
        # just have one copy per run_name
        runs: pd.DataFrame = best_runs[best_runs["tags.mlflow.runName"] == run_name]
        best_run = get_best_imputation_model_per_run_name(runs, best_metric_cfg)
        col_name = get_best_imputation_col_name(best_metric_cfg)
        best_run = threshold_filter_run(best_run, col_name, best_metric_cfg)
        best_unique_runs = pd.concat([best_unique_runs, best_run])

    return best_unique_runs


def parse_imputation_run_name_for_ensemble(run_name: str) -> str:
    fields = run_name.replace("___", "__").split("__")
    if len(fields) == 2:
        model_name, anomaly_source = run_name.split("__")
    elif len(fields) == 3:
        model_name, anomaly_source, extra = run_name.split("__")
        anomaly_source += extra
    else:
        logger.error("Unknown parsing for run name: {}".format(run_name))
        raise ValueError("Unknown parsing for run name: {}".format(run_name))

    return model_name, anomaly_source


def filter_runs_for_gt(
    best_runs,
    best_metric_cfg,
    cfg,
    task,
    return_best_gt: bool = False,
    gt_on: str = "anomaly",
):
    if task == "classification":
        if return_best_gt and gt_on is None:
            logger.debug("When you want gt anomaly AND gt imputation")

    best_runs_filtered = pd.DataFrame()
    # iterate through dataframe rows
    for index, row in best_runs.iterrows():
        run_name = row["tags.mlflow.runName"]

        if task == "imputation":
            model_name, anomaly_source = parse_imputation_run_name_for_ensemble(
                run_name
            )

            on_gt: bool = anomaly_source.startswith("pupil_gt_")
            if return_best_gt:
                if on_gt:
                    logger.debug(f"Keeping the run with GT: {run_name}")
                    best_runs_filtered = pd.concat(
                        [best_runs_filtered, row.to_frame().T]
                    )
            else:
                if not on_gt:
                    logger.debug(f"Keeping the run without GT: {run_name}")
                    best_runs_filtered = pd.concat(
                        [best_runs_filtered, row.to_frame().T]
                    )

        elif task == "classification":
            fields = run_name.split("__")
            try:
                if len(fields) == 2:
                    # ensembling broke the naming convention
                    model_name, imputation_source = run_name.split("__")
                    anomaly_source = imputation_source
                    feature_name = 'simple1.0'
                    logger.warning('Hard-coded feature name "{}" for multi-classifier ensemble'.format(feature_name))
                elif len(fields) == 4:
                    model_name, feature_name, imputation_source, anomaly_source = (
                        run_name.split("__")
                    )
            except Exception as e:
                logger.error(
                    "Could not get the run_name = {} with error: {}".format(run_name, e)
                )
                raise e

            anomaly_on_gt: bool = anomaly_source.startswith("pupil-gt")
            imputation_on_gt: bool = imputation_source.startswith("pupil-gt")
            on_gt: bool = anomaly_on_gt and imputation_on_gt
            some_gt: bool = anomaly_on_gt or imputation_on_gt

            if return_best_gt:
                if on_gt and gt_on is None:
                    logger.debug(
                        f"Keeping the run with GT (both Anomaly and Imputation): {run_name}"
                    )
                    best_runs_filtered = pd.concat(
                        [best_runs_filtered, row.to_frame().T]
                    )
                elif some_gt and not on_gt:
                    if gt_on == "anomaly" and anomaly_on_gt and not imputation_on_gt:
                        logger.debug(f"Keeping the run with Anomaly GT: {run_name}")
                        best_runs_filtered = pd.concat(
                            [best_runs_filtered, row.to_frame().T]
                        )
                    elif (
                        gt_on == "imputation" and imputation_on_gt and not anomaly_on_gt
                    ):
                        logger.debug(f"Keeping the run with Imputation GT: {run_name}")
                        best_runs_filtered = pd.concat(
                            [best_runs_filtered, row.to_frame().T]
                        )

            else:
                if not some_gt:
                    logger.debug(f"Keeping the run without GT: {run_name}")
                    best_runs_filtered = pd.concat(
                        [best_runs_filtered, row.to_frame().T]
                    )
        else:
            logger.error(f"Task {task} not implemented yet")
            raise NotImplementedError(f"Task {task} not implemented yet")

    if best_runs_filtered.shape[0] == 0:
        logger.warning("No runs after filtering!")

    return best_runs_filtered


def filter_for_detection(detection_filter_reject, best_runs_out):
    logger.info("Rejecting runs with zeroshot in the name (and use the finetuned)")
    runs_out = pd.DataFrame()
    for index, row in best_runs_out.iterrows():
        run_name = row["tags.mlflow.runName"]
        if detection_filter_reject in run_name:
            logger.debug(f"Rejecting the run: {run_name}")
        else:
            runs_out = pd.concat([runs_out, row.to_frame().T])

    return runs_out


def get_non_moment_models(best_runs_out):
    runs_out = pd.DataFrame()
    for index, row in best_runs_out.iterrows():
        run_name = row["tags.mlflow.runName"]
        model_name, _ = parse_imputation_run_name_for_ensemble(run_name)
        if "MOMENT" not in model_name:
            runs_out = pd.concat([runs_out, row.to_frame().T])
    return runs_out


def get_best_moment(best_metric_cfg, runs_moment):
    col_name = get_best_imputation_col_name(best_metric_cfg)
    if best_metric_cfg["direction"] == "DESC":
        runs_moment = runs_moment.sort_values(by=col_name, ascending=False)
    elif best_metric_cfg["direction"] == "ASC":
        runs_moment = runs_moment.sort_values(by=col_name, ascending=True)
    else:
        logger.error(f"Direction {best_metric_cfg['direction']} not implemented")
        raise NotImplementedError(
            f"Direction {best_metric_cfg['direction']} not implemented"
        )
    runs_moment = runs_moment.iloc[[0]]
    return runs_moment


def get_unique_sources(best_runs_out):
    unique_sources, model_names = [], []
    for index, row in best_runs_out.iterrows():
        run_name = row["tags.mlflow.runName"]
        model_name = run_name.split("_")[0]
        source = run_name.split("__")[1]
        unique_sources.append(source)
        model_names.append(model_name)
    return list(set(unique_sources))


def keep_moment_models(runs_source):
    df_runs = pd.DataFrame()
    for idx, row in runs_source.iterrows():
        run_name = row["tags.mlflow.runName"]
        model_name, source_name = parse_imputation_run_name_for_ensemble(run_name)
        if model_name.startswith("MOMENT"):
            df_runs = pd.concat([df_runs, row.to_frame().T])

    return df_runs


def get_best_moments_per_source(best_runs_out, best_metric_cfg):
    runs_moment = pd.DataFrame()
    unique_sources = get_unique_sources(best_runs_out)
    for unique_source in unique_sources:
        runs_source = best_runs_out[
            best_runs_out["tags.mlflow.runName"].str.contains(unique_source)
        ]
        runs_moment_as_model = keep_moment_models(runs_source)
        if runs_moment_as_model.shape[0] > 0:
            runs_moment_per_source = get_best_moment(
                best_metric_cfg, runs_moment_as_model
            )
            runs_moment = pd.concat([runs_moment, runs_moment_per_source])
        else:
            runs_moment = None
    return runs_moment


def get_best_moment_variant(best_runs_out, best_metric_cfg, return_best_gt):
    logger.info("Getting best MOMENT variant")
    non_moment_runs = get_non_moment_models(best_runs_out)
    if return_best_gt:
        # easier task as just filter MOMENT as they all have the same source
        runs_moment = best_runs_out[
            best_runs_out["tags.mlflow.runName"].str.contains("MOMENT")
        ]
        runs_moment = get_best_moment(best_metric_cfg, runs_moment)
    else:
        runs_moment = get_best_moments_per_source(best_runs_out, best_metric_cfg)

    if runs_moment is None:
        runs_out = non_moment_runs
    elif non_moment_runs is None:
        runs_out = runs_moment
    else:
        runs_out = pd.concat([non_moment_runs, runs_moment])

    return runs_out


def get_imputation_runs(
    best_runs,
    best_metric_cfg,
    cfg,
    task,
    return_best_gt,
    detection_filter_reject: str = "zeroshot",
):
    # Keep unique run_names
    best_unique_runs = get_best_unique_imputation_models(
        best_runs, best_metric_cfg, cfg, task
    )

    # Whether to keep only GT or kick out the GT
    best_runs_out = filter_runs_for_gt(
        best_unique_runs, best_metric_cfg, cfg, task, return_best_gt
    )

    if best_runs_out.shape[0] == 0:
        logger.warning("No runs after filtering!")
        return None

    # e.g. MOMENT has finetuned and zeroshot models, and let's just use the finetuned models here in the ensemble
    best_runs_out = filter_for_detection(detection_filter_reject, best_runs_out)

    # Keep only the best MOMENT model
    best_runs_out_final = get_best_moment_variant(
        best_runs_out, best_metric_cfg, return_best_gt
    )

    return best_runs_out_final


def get_best_unique_classification_models(
    best_runs: pd.Series, best_metric_cfg: DictConfig, cfg: DictConfig, task: str
):
    best_unique_runs = pd.DataFrame()
    unique_run_names = best_runs["tags.mlflow.runName"].unique()
    for run_name in unique_run_names:
        # if you have ran the same config multiple times, pick the best run, most likely you
        # just have one copy per run_name
        runs: pd.DataFrame = best_runs[best_runs["tags.mlflow.runName"] == run_name]
        # the same logic for classification and imputation:
        best_run = get_best_imputation_model_per_run_name(runs, best_metric_cfg)
        best_unique_runs = pd.concat([best_unique_runs, best_run])

    return best_unique_runs


def drop_embedding_cls_runs(best_runs_out):

    runs_out = pd.DataFrame()
    for idx, row in best_runs_out.iterrows():
        run_name = row["tags.mlflow.runName"]
        if 'embedding' not in run_name:
            runs_out = pd.concat([runs_out, pd.DataFrame(row).T])

    return runs_out

def get_list_of_good_models():
    return ['TabPFN', 'TabM', 'XGBOOST', 'CATBOOST']


def keep_the_good_models(best_runs_out):
    good_models = get_list_of_good_models()
    runs_out = pd.DataFrame()
    for idx, row in best_runs_out.iterrows():
        run_name = row["tags.mlflow.runName"]
        if any(model in run_name for model in good_models):
            runs_out = pd.concat([runs_out, pd.DataFrame(row).T])

    return runs_out


def keep_cls_runs_when_both_imputation_and_outlier_are_ensemble(best_runs_out):

    runs_out = pd.DataFrame()
    for idx, row in best_runs_out.iterrows():
        run_name = row["tags.mlflow.runName"]
        fields = run_name.split('__')
        if len(fields) == 2:
            cls, imput = run_name.split('__')
            outlier = 'anomaly'
        elif len(fields) == 4:
            cls, feat, imput, outlier = run_name.split('__')
        else:
            logger.error('Unknown number of fields in run_name, n = {}'.format(len(fields)))
            raise ValueError('Unknown number of fields in run_name, n = {}'.format(len(fields)))

        if 'ensemble' in imput:
            if outlier == 'anomaly': # "anomaly_ensemble"
                runs_out = pd.concat([runs_out, pd.DataFrame(row).T])

    return runs_out


def get_classification_runs(
    best_runs,
    best_metric_cfg,
    cfg,
    task,
    return_best_gt: bool = True,
    gt_on: str = "anomaly",
    return_only_ensembled_inputs: bool = False
):
    # Keep unique run_names
    best_unique_runs = get_best_unique_classification_models(
        best_runs, best_metric_cfg, cfg, task
    )

    # Whether to keep only GT or kick out the GT
    best_runs_out = filter_runs_for_gt(
        best_unique_runs, best_metric_cfg, cfg, task, return_best_gt, gt_on=gt_on
    )

    # Kick out embeddings
    best_runs_out = drop_embedding_cls_runs(best_runs_out)

    # Keep just the "good models", not the sanity check ones loke LogisticRegression
    best_runs_out = keep_the_good_models(best_runs_out)

    if return_only_ensembled_inputs:
        best_runs_out = keep_cls_runs_when_both_imputation_and_outlier_are_ensemble(best_runs_out)


    return best_runs_out


def get_used_models_from_mlflow(
    experiment_name: str,
    cfg: DictConfig,
    task: str = "anomaly_detection",
    exclude_ensemble: bool = True,
    return_odd_number_of_models: bool = False,
    return_best_gt: bool = False,
    return_anomaly_ensembles: bool = False,
    gt_on: str = None,
    include_all_variants: bool = False,
    return_all_runs: bool = False,
    return_only_ensembled_inputs: bool = False,
) -> dict:
    if task == "anomaly_detection":
        best_metric_cfg = cfg["OUTLIER_DETECTION"]["best_metric"]
    elif task == "imputation":
        best_metric_cfg = cfg["IMPUTATION_METRICS"]["best_metric"]
    elif task == "classification":
        best_metric_cfg = cfg["CLASSIFICATION_SETTINGS"]["BEST_METRIC"]
    else:
        logger.error(f"Task {task} not implemented yet")
        raise NotImplementedError(f"Task {task} not implemented yet")

    # best_runs contain all the hyperparameter combinations
    best_runs: pd.Series = mlflow.search_runs(experiment_names=[experiment_name])
    if not return_all_runs:
        if best_runs.shape[0] > 0:
            if exclude_ensemble:
                if task == "anomaly_detection":
                    best_runs = exclude_ensembles_from_mlflow_runs(best_runs)
                elif task == "imputation":
                    best_runs = exclude_imputation_ensembles_from_mlflow_runs(best_runs)
                    if return_anomaly_ensembles:
                        best_runs = keep_only_imputations_from_anomaly_ensembles(
                            best_runs
                        )
                    else:
                        best_runs = exclude_ensembles_from_mlflow_runs(best_runs)
            else:
                logger.info(
                    "Including runs with 'ensemble' in the name (i.e. when you want to featurize)"
                )
            if task == "anomaly_detection":
                best_unique_models = get_anomaly_runs(
                    best_runs,
                    best_metric_cfg,
                    cfg,
                    task,
                    return_odd_number_of_models,
                    include_all_variants=include_all_variants,
                )

            elif task == "imputation":
                # Get two types of ensemble:
                # 1) Imputation models trained on the gt (sets baseline for ensemble performance)
                # return_best_gt=True
                # 2) All possible imputation models excluding the gt (this is the "real-world" scenario as you might
                #    not have the gt available)
                # return_best_gt=False
                best_unique_models = get_imputation_runs(
                    best_runs, best_metric_cfg, cfg, task, return_best_gt
                )
            elif task == "classification":
                best_unique_models = get_classification_runs(
                    best_runs, best_metric_cfg, cfg, task, return_best_gt, gt_on=gt_on,
                    return_only_ensembled_inputs=return_only_ensembled_inputs
                )
            else:
                logger.error(f"Task {task} not implemented yet")
                raise NotImplementedError(f"Task {task} not implemented yet")

            return best_unique_models

        else:
            logger.warning(
                "Did not find previous runs, experiment name = {}".format(
                    experiment_name
                )
            )
            return {}
    else:
        logger.info("Returning all runs")
        if exclude_ensemble:
            best_runs = exclude_imputation_ensembles_from_mlflow_runs(best_runs)
        return best_runs


def ensemble_the_imputation_output_dicts(
    results_per_model: dict, ensembled_outputs: dict, i: int, submodel: str
) -> dict:
    if len(ensembled_outputs) == 0:
        ensembled_outputs = results_per_model

    for split in results_per_model["imputation"].keys():
        for split_key in results_per_model["imputation"][split].keys():
            # You are adding input (3d array, no_subjects, no_timepoints, no_features)
            # to the output (4d array, 4th dimension contains the submodel)
            input_array = results_per_model["imputation"][split][split_key][
                "imputation_dict"
            ]["imputation"]["mean"]
            output_array = ensembled_outputs["imputation"][split][split_key][
                "imputation_dict"
            ]["imputation"]["mean"]
            # TODO! CIpos/CIneg

            input_array = input_array[
                :, :, :, None
            ]  # add 4th dimension for the first submodel
            if len(output_array.shape) == 3:
                output_array = input_array  # for first submodel these are the same, or 0th 3rd axis
                ensembled_outputs["imputation"][split][split_key]["imputation_dict"][
                    "imputation"
                ]["mean"] = output_array

            elif len(output_array.shape) == 4:
                assert (
                    input_array.shape[0] == output_array.shape[0]
                ), "Number of shapes do not match"
                output_array = np.concatenate((output_array, input_array), axis=3)
                ensembled_outputs["imputation"][split][split_key]["imputation_dict"][
                    "imputation"
                ]["mean"] = output_array
                logger.debug("Ensemble output shape: {}".format(output_array.shape))
            else:
                logger.error(f"Input array shape is not 4d, but {input_array.shape}")
                raise ValueError(
                    f"Input array shape is not 4d, but {input_array.shape}"
                )

    # Drop unwanted keys
    ensembled_outputs.pop("train", None)
    ensembled_outputs.pop("timing", None)
    ensembled_outputs.pop("model_info", None)
    ensembled_outputs.pop("mlflow", None)

    return ensembled_outputs


def compute_ensemble_stats(
    ensembled_outputs: dict, ensemble_name: str, n: int, cfg: DictConfig
):
    def compute_numpy_array_stats(input_array: np.ndarray):
        assert len(input_array.shape) == 4, "Input array is not 4d"
        dict_out = {}
        dict_out["mean"] = np.nanmean(input_array, axis=3)
        assert len(dict_out["mean"].shape) == 3, "Mean array is not 3d"
        dict_out["std"] = np.nanstd(input_array, axis=3)
        dict_out["n"] = input_array.shape[3]
        assert (
            dict_out["n"] == n
        ), "Number of submodels used for stats does not match the number of submodel names"
        # assuming this is now normally distributed, you could test this as well? TODO!
        dict_out["CI"] = 1.96 * dict_out["std"] / np.sqrt(input_array.shape[3])
        return dict_out

    ensembled_output = ensembled_outputs
    for split in ensembled_outputs["imputation"].keys():
        if split not in ensembled_output["imputation"]:
            ensembled_output["imputation"][split] = {}
        for split_key in ensembled_outputs["imputation"][split].keys():
            if split_key not in ensembled_output["imputation"][split]:
                ensembled_output["imputation"][split][split_key] = {}
            input_array = ensembled_outputs["imputation"][split][split_key][
                "imputation_dict"
            ]["imputation"]["mean"]
            output_dict = compute_numpy_array_stats(input_array=input_array)
            ensembled_output["imputation"][split][split_key]["imputation_dict"][
                "imputation"
            ] = output_dict

    return ensembled_output


def ensemble_the_imputation_results(
    ensemble_name: str, mlflow_ensemble: dict, cfg: DictConfig
):
    ensembled_outputs = {}
    submodel_run_names = []

    for i, submodel in enumerate(mlflow_ensemble.keys()):
        logger.info(
            f"Getting the results of the model: {submodel} (#{i+1}/{len(mlflow_ensemble.keys())})"
        )
        results_per_model = get_imputation_results_from_mlflow(
            mlflow_run=mlflow_ensemble[submodel], model_name=submodel, cfg=cfg
        )

        # quick'n'dirty save of the submodel run
        submodel_run_names.append(results_per_model["mlflow"]["run_info"]["run_name"])

        ensembled_outputs = ensemble_the_imputation_output_dicts(
            results_per_model=results_per_model,
            ensembled_outputs=ensembled_outputs,
            i=i,
            submodel=submodel,
        )
        # average the metrics from submodels, not metrics from averaged imputations! TODO!
        # ensembled_outputs['averaged_metrics']

    # Add some params (as in MLflow params)
    ensembled_outputs["params"] = {
        "model": ensemble_name,
        "submodels": submodel_run_names,
    }

    # You have now 4d Numpy arrays that you can compute whatever stats you like from all the submodels
    ensembled_output = compute_ensemble_stats(
        ensembled_outputs, ensemble_name, n=len(mlflow_ensemble.keys()), cfg=cfg
    )

    # Add the MLflow run info for the submodels
    ensembled_output["mlflow_ensemble"] = mlflow_ensemble
    # Results of how many submodels were used for computing the stats, should be as many as oyu had submodel names
    n_out = ensembled_outputs["imputation"]["train"]["gt"]["imputation_dict"][
        "imputation"
    ]["n"]
    assert len(submodel_run_names) == n_out, (
        f"It seems that you computed stats from {n_out} even though you had "
        f"{len(submodel_run_names)} submodels"
    )

    return ensembled_output


def get_ensemble_permutations(
    best_unique_models: dict, ensemble_cfg: DictConfig, cfg: DictConfig
):
    def get_ensemble_name(submodel_names, delimiter="-"):
        name_out = f"ensemble{delimiter}"
        for i, name in enumerate(submodel_names):
            if i > 0:
                name_out += delimiter + name
            else:
                name_out += name
        return name_out

    # placeholder now, when you start to have models. Like 10 models and you want to create all possible ensembles
    # with 5 submodels, return some indices here TODO!
    submodel_names = sorted(list(best_unique_models.keys()))
    ensembles = {}
    ensemble_names = [get_ensemble_name(submodel_names)]

    # TODO! placeholder now
    for name in ensemble_names:
        ensembles[name] = best_unique_models

    return ensembles


def get_imputation_results_from_for_ensembling(experiment_name: str, cfg: DictConfig):
    # Get the best hyperparameter combination of each model architecture
    best_unique_models = get_used_models_from_mlflow(experiment_name, cfg)

    if len(best_unique_models) == 0:
        logger.warning("No models found for ensembling, returning empty dict")
        return {}
    elif len(best_unique_models) == 1:
        logger.warning("Only one model found, no need for ensembling")
        return {}
    else:
        # Define the permutations of the (sub)models for the ensemble
        mlflow_ensembles = get_ensemble_permutations(
            best_unique_models=best_unique_models,
            ensemble_cfg=cfg["IMPUTATION_ENSEMBLING"],
            cfg=cfg,
        )

        # Get the imputation outputs (forward passes) of the submodels and average the responses, so you can
        # compute the ensemble metrics, and use downstream for PLR featurization and later for classification
        ensembled_output = {}
        for ensemble_name in mlflow_ensembles.keys():
            if if_recreate_ensemble(ensemble_name, experiment_name, cfg):
                ensembled_output[ensemble_name] = ensemble_the_imputation_results(
                    ensemble_name=ensemble_name,
                    mlflow_ensemble=mlflow_ensembles[ensemble_name],
                    cfg=cfg,
                )
            else:
                logger.info(
                    f"Ensemble {ensemble_name} already exists, skipping the ensemble creation"
                )

    return ensembled_output


def get_gt_imputation_labels(sources):
    labels = {}
    df = sources["pupil_gt"]["df"]
    for split in df.keys():
        labels[split] = df[split]["labels"]["imputation_mask"].astype(int)
    return labels


def get_metadata_dict_from_sources(sources: dict):
    first_source_key = list(sources.keys())[0]
    df = sources[first_source_key]["df"]
    metadata_dict = {}
    for split in df.keys():
        metadata_dict[split] = df[split]["metadata"]
    return metadata_dict


def combine_ensembles_into_one_df(best_unique_models: dict):
    df_out = pd.DataFrame()
    for ensemble_name in best_unique_models.keys():
        if isinstance(best_unique_models[ensemble_name], pd.DataFrame):
            df_out = pd.concat([df_out, best_unique_models[ensemble_name]], axis=0)
    if df_out.empty:
        return None
    else:
        return df_out


def aggregate_codes(df):
    codes = {"train": None, "test": None}
    cols = []

    for idx, row in df.iterrows():
        if row["params.codes_train"] is not None:
            train_codes = row["params.codes_train"].split(" ")  # e.g. 145 codes
            test_codes = row["params.codes_test"].split(" ")  # e.g. 63 codes
            cols.append(row["tags.mlflow.runName"])
            if codes["test"] is None:
                codes["train"] = np.array(train_codes)[:, np.newaxis]
                codes["test"] = np.array(test_codes)[:, np.newaxis]
            else:
                codes["train"] = np.concatenate(
                    (codes["train"], np.array(train_codes)[:, np.newaxis]), axis=1
                )
                codes["test"] = np.concatenate(
                    (codes["test"], np.array(test_codes)[:, np.newaxis]), axis=1
                )
        else:
            logger.warning(
                f'run_name: "{row["tags.mlflow.runName"]}" does not have codes saved'
            )

    codes["train"] = pd.DataFrame(codes["train"], columns=cols)
    codes["test"] = pd.DataFrame(codes["test"], columns=cols)

    return codes


def are_codes_the_same(df: pd.DataFrame):
    same_codes = df.eq(df.iloc[:, 0], axis=0)
    run_names = list(same_codes.columns)
    nonmatching_codes = same_codes.sum(axis=0) != same_codes.shape[0]
    all_submodels_have_same_codes = np.all(same_codes)
    if not all_submodels_have_same_codes:
        logger.error("All the submodels do not have the same codes")
        for i, nonmatch_code in enumerate(nonmatching_codes):
            if nonmatch_code:
                logger.error(" run_name = {}".format(run_names[i]))

    return all_submodels_have_same_codes


def check_codes_used(best_unique_models: dict):
    """
    Are all the models trained with same subjects
    """
    df_mlflow = combine_ensembles_into_one_df(best_unique_models)
    if df_mlflow is not None:
        codes = aggregate_codes(df_mlflow)
        for split in codes.keys():
            are_codes_the_same(df=codes[split])
        return best_unique_models
    else:
        return None


def get_grouped_classification_runs(
    best_unique_models, experiment_name: str, cfg: DictConfig, task: str
):
    # When both anomaly detection and imputation come from ground truth
    best_unique_models["pupil_gt"] = get_used_models_from_mlflow(
        experiment_name, cfg, task, return_best_gt=True, gt_on=None
    )

    # Only anomaly is ground truth
    best_unique_models["anomaly_gt"] = get_used_models_from_mlflow(
        experiment_name, cfg, task, return_best_gt=True, gt_on="anomaly"
    )

    # Kinda useless this configuration in practice?
    # best_unique_models["exclude_gt"] = get_used_models_from_mlflow(
    #     experiment_name, cfg, task, return_best_gt=False
    # )

    # When both anomaly and imputation results come from ensembles
    # i.e. you have here ensembled anomaly detection, used that as input for imputation methods,
    # and then ensembled those imputation methods, and now we are then ensembling different classifiers
    # for a "full-chain" of ensembled models
    best_unique_models["ensembled_input"] = get_used_models_from_mlflow(
        experiment_name, cfg, task, return_best_gt=False, return_only_ensembled_inputs=True
    )

    best_unique_models = check_codes_used(best_unique_models)

    return best_unique_models


def get_results_from_mlflow_for_ensembling(
    experiment_name: str, cfg: DictConfig, task: str, recompute_metrics: bool = False
):
    best_unique_models = {}
    if task == "anomaly_detection":
        # hacky way to get the granular metrics computed, note also that this does not compute granular metric
        # for all the models (non-best MOMENTs and worst model if there are even number of models)
        cfg_tmp = copy.deepcopy(cfg)
        with open_dict(cfg_tmp):
            cfg_tmp["OUTLIER_DETECTION"]["best_metric"][
                "ensemble_quality_threshold"
            ] = None

        # recompute all the metrics (and get the granular metrics that were not included in the "first pass" of
        # evaluation (during the training
        if recompute_metrics:
            best_unique_models["pupil_gt_all_variants"] = get_used_models_from_mlflow(
                experiment_name, cfg_tmp, task, include_all_variants=True
            )

        else:
            # we have only outlier mask (0 or 1), so return odd number of models to have a "winner vote" for each timepoint
            best_unique_models["pupil_gt_thresholded"] = get_used_models_from_mlflow(
                experiment_name, cfg, task, return_odd_number_of_models=True
            )

            best_unique_models["pupil_gt"] = get_used_models_from_mlflow(
                experiment_name, cfg_tmp, task, return_odd_number_of_models=True
            )

    elif task == "imputation":
        if recompute_metrics:
            # recompute the metrics
            cfg_tmp = copy.deepcopy(cfg)
            with open_dict(cfg_tmp):
                cfg_tmp["IMPUTATION_METRICS"]["best_metric"][
                    "ensemble_quality_threshold"
                ] = None
            best_unique_models["all_runs"] = get_used_models_from_mlflow(
                experiment_name, cfg_tmp, task, return_all_runs=True
            )

        else:
            best_unique_models["pupil_gt"] = get_used_models_from_mlflow(
                experiment_name, cfg, task, return_best_gt=True
            )
            best_unique_models["anomaly_ensemble"] = get_used_models_from_mlflow(
                experiment_name,
                cfg,
                task,
                return_best_gt=False,
                return_anomaly_ensembles=True,
            )
            cfg_tmp = copy.deepcopy(cfg)
            with open_dict(cfg_tmp):
                cfg_tmp["IMPUTATION_METRICS"]["best_metric"][
                    "ensemble_quality_threshold"
                ] *= cfg_tmp["IMPUTATION_METRICS"]["best_metric"][
                    "gt_exclude_multiplier"
                ]
            best_unique_models["exclude_gt"] = get_used_models_from_mlflow(
                experiment_name,
                cfg_tmp,
                task,
                return_best_gt=False,
                return_anomaly_ensembles=False,
            )

    elif task == "classification":
        best_unique_models = get_grouped_classification_runs(
            best_unique_models, experiment_name, cfg, task
        )
    else:
        logger.error(f"Task {task} not implemented yet")
        raise NotImplementedError(f"Task {task} not implemented yet")

    # Display the submodel runs of the ensemble
    if best_unique_models is not None:
        ensemble_names = list(best_unique_models.keys())
        for ensemble_name in ensemble_names:
            if best_unique_models[ensemble_name] is None:
                logger.warning(
                    f"No models found for ensemble {ensemble_name}, skipping the ensemble creation (pop out)"
                )
                best_unique_models.pop(ensemble_name)

            else:
                logger.info(
                    f"Ensemble name {ensemble_name} has {len(best_unique_models[ensemble_name])} submodels"
                )

                if isinstance(best_unique_models[ensemble_name], pd.Series):
                    for key, runs in best_unique_models[ensemble_name].items():
                        logger.info(f" {key}: {len(runs)}")

                elif isinstance(best_unique_models[ensemble_name], pd.DataFrame):
                    for idx, row in best_unique_models[ensemble_name].iterrows():
                        logger.info(f"run_name: {row['tags.mlflow.runName']}")

                logger.info("")
    else:
        # Happens when you run your demo data for example
        logger.warning('None of the ensembles had any submodules found!')
        logger.warning('OK for the demo data running case')

    return best_unique_models
