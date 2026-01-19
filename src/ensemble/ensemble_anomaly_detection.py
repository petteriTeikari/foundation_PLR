import numpy as np
import pandas as pd
from omegaconf import DictConfig
from loguru import logger
import mlflow
import os


from src.anomaly_detection.outlier_sklearn import eval_on_all_outlier_difficulty_levels
from src.log_helpers.local_artifacts import load_results_dict


def get_anomaly_results_per_model(
    model_name: str, outlier_artifacts: dict, pred_masks: dict, labels: dict, idx: int
):
    if model_name == "TimesNet" or model_name == "TimesNet-orig":
        arrays = outlier_artifacts["results_best"]["outlier_train"]["results_dict"][
            "split_results"
        ]["arrays"]
        pred_masks["train"].append(arrays["pred_mask"])
        labels["train"] = arrays["trues"]
        arrays = outlier_artifacts["results_best"]["outlier_test"]["results_dict"][
            "split_results"
        ]["arrays"]
        pred_masks["test"].append(arrays["pred_mask"])
        labels["test"] = arrays["trues"]
    elif "UniTS-Outlier" in model_name:
        arrays = outlier_artifacts["results"]["outlier_train"]["arrays"]
        pred_masks["train"].append(arrays["pred_mask"])
        labels["train"] = arrays["trues"]
        arrays = outlier_artifacts["results"]["outlier_test"]["arrays"]
        pred_masks["test"].append(arrays["pred_mask"])
        labels["test"] = arrays["trues"]
    elif model_name == "MOMENT":
        arrays = outlier_artifacts["preds"]["outlier_train"]["arrays"]
        pred_masks["train"].append(arrays["pred_mask"].astype(int))
        # labels["train"] = arrays["trues"] # No labels
        arrays = outlier_artifacts["preds"]["outlier_test"]["arrays"]
        pred_masks["test"].append(arrays["pred_mask"].astype(int))
        # labels["test"] = arrays["trues"] # No labels

    else:  # sklearn models
        pred_masks["train"].append(outlier_artifacts["train"]["arrays"]["pred_mask"])
        pred_masks["test"].append(outlier_artifacts["test"]["arrays"]["pred_mask"])
        labels["train"] = outlier_artifacts["train"]["arrays"]["trues"]
        labels["test"] = outlier_artifacts["test"]["arrays"]["trues"]

    # we collect numpy arrays to a list of n submodels, thus this should be the same as the index of for loop
    # no_of_masks = len(pred_masks["test"])

    pred_mask_sum = np.sum(pred_masks["test"][idx])
    pred_mask_size = np.size(pred_masks["test"][idx])
    pred_mask_percent = 100 * (pred_mask_sum / pred_mask_size)
    logger.info(f"{model_name} {pred_mask_percent:.3f}% of model predictions are TRUE")

    label_mask_sum = np.sum(labels["test"])
    label_mask_size = np.size(labels["test"])
    label_mask_percent = 100 * (label_mask_sum / label_mask_size)
    logger.info(
        f"{model_name} {label_mask_percent:.3f}% of ground truth labels are TRUE (should be same for all models)"
    )

    return pred_masks, labels


def write_granular_outlier_metrics(metrics):
    if "outlier_test" not in metrics:
        metrics["outlier_test"] = metrics["test"]
        metrics["outlier_train"] = metrics["train"]

    for split in metrics:
        for granularity in metrics[split]:
            granular_scalars = metrics[split][granularity]["scalars"]["global"]
            for metric, value in granular_scalars.items():
                if value is not None:
                    if granularity == "all":
                        granularity_out = ""
                    else:
                        granularity_out = "__" + granularity

                    if isinstance(value, np.ndarray):
                        mlflow.log_metric(
                            f"{split}/{metric}_lo{granularity_out}", value[0]
                        )
                        mlflow.log_metric(
                            f"{split}/{metric}_hi{granularity_out}", value[1]
                        )
                    else:
                        mlflow.log_metric(f"{split}/{metric}{granularity_out}", value)
                        logger.debug(f"{split}/{metric}{granularity_out}: {value}")


def get_granular_outlier_metrics(data_dict, pred_masks, idx):
    metrics = {}
    for split in pred_masks.keys():
        preds = pred_masks[split][idx]
        assert isinstance(preds, np.ndarray)
        metrics[split] = eval_on_all_outlier_difficulty_levels(
            data_dict, preds, split.replace("outlier_", "")
        )
    return metrics


def compute_granular_metrics(
    run_id,
    mlflow_run: pd.Series,
    model_name: str,
    pred_masks: dict,
    idx: int,
    sources: dict,
    debug_verbose: bool = True,
):
    if "metrics.train/f1__easy_DEBUG" not in mlflow_run:
        # computing the granular metrics as they were not initially computed during the run
        metrics = get_granular_outlier_metrics(sources, pred_masks, idx)
        if debug_verbose:
            pred_mask_sum = np.sum(pred_masks["test"][idx])
            pred_mask_size = np.size(pred_masks["test"][idx])
            pred_mask_percent = 100 * (pred_mask_sum / pred_mask_size)
            logger.info(f"{model_name} {pred_mask_percent:.3f}% of pred_mask")

        # re-open the original mlflow_run and write the values
        with mlflow.start_run(run_id=run_id):
            write_granular_outlier_metrics(metrics)
            mlflow.end_run()

    else:
        logger.info("Granular metrics already computed")

    return metrics


def get_anomaly_masks_and_labels(ensembled_output: dict, sources: dict):
    pred_masks = {"train": [], "test": []}
    labels = {"train": [], "test": []}
    # assert (
    #     len(ensembled_output) % 2 != 0
    # ), "The number of models to ensemble must be odd"
    for idx, (model_name, mlflow_run) in enumerate(ensembled_output.items()):
        run_id = mlflow_run["run_id"]
        if isinstance(run_id, pd.Series):
            run_id = run_id.values[0]
        run_name = mlflow_run["tags.mlflow.runName"]
        if isinstance(run_name, pd.Series):
            run_name = run_name.values[0]

        if "MOMENT" in model_name:
            # quick'n'dirty fix for all the different MOMENT variants still being saved as MOMENT in the artifacts
            model_name = "MOMENT"
        if "TimesNet" in model_name:
            # quick'n'dirty fix for all the different MOMENT variants still being saved as MOMENT in the artifacts
            model_name = "TimesNet"

        from src.anomaly_detection.anomaly_utils import get_artifact

        outlier_artifacts_path = get_artifact(
            run_id, run_name, model_name, subdir="outlier_detection"
        )
        logger.info(
            f'Load ({run_id}) artifact file "{os.path.split(outlier_artifacts_path)[1]}" from "{outlier_artifacts_path}"'
        )
        outlier_artifacts = load_results_dict(outlier_artifacts_path)
        try:
            pred_masks, labels = get_anomaly_results_per_model(
                model_name, outlier_artifacts, pred_masks, labels, idx
            )
        except KeyError as e:
            logger.error(
                "Either harmonize the outputs of the models, or add parsing for the new model"
            )
            raise e

        _ = compute_granular_metrics(
            run_id, mlflow_run, model_name, pred_masks, idx, sources
        )

    # concantenate the results to a 3d numpy array
    pred_masks["train"] = np.stack(
        pred_masks["train"], axis=0
    )  # (no_of_models, no_of_patients, no_of_timepoints)
    pred_masks["test"] = np.stack(
        pred_masks["test"], axis=0
    )  # (no_of_models, no_of_patients, no_of_timepoints)

    return pred_masks, labels


def ensemble_masks(preds_3d: np.ndarray, method: str = "over_0.5"):
    preds_2d_float = np.mean(preds_3d, axis=0)
    if method == "over_0.5":
        preds_2d = (preds_2d_float > 0.5).astype(int)
    elif method == "over_0":
        # instead of averaging, and requiring the output to be over 0.5, you could binarize the output
        # so that the outlier is True when it is non-zero, more aggressive and you get more false positives
        preds_2d = (preds_2d_float > 0).astype(int)
    else:
        logger.error(
            "Unknown method for ensembling the masks, method: {}".format(method)
        )
        raise ValueError(
            "Unknown method for ensembling the masks, method: {}".format(method)
        )
    return preds_2d


def compute_ensemble_anomaly_metrics(pred_masks, labels, sources):
    """
    see e.g. metrics_per_split() in src/anomaly_detection/anomaly_detection_metrics_wrapper.py
    """
    assert (
        len(pred_masks["train"].shape) == 3
    ), "The pred_masks should be a 3D numpy array"
    assert (
        len(pred_masks["test"].shape) == 3
    ), "The pred_masks should be a 3D numpy array"

    metrics = {}
    for split in labels.keys():
        preds_3d = pred_masks[split]
        preds_2d = ensemble_masks(preds_3d)
        # labels_array = labels[split]
        metrics[split] = eval_on_all_outlier_difficulty_levels(
            sources, preds_2d, split.replace("outlier_", "")
        )
        # metrics[split] = get_scalar_outlier_metrics(preds=preds_2d, gt=labels_array)

    return metrics


def ensemble_anomaly_detection(
    ensembled_output: dict,
    cfg: DictConfig,
    experiment_name: str,
    ensemble_name: str,
    sources: dict,
):
    pred_masks, labels = get_anomaly_masks_and_labels(ensembled_output, sources)
    metrics = compute_ensemble_anomaly_metrics(pred_masks, labels, sources)

    return metrics, pred_masks
