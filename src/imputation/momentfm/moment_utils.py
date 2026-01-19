from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from loguru import logger
from momentfm import MOMENTPipeline
import mlflow


from src.data_io.data_utils import get_no_of_windows, unpad_glaucoma_PLR
from src.imputation.train_torch_utils import create_torch_dataloaders

from typing import Literal

from src.log_helpers.log_naming_uris_and_dirs import (
    get_outlier_detection_experiment_name,
)

_TASK_TYPES = Literal["imputation", "outlier_detection", "ts_cls"]


def remove_none_variants(mlflow_runs):
    # ensembles have Momemnt in the run_name but no variant in the expected column name
    variant_col = mlflow_runs["params.pretrained_model_name_or_path"].to_numpy()
    variant_is_none = len(variant_col) * [False]
    for i, variant in enumerate(variant_col):
        if variant is None:
            variant_is_none[i] = True
    return mlflow_runs.drop(mlflow_runs[variant_is_none].index)


def get_best_moment_run(
    experiment_name, cfg, model_variant: str, sort_by: str
) -> pd.Series:
    from src.anomaly_detection.anomaly_utils import sort_anomaly_detection_runs

    # Get runs from the experiment
    mlflow_runs = mlflow.search_runs(experiment_names=[experiment_name])

    # Get only the ones with MOMENT and finetuned in the tags.mlflow.runName
    if mlflow_runs.shape[0] > 0:
        mlflow_runs = mlflow_runs[
            mlflow_runs["tags.mlflow.runName"].str.contains("MOMENT")
        ]
        mlflow_runs = mlflow_runs[
            mlflow_runs["tags.mlflow.runName"].str.contains("finetune")
        ]
        mlflow_runs = remove_none_variants(mlflow_runs)
        # Check that the model variant is correct (large, base, small)
        mlflow_runs = mlflow_runs[
            mlflow_runs["params.pretrained_model_name_or_path"].str.contains(
                model_variant
            )
        ]

        if mlflow_runs.shape[0] == 0:
            logger.warning("No MOMENT finetuned model found in MLflow")
            logger.warning(
                "Do you want to allow finetuning here without outlier detection?"
            )
            logger.error("No MOMENT finetuned model found in MLflow")
            raise ValueError("No MOMENT finetuned model found in MLflow")
        else:
            # Get then the best run of these
            best_dict = cfg["OUTLIER_DETECTION"][sort_by]
            mlflow_run: pd.Series = sort_anomaly_detection_runs(
                mlflow_runs, sort_by=sort_by, best_string=best_dict["string"]
            )

        return mlflow_run
    else:
        return None


def load_finetuned_moment_model_from_mlflow(
    cfg, model, model_variant: str, task: str, sort_by="best_loss"
):
    from src.anomaly_detection.anomaly_utils import (
        get_moment_model_from_mlflow_artifacts,
    )

    experiment_name = get_outlier_detection_experiment_name(cfg)
    mlflow_run = get_best_moment_run(experiment_name, cfg, model_variant, sort_by)
    if mlflow_run is None:
        logger.error("No MOMENT finetuned model found in MLflow")
        logger.error("Using the default MOMENT model")
        return None
    else:
        model = get_moment_model_from_mlflow_artifacts(
            run_id=mlflow_run["run_id"],
            run_name=mlflow_run["tags.mlflow.runName"],
            model=model,
            device=cfg["DEVICE"]["device"],
            cfg=cfg,
            task=task,
        )
        return model


def import_moment_from_mlflow(
    model_cfg: DictConfig,
    cfg: DictConfig,
    task: str,
    model_kwargs: dict,
    load_from_mlflow: bool = True,
):
    model = MOMENTPipeline.from_pretrained(
        pretrained_model_name_or_path=model_cfg["MODEL"][
            "pretrained_model_name_or_path"
        ],
        model_kwargs=model_kwargs,
    )
    model.init()

    if task != "embedding":
        if load_from_mlflow:
            model_variant = model_cfg["MODEL"]["pretrained_model_name_or_path"]
            logger.info(
                "Updating the model weights with the fine-tuning done with outlier detection"
            )
            logger.info("Model variant = {}".format(model_variant))
            model_finetuned = load_finetuned_moment_model_from_mlflow(
                cfg, model, model_variant=model_variant, task=task
            )
            if model_finetuned is not None:
                model = model_finetuned

    return model


def import_moment_model(model_cfg: DictConfig, task: str, cfg: DictConfig):
    logger.info("Importing pretrained MOMENT model")
    model_kwargs = dict(model_cfg["MODEL"]["model_kwargs"])
    if model_cfg["MODEL"]["detection_type"] == "fine-tune":
        # see e.g. moment-research/moment/tasks/anomaly_detection_finetune.py
        model_kwargs = {**model_kwargs, **{"do_not_copy_head": False}}
        logger.info("Fine-tuning the model, 'do_not_copy_head'=False")
    try:
        # 1.39 GB model, so if you run this on cloud, see how to cache this somewhere maybe, EFS?
        if task == "outlier_detection":
            model = MOMENTPipeline.from_pretrained(
                pretrained_model_name_or_path=model_cfg["MODEL"][
                    "pretrained_model_name_or_path"
                ],
                model_kwargs=model_kwargs,
            )
        elif task == "imputation":
            # imputation now uses the exact same model finetuned in the outlier detection task (reconstruction)
            # So we can actually load the "finetuned model" from our MLflow, and the "zeroshot model" from
            # HuggingFace or wherever the MOMENT people are hosting the model
            logger.info("Loading the pretrained model for imputation")
            for key, item in model_cfg["MODEL"].items():
                logger.info(f"{key} = {item}")
            model = import_moment_from_mlflow(
                model_cfg=model_cfg,
                cfg=cfg,
                task=task,
                model_kwargs=model_kwargs,
                load_from_mlflow=model_cfg["MODEL"]["detection_type"] == "fine-tune",
            )
        else:
            logger.error("Unknown task = {}".format(task))
            raise ValueError("Unknown task = {}".format(task))

        model.init()
    except ImportError as e:
        logger.error("Could not import pretrained MOMENT model")
        raise e
    return model


def check_outlier_detection_dataloader(
    split: str, model_cfg: DictConfig, no_of_outliers: int, i: int
):
    if split == "outlier":
        # "outlier_train" and "outlier_test" are the "pupil_orig" pupil sizes and contain the blink artifacts,
        # and we would like to learn to detect them all, thus there should be the outliers masked
        if no_of_outliers == 0:
            # this might raise an error with a single subject batch sizes (batch size = 4)
            # or even smaller sizes (batch_sz = 1 - 3) as we had windowed data, as you might have
            # a window that has none outliers
            logger.error(
                f"No outlier time points masked, seems unlikely per batch!, batch {i}"
            )
            raise ValueError(
                f"No outlier time points masked, seems unlikely per batch!, batch {i}"
            )

    else:
        # You are learning to reconstruct the signal with "train" and "test"
        if (
            model_cfg["MODEL"]["detection_type"] == "fine-tune"
            or model_cfg["MODEL"]["detection_type"] == "zero-shot"
        ):
            if no_of_outliers > 0:
                logger.error(
                    f"Outlier time points masked, should not be any! batch {i}"
                )
                raise ValueError(
                    f"Outlier time points masked, should not be any! batch {i}"
                )
        else:
            logger.error(
                "Unknown detection type = {}".format(
                    model_cfg["MODEL"]["detection_type"]
                )
            )
            raise ValueError(
                "Unknown detection type = {}".format(
                    model_cfg["MODEL"]["detection_type"]
                )
            )


def check_imputation_dataloader(
    split: str, model_cfg: DictConfig, no_of_outliers: int, i: int
):
    if (
        model_cfg["MODEL"]["detection_type"] == "fine-tune"
        or model_cfg["MODEL"]["detection_type"] == "zero-shot"
    ):
        # if you have very small batch sizes, and garbage outlier detection algorithm, it might well be that you
        # don't have any missing data masked here. This check is done upstream in "dataset_imputation_selector()"
        # and it should pick up if there are no masked points in the whole dataset
        if no_of_outliers == 0:
            logger.error(f"No missing time points masked, should be some! batch {i}")
            raise ValueError(
                f"No missing time points masked, should be some! batch {i}"
            )
    else:
        logger.error(
            "Unknown detection type = {}".format(model_cfg["MODEL"]["detection_type"])
        )
        raise ValueError(
            "Unknown detection type = {}".format(model_cfg["MODEL"]["detection_type"])
        )


def check_dataloaders(
    dataloader,
    model_cfg: DictConfig,
    split: str = "train",
    task: str = "imputation",
    # this is not bulletproof if True, as individual batches can easily be without any True mask
    # if getting output from a garbage outlier detection algorithm
    check_task_specific: bool = False,
):
    for i, (batch_x, batch_masks, input_masks) in enumerate(dataloader):
        batch_x = batch_x.float()
        nans_in_input_mask = torch.isnan(input_masks).sum()
        assert nans_in_input_mask == 0, "NaNs in the input mask"
        input_data = batch_x[input_masks == 1]  # length should be 1981
        nan_sum = input_data.isnan().sum()
        # no_of_invalid_points = (input_masks == 0).sum()
        if nan_sum > 0:
            logger.error(f"split = {split}, NaNs in the input data, {i}th batch")
            raise ValueError(f"split = {split}, NaNs in the input data, {i}th batch")
        mask_points = batch_masks[input_masks == 1]
        no_of_outliers = mask_points.sum()
        if task == "outlier_detection":
            if check_task_specific:
                check_outlier_detection_dataloader(
                    split=split, model_cfg=model_cfg, no_of_outliers=no_of_outliers, i=i
                )
        elif task == "imputation":
            if check_task_specific:
                check_imputation_dataloader(
                    split=split, model_cfg=model_cfg, no_of_outliers=no_of_outliers, i=i
                )
        else:
            logger.error("Unknown task = {}".format(task))
            raise ValueError("Unknown task = {}".format(task))

    return dataloader


def print_out_missingness_ratios(dataloaders):
    for split, dataloader in dataloaders.items():
        no_mask_points = 0
        no_total_points = 0
        for i, (batch_x_gt, batch_y, _) in enumerate(dataloader):
            mask = batch_y.detach().cpu().numpy()
            no_mask_points += np.nansum(mask)
            no_total_points += mask.size
        mask_percentage = 100 * no_mask_points / no_total_points
        # e.g. for outlier_detection, train/test are 0.00%, but for the outlier_train/test
        # you should have non-zero mask percentage
        logger.info(f'Split = "{split}": {mask_percentage:.2f}% are missing from mask')
        if "outlier" in split:
            assert (
                mask_percentage > 0
            ), "You should have some points masked for the outlier split = {}".format(
                split
            )


def check_output_for_nans(batch_x, output, input_masks, labels):
    no_samples, no_chans, no_timepoints = output.reconstruction.shape  # (1,16,512)
    size = no_samples * no_timepoints * no_chans  # (8192, )
    reconstruction = remove_empty_channel(output.reconstruction)  # (16,512)
    # these now get flattened which is okay for loss
    valid_x = remove_empty_channel(batch_x)[input_masks == 1]  # (7924,)
    valid_recon = reconstruction[input_masks == 1]  # (7924,)
    valid_labels = labels[input_masks == 1]  # (7924,)
    nansum = torch.isnan(valid_recon).sum()
    if nansum == size:
        logger.error("All values in the output are NaNs")
        raise ValueError("All values in the output are NaNs")
    elif nansum > 0:
        logger.warning(
            "Some values (n={}, {:.2f}%) in the output are NaNs".format(
                nansum, (nansum / size) * 100
            )
        )

    assert valid_x.shape == valid_recon.shape, "Shapes do not match"
    assert valid_x.shape == valid_labels.shape, "Shapes do not match"
    return valid_x, valid_recon, valid_labels


def init_torch_training(
    data_dict: dict,
    cfg: DictConfig,
    model_cfg: DictConfig,
    run_name: str,
    model_name: str = None,
    task: _TASK_TYPES = "imputation",
    create_outlier_dataloaders=False,  # for outlier_detections True
):
    # Create torch dataloader for both zero-shot imputation and fine-tuning
    dataloaders = create_torch_dataloaders(
        task=task,
        model_name=model_name,
        data_dict_df=data_dict["df"],
        model_cfg=model_cfg,
        cfg=cfg,
        create_outlier_dataloaders=create_outlier_dataloaders,
    )

    # Print out the missingness ratios in splits
    print_out_missingness_ratios(dataloaders)

    # Check the dataloaders
    if model_cfg["check_dataloader"]:
        logger.info("Checking the dataloaders for NaNs/invalid values")
        for split, dataloader in dataloaders.items():
            check_dataloaders(
                dataloader=dataloader, split=split, model_cfg=model_cfg, task=task
            )
    else:
        logger.debug("Skipping the dataloader check for NaNs/invalid values")

    return dataloaders


def get_ts_length(dataloader):
    _, no_timepoints = dataloader.dataset.tensors[0].shape
    return no_timepoints


def get_ts_no_of_subjects(dataloader):
    no_subjects, _ = dataloader.dataset.tensors[0].shape
    return no_subjects


def get_ts_batch_sz(dataloader):
    batch_sz = dataloader.batch_size
    return batch_sz


def init_arrays_for_moment_fm(dataloader):
    no_subjects = get_ts_no_of_subjects(dataloader)
    no_timepoints = get_ts_length(dataloader)
    trues = np.zeros((no_subjects, no_timepoints))
    preds = np.zeros((no_subjects, no_timepoints))
    masks = np.zeros((no_subjects, no_timepoints))

    return trues, preds, masks


def check_reshape_of_moment_arrays(results_out):
    no_samples_flat = len(results_out["split_results"]["arrays_flat"]["trues_valid"])
    no_samples_2d_array = results_out["split_results"]["arrays"]["trues"].size
    assert (
        no_samples_flat == no_samples_2d_array
    ), "Shape mismatch between flat and reshaped 2d arrays"


def reshape_np_array_windows(array_dict, cfg, outlier_model_cfg):
    def reshape_func(array, cfg, outlier_model_cfg):
        if array is not None:
            if len(array.shape) == 2:
                array_2D = array
            else:
                return array
            array_out = reshape_array_to_original_shape(
                array_2D, cfg, outlier_model_cfg
            )
            assert (
                array_out.shape[1] == cfg["DATA"]["PLR_length"]
            ), "Shape mismatch ({})".format(array_out.shape)
        else:
            return array
        return array_out

    reshaped = {}
    for array_name, array in array_dict.items():
        reshaped[array_name] = reshape_func(array, cfg, outlier_model_cfg)

    return reshaped


def reshape_finetune_arrays(results, split, outlier_model_cfg, cfg):
    # Loop through all the 2D arrays and reshape them to the original shape
    results_out = deepcopy(results)
    for category_name, category in results.items():
        for data_type in category.keys():
            if "arrays" in data_type:
                array_dict = results[category_name][data_type]
                results_out[category_name][data_type] = reshape_np_array_windows(
                    array_dict, cfg, outlier_model_cfg
                )

    return results_out


def reshape_array_to_original_shape(array, cfg, outlier_model_cfg, dim: int = 2):
    """
    Reshape the array to the original shape (e.g. from (64,512) to (16,1981))
    array: np.array
        shape: (no_subjects, no_timepoints)
    """
    window_size = outlier_model_cfg["TORCH"]["DATASET"]["trim_to_size"]
    windows_per_subject = get_no_of_windows(
        length_PLR=cfg["DATA"]["PLR_length"],
        window_size=window_size,
    )

    if dim == 2:
        array = np.reshape(
            array,
            (
                array.shape[0] // windows_per_subject,
                array.shape[1] * windows_per_subject,
            ),
        )
    elif dim == 1:
        no_subjects = int(array.shape[0] / windows_per_subject / window_size)
        array = np.reshape(array, (no_subjects, window_size * windows_per_subject))
    else:
        logger.error("Only 1D/2D Reshaping supported now, not {}".format(dim))
        raise ValueError("Only 1D/2D Reshaping supported now, not {}".format(dim))

    array = unpad_glaucoma_PLR(array, length_PLR=cfg["DATA"]["PLR_length"])
    assert np.isnan(array).sum() == 0, "NaNs in the reshaped array"

    return array


def reshape_to_original_lengths(
    results: dict, split: str, outlier_model_cfg: DictConfig, cfg: DictConfig
):
    results_out = {}
    logger.info(
        "{} | Reshaping the results to the original lengths (e.g. from (64,512) to (16,1981))".format(
            split
        )
    )

    for output_key, array in results.items():
        array = reshape_array_to_original_shape(array, cfg, outlier_model_cfg)
        results_out[output_key] = array
        logger.debug(f"Reshaped {output_key} to shape {array.shape}")

    return results_out


def add_empty_channel(x):
    assert len(x.shape) == 2, "Input must be 2D"
    return x.unsqueeze(1)


def remove_empty_channel(x):
    assert len(x.shape) == 3, "Input must be 3D"
    return x.squeeze(1)
