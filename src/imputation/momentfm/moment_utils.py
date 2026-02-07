from copy import deepcopy
from typing import Literal

import mlflow
import numpy as np
import pandas as pd
import torch
from loguru import logger
from momentfm import MOMENTPipeline
from omegaconf import DictConfig

from src.data_io.data_utils import get_no_of_windows, unpad_glaucoma_PLR
from src.imputation.train_torch_utils import create_torch_dataloaders
from src.log_helpers.log_naming_uris_and_dirs import (
    get_outlier_detection_experiment_name,
)

_TASK_TYPES = Literal["imputation", "outlier_detection", "ts_cls"]


def remove_none_variants(mlflow_runs):
    """Filter out MLflow runs with None model variants.

    Removes ensemble runs that have 'MOMENT' in the run name but lack
    a pretrained model path in the expected parameter column.

    Parameters
    ----------
    mlflow_runs : pd.DataFrame
        DataFrame of MLflow runs with 'params.pretrained_model_name_or_path' column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with None variant rows removed.
    """
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
    """Get the best fine-tuned MOMENT run from MLflow.

    Searches for MOMENT fine-tuned runs in the specified experiment
    and returns the best one based on the specified metric.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name to search.
    cfg : DictConfig
        Configuration containing OUTLIER_DETECTION settings for sorting.
    model_variant : str
        MOMENT model variant to filter by (e.g., 'large', 'base', 'small').
    sort_by : str
        Metric key to sort runs by (e.g., 'best_loss').

    Returns
    -------
    pd.Series or None
        Best MLflow run as a Series, or None if no runs found.

    Raises
    ------
    ValueError
        If no MOMENT fine-tuned runs are found for the specified variant.
    """
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
    """Load a fine-tuned MOMENT model from MLflow artifacts.

    Retrieves the best fine-tuned MOMENT run and loads the model weights
    from the associated MLflow artifacts.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration.
    model : MOMENTPipeline
        Base MOMENT model to update with fine-tuned weights.
    model_variant : str
        Model variant identifier (e.g., 'large', 'base').
    task : str
        Task name for context ('imputation' or 'outlier_detection').
    sort_by : str, optional
        Metric to sort runs by. Default is 'best_loss'.

    Returns
    -------
    MOMENTPipeline or None
        Model with loaded fine-tuned weights, or None if no runs found.
    """
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
    """Import MOMENT model, optionally loading fine-tuned weights from MLflow.

    Creates a MOMENT pipeline from pretrained weights and optionally updates
    with fine-tuned weights stored in MLflow artifacts.

    Parameters
    ----------
    model_cfg : DictConfig
        Model configuration with pretrained model path.
    cfg : DictConfig
        Full Hydra configuration.
    task : str
        Task name ('imputation', 'outlier_detection', or 'embedding').
    model_kwargs : dict
        Additional keyword arguments for model initialization.
    load_from_mlflow : bool, optional
        Whether to load fine-tuned weights from MLflow. Default is True.

    Returns
    -------
    MOMENTPipeline
        Initialized MOMENT model, optionally with fine-tuned weights.
    """
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
    """Import and configure a MOMENT model for the specified task.

    Creates and initializes a MOMENT pipeline, handling both zero-shot
    (pretrained) and fine-tuned configurations based on detection_type.

    Parameters
    ----------
    model_cfg : DictConfig
        Model configuration containing MODEL settings (pretrained path,
        model_kwargs, detection_type).
    task : str
        Task type: 'outlier_detection', 'imputation', or 'embedding'.
    cfg : DictConfig
        Full Hydra configuration.

    Returns
    -------
    MOMENTPipeline
        Configured MOMENT model ready for the specified task.

    Raises
    ------
    ValueError
        If task is not one of the supported types.
    ImportError
        If pretrained model cannot be loaded.
    """
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
    """Validate dataloader batch for outlier detection task.

    Ensures appropriate masking based on split type: outlier splits should
    have masked points, while train/test splits should not (for reconstruction).

    Parameters
    ----------
    split : str
        Split type ('outlier', 'train', 'test', etc.).
    model_cfg : DictConfig
        Model configuration with detection_type setting.
    no_of_outliers : int
        Number of masked outlier points in the batch.
    i : int
        Batch index for error reporting.

    Raises
    ------
    ValueError
        If masking doesn't match expected behavior for the split type.
    """
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
    """Validate dataloader batch for imputation task.

    Ensures that batches contain some masked points for imputation.

    Parameters
    ----------
    split : str
        Split type (unused but kept for interface consistency).
    model_cfg : DictConfig
        Model configuration with detection_type setting.
    no_of_outliers : int
        Number of masked points in the batch.
    i : int
        Batch index for error reporting.

    Raises
    ------
    ValueError
        If no points are masked (nothing to impute) or unknown detection type.
    """
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
    """Validate PyTorch dataloader for NaN values and proper masking.

    Iterates through all batches to check for invalid data and optionally
    verify task-specific masking requirements.

    Parameters
    ----------
    dataloader : DataLoader
        PyTorch DataLoader to validate.
    model_cfg : DictConfig
        Model configuration for task-specific checks.
    split : str, optional
        Split name for error reporting. Default is 'train'.
    task : str, optional
        Task type for validation ('imputation' or 'outlier_detection').
        Default is 'imputation'.
    check_task_specific : bool, optional
        Whether to perform task-specific masking checks. Default is False.

    Returns
    -------
    DataLoader
        The validated dataloader (unchanged).

    Raises
    ------
    ValueError
        If NaN values found in input data or task-specific checks fail.
    """
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
    """Log the percentage of masked (missing) points per split.

    Parameters
    ----------
    dataloaders : dict
        Dictionary mapping split names to PyTorch DataLoaders.

    Raises
    ------
    AssertionError
        If outlier splits have 0% missing points (unexpected).
    """
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
    """Validate model output for NaN values and return valid arrays.

    Checks reconstruction output for NaN values and extracts only the
    valid (non-padded) portions of the data.

    Parameters
    ----------
    batch_x : torch.Tensor
        Input batch tensor.
    output : object
        Model output with 'reconstruction' attribute of shape (samples, channels, timepoints).
    input_masks : torch.Tensor
        Mask indicating valid input positions (1 = valid).
    labels : torch.Tensor
        Ground truth labels/targets.

    Returns
    -------
    tuple
        (valid_x, valid_recon, valid_labels) - tensors containing only
        valid positions.

    Raises
    ------
    ValueError
        If all output values are NaN.

    Warnings
    --------
    Logs warning if some (but not all) output values are NaN.
    """
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
    """Initialize PyTorch dataloaders for MOMENT training/inference.

    Creates dataloaders, prints missingness statistics, and optionally
    validates dataloader contents.

    Parameters
    ----------
    data_dict : dict
        Data dictionary containing 'df' with data per split.
    cfg : DictConfig
        Full Hydra configuration.
    model_cfg : DictConfig
        Model-specific configuration with TORCH settings and check_dataloader flag.
    run_name : str
        Run name for logging context.
    model_name : str, optional
        Model name for dataset creation. Default is None.
    task : {'imputation', 'outlier_detection', 'ts_cls'}, optional
        Task type for dataloader configuration. Default is 'imputation'.
    create_outlier_dataloaders : bool, optional
        Whether to create separate outlier train/test dataloaders.
        Default is False.

    Returns
    -------
    dict
        Dictionary mapping split names to PyTorch DataLoaders.
    """
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
    """Get the time series length from a DataLoader.

    Parameters
    ----------
    dataloader : DataLoader
        PyTorch DataLoader with TensorDataset.

    Returns
    -------
    int
        Number of timepoints in the time series.
    """
    _, no_timepoints = dataloader.dataset.tensors[0].shape
    return no_timepoints


def get_ts_no_of_subjects(dataloader):
    """Get the number of subjects from a DataLoader.

    Parameters
    ----------
    dataloader : DataLoader
        PyTorch DataLoader with TensorDataset.

    Returns
    -------
    int
        Number of subjects (samples) in the dataset.
    """
    no_subjects, _ = dataloader.dataset.tensors[0].shape
    return no_subjects


def get_ts_batch_sz(dataloader):
    """Get the batch size from a DataLoader.

    Parameters
    ----------
    dataloader : DataLoader
        PyTorch DataLoader.

    Returns
    -------
    int
        Batch size of the dataloader.
    """
    batch_sz = dataloader.batch_size
    return batch_sz


def init_arrays_for_moment_fm(dataloader):
    """Initialize arrays for storing MOMENT model predictions.

    Parameters
    ----------
    dataloader : DataLoader
        PyTorch DataLoader to determine array dimensions.

    Returns
    -------
    tuple
        (trues, preds, masks) - zero-initialized numpy arrays of shape
        (no_subjects, no_timepoints).
    """
    no_subjects = get_ts_no_of_subjects(dataloader)
    no_timepoints = get_ts_length(dataloader)
    trues = np.zeros((no_subjects, no_timepoints))
    preds = np.zeros((no_subjects, no_timepoints))
    masks = np.zeros((no_subjects, no_timepoints))

    return trues, preds, masks


def check_reshape_of_moment_arrays(results_out):
    """Validate that flat and 2D array shapes are consistent.

    Parameters
    ----------
    results_out : dict
        Results dictionary containing 'split_results' with 'arrays_flat'
        and 'arrays' sub-dictionaries.

    Raises
    ------
    AssertionError
        If the total number of elements differs between flat and 2D arrays.
    """
    no_samples_flat = len(results_out["split_results"]["arrays_flat"]["trues_valid"])
    no_samples_2d_array = results_out["split_results"]["arrays"]["trues"].size
    assert (
        no_samples_flat == no_samples_2d_array
    ), "Shape mismatch between flat and reshaped 2d arrays"


def reshape_np_array_windows(array_dict, cfg, outlier_model_cfg):
    """Reshape windowed arrays back to original PLR signal length.

    Applies reshaping to all arrays in the dictionary, converting from
    windowed format (e.g., 64x512) back to original format (e.g., 16x1981).

    Parameters
    ----------
    array_dict : dict
        Dictionary of arrays keyed by name to reshape.
    cfg : DictConfig
        Configuration with DATA.PLR_length.
    outlier_model_cfg : DictConfig
        Model configuration with TORCH.DATASET.trim_to_size.

    Returns
    -------
    dict
        Dictionary with reshaped arrays.
    """

    def reshape_func(array, cfg, outlier_model_cfg):
        """Reshape single array to original shape, or return unchanged if None or not 2D."""
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
    """Reshape all arrays in fine-tuning results to original PLR length.

    Iterates through results categories and reshapes any 'arrays' entries.

    Parameters
    ----------
    results : dict
        Results dictionary with nested structure containing 'arrays' entries.
    split : str
        Split name (unused but kept for logging/debugging).
    outlier_model_cfg : DictConfig
        Model configuration for window size.
    cfg : DictConfig
        Configuration for PLR length.

    Returns
    -------
    dict
        Deep copy of results with reshaped arrays.
    """
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
    """Reshape all result arrays from windowed to original PLR length.

    Parameters
    ----------
    results : dict
        Dictionary of arrays keyed by output type.
    split : str
        Split name for logging.
    outlier_model_cfg : DictConfig
        Model configuration with window size.
    cfg : DictConfig
        Configuration with PLR length.

    Returns
    -------
    dict
        Dictionary with reshaped arrays.
    """
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
    """Add a channel dimension to a 2D tensor.

    Parameters
    ----------
    x : torch.Tensor
        2D tensor of shape (batch, timepoints).

    Returns
    -------
    torch.Tensor
        3D tensor of shape (batch, 1, timepoints).

    Raises
    ------
    AssertionError
        If input is not 2D.
    """
    assert len(x.shape) == 2, "Input must be 2D"
    return x.unsqueeze(1)


def remove_empty_channel(x):
    """Remove the channel dimension from a 3D tensor.

    Parameters
    ----------
    x : torch.Tensor
        3D tensor of shape (batch, 1, timepoints).

    Returns
    -------
    torch.Tensor
        2D tensor of shape (batch, timepoints).

    Raises
    ------
    AssertionError
        If input is not 3D.
    """
    assert len(x.shape) == 3, "Input must be 3D"
    return x.squeeze(1)
