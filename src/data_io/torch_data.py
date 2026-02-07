import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import TensorDataset

from src.classification.xgboost_cls.xgboost_utils import encode_labels_to_integers
from src.data_io.data_utils import transform_data_for_momentfm

# See for a Class example:
# https://github.com/moment-timeseries-foundation-model/moment/blob/main/momentfm/data/anomaly_detection_dataset.py


def trim_data(x):
    """Trim PLR data to remove edge artifacts.

    Removes the first 3 and last 2 timepoints from PLR recordings
    to get a clean 1976-sample signal.

    Parameters
    ----------
    x : np.ndarray
        Input array with shape (n_subjects, n_timepoints).

    Returns
    -------
    np.ndarray
        Trimmed array with shape (n_subjects, 1976).
    """
    return x[:, 3:1979]  # 1976


def nan_padding(x, n: int = 1981):
    """Pad trimmed data back to original length with NaN values.

    Inverse operation of trim_data, fills edge positions with NaN.

    Parameters
    ----------
    x : np.ndarray
        Trimmed input array with shape (n_subjects, 1976).
    n : int, optional
        Target output length, by default 1981.

    Returns
    -------
    np.ndarray
        Padded array with shape (n_subjects, n) with NaN at edges.
    """
    x_out = np.zeros((x.shape[0], n))
    x_out[:, :] = np.nan
    x_out[:, 3:1979] = x
    return x_out


def get_outlier_data(data_dict, split):
    """Extract outlier detection data and masks from data dictionary.

    Retrieves the imputed original pupil data and corresponding outlier
    mask for evaluating outlier detection algorithms.

    Parameters
    ----------
    data_dict : dict
        Hierarchical data dictionary with split keys.
    split : str
        Data split to extract ("train" or "test").

    Returns
    -------
    tuple of np.ndarray
        X: pupil data array (n_subjects, n_timepoints)
        mask: outlier mask array where 1=outlier, 0=normal

    Raises
    ------
    AssertionError
        If no outliers are labeled in the mask.
    """
    # I.e. you use this to evaluate whether your network can remove outliers
    X = data_dict[split]["data"]["pupil_orig_imputed"]
    # Mask gives 1 for outliers, 0 for normal values
    # as in the labels or "mask" in the Moment code
    mask = data_dict[split]["labels"]["outlier_mask"]
    outlier_ratio = mask.sum() / mask.size
    logger.debug(f"outlier_ratio: {outlier_ratio}")
    assert mask.sum() > 0, "No outliers labeled in the mask"

    return X, mask


def pick_pupil_data_col(train_on, data_dict, split):
    """Select appropriate pupil data column based on training configuration.

    Retrieves the correct data column (ground truth, raw imputed, or original
    imputed) and corresponding mask based on the train_on parameter.

    Parameters
    ----------
    train_on : str
        Data column to use: "pupil_gt", "pupil_raw_imputed", or "pupil_orig_imputed".
    data_dict : dict
        Hierarchical data dictionary with split keys.
    split : str
        Data split to extract ("train" or "test").

    Returns
    -------
    tuple of np.ndarray
        X: pupil data array (n_subjects, n_timepoints)
        mask: corresponding mask array (zeros for gt/raw, outlier_mask for orig)

    Raises
    ------
    ValueError
        If train_on parameter is not recognized.
    """
    if train_on == "pupil_gt":
        # This is the denoised (clean signal)
        X = data_dict[split]["data"]["pupil_gt"]
        mask: np.ndarray = np.zeros_like(X)
        logger.info('Picking "pupil_gt" data')
    elif train_on == "pupil_raw_imputed":
        # The raw data (with no outliers)
        X = data_dict[split]["data"]["pupil_raw_imputed"]
        mask: np.ndarray = np.zeros_like(X)
        logger.info('Picking "pupil_raw_data" data')
    elif train_on == "pupil_orig_imputed":
        # The original raw data that most of the other methods used to train
        X = data_dict[split]["data"]["pupil_orig_imputed"]
        mask = data_dict[split]["labels"]["outlier_mask"]
        logger.info('Picking "pupil_orig_data" data')
    else:
        logger.error("Unknown train_on = {}".format(train_on))
        raise ValueError("Unknown train_on = {}".format(train_on))
        # mask comes from the outlier dataloaders, we just reconstruct the clean signal

    return X, mask


def dataset_outlier_detection_selector(
    detection_type: str, train_on: str, split: str, split_data: str, data_dict: dict
):
    """Select data for outlier detection based on detection type and split.

    Routes data selection based on whether fine-tuning or zero-shot detection
    is used, and whether outlier-specific splits are requested.

    Parameters
    ----------
    detection_type : str
        Detection approach: "fine-tune" or "zero-shot".
    train_on : str
        Data column to use for training.
    split : str
        Data split ("train" or "test").
    split_data : str
        Specific split type (e.g., "outlier_train", "outlier_test").
    data_dict : dict
        Hierarchical data dictionary.

    Returns
    -------
    tuple of np.ndarray
        X: data array and mask: outlier mask array.

    Raises
    ------
    ValueError
        If detection_type is not recognized.
    AssertionError
        If outlier split requested but mask has no outliers.
    """
    if detection_type == "fine-tune":
        if "outlier" in split_data:
            # You only need the outlier split for outlier detection. When doing imputation training,
            # You train for the reconstruction, and mask out the missing time points, and check out
            # how well the "vanilla test" is reconstructed on missing mask points
            X, mask = get_outlier_data(data_dict, split)
            assert mask.sum() > 0, "No outliers labeled in the mask"
        else:
            X, mask = pick_pupil_data_col(train_on, data_dict, split)

    elif detection_type == "zero-shot":
        if "outlier" in split_data:
            X, mask = get_outlier_data(data_dict, split)
        else:
            X, mask = pick_pupil_data_col(train_on, data_dict, split)

    else:
        logger.error("Unknown detection type = {}".format(detection_type))
        raise ValueError("Unknown detection type = {}".format(detection_type))

    return X, mask


def dataset_ts_cls_selector(
    detection_type: str, train_on: str, split: str, data_dict: dict
):
    """Select data for time series classification task.

    Extracts features and class labels for binary classification,
    encoding string labels to integers.

    Parameters
    ----------
    detection_type : str
        Detection approach: "fine-tune" or "full-finetune".
    train_on : str
        Data column to use (not used directly, for interface consistency).
    split : str
        Data split ("train" or "test").
    data_dict : dict
        Hierarchical data dictionary.

    Returns
    -------
    tuple
        X: feature array (n_subjects, n_features)
        labels: integer class labels (n_subjects,)

    Raises
    ------
    ValueError
        If detection_type is not recognized.
    AssertionError
        If number of unique classes is not 2, or if label count mismatches X.
    """
    if detection_type == "fine-tune" or detection_type == "full-finetune":
        X = data_dict[split]["data"]["X"]
        labels = data_dict[split]["labels"]["class_label"][:, 0]
        assert (
            len(np.unique(labels)) == 2
        ), "You have != 2 classes, unique classes = {}".format(np.unique(labels))
        labels = encode_labels_to_integers(labels)
        assert (
            len(labels) == X.shape[0]
        ), "Labels and X must have the same number of samples"
    else:
        logger.error("Unknown detection type = {}".format(detection_type))
        raise ValueError("Unknown detection type = {}".format(detection_type))

    return X, labels


def dataset_imputation_selector(
    detection_type: str, train_on: str, split: str, data_dict: dict
):
    """Select data for imputation task.

    Extracts data and missingness mask for training imputation models
    to reconstruct missing values.

    Parameters
    ----------
    detection_type : str
        Detection approach: "fine-tune" or "zero-shot".
    train_on : str
        Data column: "pupil_gt" or "pupil_raw_imputed".
    split : str
        Data split ("train" or "test").
    data_dict : dict
        Hierarchical data dictionary.

    Returns
    -------
    tuple of np.ndarray
        X: data array and mask: missingness mask.

    Raises
    ------
    ValueError
        If detection_type or train_on is not recognized.
    NotImplementedError
        If train_on is "pupil_raw_imputed" (not yet implemented).
    AssertionError
        If mask has no missing points.
    """
    if detection_type == "fine-tune":
        if train_on == "pupil_gt":
            # This is the denoised (clean signal)
            X = data_dict[split]["data"]["X"]
        elif train_on == "pupil_raw_imputed":
            # X and X_GT are the same if you decided to train on pupil_gt, and different with something
            # else, and with "pupil_raw_imputed", the X would already be this? Debug more if you actually start
            # using these
            logger.error("Not yet implemented")
            raise NotImplementedError("Not yet implemented")
        else:
            # Implement here if you use pupil_raw or pupil_orig, or should not need implementation?
            # just the correct column options?
            logger.error("Unknown train_on = {}".format(train_on))
            raise ValueError("Unknown train_on = {}".format(train_on))
        mask = data_dict[split]["data"]["mask"]

    elif detection_type == "zero-shot":
        X = data_dict[split]["data"]["X"]
        mask = data_dict[split]["data"]["mask"]

    else:
        logger.error("Unknown detection type = {}".format(detection_type))
        raise ValueError("Unknown detection type = {}".format(detection_type))

    # In the imputation task, we don't have at the moment any outlier_x split so the mask should contain
    # some missingness points, otherwise we can't compute any metrics for the imputation performance
    assert mask.sum() > 0, "No outliers labeled in the mask"

    return X, mask


def dataset_data_array_selector(
    split_data,
    task,
    data_dict,
    detection_type: str = "zero-shot",
    train_on: str = "gt",
):
    """Main dispatcher for selecting data arrays based on task and split.

    Routes data extraction to appropriate task-specific selector based on
    the task type (outlier detection, imputation, or classification).

    Parameters
    ----------
    split_data : str
        Split specification: "train", "test", "outlier_train", "outlier_test".
    task : str
        Task type: "outlier_detection", "imputation", or "ts_cls".
    data_dict : dict
        Hierarchical data dictionary.
    detection_type : str, optional
        Detection approach, by default "zero-shot".
    train_on : str, optional
        Data column to use, by default "gt".

    Returns
    -------
    tuple of np.ndarray
        X: data array and mask/label array depending on task.

    Raises
    ------
    ValueError
        If split_data or task is not recognized.
    AssertionError
        If X contains NaN values or shape mismatch with mask.
    """
    if split_data == "outlier_test":
        # "outlier" is a "virtual split" that takes data from different processing level,
        # but we want to use the "test" or "val" split now as it is not used for training directly
        # (well now it has been used to select the best model, so there is no real extensive validation atm)
        split = "test"
    elif split_data == "outlier_train":
        split = "train"  # same as with the outlier_test
    else:
        # otherwise these are the same
        split = split_data

    valid_splits = ["train", "test", "outlier_test", "outlier_train"]
    if split in valid_splits:
        if task == "outlier_detection":
            X, mask = dataset_outlier_detection_selector(
                detection_type=detection_type,
                train_on=train_on,
                split=split,
                split_data=split_data,
                data_dict=data_dict,
            )
        elif task == "imputation":
            X, mask = dataset_imputation_selector(
                detection_type=detection_type,
                train_on=train_on,
                split=split,
                data_dict=data_dict,
            )
        elif task == "ts_cls":
            # well this is X, and label
            X, label = dataset_ts_cls_selector(
                detection_type=detection_type,
                train_on=train_on,
                split=split,
                data_dict=data_dict,
            )
            assert np.isnan(X).sum() == 0, "Missing values in the data"
            return X, label
        else:
            logger.error("Unkown task = {}".format(task))
            raise ValueError("Unkown task = {}".format(task))

        # Depending on the algorithm, you might want to use the "pupil_orig" or "pupil_raw" data as
        # well without the imputation, if NaNs or missing problems in general (irregular sampling) is okay
        assert isinstance(X, np.ndarray), "X must be a Numpy array, not {}".format(
            type(X)
        )
        assert np.isnan(X).sum() == 0, "Missing values in the data"
    else:
        logger.error("Unrecognized split = {}".format(split))
        raise ValueError("Unrecognized split = {}".format(split))

    assert (
        X.shape[0] == mask.shape[0]
    ), "X and mask must have the same number of samples"

    return X, mask


def pick_splits_from_data_dict_to_ts(data_dict_df, model_cfg, train_on):
    """Extract train/test splits from data dictionary for time series export.

    Organizes data into split-specific dictionaries with X (data), y (outlier mask),
    and time arrays for downstream processing.

    Parameters
    ----------
    data_dict_df : dict
        Hierarchical data dictionary with split keys containing data arrays.
    model_cfg : DictConfig
        Model configuration (not currently used but kept for interface).
    train_on : str
        Data column to extract (e.g., "pupil_orig_imputed").

    Returns
    -------
    dict
        Dictionary with "train" and "test" keys, each containing:
        - X: data array
        - y: outlier mask
        - time: time vector

    References
    ----------
    - https://github.com/eBay/RANSynCoders/blob/main/example.ipynb
    """
    data_splits = {}
    # train_on = "pupil_orig_imputed"  # model_cfg["MODEL"]["train_on"]
    for split in data_dict_df.keys():
        data_splits[split] = {}
        data_splits[split]["X"] = data_dict_df[split]["data"][train_on]
        data_splits[split]["y"] = data_dict_df[split]["labels"]["outlier_mask"]
        data_splits[split]["time"] = data_dict_df[split]["time"]["time"]

    return data_splits


def create_dataset_from_numpy(
    data_dict_df: dict,
    dataset_cfg: DictConfig,
    model_cfg: DictConfig,
    split: str,
    task: str = "imputation",
    model_name: str = None,
):
    """Create a PyTorch TensorDataset from numpy arrays.

    Converts data dictionary arrays to PyTorch tensors and optionally
    applies trimming for foundation model compatibility.

    Parameters
    ----------
    data_dict_df : dict
        Hierarchical data dictionary containing numpy arrays.
    dataset_cfg : DictConfig
        Dataset configuration with trim_to_size and other settings.
    model_cfg : DictConfig
        Model configuration with MODEL settings (detection_type, train_on).
    split : str
        Data split to use ("train", "test", "outlier_train", "outlier_test").
    task : str, optional
        Task type for data selection, by default "imputation".
    model_name : str, optional
        Model name for trim configuration, by default None.

    Returns
    -------
    TensorDataset
        PyTorch dataset with (X, mask, input_mask) tensors.
    """
    # Pick the needed arrays from the model artifacts dictionary
    X, mask = dataset_data_array_selector(
        split_data=split,
        task=task,
        data_dict=data_dict_df,
        detection_type=model_cfg["MODEL"]["detection_type"],
        train_on=model_cfg["MODEL"]["train_on"],
    )

    if dataset_cfg["trim_to_size"] is not None:
        # This takes fixed windows, the Class-based sliding window sampler might be better in the future
        # if MOMENT shows some promise
        X, mask, input_mask = transform_data_for_momentfm(
            X, mask, dataset_cfg, model_name
        )
    else:
        input_mask = np.zeros((X.shape[0], X.shape[1]))
        input_mask[~np.isnan(X)] = 1

    tensor_X = torch.Tensor(X)  # transform to torch tensor
    tensor_mask = torch.Tensor(mask)
    tensor_input_mask = torch.Tensor(input_mask)
    dataset = TensorDataset(tensor_X, tensor_mask, tensor_input_mask)

    return dataset
