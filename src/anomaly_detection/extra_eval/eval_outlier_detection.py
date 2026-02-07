import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# from src.submodules.UniTS.utils.extra_eval.adjf1 import adjbestf1_with_threshold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.anomaly_detection.anomaly_detection_metrics_wrapper import (
    adjbestf1_with_threshold,
)


def get_padding_indices(
    length_orig: int = 1981, length_padded: int = 2048
) -> Tuple[int, int]:
    """
    Calculate start and end indices for extracting original signal from padded array.

    Parameters
    ----------
    length_orig : int, optional
        Original signal length before padding, by default 1981.
    length_padded : int, optional
        Length after padding, by default 2048.

    Returns
    -------
    tuple of int
        Start and end indices (start_idx, end_idx) for slicing the padded array.
    """
    no_points_pad = length_padded - length_orig  # 67
    start_idx = no_points_pad // 2  # 33
    end_idx = start_idx + length_orig  # 2014
    return start_idx, end_idx


def unpad_glaucoma_PLR(array: np.ndarray, length_PLR: int = 1981) -> np.ndarray:
    """
    Remove padding from PLR signal array to restore original length.

    Parameters
    ----------
    array : np.ndarray
        Padded array with shape (n_subjects, padded_length).
    length_PLR : int, optional
        Original PLR signal length to restore, by default 1981.

    Returns
    -------
    np.ndarray
        Unpadded array with shape (n_subjects, length_PLR).
    """
    start_idx, end_idx = get_padding_indices(
        length_orig=length_PLR, length_padded=array.shape[1]
    )
    array_out = array[:, start_idx:end_idx]
    return array_out


def get_no_of_windows(length_PLR: int = 1981, window_size: int = 512) -> int:
    """
    Calculate the number of windows needed to cover the full PLR signal.

    Parameters
    ----------
    length_PLR : int, optional
        Total length of the PLR signal, by default 1981.
    window_size : int, optional
        Size of each window, by default 512.

    Returns
    -------
    int
        Number of windows (ceiling division).
    """
    return np.ceil(length_PLR / window_size).astype(int)


def reshape_array(
    array: np.ndarray, window_size: int = 500, length_PLR: int = 1981
) -> np.ndarray:
    """
    Reshape windowed array back to original subject-wise shape.

    Converts from windowed format (e.g., 64 windows x 512 points) back to
    subject format (e.g., 16 subjects x 1981 points) and removes padding.

    Parameters
    ----------
    array : np.ndarray
        Input array in windowed format, shape (n_windows, window_size) or
        (n_windows, window_size, 1) or flattened 1D.
    window_size : int, optional
        Size of each window, by default 500.
    length_PLR : int, optional
        Original PLR signal length, by default 1981.

    Returns
    -------
    np.ndarray
        Reshaped array with shape (n_subjects, length_PLR).

    Raises
    ------
    ValueError
        If array has more than 3 dimensions.
    """
    windows_per_subject = get_no_of_windows(
        length_PLR=length_PLR,
        window_size=window_size,
    )

    dim = len(array.shape)
    if dim == 3:
        array = np.squeeze(array)
        dim = len(array.shape)

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

    array = unpad_glaucoma_PLR(array, length_PLR=length_PLR)
    assert np.isnan(array).sum() == 0, "NaNs in the reshaped array"

    return array


def adjustment(gt: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply point-adjustment to predictions based on ground truth anomaly segments.

    When a prediction correctly identifies any point within an anomaly segment,
    all points in that segment are marked as correctly predicted.

    Parameters
    ----------
    gt : array-like
        Ground truth binary labels (1 = anomaly, 0 = normal).
    pred : array-like
        Predicted binary labels (1 = anomaly, 0 = normal).

    Returns
    -------
    tuple of array-like
        Adjusted (gt, pred) arrays with point-adjustment applied.
    """
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def outlier_scalar_metric_wrapper(
    preds: np.ndarray,
    gt: np.ndarray,
    score: Optional[np.ndarray] = None,
    cfg: Optional[DictConfig] = None,
    adj_f1_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute scalar metrics for outlier detection evaluation.

    Parameters
    ----------
    preds : np.ndarray
        Binary predictions (1 = outlier, 0 = normal).
    gt : np.ndarray
        Ground truth binary labels.
    score : np.ndarray, optional
        Continuous anomaly scores for adjusted F1 computation.
    cfg : DictConfig, optional
        Configuration object (currently unused).
    adj_f1_threshold : float, optional
        Pre-computed threshold for adjusted F1; if None, optimal is found.

    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall, f1, support, fp,
        unadjusted_f1, and adj_f1_threshold.
    """
    warnings.simplefilter("ignore")
    if len(gt.shape) > 1:
        gt = gt.flatten()
        preds = preds.flatten()
        if score is not None:
            score = score.flatten()

    no_unique_gt = len(np.unique(gt))
    assert no_unique_gt <= 2, "You have {} classes, should be binary now".format(
        no_unique_gt
    )

    metrics = {}
    metrics["accuracy"] = accuracy_score(gt, preds)

    (
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["support"],
    ) = precision_recall_fscore_support(gt, preds, average="binary")

    try:
        _, fp, _, _ = confusion_matrix(gt.flatten(), preds.flatten()).ravel()
        metrics["fp"] = float(fp) / len(gt.flatten())
    except Exception:
        # if no outliers for given subject
        metrics["fp"] = np.nan
    warnings.resetwarnings()

    # adjusted f1
    if score is not None:
        metrics["unadjusted_f1"] = deepcopy(metrics["f1"])
        metrics["f1"], metrics["adj_f1_threshold"] = adjbestf1_with_threshold(
            y_true=gt, y_scores=score.flatten(), adj_f1_threshold=adj_f1_threshold
        )
    else:
        metrics["unadjusted_f1"], metrics["adj_f1_threshold"] = None, None

    return metrics


def subjectwise_outlier_metric_wrapper(
    preds: np.ndarray,
    gt: np.ndarray,
    score: np.ndarray,
    cfg: Optional[DictConfig] = None,
    global_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[Any]]:
    """
    Compute outlier detection metrics for each subject individually.

    Parameters
    ----------
    preds : np.ndarray
        Binary predictions with shape (n_subjects, n_timepoints).
    gt : np.ndarray
        Ground truth labels with shape (n_subjects, n_timepoints).
    score : np.ndarray
        Continuous anomaly scores with shape (n_subjects, n_timepoints).
    cfg : DictConfig, optional
        Configuration object passed to metric wrapper.
    global_metrics : dict, optional
        Dictionary containing global adj_f1_threshold to use for all subjects.

    Returns
    -------
    dict
        Dictionary with lists of per-subject metric values for each metric key.
    """
    no_subjects, no_timepoints = gt.shape
    metrics_subjectwise = {}

    # use the global threshold for all the subject and don't pick a threshold for each subject
    adj_f1_threshold = global_metrics["adj_f1_threshold"]

    for subj_idx in range(no_subjects):
        gt_subject = gt[subj_idx, :]
        pred_subject = preds[subj_idx, :]
        if score is not None:
            score_subject = score[subj_idx, :]
        else:
            score_subject = None
        metrics_subject = outlier_scalar_metric_wrapper(
            preds=pred_subject,
            gt=gt_subject,
            score=score_subject,
            cfg=cfg,
            adj_f1_threshold=adj_f1_threshold,
        )
        if subj_idx == 0:
            for key, value in metrics_subject.items():
                metrics_subjectwise[key] = [value]
        else:
            for key, value in metrics_subject.items():
                metrics_subjectwise[key].append(value)

    return metrics_subjectwise


def update_CI_to_outlier_scalars(
    metrics_subjectwise: Dict[str, List[Any]],
    global_metrics: Dict[str, Any],
    p: float = 0.05,
) -> Dict[str, Any]:
    """
    Add confidence interval bounds to global metrics from subject-wise values.

    Parameters
    ----------
    metrics_subjectwise : dict
        Dictionary with lists of per-subject metric values.
    global_metrics : dict
        Dictionary of global metrics to update with CI bounds.
    p : float, optional
        Percentile for CI bounds (lower=p, upper=100-p), by default 0.05.

    Returns
    -------
    dict
        Updated global_metrics with '{key}_CI' entries containing [lower, upper] bounds.
    """
    for key in metrics_subjectwise.keys():
        array_of_values = np.array(metrics_subjectwise[key])
        if np.all(array_of_values == None):  # noqa E711
            ci = None
        else:
            ci = np.nanpercentile(array_of_values, [p, 100 - p])
        global_metrics[f"{key}_CI"] = ci

    return global_metrics


def get_scalar_outlier_metrics(
    preds: np.ndarray,
    gt: np.ndarray,
    score: np.ndarray,
    cfg: Optional[DictConfig] = None,
    threshold: Optional[float] = None,
    use_detection_adjustment: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute global and subject-wise outlier detection metrics.

    Parameters
    ----------
    preds : np.ndarray
        Binary predictions with shape (n_subjects, n_timepoints).
    gt : np.ndarray
        Ground truth labels with shape (n_subjects, n_timepoints).
    score : np.ndarray
        Continuous anomaly scores with shape (n_subjects, n_timepoints).
    cfg : DictConfig, optional
        Configuration object passed to metric wrappers.
    threshold : float, optional
        Pre-computed threshold for adjusted F1.
    use_detection_adjustment : bool, optional
        Whether to apply point-adjustment to predictions, by default True.

    Returns
    -------
    dict
        Dictionary with 'global' and 'subjectwise' metric dictionaries.
    """
    if use_detection_adjustment:
        gt, preds = detection_adjustment(gt=gt, pred=preds)

    metrics = {}
    metrics["global"] = outlier_scalar_metric_wrapper(
        preds=preds, gt=gt, score=score, cfg=cfg, adj_f1_threshold=threshold
    )
    metrics["subjectwise"] = subjectwise_outlier_metric_wrapper(
        preds=preds, gt=gt, score=score, cfg=cfg, global_metrics=metrics["global"]
    )
    metrics["global"] = update_CI_to_outlier_scalars(
        metrics_subjectwise=metrics["subjectwise"], global_metrics=metrics["global"]
    )

    return metrics


def metrics_per_split(
    energy: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """
    Compute outlier detection metrics for a data split (train or test).

    Parameters
    ----------
    energy : np.ndarray
        Anomaly scores/energy values with shape (n_subjects, n_timepoints).
    labels : np.ndarray
        Ground truth binary labels with shape (n_subjects, n_timepoints).
    threshold : float
        Threshold for converting energy scores to binary predictions.

    Returns
    -------
    dict
        Dictionary with 'scalars' (metrics), 'arrays' (trues, pred_mask),
        and 'arrays_flat' keys.
    """
    metrics_test = {"scalars": {}, "arrays": {}, "arrays_flat": {}}
    metrics_test["arrays"]["trues"] = labels

    pred = (energy > threshold).astype(int)
    metrics_test["arrays"]["pred_mask"] = pred

    # pred = pred.flatten()
    # labels = labels.flatten()
    gt = labels.astype(int)  # e.g. (32000,)
    assert pred.shape == gt.shape
    # The vanilla metrics
    metrics_test["scalars"] = get_scalar_outlier_metrics(
        preds=pred, gt=gt, score=energy, cfg=None, threshold=None
    )

    return metrics_test


def detection_adjustment(
    gt: np.ndarray,
    pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply point-adjustment to predictions, handling both 1D and 2D arrays.

    Wrapper around adjustment() that preserves array shape after flattening
    and adjustment.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth binary labels, 1D or 2D (n_subjects, n_timepoints).
    pred : np.ndarray
        Predicted binary labels, same shape as gt.

    Returns
    -------
    tuple of np.ndarray
        Adjusted (gt, pred) arrays with original shape preserved.
    """
    if gt.ndim == 1:
        # no_subjects, no_timepoints = None, gt.shape
        gt, pred = adjustment(gt.flatten(), pred.flatten())
    else:
        no_subjects, no_timepoints = gt.shape
        gt, pred = adjustment(gt.flatten(), pred.flatten())
        gt = np.reshape(gt, (no_subjects, no_timepoints))
        pred = np.reshape(pred, (no_subjects, no_timepoints))

    pred = np.array(pred)
    gt = np.array(gt)
    return gt, pred


def get_score(
    criterion: nn.Module,
    batch_x: torch.Tensor,
    outputs: torch.Tensor,
) -> np.ndarray:
    """
    Compute reconstruction error score from model outputs.

    Parameters
    ----------
    criterion : nn.Module
        Loss function (e.g., MSELoss with reduction='none').
    batch_x : torch.Tensor
        Input batch with shape (batch_size, seq_len, n_features).
    outputs : torch.Tensor
        Model reconstruction outputs with same shape as batch_x.

    Returns
    -------
    np.ndarray
        Mean reconstruction error per sample, shape (batch_size, seq_len).
    """
    score = torch.mean(criterion(batch_x, outputs), dim=-1)
    score = score.detach().cpu().numpy()
    return score


def forward_three_item_loader(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Forward pass through dataloader returning 3 items per batch.

    Used for Foundation_PLR loaders that return (batch_x, batch_y, input_mask).

    Parameters
    ----------
    loader : DataLoader
        PyTorch DataLoader yielding (batch_x, batch_y, mask) tuples.
        batch_x: shape (batch_size, seq_len, n_features), e.g., (128, 100, 25).
        batch_y: shape (batch_size, seq_len, 1), e.g., (128, 100, 1).
    model : nn.Module
        Reconstruction model to evaluate.
    device : torch.device
        Device to run inference on.
    criterion : nn.Module
        Loss function for computing reconstruction error.

    Returns
    -------
    tuple of lists
        (attens_energy, recon, labels) where each is a list of np.ndarray batches.
    """
    attens_energy = []
    labels = []
    recon = []
    with torch.no_grad():
        for i, (batch_x, batch_y, _) in enumerate(loader):
            batch_x = batch_x.float().to(device)  # e.g (16,100)
            batch_x = batch_x.unsqueeze(2)  # e.g. (16,100,1)
            outputs = model(batch_x, None, None, None)  # e.g. (16,100,1)
            score = get_score(criterion, batch_x, outputs)

            attens_energy.append(score)
            recon.append(outputs.cpu().detach().numpy())
            labels.append(batch_y.cpu().detach().numpy())

    return attens_energy, recon, labels


def forward_two_item_loader(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    criterion: nn.Module,
    task_id: int = 0,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Forward pass through dataloader returning 2 items per batch.

    Used for UniTS-style loaders that return only (batch_x, batch_y).

    Parameters
    ----------
    loader : DataLoader
        PyTorch DataLoader yielding (batch_x, batch_y) tuples.
    model : nn.Module
        Reconstruction model with task_id and task_name support.
    device : torch.device
        Device to run inference on.
    criterion : nn.Module
        Loss function for computing reconstruction error.
    task_id : int, optional
        Task identifier for multi-task models, by default 0.

    Returns
    -------
    tuple of lists
        (attens_energy, recon, labels) where each is a list of np.ndarray batches.
    """

    attens_energy = []
    labels = []
    recon = []
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(tqdm(loader, "attens_energy")):
            batch_x = batch_x.float().to(device)  # e.g. (256, 500, 1)
            # reconstruction
            outputs = model(
                batch_x,
                None,
                None,
                None,
                task_id=task_id,
                task_name="anomaly_detection",
            )
            # criterion
            score = get_score(criterion, batch_x, outputs)

            attens_energy.append(score)
            recon.append(outputs.cpu().detach().numpy())
            labels.append(batch_y.cpu().detach().numpy())

    return attens_energy, recon, labels


def reshape_arrays_to_input_lengths(
    attens_energy: np.ndarray,
    labels: np.ndarray,
    recon: np.ndarray,
    length_PLR: int = 1981,
    window_size: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reshape all output arrays from windowed format back to original PLR length.

    Parameters
    ----------
    attens_energy : np.ndarray
        Anomaly scores in windowed format.
    labels : np.ndarray
        Ground truth labels in windowed format.
    recon : np.ndarray
        Reconstruction outputs in windowed format.
    length_PLR : int, optional
        Original PLR signal length, by default 1981.
    window_size : int, optional
        Size of each window, by default 500.

    Returns
    -------
    tuple of np.ndarray
        (attens, labels, recon) all with shape (n_subjects, length_PLR).
    """
    attens = reshape_array(
        attens_energy, window_size=window_size, length_PLR=length_PLR
    )
    recon = reshape_array(recon, window_size=window_size, length_PLR=length_PLR)
    labels = reshape_array(array=labels, window_size=window_size, length_PLR=length_PLR)
    assert attens.shape == recon.shape == labels.shape
    return attens, labels, recon


def get_attens_energy(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    criterion: nn.Module,
    window_size: int = 100,
    length_PLR: int = 1981,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute anomaly scores, labels, and reconstructions from a dataloader.

    Automatically detects loader format (2-item or 3-item) and processes
    accordingly.

    Parameters
    ----------
    loader : DataLoader
        PyTorch DataLoader for the dataset.
    model : nn.Module
        Reconstruction model to evaluate.
    device : torch.device
        Device to run inference on.
    criterion : nn.Module
        Loss function for computing reconstruction error.
    window_size : int, optional
        Size of each window, by default 100.
    length_PLR : int, optional
        Original PLR signal length, by default 1981.

    Returns
    -------
    tuple of np.ndarray
        (attens_energy, labels, recon) all with shape (n_subjects, length_PLR).

    Raises
    ------
    ValueError
        If dataloader returns unsupported number of items per batch.
    """
    example: list = next(iter(loader))
    no_of_items_in_batch = len(example)

    if no_of_items_in_batch == 3:
        # in the Foundation_PLR, there are batch_x, batch_y, and input mask
        # to cater originally for Momemnt's needs with the input mask being able to mask NaN values
        attens_energy, recon, labels = forward_three_item_loader(
            loader, model, device, criterion
        )
    elif no_of_items_in_batch == 2:
        # UniTS has only data and labels
        attens_energy, recon, labels = forward_two_item_loader(
            loader, model, device, criterion
        )
    else:
        logger.error(
            "No of items = {} in dataloader not supported".format(no_of_items_in_batch)
        )
        raise ValueError(
            "No of items = {} in dataloader not supported".format(no_of_items_in_batch)
        )

    attens_energy = np.concatenate(attens_energy, axis=0)
    recon = np.concatenate(recon, axis=0)  # [:,:,0]
    labels = np.concatenate(labels, axis=0)  # [:,:,0]
    attens_energy, labels, recon = reshape_arrays_to_input_lengths(
        attens_energy=attens_energy,
        labels=labels,
        recon=recon,
        window_size=window_size,
        length_PLR=length_PLR,
    )
    assert attens_energy.shape[1] == length_PLR
    assert attens_energy.shape[1] == labels.shape[1] == recon.shape[1]

    outlier_percentage = 100 * (labels.sum() / labels.size)
    logger.info(f"Outliers masked {outlier_percentage:2f}%")

    return attens_energy, labels, recon


def combine_lists_of_ndarrays(
    list1: List[np.ndarray], list2: List[np.ndarray]
) -> np.ndarray:
    """
    Combine two lists of numpy arrays into a single concatenated array.

    Parameters
    ----------
    list1 : list
        First list of numpy arrays to combine.
    list2 : list
        Second list of numpy arrays to combine.

    Returns
    -------
    np.ndarray
        Flattened and concatenated 1D array from both input lists.
    """

    def convert_list_to_array(list: List[np.ndarray]) -> np.ndarray:
        """
        Stack and flatten a list of numpy arrays into a 1D array.

        Parameters
        ----------
        list : list of np.ndarray
            List of numpy arrays to combine.

        Returns
        -------
        np.ndarray
            Flattened 1D array containing all elements from the input arrays.
        """
        return np.vstack(list).flatten()

    array1 = convert_list_to_array(list=list1)
    array2 = convert_list_to_array(list=list2)
    return np.concatenate((array1, array2), axis=0)


def convert_flat_to_2d_array(
    flat_arrays: Dict[str, np.ndarray],
    window_size: int = 500,
    length_PLR: int = 1981,
) -> Dict[str, np.ndarray]:
    """
    Convert dictionary of flat arrays to 2D subject-wise format.

    Parameters
    ----------
    flat_arrays : dict
        Dictionary mapping keys to flattened numpy arrays.
    window_size : int, optional
        Size of each window for reshaping, by default 500.
    length_PLR : int, optional
        Original PLR signal length, by default 1981.

    Returns
    -------
    dict
        Dictionary with same keys, values reshaped to (n_subjects, length_PLR).
    """
    dict_out = {}
    for key, array in flat_arrays.items():
        dict_out[key] = reshape_array(
            array=array, window_size=window_size, length_PLR=length_PLR, dim=1
        )


def eval_outlier_detection(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    _device_id: int = 0,
    device: Optional[torch.device] = None,
    task_id: Optional[int] = None,
    features: str = "S",
    anomaly_ratio: float = 10,
    # these are now annoyingly hard-coded :(
    window_size: int = 500,
    length_PLR: int = 1981,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate anomaly detection model on train and test splits.

    Computes reconstruction errors, determines threshold from combined data,
    and evaluates detection performance on both splits.

    Parameters
    ----------
    model : nn.Module
        Trained reconstruction model to evaluate.
    train_loader : DataLoader
        DataLoader for training data.
    test_loader : DataLoader
        DataLoader for test data.
    _device_id : int, optional
        GPU device ID (unused), by default 0.
    device : torch.device, optional
        Device for inference.
    task_id : int, optional
        Task identifier for multi-task models.
    features : str, optional
        Feature type indicator, by default "S".
    anomaly_ratio : float, optional
        Initial anomaly ratio estimate (overridden by actual ratio), by default 10.
    window_size : int, optional
        Size of each window, by default 500.
    length_PLR : int, optional
        Original PLR signal length, by default 1981.

    Returns
    -------
    dict
        Dictionary with 'outlier_train' and 'outlier_test' keys, each containing
        'scalars' (metrics), 'arrays' (predictions, ground truth), and
        reconstruction outputs.
    """
    # https://github.com/thuml/Time-Series-Library/blob/c80b851784184e088cc12afc9472fe55a06d6ada/exp/exp_anomaly_detection.py#L128
    model.eval()
    warnings.simplefilter("ignore")
    # size_average and reduce args will be deprecated, please use reduction='none' instead.
    criterion = nn.MSELoss(reduce=False)
    warnings.resetwarnings()
    metrics_test = {}

    # 1) static on the train set
    logger.info('Getting the train set "attens_energy"')
    train_energy, train_labels, train_recon = get_attens_energy(
        loader=train_loader,
        model=model,
        device=device,
        criterion=criterion,
        window_size=window_size,
        length_PLR=length_PLR,
    )  # e.g. (32000,)

    # the test set
    logger.info('Getting the test set "attens_energy"')
    test_energy, test_labels, test_recon = get_attens_energy(
        loader=test_loader,
        model=model,
        device=device,
        criterion=criterion,
        window_size=window_size,
        length_PLR=length_PLR,
    )  # e.g. (32000,)

    # 2) find the threshold
    combined_energy = np.concatenate([train_energy, test_energy], axis=0).flatten()
    combined_labels = np.concatenate([train_labels, test_labels], axis=0).flatten()

    # instead of a prior, use the actual anomaly ratio
    anomaly_ratio = 100 * (np.sum(combined_labels) / combined_labels.shape[0])  # 15.625
    threshold = np.percentile(
        combined_energy, 100 - anomaly_ratio
    )  # float threshold, e.g. 0.00389
    logger.info(
        f"Threshold = {threshold:.5f}, at anomaly_percentage = {anomaly_ratio:.2f}%"
    )

    # (3) evaluation on the splits
    logger.info("Getting the evaluation metrics")
    metrics_test["outlier_train"] = metrics_per_split(
        energy=train_energy, labels=train_labels, threshold=threshold
    )
    metrics_test["outlier_test"] = metrics_per_split(
        energy=test_energy, labels=test_labels, threshold=threshold
    )

    # add reconstructions to output dictionary
    metrics_test["outlier_test"]["arrays"]["preds"] = test_recon
    metrics_test["outlier_train"]["arrays"]["preds"] = train_recon

    return metrics_test
