import math
from copy import deepcopy

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from src.imputation.momentfm.moment_utils import reshape_array_to_original_shape


class EarlyStopping:
    """
    Early stopping callback for training.

    Monitors validation loss and stops training when no improvement
    is observed for a specified number of epochs.

    Parameters
    ----------
    patience : int, optional
        Number of epochs to wait for improvement. Default is 7.
    verbose : bool, optional
        Whether to print messages on checkpoint save. Default is False.
    delta : float, optional
        Minimum change to qualify as improvement. Default is 0.

    Attributes
    ----------
    counter : int
        Number of epochs since last improvement.
    best_score : float or None
        Best validation score seen.
    early_stop : bool
        Flag indicating training should stop.
    val_loss_min : float
        Minimum validation loss seen.
    """

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        """
        Check if training should stop and save checkpoint if improved.

        Parameters
        ----------
        val_loss : float
            Current validation loss.
        model : torch.nn.Module
            Model to checkpoint.
        path : str
            Directory for checkpoint file.
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """
        Save model checkpoint.

        Parameters
        ----------
        val_loss : float
            Current validation loss.
        model : torch.nn.Module
            Model to save.
        path : str
            Directory for checkpoint file.
        """
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = val_loss


def adjust_learning_rate(
    optimizer, epoch, lradj: str, learning_rate: float, train_epochs: int
):
    """
    Adjust learning rate based on epoch and schedule type.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to adjust.
    epoch : int
        Current epoch number.
    lradj : str
        Learning rate adjustment type: 'type1', 'type2', or 'cosine'.
    learning_rate : float
        Base learning rate.
    train_epochs : int
        Total number of training epochs (for cosine schedule).

    Notes
    -----
    - type1: Halves LR every epoch
    - type2: Predefined schedule at specific epochs
    - cosine: Cosine annealing schedule
    """
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if lradj == "type1":
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif lradj == "cosine":
        lr_adjust = {
            epoch: learning_rate / 2 * (1 + math.cos(epoch / train_epochs * math.pi))
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        logger.debug("Updating learning rate to {}".format(lr))


def reshape_arrays_to_original_shape(
    best_arrays: dict, cfg: DictConfig, outlier_model_cfg: DictConfig
):
    """
    Reshape flattened arrays back to original input shape.

    Converts arrays from batched/flattened format back to
    (n_subjects, n_timepoints) format.

    Parameters
    ----------
    best_arrays : dict
        Dictionary of arrays per split to reshape.
    cfg : DictConfig
        Full Hydra configuration with PLR_length.
    outlier_model_cfg : DictConfig
        Model configuration with window size.

    Returns
    -------
    dict
        Reshaped arrays per split.

    Raises
    ------
    ValueError
        If array shape is not 1D or 2D.

    Examples
    --------
    Reshapes (32000,) -> (16, 1981) or (16, 2000) with padding.
    """
    best_arrays_out = deepcopy(best_arrays)
    for split in best_arrays.keys():
        best_arrays_out[split] = {}
        for key, array in best_arrays[split].items():
            if len(array.shape) == 2:
                # e.g. (no_batches, no_time_points)
                best_arrays_out[split][key] = reshape_array_to_original_shape(
                    array, cfg, outlier_model_cfg
                )

            elif len(array.shape) == 1:
                # e.g. (no_samples) = (no_batches*no_time_points)
                best_arrays_out[split][key] = reshape_array_to_original_shape(
                    array, cfg, outlier_model_cfg, dim=1
                )

            else:
                logger.error("Unsupported array shape = {}".format(array.shape))
                raise ValueError("Unsupported array shape = {}".format(array.shape))

    return best_arrays_out
