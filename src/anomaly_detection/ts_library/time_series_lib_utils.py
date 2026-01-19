import math
from copy import deepcopy

from loguru import logger
import numpy as np
import torch
from omegaconf import DictConfig

from src.imputation.momentfm.moment_utils import reshape_array_to_original_shape


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
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
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = val_loss


def adjust_learning_rate(
    optimizer, epoch, lradj: str, learning_rate: float, train_epochs: int
):
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
    reshape flattened arrays back to input shape, e.g. (32000,) -> (16,1981) (or (16,2000) with padding
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
