import warnings

import torch
from loguru import logger

from src.imputation.momentfm.moment_utils import (
    add_empty_channel,
    check_output_for_nans,
    remove_empty_channel,
)


def momentfm_forward_pass(
    model,
    batch_x,
    labels,  # same as batch_masks
    input_masks,
    device,
    criterion=None,
    anomaly_criterion: str = None,
    detect_anomalies: bool = False,
    task_name: str = None,
):
    """
    Execute forward pass through MOMENT model.

    Handles data formatting, model inference, and loss computation for
    reconstruction-based outlier detection.

    Parameters
    ----------
    model : torch.nn.Module
        MOMENT model.
    batch_x : torch.Tensor
        Input batch of shape (batch_sz, no_timesteps).
    labels : torch.Tensor
        Outlier labels of shape (batch_sz, no_timesteps).
    input_masks : torch.Tensor
        Masks indicating valid timepoints.
    device : str
        Device to run on ('cpu' or 'cuda').
    criterion : torch.nn.Module, optional
        Loss function. If None, loss is not computed. Default is None.
    anomaly_criterion : str, optional
        Anomaly detection criterion type. Default is None.
    detect_anomalies : bool, optional
        Whether to use detect_anomalies method. Default is False.
    task_name : str, optional
        Task name for logging. Default is None.

    Returns
    -------
    tuple
        A tuple containing:
        - output : object
            MOMENT model output with reconstruction.
        - loss : torch.Tensor or None
            Reconstruction loss if criterion provided.
        - valid_dict : dict
            Dictionary with valid_x, valid_recon, valid_labels.

    Raises
    ------
    NotImplementedError
        If detect_anomalies=True (produces unexpected results).
    ValueError
        If computed loss is NaN.
    """
    # The data is expected to be 3D (batch_sz, no_channels, no_timesteps)
    batch_x = add_empty_channel(batch_x.to(device).float())
    assert len(batch_x.shape) == 3, (
        "Input must be 3D (batch_sz, no_channels, no_timesteps)"
    )

    # If you for example generate the masks on the fly. DOne on the imputation tutorial
    if labels is not None:
        # no_of_missing_values = torch.nansum(labels)
        labels = labels.to(device).long()
        assert batch_x.shape[0] == labels.shape[0], "Batch size mismatch"
        assert labels.shape[1] == batch_x.shape[2], "Number of timepoints do not match"

    # what points are valid (and what are not, e.g. containing NaNs from padding)
    # no_of_padded_points = torch.isnan(batch_x).sum() # e.g. 268 for 16x512 batch
    input_masks = input_masks.to(device).float()

    if detect_anomalies:
        raise NotImplementedError("Seems to produce something funky")
        # output = model.detect_anomalies(
        #     x_enc=batch_x,
        #     input_mask=input_masks,
        #     anomaly_criterion=anomaly_criterion
        # )
        # valid_x, valid_recon = check_output_for_nans(
        #     batch_x, output, input_masks
        # )
    else:
        input_masks = add_empty_channel(input_masks)
        # Warning: None of the inputs have requires_grad=True. Gradients will be None?
        # https://discuss.pytorch.org/t/question-about-requires-grad-true/110586/3
        # inputs[0] (16,64,1024)
        # inputs[1] (16,1,1,64) -> check_backward_validity
        # torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly.
        #     In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False
        #     is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True.
        warnings.simplefilter("ignore")
        output = model(
            x_enc=batch_x,
            input_masks=input_masks,
            anomaly_criterion=anomaly_criterion,
            batch_masks=labels,
            # use_reentrant=False,
        )
        warnings.resetwarnings()
        valid_x, valid_recon, valid_labels = check_output_for_nans(
            batch_x, output, remove_empty_channel(input_masks), labels
        )

    if criterion is not None:
        loss = criterion(valid_recon, valid_x)
        if torch.isnan(loss):
            logger.error("Loss is NaN")
            raise ValueError("Loss is NaN")
    else:
        loss = None

    return (
        output,
        loss,
        {"valid_x": valid_x, "valid_recon": valid_recon, "valid_labels": valid_labels},
    )
