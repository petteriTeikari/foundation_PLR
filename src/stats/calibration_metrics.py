"""
Calibration metrics for classification models.

Provides ECE (Expected Calibration Error), Brier score, and calibration curves.
"""

import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from torchmetrics.classification import BinaryCalibrationError


def get_calibration_curve(model, y_true, preds: dict, n_bins: int):
    """
    Compute calibration curve using sklearn.

    Parameters
    ----------
    model : object
        Classifier model (unused, kept for API compatibility).
    y_true : np.ndarray
        True binary labels.
    preds : dict
        Dictionary with 'y_pred_proba' key.
    n_bins : int
        Number of bins for calibration curve.

    Returns
    -------
    dict
        Dictionary with 'prob_true' and 'prob_pred' arrays.
    """
    prob_true, prob_pred = calibration_curve(
        y_true, preds["y_pred_proba"], n_bins=n_bins
    )
    # disp = CalibrationDisplay.from_predictions(y_true, preds["y_pred_proba"])
    return {"prob_true": prob_true, "prob_pred": prob_pred}


def get_calibration_metrics(
    model,
    metrics,
    y_true: np.ndarray,
    preds: dict,
    n_bins: int = 3,
):
    """
    TODO! Have some calibration metrics? ECE? (AU)RC? Brier Score?
    Alasalmi et al. 2020, Better Classifier Calibration for Small Datasets, https://doi.org/10.1145/3385656
    https://scholar.google.co.uk/scholar?cites=12369575194770427495&as_sdt=2005&sciodt=0,5&hl=en

    Nixon et al. 2019, Measuring Calibration in Deep Learning, https://openreview.net/forum?id=r1la7krKPS
    https://scholar.google.fi/scholar?cites=671990448700625194&as_sdt=2005&sciodt=0,5&hl=en

    Are XGBoost probabilities well-calibrated? - "No, they are not well-calibrated."
    https://stats.stackexchange.com/a/617182/294507
      "Do note that "badly" calibrated probabilities are not synonymous with a useless model but I would urge
      one doing an extra calibration step (i.e. Platt scaling, isotonic regression or beta calibration)
      if using the raw probabilities is of importance."

    https://stats.stackexchange.com/a/619981/294507
      "XGBoost is well calibrated providing you optimise for log_loss (as objective and in hyperparameter search)."


    """
    # Sorta hybrid, both classifier and calibration
    metrics["metrics"]["scalars"]["Brier"] = brier_score_loss(
        y_true, preds["y_pred_proba"]
    )

    # ECE probably enough from the "basics"? even with its flaws
    ece = BinaryCalibrationError(n_bins=n_bins, norm="max")
    preds_torch = torch.tensor(preds["y_pred_proba"])
    target = torch.tensor(y_true.astype(float))
    metrics["metrics"]["scalars"]["ECE"] = ece(preds_torch, target).item()

    # Get calibration curve, not a lot of points with our tiny data, but they get averaged
    # a bit then over iters?
    # See e.g. https://github.com/huyng/incertae/blob/master/ensemble_classification.ipynb
    metrics["metrics"]["arrays"]["calibration_curve"] = get_calibration_curve(
        model, y_true, preds, n_bins=n_bins
    )

    return metrics
