from typing import Literal, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def prr_class(
    labels: ArrayLike, probs: ArrayLike, measure: ArrayLike, rev: bool
) -> float:
    """Compute prediction rejection ratio for classification tasks.

    Evaluates uncertainty-based rejection by comparing against random
    and oracle rejection baselines.

    Parameters
    ----------
    labels : array-like
        Ground truth class labels.
    probs : array-like of shape (n_samples, n_classes)
        Predicted class probabilities.
    measure : array-like
        Uncertainty measure for each sample used for rejection ordering.
    rev : bool
        If True, reverse the sorting order of the uncertainty measure.

    Returns
    -------
    float
        Rejection ratio as a percentage, indicating how well the uncertainty
        measure identifies misclassifications compared to oracle rejection.
    """
    # Get predictions
    preds = np.argmax(probs, axis=1)

    if rev:
        inds = np.argsort(measure)[::-1]
    else:
        inds = np.argsort(measure)

    total_data = np.float32(preds.shape[0])
    errors, percentages = [], []

    for i in range(preds.shape[0]):
        errors.append(
            np.sum(np.asarray(labels[inds[:i]] != preds[inds[:i]], dtype=np.float32))
            * 100.0
            / total_data
        )
        percentages.append(float(i + 1) / total_data * 100.0)
    errors, percentages = np.asarray(errors)[:, np.newaxis], np.asarray(percentages)

    base_error = errors[-1]
    n_items = errors.shape[0]
    auc_uns = 1.0 - auc(percentages / 100.0, errors[::-1] / 100.0)

    random_rejection = np.asarray(
        [base_error * (1.0 - float(i) / float(n_items)) for i in range(n_items)],
        dtype=np.float32,
    )
    auc_rnd = 1.0 - auc(percentages / 100.0, random_rejection / 100.0)
    orc_rejection = np.asarray(
        [
            base_error * (1.0 - float(i) / float(base_error / 100.0 * n_items))
            for i in range(int(base_error / 100.0 * n_items))
        ],
        dtype=np.float32,
    )
    orc = np.zeros_like(errors)
    orc[0 : orc_rejection.shape[0]] = orc_rejection
    auc_orc = 1.0 - auc(percentages / 100.0, orc / 100.0)

    rejection_ratio = (auc_uns - auc_rnd) / (auc_orc - auc_rnd) * 100.0
    return rejection_ratio


def prr_regression(
    targets: ArrayLike, preds: ArrayLike, measure: ArrayLike, pos_label: int = 1
) -> float:
    """Compute prediction rejection ratio for regression tasks.

    Evaluates how well an uncertainty measure can identify high-error predictions
    compared to random and oracle rejection strategies.

    Parameters
    ----------
    targets : array-like
        Ground truth target values.
    preds : array-like
        Predicted values.
    measure : array-like
        Uncertainty measure for each sample used for rejection ordering.
    pos_label : int, optional
        If not 1, the measure is negated before sorting. Default is 1.

    Returns
    -------
    float
        Area under the rejection ratio curve, normalized between random
        and optimal rejection performance.
    """
    if pos_label != 1:
        measure_loc = -1.0 * measure
    else:
        measure_loc = measure
    preds = np.squeeze(preds)
    # Compute total MSE
    error = (preds - targets) ** 2
    MSE_0 = np.mean(error)
    # print 'BASE MSE', MSE_0

    # Create array
    array = np.concatenate(
        (
            preds[:, np.newaxis],
            targets[:, np.newaxis],
            error[:, np.newaxis],
            measure_loc[:, np.newaxis],
        ),
        axis=1,
    )

    # Results arrays
    results_max = [[0.0, 0.0]]
    results_var = [[0.0, 0.0]]
    results_min = [[0.0, 0.0]]

    optimal_ranking = array[:, 2].argsort()
    sorted_array = array[optimal_ranking]  # Sort by error

    for i in range(1, array.shape[0]):
        x = np.concatenate((sorted_array[:-i, 0], sorted_array[-i:, 1]), axis=0)
        mse = np.mean((x - sorted_array[:, 1]) ** 2)
        # Best rejection
        results_max.append([float(i) / float(array.shape[0]), (MSE_0 - mse) / MSE_0])
        # Random Rejection
        results_min.append(
            [float(i) / float(array.shape[0]), float(i) / float(array.shape[0])]
        )

    uncertainty_ranking = array[:, 3].argsort()
    sorted_array = array[uncertainty_ranking]  # Sort by uncertainty

    for i in range(1, array.shape[0]):
        x = np.concatenate((sorted_array[:-i, 0], sorted_array[-i:, 1]), axis=0)
        mse = np.mean((x - sorted_array[:, 1]) ** 2)
        results_var.append([float(i) / float(array.shape[0]), (MSE_0 - mse) / MSE_0])

    max_auc = auc([x[0] for x in results_max], [x[1] for x in results_max])
    var_auc = auc([x[0] for x in results_var], [x[1] for x in results_var])
    min_auc = auc([x[0] for x in results_min], [x[1] for x in results_min])

    AUC_RR = (var_auc - min_auc) / (max_auc - min_auc)

    return AUC_RR


def ood_detect(
    domain_labels: ArrayLike,
    in_measure: ArrayLike,
    out_measure: ArrayLike,
    mode: Literal["PR", "ROC"],
    pos_label: int = 1,
) -> float:
    """Evaluate out-of-distribution detection using uncertainty scores.

    Computes detection performance metrics for distinguishing in-distribution
    from out-of-distribution samples based on uncertainty measures.

    Parameters
    ----------
    domain_labels : array-like
        Binary labels indicating in-distribution (0) vs out-of-distribution (1).
    in_measure : array-like
        Uncertainty scores for in-distribution samples.
    out_measure : array-like
        Uncertainty scores for out-of-distribution samples.
    mode : {'PR', 'ROC'}
        Evaluation mode: 'PR' for precision-recall AUC, 'ROC' for ROC AUC.
    pos_label : int, optional
        If not 1, scores are negated. Default is 1.

    Returns
    -------
    float
        Area under the curve (AUPR or AUROC depending on mode).
    """
    scores = np.concatenate((in_measure, out_measure), axis=0)
    scores = np.asarray(scores, dtype=np.float128)
    if pos_label != 1:
        scores *= -1.0

    if mode == "PR":
        precision, recall, thresholds = precision_recall_curve(domain_labels, scores)
        aupr = auc(recall, precision)
        return aupr

    elif mode == "ROC":
        roc_auc = roc_auc_score(domain_labels, scores)
        return roc_auc


def nll_regression(
    target: ArrayLike,
    mu: ArrayLike,
    var: ArrayLike,
    epsilon: float = 1e-8,
    raw: bool = False,
) -> Union[float, NDArray[np.floating]]:
    """Compute negative log-likelihood for Gaussian regression predictions.

    Assumes predictions follow a Gaussian distribution with given mean and variance.

    Parameters
    ----------
    target : array-like
        Ground truth target values.
    mu : array-like
        Predicted mean values.
    var : array-like
        Predicted variance values.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-8.
    raw : bool, optional
        If True, return per-sample NLL; if False, return mean NLL. Default is False.

    Returns
    -------
    float or ndarray
        Mean NLL if raw=False, per-sample NLL array if raw=True.
    """
    nll = (
        (target - mu) ** 2 / (2.0 * var + epsilon)
        + np.log(var + epsilon) / 2.0
        + np.log(2 * np.pi) / 2.0
    )
    if raw:  # for individual predictions
        return nll
    return np.mean(nll)


def nll_class(
    target: ArrayLike, probs: ArrayLike, epsilon: float = 1e-10
) -> NDArray[np.floating]:
    """Compute negative log-likelihood for binary classification predictions.

    Parameters
    ----------
    target : array-like
        Ground truth binary labels (0 or 1).
    probs : array-like of shape (n_samples, 2)
        Predicted probabilities for each class.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-10.

    Returns
    -------
    ndarray
        Per-sample negative log-likelihood values.
    """
    log_p = -np.log(probs + epsilon)
    return target * log_p[:, 1] + (1 - target) * log_p[:, 0]


def ens_nll_regression(
    target: ArrayLike,
    preds: ArrayLike,
    epsilon: float = 1e-8,
    raw: bool = False,
) -> Union[float, NDArray[np.floating]]:
    """Compute ensemble negative log-likelihood for regression predictions.

    Combines predictions from multiple ensemble members using mixture-of-Gaussians.

    Parameters
    ----------
    target : array-like
        Ground truth target values.
    preds : array-like of shape (n_members, n_samples, 2)
        Predictions from ensemble members, where [:, :, 0] is mean
        and [:, :, 1] is variance.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-8.
    raw : bool, optional
        If True, return per-sample NLL; if False, return mean NLL. Default is False.

    Returns
    -------
    float or ndarray
        Mean ensemble NLL if raw=False, per-sample NLL array if raw=True.
    """
    mu = preds[:, :, 0]
    var = preds[:, :, 1]
    nll = (
        (target - mu) ** 2 / (2.0 * var + epsilon)
        + np.log(var + epsilon) / 2.0
        + np.log(2 * np.pi) / 2.0
    )
    proba = np.exp(-1 * nll)
    if raw:  # for individual predictions
        return -1 * np.log(np.mean(proba, axis=0))  # for individual predictions
    return np.mean(-1 * np.log(np.mean(proba, axis=0)))


def calc_rmse(
    preds: ArrayLike, target: ArrayLike, raw: bool = False
) -> Union[float, NDArray[np.floating]]:
    """Calculate root mean squared error between predictions and targets.

    Parameters
    ----------
    preds : array-like
        Predicted values.
    target : array-like
        Ground truth target values.
    raw : bool, optional
        If True, return per-sample squared errors; if False, return RMSE.
        Default is False.

    Returns
    -------
    float or ndarray
        RMSE if raw=False, per-sample squared errors if raw=True.
    """
    if raw:
        return (preds - target) ** 2  # for individual predictions
    return np.sqrt(np.mean((preds - target) ** 2))


def ens_rmse(
    target: ArrayLike,
    preds: ArrayLike,
    epsilon: float = 1e-8,
    raw: bool = False,
) -> Union[float, NDArray[np.floating]]:
    """Calculate RMSE for ensemble predictions using averaged means.

    Parameters
    ----------
    target : array-like
        Ground truth target values.
    preds : array-like of shape (n_members, n_samples, 2)
        Predictions from ensemble members, where [:, :, 0] is mean.
    epsilon : float, optional
        Unused parameter kept for API consistency. Default is 1e-8.
    raw : bool, optional
        If True, return per-sample squared errors; if False, return RMSE.
        Default is False.

    Returns
    -------
    float or ndarray
        RMSE if raw=False, per-sample squared errors if raw=True.
    """
    means = preds[:, :, 0]
    avg_mean = np.mean(means, axis=0)
    if raw:  # for individual predictions
        return calc_rmse(avg_mean, target, raw=True)
    return calc_rmse(avg_mean, target)
