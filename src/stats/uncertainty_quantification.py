import sys

import numpy as np
from loguru import logger


def sec_classification(y_true, y_pred, conf):
    """Compute the AURC.
    Args:
    y_true: true labels, vector of size n_test
    y_pred: predicted labels by the classifier, vector of size n_test
    conf: confidence associated to y_pred, vector of size n_test
    Returns:
    conf: confidence sorted (in decreasing order)
    risk_cov: risk vs coverage (increasing coverage from 0 to 1)
    aurc: AURC
    eaurc: Excess AURC
    """
    assert len(y_true) == len(y_pred), "pred and label lengths do not match"
    n = len(y_true)
    ind = np.argsort(conf)
    y_true, y_pred, conf = y_true[ind][::-1], y_pred[ind][::-1], conf[ind][::-1]
    risk_cov = np.divide(
        np.cumsum(y_true != y_pred).astype("float"), np.arange(1, n + 1)
    )
    nrisk = np.sum(y_true != y_pred)
    aurc = np.mean(risk_cov)
    opt_aurc = (1.0 / n) * np.sum(
        np.divide(
            np.arange(1, nrisk + 1).astype("float"), n - nrisk + np.arange(1, nrisk + 1)
        )
    )
    eaurc = aurc - opt_aurc
    coverage = np.linspace(0, 1, num=len(risk_cov))

    if aurc == 0:
        logger.debug("AURC is 0, as in no risk in the model at any coverage")
        logger.debug("Is this a bug, or you got this with debug or something?")

    return coverage, risk_cov, aurc, eaurc


def risk_coverage(p_mean, p_std, y_pred, y_true):
    """
    args:
        p_mean: np.ndarray, shape (n_subjects,)
        p_std: np.ndarray, shape (n_subjects,)
        preds: np.ndarray, shape (n_subjects, n_iters in bootstrap)
        y_true: np.ndarray, shape (n_subjects,)
    """
    # AURC (Risk-Coverage curve)
    # https://github.com/IdoGalil/benchmarking-uncertainty-estimation-performance
    # Ding et al. (2020): "Revisiting the Evaluation of Uncertainty Estimation and Its Application to
    # Explore Model Complexity-Uncertainty Trade-Off"

    # the "stdev" variant of AURC
    assert len(p_mean) == len(y_pred), "pred and probs lengths do not match"
    assert len(y_true) == len(y_pred), "pred and label lengths do not match"
    conf = -p_std[np.arange(p_std.shape[0])]
    coverage, risk_cov, aurc, eaurc = sec_classification(y_true, y_pred, conf)

    return coverage, risk_cov, aurc, eaurc


def get_sample_mean_and_std(preds: np.ndarray):
    p_mean = preds.mean(axis=1)  # shape (n_subjects,)
    p_std = preds.std(axis=1)  # shape (n_subjects,)
    y_pred = (p_mean > 0.5).astype(int)  # shape (n_subjects,)

    return p_mean, p_std, y_pred


def risk_coverage_wrapper(preds: np.ndarray, y_true: np.ndarray):
    p_mean, p_std, y_pred = get_sample_mean_and_std(preds)
    coverage, risk_cov, aurc, eaurc = risk_coverage(p_mean, p_std, y_pred, y_true)

    return coverage, risk_cov, aurc, eaurc


def uncertainty_wrapper_from_subject_codes(
    p_mean: np.ndarray, p_std: np.ndarray, y_true: np.ndarray, split: str
):
    # we don't have nice equal-sized np.ndarrays from train/val bootstrap so we compute these from the mean/std
    # we could also compute these from the subject codes, but we don't have them in the current setup
    metrics = {"scalars": {}, "arrays": {}}
    assert len(y_true) == len(p_mean), "pred and label lengths do not match"
    y_pred = (p_mean > 0.5).astype(int)  # shape (n_subjects,)
    (
        metrics["arrays"]["coverage"],
        metrics["arrays"]["risk"],
        metrics["scalars"]["AURC"],
        metrics["scalars"]["AURC_E"],
    ) = risk_coverage(p_mean, p_std, y_pred, y_true)

    return metrics


def get_uncertainties(preds):
    """
    Epistemic uncertainty = Standard deviation of the Monte Carlo sample of estimated value
    Aleatoric uncertainty = Square root of the mean of the Monte Carlo sample of variance estimates
    https://shrmtmt.medium.com/beyond-average-predictions-embracing-variability-with-heteroscedastic-loss-in-deep-learning-f098244cad6f
    https://stackoverflow.com/a/63397197/6412152

    Note! that there are multiple ways to estimate epistemic and aleatoric uncertainty
    And whole another thing if the bootstrap output qualifies as a source of these
    """

    metrics = {"scalars": {}, "arrays": {}}

    # Calculating mean across multiple bootstrap iters
    mean = np.mean(preds, axis=-1)[:, np.newaxis]  # shape (n_samples, n_classes)

    # Calculating variance across multiple bootstrap iters
    # variance = np.var(preds, axis=-1)[:, np.newaxis]  # shape (n_samples, n_classes)

    # epistemic (check these)
    # metrics["scalars"]['UQ_epistemic'] = float(np.std(mean, axis=0))

    # aleatoric
    # preds_var_mean = np.mean(variance, axis=0)
    # metrics["scalars"]['UQ_aleatoric'] = float(np.sqrt(preds_var_mean))

    epsilon = sys.float_info.min
    # Calculating entropy across multiple bootstrap iters (predictive_entropy)
    # https://github.com/kyle-dorman/bayesian-neural-network-blogpost
    metrics["arrays"]["entropy"] = -np.sum(
        mean * np.log(mean + epsilon), axis=-1
    )  # shape (n_samples,)
    metrics["scalars"]["entropy"] = np.mean(metrics["arrays"]["entropy"])

    # Calculating mutual information across multiple bootstrap iters
    # The Mutual Information Metric to estimate the epistemic uncertainty of an ensemble of estimators.
    # A higher mutual information can be interpreted as a higher epistemic uncertainty.
    # https://torch-uncertainty.github.io/api.html#diversity
    metrics["arrays"]["mutual_information"] = metrics["arrays"]["entropy"] - np.mean(
        np.sum(-preds * np.log(preds + epsilon), axis=-1),
        axis=0,
    )
    metrics["scalars"]["MI"] = np.mean(metrics["arrays"]["mutual_information"])

    return metrics


def uncertainty_metrics(preds, y_true):
    # Entropy, Mutual Information, epistemic and aleatoric uncertainty
    metrics = get_uncertainties(preds)

    # AURC (Risk-Coverage curve), the std variant
    (
        metrics["arrays"]["coverage"],
        metrics["arrays"]["risk"],
        metrics["scalars"]["AURC"],
        metrics["scalars"]["AURC_E"],
    ) = risk_coverage_wrapper(preds, y_true)

    return metrics


def uncertainty_wrapper(
    preds: np.ndarray,
    y_true: np.ndarray,
    key: str,
    split: str,
    return_placeholder: bool = False,
):
    """
    See predict_and_decompose_uncertainty_tf() in uncertainty_baselines
    This implementation copied straight from there (Nado et al. 2022, https://arxiv.org/abs/2106.04015)
    epistemic uncertainty (MI), and aleatoric uncertainty (expected entropy)
    https://github.com/google/uncertainty-baselines

    See also:
        https://torch-uncertainty.github.io/api.html#diversity
        https://github.com/kyle-dorman/bayesian-neural-network-blogpost
        https://github.com/yizhanyang/Uncertainty-Estimation-BNN/blob/master/main.py
        https://github.com/yaringal/ConcreteDropout/blob/master/concrete-dropout-pytorch.ipynb
        https://github.com/rutgervandeleur/uncertainty/tree/master
        https://github.com/Kyushik/Predictive-Uncertainty-Estimation-using-Deep-Ensemble/blob/master/Ensemble_Regression_ToyData_Torch.ipynb
            means = torch.stack([tup[0] for tup in MC_samples]).view(K_test, X_val.shape[0]).cpu().data.numpy()
            logvar = torch.stack([tup[1] for tup in MC_samples]).view(K_test, X_val.shape[0]).cpu().data.numpy()
            epistemic_uncertainty = np.var(means, 0).mean(0)
            logvar = np.mean(logvar, 0)
            aleatoric_uncertainty = np.exp(logvar).mean(0)

    See also "Uncertainty in Gradient Boosting via Ensembles", https://arxiv.org/abs/2006.10562
    https://github.com/yandex-research/GBDT-uncertainty
    https://github.com/yandex-research/GBDT-uncertainty/blob/main/aggregate_results_classification.py
    - See also our tutorials on uncertainty estimation with CatBoost:
      https://towardsdatascience.com/tutorial-uncertainty-estimation-with-catboost-255805ff217e
      https://github.com/catboost/catboost/blob/master/catboost/tutorials/uncertainty/uncertainty_regression.ipynb

    Maybe a conformal prediction with the classifier could be nice?
    https://github.com/PacktPublishing/Practical-Guide-to-Applied-Conformal-Prediction/blob/main/Chapter_05_TCP.ipynb

    Args:
        preds: np.ndarray, shape (n_subjects, n_iters in bootstrap), class 1 probability (e.g. glaucoma probability)
        key: str, key for the uncertainty
        return_placeholder: bool, return placeholder if True
    """
    if return_placeholder:
        return {}
    else:
        no_subjects = preds.shape[0]
        if no_subjects > 0:
            metrics = uncertainty_metrics(preds, y_true)
            return metrics
        else:
            return {}
