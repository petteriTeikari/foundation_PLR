import numpy as np


def convert_winning_probs_to_class1_probs(probs_winning, y_pred):
    """
    Convert winning class probabilities to class 1 probabilities.

    TabPFN may return probability of the winning class. This converts
    to standard format where we always have probability of class 1.

    Parameters
    ----------
    probs_winning : np.ndarray
        Probabilities of the winning (predicted) class.
    y_pred : np.ndarray
        Predicted class labels (0 or 1).

    Returns
    -------
    np.ndarray
        Probabilities of class 1 for all samples.
    """
    probs_class1 = np.zeros_like(probs_winning)
    probs_class1[:] = np.nan
    for idx, (probs_w, y) in enumerate(zip(probs_winning, y_pred)):
        if y == 1:
            probs_class1[idx] = probs_w
        else:
            probs_class1[idx] = 1 - probs_w

    return probs_class1
