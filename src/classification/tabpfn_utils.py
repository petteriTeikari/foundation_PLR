import numpy as np


def convert_winning_probs_to_class1_probs(probs_winning, y_pred):
    probs_class1 = np.zeros_like(probs_winning)
    probs_class1[:] = np.nan
    for idx, (probs_w, y) in enumerate(zip(probs_winning, y_pred)):
        if y == 1:
            probs_class1[idx] = probs_w
        else:
            probs_class1[idx] = 1 - probs_w

    return probs_class1
