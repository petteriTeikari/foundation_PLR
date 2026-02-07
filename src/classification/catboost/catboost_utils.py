import warnings

import numpy as np
from loguru import logger


def rearrange_the_split(results: dict) -> dict:
    """
    Rearrange CatBoost ensemble results to match bootstrap structure.

    Converts from {split: {metrics}} to {metric_type: {split: metrics}}
    structure used by bootstrap evaluation.

    Parameters
    ----------
    results : dict
        Results dictionary keyed by split name.

    Returns
    -------
    dict
        Rearranged results with metric types as top-level keys.
    """
    # Match the bootstrap eval structure and add the split "inside the dict"
    splits = list(results.keys())
    split_picked = splits[0]
    results_out = results[split_picked]
    for split in results.keys():
        if split != split_picked:
            split_metrics = results[split]
            for metric_type, dict_results in split_metrics.items():
                # e.g. dict_keys(['metrics_iter', 'metrics_stats', 'subjectwise_stats', 'subject_global_stats'])
                results_out[metric_type][split] = dict_results[split]

    return results_out


def get_ensembled_response(
    dict_of_subjects: dict, p: float = 0.05
) -> tuple[np.ndarray, dict]:
    """
    Compute mean response and statistics from subject-keyed predictions.

    Aggregates predictions from multiple ensemble members per subject.

    Parameters
    ----------
    dict_of_subjects : dict
        Predictions keyed by subject index, values are lists of predictions
        from ensemble members.
    p : float, default 0.05
        Percentile for confidence interval (e.g., 0.05 for 5-95% CI).

    Returns
    -------
    tuple
        (array, stats_dict) where array is mean predictions per subject and
        stats_dict contains mean, std, CI per subject.
    """
    # TODO! get the p from the cfg rather than being hard-coded
    array, stats_dict = [], {}
    warnings.simplefilter("ignore")
    for idx, dict_of_submodels in dict_of_subjects.items():
        mean_response = np.mean(dict_of_submodels)
        array.append(mean_response)
        stats_dict[idx] = {
            "mean": mean_response,
            "std": np.std(dict_of_submodels),
            "CI": np.nanpercentile(dict_of_submodels, [p, 100 - p]),
        }
    warnings.resetwarnings()
    return np.array(array), stats_dict


def get_ensembled_response_from_array(array_dict: dict) -> tuple[dict, dict]:
    """
    Compute mean response and statistics from 2D prediction arrays.

    Parameters
    ----------
    array_dict : dict
        Dictionary with prediction keys, values are arrays of shape
        (n_subjects, n_ensemble_members).

    Returns
    -------
    tuple
        (mean_arrays, stats_dict) where mean_arrays contains mean per subject
        and stats_dict contains mean, std, CI per prediction key.
    """
    mean_arrays = {}
    stats_dict = {}
    for key, array in array_dict.items():
        mean_response = np.mean(
            array, axis=1
        )  # (no_subjects, no_submodels) -> (no_subjects)
        mean_arrays[key] = mean_response
        stats_dict[key] = {
            "mean": mean_response,
            "std": np.std(array, axis=1),
            "CI": np.nanpercentile(array, [5, 95], axis=1),
        }

    return mean_arrays, stats_dict


def convert_array_dicts_to_arrays(array_dicts: dict) -> tuple[dict, dict]:
    """
    Convert subject-keyed prediction dicts to aggregated arrays.

    Parameters
    ----------
    array_dicts : dict
        Predictions with structure:
        - y_pred: dict with subject indices as keys, lists of predictions as values
        - y_pred_proba: same structure
        - labels: same structure

    Returns
    -------
    tuple
        (arrays, stats_dict) where arrays contains mean predictions per key
        and stats_dict contains per-subject statistics.
    """
    arrays = {}
    stats_dict = {}
    for key, dict_of_subjects in array_dicts.items():
        arrays[key], stats_dict[key] = get_ensembled_response(dict_of_subjects)

    return arrays, stats_dict


def get_catboost_preds_from_results_for_bootstrap(
    split_results: dict, split: str
) -> dict:
    """
    Extract predictions from CatBoost ensemble results for bootstrap.

    Handles both dict-based (train/val) and array-based (test) prediction
    storage formats.

    Parameters
    ----------
    split_results : dict
        Results dictionary for a split with metrics_iter.
    split : str
        Split name ('train', 'val', 'test').

    Returns
    -------
    dict
        Predictions with 'y_pred_proba' and 'y_pred' keys.

    Raises
    ------
    ValueError
        If predictions not found in expected locations.
    """
    split_results = split_results["metrics_iter"][split]
    if "preds_dict" in split_results:
        array_dicts = split_results["preds_dict"]["arrays"]
        arrays, _ = convert_array_dicts_to_arrays(array_dicts)
        preds = {
            "y_pred_proba": arrays["y_pred_proba"],
            "y_pred": (arrays["y_pred_proba"] > 0.5).astype(int),
        }
    elif "preds" in split_results:
        predictions = split_results["preds"]["arrays"]["predictions"]
        arrays, _ = get_ensembled_response_from_array(array_dict=predictions)
        preds = {
            "y_pred_proba": arrays["y_pred_proba"],
            "y_pred": (arrays["y_pred_proba"] > 0.5).astype(int),
        }
    else:
        logger.error("Where are your predictions?")
        logger.error(split_results)
        raise ValueError("Where are your predictions?")

    return preds
