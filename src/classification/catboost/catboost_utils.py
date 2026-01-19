import warnings

import numpy as np
from loguru import logger


def rearrange_the_split(results):
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


def get_ensembled_response(dict_of_subjects: dict, p=0.05):
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


def get_ensembled_response_from_array(array_dict):
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


def convert_array_dicts_to_arrays(array_dicts) -> dict:
    """
    array_dicts: dict
        y_pred: dict (no_subjects,)
            '0': list (no_of_submodels in an ensemble)
            '1':  list (no_of_submodels in an ensemble)
            ...
        y_pred_proba_ dict
            same as above
        labels: dict
            same as above
    """
    arrays = {}
    stats_dict = {}
    for key, dict_of_subjects in array_dicts.items():
        arrays[key], stats_dict[key] = get_ensembled_response(dict_of_subjects)

    return arrays, stats_dict


def get_catboost_preds_from_results_for_bootstrap(split_results: dict, split: str):
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
