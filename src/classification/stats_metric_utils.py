import warnings
from copy import deepcopy

import numpy as np
from omegaconf import DictConfig
from loguru import logger
from scipy.interpolate import interp1d

from src.classification.catboost.catboost_ensemble import ensemble_uncertainties
from src.classification.catboost.catboost_utils import (
    get_catboost_preds_from_results_for_bootstrap,
)
from src.classification.tabm.tabm_utils import get_tabm_preds_from_results_for_bootstrap
from src.stats.calibration_metrics import get_calibration_metrics
from src.stats.classifier_metrics import get_classifier_metrics
from src.stats.uncertainty_quantification import (
    uncertainty_wrapper,
    uncertainty_wrapper_from_subject_codes,
)


def interpolation_wrapper(x, y, x_new, n_samples: int, metric: str, kind="linear"):
    def clip_illegal_calibration_values(y_new):
        y_new[y_new < 0] = 0
        y_new[y_new > 1] = 1
        return y_new

    warnings.simplefilter("ignore")
    assert len(x) == len(y), "x and y must have the same length"
    if metric == "calibration_curve":
        # so few points and does not behave well
        x_new = np.linspace(0, 1, 10)
        f = interp1d(x, y, kind="linear", fill_value="extrapolate")
        y_new = f(x_new)
        y_new = clip_illegal_calibration_values(y_new)

    else:
        if len(x) > 1:
            f = interp1d(x, y, kind="linear")
            y_new = f(x_new)
        else:
            # you got only one value, not a vector
            y_new = np.repeat(y, n_samples)
    warnings.resetwarnings()

    return x_new, y_new


def bootstrap_get_array_axis_names(metric: str):
    if metric == "AUROC":
        x_name = "fpr"
        y_name = "tpr"
    elif metric == "AUPR":
        x_name = "recall"
        y_name = "precision"
    elif metric == "calibration_curve":
        # https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
        x_name = "prob_pred"
        y_name = "prob_true"
    else:
        logger.error(
            f"Unknown metric: {metric} (only AUROC, AUPR and calibration curve supported)"
        )
        raise ValueError

    return x_name, y_name


def bootstrap_interpolate_metric_arrays(arrays: dict, n_samples: int = 200) -> dict:
    arrays_out = {}
    for metric in arrays.keys():
        arrays_out[metric] = {}
        x_name, y_name = bootstrap_get_array_axis_names(metric)

        # Interpolate (actual metric)
        arrays_out[metric][x_name], arrays_out[metric][y_name] = interpolation_wrapper(
            x=arrays[metric][x_name],
            y=arrays[metric][y_name],
            x_new=np.linspace(
                arrays[metric][x_name][0], arrays[metric][y_name][-1], n_samples
            ),
            n_samples=n_samples,
            metric=metric,
        )

        # Interpolate (thresholds as well)
        if "thresholds" in arrays[metric]:
            # exist for AUROC and AUPR
            _, arrays_out[metric]["thresholds"] = interpolation_wrapper(
                x=np.linspace(0, 1, len(arrays[metric]["thresholds"])),
                y=arrays[metric]["thresholds"],
                x_new=np.linspace(0, 1, n_samples),
                n_samples=n_samples,
                metric="thresholds",
            )

    return arrays_out


def bootstrap_aggregate_arrays(
    arrays: dict, metrics_per_split: dict, main_key: str = "metrics"
):
    # For first bootstrap iteration, initialize the metrics_per_split dict
    if main_key not in metrics_per_split.keys():
        metrics_per_split[main_key] = {}
    if "arrays" not in metrics_per_split[main_key].keys():
        metrics_per_split[main_key]["arrays"] = {}

    # Go through all the scalar metrics and aggregate them, flexible, so you can add new metrics without changing this
    for metric in arrays.keys():
        if metric not in metrics_per_split[main_key]["arrays"].keys():
            # For first iteration
            metrics_per_split[main_key]["arrays"][metric] = {}
        values = arrays[metric]
        for variable in values.keys():
            array_var: np.ndarray = values[variable][:, np.newaxis]  # e,g, (200, 1)
            if variable not in metrics_per_split[main_key]["arrays"][metric].keys():
                # For first iteration, (curve_length, no_iterations)
                # print(1, ' ', metric, variable, array_var.shape)
                metrics_per_split[main_key]["arrays"][metric][variable] = array_var
            else:
                # When there is something already here
                # print(2, " ", metric, variable, array_var.shape)
                try:
                    metrics_per_split[main_key]["arrays"][metric][variable] = np.hstack(
                        (
                            metrics_per_split[main_key]["arrays"][metric][variable],
                            array_var,
                        )
                    )
                except Exception as e:
                    logger.error(f"Could not stack arrays: {e}")
                    logger.error(
                        f'previous shape: '
                        f'{metrics_per_split[main_key]["arrays"][metric][variable].shape[0]}, '
                        f'and current shape: '
                        f'{array_var.shape}'
                    )
                    raise e

    return metrics_per_split


def bootstrap_aggregate_scalars(metrics_dict: dict, metrics_per_split: dict):
    # For first bootstrap iteration, initialize the metrics_per_split dict
    if "metrics" not in metrics_per_split.keys():
        metrics_per_split["metrics"] = {}
    if "scalars" not in metrics_per_split["metrics"].keys():
        metrics_per_split["metrics"]["scalars"] = {}

    # Go through all the scalar metrics and aggregate them, flexible, so you can add new metrics without changing this
    for metric in metrics_dict["metrics"]["scalars"].keys():
        if metric not in metrics_per_split["metrics"]["scalars"].keys():
            # For first iteration
            metrics_per_split["metrics"]["scalars"][metric] = [
                metrics_dict["metrics"]["scalars"][metric]
            ]
        else:
            # When there is something already here
            metrics_per_split["metrics"]["scalars"][metric].append(
                metrics_dict["metrics"]["scalars"][metric]
            )

    return metrics_per_split


def bootstrap_aggregate_by_subject_per_split(
    arrays: dict,
    metrics_per_split: dict,
    codes_per_split: np.ndarray,
    main_key: str,
    subkey: str = "predictions",
    is_init_with_correct_codes: bool = False,
):
    # For first bootstrap iteration, initialize the metrics_per_split dict
    if main_key not in metrics_per_split.keys():
        metrics_per_split[main_key] = {}
    if "arrays" not in metrics_per_split[main_key].keys():
        metrics_per_split[main_key]["arrays"] = {}
    code_warnings = []

    # Aggregate in a dict based on the subject code. Remember that in the bootstrap resample, you
    # get the same code multiple times in each iteration, so you get less unique codes per iteration
    # than you have samples (if you are counting the number of codes added per iteration and are confused)
    for metric in arrays[subkey].keys():
        if metric not in metrics_per_split[main_key]["arrays"].keys():
            metrics_per_split[main_key]["arrays"][metric] = {}
        values: np.ndarray = arrays[subkey][metric]
        assert len(values) == len(
            codes_per_split
        ), "Values and codes must have the same length"

        for i, code in enumerate(codes_per_split):
            value = values[i]
            if code not in metrics_per_split[main_key]["arrays"][metric].keys():
                # When this code has not yet been added
                if is_init_with_correct_codes:
                    # if you are trying to add a code that is already there, you are doing something wrong
                    logger.debug(
                        "You are trying to add a code that is not in the split?, code = {}".format(
                            code
                        )
                    )
                    code_warnings.append(code)
                else:
                    # if you have an empty the dict, then you can append
                    metrics_per_split[main_key]["arrays"][metric][code] = [value]
            else:
                # Just keep on appending then over the iterations
                metrics_per_split[main_key]["arrays"][metric][code].append(value)

    if len(code_warnings) > 0:
        logger.warning(
            "You tried to add {} codes that are not in the split".format(
                len(code_warnings)
            )
        )
        logger.warning(f"Codes that are not in the split: {code_warnings}")

    return metrics_per_split


def bootstrap_aggregate_subjects(
    metrics_per_split: dict, codes_per_split: np.ndarray, split: str, preds: dict
):
    # Aggregate the predictions as well so you could get average probabilty per patient,
    # and some uncertainty quantification (aleatoric and epistemic uncertainty)?

    # Add a "dummy key" so that the array code above works for this without modifications
    arrays_preds = {"predictions": preds}
    if split == "test":
        # The subjects are always the same for each iteration
        metrics_per_split = bootstrap_aggregate_arrays(
            arrays=arrays_preds, metrics_per_split=metrics_per_split, main_key="preds"
        )
        from src.ensemble.ensemble_classification import check_metrics_iter_preds

        check_metrics_iter_preds(
            dict_arrays=metrics_per_split["preds"]["arrays"]["predictions"]
        )
    elif split == "train" or split == "val":
        # Subjects are not the same now for each iterations, so we need to aggregate them
        # by the subject code
        metrics_per_split = bootstrap_aggregate_by_subject_per_split(
            arrays=arrays_preds,
            metrics_per_split=metrics_per_split,
            codes_per_split=codes_per_split,
            main_key="preds_dict",
        )
        from src.ensemble.ensemble_classification import check_metrics_iter_preds_dict

        check_metrics_iter_preds_dict(
            dict_arrays=metrics_per_split["preds_dict"]["arrays"]
        )
    else:
        logger.error(f"Unknown split: {split}")
        raise ValueError

    return metrics_per_split


def bootstrap_metrics_per_split(
    X: np.ndarray,
    y_true: np.ndarray,
    preds: dict,
    model,
    model_name: str,
    metrics_per_split: dict,
    codes_per_split: np.ndarray,
    method_cfg: DictConfig,
    cfg: DictConfig,
    split: str,
    skip_mlflow: bool = False,
    recompute_for_ensemble: bool = False,
):
    assert len(y_true) == len(
        preds["y_pred"]
    ), "y_true and y_pred must have the same length"
    assert len(y_true) == len(
        codes_per_split
    ), "y_true and codes_per_split must have the same length"

    # Get the basic metrics that you want
    metrics_dict = get_classifier_metrics(
        y_true, preds=preds, cfg=cfg, skip_mlflow=skip_mlflow, model_name=model_name
    )

    # Note with such few samples for calibration&uncertainty, results might be crap, do after the bootstrap?
    # Doing now both
    # Get calibration metrics
    metrics_dict = get_calibration_metrics(model, metrics_dict, y_true, preds=preds)

    # Interpolate the ROC and PR curves to a shared fixed length so you can aggregate them and do stats
    interpolated_arrays = bootstrap_interpolate_metric_arrays(
        arrays=metrics_dict["metrics"]["arrays"], n_samples=method_cfg["curve_x_length"]
    )

    # aggregate the scalar metrics
    metrics_per_split = bootstrap_aggregate_scalars(
        metrics_dict=metrics_dict, metrics_per_split=metrics_per_split
    )

    # aggregate the array curves
    metrics_per_split = bootstrap_aggregate_arrays(
        arrays=interpolated_arrays, metrics_per_split=metrics_per_split
    )

    # Aggregate the subject predictions (i.e. preds)
    if not recompute_for_ensemble:
        # during a "live" bootstrapping (as in when not ensembling from results loaded from MLflow)
        # ENSEMBLING: this is basically created already by the previous ensembling functions
        metrics_per_split = bootstrap_aggregate_subjects(
            metrics_per_split=metrics_per_split,
            codes_per_split=codes_per_split,
            split=split,
            preds=preds,
        )

    return metrics_per_split


def bootstrap_predict(model, X, i, split, debug_aggregation: bool = True):
    try:
        predict_probs = model.predict_proba(X)  # (n_samples, n_classes), e.g. (72,2)
        preds = {
            "y_pred_proba": predict_probs[
                :, 1
            ],  # (n_samples,), e.g. (72,) for the class 1 (e.g. glaucoma)
            "y_pred": model.predict(X),  # (n_samples,), e.g. (72,)
        }
    except Exception as e:
        logger.error(f"Could not get prediction from the model: {e}")
        raise e

    if debug_aggregation:
        if split == "test":
            logger.info(
                f"DEBUG iter #{i+1}: 1st sample probs the test split: {preds['y_pred_proba'][0]}"
            )

    return preds


def tabm_demodata_fix(preds):

    no_pred_length = len(preds["y_pred_proba"])
    no_pred_labels = len(preds["label"])

    if no_pred_length != no_pred_labels:
        if no_pred_length == 2*no_pred_labels:
            preds["y_pred"] = preds["y_pred"][0:no_pred_labels]
            preds["y_pred_proba"] = preds["y_pred_proba"][0:no_pred_labels]
        else:
            logger.error('Number of predictions does not match number of labels')
            raise RuntimeError('Number of predictions does not match number of labels')

    return preds


def bootstrap_metrics(
    i,
    model,
    dict_splits,
    metrics,
    results_per_iter,
    method_cfg,
    cfg: DictConfig,
    debug_aggregation: bool = False,
    model_name: str = None,
):
    """
    i: int
        which bootstrap iteration, or a submodel of the ensemble
    dict_splits: dict
        test: dict
            X: np.ndarray
            y: np.ndarray
            w: np.ndarray
            codes: np.ndarray
    metrics: dict, e.g. {} on i=0
    results_per_iter, e.g. None on i=0
    """
    for split in dict_splits.keys():
        if split not in metrics:
            metrics[split] = {}
        X = dict_splits[split]["X"]
        y_true = dict_splits[split]["y"]
        if results_per_iter is None:
            # for sklearn API like models, we can do predict() here
            preds = bootstrap_predict(
                model, X, i, split, debug_aggregation=debug_aggregation
            )
        else:
            # more custom models like TabM, have already the preds done
            if model_name is not None:
                if model_name == "TabM":
                    preds = get_tabm_preds_from_results_for_bootstrap(
                        split_results=results_per_iter[split]
                    )
                elif model_name == "CATBOOST":
                    preds = get_catboost_preds_from_results_for_bootstrap(
                        split_results=results_per_iter[split], split=split
                    )
                elif model_name == "TabPFN":
                    preds = results_per_iter[split]
                else:
                    logger.error(
                        "Some novel model name for bootstrap? model_name = {}".format(
                            model_name
                        )
                    )
                    raise ValueError(
                        "Some novel model name for bootstrap? model_name = {}".format(
                            model_name
                        )
                    )

        # add the label here (if you want to plot some probs distributions as a function of label, apply threshold
        # tuning, or whatever)
        preds["label"] = y_true

        # hacky fix if you are running TabM, on demo data
        if model_name == 'TabM':
            preds = tabm_demodata_fix(preds)

        assert preds["label"].shape[0] == preds["y_pred_proba"].shape[0], (
            f"label ({preds['label'].shape[0]}) and "
            f"probs ({preds['y_pred_proba'].shape[0]}) should have the same length"
        )

        # TODO! Add the "granular label", when you add it to the input data?
        #  early, moderate, severe, etc. for glaucoma, or whatever you have

        warnings.simplefilter("ignore")
        metrics[split] = bootstrap_metrics_per_split(
            X,
            y_true,
            preds,
            model,
            model_name,
            metrics_per_split=metrics[split],
            codes_per_split=dict_splits[split]["codes"],
            method_cfg=method_cfg,
            cfg=cfg,
            split=split,
        )
        warnings.resetwarnings()

    return metrics


def get_p_from_alpha(alpha: float = 0.95):
    return np.round(
        ((1.0 - alpha) / 2.0) * 100, 1
    )  # e.g. 2.5 with alpha=0.95 (2.5 - 97.5%)


def bootstrap_scalar_stats_per_metric(values: np.ndarray, method_cfg: DictConfig):
    dict_out = {}
    dict_out["n"] = len(values)
    warnings.simplefilter("ignore")
    dict_out["mean"] = np.nanmean(values)
    dict_out["std"] = np.nanstd(values)

    # Confidence intervals
    p = get_p_from_alpha(alpha=method_cfg["alpha_CI"])
    if dict_out["std"] != 0:
        # all the values are the same, no point in trying to estimate CI, and get a warning clogging up your logs
        dict_out["ci"] = np.nanpercentile(values, [p, 100 - p])
    else:
        dict_out["ci"] = np.array((np.nan, np.nan))
    warnings.resetwarnings()

    return dict_out


def convert_inf_to_nan(values: np.ndarray):
    # vector-based for 2D arrays? instead of the loop
    if np.any(np.isinf(values)):
        if len(values.shape) == 2:
            for row in range(values.shape[0]):
                for col in range(values.shape[1]):
                    is_inf = np.isinf(values[row, col])
                    if is_inf:
                        values[row, col] = np.nan
        else:
            raise NotImplementedError(f"Not implemented, {values.shape}dim array")
    return values


def get_array_stats_per_metric(
    values: np.ndarray, method_cfg: DictConfig, inf_to_nan: bool = True
):
    if inf_to_nan:
        values = convert_inf_to_nan(values)
    warnings.simplefilter("ignore")
    dict_out = {}
    dict_out["mean"] = np.mean(values, axis=1)
    dict_out["std"] = np.mean(values, axis=1)

    # Confidence intervals
    p = round(((1.0 - method_cfg["alpha_CI"]) / 2.0) * 100, 1)
    if ~np.all(values.flatten() == values.flatten()[0]):
        dict_out["ci"] = np.nanpercentile(values, [p, 100 - p], axis=1)
    else:
        # if all the values are the same, no point in trying to estimate CI
        dict_out["ci"] = np.array((np.nan, np.nan))
    warnings.resetwarnings()
    return dict_out


def bootstrap_scalar_stats(metrics_per_split, method_cfg, split):
    metrics_out = {}
    for metric in metrics_per_split.keys():
        metrics_out[metric] = bootstrap_scalar_stats_per_metric(
            values=metrics_per_split[metric],
            method_cfg=method_cfg,
        )
    return metrics_out


def bootstrap_array_stats(metrics_per_split, method_cfg):
    metrics_out = {}
    for metric in metrics_per_split.keys():
        metrics_out[metric] = {}
        for variable in metrics_per_split[metric].keys():
            metrics_out[metric][variable] = get_array_stats_per_metric(
                values=metrics_per_split[metric][variable], method_cfg=method_cfg
            )

    return metrics_out


def check_bootstrap_probability_predictions(metrics: dict):
    def check_probs_array(probs_array, split):
        if isinstance(probs_array, list):
            # These are list if coming from train/val
            if len(probs_array) > 1:
                # (n_samples, n_bootstrap_iters)
                probs_array = np.array(probs_array)[np.newaxis, :]
            else:
                # if you run only for couple of iterations, you might have just
                # one sample per subject, and stdev will obviously be zero
                return
        # stdev = np.nanstd(probs_array, axis=1)
        probs_are_the_same = np.all(probs_array.flatten() == probs_array.flatten()[0])
        if probs_are_the_same:
            logger.warning(
                f"Class probabilities across all the {probs_array.shape[1]} "
                f"bootstrap iteration seem to be the same"
            )
            logger.warning(
                "Either your model is very deterministic on different bootstrap iters or you have a bug"
            )
            logger.error(
                "Either the model does not get updated on each iteration, "
                "you do not use the bootstrap samples?"
            )
            if np.all(probs_array == 0.5):
                logger.warning(
                    "All probabilities across the bootstrap iteration seem to be 0.5 with the model"
                    "failing to learning anything?"
                )
            logger.error(
                "Not raising an exception here as I guess with garbage outlier detection + imputation, "
                'you might get "unlearnable input data"?'
            )

    for split, split_dict in metrics.items():
        if "preds_dict" in split_dict:  # train/val
            probs_dict = split_dict["preds_dict"]["arrays"]["y_pred_proba"]
            for code, probs_array in probs_dict.items():
                check_probs_array(probs_array, split)
        elif "preds" in split_dict:
            probs_array = split_dict["preds"]["arrays"]["predictions"]["y_pred_proba"]
            check_probs_array(probs_array, split)
        else:
            logger.error("How come an error here?")
            raise ValueError


def bootstrap_compute_stats(metrics, method_cfg, call_from: str, verbose: bool = True):
    if verbose:
        logger.info("Compute Bootstrap statistics (AUROC, ROC Curves, etc.)")
    warnings.simplefilter("ignore")
    if call_from != "ts_ensemble":
        check_bootstrap_probability_predictions(metrics)
    else:
        logger.info("Skip bootstrap check")
    metrics_stats = {}
    for split in metrics.keys():
        metrics_stats[split] = {}
        metrics_stats[split]["metrics"] = {}
        if "metrics" not in metrics[split]:
            logger.info(
                f"Skip the split {split} as no metrics found (e.g. with ensembling val is skipped)"
            )
            metrics_stats[split]["metrics"]["scalars"] = None
            metrics_stats[split]["metrics"]["arrays"] = None
        else:
            metrics_stats[split]["metrics"]["scalars"] = bootstrap_scalar_stats(
                metrics_per_split=metrics[split]["metrics"]["scalars"],
                method_cfg=method_cfg,
                split=split,
            )
            metrics_stats[split]["metrics"]["arrays"] = bootstrap_array_stats(
                metrics_per_split=metrics[split]["metrics"]["arrays"],
                method_cfg=method_cfg,
            )

    warnings.resetwarnings()

    return metrics_stats


def bootstrap_subject_stats_numpy_array(
    preds_per_key: np.ndarray, labels: np.ndarray, key: str
):
    """
    args:
        preds_per_key: np.ndarray, shape (n_subjects, n_iterations)
    """
    dict_out = {}
    warnings.simplefilter("ignore")
    dict_out["mean"] = np.mean(preds_per_key, axis=1)
    dict_out["std"] = np.std(preds_per_key, axis=1)
    warnings.resetwarnings()

    return dict_out


def aggregate_dict_subjects(dict_out, stats_per_code):
    for key, scalar_in_array in stats_per_code.items():
        if key not in dict_out.keys():
            dict_out[key] = scalar_in_array
        else:
            dict_out[key] = np.hstack((dict_out[key], scalar_in_array))

    return dict_out


def bootstrap_subject_stats_dict(
    preds_per_key: dict,
    labels: np.ndarray,
    codes_train: np.ndarray,
    key: str,
    split=str,
    verbose: bool = True,
    check_preds: bool = False,
):
    warnings.simplefilter("ignore")
    dict_out = {}
    for code in preds_per_key.keys():
        array_per_code = np.array(preds_per_key[code])[np.newaxis, :]
        if check_preds:
            check_indiv_code_for_different_preds(code, list(array_per_code), key)
        stats_per_code = bootstrap_subject_stats_numpy_array(
            array_per_code, labels=labels, key=key
        )
        dict_out = aggregate_dict_subjects(dict_out, stats_per_code)

    # Uncertainty here for the train/val splits
    if key == "y_pred_proba":
        if verbose:
            logger.info(
                "Compute uncertainty quantification, split = {}, key = {}".format(
                    split, key
                )
            )
        assert len(labels) == len(
            preds_per_key
        ), "label and prediction lengths do not match"
        uq_dict = uncertainty_wrapper_from_subject_codes(
            p_mean=dict_out["mean"], p_std=dict_out["std"], y_true=labels, split=split
        )
        warnings.resetwarnings()
    else:
        uq_dict = {}

    return dict_out, uq_dict


def sort_dict_keys_based_on_list(
    dict_to_sort: dict, list_to_sort_by: list, sort_list: bool = True
):
    # sort the keys based on original train codes as you now get arrays for the stats
    dict_to_sort = dict(sorted(dict_to_sort.items()))

    if sort_list:
        # Only need to sort when you are bootstrapping, does not matter for CatBoost Ensemble
        try:
            return {k: dict_to_sort[k] for k in list_to_sort_by}
        except Exception as e:
            logger.error(f"Could not sort the dict: {e}")
            raise e
    else:
        return dict_to_sort


def bootstrap_check_that_samples_different(
    preds_per_key: dict, key: str, check_preds: bool = False
):
    for subject_code in list(preds_per_key.keys()):
        preds_per_code = preds_per_key[subject_code]
        if check_preds:
            check_indiv_code_for_different_preds(subject_code, preds_per_code, key)


def check_indiv_code_for_different_preds(
    subject_code: str, preds_per_code: list, key: str
):
    """
    Tricky check as if you would have clean data and your model predicts 0.5 for both classes, you would clearly have
    a problem. But now as we are doing this downstream analysis, the input data at worse can be garbage. Needlessly
    will raise issues then for these bad quality data runs
    """
    if key == "y_pred_proba":
        # could happen that class labels are same for tiny bootstraps?
        # or badly functioning model maybe just outputs all the classes the same?
        if len(np.unique(preds_per_code)) == 1:
            logger.warning(f"Subject {subject_code} has the same predictions")
            logger.warning(preds_per_code)
            raise ValueError


def bootstrap_compute_subject_stats(
    metrics_iter,
    dict_arrays,
    method_cfg,
    sort_list: bool = True,
    call_from: str = None,
    verbose: bool = True,
):
    if verbose:
        logger.info(
            "Compute subject-wise Bootstrap statistics (class probabiities, uncertainty quantification, etc."
        )
    subject_stats = {}
    for split in metrics_iter.keys():
        labels, _ = get_labels_and_codes(split, dict_arrays, call_from)
        subject_stats[split] = {}
        subject_stats[split]["preds"] = {}

        dict_per_split = metrics_iter[split]
        if "preds_dict" in dict_per_split.keys():
            if call_from == "CATBOOST":
                codes = dict_arrays[f"subject_codes_{split}"]
            elif call_from == "classification_ensemble":
                # examine later why the codes do not seem to match, if you want to use the classification ensembles
                # which do not seem to provide much, just use "the internal ensembling" of Tree-based methods
                codes = None  # dict_arrays[f"subject_codes_{split}"]
            else:
                codes = dict_arrays["subject_codes_train"]
            # train/val split with the subject codes
            preds_per_split = dict_per_split["preds_dict"]["arrays"]
            for key in preds_per_split.keys():
                if codes is None:
                    # don't use the metadata codes (there is some glitch?) for the ensemble
                    # use directly the prediction codes for sorting
                    codes = sorted(list(set(list(preds_per_split[key].keys()))))
                preds_per_key = sort_dict_keys_based_on_list(
                    preds_per_split[key], list(codes), sort_list=sort_list
                )
                assert len(labels) == len(preds_per_key), (
                    f"label ({len(labels)}) and pred ({len(preds_per_key)}) "
                    f"lengths do not match"
                )
                bootstrap_check_that_samples_different(preds_per_key, key)
                if key == "y_pred_proba":
                    subject_stats[split]["preds"][key], subject_stats[split]["uq"] = (
                        bootstrap_subject_stats_dict(
                            preds_per_key=preds_per_key,
                            labels=labels,
                            codes_train=codes,
                            key=key,
                            split=split,
                            verbose=verbose,
                        )
                    )
                else:
                    subject_stats[split]["preds"][key], _ = (
                        bootstrap_subject_stats_dict(
                            preds_per_key=preds_per_key,
                            labels=labels,
                            codes_train=codes,
                            key=key,
                            split=split,
                            verbose=verbose,
                        )
                    )
            subject_stats[split]["subject_code"] = codes
            subject_stats[split]["labels"] = labels
            assert len(subject_stats[split]["subject_code"]) == len(
                preds_per_key
            ), "Codes and predictions must have the same length"
            assert len(subject_stats[split]["subject_code"]) == len(
                labels
            ), "Codes and labels must have the same length"

        elif "preds" in dict_per_split.keys():
            # test split
            codes = dict_arrays["subject_codes_test"]
            preds_per_split = dict_per_split["preds"]["arrays"]["predictions"]
            for key in preds_per_split.keys():
                preds_per_key = preds_per_split[key]
                subject_stats[split]["preds"][key] = (
                    bootstrap_subject_stats_numpy_array(
                        preds_per_key=preds_per_key, labels=labels, key=key
                    )
                )
                # Uncertainty Quantification
                if key == "y_pred_proba":
                    subject_stats[split]["uq"] = uncertainty_wrapper(
                        preds=preds_per_key, y_true=labels, key=key, split=split
                    )

            subject_stats[split]["subject_code"] = codes
            subject_stats[split]["labels"] = labels
            assert len(subject_stats[split]["subject_code"]) == len(
                preds_per_key
            ), "Codes and predictions must have the same length"
            assert len(subject_stats[split]["subject_code"]) == len(
                labels
            ), "Codes and labels must have the same length"

        else:
            logger.error(f"Where are the predictions now? {dict_per_split.keys()}")
            raise ValueError

    return subject_stats


def global_subject_stats(
    values: np.ndarray,
    labels: np.ndarray,
    key: str,
    variable: str,
    method_cfg: DictConfig,
):
    dict_out = {}
    # not much point in averaging all the subject probabilities together without accounting for the label
    unique_labels = np.unique(labels)
    for label in unique_labels:
        values_per_label = values[labels == label]
        dict_out[label] = bootstrap_scalar_stats_per_metric(
            values_per_label, method_cfg=method_cfg
        )

    return dict_out


def get_labels_and_codes(split, dict_arrays, call_from):
    # These splits are now "bootstrapping splits" so the labels, codes are from the original Train
    if split == "train" or split == "val":
        if call_from is None or call_from == "classification_ensemble":
            labels, codes = dict_arrays["y_train"], dict_arrays["subject_codes_train"]
        elif call_from == "CATBOOST":
            # This is actually a call for the ensemble evaluation, the Catboost bootstrap call is handled
            # normally as the None condition above
            labels, codes = (
                dict_arrays[f"y_{split}"],
                dict_arrays[f"subject_codes_{split}"],
            )
    elif split == "test":
        labels, codes = dict_arrays["y_test"], dict_arrays["subject_codes_test"]
    else:
        logger.error(f"Unknown split: {split}")
        raise ValueError
    assert len(labels) == len(codes), "Labels and codes must have the same length"

    return labels, codes


def bootstrap_compute_global_subject_stats(
    subjectwise_stats, method_cfg, verbose: bool = True
):
    # Compute the "mean response" of the subjects, e.g. scalar mean UQ metric to describe the whole model uncertainty

    subject_global_stats = {}
    for split in subjectwise_stats.keys():
        subject_global_stats[split] = {}
        labels = subjectwise_stats[split]["labels"]
        for key in subjectwise_stats[split]["preds"].keys():
            preds_per_key = subjectwise_stats[split]["preds"][key]
            subject_global_stats[split][key] = {}
            for variable in preds_per_key.keys():
                # Note! Now we just compute stats for whatever variables you have here, not like UQ epistemic
                # uncertainty computed from y_pred (class) is necesssarily what you need, but pick the correct combos
                # when visualizing and logging to MLflow. Easier than cherrypicking here what makes sense and not
                values: np.ndarray = preds_per_key[variable]
                subject_global_stats[split][key][variable] = global_subject_stats(
                    values, labels, key, variable, method_cfg
                )

    return subject_global_stats


def compute_uq_unks_from_dict_of_subjects(probs_dict: dict):
    """
    i.e. train/val splits that do not have the same number of samples for each subject
    """
    uq = None
    iters = []
    for code, list_of_probs in probs_dict.items():
        probs_code = np.array(list_of_probs)[
            :, np.newaxis, np.newaxis
        ]  # (esize/no_iter, 1, 1)
        probs_code = np.concatenate(
            [probs_code, 1 - probs_code], axis=2
        )  # (n_samples, 1, n_classes=2)
        iters.append(probs_code.shape[0])
        uq_code: dict = ensemble_uncertainties(probs=probs_code)
        # TODO! compute AURC here as well? instead of before?
        if uq is None:
            uq = deepcopy(uq_code)
        else:
            for key in uq.keys():
                uq[key] = np.vstack((uq[key], uq_code[key]))

    assert uq["confidence"].shape[0] == len(
        probs_dict
    ), "You did not compute probs ({}) for all the subjects ({})".format(
        uq["confidence"].shape[0], len(probs_dict)
    )

    return uq


def compute_uq_for_subjectwise_stats(
    metrics_iter, subjectwise_stats, verbose: bool = True
):
    from src.classification.catboost.catboost_main import (
        combine_unks_with_subjectwise_stats,
    )

    uq = {}
    for split in metrics_iter.keys():
        if "preds_dict" in metrics_iter[split].keys():
            probs_dict = metrics_iter[split]["preds_dict"]["arrays"]["y_pred_proba"]
            assert isinstance(probs_dict, dict), "Probs must be a dictionary"
            uq[split] = compute_uq_unks_from_dict_of_subjects(probs_dict)
        elif "preds" in metrics_iter[split].keys():
            probs = metrics_iter[split]["preds"]["arrays"]["predictions"][
                "y_pred_proba"
            ]
            assert isinstance(probs, np.ndarray), "Probs must be a numpy array"
            probs = probs.T[:, :, np.newaxis]  # (n_iter, n_samples, 1)
            probs = np.concatenate(
                [probs, 1 - probs], axis=2
            )  # (n_iter, n_samples, n_classes=2)
            uq[split] = ensemble_uncertainties(probs)
        else:
            logger.error(f"Where are the predictions now? {metrics_iter[split].keys()}")
            raise ValueError

        subjectwise_stats = combine_unks_with_subjectwise_stats(
            subjectwise_stats, unks=uq[split], split=split
        )

    return subjectwise_stats
