from loguru import logger

import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_PLR_data(
    X,
    preprocess_cfg,
    preprocess_dict: dict = None,
    data_filtering: str = "gt",
    split: str = "train",
):
    if preprocess_dict is None:
        preprocess_dict = {}

    if preprocess_cfg["standardize"]:
        use_precomputed, mean, std, filterkey = if_use_precomputed(
            preprocess_dict, preprocess_cfg, split, data_filtering
        )
        if use_precomputed:
            X = standardize_with_precomputed_stats(
                X, preprocess_dict, data_filtering, filterkey, split
            )
        else:
            preprocess_dict, X = compute_stats_and_standardize(
                preprocess_dict, X, data_filtering, split
            )

    logger.debug(
        'Number of NaNs in the "{}" data: {}'.format(data_filtering, np.isnan(X).sum())
    )

    return X, preprocess_dict


def if_use_precomputed(preprocess_dict, preprocess_cfg, split, data_filtering):
    mean, std, filterkey = None, None, "gt"
    if len(preprocess_dict) == 0:
        return False, mean, std, filterkey
    elif "standardize" in preprocess_dict:
        if preprocess_cfg["use_gt_stats_for_raw"]:
            logger.debug("Use mean&stdev from GT for raw data")
            if "gt" in preprocess_dict["standardize"]:
                mean = preprocess_dict["standardize"]["gt"]["mean"]
                std = preprocess_dict["standardize"]["gt"]["std"]
                log_stats_msg(mean, std, split, data_filtering, "precomputed")
                filterkey = "gt"
                return True, mean, std, filterkey
            else:
                return False, mean, std, filterkey
        else:
            raise NotImplementedError("Not implemented yet")
            # if data_filtering in preprocess_dict["standardize"]:
            #     mean = preprocess_dict["standardize"][data_filtering]["mean"]
            #     std = preprocess_dict["standardize"][data_filtering]["std"]
            #     log_stats_msg(mean, std, split, data_filtering, "precomputed")
            #     filterkey = data_filtering
            #     return True, mean, std, filterkey
            # else:
            #     return False, mean, std, filterkey


def log_stats_msg(mean, std, split, data_filtering, call_from="precomputed"):
    if call_from == "precomputed":
        string = "Mean&Std already precomputed"
    elif call_from == "standardize":
        string = "STATS after standardization"
    else:
        raise NotImplementedError("Unknown call_from = {}".format(call_from))

    logger.debug(
        "{}: mean = {}, std = {}, split = {}, data_filtering = {}".format(
            string, mean, std, split, data_filtering
        )
    )


def standardize_with_precomputed_stats(
    X, preprocess_dict, data_filtering, filterkey, split
):
    X = (X - preprocess_dict["standardize"][filterkey]["mean"]) / preprocess_dict[
        "standardize"
    ][filterkey]["std"]
    logger.debug(
        "Data has been standardized, mean = {}, std = {}".format(
            np.nanmean(X), np.nanstd(X)
        )
    )
    return X


def compute_stats_and_standardize(
    preprocess_dict: dict, X: np.ndarray, data_filtering, split
):
    no_samples = X.shape[0] * X.shape[1]
    scaler = StandardScaler()
    scaler.fit(X.reshape(no_samples, -1))
    X = scaler.transform(X.reshape(no_samples, -1)).reshape(X.shape)
    preprocess_dict["standardize"] = {}
    print_stdz_stats(scaler, split, data_filtering)

    if "standardize" not in preprocess_dict:
        preprocess_dict["standardize"] = {}

    if data_filtering not in preprocess_dict["standardize"]:
        preprocess_dict["standardize"][data_filtering] = {}

    preprocess_dict["standardize"][data_filtering]["mean"] = float(scaler.mean_)
    preprocess_dict["standardize"][data_filtering]["std"] = float(scaler.scale_)

    log_stats_msg(np.nanmean(X), np.nanstd(X), split, data_filtering, "standardize")

    return preprocess_dict, X


def print_stdz_stats(scaler, split, data_filtering):
    if split == "train" and data_filtering == "gt":
        # Print only once the standardized stats to reduce clutter
        logger.info(
            "Standardized (split = {}, split_key = {}), mean = {}, std = {}".format(
                split, data_filtering, scaler.mean_, scaler.scale_
            )
        )
    else:
        logger.debug(
            "Standardized, mean = {}, std = {}".format(scaler.mean_, scaler.scale_)
        )


def debug_triplet_stats(X_gt, X_gt_missing, X_raw, split):
    def stats_per_split(X, split):
        logger.debug(
            "{}: mean = {}, std = {}, no_NaN = {}".format(
                split, np.nanmean(X), np.nanstd(X), np.isnan(X).sum()
            )
        )
        return {"mean": np.nanmean(X), "std": np.nanstd(X), "no_NaN": np.isnan(X).sum()}

    logger.debug("DEBUG FOR THE 'FILTERING TRIPLET', split = {}:".format(split))
    stats_per_split(X_gt, "GT")
    stats_per_split(X_gt_missing, "GT_MISSING")
    stats_per_split(X_raw, "RAW")

    return None


def destandardize_for_imputation_metric(
    targets: np.ndarray, predictions: np.ndarray, stdz_dict: dict
):
    if stdz_dict["standardized"]:
        targets = destandardize_numpy(targets, stdz_dict["mean"], stdz_dict["stdev"])
        predictions = destandardize_numpy(
            predictions, stdz_dict["mean"], stdz_dict["stdev"]
        )

    return targets, predictions


def destandardize_dict(imputation_dict: dict, mean, std):
    logger.debug(
        "De-standardizing the imputation results with mean = {} and std = {}".format(
            mean, std
        )
    )
    imputation_dict["mean"] = imputation_dict["mean"] * std + mean
    # TODO! Also for the confidence intervals (CI)
    return imputation_dict


def destandardize_numpy(X, mean, std):
    logger.debug(
        "De-standardizing the imputation results with mean = {} and std = {}".format(
            mean, std
        )
    )
    return X * std + mean


def destandardize_for_imputation_metrics(targets, predictions, preprocess_dict):
    predictions_mean = np.nanmean(predictions)
    targets_mean = np.nanmean(targets)
    predictions_larger_ratio = abs(predictions_mean) / abs(targets_mean)
    if predictions_larger_ratio > 100:
        logger.debug(
            "Predictions are larger than targets by a factor of {}".format(
                predictions_larger_ratio
            )
        )
        logger.debug(
            "It seems that your predictions are inverse transformed (destandardized) and targets are not"
        )
        logger.debug(
            "Check if you have destandardized the predictions and targets correctly"
        )
        logger.debug("Destandardizing now the targets as well for you")
        targets = destandardize_numpy(
            targets,
            preprocess_dict["standardization"]["mean"],
            preprocess_dict["standardization"]["stdev"],
        )
    else:
        if preprocess_dict["standardization"]["standardized"]:
            targets = destandardize_numpy(
                targets,
                preprocess_dict["standardization"]["mean"],
                preprocess_dict["standardization"]["stdev"],
            )
            predictions = destandardize_numpy(
                predictions,
                preprocess_dict["standardization"]["mean"],
                preprocess_dict["standardization"]["stdev"],
            )

    return targets, predictions
