import numpy as np
from loguru import logger
from omegaconf import DictConfig
from sklearn.feature_selection import RFE


def index_with_boolean(dict_arrays, boolean_array):
    """
    Index the dictionary of arrays with a boolean array
    """
    for key, item in dict_arrays.items():
        if isinstance(item, np.ndarray):
            if len(item.shape) == 2:
                dict_arrays[key] = dict_arrays[key][:, boolean_array]
            elif len(item.shape) == 1:
                _ = "labels"
            else:
                logger.error(f"Unknown shape of item: {item.shape}")
                raise ValueError(f"Unknown shape of item: {item.shape}")
        elif isinstance(item, list):
            dict_arrays[key] = [item[i] for i in range(len(item)) if boolean_array[i]]
        else:
            logger.error(f"Unknown type of item: {type(item)}")
            raise ValueError(f"Unknown type of item: {type(item)}")

    return dict_arrays


def xgboost_feature_selection(i, model, dict_arrays, xgboost_cfg):
    """
    See e.g.
        https://towardsdatascience.com/interpretable-machine-learning-with-xgboost
    """

    feature_importances = model.feature_importances_
    assert len(feature_importances) == len(
        dict_arrays["feature_names"]
    ), "Feature importances and feature names have different lengths"
    features_in = dict_arrays["feature_names"]
    no_features_in = len(features_in)

    areZeroImportances = np.array(feature_importances) == 0
    dict_arrays = index_with_boolean(dict_arrays, ~areZeroImportances)
    feature_importances = [
        feature_importances[i]
        for i in range(len(feature_importances))
        if list(~areZeroImportances)[i]
    ]

    logger.info(
        "After first XGBoost iteration, we are left with {} features (out of {})".format(
            len(dict_arrays["feature_names"]), no_features_in
        )
    )
    logger.info("Feature importances: {}".format(feature_importances))

    return {
        "n_features": len(dict_arrays["feature_names"]),
        "feature_names": dict_arrays["feature_names"],
        "feature_importances": feature_importances,
    }, dict_arrays


def rfe_feature_selector(model, dict_arrays: dict, xgboost_cfg: DictConfig):
    logger.info(
        f'Feature selection with RFE, params: {xgboost_cfg["FEATURE_SELECTION"]["RFE"]}'
    )
    rfe = RFE(
        estimator=model,
        n_features_to_select=xgboost_cfg["FEATURE_SELECTION"]["RFE"][
            "n_features_to_select"
        ],
    )
    rfe.fit(dict_arrays["x_train"], dict_arrays["y_train"])

    rfe_ranking = rfe.ranking_
    for i in range(len(rfe_ranking)):
        logger.info(
            f'Feature {dict_arrays["feature_names"][i]} has rank {rfe_ranking[i]}'
        )

    # Drop the undesired features
    dict_arrays = index_with_boolean(dict_arrays, rfe.support_)

    logger.info(
        "After RFE, we are left with {} features (out of {})".format(
            len(dict_arrays["feature_names"]), rfe.n_features_in_
        )
    )
    logger.info("Feature names: {}".format(dict_arrays["feature_names"]))

    return dict_arrays
