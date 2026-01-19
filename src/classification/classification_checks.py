import numpy as np
from loguru import logger


def pre_tree_based_classifier_checks(
    xgboost_cfg,
    x_train,
    y_train,
    x_test,
    y_test,
    feature_names,
    x_train_w,
    x_test_w,
    feature_names_w,
):
    logger.info("Pre-checks for tree-based classifiers (PLACEHOLDER)")
    # e.g. SHAP value calculation does like multicollinear features, so you could do VIF check here
    # "variance_inflation_factor" is a function in statsmodels.stats.outliers_influence (see stats_utils.py)
    # https://etav.github.io/python/vif_factor_python.html
    # https://www.analyticsvidhya.com/blog/2020/03/one-hot-encoding-vs-label-encoding-using-scikit-learn/
    # https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python
    # https://datascience.stackexchange.com/a/12597/21125
    assert len(np.unique(y_train)) == 2, (
        "All the downstream code assumes that you have 2 classes (binary cls), "
        "but you had now {} classes in train".format(len(np.unique(y_train)))
    )
    assert len(np.unique(y_test)) == 2, (
        "All the downstream code assumes that you have 2 classes (binary cls), "
        "but you had now {} classes in test".format(len(np.unique(y_train)))
    )

    return (
        x_train,
        y_train,
        x_test,
        y_test,
        feature_names,
        x_train_w,
        x_test_w,
        feature_names_w,
    )
