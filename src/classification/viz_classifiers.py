from loguru import logger


def shapley_wrapper(model, X):
    try:
        import shap
    except ImportError:
        # uv add shap gives:
        # RuntimeError: Cannot install on Python version 3.10.12; only versions >=3.6,<3.10 are supported.
        logger.error("Could not import SHAP")
        logger.error("Please install SHAP")
        return None
    # https://pypi.org/project/shap/
    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    return shap_values


def classifier_feature_importance(
    model, dict_arrays, metrics, xgboost_cfg, cfg, run_name
):
    # explainers = {}
    # explainers["SHAP"] = {}
    # explainers["SHAP"]['train'] = shapley_wrapper(model, X=metrics["x_train"])
    # explainers["SHAP"]['test'] = shapley_wrapper(model, X=metrics["x_test"])
    logger.info(
        "Explainer Placeholder! Shap does not install, nor MLflow autologs feature importance"
    )

    return {}
