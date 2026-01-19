from omegaconf import DictConfig
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV


def isotonic_calibration(i, model, cls_model_cfg: DictConfig, dict_arrays_iter: dict):
    if i == 0:
        logger.info("Calibrating classifier with isotonic calibration")
    model = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    model.fit(dict_arrays_iter["x_val"], dict_arrays_iter["y_val"])
    return model


def bootstrap_calibrate_classifier(
    i: int, model, cls_model_cfg: DictConfig, dict_arrays_iter: dict, weights_dict: dict
):
    """
    https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
    https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration.html#sphx-glr-auto-examples-calibration-plot-calibration-py
    https://www.kaggle.com/code/banddaniel/rain-pred-catboost-conformal-prediction-f1-0-84?scriptVersionId=147866075&cellId=33

    Whole another thing is whether your submodels (bootstrap iterations) need to be calibrated,
    and whether that will lead to a good ensembled performance?
    Wu and Gales (2020): "Should Ensemble Members Be Calibrated?"
    https://openreview.net/forum?id=wTWLfuDkvKp
    https://scholar.google.co.uk/scholar?cites=4462772606110879200&as_sdt=2005&sciodt=0,5&hl=en
    """

    if "CALIBRATION" in cls_model_cfg:
        if cls_model_cfg["CALIBRATION"]["method"] is None:
            if i == 0:
                logger.info("Skipping post-training calibration")
        elif cls_model_cfg["CALIBRATION"]["method"] == "isotonic":
            model = isotonic_calibration(i, model, cls_model_cfg, dict_arrays_iter)
        elif cls_model_cfg["CALIBRATION"]["method"] == "platt":
            raise NotImplementedError
            # i.e. "sigmoid" calibration
            # https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
            # https://ethen8181.github.io/machine-learning/model_selection/prob_calibration/prob_calibration.html
        elif cls_model_cfg["CALIBRATION"]["method"] == "beta":
            raise NotImplementedError
            # https://pypi.org/project/betacal/
            # https://stats.stackexchange.com/a/619981/294507
            # https://github.com/REFRAME/betacal/blob/master/python/tutorial/Python%20tutorial.ipynb
        elif cls_model_cfg["CALIBRATION"]["method"] == "conformal_platt":
            raise NotImplementedError
            # - https://github.com/aangelopoulos/conformal_classification
            # - https://github.com/aangelopoulos/conformal_classification/blob/master/example.ipynb
        else:
            logger.error(
                f"Unknown calibration method: {cls_model_cfg['CALIBRATION']['method']}"
            )
            raise ValueError(
                f"Unknown calibration method: {cls_model_cfg['CALIBRATION']['method']}"
            )
    else:
        if i == 0:
            logger.info(
                "No calibration speficied in your config, skipping post-training calibration"
            )

    return model
