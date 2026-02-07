import warnings

import mlflow
import numpy as np
from libauc.metrics.metrics import pauc_roc_score
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)

from src.stats._defaults import DEFAULT_TPAUC_SENSITIVITY, DEFAULT_TPAUC_SPECIFICITY


def compute_tpAUC(
    y_true: np.ndarray,
    preds: dict,
    sensitivity: float = DEFAULT_TPAUC_SENSITIVITY,
    specificity: float = DEFAULT_TPAUC_SPECIFICITY,
) -> float:
    """
    Two-way Partial AUC (tpAUC)? Sensitivity and Specificity desired for glaucoma screening / case finding?
    see e.g. A novel estimator for the two-way partial AUC https://doi.org/10.1186/s12911-023-02382-2
             https://github.com/statusrank/A-Generic-Framework-for-Optimizing-Two-way-Partial-AUC
    Glaucoma desires: (No Hard guidelines atm?)
             https://github.com/petteriTeikari/glaucoma_screening/wiki
             https://pmc.ncbi.nlm.nih.gov/articles/PMC8873198/
             https://doi.org/10.1038/s41433-024-03056-7:
               "A study looking at accuracy of VGCs in
               diagnosing glaucoma showed that there was a sensitivity of 86.2% and a specificity of 82.1% [16].
               These studies ultimately demonstrated that VGCs appear to be effective."

    This code modified from:
    https://docs.libauc.org/examples/pauc_sotas.html (https://arxiv.org/abs/2306.03065)

    args:
    y_true: np.ndarray
        Ground truth labels
    preds: dict
        Predictions, including y_pred_proba
    sensitivity: float
        Desired sensitivity, classification-specific, check your NICE guidelines etc.
    specificity: float
        Desired specificity, classification-specific, check your NICE guidelines etc.

    """
    max_fpr = 1.0 - specificity
    min_tpr = sensitivity
    y_pred = preds["y_pred_proba"]  # probabilities of class 1 (glaucoma)

    # e.g. test_pauc = pauc_roc_score(test_true, test_pred, max_fpr = 0.3, min_tpr=0.7)
    # https://docs.libauc.org/examples/pauc_sotas.html
    tpAUC = pauc_roc_score(y_true, y_pred, max_fpr=max_fpr, min_tpr=min_tpr)

    # Note! As you are computing the AUC of a subregion (higher right corner of the full ROC curve),
    # you get quite small numbers compared to your "main ROC AUC" which is kinda the motivation for this
    # that you would get more clinically relevant AUC for your specific use case, see e.g. Figure 2 of
    # Neto et al. (2024): "A novel estimator for the two-way partial AUC" https://doi.org/10.1186/s12911-023-02382-2

    return tpAUC


def prevalence_adjusted_metrics(
    sensitivity: float, specificity: float, prevalence: float
):
    """
    See e.g. https://www.ncbi.nlm.nih.gov/books/NBK430867/
             https://www.ccjm.org/content/ccjom/62/5/311.full.pdf
             https://doi.org/10.1093/biostatistics/kxr008
             https://doi.org/10.1177/0272989x9601600205
             https://doi.org/10.1016/j.jclinepi.2008.04.007

    args:
    sensitivity: float
        Sensitivity
    specificity: float
        Specificity
    prevalence: float
        Prevalence of the disease in the population
    """
    dict_out = {}
    warnings.simplefilter("ignore")
    if prevalence is None:
        logger.warning("Prevalence is None, cannot compute prevalence-adjusted metrics")
    else:
        # Remember that with low prevalence the NPV goes toward 1.0 and PPV towards zero, and vice versa
        # See Eisenberg (1995) "Accuracy and predictive values" https://www.ccjm.org/content/ccjom/62/5/311.full.pdf
        dict_out["PPV_prev"] = (sensitivity * prevalence) / (
            (sensitivity * prevalence) + ((1.0 - specificity) * (1.0 - prevalence))
        )
        dict_out["NPV_prev"] = (specificity * (1.0 - prevalence)) / (
            (specificity * (1.0 - prevalence)) + ((1.0 - sensitivity) * prevalence)
        )
        # You need prevalence-adjusted accuracy really?
    warnings.resetwarnings()
    return dict_out


def check_for_glaucoma_config(cfg, skip_mlflow=False):
    """
    You need this for tpAUC and prevalence-adjusted metrics, otherwise you can just compute
    the basic ones without any knowledge of what oyu are trying to classify
    """
    if cfg is not None:
        glaucoma_params = cfg["CLS_EVALUATION"]["glaucoma_params"]
        if not skip_mlflow:
            for key, value in glaucoma_params.items():
                mlflow.log_param(f"glaucomaParam_{key}", value)
    else:
        glaucoma_params = {"prevalence": None, "sensitivity": None, "specificity": None}

    return glaucoma_params


def get_classifier_metrics(
    y_true: np.ndarray,
    preds: dict,
    cfg: DictConfig = None,
    skip_mlflow: bool = False,
    model_name: str = None,
) -> dict:
    """
    See e.g. Table 2 of http://dx.doi.org/10.1136/bjophthalmol-2021-319938
    AUC
    Sensitivity
    Specificity
    PPV ‡
    NPV ‡
    Accuracy ‡
    ‡ Values calculated at a disease prevalence of 3.54%.
    """
    warnings.simplefilter("ignore")
    glaucoma_params = check_for_glaucoma_config(cfg, skip_mlflow=skip_mlflow)

    metrics = {}
    metrics["metrics"] = {}
    metrics["metrics"]["scalars"] = {}
    metrics["metrics"]["arrays"] = {}

    # pROC for Python?
    # https://stackoverflow.com/a/53180614/6412152

    # Without any stats or anything, what you see in any data science tutorial:
    # (AU)ROC
    # keepming the key names the same for arrays and scalars, even though strictly speaking no Area Under in
    # the arrays yet computed
    try:
        metrics["metrics"]["arrays"]["AUROC"] = {}
        (
            metrics["metrics"]["arrays"]["AUROC"]["fpr"],
            metrics["metrics"]["arrays"]["AUROC"]["tpr"],
            metrics["metrics"]["arrays"]["AUROC"]["thresholds"],
        ) = roc_curve(y_true, preds["y_pred_proba"])
        metrics["metrics"]["scalars"]["AUROC"] = auc(
            metrics["metrics"]["arrays"]["AUROC"]["fpr"],
            metrics["metrics"]["arrays"]["AUROC"]["tpr"],
        )
    except Exception as e:
        logger.error(f"Could not compute AUROC: {e}")
        (
            metrics["metrics"]["arrays"]["AUROC"]["fpr"],
            metrics["metrics"]["arrays"]["AUROC"]["tpr"],
            metrics["metrics"]["arrays"]["AUROC"]["thresholds"],
        ) = (np.nan, np.nan, np.nan)
        metrics["metrics"]["scalars"]["AUROC"] = np.nan

    # Precision-Recall
    metrics["metrics"]["arrays"]["AUPR"] = {}
    try:
        (
            metrics["metrics"]["arrays"]["AUPR"]["precision"],
            metrics["metrics"]["arrays"]["AUPR"]["recall"],
            metrics["metrics"]["arrays"]["AUPR"]["thresholds"],
        ) = precision_recall_curve(y_true, preds["y_pred_proba"])
        metrics["metrics"]["scalars"]["AUPR"] = auc(
            x=metrics["metrics"]["arrays"]["AUPR"]["recall"],
            y=metrics["metrics"]["arrays"]["AUPR"]["precision"],
        )
    except Exception as e:
        logger.error(f"Could not compute AUPR: {e}")
        (
            metrics["metrics"]["arrays"]["AUPR"]["precision"],
            metrics["metrics"]["arrays"]["AUPR"]["recall"],
            metrics["metrics"]["arrays"]["AUPR"]["thresholds"],
        ) = (np.nan, np.nan, np.nan)
        metrics["metrics"]["scalars"]["AUPR"] = np.nan

    # F1
    try:
        metrics["metrics"]["scalars"]["F1"] = f1_score(
            y_true, preds["y_pred"], average="binary", zero_division=np.nan
        )
    except Exception as e:
        logger.error(f"Could not compute F1: {e}")
        metrics["metrics"]["scalars"]["F1"] = np.nan

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, preds["y_pred"])
    try:
        tn, fp, fn, tp = conf_matrix.ravel()

        # Sensitivity
        metrics["metrics"]["scalars"]["sensitivity"] = tp / (tp + fn)

        # Specificity
        metrics["metrics"]["scalars"]["specificity"] = tn / (tn + fp)

        # PPV
        metrics["metrics"]["scalars"]["PPV"] = tp / (tp + fp)

        # NPV
        metrics["metrics"]["scalars"]["NPV"] = tn / (tn + fn)

        # Accuracy
        metrics["metrics"]["scalars"]["accuracy"] = accuracy_score(
            y_true, preds["y_pred"]
        )

        # Balanced Accuracy
        metrics["metrics"]["scalars"]["accuracy_balanced"] = balanced_accuracy_score(
            y_true, preds["y_pred"]
        )

        # Prevalence-adjusted metrics
        prevalence_adjusted = prevalence_adjusted_metrics(
            sensitivity=metrics["metrics"]["scalars"]["sensitivity"],
            specificity=metrics["metrics"]["scalars"]["specificity"],
            prevalence=glaucoma_params["prevalence"],
        )
        metrics["metrics"]["scalars"] = {
            **metrics["metrics"]["scalars"],
            **prevalence_adjusted,
        }

        # Two-way Partial AUC (tpAUC)
        try:
            metrics["metrics"]["scalars"]["tpAUC"] = compute_tpAUC(
                y_true,
                preds,
                sensitivity=glaucoma_params["tpAUC_sensitivity"],
                specificity=glaucoma_params["tpAUC_specificity"],
            )
        except Exception as e:
            logger.error(f"Could not compute tpAUC: {e}")
            metrics["metrics"]["scalars"]["tpAUC"] = np.nan
    except Exception as e:
        logger.error(f"Failed to compute the confusion matrix: {e}")
        logger.error(f"Confusion matrix: {conf_matrix}")
        logger.error(f"y_true: {y_true}")
        logger.error(f"preds: {preds}")
        logger.error(
            "As in 'no confusion'? This can easily happen in debug mode with small sample sizes"
        )
        metrics["metrics"]["scalars"]["sensitivity"] = np.nan
        metrics["metrics"]["scalars"]["specificity"] = np.nan
        metrics["metrics"]["scalars"]["PPV"] = np.nan
        metrics["metrics"]["scalars"]["NPV"] = np.nan
        metrics["metrics"]["scalars"]["PPV_prev"] = np.nan
        metrics["metrics"]["scalars"]["NPV_prev"] = np.nan
        metrics["metrics"]["scalars"]["accuracy"] = np.nan
        metrics["metrics"]["scalars"]["accuracy_balanced"] = np.nan
        metrics["metrics"]["scalars"]["tpAUC"] = np.nan
    warnings.resetwarnings()
    return metrics
