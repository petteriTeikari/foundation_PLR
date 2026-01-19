import pandas as pd
from loguru import logger


def convert_outlier_detection_artifacts_to_df(artifacts):
    """
    Do you actually need anything from the pickle?
    These are either on mlflow and in the df already
    train
      scalars
        accuracy
        f1
    arrays
        pred_mask
        trues
    """
    logger.info("Nothing in artifacts_dict_summary now for outlier_detection")
    return pd.DataFrame({"placeholder": [0]})


def convert_dict_of_metrics_to_df(artifacts_dict, task):
    artifacts_dict_summary_out = None
    for source_name, artifacts in artifacts_dict.items():
        if artifacts is not None:
            if task == "outlier_detection":
                artifacts_dict_summary = convert_outlier_detection_artifacts_to_df(
                    artifacts
                )
            elif task == "imputation":
                artifacts_dict_summary = pd.DataFrame({"placeholder": [0]})
            elif task == "classification":
                artifacts_dict_summary = pd.DataFrame({"placeholder": [0]})
            elif task == "featurization":
                artifacts_dict_summary = pd.DataFrame({"placeholder": [0]})
            else:
                logger.error("Task not recognized")
                raise ValueError("Task not recognized")

            if artifacts_dict_summary_out is None:
                artifacts_dict_summary_out = artifacts_dict_summary
            else:
                artifacts_dict_summary_out = pd.concat(
                    [artifacts_dict_summary_out, artifacts_dict_summary], axis=0
                )

    return artifacts_dict_summary_out
