from loguru import logger
from omegaconf import DictConfig

from src.classification.classification_ts.ts_cls_Moment import ts_cls_moment_main


# @task(
#     log_prints=True,
#     name="PLR Classifiers (One Source)",
#     description="Train classifiers for one source",
# )
def train_ts_classifier(
    source_name: str,
    features_per_source: dict,
    cls_model_name: str,
    run_name: str,
    cfg: DictConfig,
    cls_model_cfg: DictConfig,
):
    if "MOMENT" in cls_model_name:
        ts_cls_moment_main(
            source_name,
            features_per_source,
            cls_model_name,
            cls_model_cfg,
            run_name,
            cfg,
        )
    else:
        logger.error(f"Unknown classifier model: {cls_model_name}")
        raise ValueError(f"Unknown classifier model: {cls_model_name}")
