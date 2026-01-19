from loguru import logger
from omegaconf import DictConfig


# @flow(
#     log_prints=True,
#     name="Model Deployment",
#     description="Put the trained models to production",
# )
def flow_deployment(cfg: DictConfig):
    deploy = False
    logger.info(
        "Placeholder for model deployment, "
        "you could log some of the trained models to MLflow model registry?"
    )
    logger.info(
        "e.g. all the best models of different tasks so you could have an end-to-end pipeline?"
    )
    if deploy:
        logger.info("Placeholder for model deployment")
        # # Subflow 5) Model Deployment
        # deploy_artifact = deploy_PLR_models(
        #     imputation_artifacts=imputation_artifacts,
        #     classification_artifacts=classification_artifacts,
        #     cfg=cfg,
        #     name=name,
        # )
