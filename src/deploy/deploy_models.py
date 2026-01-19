from loguru import logger

from omegaconf import DictConfig


# @task(
#     log_prints=True,
#     name="Dummy task",
#     description="Placeholder",
# )
def dummy_deploy_task():
    logger.debug("PLACEHOLDER: Deploying the trained models")


# @flow(
#     log_prints=True,
#     name="PLR Model Deployment",
#     description="Deploy the trained imputation and classifier models. BentoML and FastAPI?",
# )
def deploy_PLR_models(
    imputation_artifacts: dict,
    classification_artifacts: dict,
    cfg: DictConfig,
    name: str,
):
    logger.debug("PLACEHOLDER: Deploying the trained models")
