from omegaconf import DictConfig
from loguru import logger


# @task(
#     log_prints=True,
#     name="Summary Analysis Main",
#     description="...",
# )
def summary_analysis_main(flow_results: dict, cfg: DictConfig):
    logger.debug("Summary Analysis Main")
