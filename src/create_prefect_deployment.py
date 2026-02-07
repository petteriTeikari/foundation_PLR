# See https://docs.prefect.io/3.0/get-started/quickstart#deploy-and-schedule-your-flow
# and https://docs.prefect.io/3.0/deploy/infrastructure-examples/docker#store-your-code-in-git-based-cloud-storage
import hydra
from prefect import flow
from prefect.blocks.system import Secret
from prefect.runner.storage import GitRepository

from src.orchestration.prefect_utils import pre_check_workpool

SOURCE_REPO = "https://github.com/petteriTeikari/foundation_PLR.git"


@hydra.main(version_base=None, config_path="../configs", config_name="defaults")
def prefect_deployment(cfg):
    if cfg["PREFECT"]["WORKPOOL"]["autostart"]:
        pre_check_workpool()

    # https://docs.prefect.io/3.0/deploy/infrastructure-concepts/store-flow-code#git-based-storage
    flow.from_source(
        source=GitRepository(
            url=SOURCE_REPO,
            branch="main",
            credentials={"access_token": Secret.load("github-access-token")},
        ),
        entrypoint="pipeline_PLR.py:model_test",
    ).deploy(name="private-git-testing", work_pool_name="my-work-pool", build=False)


if __name__ == "__main__":
    prefect_deployment()
