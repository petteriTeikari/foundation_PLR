import torch
from loguru import logger
import subprocess
from omegaconf import DictConfig
import hydra


# TO CONTROL PREFECT LOGGING, see:
# https://discourse.prefect.io/t/how-do-i-suppress-created-task-run-logs/750/8
# Tired of getting those "[prefect.task_runs][INFO]" and "[httpx][INFO] - HTTP Request" logs
# https://github.com/PrefectHQ/prefect/issues/5952#issuecomment-1171627274
# GET CUSTOM LOGGING WORKING with PREFECT: (i.e. mix of Loguru and Prefect logging)
# https://discourse.prefect.io/t/how-to-log-messages-in-imported-classes/1805/3
# prefect config set PREFECT_LOGGING_EXTRA_LOGGERS="root"
# export PREFECT_LOGGING_ROOT_LEVEL=INFO
# https://discourse.prefect.io/t/how-to-log-messages-in-imported-classes/1805/1


def pre_flow_prefect_checks(prefect_cfg: DictConfig):
    logger.info(
        'Hydra output directory = "{}"'.format(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        )
    )

    if prefect_cfg["SERVER"]["autostart"]:
        pre_check_server()
    # Prefect results/artifacts?
    # https://docs.prefect.io/3.0/develop/results

    # Check that CUDA is available
    if torch.cuda.is_available():
        logger.info("CUDA is available")
    else:
        logger.warning("--")
        logger.warning("-----")
        logger.warning("----------")
        logger.warning("CUDA is not available! You will be training on  (Takes time!)")
        logger.warning("----------")
        logger.warning("-----")
        logger.warning("--")


def pre_check_server():
    # see https://orion-docs.prefect.io/latest/api-ref/prefect/cli/server/
    logger.debug("PREFECT SERVER AUTOSTART=True: Trying to autostart Prefect server")
    p = subprocess.Popen(
        ["nohup", "prefect", "server", "start"], stdout=subprocess.PIPE
    )
    out, err = p.communicate()  # TODO! Can jam here, why? add some timeout?
    # output = check_output(cmd, stderr=STDOUT, timeout=seconds)?
    # https://stackoverflow.com/a/12698328/6412152

    logger.debug(out)
    if err is not None:
        logger.error(err)

    if "Port 4200 is already in use" in str(out):
        logger.info("Prefect server running")
        # Dashboard by default is here http://127.0.0.1:4200/dashboard
        # TODO! display the URL to the user, with dynamic URL
        logger.info("Prefect dashboard is at http://127.0.0.1:4200/dashboard")
    else:
        logger.info("Prefect server was not running, and it was started!")


def pre_check_workpool():
    # https://docs.prefect.io/3.0/get-started/quickstart#create-a-work-pool
    workpool_name = "my-work-pool"
    logger.debug(
        "PREFECT WORKPOOL AUTOSTART=True: Trying to autostart Prefect Work Pool"
    )
    # prefect work-pool create --type process my-work-pool
    p = subprocess.Popen(
        ["nohup", "prefect", "work-pool", "create", "--type", "process", workpool_name],
        stdout=subprocess.PIPE,
    )
    out, err = p.communicate()

    if "already exists" in str(out):
        logger.info('Prefect work pool "{}" already exists'.format(workpool_name))
    else:
        logger.info("TODO!")


def post_flow_prefect_housekeeping(prefect_cfg: DictConfig):
    logger.info("Prefect housekeeping Placeholder")
    # TODO! Stop the server for example, the following does not find any running server though?
    # p = subprocess.Popen(["nohup", "prefect", "server", "stop"], stdout=subprocess.PIPE)
    # out, err = p.communicate()
