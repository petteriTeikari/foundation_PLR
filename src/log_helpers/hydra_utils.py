import glob
import shutil
import sys
import mlflow

from loguru import logger
import os
import hydra

from src.log_helpers.log_utils import get_datetime_as_string
from src.utils import get_artifacts_dir, get_repo_root
from omegaconf import OmegaConf
from hydra import compose, initialize


def update_hydra_ouput_dir(use_gmt_time: bool = False):
    # Fake the CLI argument (update if there is more elegant method
    # TODO! This works obviously for local repo, but it does not scale to
    #  defining the artifacts directory in the config file
    # https://stackoverflow.com/a/67720433/6412152
    # Extra background
    # https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/
    # https://github.com/facebookresearch/hydra/discussions/2819#discussioncomment-7899912
    # https://stackoverflow.com/a/70777327/6412152
    artifacts_dir = os.path.join(get_artifacts_dir(service_name="hydra"))
    date_string = get_datetime_as_string()
    artifacts_dir_string = f"hydra.run.dir={os.path.join(artifacts_dir, date_string)}"
    return artifacts_dir_string


def get_hydra_output_dir():
    try:
        return hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    except Exception:
        hydra_dir = get_artifacts_dir(service_name="hydra")
        logger.debug(
            f"Failed to get the hydra output directory (you used Compose?), using now: {hydra_dir}"
        )
        return hydra_dir


def get_intermediate_hydra_log_path():
    output_dir = get_hydra_output_dir()
    log_files = glob.glob(f"{output_dir}/*.log")
    if len(log_files) == 0:
        logger.warning("No Hydra log files found in the output directory")
        return None
    elif len(log_files) > 1:
        # TODO! Pick the latest log file
        logger.error(
            "Multiple log files found in the output directory? {}".format(log_files)
        )
        raise NotImplementedError(
            "Multiple log files found in the output directory? {}".format(log_files)
        )
    else:
        return log_files[0]


def save_hydra_cfg_as_yaml(cfg, dir_output):
    # yaml_data: str = OmegaConf.to_yaml(cfg)
    cfg_path = os.path.join(dir_output, "hydra_cfg.yaml")
    with open(cfg_path, "w") as f:
        OmegaConf.save(cfg, f)
    logger.info(f"Hydra config saved as {cfg_path}")
    return cfg_path


def get_cfg_HydraCompose(args, config_dir: str = "configs"):
    # https://stackoverflow.com/a/61169706/6412152
    # The not recommended route by Hydra, but in the end not using many of the Hydra's "automatic" features
    # TO-OPTIMIZE! Re-assess this decision maybe later?
    # https://hydra.cc/docs/advanced/compose_api/
    repo_root = get_repo_root()
    abs_config_path = os.path.join(repo_root, config_dir)
    yaml_path = os.path.join(abs_config_path, f"{args.config_file}.yaml")
    if not os.path.exists(yaml_path):
        logger.error(f"Config file not found: {abs_config_path}")
        raise FileNotFoundError(f"Config file not found: {abs_config_path}")
    else:
        logger.info(f"Using Hydra config file: {abs_config_path}")

    rel_config_path = os.path.join(
        "..", "..", config_dir
    )  # from "hydra_utils.py" directory
    with initialize(version_base=None, config_path=rel_config_path):
        cfg = compose(config_name=args.config_file)

    return cfg


def add_hydra_cli_args(args):
    # e.g. ['pipeline_PLR.py', '--config-path', '../configs', '--config-name', 'defaults.yaml']
    logger.info('Hydra config path: "{}"'.format(args.config_path))
    logger.info('Hydra config name: "{}"'.format(args.config_name))
    sys.argv.append(
        update_hydra_ouput_dir()
    )  # Hack to change the Hydra output directory
    sys.argv.append(
        "--config-path"
    )  # https://github.com/facebookresearch/hydra/issues/386
    sys.argv.append(f"{args.config_path}")
    sys.argv.append(
        "--config-name"
    )  # https://github.com/facebookresearch/hydra/issues/874
    sys.argv.append(f"{args.config_name}")
    logger.debug(sys.argv)


def log_the_hydra_log_as_mlflow_artifact(
    hydra_log, suffix: str = "_train", intermediate: bool = False
):
    if hydra_log is not None:
        dir_in, fname = os.path.split(hydra_log)
        fname, ext = os.path.splitext(fname)
        fname_out = fname + f"{suffix}" + ext
        fname_out_path = os.path.join(dir_in, fname_out)
        try:
            shutil.copy(hydra_log, fname_out_path)
        except Exception as e:
            logger.error(
                "Fail to make a local copy of the hydra log (cannot log as an artifact): {}".format(
                    e
                )
            )
            return None

        try:
            if intermediate:
                mlflow.log_artifact(
                    fname_out_path, artifact_path="hydra_logs/intermediate"
                )
            else:
                mlflow.log_artifact(fname_out_path, artifact_path="hydra_logs")
        except Exception as e:
            logger.error(f"Failed to log hydra log artifact: {e}")

        # remove the temp file after it has been registered as an artifact
        try:
            os.remove(fname_out_path)
        except Exception as e:
            logger.error(f"Failed to remove the local copy of the hydra log: {e}")

    else:
        logger.warning(
            "No hydra log found to log as an artifact (normal if you use Hydra Compose)"
        )


def log_hydra_artifacts_to_mlflow(artifacts_dir, model_name, cfg, run_name):
    # Hydra log with the suffix
    logger.debug("Logging the Hydra log to MLflow")
    hydra_log = get_intermediate_hydra_log_path()
    log_the_hydra_log_as_mlflow_artifact(
        hydra_log, suffix="_imputation", intermediate=True
    )
