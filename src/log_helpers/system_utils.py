import re
import subprocess
from platform import python_version, release, system, processor
import numpy as np
import psutil
import torch
from loguru import logger
import polars as pl


def get_commit_id(return_short: bool = True) -> str:
    def get_git_revision_hash() -> str:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )

    def get_git_revision_short_hash() -> str:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )

    # Get the current git commit id
    try:
        git_hash_short = get_git_revision_short_hash()
        git_hash = get_git_revision_hash()
    except Exception as e:
        logger.warning("Failed to get the git hash, e = {}".format(e))
        git_hash_short, git_hash = np.nan, np.nan

    if return_short:
        return git_hash_short
    else:
        return git_hash


def get_processor_info():
    model_name = np.nan

    if system() == "Windows":
        all_info = processor()
        # cpuinfo better? https://stackoverflow.com/a/62888665
        logger.warning("You need to add to Windows parsing for your CPU name")

    elif system() == "Darwin":
        all_info = subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]
        ).strip()
        logger.warning("You need to add to Mac parsing for your CPU name")

    elif system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                model_name = re.sub(".*model name.*:", "", line, 1)

    else:
        logger.warning("Unknown OS = {}, cannot get the CPU name".format(system()))

    return model_name


def get_system_params():
    # CPU/Mem

    dict = {
        "CPU": get_processor_info(),
        "RAM_GB": str(round(psutil.virtual_memory().total / (1024**3), 1)),
    }
    return dict


def get_library_versions() -> dict:
    metadata = {}
    try:
        metadata["v_Python"] = python_version()
        metadata["v_Numpy"] = np.__version__
        metadata["v_Polars"] = pl.__version__
        metadata["v_OS"] = system()
        metadata["v_OS_kernel"] = release()  # in Linux systems
        metadata["v_Torch"] = str(torch.__version__)
        # https://www.thepythoncode.com/article/get-hardware-system-information-python
    except Exception as e:
        logger.warning("Problem getting library versions, error = {}".format(e))

    try:
        metadata["v_CUDA"] = torch.version.cuda
        metadata["v_CuDNN"] = torch.backends.cudnn.version()
    except Exception as e:
        logger.warning("Problem getting CUDA library versions, error = {}".format(e))

    return metadata


def get_system_param_dict() -> dict:
    # In a way, might as well log everything, but at some point you just clutter the MLflow UI
    # You could dump this dict to a file as well and log it as an artifact?
    dict = {
        "system": get_system_params(),
        "libraries": get_library_versions(),
        "git_commit": {"git": get_commit_id()},
        # DVC commit?
    }

    return dict
