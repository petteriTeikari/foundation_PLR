import shutil
import sys
from pathlib import Path

import mlflow
from loguru import logger
from mlflow.entities import ViewType

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.utils.paths import get_mlruns_dir

MLRUNS_DIR = str(get_mlruns_dir())


def get_runs_per_experiment(e):
    all_runs = mlflow.search_runs(
        experiment_ids=[e.experiment_id], run_view_type=ViewType.ALL
    )
    deleted_runs = mlflow.search_runs(
        experiment_ids=[e.experiment_id], run_view_type=ViewType.DELETED_ONLY
    )
    active_runs = mlflow.search_runs(
        experiment_ids=[e.experiment_id], run_view_type=ViewType.ACTIVE_ONLY
    )
    logger.info(
        f"All runs: {len(all_runs)}, Deleted runs: {len(deleted_runs)}, Active runs: {len(active_runs)}"
    )
    return {
        "all_runs": all_runs,
        "deleted_runs": deleted_runs,
        "active_runs": active_runs,
    }


def permanently_delete_deleted_runs(deleted_runs, mlruns_dir, dry_run: bool = True):
    """
    https://stackoverflow.com/a/63844571/6412152
    """

    def get_run_dir(artifacts_uri):
        # remove 'file:///' from the beginning and '/artifacts' from the end
        file_path = artifacts_uri.replace("file:///", "").replace("/artifacts", "")
        return file_path

    def remove_run_dir(run_dir):
        shutil.rmtree(run_dir)  # , ignore_errors=True)

    def fix_path(run_dir, mlruns_dir):
        mlruns_path = Path(mlruns_dir)
        if not mlruns_path.exists():
            logger.error(f"MLflow directory {mlruns_dir} does not exist")
            raise FileNotFoundError(f"MLflow directory {mlruns_dir} does not exist")

        # if these are originally generated on a different machine with a possibly different absolute path
        path_after_mlruns = run_dir.split("mlruns")[1][1:]
        experiment_id, run_id = path_after_mlruns.split("/")

        experiment_dir = mlruns_path / experiment_id
        if not experiment_dir.exists():
            logger.error(f"Experiment directory {experiment_dir} does not exist")
            raise FileNotFoundError(
                f"Experiment directory {experiment_dir} does not exist"
            )

        run_dir = experiment_dir / run_id
        if not run_dir.exists():
            logger.error(f"Run directory {run_dir} does not exist")
            raise FileNotFoundError(f"Run directory {run_dir} does not exist")

        return str(run_dir)

    # iterate through the Pandas Dataframe rows
    filesize = 0
    for run in deleted_runs.iterrows():
        run = run[1]
        run_dir = get_run_dir(run.artifact_uri)
        fixed_run_dir = fix_path(run_dir, mlruns_dir)
        if Path(fixed_run_dir).exists():
            # did not work the file size calculation
            nbytes = 0  # sum(d.stat().st_size for d in os.scandir(fixed_run_dir) if d.is_file())
            filesize += nbytes
        else:
            logger.warning(f"Run directory {run_dir} does not exist")

        if not dry_run:
            _ = remove_run_dir(fixed_run_dir)

    return filesize / 1024 / 1024


def clear_deleted_runs(mlruns_dir: str):
    """
    https://stackoverflow.com/questions/60088889/how-do-you-permanently-delete-an-experiment-in-mlflow
    https://github.com/mlflow/mlflow/issues/5541

    mlflow gc
    """
    mlflow.set_tracking_uri(mlruns_dir)
    exps = mlflow.search_experiments()

    for e in exps:
        logger.info(
            f"Experiment: {e.name}, ID: {e.experiment_id}, Lifecycle: {e.lifecycle_stage}"
        )
        runs = get_runs_per_experiment(e)
        _ = permanently_delete_deleted_runs(
            deleted_runs=runs["deleted_runs"], mlruns_dir=mlruns_dir, dry_run=False
        )
        logger.info(
            f"Deleted {len(runs['deleted_runs'])} runs"
        )  # , total size: {filesize_MB:.2f} MB")

    logger.info("Deletion done")


if __name__ == "__main__":
    clear_deleted_runs(mlruns_dir=MLRUNS_DIR)
