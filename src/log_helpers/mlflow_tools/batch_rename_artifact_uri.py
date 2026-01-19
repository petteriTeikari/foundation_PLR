from loguru import logger
import yaml
from pathlib import Path
from glob import glob


MLRUNS_DIR = "/home/petteri/Dropbox/manuscriptDrafts/foundationPLR/repo_PLR/src/mlruns"
INPUT_DIR = "file:///home/petteri/Dropbox/manuscriptDrafts/foundationPLR/repo_desktop_clone/foundation_PLR/src/mlruns"
REPLACE_WITH = "file://" + MLRUNS_DIR


def import_yaml_rename_and_export(
    yaml_path: str,
    string_to_replace: str,
    replace_with: str,
    cold_rename: bool = False,
):
    yaml_dict = yaml.safe_load(Path(yaml_path).read_text())
    if "artifact_location" in yaml_dict or "artifact_uri" in yaml_dict:
        if "artifact_location" in yaml_dict:
            yaml_dict["artifact_location"] = yaml_dict["artifact_location"].replace(
                string_to_replace, replace_with
            )
        elif "artifact_uri" in yaml_dict:
            yaml_dict["artifact_uri"] = yaml_dict["artifact_uri"].replace(
                string_to_replace, replace_with
            )
        if not cold_rename:
            Path(yaml_path).write_text(yaml.dump(yaml_dict))
            logger.debug(yaml_path)
            logger.debug(f"Replaced {string_to_replace} with {replace_with}")
        else:
            logger.info(yaml_path)
            logger.info(
                f"COLD RENAME: Would have Replaced {string_to_replace} with {replace_with}"
            )
        return 1
    else:
        logger.warning(f"Did not rename {yaml_path}")
        return 0


def batch_rename_meta_yaml_files(mlruns_dir: str, file_to_find: str = "meta.yaml"):
    # Find recursively in aöö the subdirectories the file_to_find and call then the import_yaml_rename_and_export
    found_files = 0
    for full_file_path in glob(f"{mlruns_dir}/**/{file_to_find}", recursive=True):
        renamed = import_yaml_rename_and_export(full_file_path, INPUT_DIR, REPLACE_WITH)
        found_files += renamed
    logger.info(f"Found {found_files} files to rename")

    # If you have had a local MLflow with a different path, you can batch rename the meta.yaml files


if __name__ == "__main__":
    batch_rename_meta_yaml_files(mlruns_dir=MLRUNS_DIR)
