import tempfile
from pathlib import Path

import mlflow


def test_artifact_write(mlflow_subdir="dummy_artifact"):
    # TO-OPTIMIZE! This is pretty much meaningful only when you
    # have modified the default tracking_uri, and want to see if you
    # can still log artifacts, the default one works pretty much always
    # "independent test suite will always work", import path from "cfg"?
    features = "testing that the artifact write works"
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir, "features.txt")
        path.write_text(features)
        mlflow.log_artifact(path, mlflow_subdir)

    # TODO! add auto-delete?
