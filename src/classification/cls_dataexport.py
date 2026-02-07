"""Data export utilities for classification arrays.

This module exports classification arrays to CSV for external analysis.
Uses environment variable CLS_EXPORT_DIR to specify the output directory.

Usage:
    export CLS_EXPORT_DIR=/path/to/export/dir
    python -c "from src.classification.cls_dataexport import export_dict_arrays; ..."
"""

import os
from pathlib import Path

import numpy as np

# Use environment variable for portable path configuration
# Falls back to a directory in the project if not set
_default_export_dir = Path(__file__).parent.parent.parent / "outputs" / "cls_export"
DIR_OUT = Path(os.environ.get("CLS_EXPORT_DIR", str(_default_export_dir)))

# Create output directories if they don't exist
DIR_OUT.mkdir(parents=True, exist_ok=True)
SUBDIR_OUT = DIR_OUT / "dict_arrays"
SUBDIR_OUT.mkdir(parents=True, exist_ok=True)


def export_dict_arrays(dict_arrays: dict[str, np.ndarray]) -> None:
    import pandas as pd

    # dump each array to a separate file in the output directory
    # this is useful for debugging and for further analysis
    # don't use pickle if you need this to "escape from Numpy issues" or something
    for key, arr in dict_arrays.items():
        df = pd.DataFrame(arr)
        df.to_csv(SUBDIR_OUT / f"{key}.csv", index=False)


def import_dict_arrays_from_csv() -> dict[str, np.ndarray]:
    import pandas as pd

    # load each array from a separate file in the output directory
    dict_arrays: dict[str, np.ndarray] = {}
    # list files in the output directory
    for fpath in SUBDIR_OUT.iterdir():
        key = fpath.stem
        df = pd.read_csv(fpath)
        dict_arrays[key] = df.to_numpy()
    return dict_arrays
