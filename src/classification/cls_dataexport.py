import os

DIR_OUT = "/home/petteri/Dropbox/manuscriptDrafts/foundationPLR/3rdParty_repos/catboost_demo/catboost_uq"
if not os.path.exists(DIR_OUT):
    raise FileNotFoundError(f"Output directory {DIR_OUT} not found")

SUBDIR_OUT = os.path.join(DIR_OUT, "dict_arrays")
if not os.path.exists(SUBDIR_OUT):
    os.makedirs(SUBDIR_OUT)


DIR_OUT = "/home/petteri/Dropbox/manuscriptDrafts/foundationPLR/3rdParty_repos/catboost_demo/catboost_uq"
if not os.path.exists(DIR_OUT):
    raise FileNotFoundError(f"Output directory {DIR_OUT} not found")

SUBDIR_OUT = os.path.join(DIR_OUT, "dict_arrays")
if not os.path.exists(SUBDIR_OUT):
    os.makedirs(SUBDIR_OUT)


def export_dict_arrays(dict_arrays: dict):
    import pandas as pd

    # dump each array to a separate file in the output directory
    # this is useful for debugging and for further analysis
    # don't use pickle if you need this to "escape from Numpy issues" or something
    for key, arr in dict_arrays.items():
        df = pd.DataFrame(arr)
        df.to_csv(os.path.join(SUBDIR_OUT, f"{key}.csv"), index=False)


def import_dict_arrays_from_csv():
    import pandas as pd

    # load each array from a separate file in the output directory
    dict_arrays = {}
    # list files in the output directory
    files = os.listdir(SUBDIR_OUT)
    for fname in files:
        key = fname.split(".")[0]
        df = pd.read_csv(os.path.join(SUBDIR_OUT, fname))
        dict_arrays[key] = df.to_numpy()
    return dict_arrays
