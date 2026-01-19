import os
import pickle

import numpy as np
import pandas as pd
from loguru import logger


def if_dicts_match(dict1, dict2):
    return True


def pickle_save(results, results_path, debug_load=True):
    with open(results_path, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if debug_load:
        results_loaded = load_results_dict(results_path)
        if_dicts_match(results, results_loaded)


def save_results_dict(
    results_dict: dict,
    results_path: str,
    name: str = None,
    debug_load: bool = True,
):
    if os.path.exists(results_path):
        logger.info(
            "Removing the existing results dictionary at {}".format(results_path)
        )
        os.remove(results_path)

    logger.info("Saving the {} dictionary to {}".format(name, results_path))
    if ".pickle" in results_path:
        pickle_save(results_dict, results_path, debug_load=debug_load)
    else:
        raise NotImplementedError(
            "Only pickle format is supported at the moment, not {}".format(format)
        )


def pickle_load(results_path):
    with open(results_path, "rb") as handle:
        try:
            return pickle.load(handle)
        except Exception as e:
            logger.error(
                "Could not load the results dictionary from pickle: {}".format(e)
            )
            import numpy

            logger.error("Numpy version: {}".format(numpy.__version__))
            logger.error(
                "If you get 'No module named 'numpy._core'' it might be an issue with Numpy versions?"
            )
            logger.error(
                "You saved with another Numpy version that you are trying to read them?"
            )
            logger.error(
                "TODO! Try to switch to something more platform-independent way of saving data"
            )
            logger.error("JSON? for the nested dictionaries?")
            raise e


def load_results_dict(results_path):
    if ".pickle" in results_path:
        return pickle_load(results_path)
    else:
        raise NotImplementedError(
            "Only pickle format is supported at the moment, not {}".format(format)
        )


def save_object_to_pickle(obj, path):
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_array_as_csv(array: np.ndarray, path: str):
    df = pd.DataFrame(array)
    df.to_csv(path, index=False)
