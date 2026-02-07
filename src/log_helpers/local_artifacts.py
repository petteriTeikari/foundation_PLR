import os
import pickle

import numpy as np
import pandas as pd
from loguru import logger


def if_dicts_match(_dict1, _dict2) -> bool:
    """Check if two dictionaries match (placeholder implementation).

    Parameters
    ----------
    _dict1 : dict
        First dictionary (unused in placeholder).
    _dict2 : dict
        Second dictionary (unused in placeholder).

    Returns
    -------
    bool
        Always returns True (placeholder - TODO: implement actual comparison).
    """
    return True


def pickle_save(results, results_path, debug_load=True) -> None:
    """Save results to pickle file with optional verification.

    Parameters
    ----------
    results : object
        Data to save.
    results_path : str
        Path to save the pickle file.
    debug_load : bool, default True
        If True, reload and verify the saved file.
    """
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
) -> None:
    """Save results dictionary to pickle file.

    Removes existing file if present before saving.

    Parameters
    ----------
    results_dict : dict
        Dictionary to save.
    results_path : str
        Path for the pickle file (must have .pickle extension).
    name : str, optional
        Name for logging purposes.
    debug_load : bool, default True
        If True, verify saved file by reloading.

    Raises
    ------
    NotImplementedError
        If results_path does not have .pickle extension.
    """
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


def pickle_load(results_path) -> object:
    """Load data from pickle file.

    Parameters
    ----------
    results_path : str
        Path to pickle file.

    Returns
    -------
    object
        Loaded data.

    Raises
    ------
    Exception
        If loading fails, often due to NumPy version mismatch.
    """
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


def load_results_dict(results_path) -> dict:
    """Load results dictionary from file.

    Parameters
    ----------
    results_path : str
        Path to results file (must be .pickle).

    Returns
    -------
    dict
        Loaded results dictionary.

    Raises
    ------
    NotImplementedError
        If file is not a pickle file.
    """
    if ".pickle" in results_path:
        return pickle_load(results_path)
    else:
        raise NotImplementedError(
            "Only pickle format is supported at the moment, not {}".format(format)
        )


def save_object_to_pickle(obj, path) -> None:
    """Save any object to pickle file.

    Parameters
    ----------
    obj : object
        Object to save.
    path : str
        Output file path.
    """
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_array_as_csv(array: np.ndarray, path: str) -> None:
    """Save NumPy array as CSV file.

    Parameters
    ----------
    array : np.ndarray
        Array to save.
    path : str
        Output CSV file path.
    """
    df = pd.DataFrame(array)
    df.to_csv(path, index=False)
