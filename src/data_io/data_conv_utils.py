import numpy as np
from loguru import logger

from src.utils import get_time_vector


def long_df_to_long_numpy(df, no_features: int = 1, size_col_name: str = "gt"):
    """Convert a long-format dataframe to a 3D numpy array.

    Reshapes flat dataframe data into the format expected by PyPOTS
    and similar time series libraries: (n_subjects, n_timepoints, n_features).

    Parameters
    ----------
    df : pl.DataFrame
        Long-format Polars dataframe with all subjects' data concatenated.
    no_features : int, optional
        Number of features per timepoint, by default 1.
    size_col_name : str, optional
        Column name containing pupil size data, by default "gt".

    Returns
    -------
    np.ndarray
        3D array with shape (n_subjects, n_timepoints, n_features).

    Raises
    ------
    AssertionError
        If the data length is not evenly divisible by the expected time steps.
    Exception
        If the specified column is not found in the dataframe.
    """
    # compared to PyPOTS examples, we have only one feature (pupil size),
    # or 2 if you consider light stimuli as a separate feature
    try:
        size_df = df.select(size_col_name)
    except Exception as e:
        logger.error(e)
        logger.error(
            "Problem finding a column? Maybe you have an outdated DuckDB database?"
        )
        raise
    X = size_df.to_numpy()
    no_of_time_steps = len(get_time_vector())
    no_of_subjects = len(X) / no_of_time_steps
    assert np.mod(no_of_subjects, 1) == 0, (
        "Number of subjects is not an integer, "
        "n_timepoints = {},"
        "n_subjects = {},"
        "n_samples = {}"
    ).format(no_of_time_steps, no_of_subjects, len(X))
    no_of_subjects = int(no_of_subjects)
    X = X.reshape(no_of_subjects, no_of_time_steps, no_features)

    logger.debug(
        f"INPUT: Dataframe data shape = {df.shape} "
        f"-> OUTPUT: Numpy data shape = {X.shape} (no_subjects, no_timesteps, no_features)"
    )

    return X
