import numpy as np
from loguru import logger


def compute_stats_per_split(X, split_name):
    """Compute and log basic statistics for a data split.

    Calculates mean and standard deviation using NaN-aware functions
    and logs the results for debugging purposes.

    Parameters
    ----------
    X : np.ndarray
        Data array for which to compute statistics.
    split_name : str
        Name of the data split (e.g., 'train_gt', 'val_missing')
        used for logging context.

    Returns
    -------
    dict
        Dictionary containing 'mean' and 'std' statistics.

    Notes
    -----
    Train splits are expected to have near-perfect standardization
    (mean=0, std=1), while validation splits may deviate slightly.
    Missing data splits differ from ground truth due to masking
    applied after standardization.
    """
    stats = {
        "mean": np.nanmean(X),
        "std": np.nanstd(X),
    }
    # You would expect the train split to have "perfect standardization" (mean=0, std=1), whereas the val split
    # is slightly off as the data is slightly different. Similarly the _missing is slightly different from the _gt
    # as the missingness masking is done after the standardization.
    logger.debug(
        "Data stats for the split_key {} | mean = {}, std = {}".format(
            split_name, stats["mean"], stats["std"]
        )
    )
    return stats
