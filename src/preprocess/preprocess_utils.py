from loguru import logger

import numpy as np


def compute_stats_per_split(X, split_name):
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
