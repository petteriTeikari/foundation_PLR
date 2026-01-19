import numpy as np
from tqdm import tqdm
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
from src.anomaly_detection.TSB_AD.utils.slidingWindows import find_length_rank


def run_Unsupervise_AD(model_name, data, **kwargs):
    if model_name == "SubPCA" or model_name == "EIF":
        results = tsb_ab_wrapper(model_name, data, **kwargs)
    else:
        error_message = f"Model function '{model_name}' is not defined."
        logger.error(error_message)
        raise NotImplementedError(error_message)

    return results


def tsb_ab_wrapper(model_name, data, subject_wise: bool = True, **kwargs):
    """
    The original TSB-AD codebase assumes univariate timeseries data with no multiple timeseries
    So we just loop these timeseries subject-by-subject now
    """
    no_samples, no_time_steps = data.shape
    scores_out = np.zeros_like(data)

    if subject_wise:
        for i in tqdm(range(no_samples), desc=f"TSB-AD Wrapper: {model_name}"):
            if model_name == "SubPCA":
                scores_out[i, :] = run_Sub_PCA(data=data[i, :][:, np.newaxis], **kwargs)
            elif model_name == "EIF":
                scores_out[i, :] = run_EIF(data=data[i, :][:, np.newaxis], **kwargs)
            else:
                logger.error(f"Model {model_name} not supported")
                raise ValueError(f"Model {model_name} not supported'")
    else:
        logger.error("Dataset-wise TSB-AD not implemented yet")
        raise NotImplementedError("Dataset-wise TSB-AD not implemented yet")

    return scores_out


def run_Sub_PCA(data, periodicity=1, n_components=None, n_jobs=1):
    from src.anomaly_detection.TSB_AD.models.PCA import PCA

    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = PCA(slidingWindow=slidingWindow, n_components=n_components)
    clf.fit(data)
    score = clf.decision_scores_
    score = (
        MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    )
    return score


def run_EIF(data, n_trees=100):
    from .models.EIF import EIF

    clf = EIF(n_trees=n_trees)
    clf.fit(data)
    score = clf.decision_scores_
    first_score = score[0]
    all_the_same = all([x == first_score for x in score])
    if all_the_same:
        logger.error("All the scores are the same for each timepoint")
        raise ValueError("All the scores are the same for each timepoint")
    else:
        score = (
            MinMaxScaler(feature_range=(0, 1))
            .fit_transform(score.reshape(-1, 1))
            .ravel()
        )

    return score
