from omegaconf import DictConfig


def pypots_data_results(data_preprocessed: dict, cfg: DictConfig):
    # Save the preprocessed data so it can be possibly be evaluated later a bit easier (?),
    # e.g. if you start mixing PyPOTS models with latest Github/arXiv magic and all the models start
    #      to expect their input to be formatted slightly differently?
    print("placeholder")
    # results["data"] = {
    #     "X_train_gt": X_train_gt,  # Denoised PLR with hand-corrected missing values
    #     "X_train_gt_missing": X_train_gt_missing,  # Denoised PLR with the missing values from _raw
    #     "X_train_raw": X_train_raw,  # What was actually measured with the actual missing values
    #     "X_val_gt": X_val_gt,
    #     "X_val_gt_missing": X_val_gt_missing,
    #     "X_val_raw": X_val_raw,
    # }
    #
    # results["data_stats"] = {}
    # for split_name in results["data"]:
    #     results["data_stats"][split_name] = compute_stats_per_split(
    #         X=results["data"][split_name], split_name=split_name
    #     )
    #
    # return results
