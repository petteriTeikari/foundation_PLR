from loguru import logger


def eval_results_housekeeping(results, cfg):
    """
    Housekeeping for the evaluation results
    """
    # Placeholder for the results
    results_clean = {}

    # Remove the placeholder results
    for model_name, model_results in results.items():
        if len(model_results) != 0:
            results_clean[model_name] = model_results
        else:
            logger.warning(
                "Model {} had no results, "
                "just a placeholder in your training loop, or is this a bug".format(
                    model_name
                )
            )

    return results_clean
