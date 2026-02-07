import polars as pl
from loguru import logger
from omegaconf import DictConfig

from src.data_io.data_import import import_data_from_duckdb, import_PLR_data_wrapper
from src.data_io.data_outliers import granularize_outlier_labels
from src.data_io.data_utils import check_for_data_lengths, combine_split_dataframes
from src.log_helpers.log_naming_uris_and_dirs import (
    experiment_name_wrapper,
    get_demo_string_to_add,
)
from src.utils import get_data_dir
from src.viz.viz_data_import import visualize_input_data
from src.viz.viz_utils import create_video_from_figures_on_disk


# @flow(
#     log_prints=True,
#     name="PLR Data Import",
#     description="Either the import routine from 'raw CSVs' or the 'final DuckDB file'",
# )
def flow_import_data(cfg: DictConfig) -> pl.DataFrame:
    """Import PLR data from either raw CSVs or DuckDB database.

    Main data import flow that handles loading, combining splits, optional
    debug subsetting, and visualization of PLR pupillometry data.

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary containing DATA, DEBUG, EXPERIMENT,
        and PREFECT settings.

    Returns
    -------
    pl.DataFrame
        Combined Polars dataframe with train and test splits indicated
        by the 'split' column.

    Raises
    ------
    NotImplementedError
        If debug mode is on but no subject subset is specified.
    """
    experiment_name = experiment_name_wrapper(
        experiment_name=cfg["PREFECT"]["FLOW_NAMES"]["DATA_IMPORT"], cfg=cfg
    )
    logger.info("FLOW | Name: {}".format(experiment_name))
    logger.info("=====================")
    data_dir = get_data_dir(data_path=cfg["DATA"]["data_path"])

    if cfg["DATA"]["import_from_DuckDB"]:
        # TODO! Change this a bit later, when you know if this DuckDB can be somewhere online
        df_train, df_test = import_data_from_duckdb(
            data_cfg=cfg["DATA"],
            data_dir=data_dir,
            use_demo_data=cfg["EXPERIMENT"]["use_demo_data"],
        )
        # Compute granularized outlier masks if not in database
        # (these columns are computed on-the-fly from outlier_mask)
        if "outlier_mask_easy" not in df_train.columns:
            df_train = granularize_outlier_labels(df_train, cfg)
            df_test = granularize_outlier_labels(df_test, cfg)
    else:
        # Task 1) Import the Polars dataframe
        df_train, df_test = import_PLR_data_wrapper(cfg, data_dir=data_dir)

    # check that each subject has good data lengths
    check_for_data_lengths(df_train, cfg)
    check_for_data_lengths(df_test, cfg)

    # If you have DEBUG MODE on, and you only want to use a subset of data
    # Combine splits to one dataframe with the column indicating the split
    if cfg["DEBUG"]["debug_n_subjects"] is not None:
        if get_demo_string_to_add() in experiment_name:
            demo_mode = True
        else:
            demo_mode = False

        df = combine_split_dataframes(
            df_train,
            df_test,
            cfg,
            debug_mode=cfg["EXPERIMENT"]["debug"],
            debug_n=cfg["DEBUG"]["debug_n_subjects"],
            pick_random=cfg["DEBUG"].get("pick_random", False),
            demo_mode=demo_mode,
        )
    else:
        logger.warning("DEBUG MODE is ON, but you did not take a subset of the data")
        logger.warning(
            "This typically reserved for Github Actions (or something) to run an end-to-end test"
        )
        raise NotImplementedError

    # Check that all the subjects that have equal number of samples
    check_for_data_lengths(df, cfg)

    if cfg["DATA"]["VISUALIZE"]["visualize_input_subjects"]:
        # Whether to visualize the input data or not
        # The "import_PLR_data_wrapper" contains some heuristics for data quality, but you could obviously
        # implement something more sophisticated here as well
        logger.info("Visualization of the input data")

        # Task 2) Visualize the data
        fig_paths = visualize_input_data(df=df, cfg=cfg)

        # Task 3) Create a MP4 video from the figures
        create_video_from_figures_on_disk(fig_paths=fig_paths, cfg=cfg)

    else:
        logger.info("Skipping the visualization of the input data")

    return df
