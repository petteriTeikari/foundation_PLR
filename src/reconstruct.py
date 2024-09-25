"""Console script for src."""
import argparse
import os
import sys
import logging

import hydra
import polars as pl
import pandas as pd
from omegaconf import DictConfig, OmegaConf

# Import environment variables
from dotenv import load_dotenv
read_ok = load_dotenv()
if not read_ok:
    logging.warning('Could not read .env file!')
else:
    logging.info('.env file imported successfully.')

src_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.split(src_path)[0]
sys.path.insert(
    0, project_path
)
from src.data_io.data_import import import_data
from src.models.model_main import model_data


@hydra.main(version_base=None, config_path="../configs", config_name="defaults")
def main(cfg: DictConfig) -> None:

    # Import data
    df_train, df_val, df_subset, input_dir, time_vec = import_data(cfg)

    # Define artifact location
    output_dir = os.path.join(project_path, 'output')
    logging.info('Output directory for the artifacts = {}'.format(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    # Model the data
    model_data(df_train, df_val, df_subset, time_vec, output_dir, cfg)


if __name__ == "__main__":
    main()  # pragma: no cover
