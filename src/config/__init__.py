"""
Configuration module for Foundation PLR.

This module provides:
- YAMLConfigLoader: Load and validate YAML configuration files
- ExperimentConfig: Pydantic models for experiment configuration
- load_experiment: Load and validate a complete experiment configuration

Usage:
    from src.config import YAMLConfigLoader, load_experiment

    # Load combos
    loader = YAMLConfigLoader()
    combos = loader.load_combos()

    # Load full experiment
    config = load_experiment("paper_2026")
"""

from src.config.loader import YAMLConfigLoader
from src.config.experiment import (
    ExperimentConfig,
    load_experiment,
    validate_experiment_config,
    list_experiments,
)

__all__ = [
    "YAMLConfigLoader",
    "ExperimentConfig",
    "load_experiment",
    "validate_experiment_config",
    "list_experiments",
]
