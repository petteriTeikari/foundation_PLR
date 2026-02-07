# Orchestration

Workflow utilities for Prefect-based pipeline orchestration and hyperparameter sweeps.

## Overview

This module provides helper functions for configuring and running the Foundation PLR pipeline via Prefect, including hyperparameter grid/list sweeps, debug shortcuts, and the two-block reproducibility flows (extraction and analysis).

## Modules

| Module | Purpose |
|--------|---------|
| `prefect_utils.py` | Prefect and Hydra setup, logging configuration |
| `hyperparameter_sweep_utils.py` | Flatten nested configs, define sweep grids |
| `hyperparamer_list_utils.py` | Create named combos from model parameters, grid/list dispatch |
| `debug_utils.py` | Reduce bootstrap iterations for faster debugging |
| `tabm_hyperparams.py` | Extract evaluation metrics from multiple runs |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `flows/` | Top-level reproducibility flows |

### `flows/`

| Module | Purpose |
|--------|---------|
| `extraction_flow.py` | Block 1: MLflow to DuckDB with privacy separation and checkpoint/resume |
| `analysis_flow.py` | Block 2: DuckDB to figures, statistics, and LaTeX artifacts |

## See Also

- `src/pipeline_PLR.py` -- Main pipeline entry point
- `configs/` -- Hydra configuration
