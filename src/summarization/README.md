# Summarization

Experiment summarization and cross-run analysis.

## Overview

Collects classification artifacts from MLflow runs and performs summary analysis across experiment configurations. Includes data wrangling for DuckDB-based aggregation.

## Modules

| Module | Purpose |
|--------|---------|
| `flow_summarization.py` | Prefect flow entry point for summarization pipeline |
| `summarization_data_wrangling.py` | Retrieve and wrangle MLflow run data into DuckDB for aggregation |
| `summarize_classification.py` | Import classification artifacts (bootstrap metrics, baselines) from MLflow |
| `summary_analysis_main.py` | Main summary analysis entry point (placeholder/stub) |

## See Also

- `src/classification/` -- Classification runs being summarized
- `src/log_helpers/` -- MLflow artifact retrieval
