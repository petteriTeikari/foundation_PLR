# Data I/O (`src/data_io/`)

Data import, export, and transformation utilities.

[![DuckDB archival database schema diagram for the Foundation PLR pupillary light reflex analysis pipeline, showing the essential_metrics table with 406 experiment configurations and 5 child tables (dca_curves, calibration_curves, retention_metrics, cohort_metrics, distribution_stats) that store STRATOS-compliant metrics extracted from MLflow runs](../../docs/repo-figures/assets/fig-repo-57-duckdb-schema-diagram.jpg)](../../docs/repo-figures/assets/fig-repo-57-duckdb-schema-diagram.jpg)

*Figure: DuckDB Archival Database Schema — the single portable artifact (~50 MB) replacing ephemeral MLflow runs (~200 GB). All figures, statistics, and LaTeX macros are generated from this database. See [schema plan](../../docs/repo-figures/figure-plans/fig-repo-57-duckdb-schema-diagram.md) for details.*

## Overview

This module handles:
- Loading PLR data from DuckDB databases
- Exporting results to DuckDB/parquet
- Data format conversions
- Subject stratification

## Module Structure

```
data_io/
├── __init__.py
├── flow_data.py                 # Prefect flow orchestration
├── data_import.py               # Import from various sources
├── duckdb_export.py             # Export to DuckDB
├── streaming_duckdb_export.py   # Streaming export for large data
│
├── data_wrangler.py             # Data transformation
├── data_utils.py                # General utilities
├── data_conv_utils.py           # Format conversion
├── ts_format.py                 # Time series formatting
├── torch_data.py                # PyTorch data utilities
│
├── data_imputation.py           # Imputation data handling
├── data_outliers.py             # Outlier data handling
│
├── define_sources_for_flow.py   # Source definition for pipeline
├── metadata_from_xlsx.py        # Metadata extraction
└── stratification_utils.py      # Train/test stratification
```

## Key Functions

### Data Import

```python
from src.data_io.flow_data import flow_import_data

# Load data for pipeline
df = flow_import_data(cfg=cfg)
```

### DuckDB Operations

```python
from src.data_io.duckdb_export import export_to_duckdb

# Export results
export_to_duckdb(
    df=results_df,
    db_path='results.duckdb',
    table_name='metrics'
)
```

### Streaming Export

```python
from src.data_io.streaming_duckdb_export import StreamingDuckDBExporter

# For large data with progress tracking
with StreamingDuckDBExporter('output.duckdb') as exporter:
    for batch in data_batches:
        exporter.write_batch(batch)
```

### Stratification

```python
from src.data_io.stratification_utils import stratified_split

# Split preserving class balance
train_df, test_df = stratified_split(
    df,
    test_size=0.3,
    stratify_columns=['class_label', 'no_outliers_bins']
)
```

## Data Sources

### Primary Database

```
/home/petteri/Dropbox/github-personal/foundation-PLR/SERI_PLR_GLAUCOMA.db
```

**Tables:**
- `train` - Training subjects
- `test` - Test subjects

**Key columns:**
- `pupil_raw` - Raw PLR signal
- `pupil_gt` - Ground truth denoised signal
- `outlier_mask` - Ground truth outlier mask
- `imputation_mask` - Imputation mask
- `class_label` - Classification label (control/glaucoma)

### Subject Counts

| Class | Count | Used For |
|-------|-------|----------|
| Control | 152 | Classification |
| Glaucoma | 56 | Classification |
| Unknown | 299 | Preprocessing only |
| **Total** | 507 | All |

## Configuration

Configure data import in `configs/defaults.yaml`:

```yaml
DATA:
  import_from_DuckDB: True
  filename_DuckDB: 'SERI_PLR_GLAUCOMA.db'
  PLR_length: 1981  # Timepoints per signal

STRATIFICATION:
  test_size: 0.3
  random_state: 42
  stratify_columns:
    - 'class_label'
    - 'no_outliers_bins'
```

## Data Flow

```
SERI_PLR_GLAUCOMA.db
        │
        ▼ data_import.py
    pl.DataFrame
        │
        ├──▶ Outlier Detection
        │
        ├──▶ Imputation
        │
        ├──▶ Featurization
        │
        └──▶ Classification
                │
                ▼ duckdb_export.py
        results.duckdb
```

## Format Utilities

### Time Series Formatting

```python
from src.data_io.ts_format import format_ts_for_model

# Format for PyPOTS/MOMENT
X_formatted = format_ts_for_model(
    signals=signals,
    masks=masks,
    model_type='pypots'
)
```

### Data Wrangling

```python
from src.data_io.data_wrangler import (
    get_subject_dict_for_featurization,
    convert_subject_dict_of_arrays_to_df,
)

# Get subject data in dict format
subject_dict = get_subject_dict_for_featurization(df, subject_code)

# Convert back to DataFrame
df = convert_subject_dict_of_arrays_to_df(subject_dict)
```

## See Also

- [ARCHITECTURE.md](../../ARCHITECTURE.md) - Pipeline overview
- [configs/defaults.yaml](../../configs/defaults.yaml) - Data configuration
