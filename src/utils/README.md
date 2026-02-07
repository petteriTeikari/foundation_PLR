# Utils Module

Utility functions and infrastructure for the Foundation PLR pipeline.

## Module Overview

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `data_mode.py` | Synthetic/production isolation | `is_synthetic_mode()`, `get_data_mode()` |
| `paths.py` | Centralized path resolution | `get_mlruns_dir()`, `get_seri_db_path()` |
| `provenance.py` | JSON metadata for reproducibility | `create_metadata()`, `compute_file_hash()` |

## data_mode.py - Synthetic Data Isolation

Central logic for the 4-gate synthetic data isolation architecture.

### Detection Hierarchy

Synthetic mode is triggered by ANY of these (in priority order):

1. **Environment variable**: `FOUNDATION_PLR_SYNTHETIC=1/true/yes`
2. **Config key**: `EXPERIMENT.is_synthetic=true`
3. **Config key**: `EXPERIMENT.experiment_prefix="synth_"`
4. **Config path**: `DATA.data_path` contains "synthetic"
5. **Filename**: contains "SYNTH_" or "synthetic"

### Constants

```python
SYNTHETIC_RUN_PREFIX = "__SYNTHETIC_"      # MLflow run name prefix
SYNTHETIC_EXPERIMENT_PREFIX = "synth_"      # MLflow experiment prefix
SYNTHETIC_ENV_VAR = "FOUNDATION_PLR_SYNTHETIC"
```

### Core Functions

| Function | Description |
|----------|-------------|
| `is_synthetic_mode()` | Check env var for synthetic mode |
| `is_synthetic_from_config(cfg)` | Check Hydra config for synthetic indicators |
| `is_synthetic_from_filename(path)` | Check filename/path for synthetic markers |
| `get_data_mode(cfg=None)` | Returns "synthetic" or "production" |

### MLflow Functions

| Function | Description |
|----------|-------------|
| `is_synthetic_run_name(name)` | Check if run name has `__SYNTHETIC_` prefix |
| `is_synthetic_experiment_name(name)` | Check if experiment has `synth_` prefix |
| `add_synthetic_prefix_to_run_name(name)` | Add prefix (idempotent) |
| `get_mlflow_tags_for_mode(synthetic)` | Get appropriate MLflow tags |
| `get_synthetic_mlflow_tags()` | Tags for synthetic runs |
| `get_production_mlflow_tags()` | Tags for production runs |

### Validation Functions

| Function | Description |
|----------|-------------|
| `validate_run_for_production_extraction(...)` | Returns False if run is synthetic |
| `validate_not_synthetic(run_name, db_path)` | Raises `SyntheticDataError` if synthetic |

### Path Functions

| Function | Description |
|----------|-------------|
| `get_synthetic_output_dir()` | Returns `outputs/synthetic/` |
| `get_synthetic_figures_dir()` | Returns `figures/synthetic/` |
| `get_results_db_path_for_mode(synthetic)` | Correct DB path by mode |
| `get_figures_dir_for_mode(synthetic)` | Correct figures dir by mode |

### Exception

```python
class SyntheticDataError(Exception):
    """Raised when synthetic data is detected in production context."""
```

### Usage Example

```python
from src.utils.data_mode import (
    is_synthetic_mode,
    get_data_mode,
    get_results_db_path_for_mode,
    validate_run_for_production_extraction,
)

# Check mode
if is_synthetic_mode():
    print("Running in synthetic mode")
    db_path = get_results_db_path_for_mode(synthetic=True)
else:
    db_path = get_results_db_path_for_mode(synthetic=False)

# Validate during extraction
is_valid = validate_run_for_production_extraction(
    run_name="LOF",
    experiment_name="PLR_Classification",
    tags={"is_synthetic": "false"}
)
```

## paths.py - Centralized Path Resolution

Single source of truth for all path access. Eliminates hardcoded paths.

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `MLRUNS_DIR` | MLflow runs directory | `~/mlruns` |
| `SERI_DB_PATH` | SERI PLR database | `<project>/SERI_PLR_GLAUCOMA.db` |
| `FOUNDATION_PLR_RESULTS_DB` | Results database | `<project>/data/public/...` |
| `PREPROCESSED_SIGNALS_DB` | Preprocessed signals | `<project>/data/...` |
| `CLASSIFICATION_EXP_ID` | MLflow experiment ID | (from MLflow) |

### Key Functions

```python
from src.utils.paths import (
    get_mlruns_dir,          # MLflow directory
    get_seri_db_path,        # SERI PLR database
    get_results_db_path,     # Extracted results DB
    get_classification_experiment_id,  # MLflow experiment
)
```

## provenance.py - Metadata for Reproducibility

Provides standard metadata format for JSON exports.

### Functions

| Function | Description |
|----------|-------------|
| `compute_file_hash(path)` | Compute truncated file hash |
| `create_metadata(generator, ...)` | Create standard metadata dict |

### Usage

```python
from src.utils.provenance import create_metadata

metadata = create_metadata(
    generator="my_script.py",
    database_path=Path("results.db"),
    table="metrics",
    query="SELECT * FROM metrics",
    description="AUROC by preprocessing method",
)

data = {
    "_metadata": metadata,
    "results": [...]
}
```

### Metadata Fields

The `create_metadata()` function returns:

```json
{
  "generated_at": "2026-02-02T12:00:00Z",
  "generator": "my_script.py",
  "database_path": "results.db",
  "database_hash": "a1b2c3d4e5f6",
  "table": "metrics",
  "query": "SELECT * FROM metrics",
  "description": "..."
}
```

## Test Coverage

Tests for this module are in:
- `tests/unit/test_data_mode.py` - 31 unit tests for isolation logic
- Integration tests reference these utilities throughout

Run tests:
```bash
pytest tests/unit/test_data_mode.py -v
```

## Related Documentation

- [README.md](../../README.md) - Project overview with isolation architecture
- [configs/data_isolation.yaml](../../configs/data_isolation.yaml) - Isolation configuration
- [ARCHITECTURE.md](../../ARCHITECTURE.md) - Full pipeline documentation
