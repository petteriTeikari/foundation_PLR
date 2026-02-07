# Configuration Domain Context

## 🚨 CRITICAL: Registry is Single Source of Truth

**The experiment parameter registry is the GROUND TRUTH:**

- **Location**: `configs/mlflow_registry/parameters/classification.yaml`
- **Python API**: `src/data_io/registry.py`

| Parameter | Count | If Different = BROKEN |
|-----------|-------|----------------------|
| Outlier methods | **11** | Never 17, never 15, EXACTLY 11 |
| Imputation methods | **8** | |
| Classifiers | **5** | |

**NEVER parse MLflow run names. ALWAYS use the registry.**

---

## Quick Reference

All configuration uses [Hydra](https://hydra.cc/):

```bash
# Default run
python src/pipeline_PLR.py

# Override config
python src/pipeline_PLR.py --config-name=hyperparam_sweep

# Override parameters
python src/pipeline_PLR.py EXPERIMENT.debug=True MODELS=CSDI
```

## Configuration Files

| File | Purpose |
|------|---------|
| `configs/defaults.yaml` | Main configuration |
| `configs/debug_run.yaml` | Debug mode |
| `configs/hyperparam_sweep.yaml` | Hyperparameter search |

## Config Structure

```
configs/
├── defaults.yaml          # Main entry point
├── OUTLIER_MODELS/        # Outlier detection methods
├── MODELS/                # Imputation methods
├── CLS_MODELS/            # Classifiers
├── CLS_HYPERPARAMS/       # Classifier hyperparameters
├── PLR_FEATURIZATION/     # Feature extraction
├── SERVICES/              # MLflow, Prefect
├── VISUALIZATION/         # Figure settings
└── schema/                # Data schemas
```

## Hydra Composition

```yaml
# defaults.yaml
defaults:
  - OUTLIER_MODELS: MOMENT    # Loads OUTLIER_MODELS/MOMENT.yaml
  - MODELS: SAITS             # Loads MODELS/SAITS.yaml
  - CLS_MODELS: CatBoost      # Loads CLS_MODELS/CatBoost.yaml
```

## Critical Parameters

| Path | Default | Description |
|------|---------|-------------|
| `EXPERIMENT.debug` | False | Debug mode (minimal data) |
| `CLS_EVALUATION.BOOTSTRAP.n_iterations` | 1000 | Bootstrap iterations |
| `CLS_EVALUATION.glaucoma_params.prevalence` | 0.0354 | Disease prevalence |
| `DATA.filename_DuckDB` | `SERI_PLR_GLAUCOMA.db` | Data source |

## Visualization Config

For figures, use `configs/VISUALIZATION/plot_hyperparam_combos.yaml`:

```yaml
standard_combos:
  - id: ground_truth
    outlier: pupil-gt
    imputation: pupil-gt
    classifier: CatBoost
  - id: best_ensemble
    outlier: ensemble-*
    imputation: CSDI
    classifier: CatBoost
```

## Loading Config in Code

```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="defaults")
def main(cfg: DictConfig):
    # Access nested config
    n_iters = cfg.CLS_EVALUATION.BOOTSTRAP.n_iterations
```

## Rules

1. **Never hardcode** values that are in config
2. **Use standard combos** from `plot_hyperparam_combos.yaml`
3. **Extend, don't duplicate** config files
4. **Check existing config** before creating new parameters
