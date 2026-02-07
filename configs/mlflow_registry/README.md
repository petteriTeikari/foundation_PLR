# MLflow Registry - SINGLE SOURCE OF TRUTH

> **Quick Visual Guide** (5-second overview)

[![Registry as Single Source of Truth: Data flow diagram showing THE PROBLEM (parsing MLflow run names leads to garbage like 'anomaly'), THE SOLUTION (registry → Python code → validation tests), and WHAT HAPPENS WHEN YOU VIOLATE (wrong counts, broken figures). Shows exactly 11 outlier methods, 8 imputation methods, 5 classifiers.](../../docs/repo-figures/assets/fig-repo-55-registry-single-source-of-truth.jpg)](../../docs/repo-figures/assets/fig-repo-55-registry-single-source-of-truth.jpg)

*Figure: The registry pattern prevents garbage data from MLflow parsing. Python code MUST use `get_valid_outlier_methods()` from `src/data_io/registry.py` - NEVER parse run names. See [Hydra docs →](https://hydra.cc/) | [DuckDB docs →](https://duckdb.org/)*

## 🚨🚨🚨 CRITICAL: THIS IS THE GROUND TRUTH 🚨🚨🚨

**This directory is the SINGLE SOURCE OF TRUTH for all experiment parameters.**

All Python code MUST:
1. **LOAD parameters from this registry** - never parse MLflow run names
2. **VALIDATE against this registry** - reject anything not defined here
3. **USE these exact counts** - 11 outlier methods, 8 imputation methods, 5 classifiers

**NEVER:**
- Parse MLflow run names to discover methods
- Accept methods not listed in this registry
- Hardcode method names in scripts (import from registry instead)
- "Vibe interpret" what methods exist

**If the registry says 11 outlier methods, there are 11. Period.**

See: `src/data_io/registry.py` for the Python API to load these values.

---

## Purpose

This directory provides the **canonical definition** of all experiment parameters. Instead of scanning `/home/petteri/mlruns/`, all code MUST read from these YAML files to:

1. **Get valid method names** - The ONLY valid outlier/imputation methods
2. **Understand parameter space** - What combinations are valid
3. **Validate extraction** - Reject runs not matching registry

## Contents

```
mlflow_registry/
├── experiments.yaml          # Experiment IDs and run counts
├── metrics/
│   ├── classification.yaml   # 26 classification metrics
│   ├── imputation.yaml       # 4 imputation metrics
│   └── outlier_detection.yaml # 4 outlier metrics
├── parameters/
│   ├── classification.yaml   # Pipeline and hyperparameters
│   ├── imputation.yaml       # Model and training params
│   └── outlier_detection.yaml # Model configurations
└── README.md                 # This file
```

## Usage Examples

### Python: Load available metrics
```python
import yaml
from pathlib import Path

registry = Path("configs/mlflow_registry")
with open(registry / "metrics/classification.yaml") as f:
    metrics = yaml.safe_load(f)

# Get all discrimination metrics
disc_metrics = metrics["metrics"]["discrimination"]
print(list(disc_metrics.keys()))  # ['AUROC', 'AUPR', 'tpAUC']
```

### Python: Get all unique outlier detection methods
```python
with open(registry / "parameters/classification.yaml") as f:
    params = yaml.safe_load(f)

outlier_methods = params["pipeline"]["anomaly_source"]["values"]
print(f"Found {len(outlier_methods)} outlier detection methods")
```

### Bash: Quick lookup
```bash
# List all experiments
cat configs/mlflow_registry/experiments.yaml | grep "id:"

# Find classification metrics
cat configs/mlflow_registry/metrics/classification.yaml | grep "display_name:"
```

## Updating the Registry

The registry was generated from MLflow on 2026-01-22. If new experiments are run:

1. **Manual update**: Edit the YAML files directly
2. **Script update** (future): Run `scripts/update_mlflow_registry.py`

Since no new experiments are planned, these files should remain stable.

## Metric Naming Conventions

| Pattern | Example | Meaning |
|---------|---------|---------|
| `metric` | `AUROC` | Base metric |
| `metric_CI_hi` | `AUROC_CI_hi` | Upper CI bound |
| `metric_CI_lo` | `AUROC_CI_lo` | Lower CI bound |
| `metric__easy` | `f1__easy` | Easy difficulty subset |
| `split/metric` | `test/mae` | Split-specific |

## See Also

- `configs/VISUALIZATION/metrics.yaml` - Metric combos for visualization
- `configs/VISUALIZATION/plot_hyperparam_combos.yaml` - Standard plotting combos
- `.claude/domains/mlflow-experiments.md` - Claude context file
