# Config → Code Mapping Reference

## How Config Values Should Be Loaded

### Hydra (Primary - for pipeline execution)

```python
@hydra.main(config_path="configs", config_name="defaults")
def main(cfg: DictConfig):
    prevalence = cfg.CLS_EVALUATION.glaucoma_params.prevalence
```

### OmegaConf (For scripts not using Hydra)

```python
from omegaconf import OmegaConf
cfg = OmegaConf.load("configs/defaults.yaml")
```

### yaml.safe_load (For visualization/analysis)

```python
import yaml
combos = yaml.safe_load(open("configs/VISUALIZATION/plot_hyperparam_combos.yaml"))
```

### Registry Module (For method names)

```python
from src.data_io.registry import get_valid_outlier_methods
methods = get_valid_outlier_methods()  # Loads from configs/mlflow_registry/
```

## Config Directory → Code Mapping

| Config Dir | Primary Consumer | Loader |
|------------|-----------------|--------|
| `configs/defaults.yaml` | Pipeline orchestration | Hydra |
| `configs/CLS_HYPERPARAMS/` | Classification training | Hydra |
| `configs/CLS_MODELS/` | Classification training | Hydra |
| `configs/MODELS/` | Imputation models | Hydra |
| `configs/OUTLIER_MODELS/` | Anomaly detection | Hydra |
| `configs/experiment/` | Experiment configs | Hydra compose |
| `configs/combos/` | Combo definitions | Hydra compose |
| `configs/VISUALIZATION/` | `src/viz/`, `src/r/` | yaml.safe_load |
| `configs/mlflow_registry/` | `src/data_io/registry.py` | yaml.safe_load |
| `configs/subjects/` | Subject selection | yaml.safe_load |
| `configs/figures_config/` | Figure dimensions/DPI | yaml.safe_load |
| `configs/PLR_FEATURIZATION/` | Feature extraction | Hydra |
| `configs/PLR_EMBEDDING/` | Embedding extraction | Hydra |
| `configs/data/` | Data paths | Hydra |
| `configs/SERVICES/` | External service config | Hydra |

## Known Config Values That MUST Come From YAML

| Value | Config Path | File |
|-------|-------------|------|
| 0.0354 | `CLS_EVALUATION.glaucoma_params.prevalence` | `defaults.yaml` |
| 1000 | `CLS_EVALUATION.BOOTSTRAP.n_iterations` | `defaults.yaml` |
| 0.95 | `CLS_EVALUATION.BOOTSTRAP.alpha_CI` | `defaults.yaml` |
| 100 | `VISUALIZATION.dpi` | `defaults.yaml` |
| 11 | Outlier method count | `mlflow_registry/parameters/classification.yaml` |
| 8 | Imputation method count | `mlflow_registry/parameters/classification.yaml` |
| 5 | Classifier count | `mlflow_registry/parameters/classification.yaml` |
