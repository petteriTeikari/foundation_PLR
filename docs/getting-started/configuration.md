# Configuration

Foundation PLR uses [Hydra](https://hydra.cc/) for configuration management.

## Configuration Structure

```
configs/
├── defaults.yaml          # Main configuration
├── VISUALIZATION/         # Figure and plotting settings
├── mlflow_registry/       # MLflow metadata
└── ...
```

## Key Configuration Values

### Classification Parameters

```yaml
CLS_EVALUATION:
  glaucoma_params:
    prevalence: 0.0354           # Disease prevalence (Tham 2014)
    tpAUC_sensitivity: 0.862     # Target sensitivity
    tpAUC_specificity: 0.821     # Target specificity

  BOOTSTRAP:
    n_iterations: 1000           # Bootstrap iterations
    alpha_CI: 0.95               # Confidence interval level
```

### Visualization Settings

```yaml
VISUALIZATION:
  dpi: 100
  figure_format: pdf
```

## Overriding Configuration

### Command Line

```bash
# Single override
python -m src.classification.flow_classification classifier=XGBoost

# Multiple overrides
python -m src.classification.flow_classification \
    classifier=CatBoost \
    CLS_EVALUATION.BOOTSTRAP.n_iterations=500
```

### Configuration Files

Create a custom config file:

```yaml
# configs/my_experiment.yaml
defaults:
  - defaults

classifier: CatBoost
outlier_method: MOMENT-gt-finetune
imputation_method: SAITS
```

Run with:

```bash
python -m src.classification.flow_classification --config-name=my_experiment
```

## Environment Variables

Hydra supports environment variable interpolation:

```yaml
data_path: ${oc.env:DATA_PATH,/default/path}
```

## See Also

- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [OmegaConf Reference](https://omegaconf.readthedocs.io/)
