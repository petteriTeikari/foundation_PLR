# Imputation

Stage 2 of the pipeline: Reconstructing missing or invalid signal segments.

## Available Methods

### Ground Truth

| Method | Description |
|--------|-------------|
| `pupil-gt` | Human-corrected ground truth signal |

### Deep Learning

| Method | Description |
|--------|-------------|
| `SAITS` | Self-Attention Imputation for Time Series |
| `CSDI` | Conditional Score-based Diffusion Imputation |
| `TimesNet` | TimesNet-based imputation |

### Foundation Models

| Method | Description |
|--------|-------------|
| `MOMENT-finetune` | MOMENT model finetuned for imputation |
| `MOMENT-zeroshot` | MOMENT zero-shot imputation |

### Traditional

| Method | Description |
|--------|-------------|
| `linear` | Linear interpolation |

### Ensembles

| Method | Description |
|--------|-------------|
| `ensemble-CSDI-MOMENT-SAITS-TimesNet` | Ensemble of deep learning methods |

## Configuration

```yaml
# In Hydra config
imputation_method: SAITS
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error vs ground truth |
| RMSE | Root Mean Square Error vs ground truth |

## API Reference

::: src.imputation.flow_imputation
    options:
      show_root_heading: true
      members: [run_imputation]
