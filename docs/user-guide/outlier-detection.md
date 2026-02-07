# Outlier Detection

Stage 1 of the pipeline: Detecting artifacts and invalid samples in PLR signals.

## Available Methods

### Ground Truth

| Method | Description |
|--------|-------------|
| `pupil-gt` | Human-annotated ground truth masks |

### Foundation Models

| Method | Description |
|--------|-------------|
| `MOMENT-gt-finetune` | MOMENT model finetuned on ground truth |
| `MOMENT-gt-zeroshot` | MOMENT model zero-shot |
| `MOMENT-orig-finetune` | MOMENT finetuned on original data |
| `UniTS-gt-finetune` | UniTS model finetuned on ground truth |
| `UniTS-orig-finetune` | UniTS finetuned on original data |
| `UniTS-orig-zeroshot` | UniTS zero-shot |
| `TimesNet-gt` | TimesNet on ground truth |
| `TimesNet-orig` | TimesNet on original data |

### Traditional Methods

| Method | Description |
|--------|-------------|
| `LOF` | Local Outlier Factor |
| `OneClassSVM` | One-Class SVM |
| `SubPCA` | Subspace PCA |
| `PROPHET` | Facebook Prophet-based detection |

### Ensembles

| Method | Description |
|--------|-------------|
| `ensemble-LOF-MOMENT-...` | Voting ensemble of multiple methods |
| `ensembleThresholded-...` | Thresholded ensemble |

## Configuration

```yaml
# In Hydra config
outlier_method: MOMENT-gt-finetune
```

## API Reference

::: src.anomaly_detection.flow_anomaly_detection
    options:
      show_root_heading: true
      members: [run_anomaly_detection]
