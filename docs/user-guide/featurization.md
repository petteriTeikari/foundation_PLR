# Featurization

Stage 3 of the pipeline: Extracting handcrafted features from PLR signals.

## Feature Types

### Amplitude Bins

Histogram-based features capturing the distribution of pupil sizes:

- Baseline diameter
- Constriction amplitude (absolute and relative)
- Max constriction diameter

### Latency Features

Timing-based features:

| Feature | Description |
|---------|-------------|
| `latency_to_constriction` | Time to reach max constriction |
| `latency_75pct` | Time to reach 75% constriction |
| `time_to_redilation` | Recovery time |
| `constriction_duration` | Duration of constriction phase |

### Velocity Features

| Feature | Description |
|---------|-------------|
| `max_constriction_velocity` | Peak constriction speed |
| `mean_constriction_velocity` | Average constriction speed |
| `max_redilation_velocity` | Peak recovery speed |

### PIPR Features

Post-Illumination Pupil Response:

| Feature | Description |
|---------|-------------|
| `pipr_6s` | PIPR at 6 seconds |
| `pipr_10s` | PIPR at 10 seconds |
| `recovery_time` | Time to baseline recovery |

## Why Handcrafted Features?

!!! important "Key Finding"
    Handcrafted features outperform foundation model embeddings by **9 percentage points** (0.830 vs 0.740 AUROC).

Foundation model embeddings were tested but underperform because:

1. Generic embeddings don't capture domain-specific PLR physiology
2. Handcrafted features encode expert knowledge about glaucoma biomarkers
3. Small dataset (N=208) doesn't benefit from high-dimensional embeddings

## Configuration

```yaml
# Featurization is fixed (not configurable)
# Uses handcrafted features only
```

## API Reference

::: src.featurization.flow_featurization
    options:
      show_root_heading: true
