# MODELS - Imputation Model Configurations

> **Quick Visual Guide** (5-second overview)

[![Imputation Model Landscape: The 8 registry-defined methods from simple to complex - Ground Truth (pupil-gt), Traditional (linear), Deep Learning (SAITS, CSDI, TimesNet), Foundation Models (MOMENT-finetune, MOMENT-zeroshot), and Ensemble. Shows complexity spectrum and config locations.](../../docs/repo-figures/assets/fig-repo-54-imputation-model-landscape.jpg)](../../docs/repo-figures/assets/fig-repo-54-imputation-model-landscape.jpg)

*Figure: Imputation model landscape showing the 8 registry methods organized by complexity. Foundation models can operate in zero-shot mode (no training required). See [SAITS paper →](https://arxiv.org/abs/2202.08516) | [CSDI paper →](https://arxiv.org/abs/2107.03502) | [MOMENT paper →](https://arxiv.org/abs/2402.03885)*

## Purpose

Configures imputation methods that reconstruct missing PLR signal segments after outlier detection. This is the second preprocessing stage before feature extraction.

## The 8 Registry Methods

**Registry source**: `configs/mlflow_registry/parameters/classification.yaml`

| Category | Method | Description |
|----------|--------|-------------|
| **Ground Truth** | `pupil-gt` | Human-denoised baseline |
| **Deep Learning** | `SAITS` | Self-attention imputation |
| | `CSDI` | Diffusion-based imputation |
| | `TimesNet` | Multi-period reconstruction |
| **Foundation Model** | `MOMENT-finetune` | MOMENT fine-tuned on PLR |
| | `MOMENT-zeroshot` | MOMENT zero-shot transfer |
| **Simple** | `linear` | Linear interpolation baseline |
| **Ensemble** | `ensemble-CSDI-MOMENT-SAITS-TimesNet` | 4-method combination |

## Files

| File | Model | Approach |
|------|-------|----------|
| `SAITS.yaml` | SAITS | Self-attention |
| `CSDI.yaml` | CSDI | Diffusion models |
| `TimesNet.yaml` | TimesNet | Multi-period transform |
| `MOMENT.yaml` | MOMENT | Foundation model |
| `MISSFOREST.yaml` | MissForest | Random forest (experimental) |
| `NuwaTS.yaml` | NuwaTS | (experimental) |

## Method Details

### SAITS (Self-Attention-based Imputation for Time Series)

Du et al. 2023 ESWA

- **Architecture**: Transformer with diagonally-masked self-attention
- **Key idea**: Jointly learns feature correlations and temporal dependencies
- **Strengths**: Handles arbitrary missing patterns, no need for explicit missingness model

```yaml
SAITS:
  n_layers: 2
  d_model: 256
  d_ffn: 128
  n_heads: 4
  d_k: 64
  d_v: 64
  dropout: 0.1
  epochs: 100
  patience: 10
```

### CSDI (Conditional Score-based Diffusion Models for Imputation)

Tashiro et al. 2021 NeurIPS

- **Architecture**: Score-based diffusion model
- **Key idea**: Learns to denoise from pure noise, conditioned on observed values
- **Strengths**: Captures complex distributions, provides uncertainty estimates

```yaml
CSDI:
  target_strategy: 'random'  # or 'hybrid', 'historical'
  num_steps: 50
  schedule: 'quad'
  channels: 64
  nheads: 8
  layers: 4
```

### TimesNet

Wu et al. 2023 ICLR

- **Architecture**: Multi-period 2D convolution
- **Key idea**: Transforms 1D series to 2D using FFT-based period finding
- **Strengths**: Captures both intra-period and inter-period variations

```yaml
TimesNet:
  seq_len: 96
  label_len: 48
  pred_len: 96
  d_model: 64
  d_ff: 64
  e_layers: 2
  top_k: 5  # Top-k periods to consider
```

### MOMENT

Goswami et al. 2024

- **Architecture**: Pre-trained time-series transformer
- **Modes**: `finetune` (PLR-adapted) or `zeroshot` (direct transfer)
- **Key idea**: Foundation model pre-trained on diverse time series
- **Strengths**: Zero-shot capability, robust to domain shift

```yaml
MOMENT:
  model_name: 'MOMENT-1-large'
  task_name: 'imputation'
  mode: 'finetune'  # or 'zeroshot'
  max_epochs: 100
  learning_rate: 1e-4
```

### Linear Interpolation

Simple baseline method.

- **Algorithm**: Linear interpolation between known endpoints
- **Key idea**: Assumes smooth signal between known points
- **Strengths**: Fast, no training required
- **Weakness**: Cannot capture complex dynamics

### Ensemble

Combines CSDI, MOMENT, SAITS, TimesNet:

- **Method**: Weighted averaging or median
- **Key idea**: Robust to individual method failures
- **Trade-off**: Higher computational cost

## Evaluation Metrics

Imputation quality is evaluated against ground truth using:

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| MAPE | Mean Absolute Percentage Error |
| MRE | Mean Relative Error |

See `configs/mlflow_registry/metrics/imputation.yaml` for full list.

## Hydra Usage

```bash
# Use specific imputation model
python src/imputation/flow_imputation.py \
    MODELS=SAITS

# Full preprocessing pipeline
python src/pipeline_PLR.py \
    OUTLIER_MODELS=MOMENT \
    MODELS=CSDI
```

## Training Notes

### GPU Requirements

| Model | GPU Memory | Training Time |
|-------|------------|---------------|
| SAITS | ~4GB | ~1h |
| CSDI | ~8GB | ~3h |
| TimesNet | ~4GB | ~2h |
| MOMENT | ~8GB | ~2h (finetune) |

### Checkpointing

All models save checkpoints for reproducibility:

```yaml
CHECKPOINT:
  save_top_k: 3
  monitor: 'val_mae'
  mode: 'min'
```

## See Also

- Outlier detection (previous stage): `../OUTLIER_MODELS/`
- Feature extraction (next stage): `../PLR_FEATURIZATION/`
- Registry: `../mlflow_registry/parameters/classification.yaml`
- Code: `src/imputation/`
- External:
  - [SAITS paper](https://arxiv.org/abs/2202.08516)
  - [CSDI paper](https://arxiv.org/abs/2107.03502)
  - [MOMENT paper](https://arxiv.org/abs/2402.03885)
  - [TimesNet paper](https://arxiv.org/abs/2210.02186)
  - [PyPOTS library](https://github.com/WenjieDu/PyPOTS)

---

**Note**: Performance comparisons are documented in the manuscript, not this repository.
