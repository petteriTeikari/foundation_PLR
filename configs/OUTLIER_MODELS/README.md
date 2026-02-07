# OUTLIER_MODELS - Outlier Detection Method Configurations

> **Quick Visual Guide** (5-second overview)

[![Outlier Detection Methods: The 11 registry-defined methods organized by category - Ground Truth (pupil-gt), Foundation Models (MOMENT, UniTS), Deep Learning (TimesNet), Traditional (LOF, OneClassSVM, PROPHET, SubPCA), and Ensembles. Shows config file locations and method characteristics.](../../docs/repo-figures/assets/fig-repo-53-outlier-detection-methods.jpg)](../../docs/repo-figures/assets/fig-repo-53-outlier-detection-methods.jpg)

*Figure: The 11 outlier detection methods in the registry, categorized by approach. CRITICAL: If code shows more than 11 methods, the extraction is broken. See [Registry Pattern (fig-repo-55)](../../docs/repo-figures/assets/fig-repo-55-registry-single-source-of-truth.jpg) for why this matters. [LOF paper →](https://dl.acm.org/doi/10.1145/342009.335388) | [MOMENT repo →](https://github.com/moment-timeseries-foundation-model/moment)*

## Purpose

Configures outlier detection methods that identify artifacts (blinks, noise, dropouts) in raw PLR signals. This is the first preprocessing stage before imputation.

## The 11 Registry Methods

**CRITICAL**: The registry defines EXACTLY 11 outlier methods. See `configs/mlflow_registry/parameters/classification.yaml`.

| Category | Method | Description |
|----------|--------|-------------|
| **Ground Truth** | `pupil-gt` | Human-annotated baseline |
| **Foundation Model** | `MOMENT-gt-finetune` | MOMENT fine-tuned on PLR |
| | `MOMENT-gt-zeroshot` | MOMENT zero-shot transfer |
| | `UniTS-gt-finetune` | UniTS fine-tuned on PLR |
| **Deep Learning** | `TimesNet-gt` | Multi-period temporal modeling |
| **Traditional** | `LOF` | Local Outlier Factor (density) |
| | `OneClassSVM` | One-class SVM (boundary) |
| | `PROPHET` | Forecast-based detection |
| | `SubPCA` | Subspace PCA anomaly |
| **Ensemble** | `ensemble-LOF-MOMENT-...` | 7-method majority voting |
| | `ensembleThresholded-...` | 3-method (DL+FM only) |

## Files

| File | Method | Approach |
|------|--------|----------|
| `LOF.yaml` | Local Outlier Factor | Density-based |
| `LOF_only.yaml` | LOF standalone | Same, minimal config |
| `OneClassSVM.yaml` | One-Class SVM | Boundary-based |
| `PROPHET.yaml` | Prophet | Forecast residuals |
| `SubPCA.yaml` | Subspace PCA | Dimensionality-based |
| `MOMENT.yaml` | MOMENT Foundation Model | Transformer-based |
| `UniTS.yaml` | UniTS Foundation Model | Unified TS transformer |
| `TimesNet.yaml` | TimesNet | Multi-period convolution |
| `EIF.yaml` | Extended Isolation Forest | Tree-based (experimental) |
| `SigLLM.yaml` | Signal LLM | LLM-based (experimental) |

## Method Categories

### Traditional Methods

**Local Outlier Factor (LOF)**
- Density-based anomaly detection
- Compares local density to neighbors
- Good for isolated outliers
- Reference: Breunig et al. 2000

**One-Class SVM**
- Boundary-based detection
- Learns decision boundary around normal data
- Works well with small datasets
- Reference: Schölkopf et al. 2001

**PROPHET**
- Time series forecasting approach
- Detects points that deviate from forecast
- Handles seasonality and trends
- Reference: Taylor & Letham 2018

**SubPCA (Subspace PCA)**
- Projects data to principal subspace
- Reconstruction error indicates anomalies
- Effective for correlated features

### Deep Learning Methods

**TimesNet**
- Multi-period temporal 2D variation modeling
- Transforms 1D signal to 2D using FFT-based period detection
- Captures both intra-period and inter-period patterns
- Reference: Wu et al. 2023 ICLR

### Foundation Model Methods

**MOMENT**
- Open time-series foundation model
- Pre-trained on large-scale time series
- Supports zero-shot and fine-tuning modes
- Reference: Goswami et al. 2024

**UniTS**
- Unified time series model
- Multi-task pre-training (forecasting, classification, anomaly)
- Domain-agnostic representations
- Reference: Gao et al. 2024

### Ensemble Methods

**Full Ensemble** (`ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune`)
- Combines 7 methods via majority voting
- Robust to individual method failures
- Higher computational cost

**Thresholded Ensemble** (`ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune`)
- 3-method ensemble (DL + FM only)
- Lower computational cost
- Focus on modern methods

## Key Parameters

### LOF

```yaml
LOF:
  n_neighbors: 20       # K in k-NN
  contamination: 0.1    # Expected outlier fraction
  algorithm: 'auto'     # Ball tree, KD tree, or brute force
```

### One-Class SVM

```yaml
OneClassSVM:
  kernel: 'rbf'
  nu: 0.1              # Upper bound on outlier fraction
  gamma: 'scale'       # Kernel coefficient
```

### MOMENT

```yaml
MOMENT:
  model_name: 'MOMENT-1-large'
  task_name: 'anomaly_detection'
  mode: 'finetune'     # or 'zeroshot'
  max_epochs: 100
```

### TimesNet

```yaml
TimesNet:
  seq_len: 96
  pred_len: 0          # Anomaly detection, not forecasting
  d_model: 32
  n_heads: 4
  e_layers: 2
```

## UniTS Training

UniTS requires separate training outside the main pipeline:

1. Export data: `write_as_numpy_for_vanilla_dataloader()`
2. Clone: `https://github.com/petteriTeikari/UniTS`
3. Place data in `UniTS/dataset/SERI_PLR_GLAUCOMA/`
4. Run training with `anomaly_PLR_args`
5. Results logged to MLflow

See the `assets/` directory for screenshot documentation.

## Hydra Usage

```bash
# Use specific outlier method
python src/outlier_detection/flow_outlier.py \
    OUTLIER_MODELS=LOF

# Multiple methods
python src/pipeline_PLR.py \
    OUTLIER_MODELS=MOMENT \
    MODELS=SAITS  # Imputation after outlier detection
```

## Evaluation Metrics

Outlier detection is evaluated against ground truth (`pupil-gt`) using:

| Metric | Description |
|--------|-------------|
| F1 Score | Harmonic mean of precision/recall |
| Precision | True positives / predicted positives |
| Recall | True positives / actual positives |
| IoU | Intersection over union of masks |

See `configs/mlflow_registry/metrics/outlier_detection.yaml` for full list.

## See Also

- Registry: `../mlflow_registry/parameters/classification.yaml`
- Imputation (next stage): `../MODELS/`
- Code: `src/outlier_detection/`
- External:
  - [LOF paper](https://dl.acm.org/doi/10.1145/342009.335388)
  - [MOMENT repo](https://github.com/moment-timeseries-foundation-model/moment)
  - [UniTS repo](https://github.com/mims-harvard/UniTS)
  - [TimesNet paper](https://arxiv.org/abs/2210.02186)

---

**Note**: Performance comparisons are documented in the manuscript, not this repository.
