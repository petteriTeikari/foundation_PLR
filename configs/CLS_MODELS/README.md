# CLS_MODELS - Classifier Model Parameters

> **Quick Visual Guides** (5-second overview)

[![Classifier Configuration Architecture: Two-tier system showing fixed model parameters (this folder) and HPO search spaces (CLS_HYPERPARAMS). CatBoost iterations=1000, XGBoost use_GPU=false, TabPFN pre-trained transformer, TabM mini-architecture, LogReg l1 penalty.](../../docs/repo-figures/assets/fig-repo-51-classifier-config-architecture.jpg)](../../docs/repo-figures/assets/fig-repo-51-classifier-config-architecture.jpg)

*Figure: Config architecture - this folder contains FIXED parameters (iterations, seeds), while `CLS_HYPERPARAMS/` contains TUNABLE search spaces. [CatBoost docs →](https://catboost.ai/docs/concepts/python-reference_train.html)*

[![Classifier Paradigms: Evolution from Linear models (LogReg - interpretable, linear boundary) to Tree Ensembles (CatBoost, XGBoost - gradient boosting, handles non-linearity) to Foundation Models (TabPFN, TabM - pre-trained, prior-data fitted). Shows maturity tiers.](../../docs/repo-figures/assets/fig-repo-52-classifier-paradigms.jpg)](../../docs/repo-figures/assets/fig-repo-52-classifier-paradigms.jpg)

*Figure: Classifier paradigms from traditional to modern. TabPFN and TabM are foundation models requiring minimal tuning. [TabPFN paper →](https://arxiv.org/abs/2207.01848) | [Grinsztajn 2022 →](https://arxiv.org/abs/2207.08815)*

## Purpose

Defines the fixed (non-tunable) model parameters for each classifier. These complement the hyperparameter search spaces in `CLS_HYPERPARAMS/`.

## Files

| File | Classifier | Paradigm |
|------|-----------|----------|
| `CATBOOST.yaml` | CatBoost | Tree Ensemble |
| `XGBOOST.yaml` | XGBoost | Tree Ensemble |
| `LogisticRegression.yaml` | Logistic Regression | Linear |
| `TabPFN.yaml` | TabPFN | Foundation Model |
| `TabM.yaml` | TabM | Foundation Model |

## Key Fixed Parameters

### CatBoost

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `iterations` | 1000 | Max boosting iterations |
| `seed` | 100 | Random seed for reproducibility |
| `used_ram_limit` | "36gb" | Memory limit |
| `use_GPU` | False | CPU training (for reproducibility) |

**Weighing Options**:
- `weigh_the_samples`: True (default)
- `weigh_the_features`: False
- `weigh_the_classes`: False

### XGBoost

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `use_cupy_arrays` | False | No GPU acceleration |
| `calibration.method` | null | No post-hoc calibration |

**Feature Selection** (optional, disabled by default):
- RFE (Recursive Feature Elimination)
- `n_features_to_select`: 5

### Logistic Regression

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `penalty` | 'l1' | Lasso regularization (sparse) |
| `C` | 0.1 | Regularization strength |
| `solver` | 'liblinear' | Good for small datasets |

### TabPFN

| Parameter | Notes |
|-----------|-------|
| Minimal config | TabPFN requires few fixed params |
| Class weighting | `weigh_the_classes: True` |

TabPFN is a pre-trained transformer model. Most "tuning" happens in the pre-training, not at inference.

### TabM

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `arch_type` | 'tabm-mini' | Architecture variant |
| `n_epochs` | 100 | Training epochs |
| `patience` | 16 | Early stopping patience |
| `batch_size` | 256 | Mini-batch size |
| `skip_model_export` | True | Save disk space (large model) |

**Note**: TabM only runs on ground truth preprocessing due to computational cost.

## Weighing Options (All Classifiers)

Each classifier supports three weighing schemes:

| Option | Field | Description |
|--------|-------|-------------|
| Sample weights | `weigh_the_samples` | Weight individual training samples |
| Feature weights | `weigh_the_features` | Weight feature importance |
| Class weights | `weigh_the_classes` | Handle class imbalance |

Weight creation methods:
- `inverse_of_variance` - Weight by inverse feature variance
- `normalize_mean` - Normalize sample weights
- `unity` - Replace NaN weights with 1.0

## Confidence Intervals

CatBoost supports two CI methods:

```yaml
CI:
  method_CI: 'BOOTSTRAP'  # or 'ENSEMBLE'
  BOOTSTRAP:
    esize: 2  # Submodels per bootstrap iteration
```

- **BOOTSTRAP**: Uses bootstrap resampling (default, STRATOS-compliant)
- **ENSEMBLE**: Uses model ensemble variance

## Hydra Usage

```bash
# Load model config
python src/classification/flow_classification.py \
    CLS_MODELS=CATBOOST

# Combined with hyperparameter space
python src/classification/flow_classification.py \
    CLS_MODELS=CATBOOST \
    CLS_HYPERPARAMS=CATBOOST_hyperparam_space

# Override specific fixed param
python src/classification/flow_classification.py \
    CLS_MODELS=CATBOOST \
    CLS_MODELS.CATBOOST.iterations=2000
```

## Two-Config Pattern

| Directory | Purpose | Example |
|-----------|---------|---------|
| `CLS_MODELS/` | Fixed params (this) | iterations=1000 |
| `CLS_HYPERPARAMS/` | Tunable search space | depth=[1,3] |

Both are loaded together via Hydra composition to define a complete classifier configuration.

## See Also

- Hyperparameter spaces: `../CLS_HYPERPARAMS/`
- Code: `src/classification/`
- External docs:
  - [CatBoost Training](https://catboost.ai/docs/concepts/python-reference_train.html)
  - [XGBoost Learning Task](https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters)
  - [TabPFN (Hollmann 2023)](https://arxiv.org/abs/2207.01848)
  - [TabM (Yandex 2024)](https://arxiv.org/abs/2410.24210)

---

**Note**: Performance comparisons are documented in the manuscript, not this repository.
