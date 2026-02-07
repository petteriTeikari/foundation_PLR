# fig-repo-26: Classifier Configuration Architecture

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-26 |
| **Title** | Classifier Configuration Architecture |
| **Complexity Level** | L2-L3 (Architecture) |
| **Target Persona** | ML Engineer, Data Scientist |
| **Location** | docs/user-guide/, ARCHITECTURE.md |
| **Priority** | P2 |
| **Aspect Ratio** | 16:10 |

## Purpose

Document how classifiers are configured in this repository: config locations, hyperparameter spaces, Hydra composition, and how to add new classifiers.

## Key Message

"Five classifiers are implemented with two-file config pattern: model settings + hyperparameter search space. Each uses Hydra composition for flexible configuration."

## Config Architecture (Verified from Repository)

### Two-File Pattern Per Classifier

```
configs/
├── CLS_MODELS/                    # Model configuration
│   ├── CATBOOST.yaml              # Ensemble size, iterations, weighing
│   ├── XGBOOST.yaml               # Feature selection, calibration
│   ├── TabPFN.yaml                # Minimal config (pre-trained)
│   ├── TabM.yaml                  # Architecture settings
│   └── LogisticRegression.yaml    # Penalty, solver, regularization
│
└── CLS_HYPERPARAMS/               # Hyperparameter optimization
    ├── CATBOOST_hyperparam_space.yaml    # Optuna search space
    ├── XGBOOST_hyperparam_space.yaml     # hyperopt search space
    ├── TabPFN_hyperparam_space.yaml      # (minimal)
    ├── TabM_hyperparam_space.yaml
    └── LogisticRegression_hyperparam_space.yaml
```

### CatBoost Config Example

```yaml
# configs/CLS_MODELS/CATBOOST.yaml
CATBOOST:
  esize: 100          # Ensemble size (SGLB uncertainty)
  iterations: 1000    # Boosting iterations
  seed: 100
  used_ram_limit: "36gb"

  MODEL:
    use_GPU: False
    CI:
      method_CI: 'BOOTSTRAP'  # or 'ENSEMBLE'
    WEIGHING:
      weigh_the_samples: True
      weigh_the_classes: False
```

```yaml
# configs/CLS_HYPERPARAMS/CATBOOST_hyperparam_space.yaml
CATBOOST:
  HYPERPARAMS:
    method: 'OPTUNA'      # HPO library
    metric_val: 'auc'
    skip_HPO: False
  SEARCH_SPACE:
    OPTUNA:
      depth: [1, 3]
      lr: [0.001, 0.01, 0.1]
      colsample_bylevel: [0.05, 1.0]
      min_data_in_leaf: [1, 100]
      l2_leaf_reg: [1, 30]
```

### XGBoost Config Example

```yaml
# configs/CLS_MODELS/XGBOOST.yaml
XGBOOST:
  FEATURE_SELECTION:
    RFE:
      use: False
      n_features_to_select: 5
  CALIBRATION:
    method: null  # or 'isotonic'
  MODEL:
    WEIGHING:
      weigh_the_samples: True
```

```yaml
# configs/CLS_HYPERPARAMS/XGBOOST_hyperparam_space.yaml
XGBOOST:
  HYPERPARAMS:
    method: 'HYPEROPT'    # HPO library (different from CatBoost!)
    metric_val: 'auc'
  SEARCH_SPACE:
    HYPEROPT:
      max_depth:
        hp_func: 'choice'
        min: 1
        max: 2
      eta:
        hp_func: 'uniform'
        min: 0
        max: 1
      # ... more hyperparameters
```

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              CLASSIFIER CONFIGURATION ARCHITECTURE                               │
│              5 Classifiers × 2 Config Files Each                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │  HYDRA COMPOSITION                                                          ││
│  │                                                                             ││
│  │  python pipeline.py +CLS_MODEL=CATBOOST +CLS_HYPERPARAMS=CATBOOST          ││
│  │                                                                             ││
│  │         ┌─────────────────────┐     ┌─────────────────────────┐            ││
│  │         │ CLS_MODELS/         │     │ CLS_HYPERPARAMS/        │            ││
│  │         │ CATBOOST.yaml       │  +  │ CATBOOST_hyperparam_    │            ││
│  │         │                     │     │ space.yaml              │            ││
│  │         │ • esize             │     │ • SEARCH_SPACE.OPTUNA   │            ││
│  │         │ • iterations        │     │ • depth, lr, l2_leaf_reg│            ││
│  │         │ • WEIGHING          │     │ • method: 'OPTUNA'      │            ││
│  │         └─────────────────────┘     └─────────────────────────┘            ││
│  │                     │                         │                             ││
│  │                     └────────────┬────────────┘                             ││
│  │                                  ▼                                          ││
│  │                    ┌──────────────────────────┐                             ││
│  │                    │  Merged cfg object       │                             ││
│  │                    │  (Hydra OmegaConf)       │                             ││
│  │                    └──────────────────────────┘                             ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│  THE 5 CLASSIFIERS                                                               │
│  ════════════════                                                                │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                            │ │
│  │  Classifier          │ Paradigm              │ HPO Library │ Key Params   │ │
│  │  ────────────────────┼───────────────────────┼─────────────┼───────────── │ │
│  │  CatBoost            │ Gradient Boosting     │ Optuna      │ depth, lr,   │ │
│  │                      │ + SGLB uncertainty    │             │ l2_leaf_reg  │ │
│  │  ────────────────────┼───────────────────────┼─────────────┼───────────── │ │
│  │  XGBoost             │ Gradient Boosting     │ hyperopt    │ max_depth,   │ │
│  │                      │                       │             │ eta, gamma   │ │
│  │  ────────────────────┼───────────────────────┼─────────────┼───────────── │ │
│  │  TabPFN              │ Transformer           │ minimal     │ pre-trained  │ │
│  │                      │ Prior-fitted network  │             │ (frozen)     │ │
│  │  ────────────────────┼───────────────────────┼─────────────┼───────────── │ │
│  │  TabM                │ Deep tabular          │ (config)    │ architecture │ │
│  │                      │                       │             │              │ │
│  │  ────────────────────┼───────────────────────┼─────────────┼───────────── │ │
│  │  LogisticRegression  │ Linear model          │ minimal     │ penalty, C,  │ │
│  │                      │ (scikit-learn)        │             │ solver       │ │
│  │                                                                            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│  CLASSIFIER PARADIGMS (Literature Context)                                       │
│  ═════════════════════════════════════════                                       │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                            │ │
│  │  LINEAR MODELS                          TREE-BASED ENSEMBLES               │ │
│  │  ─────────────                          ────────────────────               │ │
│  │                                                                            │ │
│  │  LogisticRegression                     CatBoost, XGBoost                  │ │
│  │                                                                            │ │
│  │  • Linear decision boundary             • Gradient boosting                │ │
│  │  • Interpretable coefficients           • Handles non-linearity           │ │
│  │  • Fast training                        • Feature importance              │ │
│  │  • Regularization (L1/L2)               • GPU acceleration                │ │
│  │                                                                            │ │
│  │  See: Christodoulou 2019 JCE            See: Grinsztajn 2022 NeurIPS      │ │
│  │  "ML vs LR systematic review"           "Tree-based on tabular"           │ │
│  │                                                                            │ │
│  │  ─────────────────────────────────────────────────────────────────────────│ │
│  │                                                                            │ │
│  │  FOUNDATION TABULAR MODELS                                                 │ │
│  │  ─────────────────────────                                                 │ │
│  │                                                                            │ │
│  │  TabPFN, TabPFNv2                                                          │ │
│  │                                                                            │ │
│  │  • Pre-trained on synthetic tabular data                                   │ │
│  │  • Transformer architecture                                                │ │
│  │  • No hyperparameter tuning needed                                         │ │
│  │  • Works well on small datasets                                            │ │
│  │                                                                            │ │
│  │  See: Hollmann 2023 ICLR (TabPFN)                                          │ │
│  │       TabPFNv2 2024 (Prior-data fitted networks)                           │ │
│  │                                                                            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│  ADDING A NEW CLASSIFIER                                                         │
│  ═══════════════════════                                                         │
│                                                                                  │
│  1. Create model config:      configs/CLS_MODELS/NEW_CLASSIFIER.yaml            │
│  2. Create HPO config:        configs/CLS_HYPERPARAMS/NEW_CLASSIFIER_*.yaml     │
│  3. Add to registry:          configs/mlflow_registry/parameters/classification │
│  4. Implement wrapper:        src/classification/new_classifier/                │
│  5. Register in flow:         src/classification/flow_classification.py         │
│                                                                                  │
│  Common additions from literature:                                               │
│  • LightGBM (gradient boosting, Microsoft)                                       │
│  • AutoGluon (AutoML ensemble)                                                   │
│  • NODE (Neural Oblivious Decision Ensembles)                                    │
│  • TabNet (Attentive Interpretable Tabular)                                      │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Hydra composition diagram**: How two configs merge
2. **Config file examples**: Actual YAML from repo
3. **5 classifiers table**: Paradigm, HPO library, key params
4. **Paradigm comparison**: Linear vs Tree vs Foundation (conceptual only)
5. **Extension guide**: How to add new classifiers

## Text Content

### Title Text
"Classifier Configuration Architecture"

### Caption
Five classifiers are implemented with a two-file config pattern: model settings (`configs/CLS_MODELS/`) and hyperparameter search space (`configs/CLS_HYPERPARAMS/`). CatBoost uses Optuna for HPO; XGBoost uses hyperopt. TabPFN is pre-trained and requires minimal tuning. Configs compose via Hydra overrides. To add a new classifier, create both config files, implement a wrapper in `src/classification/`, and register in the flow.

## Paradigm Comparison (Conceptual, Not Performance)

| Paradigm | Representatives | Key Characteristics |
|----------|-----------------|---------------------|
| **Linear** | LogisticRegression | Interpretable, linear boundary, fast |
| **Tree Ensemble** | CatBoost, XGBoost | Non-linear, feature importance, GPU-accelerated |
| **Foundation Tabular** | TabPFN | Pre-trained, minimal tuning, transformer-based |

## Literature References

| Topic | Reference |
|-------|-----------|
| ML vs Logistic Regression | Christodoulou et al. 2019 JCE - Systematic review |
| Tree-based on tabular | Grinsztajn et al. 2022 NeurIPS - "Why do tree-based models still outperform" |
| TabPFN | Hollmann et al. 2023 ICLR - Prior-data fitted networks |
| TabPFNv2 | 2024 - Extended pre-training, larger datasets |
| CatBoost SGLB | Malinin et al. 2021 - Uncertainty via SGLB |

## Prompts for Nano Banana Pro

### Style Prompt
Architecture diagram showing config file structure. Hydra composition flow. Table of classifiers with paradigms. Literature context boxes. No performance metrics or rankings. Technical documentation style.

### Content Prompt
Create a classifier configuration diagram:

**TOP - Hydra Composition**:
- Two config boxes merging into one
- CLS_MODELS/*.yaml + CLS_HYPERPARAMS/*.yaml

**MIDDLE - Classifier Table**:
- 5 rows: CatBoost, XGBoost, TabPFN, TabM, LogisticRegression
- Columns: Paradigm, HPO Library, Key Params
- NO performance columns

**BOTTOM LEFT - Paradigm Boxes**:
- Linear models (LogisticRegression)
- Tree ensembles (CatBoost, XGBoost)
- Foundation tabular (TabPFN)
- Brief characteristics, NO rankings

**BOTTOM RIGHT - Extension Guide**:
- Steps to add new classifier
- Common additions from literature

## Alt Text

Classifier configuration architecture diagram. Shows Hydra composition merging CLS_MODELS/*.yaml and CLS_HYPERPARAMS/*.yaml configs. Table lists 5 classifiers: CatBoost (gradient boosting, Optuna), XGBoost (gradient boosting, hyperopt), TabPFN (transformer, pre-trained), TabM (deep tabular), LogisticRegression (linear, minimal). Paradigm boxes explain conceptual differences without performance comparison. Extension guide shows 5 steps to add new classifiers.

## Status

- [x] Draft created
- [x] Updated to focus on config/architecture, not results
- [ ] Generated
- [ ] Placed in docs/user-guide/

## Note

Performance comparisons between classifiers are in the manuscript, not this repository.
See: `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/latent-methods-results/`
