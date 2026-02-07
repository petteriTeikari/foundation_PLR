# Figure Plan: fig-repo-51-classifier-config-architecture

**Target**: Repository documentation infographic
**Section**: `configs/CLS_HYPERPARAMS/` and `configs/CLS_MODELS/`
**Purpose**: Visual guide to classifier hyperparameter configuration
**Version**: 1.0

---

## Title

**Classifier Configuration Architecture**

---

## Purpose

Help developers understand:
1. How classifier hyperparameters are organized in the config system
2. What each hyperparameter controls and why
3. The search space rationale for each classifier
4. How Hydra composes these configs

---

## Visual Layout (3-column)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FIGURE TITLE: Classifier Configuration Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────────┐  ┌─────────────────────────┐ │
│  │                 │  │                     │  │                         │ │
│  │  COLUMN 1:      │  │  COLUMN 2:          │  │  COLUMN 3:              │ │
│  │  Config Files   │  │  Hyperparameter     │  │  Search Space           │ │
│  │                 │  │  Explanation        │  │  Visualization          │ │
│  │  [Tree showing  │  │                     │  │                         │ │
│  │  CLS_HYPERPARAMS│  │  [Table with each   │  │  [Mini plots showing    │ │
│  │  └── CatBoost   │  │  param, what it     │  │  the range for each     │ │
│  │  └── XGBoost    │  │  controls, and why  │  │  continuous param]      │ │
│  │  └── TabPFN     │  │  that range]        │  │                         │ │
│  │  └── TabM       │  │                     │  │  depth: [1,3] ████░░░   │ │
│  │  └── LogReg     │  │                     │  │  lr: [0.001,0.1] █████  │ │
│  │                 │  │                     │  │  l2: [1,30] ██████████  │ │
│  │  CLS_MODELS     │  │                     │  │                         │ │
│  │  └── *.yaml     │  │                     │  │                         │ │
│  │                 │  │                     │  │                         │ │
│  └─────────────────┘  └─────────────────────┘  └─────────────────────────┘ │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  BOTTOM: Hydra Composition Example                                       ││
│  │  python run.py CLS_HYPERPARAMS=CATBOOST CLS_MODELS=catboost             ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Content Elements

### Column 1: Config File Tree

```
configs/
├── CLS_HYPERPARAMS/
│   ├── CATBOOST_hyperparam_space.yaml
│   ├── XGBOOST_hyperparam_space.yaml
│   ├── TABPFN_hyperparam_space.yaml
│   ├── TABM_hyperparam_space.yaml
│   └── LOGREG_hyperparam_space.yaml
└── CLS_MODELS/
    ├── catboost.yaml
    ├── xgboost.yaml
    ├── tabpfn.yaml
    ├── tabm.yaml
    └── logreg.yaml
```

### Column 2: Hyperparameter Table

| Classifier | Parameter | What It Controls | Range | Rationale |
|------------|-----------|------------------|-------|-----------|
| **CatBoost** | `depth` | Tree depth | [1, 3] | Shallow trees prevent overfitting on small N |
| | `lr` | Learning rate | [0.001, 0.1] | Standard range for gradient boosting |
| | `l2_leaf_reg` | L2 regularization | [1, 30] | Prevents leaf overfitting |
| **XGBoost** | `max_depth` | Tree depth | [1, 3] | Same rationale as CatBoost |
| | `eta` | Learning rate | [0.001, 0.1] | XGBoost naming convention |
| | `lambda` | L2 regularization | [1, 30] | Regularization strength |
| **TabPFN** | N/A | Zero-shot | - | Pretrained, no tuning needed |
| **TabM** | `num_emb` | Embedding dim | [32, 128] | Feature representation capacity |
| **LogReg** | `C` | Inverse regularization | [0.01, 100] | Classic regularization parameter |

### Column 3: Search Space Visualization

Mini horizontal bar charts showing:
- Where the range sits relative to theoretical min/max
- Log vs linear scale indication
- Default value marker (if applicable)

---

## Key Messages

1. **Shallow trees for small datasets**: depth [1,3] not [1,10] because N=208
2. **Consistent ranges across boosters**: CatBoost and XGBoost use same effective ranges
3. **TabPFN is zero-shot**: No hyperparameter tuning needed
4. **Hydra composition**: Mix and match configs via CLI

---

## Technical Specifications

- **Aspect ratio**: 16:9 (landscape)
- **Resolution**: 300 DPI
- **Background**: #FBF9F3 (Economist off-white)
- **Typography**: Sans-serif, dark grey (#333333)
- **Generation method**: Mermaid + Python for search space bars

---

## Data Source

- `configs/CLS_HYPERPARAMS/*.yaml`
- `configs/CLS_MODELS/*.yaml`

---

## Related Documentation

- **README to create**: `configs/CLS_HYPERPARAMS/README.md`
- **Existing**: `configs/README.md` (Mermaid Hydra diagram)
- **Related infographic**: fig-repo-52 (Classifier Paradigms - conceptual)

---

## References

- Prokhorenkova et al. 2018 "CatBoost: unbiased boosting with categorical features"
- Chen & Guestrin 2016 "XGBoost: A Scalable Tree Boosting System"
- Hollmann et al. 2023 "TabPFN: A Transformer That Solves Small Tabular Classification Problems"
- Grinsztajn et al. 2022 "Why do tree-based models still outperform deep learning on tabular data?"

---

*Figure plan created: 2026-02-02*
*For: configs/CLS_HYPERPARAMS/ and configs/CLS_MODELS/ documentation*
