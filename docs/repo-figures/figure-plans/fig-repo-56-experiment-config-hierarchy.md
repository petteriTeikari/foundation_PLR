# Figure Plan: fig-repo-56-experiment-config-hierarchy

**Target**: Repository documentation infographic
**Section**: `configs/` - Hydra Composition System
**Purpose**: Explain the composable, hierarchical configuration architecture
**Version**: 2.0 (Enhanced - full Hydra composition power)

---

## Title

**Hierarchical Experiment Configuration: Composable Hydra Architecture**

---

## Purpose

Help developers understand:
1. How Hydra's **config groups** enable reusable, composable configurations
2. The **hierarchical override system** (defaults → groups → experiment → CLI)
3. How the **same CatBoost config** can be reused across paper_2026, paper_2027, etc.
4. How to **selectively include/exclude** components (e.g., use only 2 of 5 classifiers)
5. The **DRY principle** - define once, compose everywhere

---

## Key Concept: Config Groups as Reusable Building Blocks

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │           HYDRA CONFIG GROUPS = REUSABLE LEGO BLOCKS         │
                    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  CONFIG GROUPS (defined ONCE, reused across ALL experiments)                     │
    │                                                                                  │
    │  configs/                                                                        │
    │  ├── CLS_MODELS/              ← Config Group: Classifier model settings          │
    │  │   ├── catboost.yaml           [Reusable across any experiment]               │
    │  │   ├── xgboost.yaml                                                           │
    │  │   ├── tabpfn.yaml                                                            │
    │  │   ├── tabm.yaml                                                              │
    │  │   └── logreg.yaml                                                            │
    │  │                                                                               │
    │  ├── CLS_HYPERPARAMS/         ← Config Group: HPO search spaces                  │
    │  │   ├── CATBOOST_hyperparam_space.yaml                                         │
    │  │   ├── XGBOOST_hyperparam_space.yaml                                          │
    │  │   └── ...                                                                     │
    │  │                                                                               │
    │  ├── MODELS/                  ← Config Group: Imputation models                  │
    │  │   ├── SAITS.yaml                                                             │
    │  │   ├── CSDI.yaml                                                              │
    │  │   ├── MOMENT.yaml                                                            │
    │  │   └── TimesNet.yaml                                                          │
    │  │                                                                               │
    │  ├── OUTLIER_MODELS/          ← Config Group: Outlier detection                  │
    │  │   ├── MOMENT.yaml                                                            │
    │  │   ├── LOF.yaml                                                               │
    │  │   └── ...                                                                     │
    │  │                                                                               │
    │  ├── data/                    ← Config Group: Dataset specifications             │
    │  │   ├── plr_glaucoma.yaml                                                      │
    │  │   └── plr_synthetic.yaml   [Future datasets]                                 │
    │  │                                                                               │
    │  └── combos/                  ← Config Group: Method combinations                │
    │      ├── paper_2026_combos.yaml                                                 │
    │      ├── quick_test_combos.yaml                                                 │
    │      └── ablation_combos.yaml                                                   │
    │                                                                                  │
    └─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Visual Layout: Multi-Experiment Reuse Pattern

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  FIGURE TITLE: Hierarchical Experiment Configuration - Composable Hydra Architecture │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  LAYER 1: EXPERIMENTS (Top-level orchestration)                                      │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                      │
│                                                                                      │
│   ┌────────────────┐    ┌────────────────┐    ┌────────────────┐                    │
│   │  paper_2026    │    │  paper_2027    │    │  ablation_     │                    │
│   │  .yaml         │    │  .yaml         │    │  study.yaml    │                    │
│   │                │    │                │    │                │                    │
│   │  defaults:     │    │  defaults:     │    │  defaults:     │                    │
│   │  - CLS_MODELS: │    │  - CLS_MODELS: │    │  - CLS_MODELS: │                    │
│   │    catboost    │    │    catboost    │    │    catboost    │                    │
│   │  - combos:     │    │  - combos:     │    │  - combos:     │                    │
│   │    paper_2026  │    │    paper_2027  │    │    ablation    │                    │
│   │  - data:       │    │  - data:       │    │  - data:       │                    │
│   │    plr_glaucoma│    │    plr_v2      │    │    plr_glaucoma│                    │
│   └───────┬────────┘    └───────┬────────┘    └───────┬────────┘                    │
│           │                     │                     │                              │
│           └─────────────────────┼─────────────────────┘                              │
│                                 │                                                    │
│                    ALL SHARE THE SAME CatBoost CONFIG!                               │
│                                 │                                                    │
│                                 ▼                                                    │
│  LAYER 2: CONFIG GROUPS (Reusable components)                                        │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                        │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                               │   │
│  │  CLS_MODELS/                   CLS_HYPERPARAMS/            MODELS/            │   │
│  │  ┌────────────┐               ┌────────────────┐          ┌────────────┐     │   │
│  │  │ catboost   │◄──────────────│ CATBOOST_      │          │ SAITS      │     │   │
│  │  │ .yaml      │  references   │ hyperparam_    │          │ .yaml      │     │   │
│  │  │            │               │ space.yaml     │          │            │     │   │
│  │  │ iterations │               │                │          │ d_model    │     │   │
│  │  │ task_type  │               │ depth: [1,3]   │          │ n_layers   │     │   │
│  │  │ loss_func  │               │ lr: [.001,.1]  │          │ dropout    │     │   │
│  │  └────────────┘               └────────────────┘          └────────────┘     │   │
│  │  ┌────────────┐               ┌────────────────┐          ┌────────────┐     │   │
│  │  │ xgboost    │               │ XGBOOST_...    │          │ CSDI       │     │   │
│  │  │ .yaml      │               │ .yaml          │          │ .yaml      │     │   │
│  │  └────────────┘               └────────────────┘          └────────────┘     │   │
│  │  ┌────────────┐               ┌────────────────┐          ┌────────────┐     │   │
│  │  │ tabpfn     │               │ ...            │          │ MOMENT     │     │   │
│  │  │ .yaml      │               │                │          │ .yaml      │     │   │
│  │  └────────────┘               └────────────────┘          └────────────┘     │   │
│  │                                                                               │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                 │                                                    │
│                                 ▼                                                    │
│  LAYER 3: FALLBACK DEFAULTS (Base parameters)                                        │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                        │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                         defaults.yaml                                         │   │
│  │                    (Always loaded as base layer)                              │   │
│  │                                                                               │   │
│  │  BOOTSTRAP:              VISUALIZATION:          CLS_EVALUATION:              │   │
│  │    n_iterations: 1000      dpi: 100               glaucoma_params:            │   │
│  │    alpha_CI: 0.95          figure_format: png       prevalence: 0.0354        │   │
│  │    random_state: 42                                                           │   │
│  │                                                                               │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Visual Layout: Selective Composition (Choose What You Need)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  SELECTIVE COMPOSITION: Use Only What Each Experiment Needs                          │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌─────────────────────────────┐         ┌─────────────────────────────┐            │
│  │  paper_2026.yaml            │         │  quick_test.yaml            │            │
│  │  (Full experiment)          │         │  (Fast iteration)           │            │
│  ├─────────────────────────────┤         ├─────────────────────────────┤            │
│  │                             │         │                             │            │
│  │  Classifiers: ALL 5         │         │  Classifiers: 1 only        │            │
│  │  ┌───┐┌───┐┌───┐┌───┐┌───┐ │         │  ┌───┐                      │            │
│  │  │Cat││XGB││Tab││TbM││Log│ │         │  │Cat│                      │            │
│  │  │   ││   ││PFN││   ││Reg│ │         │  │   │                      │            │
│  │  └───┘└───┘└───┘└───┘└───┘ │         │  └───┘                      │            │
│  │                             │         │                             │            │
│  │  Outlier methods: ALL 11    │         │  Outlier methods: 2 only    │            │
│  │  ████████████████████████   │         │  ████                       │            │
│  │                             │         │  pupil-gt, MOMENT-gt-ft     │            │
│  │                             │         │                             │            │
│  │  Imputation: ALL 8          │         │  Imputation: 2 only         │            │
│  │  ████████████████████████   │         │  ████                       │            │
│  │                             │         │  pupil-gt, SAITS            │            │
│  │                             │         │                             │            │
│  │  Bootstrap: 1000            │         │  Bootstrap: 10              │            │
│  │                             │         │                             │            │
│  └─────────────────────────────┘         └─────────────────────────────┘            │
│                                                                                      │
│  YAML Implementation:                                                                │
│                                                                                      │
│  # paper_2026_combos.yaml                # quick_test_combos.yaml                   │
│  outlier_methods:                        outlier_methods:                           │
│    - pupil-gt                              - pupil-gt                               │
│    - MOMENT-gt-finetune                    - MOMENT-gt-finetune                     │
│    - MOMENT-gt-zeroshot                                                             │
│    - UniTS-gt-finetune                   imputation_methods:                        │
│    - TimesNet-gt                           - pupil-gt                               │
│    - LOF                                   - SAITS                                  │
│    - OneClassSVM                                                                    │
│    - PROPHET                             classifiers:                               │
│    - SubPCA                                - CatBoost                               │
│    - ensemble-LOF-MOMENT-...                                                        │
│    - ensembleThresholded-...                                                        │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Visual Layout: Override Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  OVERRIDE HIERARCHY: Later Overrides Earlier (Like CSS Specificity)                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  PRIORITY (highest → lowest):                                                        │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │ 5. CLI OVERRIDES                                               HIGHEST ▲    │    │
│  │    python run.py experiment=paper_2026 BOOTSTRAP.n_iterations=100           │    │
│  │                                                                              │    │
│  │    • Overrides EVERYTHING below                                              │    │
│  │    • Use for quick testing, debugging, single runs                           │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                          │                                           │
│                                          ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │ 4. EXPERIMENT CONFIG                                                         │    │
│  │    configs/experiment/paper_2026.yaml                                        │    │
│  │                                                                              │    │
│  │    • Experiment-specific overrides                                           │    │
│  │    • MLflow settings, output paths                                           │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                          │                                           │
│                                          ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │ 3. COMBO CONFIG                                                              │    │
│  │    configs/combos/paper_2026_combos.yaml                                     │    │
│  │                                                                              │    │
│  │    • Which methods to run (outlier, imputation, classifier)                  │    │
│  │    • Subset selection per experiment                                         │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                          │                                           │
│                                          ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │ 2. CONFIG GROUP DEFAULTS                                                     │    │
│  │    configs/CLS_MODELS/catboost.yaml, configs/MODELS/SAITS.yaml, etc.         │    │
│  │                                                                              │    │
│  │    • Reusable component settings                                             │    │
│  │    • Shared across all experiments that use them                             │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                          │                                           │
│                                          ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │ 1. BASE DEFAULTS                                               LOWEST ▼     │    │
│  │    configs/defaults.yaml                                                     │    │
│  │                                                                              │    │
│  │    • Foundational parameters (bootstrap, visualization, constants)           │    │
│  │    • Always loaded first                                                     │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  EXAMPLE: Where does BOOTSTRAP.n_iterations come from?                               │
│                                                                                      │
│  defaults.yaml:              n_iterations: 1000  ← Defined here                      │
│  paper_2026.yaml:            (not set)           ← Inherited                         │
│  CLI override:               n_iterations=100    ← WINS (highest priority)           │
│                                                                                      │
│  Result: n_iterations = 100                                                          │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Visual Layout: Defaults List Pattern (Hydra Core Concept)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  THE DEFAULTS LIST: Hydra's Composition Mechanism                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  # configs/experiment/paper_2026.yaml                                                │
│                                                                                      │
│  defaults:                                                                           │
│    - _self_                         # This file's own keys (lowest priority here)    │
│    - /combos: paper_2026_combos     # Include from combos/ config group              │
│    - /data: plr_glaucoma            # Include from data/ config group                │
│    - /CLS_MODELS: catboost          # Include specific classifier model              │
│    - /CLS_HYPERPARAMS: CATBOOST_hyperparam_space                                     │
│                                                                                      │
│  # Experiment-specific overrides                                                     │
│  mlflow:                                                                             │
│    experiment_name: "paper_2026_main"                                                │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                              │    │
│  │  WHAT HAPPENS WHEN HYDRA LOADS THIS:                                         │    │
│  │                                                                              │    │
│  │  1. Load configs/defaults.yaml (always first)                                │    │
│  │                          │                                                   │    │
│  │                          ▼                                                   │    │
│  │  2. Load configs/combos/paper_2026_combos.yaml                               │    │
│  │     → Adds: outlier_methods, imputation_methods, classifiers lists           │    │
│  │                          │                                                   │    │
│  │                          ▼                                                   │    │
│  │  3. Load configs/data/plr_glaucoma.yaml                                      │    │
│  │     → Adds: database path, subject splits, table names                       │    │
│  │                          │                                                   │    │
│  │                          ▼                                                   │    │
│  │  4. Load configs/CLS_MODELS/catboost.yaml                                    │    │
│  │     → Adds: CatBoost model configuration                                     │    │
│  │                          │                                                   │    │
│  │                          ▼                                                   │    │
│  │  5. Load configs/CLS_HYPERPARAMS/CATBOOST_hyperparam_space.yaml              │    │
│  │     → Adds: Optuna search space for CatBoost                                 │    │
│  │                          │                                                   │    │
│  │                          ▼                                                   │    │
│  │  6. Apply this file's own keys (mlflow.experiment_name)                      │    │
│  │                          │                                                   │    │
│  │                          ▼                                                   │    │
│  │  FINAL: Merged configuration object ready for use                            │    │
│  │                                                                              │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Visual Layout: Multi-Experiment Reuse Example

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  REAL-WORLD SCENARIO: Same Classifier, Different Experiments                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Scenario: You optimize CatBoost once, reuse it everywhere                           │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                              │    │
│  │            configs/CLS_MODELS/catboost.yaml                                  │    │
│  │                                                                              │    │
│  │            ┌──────────────────────────────┐                                  │    │
│  │            │  CATBOOST:                   │                                  │    │
│  │            │    iterations: 1000          │                                  │    │
│  │            │    task_type: GPU            │                                  │    │
│  │            │    loss_function: Logloss    │                                  │    │
│  │            │    early_stopping_rounds: 50 │                                  │    │
│  │            └──────────────────────────────┘                                  │    │
│  │                           │                                                  │    │
│  │          ┌────────────────┼────────────────┬────────────────┐               │    │
│  │          │                │                │                │               │    │
│  │          ▼                ▼                ▼                ▼               │    │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │    │
│  │  │ paper_2026   │ │ paper_2027   │ │ ablation_    │ │ retina_      │       │    │
│  │  │ .yaml        │ │ .yaml        │ │ fm_only.yaml │ │ only.yaml    │       │    │
│  │  │              │ │              │ │              │ │              │       │    │
│  │  │ Uses CatBoost│ │ Uses CatBoost│ │ Uses CatBoost│ │ Uses CatBoost│       │    │
│  │  │ + XGBoost    │ │ + TabPFNv2   │ │ only         │ │ + LogReg     │       │    │
│  │  │ + TabPFN     │ │ + TabM       │ │              │ │              │       │    │
│  │  │ + TabM       │ │              │ │              │ │              │       │    │
│  │  │ + LogReg     │ │              │ │              │ │              │       │    │
│  │  │              │ │              │ │              │ │              │       │    │
│  │  │ 11 outlier   │ │ 11 outlier   │ │ 4 FM outlier │ │ 11 outlier   │       │    │
│  │  │ methods      │ │ methods      │ │ methods only │ │ methods      │       │    │
│  │  │              │ │              │ │              │ │              │       │    │
│  │  │ All 8        │ │ All 8        │ │ FM imputers  │ │ All 8        │       │    │
│  │  │ imputers     │ │ imputers     │ │ only         │ │ imputers     │       │    │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘       │    │
│  │                                                                              │    │
│  │  KEY INSIGHT: catboost.yaml is defined ONCE                                  │    │
│  │               → Reused by 4 different experiments                            │    │
│  │               → Change it once, affects all experiments                      │    │
│  │                                                                              │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## CLI Usage Examples

```bash
# Full paper experiment (all methods, 1000 bootstrap)
python run.py experiment=paper_2026

# Quick test (subset, 10 bootstrap)
python run.py experiment=paper_2026 combos=quick_test_combos BOOTSTRAP.n_iterations=10

# Override specific classifier
python run.py experiment=paper_2026 CLS_MODELS=xgboost

# Use different dataset
python run.py experiment=paper_2026 data=plr_synthetic

# FM-only ablation (via different combo file)
python run.py experiment=ablation_fm_only

# Single method test with CLI override
python run.py experiment=paper_2026 \
    combos.outlier_methods=[pupil-gt] \
    combos.imputation_methods=[SAITS] \
    combos.classifiers=[CatBoost]

# Multirun: sweep across classifiers
python run.py -m experiment=paper_2026 CLS_MODELS=catboost,xgboost,tabpfn
```

---

## Content Elements

### Config Group Inventory

| Config Group | Contents | Purpose |
|--------------|----------|---------|
| `CLS_MODELS/` | 5 classifier configs | Model hyperparameters (fixed) |
| `CLS_HYPERPARAMS/` | 5 HPO spaces | Search spaces for Optuna |
| `MODELS/` | 7 imputation models | Deep learning imputation settings |
| `OUTLIER_MODELS/` | 10 outlier configs | Outlier detection settings |
| `combos/` | Method combinations | Which methods to run per experiment |
| `data/` | Dataset specifications | Database paths, subject splits |
| `experiment/` | Top-level orchestration | Defaults list + overrides |

### DRY Principle Benefits

| Without Hydra | With Hydra |
|---------------|------------|
| Copy CatBoost config to each experiment | Define once in `CLS_MODELS/catboost.yaml` |
| 5 experiments × 5 classifiers = 25 files | 5 classifier files + 5 experiment files |
| Change requires editing multiple files | Change once, affects all |
| Drift between experiments | Guaranteed consistency |

---

## Key Messages

1. **Config Groups = Reusable Lego Blocks**: Define once, compose everywhere
2. **Defaults List**: Hydra's mechanism for composing configurations hierarchically
3. **Override Hierarchy**: CLI > experiment > combos > groups > defaults
4. **Selective Composition**: Each experiment chooses its subset of methods
5. **DRY Principle**: CatBoost is defined ONCE, reused across all experiments

---

## Technical Specifications

- **Aspect ratio**: 16:10 (taller for detailed hierarchy)
- **Resolution**: 300 DPI
- **Background**: #FBF9F3 (Economist off-white)
- **Typography**: Sans-serif, monospace for config paths
- **Colour coding**:
  - Deep blue (`primary_pathway`) for experiment configs
  - Gold (`highlight_accent`) for combo configs
  - Teal (`cone_S`) for data configs
  - Grey (`secondary_pathway`) for base defaults

---

## References

- **Hydra Documentation**: https://hydra.cc/docs/intro/
- **Hydra Jupyter Notebook Example**: https://github.com/facebookresearch/hydra/blob/main/examples/jupyter_notebooks/compose_configs_in_notebook.ipynb
- **DVC + Hydra Composition**: https://dvc.org/doc/user-guide/experiment-management/hydra-composition
- **OmegaConf (underlying config library)**: https://omegaconf.readthedocs.io/

---

## Related Documentation

- **README**: `configs/experiment/README.md`
- **Root configs README**: `configs/README.md` (existing Mermaid diagrams)
- **Related infographic**: fig-repo-05 (Hydra Config System - basic intro)

---

*Figure plan created: 2026-02-02*
*Version 2.0: Enhanced to show full Hydra composition power, multi-experiment reuse, selective composition*
*For: configs/ documentation - composable hierarchical architecture*
