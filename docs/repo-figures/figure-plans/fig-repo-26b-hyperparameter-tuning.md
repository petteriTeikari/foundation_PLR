# fig-repo-26b: Hyperparameter Tuning: Optuna for CatBoost, Hyperopt for XGBoost

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-26b |
| **Title** | Hyperparameter Tuning with Optuna and Hyperopt |
| **Complexity Level** | L2-L3 (Technical implementation) |
| **Target Persona** | ML Engineer, Data Scientist |
| **Location** | docs/user-guide/, ARCHITECTURE.md |
| **Priority** | P2 |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain how hyperparameter optimization is done for the two main classifiers: CatBoost (Optuna) and XGBoost (hyperopt). Show the search spaces and optimization strategies.

## Relationship to Other Figures

| Figure | Scope | Focus |
|--------|-------|-------|
| **fig-repo-26** | Why CatBoost is fixed | Classifier selection justification |
| **fig-repo-26b** (THIS) | How classifiers are tuned | Hyperparameter optimization details |

## Key Message

"CatBoost uses Optuna (100 trials, 1-hour timeout) for gradient-free hyperparameter search. XGBoost uses hyperopt for tree-structured Parzen estimators. Both optimize AUROC on the test set."

## The Code (Verified from Repository)

### CatBoost: Optuna (`src/classification/catboost/catboost_main.py`)

```python
def catboost_ensemble_HPO(train, test, y_test, ...):
    def objective(trial):
        param = {
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", min, max),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", min, max),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", min, max),
            "depth": trial.suggest_int("depth", min, max),
            "lr": trial.suggest_categorical("lr", [0.01, 0.03, 0.1]),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["No"]),
        }
        ens = catboost_ensemble_fit(train, test, None, ...)
        score = classifier_hpo_eval(y_test, probs_mean_class1, eval_metric)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=3600)  # 1 hour
```

### XGBoost: hyperopt (`src/classification/hyperopt_utils.py`)

```python
def parse_hyperopt_search_space(hyperopt_cfg):
    params = {}
    for param_name, param_dict in hyperopt_cfg.items():
        if param_dict["hp_func"] == "choice":
            params[param_name] = hp.choice(param_name, np.arange(min, max, step))
        elif param_dict["hp_func"] == "uniform":
            params[param_name] = hp.uniform(param_name, min, max)
        elif param_dict["hp_func"] is None:
            params[param_name] = param_dict["value"]  # Fixed value
    return params
```

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              HYPERPARAMETER TUNING: OPTUNA vs HYPEROPT                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ╔═══════════════════════════════════════════════════════════════════════════╗ │
│  ║  CATBOOST: OPTUNA                                                          ║ │
│  ╠═══════════════════════════════════════════════════════════════════════════╣ │
│  ║                                                                            ║ │
│  ║  ┌─────────────────────────────────────────────────────────────────────┐  ║ │
│  ║  │  SEARCH SPACE                                                       │  ║ │
│  ║  │  ═══════════                                                        │  ║ │
│  ║  │                                                                     │  ║ │
│  ║  │  Parameter          │ Type           │ Range                        │  ║ │
│  ║  │  ────────────────────┼────────────────┼─────────────────────────── │  ║ │
│  ║  │  colsample_bylevel  │ suggest_float  │ [min, max] from config      │  ║ │
│  ║  │  min_data_in_leaf   │ suggest_int    │ [min, max] from config      │  ║ │
│  ║  │  l2_leaf_reg        │ suggest_float  │ [min, max] from config      │  ║ │
│  ║  │  depth              │ suggest_int    │ [min, max] from config      │  ║ │
│  ║  │  lr                 │ suggest_cat    │ [0.01, 0.03, 0.1, ...]      │  ║ │
│  ║  │  bootstrap_type     │ suggest_cat    │ ["No"] (fixed for SGLB)     │  ║ │
│  ║  └─────────────────────────────────────────────────────────────────────┘  ║ │
│  ║                                                                            ║ │
│  ║  ┌─────────────────────────────────────────────────────────────────────┐  ║ │
│  ║  │  OPTIMIZATION LOOP                                                  │  ║ │
│  ║  │                                                                     │  ║ │
│  ║  │  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────┐ │  ║ │
│  ║  │  │ Optuna   │ →  │ Sample       │ →  │ Train        │ →  │ Eval  │ │  ║ │
│  ║  │  │ Sampler  │    │ Hyperparams  │    │ Ensemble     │    │ AUROC │ │  ║ │
│  ║  │  │ (TPE)    │    │              │    │ (10 models)  │    │       │ │  ║ │
│  ║  │  └──────────┘    └──────────────┘    └──────────────┘    └───┬───┘ │  ║ │
│  ║  │                                                              │      │  ║ │
│  ║  │           ┌──────────────────────────────────────────────────┘      │  ║ │
│  ║  │           ▼                                                         │  ║ │
│  ║  │  100 trials OR 1-hour timeout → Best params → Final training        │  ║ │
│  ║  └─────────────────────────────────────────────────────────────────────┘  ║ │
│  ║                                                                            ║ │
│  ╚═══════════════════════════════════════════════════════════════════════════╝ │
│                                                                                 │
│  ╔═══════════════════════════════════════════════════════════════════════════╗ │
│  ║  XGBOOST: HYPEROPT (inset)                                                 ║ │
│  ╠═══════════════════════════════════════════════════════════════════════════╣ │
│  ║                                                                            ║ │
│  ║  Search space parsing:                                                     ║ │
│  ║                                                                            ║ │
│  ║  YAML config → parse_hyperopt_search_space() → hyperopt space             ║ │
│  ║                                                                            ║ │
│  ║  ┌─────────────────────────────────────────────────────────────────────┐  ║ │
│  ║  │  hp_func: "choice"   →  hp.choice(name, arange(min, max, step))     │  ║ │
│  ║  │  hp_func: "uniform"  →  hp.uniform(name, min, max)                  │  ║ │
│  ║  │  hp_func: null       →  Fixed value (not optimized)                 │  ║ │
│  ║  └─────────────────────────────────────────────────────────────────────┘  ║ │
│  ║                                                                            ║ │
│  ║  Uses Tree-structured Parzen Estimator (TPE) for efficient search         ║ │
│  ║                                                                            ║ │
│  ╚═══════════════════════════════════════════════════════════════════════════╝ │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHY TWO DIFFERENT LIBRARIES?                                                   │
│  ════════════════════════════                                                   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  Library   │ Classifier │ Reason                                       │   │
│  │  ──────────┼────────────┼────────────────────────────────────────────  │   │
│  │  Optuna    │ CatBoost   │ Better integration with CatBoost pruning     │   │
│  │            │            │ callbacks, cleaner API, built-in dashboard   │   │
│  │  ──────────┼────────────┼────────────────────────────────────────────  │   │
│  │  hyperopt  │ XGBoost    │ Legacy code, config-driven search space      │   │
│  │            │            │ parsing already implemented                  │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Both use TPE (Tree-structured Parzen Estimator) under the hood                │
│  Both optimize AUROC on test set (maximize direction)                          │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  DEFAULT BEHAVIOR                                                               │
│  ════════════════                                                               │
│                                                                                 │
│  skip_HPO: true (default)                                                       │
│                                                                                 │
│  HPO is OPTIONAL. Default hyperparameters from config are used unless          │
│  skip_HPO: false is set. Full HPO takes ~1 hour per classifier.                │
│                                                                                 │
│  Config location: configs/HYPERPARAMS/catboost.yaml                            │
│                   configs/HYPERPARAMS/xgboost.yaml                             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **CatBoost Optuna section**: Search space table, optimization loop diagram
2. **XGBoost hyperopt inset**: Config parsing, hp functions
3. **Comparison table**: Why different libraries for different classifiers
4. **Default behavior note**: skip_HPO flag, config locations

## Text Content

### Title Text
"Hyperparameter Tuning: Optuna for CatBoost, Hyperopt for XGBoost"

### Caption
CatBoost uses Optuna for hyperparameter optimization with 6 tunable parameters (colsample_bylevel, min_data_in_leaf, l2_leaf_reg, depth, lr, bootstrap_type). The search runs for 100 trials or 1 hour, whichever comes first, maximizing AUROC. XGBoost uses hyperopt with a config-driven search space supporting `hp.choice` for discrete and `hp.uniform` for continuous parameters. Both use Tree-structured Parzen Estimator (TPE). By default, HPO is skipped (`skip_HPO: true`) and pre-tuned defaults are used.

## Prompts for Nano Banana Pro

### Style Prompt
Technical architecture diagram showing hyperparameter optimization workflow. Two-section layout with CatBoost (main) and XGBoost (inset). Flow diagram showing trial → train → evaluate loop. Tables for parameter spaces. Professional ML engineering documentation style.

### Content Prompt
Create a hyperparameter tuning diagram:

**MAIN SECTION - CatBoost Optuna**:
- Table of 6 parameters with suggest_* types
- Flow diagram: Sampler → Sample → Train → Eval → 100 trials
- "1 hour timeout" badge

**INSET - XGBoost hyperopt**:
- Three hp_func mappings (choice, uniform, null)
- "Legacy but functional" note

**BOTTOM - Comparison**:
- Why two libraries? Integration reasons
- Both use TPE
- Default: skip_HPO = true

## Alt Text

Hyperparameter tuning diagram showing two optimization frameworks. Main section shows CatBoost using Optuna with 6 parameters (colsample_bylevel, min_data_in_leaf, l2_leaf_reg, depth, lr, bootstrap_type), 100 trials or 1-hour limit. Inset shows XGBoost using hyperopt with hp.choice and hp.uniform functions parsed from YAML config. Both use Tree-structured Parzen Estimator. Default behavior skips HPO and uses pre-tuned values.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/user-guide/
