# fig-repo-29: MLflow Experiment Tracking

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-29 |
| **Title** | MLflow Experiment Tracking |
| **Complexity Level** | L2 (Technical concept) |
| **Target Persona** | ML Engineer, Data Scientist |
| **Location** | docs/user-guide/, ARCHITECTURE.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain how MLflow stores experiments, the run naming convention, and how to find results.

## Key Message

"542 MLflow runs capture all (outlier × imputation × classifier) combinations. Each run stores 1000 bootstrap iterations as a pickle file for reproducible analysis."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MLFLOW EXPERIMENT TRACKING                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  WHAT IS MLFLOW?                                                                │
│  ═══════════════                                                                │
│                                                                                 │
│  An open-source platform for managing the ML lifecycle:                         │
│  • Experiment tracking (what we use)                                            │
│  • Model packaging                                                              │
│  • Model registry                                                               │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  EXPERIMENT STRUCTURE                                                           │
│  ════════════════════                                                           │
│                                                                                 │
│  /home/petteri/mlruns/                                                          │
│  └── 253031330985650090/                    ← Experiment ID                     │
│      ├── 0a1b2c3d.../                       ← Run 1                             │
│      │   ├── meta.yaml                      ← Run metadata                      │
│      │   ├── params/                        ← Hyperparameters                   │
│      │   │   ├── outlier_method             ← "MOMENT-gt-finetune"              │
│      │   │   ├── imputation_method          ← "SAITS"                           │
│      │   │   └── classifier                 ← "CatBoost"                        │
│      │   ├── metrics/                       ← Logged metrics                    │
│      │   │   └── test_auroc                 ← 0.9099                            │
│      │   └── artifacts/                     ← Stored files                      │
│      │       └── bootstrap_results.pkl      ← 1000 iterations!                  │
│      │                                                                          │
│      ├── 1b2c3d4e.../                       ← Run 2                             │
│      ├── 2c3d4e5f.../                       ← Run 3                             │
│      └── ...                                ← 542 total runs                    │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  RUN NAMING CONVENTION                                                          │
│  ═════════════════════                                                          │
│                                                                                 │
│  run_name: "simple__MOMENT-gt-finetune__SAITS__CatBoost"                        │
│             ├────┘  ├───────────────────┘ ├───┘ ├───────┘                       │
│             │       │                     │     │                               │
│             │       │                     │     └─ Classifier                   │
│             │       │                     └─ Imputation method                  │
│             │       └─ Outlier detection method                                 │
│             └─ Featurization type ("simple" = handcrafted)                      │
│                                                                                 │
│  This naming convention allows parsing, BUT we use the REGISTRY instead!        │
│  (See: configs/mlflow_registry/)                                                │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  BOOTSTRAP PICKLE CONTENTS                                                      │
│  ═════════════════════════                                                      │
│                                                                                 │
│  bootstrap_results.pkl contains:                                                │
│                                                                                 │
│  {                                                                              │
│    "auroc": [0.907, 0.921, 0.895, ...],    # 1000 AUROC values                  │
│    "brier": [0.131, 0.128, 0.134, ...],    # 1000 Brier scores                  │
│    "y_true": [[0,1,0,1,...], ...],         # True labels per iteration          │
│    "y_prob": [[0.2,0.8,...], ...],         # Predictions per iteration          │
│    "train_indices": [[1,5,8,...], ...],    # Bootstrap sample indices           │
│    "test_indices": [[2,3,4,...], ...],     # OOB indices                        │
│  }                                                                              │
│                                                                                 │
│  This enables computing ANY metric post-hoc (STRATOS compliance!)               │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ACCESSING RESULTS                                                              │
│  ════════════════                                                               │
│                                                                                 │
│  # Option 1: MLflow UI                                                          │
│  mlflow ui --port 5000                                                          │
│  → http://localhost:5000                                                        │
│                                                                                 │
│  # Option 2: Python API                                                         │
│  import mlflow                                                                  │
│  runs = mlflow.search_runs(experiment_ids=["253031330985650090"])               │
│                                                                                 │
│  # Option 3: Use our extraction (RECOMMENDED)                                   │
│  # Results already in DuckDB: data/public/foundation_plr_results.db             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **What is MLflow**: Brief introduction
2. **Directory structure**: Folder tree showing experiments, runs, artifacts
3. **Naming convention**: Breakdown of run name components
4. **Pickle contents**: What's stored in bootstrap_results.pkl
5. **Access methods**: UI, Python API, DuckDB extraction

## Text Content

### Title Text
"MLflow Experiment Tracking: 542 Runs, 542,000 Bootstrap Iterations"

### Caption
MLflow tracks all 542 experiment runs, each representing a unique (outlier × imputation × classifier) combination. Each run stores 1000 bootstrap iterations in a pickle file, enabling post-hoc computation of any STRATOS metric. Run names encode the configuration (featurization__outlier__imputation__classifier), but the registry in configs/mlflow_registry/ is the authoritative source for valid method names.

## Prompts for Nano Banana Pro

### Style Prompt
Directory tree diagram showing MLflow structure. Folder icons with nested hierarchy. Code block showing pickle contents. Command examples at bottom. Clean, technical documentation aesthetic.

### Content Prompt
Create an MLflow tracking diagram:

**TOP - Structure**:
- Directory tree: mlruns/ → experiment/ → run/ → params, metrics, artifacts
- Highlight bootstrap_results.pkl

**MIDDLE - Naming**:
- Example run name with arrows pointing to each component
- Labels: featurization, outlier, imputation, classifier

**BOTTOM LEFT - Pickle Contents**:
- JSON-like representation showing arrays for auroc, brier, y_true, y_prob

**BOTTOM RIGHT - Access**:
- Three options: MLflow UI, Python API, DuckDB

## Alt Text

MLflow experiment tracking diagram. Directory structure shows mlruns containing experiment 253031330985650090 with 542 runs. Each run contains params (outlier_method, imputation_method, classifier), metrics (test_auroc), and artifacts (bootstrap_results.pkl with 1000 iterations). Run naming convention breaks down as featurization__outlier__imputation__classifier. Three access methods: MLflow UI, Python API, or extracted DuckDB.

## Related Figures

- **fig-repo-04**: MLflow as lab notebook (ELI5 version for README)
- **fig-repro-20**: DuckDB single source (where MLflow data goes after extraction)

## Cross-References

This figure subsumes the reproducibility angle previously in **fig-repro-21** (archived).

Reader flow: **fig-repo-04** (ELI5) → **THIS FIGURE** (technical details) → **fig-repro-20** (extraction to DuckDB)

## Status

- [x] Draft created
- [x] Subsumes fig-repro-21 (archived)
- [ ] Generated
- [ ] Placed in ARCHITECTURE.md
