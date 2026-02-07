# fig-repo-06: How to Add a New Classifier

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-06 |
| **Title** | Adding a New Classifier to the Pipeline |
| **Complexity Level** | L4 (Technical how-to with config architecture) |
| **Target Persona** | Research Scientist, ML Engineer |
| **Location** | CONTRIBUTING.md, docs/tutorials/ |
| **Priority** | P2 (High - prevents misinformation) |
| **Aspect Ratio** | 16:14 (wider to accommodate full config hierarchy) |

## Purpose

Step-by-step guide for contributors who want to add new classifiers (like TabPFNv2.5, MIRA) to the existing pipeline, showing the CORRECT config hierarchy.

## CRITICAL: Config Architecture Clarification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING vs POST-HOC ANALYSIS                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   TRAINING CONFIG (Hydra composition)       POST-HOC REGISTRY               │
│   ──────────────────────────────────        ────────────────────            │
│                                                                             │
│   configs/                                  configs/mlflow_registry/        │
│   ├── CLS_MODELS/                           └── parameters/                 │
│   │   ├── CATBOOST.yaml  ◄── NEW            │   └── classification.yaml    │
│   │   ├── XGBOOST.yaml       CLASSIFIER                                     │
│   │   ├── TabPFN.yaml        GOES HERE!     Documents WHAT WAS TESTED       │
│   │   ├── TabM.yaml                         (for analysis & visualization)  │
│   │   └── LogisticRegression.yaml                                           │
│   │                                         Updated AFTER experiments run,  │
│   ├── CLS_HYPERPARAMS/                      NOT before training!            │
│   │   ├── CATBOOST_hyperparam_space.yaml                                    │
│   │   ├── XGBOOST_hyperparam_space.yaml                                     │
│   │   └── ...                                                               │
│   │                                                                         │
│   └── defaults.yaml  ◄── Hydra composes all                                 │
│                                                                             │
│   python -m src.classification.flow_classification CLS_MODELS=TabPFNv25     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Message

"Adding a new classifier requires: (1) Python wrapper, (2) CLS_MODELS config, (3) CLS_HYPERPARAMS config, (4) tests. The mlflow_registry is updated AFTER running experiments for post-hoc analysis."

## Visual Concept

**16:14 aspect ratio - Two-column layout:**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              ADDING A NEW CLASSIFIER: Complete Config Architecture               │
│                                                                                  │
│  LEFT COLUMN: TRAINING PIPELINE              RIGHT COLUMN: POST-HOC ANALYSIS     │
│  ═══════════════════════════════════         ════════════════════════════════    │
│                                                                                  │
│  ┌──────────────────────────────────┐        ┌──────────────────────────────┐   │
│  │ STEP 1: Create Python Wrapper    │        │ LATER: Register for Analysis │   │
│  │ src/classifiers/tabpfn_v25.py    │        │                              │   │
│  │ ┌────────────────────────────┐   │        │ ONLY AFTER experiments run!  │   │
│  │ │class TabPFNv25Classifier:  │   │        │                              │   │
│  │ │  def fit(X, y): ...        │   │        │ configs/mlflow_registry/     │   │
│  │ │  def predict_proba(X): ... │   │        │   parameters/                │   │
│  │ └────────────────────────────┘   │        │     classification.yaml      │   │
│  └──────────────────────────────────┘        │                              │   │
│              │                               │ model_name:                  │   │
│              ▼                               │   values:                    │   │
│  ┌──────────────────────────────────┐        │     - CatBoost               │   │
│  │ STEP 2: CLS_MODELS Config        │        │     - XGBoost                │   │
│  │ configs/CLS_MODELS/TabPFNv25.yaml│        │     - TabPFN                 │   │
│  │ ┌────────────────────────────┐   │        │     - TabM                   │   │
│  │ │HYPERPARAMS:                │   │        │     - LogisticRegression     │   │
│  │ │  enabled: True             │   │        │     + TabPFNv25  # ADDED     │   │
│  │ │  search_space_file:        │   │        │   count: 6  # UPDATED        │   │
│  │ │    TabPFNv25_hyperparam... │   │        │                              │   │
│  │ └────────────────────────────┘   │        │ Purpose: Query MLflow runs   │   │
│  └──────────────────────────────────┘        │ for analysis & visualization │   │
│              │                               └──────────────────────────────┘   │
│              ▼                                                                   │
│  ┌──────────────────────────────────┐        ┌──────────────────────────────┐   │
│  │ STEP 3: CLS_HYPERPARAMS Config   │        │ CONFIG PRIORITY (Hydra)      │   │
│  │ configs/CLS_HYPERPARAMS/         │        │                              │   │
│  │   TabPFNv25_hyperparam_space.yaml│        │  defaults.yaml               │   │
│  │ ┌────────────────────────────┐   │        │       │                      │   │
│  │ │search_space:               │   │        │       ▼ composes             │   │
│  │ │  n_ensemble_configurations:│   │        │  CLS_MODELS/*.yaml           │   │
│  │ │    range: [8, 32]          │   │        │       │                      │   │
│  │ │  learning_rate:            │   │        │       ▼ references           │   │
│  │ │    range: [0.001, 0.1]     │   │        │  CLS_HYPERPARAMS/*.yaml      │   │
│  │ │    log_scale: True         │   │        │       │                      │   │
│  │ └────────────────────────────┘   │        │       ▼ CLI override         │   │
│  └──────────────────────────────────┘        │  python -m ... CLS_MODELS=X  │   │
│              │                               └──────────────────────────────┘   │
│              ▼                                                                   │
│  ┌──────────────────────────────────┐                                           │
│  │ STEP 4: Add Tests                │                                           │
│  │ tests/test_classifiers/          │                                           │
│  │   test_tabpfn_v25.py             │                                           │
│  │ ┌────────────────────────────┐   │                                           │
│  │ │def test_tabpfn_fit():      │   │                                           │
│  │ │  clf = TabPFNv25Classifier()│  │                                           │
│  │ │  clf.fit(X_train, y_train) │   │                                           │
│  │ │  assert clf.predict_proba(.│   │                                           │
│  │ └────────────────────────────┘   │                                           │
│  └──────────────────────────────────┘                                           │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ ✓ RUN: python -m src.classification.flow_classification CLS_MODELS=TabPFNv25 │
│  │ ✓ NOW WORKS WITH: all 11 outlier × 8 imputation = 88 preprocessing combos!   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

### Required Elements
1. **Two-column layout**: Training (left) vs Post-hoc (right)
2. **Clear separation**: "Training Config" vs "Analysis Registry"
3. **Four training steps**: wrapper, CLS_MODELS, CLS_HYPERPARAMS, tests
4. **Config priority diagram**: Hydra composition flow
5. **"LATER" callout**: mlflow_registry is POST-HOC
6. **File paths**: Exact paths for each component

### Critical Clarifications
1. `CLS_MODELS/` = Classifier model configuration (training)
2. `CLS_HYPERPARAMS/` = Hyperparameter search space (training)
3. `mlflow_registry/` = Documents what was tested (analysis/viz ONLY)
4. Hydra composes configs at runtime via defaults

## Text Content

### Title Text
"Adding a Classifier: Training Config vs Analysis Registry"

### Labels/Annotations

**Left Column (Training Pipeline):**
- Header: "TRAINING PIPELINE - Where to add new classifier"
- Step 1: "src/classifiers/ - Python wrapper with scikit-learn API"
- Step 2: "configs/CLS_MODELS/ - Model configuration"
- Step 3: "configs/CLS_HYPERPARAMS/ - Search space definition"
- Step 4: "tests/test_classifiers/ - Unit tests"

**Right Column (Post-Hoc):**
- Header: "POST-HOC ANALYSIS - Updated AFTER training"
- Callout: "ONLY after experiments complete!"
- Purpose: "Defines valid methods for analysis & visualization"
- Note: "Not for training configuration"

**Config Priority Box:**
- "defaults.yaml → composes → CLS_MODELS → references → CLS_HYPERPARAMS → CLI override"

### Caption (for embedding)
Adding a new classifier requires a Python wrapper in src/classifiers/, model config in configs/CLS_MODELS/, and hyperparameter space in configs/CLS_HYPERPARAMS/. The mlflow_registry is only updated AFTER running experiments, for post-hoc analysis and visualization.

## Prompts for Nano Banana Pro

### Style Prompt
Developer documentation style with two-column comparison layout. Left column shows training workflow (green/blue accents). Right column shows post-hoc analysis (orange/yellow accents, with "LATER" warning). Use dark-mode code snippets. File paths in monospace. Include Hydra config composition arrows. 16:14 aspect ratio.

### Content Prompt
Create a two-column infographic (16:14 aspect) for classifier configuration:

**LEFT COLUMN - "Training Pipeline" (4 steps):**
1. Python wrapper (src/classifiers/new_classifier.py) with class snippet
2. CLS_MODELS config (configs/CLS_MODELS/NewClassifier.yaml) with YAML snippet
3. CLS_HYPERPARAMS config (configs/CLS_HYPERPARAMS/NewClassifier_hyperparam_space.yaml) with search space snippet
4. Tests (tests/test_classifiers/test_new_classifier.py)

**RIGHT COLUMN - "Post-Hoc Analysis":**
- Warning callout: "ONLY AFTER experiments run!"
- mlflow_registry/parameters/classification.yaml
- "Purpose: Query MLflow for analysis & visualization"
- Config priority diagram showing Hydra composition: defaults.yaml → CLS_MODELS → CLS_HYPERPARAMS → CLI override

**BOTTOM:**
- Success banner with CLI command and "works with 88 preprocessing combinations"

### Refinement Notes
- CRITICAL: The two columns must be visually distinct
- Left = "where to ADD" (green checkmarks, action steps)
- Right = "when to UPDATE" (orange warning, post-hoc)
- Hydra composition flow should be clear arrows
- Don't make mlflow_registry look like a required training step

## Alt Text

Two-column infographic showing classifier configuration architecture: Left column shows four training steps (Python wrapper, CLS_MODELS config, CLS_HYPERPARAMS config, tests); Right column shows post-hoc mlflow_registry that is only updated after experiments complete. Includes Hydra config priority diagram showing composition flow.

## Technical Notes

### File Mapping

| Purpose | Directory | When Modified |
|---------|-----------|---------------|
| Python wrapper | `src/classifiers/` | During development |
| Model config | `configs/CLS_MODELS/` | During development |
| Hyperparam space | `configs/CLS_HYPERPARAMS/` | During development |
| Tests | `tests/test_classifiers/` | During development |
| Analysis registry | `configs/mlflow_registry/` | **AFTER experiments** |

### Hydra Composition Flow

```yaml
# configs/defaults.yaml (excerpt)
defaults:
  - _self_
  - CLS_MODELS: XGBOOST           # Can override: CLS_MODELS=TabPFNv25
  - CLS_HYPERPARAMS: XGBOOST_hyperparam_space
```

### Why This Matters

**WRONG understanding**: "I need to add my classifier to mlflow_registry to train it"

**CORRECT understanding**: "mlflow_registry documents what experiments HAVE BEEN run, for analysis and visualization purposes. Training uses CLS_MODELS and CLS_HYPERPARAMS."

## Status

- [x] Draft created
- [x] Factual accuracy reviewed (config architecture verified)
- [ ] Review passed
- [ ] Generated (16:14 aspect ratio)
- [ ] Placed in CONTRIBUTING.md, docs/tutorials/
