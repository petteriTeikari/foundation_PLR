# Foundation PLR - Complete Context for External AI Consultation
# ==============================================================
#
# This document provides COMPLETE context for AI systems (Gemini, OpenAI, etc.)
# that don't have access to the codebase. Copy this entire document when
# seeking second opinions on architecture, implementation, or analysis.
#
# Last Updated: 2026-01-22
# Repository: foundation_PLR (private)

---

## 1. PROJECT OVERVIEW

### 1.1 Research Question

> **How do preprocessing choices (outlier detection → imputation) affect
> downstream classification performance when using handcrafted physiological
> features for glaucoma screening from pupillary light reflex (PLR) signals?**

Specifically:
1. Do foundation models (MOMENT, UniTS, TimesNet, SAITS) provide better
   preprocessing than traditional methods (LOF, SVM, linear interpolation)?
2. How do errors introduced at outlier detection propagate through imputation
   and ultimately degrade classifier performance?
3. Can automated preprocessing match human-annotated ground truth quality?

### 1.2 What This Is NOT About

- **NOT** comparing classifiers (CatBoost vs XGBoost vs TabPFN)
- **NOT** generic ML benchmarking
- **NOT** finding the "best classifier"

The classifier is FIXED (CatBoost - it's the best, that's established).
The VARIABLE is preprocessing (outlier detection × imputation method).

### 1.3 Key Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Best AUROC | 0.913 | With ground truth preprocessing + CatBoost |
| Preprocessing effect size | η²=0.15 | Meaningful but not dominant |
| Handcrafted vs Embeddings | 0.830 vs 0.740 | **9pp gap - embeddings are NOT useful!** |
| FM for preprocessing | Competitive | FMs useful for outlier detection & imputation |

---

## 2. THE PIPELINE (Error Propagation Chain)

```
Raw PLR Signal (with blinks, artifacts, noise)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ [1] OUTLIER DETECTION (11 methods)                              │
│     Question: Can FMs detect artifacts as well as humans?       │
│                                                                 │
│     Ground Truth:   pupil-gt (human annotation)                 │
│     Foundation:     MOMENT-gt-finetune, UniTS-gt-finetune, etc. │
│     Traditional:    LOF, OneClassSVM, PROPHET                   │
│     Ensembles:      ensemble-LOF-MOMENT-..., ensembleThresholded│
│                                                                 │
│     ERRORS HERE → propagate downstream!                         │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ [2] IMPUTATION (8 methods)                                      │
│     Question: Can FMs reconstruct missing segments well?        │
│                                                                 │
│     Ground Truth:   pupil-gt (human denoised signal)            │
│     Deep Learning:  SAITS, CSDI, TimesNet                       │
│     Foundation:     MOMENT-finetune, MOMENT-zeroshot            │
│     Ensembles:      ensemble-CSDI-MOMENT-SAITS-TimesNet         │
│                                                                 │
│     ERRORS HERE → affect feature extraction!                    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ [3] FEATURIZATION (5 methods)                                   │
│     FIXED: Using handcrafted features (amplitude bins + latency)│
│                                                                 │
│     Options tested:                                             │
│     - handcrafted_features (amplitude histogram + PIPR/MEDFA)   │
│     - embeddings (FM latent representations) ← 9pp WORSE!       │
│     - pupil_gt, anomaly_gt, ensembled_input                     │
│                                                                 │
│     NOT using FM embeddings for features - they underperform!   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ [4] CLASSIFICATION (5 classifiers, but FIXED to CatBoost)       │
│                                                                 │
│     Tested: CatBoost, XGBoost, TabPFN, TabM, LogisticRegression │
│     Winner: CatBoost (0.878 mean, 0.913 max)                    │
│                                                                 │
│     DO NOT compare classifiers - use CatBoost and measure       │
│     how PREPROCESSING affects it                                │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
AUROC (measure of glaucoma vs control discrimination)
```

---

## 3. DATA INFRASTRUCTURE

### 3.1 Data Sources

| Source | Path | Contents |
|--------|------|----------|
| Raw PLR + Ground Truth | `SERI_PLR_GLAUCOMA.db` (SQLite) | 507 subjects, 1M+ timepoints |
| MLflow Experiments | `/home/petteri/mlruns/` | 410+ experiment runs |
| Derived Results | `foundation_plr_results.db` (DuckDB) | Simplified metrics view |
| CD Diagram Data | `cd_diagram_data.duckdb` | Pre-computed Friedman stats |

### 3.2 Subject Counts (CRITICAL!)

| Task | N Subjects | Reason |
|------|------------|--------|
| **Outlier Detection** | **507** | All subjects have ground truth masks |
| **Imputation** | **507** | All subjects have ground truth signals |
| **Classification** | **208** | Only 152 Control + 56 Glaucoma have labels |

The 299 unlabeled subjects CAN be used for self-supervised pretraining but
NOT for classification evaluation.

### 3.3 Database Schema (SERI_PLR_GLAUCOMA.db)

```sql
-- Main tables
CREATE TABLE train (
    subject_code TEXT PRIMARY KEY,
    pupil_raw BLOB,          -- Raw PLR signal (float32 array)
    pupil_gt BLOB,           -- Human-denoised ground truth signal
    outlier_mask BLOB,       -- Binary mask (1 = outlier/artifact)
    imputation_mask BLOB,    -- Binary mask (1 = needs imputation)
    class_label INTEGER,     -- 0=Control, 1=Glaucoma, NULL=Unknown
    -- ... additional columns
);

CREATE TABLE test (
    -- Same schema as train
);
```

### 3.4 MLflow Experiment Structure

```
/home/petteri/mlruns/
├── 253031330985650090/     # PLR_Classification experiment (410 runs)
│   ├── <run_id>/
│   │   ├── params/         # Hyperparameters
│   │   │   ├── outlier_method
│   │   │   ├── imputation_method
│   │   │   ├── featurization
│   │   │   └── classifier
│   │   ├── metrics/        # Performance metrics
│   │   │   ├── auroc_mean
│   │   │   ├── auroc_std
│   │   │   ├── bootstrap_iterations (1000)
│   │   │   └── ...
│   │   └── artifacts/      # Saved models, predictions
│   │       └── bootstrap_metrics.pkl
```

### 3.5 DuckDB Results Schema (foundation_plr_results.db)

```sql
-- Simplified view for visualization
CREATE TABLE essential_metrics (
    config_id INTEGER PRIMARY KEY,
    outlier_method TEXT,
    imputation_method TEXT,
    featurization TEXT,
    classifier TEXT,
    auroc_mean FLOAT,
    auroc_std FLOAT,
    auroc_ci_lower FLOAT,
    auroc_ci_upper FLOAT,
    brier_mean FLOAT,
    n_bootstrap INTEGER
);
```

---

## 4. CONFIGURATION SYSTEM

### 4.1 Config Files Overview

| Config | Purpose | Location |
|--------|---------|----------|
| `plot_hyperparam_combos.yaml` | Fixed combos for figures | `config/` |
| `demo_subjects.yaml` | 12 subjects for traces | `config/` |
| `figure_registry.yaml` | Figure metadata catalog | `config/` |
| `methods.yaml` | Method names & display | `config/` |
| `colors.yaml` | Paul Tol colorblind palette | `config/` |

### 4.2 Hyperparam Combos (plot_hyperparam_combos.yaml)

```yaml
# Standard 4 combos (main figures)
standard_combos:
  - id: "ground_truth"
    outlier_method: "pupil-gt"
    imputation_method: "pupil-gt"
    classifier: "CatBoost"
    auroc: 0.9110

  - id: "best_ensemble"
    outlier_method: "ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune"
    imputation_method: "CSDI"
    classifier: "CatBoost"
    auroc: 0.9130

  - id: "best_single_fm"
    outlier_method: "MOMENT-gt-finetune"
    imputation_method: "SAITS"
    classifier: "CatBoost"
    auroc: 0.9099

  - id: "traditional"
    outlier_method: "LOF"
    imputation_method: "SAITS"
    classifier: "TabPFN"
    auroc: 0.8599

# Extended 5 combos (supplementary)
extended_combos:
  - id: "moment_full"     # MOMENT for both tasks
  - id: "lof_moment"      # Traditional outlier + FM imputation
  - id: "timesnet_full"   # TimesNet for both tasks
  - id: "units_pipeline"  # UniTS outlier + SAITS
  - id: "simple_baseline" # OC-SVM + MOMENT-zeroshot
```

### 4.3 Hydra Experiment Config (for running experiments)

```yaml
# config/experiment/classification.yaml
defaults:
  - _self_
  - override /preprocessing: moment_saits
  - override /classifier: catboost

preprocessing:
  outlier:
    method: "MOMENT-gt-finetune"
    threshold: 0.5
  imputation:
    method: "SAITS"
    n_epochs: 100

classifier:
  name: "CatBoost"
  params:
    iterations: 1000
    depth: 6
    learning_rate: 0.03

evaluation:
  n_bootstrap: 1000
  metrics: ["auroc", "aupr", "brier_score"]
```

---

## 5. VISUALIZATION INFRASTRUCTURE

### 5.1 Figure Generation Scripts

| Script | Output | Combos |
|--------|--------|--------|
| `factorial_matrix.py` | `fig_M3_factorial_matrix` | None (static) |
| `featurization_comparison.py` | `fig_R7_featurization_comparison` | None (aggregated) |
| `foundation_model_dashboard.py` | `fig_R8_foundation_model_dashboard` | None (by task) |
| `retained_metric.py` | `fig_retained_multi_metric` | 4 standard |
| `calibration_plot.py` | `fig_calibration_smoothed` | 4 standard |
| `dca_plot.py` | `fig_dca_curves` | 4 standard |
| `cd_diagram_preprocessing.py` | `cd_preprocessing_comparison` | Top 12 |

### 5.2 Styling System (plot_config.py)

```python
# Paul Tol colorblind-safe palette
COLORS = {
    'ground_truth': '#666666',     # Gray
    'best_ensemble': '#0072B2',    # Blue
    'best_single_fm': '#56B4E9',   # Light blue
    'traditional': '#E69F00',      # Orange
}

# Economist/ggplot2 style
STYLE = {
    'background': '#FAFAFA',       # Light gray
    'grid': 'y-axis only',
    'spines': 'left and bottom only',
    'font': 'Helvetica Neue',
}
```

### 5.3 Figure Registry (figure_registry.yaml)

```yaml
main_figures:
  fig_retained_multi_metric:
    combo_requirement: "standard_4"
    max_curves: 4
    required_combos: ["ground_truth", "best_ensemble", ...]
    outputs:
      pdf: "fig_retained_multi_metric.pdf"
      json: "data/fig_retained_multi_metric.json"
      json_privacy: "public"  # Can be committed

  fig_subject_traces:
    combo_requirement: "extended_6"
    outputs:
      json_privacy: "PRIVATE"  # DO NOT commit - contains subject IDs
```

---

## 6. CURRENT PROBLEMS TO SOLVE

### 6.1 Broken Figures (Priority P0)

1. **`fig_retained_multi_metric`**: Shows only 1 combo, should show 4
2. **`fig_retained_auroc`**: Redundant - DELETE (use multi_metric instead)
3. **`fig_retained_brier`**: Redundant - DELETE
4. **`fig_retained_netbenefit`**: Redundant - DELETE
5. **`fig_retained_scaledbrier`**: Redundant - DELETE

### 6.2 Styling Problems (Priority P1)

1. Generic matplotlib defaults - need Economist/ggplot2 style
2. Inconsistent fonts across figures
3. No CI bands on comparison figures
4. Grid lines too prominent

### 6.3 Architecture Questions (Need Input)

1. **Figure generation**: Should each script be standalone or use central generator?
2. **Config loading**: Is YAML hierarchy (combos → figure_registry → methods) correct?
3. **Data flow**: MLflow → DuckDB → Visualization - is this over-engineered?
4. **Privacy**: How to handle subject-level JSON data safely?

---

## 7. DESIRED OUTCOMES

### 7.1 Immediate Goals

- [ ] Fix retention figures to show 4 combos
- [ ] Apply Economist styling to all figures
- [ ] Delete redundant single-metric figures
- [ ] Ensure all figures have reproducibility JSON

### 7.2 Architecture Goals

- [ ] Single source of truth for figure specs (figure_registry.yaml)
- [ ] Config-driven generation (no hardcoded method names)
- [ ] Consistent styling across all figures
- [ ] Clear privacy handling for subject data

### 7.3 Documentation Goals

- [ ] Hierarchical CLAUDE.md with progressive disclosure
- [ ] Domain-specific context files for each subsystem
- [ ] Validation scripts to check figure completeness

---

## 8. QUESTIONS FOR EXTERNAL REVIEW

1. **Is the data infrastructure over-engineered?**
   - Raw data in SQLite → MLflow experiments → DuckDB views → JSON for viz
   - Could this be simpler?

2. **Is the config system appropriate?**
   - YAML for combos, subjects, figure registry, methods, colors
   - Should these be consolidated?

3. **How should figure generation be organized?**
   - Current: Individual scripts + central generator
   - Alternative: Single script with figure registry dispatch?

4. **Best practices for AI-assisted code documentation?**
   - Currently using CLAUDE.md + domain context files
   - Is there a better pattern for progressive disclosure?

5. **Error propagation analysis approach?**
   - Currently: Compare AUROC across (outlier × imputation) grid
   - Should we do causal analysis? Structural equation modeling?

---

## 9. FILE TREE (Key Paths)

```
foundation_PLR/
├── CLAUDE.md                    # Main context (read this first)
├── .claude/
│   ├── CLAUDE.md                # Behavior contract
│   ├── CONTEXT_FOR_EXTERNAL_AI.md  # This file
│   └── domains/
│       ├── visualization.md     # Viz-specific context
│       ├── mlflow-experiments.md # MLflow details
│       └── manuscript.md        # Paper context
├── config/
│   ├── plot_hyperparam_combos.yaml  # Fixed combos for figures
│   ├── demo_subjects.yaml       # 12 subjects for traces
│   ├── figure_registry.yaml     # Figure metadata catalog
│   ├── methods.yaml             # Method names
│   └── colors.yaml              # Color palette
├── src/
│   ├── viz/                     # Visualization scripts
│   │   ├── plot_config.py       # Styling config
│   │   ├── generate_all_figures.py
│   │   ├── retained_metric.py
│   │   ├── calibration_plot.py
│   │   └── ...
│   ├── config/
│   │   └── loader.py            # Config loading utilities
│   └── data_io/
│       └── duckdb_export.py     # MLflow → DuckDB
├── scripts/
│   └── validate_figures.py      # Figure validation
└── tests/
    └── unit/
        └── test_*.py            # Unit tests
```

---

**END OF CONTEXT DOCUMENT**

When consulting external AI, copy everything above this line.
Include specific code snippets as needed for the question at hand.
