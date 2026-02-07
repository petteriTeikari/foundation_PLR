# Figure Plans - Claude Instructions

## CRITICAL RULE: Repository Figures Document CODE, Not Results

**Repository figure plans document HOW the code works, not WHAT the results are.**

### What Repository Figures SHOULD Show

| ✅ GOOD | ❌ BAD |
|---------|--------|
| Config file locations and structure | Performance metrics (AUROC, accuracy) |
| Code paths and module architecture | Method comparisons ("X beats Y by Zpp") |
| Hydra composition patterns | Bar charts of results |
| Hyperparameter search spaces | Rankings of methods |
| How to add new methods | "The finding" or "key result" |
| Directory structures | Statistical comparisons |

### Why This Separation Exists

- **Repository figures**: For developers understanding the codebase
- **Manuscript figures**: For readers understanding the research findings

Results belong in:
- `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/latent-methods-results/`

### Required Elements in Figure Plans

Every figure plan MUST include:

1. **Config locations**: Where are the YAML files?
2. **Code paths**: Which Python/R modules implement this?
3. **Hydra composition**: How do configs compose?
4. **Extension guide**: How would a user add a new variant?

### Self-Check Before Writing a Figure Plan

Ask yourself:
- [ ] Am I showing HOW something is configured, or HOW WELL it performs?
- [ ] Would this help a developer understand the code?
- [ ] Are there any AUROC/accuracy/performance numbers? (If yes, REMOVE them)
- [ ] Does this reference actual file paths in the repo?
- [ ] Is there a "Results are in the manuscript" note?

## Classifier Documentation Guidelines

When documenting classifiers (CatBoost, XGBoost, TabPFN, etc.):

### SHOW These

1. **Config structure**:
   - `configs/CLS_MODELS/{CLASSIFIER}.yaml` - Model settings
   - `configs/CLS_HYPERPARAMS/{CLASSIFIER}_hyperparam_space.yaml` - HPO space

2. **Key hyperparameters** (without stating which values are "best"):
   - CatBoost: `depth`, `lr`, `l2_leaf_reg`, `colsample_bylevel`
   - XGBoost: `max_depth`, `eta`, `gamma`, `reg_lambda`, `n_estimators`
   - LogisticRegression: `penalty`, `C`, `solver`

3. **HPO method used**:
   - CatBoost: Optuna
   - XGBoost: hyperopt

4. **Architecture differences** (conceptual):
   - Logistic Regression: Linear decision boundary
   - Tree-based (XGBoost, CatBoost): Gradient boosting, handles non-linearity
   - TabPFN: Transformer-based, prior-data fitted network

5. **How to add new classifiers**: Extension pattern

### DON'T SHOW These

- "CatBoost achieved 0.878 AUROC"
- "XGBoost is 2nd best"
- Rankings or comparisons
- "The optimal depth is 3"

## Featurization Documentation Guidelines

When documenting featurization (handcrafted vs embeddings):

### SHOW These

1. **Config locations**:
   - `configs/PLR_FEATURIZATION/` for handcrafted
   - `configs/PLR_EMBEDDING/` for embeddings

2. **Feature definitions** (YAML structure):
   - `time_from`, `time_start`, `time_end`, `stat`

3. **MOMENT task modes**:
   - `"embedding"`, `"reconstruction"`, `"forecasting"`

4. **Code paths** for each approach

5. **External references** (MOMENT repo, tutorials)

### DON'T SHOW These

- "Handcrafted beats embeddings by 9pp"
- AUROC comparisons
- "Use handcrafted because it's better"

## Template: Classifier/Method Figure Plan

```markdown
# fig-repo-XX: [Method Name] Configuration

## Purpose
Document how [Method] is configured in this repository.

## Config Architecture

### Model Config
Location: `configs/CLS_MODELS/{METHOD}.yaml`
```yaml
# Actual YAML from repo
```

### Hyperparameter Space
Location: `configs/CLS_HYPERPARAMS/{METHOD}_hyperparam_space.yaml`
```yaml
# Actual YAML from repo
```

## Code Path
```
src/classification/
├── flow_classification.py
└── {method}/
    └── {method}_main.py
```

## Hydra Composition
How this config is loaded via Hydra overrides.

## Adding a New Variant
Steps to add a similar method.

## References
- Paper citations
- GitHub repos
- Documentation links

Note: Performance comparisons are in the manuscript, not this repository.
```

---

## Progressive Disclosure Documentation Philosophy

Repository documentation follows a **progressive disclosure** pattern to lower the barrier to entry:

```
┌─────────────────────────────────────────────────────────────────────┐
│  LEVEL 1: INFOGRAPHIC (5 seconds)                                   │
│  Visual diagram showing key concepts - instant overview             │
│                                                                     │
│  LEVEL 2: README.md (2 minutes)                                     │
│  Tutorial explaining each parameter with rationale                  │
│                                                                     │
│  LEVEL 3: YAML COMMENTS (30 seconds scan)                           │
│  Brief header + link to README (not duplicated content)             │
│                                                                     │
│  LEVEL 4: EXTERNAL REFERENCES (deep dive)                           │
│  Links to papers, docs, tutorials                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Principle: Single Source of Truth

Information lives in ONE place:
- **Infographics** → Visual overview of architecture/concepts
- **READMEs** → Full parameter explanations and rationale
- **YAML comments** → Minimal header + LINK to README
- **External refs** → Academic papers, official docs

**NEVER duplicate content between levels.** YAML comments should NOT repeat what's in README.

### Infographic Index

| ID | Title | Config Directories Covered |
|----|-------|---------------------------|
| fig-repo-51 | Classifier Configuration Architecture | `CLS_HYPERPARAMS/`, `CLS_MODELS/` |
| fig-repo-52 | Classifier Paradigms | Conceptual classifier understanding |
| fig-repo-53 | Outlier Detection Methods | `OUTLIER_MODELS/`, registry alignment |
| fig-repo-54 | Imputation Model Landscape | `MODELS/` |
| fig-repo-55 | Registry as Single Source of Truth | `mlflow_registry/` |
| fig-repo-56 | Experiment Configuration Hierarchy | `experiment/`, `combos/`, `data/` |

---

## Literature Context (Use When Relevant)

When adding literature context to classifier figures, reference these paradigms:

| Paradigm | Examples | Key Idea |
|----------|----------|----------|
| Linear models | Logistic Regression, SVM | Linear decision boundary, interpretable |
| Tree ensembles | XGBoost, CatBoost, LightGBM | Gradient boosting, feature importance |
| Neural tabular | TabNet, NODE | Deep learning on tabular data |
| Foundation tabular | TabPFNv2, TabR | Pre-trained on synthetic/real data |

Common references:
- Christodoulou 2019 JCE (ML vs LR systematic review)
- Grinsztajn 2022 NeurIPS (tree-based models on tabular)
- Hollmann 2023 ICLR (TabPFN)
- TabPFNv2 (2024) - Prior-data fitted networks
