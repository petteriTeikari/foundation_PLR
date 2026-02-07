# Config Documentation & Infographics Plan

**Status:** REVIEWED - Ready for implementation
**Goal:** Lower barrier to entry for developers via progressive disclosure documentation
**Scope:** All `configs/` subdirectories
**Reviewer:** Plan agent (iteration 1)

---

## High-Level Vision

### The Problem

A developer opens `configs/CLS_HYPERPARAMS/CATBOOST_hyperparam_space.yaml` and sees:

```yaml
depth:
  - 1
  - 3
lr:
  - 0.001
  - 0.1
```

**Questions they have:**
- What does `depth` actually control in CatBoost?
- Why is the range [1, 3] and not [1, 10]?
- What's the computational/accuracy tradeoff?
- Where can I learn more?

### The Solution: Progressive Disclosure

```
┌─────────────────────────────────────────────────────────────────────┐
│  LEVEL 1: INFOGRAPHIC (5 seconds)                                   │
│  Visual diagram showing key concepts                                │
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

**Key Principle:** Information lives in ONE place. YAML files link to READMEs. READMEs link to infographics. No duplication.

---

## CLAUDE.md Updates Required

### 1. Update `docs/repo-figures/CLAUDE.md`

Add section on progressive disclosure philosophy and documentation purpose.

### 2. Create `docs/repo-figures/plans-TODO/CLAUDE.md`

Subdirectory-specific rules allowing both:
- Code/config documentation (paths, composition)
- Methodological context (brief theory without results)

---

## Current State Audit

### Existing README Files (4 total)

| File | Lines | Quality | Action |
|------|-------|---------|--------|
| `configs/README.md` | 254 | **Comprehensive** (Mermaid diagrams, Hydra examples) | UPDATE: Add infographic index |
| `configs/mlflow_registry/README.md` | 106 | **Comprehensive** | UPDATE: Add infographic reference |
| `configs/experiment/README.md` | 103 | **Comprehensive** | UPDATE: Add infographic reference |
| `configs/OUTLIER_MODELS/README.md` | 34 | **Basic** | EXPAND: Add method explanations |

### Existing Figure Plans (102 total in `docs/repo-figures/figure-plans/`)

Relevant existing plans:
- `fig-repo-05-hydra-config-system.md` - Hydra composition (DO NOT DUPLICATE)
- `fig-repo-11-stratos-metrics.md` - Metrics (DO NOT DUPLICATE)

### Config Directory Inventory

| Directory | Files | README | Infographic Needed | Priority |
|-----------|-------|--------|-------------------|----------|
| `CLS_HYPERPARAMS/` | 5 | **NO** | YES (fig-repo-51) | HIGH |
| `CLS_MODELS/` | 5 | **NO** | Covered by fig-repo-51 | MEDIUM |
| `CLS_TS_MODELS/` | 1 | **NO** | NO (niche) | LOW |
| `MODELS/` | 7 | **NO** | YES (fig-repo-54) | HIGH |
| `OUTLIER_MODELS/` | 10+README | **YES (basic)** | YES (fig-repo-53) | HIGH |
| `PLR_FEATURIZATION/` | 2 | **NO** | NO (simple, README sufficient) | MEDIUM |
| `PLR_EMBEDDING/` | 1 | **NO** | NO (niche) | LOW |
| `VISUALIZATION/` | 10 | **NO** | NO (self-documenting colors) | LOW |
| `mlflow_registry/` | 12 | **YES** | YES (fig-repo-55) | HIGH |
| `experiment/` | 3+README | **YES** | YES (fig-repo-56) | MEDIUM |
| `combos/` | 2 | **NO** | Covered by fig-repo-56 | MEDIUM |
| `data/` | 2 | **NO** | Covered by fig-repo-56 | LOW |
| `figures_config/` | 2 | **NO** | NO (simple) | LOW |
| `mlflow_config/` | 2 | **NO** | NO (simple) | LOW |
| `schema/` | 1 | **NO** | NO (niche) | LOW |
| `subjects/` | 1 | **NO** | NO (simple) | LOW |
| `SERVICES/` | 1 | **NO** | NO (simple) | LOW |

---

## Infographics Plan (6 Total)

### Consolidated from original 10 → 6

| ID | Title | Purpose | Covers |
|----|-------|---------|--------|
| **fig-repo-51** | Classifier Configuration Architecture | Hyperparameters + architecture for all 5 classifiers | `CLS_HYPERPARAMS/`, `CLS_MODELS/` |
| **fig-repo-52** | Classifier Paradigms | Linear → Tree Ensemble → Foundation model evolution | Conceptual understanding |
| **fig-repo-53** | Outlier Detection Methods | The 11 registry methods categorized | `OUTLIER_MODELS/`, registry alignment |
| **fig-repo-54** | Imputation Model Landscape | SAITS, CSDI, MOMENT, TimesNet, linear | `MODELS/` |
| **fig-repo-55** | Registry as Single Source of Truth | Data flow showing registry → code → validation | `mlflow_registry/` |
| **fig-repo-56** | Experiment Configuration Hierarchy | How paper_2026.yaml composes sub-configs | `experiment/`, `combos/`, `data/` |

### Removed Infographics (Rationale)

| Original ID | Reason Removed |
|-------------|----------------|
| fig-repo-41 | Merged into fig-repo-51 (tree depth is detail, not separate) |
| fig-repo-45 | Risks showing results; featurization is simple enough for README |
| fig-repo-46 | Colors are self-documenting in YAML with good comments |
| fig-repo-48 | Already exists as fig-repo-05 (Hydra composition) |
| fig-repo-49 | Already in configs/README.md (Mermaid diagram) |

---

## README Plan

### READMEs to Create (6)

| Directory | Priority | Content Focus |
|-----------|----------|---------------|
| `CLS_HYPERPARAMS/` | HIGH | Each classifier's hyperparameters explained |
| `CLS_MODELS/` | MEDIUM | Classifier fixed params, weighting strategies |
| `MODELS/` | HIGH | Deep learning imputation models |
| `PLR_FEATURIZATION/` | MEDIUM | Handcrafted feature extraction |
| `combos/` | MEDIUM | Combo rationale, naming conventions |
| `VISUALIZATION/` | LOW | Figure styling guide |

### READMEs to Update (4)

| File | Updates Needed |
|------|----------------|
| `configs/README.md` | Add infographic index section |
| `mlflow_registry/README.md` | Add fig-repo-55 reference |
| `experiment/README.md` | Add fig-repo-56 reference |
| `OUTLIER_MODELS/README.md` | Expand with 11 method explanations |

### README Template

```markdown
# [Directory Name]

> **Quick Guide**: See [infographic](../repo-figures/fig-repo-XX.png) for visual overview

## Purpose

One sentence explaining what this directory contains.

## Files

| File | Purpose |
|------|---------|
| `file1.yaml` | Description |
| `file2.yaml` | Description |

## Key Parameters Explained

### Parameter 1

**What it does**: Brief explanation
**Why this range**: Rationale for the search space
**Trade-offs**: Computational vs accuracy considerations

### Parameter 2

...

## Hydra Usage

```bash
python src/pipeline.py COMPONENT=file1
```

## See Also

- Related configs: `../OTHER_DIR/`
- External docs: [Link to official documentation]

---

**Note**: Performance results are documented in the manuscript, not this repository.
```

---

## YAML Comment Template (Minimal)

**Principle:** YAML comments should be minimal. Explanations live in README.

```yaml
_version: "1.0.0"
# ═══════════════════════════════════════════════════════════════════
# CATBOOST HYPERPARAMETER SEARCH SPACE
# ═══════════════════════════════════════════════════════════════════
# See README.md for parameter explanations
# See docs/repo-figures/fig-repo-51.png for visual guide
# ═══════════════════════════════════════════════════════════════════

CATBOOST:
  HYPERPARAMS:
    method: 'OPTUNA'
    metric_val: 'auc'

  SEARCH_SPACE:
    OPTUNA:
      depth: [1, 3]        # Tree depth - see README
      lr: [0.001, 0.1]     # Learning rate - see README
      l2_leaf_reg: [1, 30] # Regularization - see README
```

**Maximum comment overhead: 10 lines header + 1 inline comment per param**

---

## Implementation Phases

### Phase 1: Foundation (3 days)

- [ ] Update `docs/repo-figures/CLAUDE.md` with progressive disclosure philosophy
- [ ] Create `docs/repo-figures/plans-TODO/CLAUDE.md`
- [ ] Update `configs/README.md` with infographic index placeholder
- [ ] Validate README template with `CLS_HYPERPARAMS/`

**Deliverables:**
- Updated CLAUDE.md files
- `CLS_HYPERPARAMS/README.md` (first README)

### Phase 2: Core Classifier Docs (1 week)

- [ ] Create fig-repo-51 plan (Classifier Config Architecture)
- [ ] Create fig-repo-52 plan (Classifier Paradigms)
- [ ] Generate fig-repo-51 infographic
- [ ] Generate fig-repo-52 infographic
- [ ] Create `CLS_MODELS/README.md`
- [ ] Add minimal comments to 5 `*_hyperparam_space.yaml` files

**Deliverables:**
- 2 infographics
- 2 READMEs
- 5 YAML files with improved headers

### Phase 3: Preprocessing Pipeline (1 week)

- [ ] Create fig-repo-53 plan (Outlier Detection - aligned with 11 registry methods)
- [ ] Create fig-repo-54 plan (Imputation Models)
- [ ] Generate fig-repo-53 infographic
- [ ] Generate fig-repo-54 infographic
- [ ] Expand `OUTLIER_MODELS/README.md` (currently 34 lines → ~150 lines)
- [ ] Create `MODELS/README.md`
- [ ] Create `PLR_FEATURIZATION/README.md`

**Deliverables:**
- 2 infographics
- 3 READMEs (1 expansion, 2 new)

### Phase 4: System Understanding (1 week)

- [ ] Create fig-repo-55 plan (Registry Pattern)
- [ ] Create fig-repo-56 plan (Experiment Hierarchy)
- [ ] Generate fig-repo-55 infographic
- [ ] Generate fig-repo-56 infographic
- [ ] Update `mlflow_registry/README.md` with infographic reference
- [ ] Update `experiment/README.md` with infographic reference
- [ ] Create `combos/README.md`

**Deliverables:**
- 2 infographics
- 3 README updates

### Phase 5: Polish (3 days)

- [ ] Create `VISUALIZATION/README.md`
- [ ] Cross-reference all READMEs
- [ ] Validate all infographic links
- [ ] Update main `configs/README.md` index with all infographic links
- [ ] Document `_version_manifest.yaml` usage

**Deliverables:**
- 1 README
- All cross-references validated

---

## Infographic Design Guidelines

### DO:
- Use Mermaid diagrams (render in GitHub, maintainable)
- Generate programmatically where possible (from config data)
- Keep text minimal - visual should be self-explanatory
- Include legend/key
- Reference specific config files

### DON'T:
- Hand-draw ASCII art (becomes stale)
- Include performance numbers (results belong in manuscript)
- Duplicate information that's in README
- Create overly complex diagrams

### Tool Recommendations

| Tool | Use Case | Maintainability |
|------|----------|-----------------|
| Mermaid | Architecture diagrams, flowcharts | HIGH (text-based, versioned) |
| Python/matplotlib | Generated from config data | HIGH (code generates diagram) |
| Draw.io/Figma | Complex conceptual diagrams | MEDIUM (needs manual updates) |

---

## Registry Alignment (Critical)

### fig-repo-53 MUST show exactly 11 outlier methods:

| # | Method | Category |
|---|--------|----------|
| 1 | `pupil-gt` | Ground Truth |
| 2 | `MOMENT-gt-finetune` | Foundation Model |
| 3 | `MOMENT-gt-zeroshot` | Foundation Model |
| 4 | `UniTS-gt-finetune` | Foundation Model |
| 5 | `TimesNet-gt` | Deep Learning |
| 6 | `LOF` | Traditional |
| 7 | `OneClassSVM` | Traditional |
| 8 | `PROPHET` | Traditional |
| 9 | `SubPCA` | Traditional |
| 10 | `ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune` | Ensemble |
| 11 | `ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune` | Ensemble |

**Source:** `configs/mlflow_registry/parameters/classification.yaml`

---

## Anti-Patterns to Avoid

| Anti-Pattern | How We Avoid It |
|--------------|-----------------|
| Results in docs | No AUROC/accuracy. Say "conceptually different" not "better" |
| Duplication | YAML links to README. README links to infographic. |
| Unmaintainable diagrams | Use Mermaid or code-generated. No hand-drawn ASCII. |
| Over-documentation | 6 infographics, not 10. Not every config needs one. |
| Stale references | Use relative paths. Test links in CI. |

---

## Success Criteria

1. **Developer can understand any config in <5 minutes**
   - Infographic provides instant visual context
   - README explains all parameters
   - YAML comments link to details (not duplicate)

2. **Zero results in documentation**
   - No AUROC, accuracy, or performance numbers
   - No "X is better than Y"
   - Methodology/concepts only

3. **Single source of truth maintained**
   - Method counts match registry exactly
   - Colors/combos reference configs, not hardcoded

4. **Maintainable**
   - Mermaid/code-generated diagrams where possible
   - README template ensures consistency
   - Version manifest tracks config changes

---

## Open Items

1. **Infographic generation tool**: Decide between Mermaid-only vs Python generation
2. **CI validation**: Add check that README links to infographics work
3. **Version manifest documentation**: Add section explaining `_version_manifest.yaml`

---

## References

### For Hyperparameter Explanations

| Classifier | Key Reference |
|------------|---------------|
| CatBoost | Prokhorenkova et al. 2018 "CatBoost: unbiased boosting with categorical features" |
| XGBoost | Chen & Guestrin 2016 "XGBoost: A Scalable Tree Boosting System" |
| LogReg | Hastie et al. ESL Chapter 4 |
| TabPFN | Hollmann et al. 2023 ICLR "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second" |
| Tree-based advantage | Grinsztajn et al. 2022 NeurIPS "Why do tree-based models still outperform deep learning on tabular data?" |

### For Imputation Models

| Model | Key Reference |
|-------|---------------|
| SAITS | Du et al. 2023 "SAITS: Self-Attention-based Imputation for Time Series" |
| CSDI | Tashiro et al. 2021 "CSDI: Conditional Score-based Diffusion Models" |
| MOMENT | Goswami et al. 2024 "MOMENT: A Family of Open Time-series Foundation Models" |

### For Outlier Detection

| Method | Key Reference |
|--------|---------------|
| LOF | Breunig et al. 2000 "LOF: Identifying Density-Based Local Outliers" |
| One-Class SVM | Schölkopf et al. 2001 "Estimating the Support of a High-Dimensional Distribution" |
| PROPHET | Taylor & Letham 2018 "Forecasting at Scale" |
