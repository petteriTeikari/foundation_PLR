# Hierarchical Experiment Configuration System

## User Request (Verbatim)

> We need to think of one thing more, not sure if it is relevant for this plan though. Now we have the "paper experiment" as in the hyperparameters combos published in an academic paper, then the debug and this synthetic! The two last Prefect flows though use then hardcoded combos which is fine now, but what about in the future when someone likes to run a new experiment with new combos for the plots without destroying the possibility to use the old combos? We need to add one more level of hierarchy to the .yaml config system, and make it Hydra-like composable? As in we could have "paper_2026.yaml" config that knows that data to load in, what factorial design to reproduce, what combos to plot, what subject examples to plot, etc. And a year from now someone might want to explore another set of TSFM models, so there so could be a new "paper_2027.yaml" added that would pick the correct .yamls then? And when and if someone find more bugs, then we can always run the paper_2026 experiment again exactly in terms of data and all? What do you say? docs/planning/hierarchical-experiment-config.md need to be created for this with comprehensive multi-hypothesis planning of different design choices involved with multiple code reviewer agents optimizing the plan! Then we should integrate the recommendation for the docs/planning/pipeline-robustness-plan.md (if needed) or run another plan after this plan? Think of all these and save my prompt verbatim

---

## Executive Summary

The current configuration system has three experiment types (paper, debug, synthetic) but lacks a versioning and composition layer that would allow:
1. **Reproducibility**: Re-running paper_2026 experiment exactly as published
2. **Extensibility**: Adding paper_2027 experiments without breaking existing ones
3. **Isolation**: Each experiment version is self-contained with its own combos, data, subjects

This document proposes a hierarchical, Hydra-composable experiment configuration architecture.

---

## Current State Analysis

### Existing Configuration Locations

| Config Type | Location | Purpose |
|-------------|----------|---------|
| Main defaults | `configs/defaults.yaml` | Pipeline parameters |
| Visualization | `configs/VISUALIZATION/` | Figure settings |
| Plot combos | `configs/VISUALIZATION/plot_hyperparam_combos.yaml` | Standard 4 combos |
| Demo subjects | `configs/demo_subjects.yaml` | 8 hand-picked subjects |
| MLflow registry | `configs/mlflow_registry/` | Method definitions |
| Figure registry | `configs/VISUALIZATION/figure_registry.yaml` | Figure specs |

### Current Problems

1. **No experiment versioning**: Changing combos for 2027 would overwrite 2026 settings
2. **Hardcoded references**: Prefect flows reference specific YAML files directly
3. **No composition**: Can't easily swap datasets, combos, or subjects independently
4. **Reproducibility risk**: Bug fixes might change behavior of "frozen" experiments

---

## Proposed Architecture

### Design Hypothesis 1: Flat Experiment Configs (Simple)

```yaml
# configs/experiments/paper_2026.yaml
experiment:
  name: "paper_2026"
  version: "1.0.0"

data:
  database: "SERI_PLR_GLAUCOMA.db"
  n_subjects_classification: 208
  n_subjects_preprocessing: 507

visualization:
  combos_file: "combos_paper_2026.yaml"
  demo_subjects_file: "demo_subjects_paper_2026.yaml"
  figure_registry_file: "figure_registry_paper_2026.yaml"

factorial_design:
  outlier_methods: 11
  imputation_methods: 8
  classifiers: 5
```

**Pros**: Simple, explicit, easy to understand
**Cons**: Duplication across experiment files, hard to share common settings

---

### Design Hypothesis 2: Hydra Composition (Recommended)

```yaml
# configs/experiment/paper_2026.yaml
defaults:
  - /data: seri_plr_2026
  - /combos: paper_2026
  - /subjects: demo_8_subjects
  - /figures: publication_ready
  - /mlflow: production

experiment:
  name: "Foundation PLR Paper 2026"
  version: "1.0.0"
  frozen: true  # Prevent accidental modifications

metadata:
  doi: "10.xxxx/foundation-plr-2026"
  publication_date: "2026-XX-XX"
  authors: ["..."]
```

```yaml
# configs/experiment/paper_2027.yaml
defaults:
  - /data: seri_plr_2026  # Same data
  - /combos: paper_2027   # New combos with more TSFM models
  - /subjects: demo_12_subjects  # Extended subject set
  - /figures: publication_ready
  - /mlflow: production

experiment:
  name: "Foundation PLR Extension 2027"
  version: "1.0.0"
  extends: "paper_2026"  # Documents lineage
```

**Directory Structure**:
```
configs/
├── experiment/
│   ├── paper_2026.yaml
│   ├── paper_2027.yaml
│   ├── debug.yaml
│   └── synthetic.yaml
├── data/
│   ├── seri_plr_2026.yaml
│   └── synthetic_small.yaml
├── combos/
│   ├── paper_2026.yaml
│   ├── paper_2027.yaml
│   └── debug_minimal.yaml
├── subjects/
│   ├── demo_8_subjects.yaml
│   └── demo_12_subjects.yaml
├── figures/
│   ├── publication_ready.yaml
│   └── draft_quality.yaml
└── mlflow/
    ├── production.yaml
    └── local_testing.yaml
```

**Pros**: Hydra-native, composable, DRY, clear inheritance
**Cons**: More complex directory structure, learning curve

---

### Design Hypothesis 3: Git-Tagged Configs (Version Control Native)

Use git tags to version entire config directories:
- `experiment/paper_2026_v1.0.0` tag
- Checkout specific tag to reproduce

**Pros**: Uses existing git infrastructure
**Cons**: Doesn't support running multiple experiments in parallel, messy history

---

## Multi-Reviewer Analysis

### Reviewer 1: Reproducibility Expert

**Concerns:**
- How do we guarantee bitwise reproducibility across years?
- What about dependency versions (uv.lock) per experiment?
- Random seed management across experiments?

**Recommendations:**
1. Each experiment config should reference a specific `uv.lock` hash
2. Include `random_seed` in experiment config
3. Store DuckDB checksums for data verification

```yaml
experiment:
  name: "paper_2026"
  reproducibility:
    uv_lock_sha256: "abc123..."
    data_checksum: "def456..."
    random_seed: 42
    python_version: "3.11.10"
```

### Reviewer 2: Code Architecture Expert

**Concerns:**
- How do Prefect flows consume these configs?
- Backwards compatibility with existing code?
- Config validation and schema enforcement?

**Recommendations:**
1. Create `ExperimentConfig` dataclass with Pydantic validation
2. Update Prefect flows to accept `--experiment=paper_2026`
3. Add config schema validation in pre-commit

```python
# src/config/experiment.py
from pydantic import BaseModel
from typing import Literal

class ExperimentConfig(BaseModel):
    name: str
    version: str
    frozen: bool = False
    data: DataConfig
    combos: CombosConfig
    subjects: SubjectsConfig
```

### Reviewer 3: User Experience Expert

**Concerns:**
- How does a new user run the paper_2026 experiment?
- How does a researcher create paper_2027?
- Documentation and discoverability?

**Recommendations:**
1. Single command: `make run-experiment EXPERIMENT=paper_2026`
2. Template generator: `make new-experiment NAME=paper_2027 BASE=paper_2026`
3. Add `configs/experiments/README.md` with clear instructions

```bash
# Run the 2026 paper experiment
make run-experiment EXPERIMENT=paper_2026

# Create a new experiment based on 2026
make new-experiment NAME=paper_2027 BASE=paper_2026
```

### Reviewer 4: CI/CD Expert

**Concerns:**
- How do we test all experiment configs in CI?
- What happens when a bug fix affects multiple experiments?
- Dependency between experiments?

**Recommendations:**
1. CI job that validates all experiment configs load correctly
2. "Frozen" experiments run in separate CI pipeline
3. Create `experiments.lock` file listing all valid experiments

```yaml
# .github/workflows/experiment-validation.yml
jobs:
  validate-experiments:
    steps:
      - name: Validate all experiment configs
        run: |
          for exp in configs/experiment/*.yaml; do
            python -c "from src.config import load_experiment; load_experiment('$exp')"
          done
```

---

## Implementation Plan

### Phase 1: Config Structure (Week 1) - COMPLETED 2026-02-01

1. [x] Create `configs/experiment/` directory
2. [x] Create sub-directories: `data/`, `combos/`, `subjects/`, `figures_config/`, `mlflow_config/`
3. [x] Created base configs for each category
4. [x] Create `paper_2026.yaml` experiment config

**Files created:**
- `configs/experiment/paper_2026.yaml` - Main paper experiment
- `configs/experiment/synthetic.yaml` - CI testing experiment
- `configs/data/seri_plr_2026.yaml` - SERI dataset config
- `configs/data/synthetic_small.yaml` - Synthetic dataset config
- `configs/combos/paper_2026.yaml` - Standard combos
- `configs/combos/debug_minimal.yaml` - Debug combos
- `configs/subjects/demo_8_subjects.yaml` - Demo subject selection
- `configs/figures_config/publication_ready.yaml` - Publication quality
- `configs/figures_config/draft_quality.yaml` - Draft quality
- `configs/mlflow_config/production.yaml` - Production MLflow
- `configs/mlflow_config/local_testing.yaml` - Local testing MLflow
- `configs/experiment/README.md` - Documentation

### Phase 2: Validation Layer (Week 2) - COMPLETED 2026-02-02

1. [x] Create `src/config/experiment.py` with Pydantic models
2. [x] Add `load_experiment()` function with validation
3. [x] Create `src/config/loader.py` for YAML loading
4. [x] Add `list_experiments()` and `validate_experiment_config()`

**Files created:**
- `src/config/__init__.py` - Module exports
- `src/config/loader.py` - YAMLConfigLoader with caching
- `src/config/experiment.py` - Pydantic models (ExperimentConfig, etc.)
- `tests/unit/test_experiment_config.py` - 14 unit tests

**Usage:**
```python
from src.config import load_experiment, list_experiments
experiments = list_experiments()  # ['paper_2026', 'synthetic']
cfg = load_experiment('paper_2026')
print(f'{cfg.name} v{cfg.version}, frozen={cfg.is_frozen}')
```

### Phase 3: Flow Integration (Week 3) - COMPLETED 2026-02-02

1. [x] Update `Makefile` with experiment targets
2. [x] Create `make new-experiment` template generator
3. [x] Add experiment validation script

**Make targets added:**
- `make list-experiments` - List available experiments
- `make run-experiment EXPERIMENT=paper_2026` - Run specific experiment
- `make new-experiment NAME=X BASE=Y` - Create from template
- `make validate-experiments` - Validate all configs

**Files created:**
- `scripts/validate_experiments.py` - Validation script

### Phase 4: CI Integration (Week 4)

1. [ ] Add experiment validation CI job
2. [ ] Add frozen experiment smoke tests
3. [ ] Create `experiments.lock` tracking file
4. [ ] Add experiment diff tool for comparing versions

---

## Migration Strategy

### Step 1: Non-Breaking Addition

Add new structure alongside existing configs. No breaking changes.

```
configs/
├── defaults.yaml           # Keep existing
├── VISUALIZATION/          # Keep existing
├── experiment/             # NEW
│   └── paper_2026.yaml
```

### Step 2: Gradual Adoption

Update code to prefer new structure with fallback to old:

```python
def load_combos(experiment: str = None):
    if experiment:
        return load_experiment_combos(experiment)
    else:
        # Fallback to legacy
        return load_yaml("configs/VISUALIZATION/plot_hyperparam_combos.yaml")
```

### Step 3: Deprecation

After all code migrated, deprecate direct config file references.

---

## Integration with Pipeline Robustness Plan

This plan **complements** `docs/planning/pipeline-robustness-plan.md`:

| Robustness Plan Phase | Integration Point |
|----------------------|-------------------|
| Phase 2: Unit Tests | Add tests for `load_experiment()` validation |
| Phase 3: Integration Tests | Use experiment configs for CI smoke tests |
| Phase 4: Runtime Assertions | Validate experiment config at pipeline start |
| Phase 5: Pre-commit Hooks | Add experiment config schema validation |

**Recommendation**: Implement this plan **after** completing Phases 1-3 of the robustness plan, then integrate into Phases 4-5.

---

## Open Questions

1. **Should experiment configs be immutable after publication?**
   - Option A: Strict immutability (create paper_2026_v2 for fixes)
   - Option B: Allow bug fixes with changelog

2. **How to handle external data dependencies?**
   - Option A: Store data checksums only
   - Option B: Use DVC for data versioning

3. **Should synthetic/debug experiments follow the same structure?**
   - Option A: Yes, for consistency
   - Option B: No, keep them simpler

---

## Decision Matrix

| Criterion | Hypothesis 1 (Flat) | Hypothesis 2 (Hydra) | Hypothesis 3 (Git Tags) |
|-----------|---------------------|----------------------|-------------------------|
| Reproducibility | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Maintainability | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| User Experience | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| CI/CD Integration | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Complexity | ⭐⭐⭐⭐⭐ (low) | ⭐⭐⭐ (medium) | ⭐⭐⭐⭐ (low-medium) |

**Recommendation**: **Hypothesis 2 (Hydra Composition)** provides the best balance of reproducibility, maintainability, and extensibility for a research codebase.

---

## Next Steps

1. [ ] Review this document with stakeholders
2. [ ] Decide on open questions
3. [ ] Create implementation tickets
4. [ ] Begin Phase 1 after robustness plan Phase 3 complete

---

## Appendix: Example Full Experiment Config

```yaml
# configs/experiment/paper_2026.yaml
defaults:
  - /data: seri_plr_2026
  - /combos: paper_2026
  - /subjects: demo_8_subjects
  - /figures: publication_ready
  - /mlflow: production
  - _self_

experiment:
  name: "Foundation PLR Paper 2026"
  version: "1.0.0"
  frozen: true
  description: |
    Main experiment for the Foundation PLR paper evaluating
    TSFM models for PLR preprocessing.

metadata:
  doi: "10.xxxx/foundation-plr-2026"
  publication_date: "2026-XX-XX"
  manuscript_repo: "/path/to/sci-llm-writer/manuscripts/foundationPLR"

reproducibility:
  random_seed: 42
  python_version: "3.11.10"
  uv_lock_sha256: "abc123def456..."
  data_checksums:
    seri_plr_glaucoma_db: "sha256:..."
    mlruns_pickle_count: 542

factorial_design:
  outlier_methods: 11  # From mlflow_registry
  imputation_methods: 8
  classifiers: 5
  featurization_methods: 2

expected_results:
  ground_truth_auroc: 0.911
  best_ensemble_auroc: 0.913
  n_bootstrap_iterations: 1000
```
