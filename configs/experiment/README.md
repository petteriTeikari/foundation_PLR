# Experiment Configuration System

> **Quick Visual Guides** (5-second overview)

[![Experiment Configuration Hierarchy: Shows Hydra's composable config system - config groups as reusable Lego blocks (CLS_MODELS, MODELS, OUTLIER_MODELS), multi-experiment reuse (paper_2026, paper_2027 sharing CatBoost), selective composition, and override hierarchy (CLI > experiment > combos > groups > defaults).](../../docs/repo-figures/assets/fig-repo-56-experiment-config-hierarchy.jpg)](../../docs/repo-figures/assets/fig-repo-56-experiment-config-hierarchy.jpg)

*Figure: Hydra composition hierarchy - config groups are reusable building blocks. One CatBoost.yaml serves all experiments. [Hydra tutorial →](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/)*

[![Hierarchical Experiment Config System: Shows paper_2026.yaml composing from 5 sub-configs (data, combos, subjects, figures_config, mlflow_config), Pydantic validation layer, Make commands (new-experiment, validate-experiments), and frozen vs unfrozen experiments.](../../docs/repo-figures/assets/fig-repo-43-experiment-config-system.jpg)](../../docs/repo-figures/assets/fig-repo-43-experiment-config-system.jpg)

*Figure: Experiment config with Pydantic validation. Frozen experiments (paper_2026) cannot be modified after publication. Use `make new-experiment` to create new ones. [Pydantic docs →](https://docs.pydantic.dev/)*

---

This directory contains Hydra-composable experiment configurations for the Foundation PLR pipeline.

## Quick Start

```bash
# Run paper 2026 experiment
python src/pipeline_PLR.py +experiment=paper_2026

# Run synthetic experiment (CI testing)
python src/pipeline_PLR.py +experiment=synthetic

# Or via Make:
make run-experiment EXPERIMENT=paper_2026
```

## Directory Structure

```
configs/
├── experiment/          # Top-level experiment configs
│   ├── paper_2026.yaml  # Production paper config (FROZEN)
│   ├── synthetic.yaml   # CI testing config
│   └── README.md        # This file
├── data/                # Data source configurations
│   ├── seri_plr_2026.yaml
│   └── synthetic_small.yaml
├── combos/              # Hyperparameter combinations
│   ├── paper_2026.yaml
│   └── debug_minimal.yaml
├── subjects/            # Demo subject selections
│   └── demo_8_subjects.yaml
├── figures_config/      # Figure styling
│   ├── publication_ready.yaml
│   └── draft_quality.yaml
└── mlflow_config/       # MLflow settings
    ├── production.yaml
    └── local_testing.yaml
```

## Composition Pattern

Experiment configs use Hydra's composition feature:

```yaml
defaults:
  - /data: seri_plr_2026      # Data source
  - /combos: paper_2026        # Method combinations
  - /subjects: demo_8_subjects # Demo subjects
  - /figures_config: publication_ready
  - /mlflow_config: production
  - _self_
```

## Creating a New Experiment

1. **Copy an existing experiment config:**
   ```bash
   cp configs/experiment/paper_2026.yaml configs/experiment/paper_2027.yaml
   ```

2. **Modify the defaults to reference new configs:**
   ```yaml
   defaults:
     - /combos: paper_2027  # Your new combos
   ```

3. **Create new component configs as needed:**
   ```bash
   cp configs/combos/paper_2026.yaml configs/combos/paper_2027.yaml
   ```

4. **Update the new experiment metadata:**
   ```yaml
   experiment:
     name: "Foundation PLR Extension 2027"
     version: "1.0.0"
     extends: "paper_2026"  # Document lineage
   ```

## Frozen Experiments

Experiments marked with `frozen: true` should NOT be modified after publication.
If fixes are needed, create a new version:

```yaml
# paper_2026_v2.yaml
experiment:
  name: "Foundation PLR Paper 2026 v2"
  version: "2.0.0"
  frozen: true
  extends: "paper_2026"
  changelog: |
    - Fixed calibration computation bug
```

## See Also

- `docs/planning/hierarchical-experiment-config.md` - Design document
- `docs/planning/pipeline-robustness-plan.md` - Robustness improvements
- `configs/mlflow_registry/README.md` - Method registry (SINGLE SOURCE OF TRUTH)
