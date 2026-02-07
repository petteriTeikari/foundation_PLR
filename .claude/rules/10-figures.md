# FIGURE RULES - Before Any Visualization

## Mandatory Steps

1. **CHECK** `configs/VISUALIZATION/figure_registry.yaml` for figure specification
2. **LOAD** combos from `configs/VISUALIZATION/plot_hyperparam_combos.yaml`
3. **CALL** `setup_style()` before any matplotlib operations
4. **SAVE** JSON data for reproducibility

## Constraints

| Rule | Limit | Enforcement |
|------|-------|-------------|
| Main figure curves | MAX 4 | Validator |
| Supplementary curves | MAX 8 | Validator |
| Ground truth | REQUIRED | All comparisons |
| Hardcoded combos | FORBIDDEN | Validator |
| Hardcoded colors | FORBIDDEN | Use COLORS dict |

**NOTE**: Main vs supplementary assignment may change during manuscript preparation.
Code should support both modes - control via config, not hardcoded limits.

## Standard 4 Combos (Main Figures)

From `configs/VISUALIZATION/plot_hyperparam_combos.yaml`:

| ID | Outlier | Imputation | AUROC |
|----|---------|------------|-------|
| ground_truth | pupil-gt | pupil-gt | 0.9110 |
| best_ensemble | Ensemble | CSDI | 0.9130 |
| best_single_fm | MOMENT-gt-finetune | SAITS | 0.9099 |
| traditional | LOF | SAITS | 0.8599 |

## Correct Pattern

```python
import yaml
from plot_config import setup_style, COLORS, save_figure

# 1. Load combos from YAML (never hardcode!)
cfg = yaml.safe_load(open("configs/VISUALIZATION/plot_hyperparam_combos.yaml"))
combos = cfg["standard_combos"]

# 2. Setup style first
setup_style()

# 3. Use semantic colors
for combo in combos:
    ax.plot(..., color=COLORS[combo["id"]])

# 4. Save with data
save_figure(fig, "fig_name", data=data_dict)
```

## Privacy

- Subject-level JSON files are PRIVATE (excluded from git)
- Check `json_privacy` field in figure_registry.yaml
- Pattern: `**/subject_*.json`, `**/individual_*.json`
