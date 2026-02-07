# /figures - Figure Management

Manage manuscript figures for Foundation PLR.

## Usage

- `/figures` or `/figures list` - List all figures from figure_registry.yaml
- `/figures run <ID>` - Generate a specific figure (e.g., `/figures run R7`)
- `/figures all` - Generate all figures
- `/figures status` - Show which figures exist and which need regeneration

## Figure Registry

All figures are defined in `configs/VISUALIZATION/figure_registry.yaml`. This is the single source of truth.

## Generation Commands

```bash
# All figures
python src/viz/generate_all_figures.py

# Specific figure
python src/viz/generate_all_figures.py --figure <ID>

# List available
python src/viz/generate_all_figures.py --list
```

## Standard Combos (Main Figures)

Main figures use 4 standard combos from `configs/VISUALIZATION/plot_hyperparam_combos.yaml`:
- `ground_truth` - pupil-gt + pupil-gt + CatBoost (AUROC: 0.9110)
- `best_ensemble` - Ensemble + CSDI + CatBoost (AUROC: 0.9130)
- `best_single_fm` - MOMENT-gt-finetune + SAITS + CatBoost (AUROC: 0.9099)
- `traditional` - LOF + SAITS + TabPFN (AUROC: 0.8599)

## Output Locations

- Figures: `figures/generated/` (PDF, PNG)
- Data: `figures/generated/data/` (JSON for reproducibility)
- Manuscript copy: `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/figures/generated/`

## Critical Rules

1. **NEVER hardcode combo names** - always load from YAML
2. **ALWAYS include ground_truth** in comparison figures
3. **Main figures: max 4 curves**
4. **ALWAYS call setup_style()** before plotting
