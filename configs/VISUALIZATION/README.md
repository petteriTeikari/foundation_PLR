# VISUALIZATION - Figure and Plot Configuration

## Purpose

Centralized configuration for all visualization aspects: colors, styles, figure dimensions, data filters, and standard method combinations for plotting.

## Files Overview

| File | Purpose |
|------|---------|
| `colors.yaml` | Semantic color palette definitions |
| `figure_style.yaml` | Typography, DPI, theme settings |
| `figure_registry.yaml` | Registry of all manuscript figures |
| `figure_layouts.yaml` | Panel layouts and dimensions |
| `plot_hyperparam_combos.yaml` | Standard method combos for plots |
| `methods.yaml` | Method display names and abbreviations |
| `metrics.yaml` | Metric groupings for different analyses |
| `demo_subjects.yaml` | 8 demo subjects for case studies |
| `data_filters.yaml` | Query filters for data extraction |

## Color Palette

`colors.yaml` defines semantic colors aligned with the Economist style:

### Background Colors

| Variable | Hex | Usage |
|----------|-----|-------|
| `--color-background` | #FBF9F3 | Economist off-white (mandatory) |
| `--color-background-panel` | #F5F3EF | Slightly darker panels |
| `--color-background-callout` | #FFFFFF | White callout boxes |

### Semantic Colors

| Variable | Hex | Usage |
|----------|-----|-------|
| `--color-ground-truth` | #D4A03C | Gold - human baseline |
| `--color-foundation-model` | #3EBCD2 | Teal - foundation models |
| `--color-deep-learning` | #006BA2 | Deep blue - DL methods |
| `--color-traditional` | #999999 | Gray - traditional methods |
| `--color-ensemble` | #379A8B | Teal-green - ensembles |

### STRATOS Metric Colors

| Variable | Hex | Usage |
|----------|-----|-------|
| `--color-auroc` | #006BA2 | Discrimination |
| `--color-calibration` | #D4A03C | Calibration metrics |
| `--color-net-benefit` | #379A8B | Clinical utility |
| `--color-brier` | #C44536 | Overall performance |

## Figure Style

`figure_style.yaml` defines:

```yaml
figure_style:
  dpi: 300                    # Publication quality
  font_family: "Helvetica"    # Sans-serif
  font_size_title: 14
  font_size_label: 11
  font_size_tick: 9
  line_width: 1.5
  marker_size: 6
```

## Figure Registry

`figure_registry.yaml` catalogs all manuscript figures:

```yaml
figures:
  fig_M3:
    id: "fig_M3_factorial_matrix"
    title: "Factorial Experiment Design"
    type: "matrix"
    # ...
  fig_R7:
    id: "fig_R7_featurization_comparison"
    title: "Handcrafted vs Embeddings"
    type: "bar_comparison"
    # ...
```

## Plot Hyperparam Combos

`plot_hyperparam_combos.yaml` defines standard combinations for consistent plotting.

See `../combos/README.md` for detailed explanation.

### Standard 4 Combos

1. `ground_truth` - pupil-gt + pupil-gt + CatBoost
2. `best_ensemble` - Ensemble + CSDI + CatBoost
3. `best_single_fm` - MOMENT-gt-finetune + SAITS + CatBoost
4. `traditional` - LOF + SAITS + TabPFN

## Demo Subjects

`demo_subjects.yaml` defines 8 hand-picked subjects for visualization:

| ID | Class | Outlier % | Purpose |
|----|-------|-----------|---------|
| H001-H002 | Control | Low | Clean waveforms |
| H003-H004 | Control | High | Challenging cases |
| G001-G002 | Glaucoma | Low | Clear pathology |
| G003-G004 | Glaucoma | High | Challenging + pathology |

**Privacy Note**: Public IDs (Hxxx, Gxxx) map to private PLRxxxx codes via `data/private/subject_lookup.yaml`.

## Data Filters

`data_filters.yaml` defines query filters:

```yaml
filters:
  handcrafted_only:
    featurization: "simple"

  catboost_only:
    classifier: "CatBoost"

  ground_truth_only:
    outlier_method: "pupil-gt"
    imputation_method: "pupil-gt"
```

## Usage Patterns

### Python

```python
import yaml
from pathlib import Path

viz_config = Path("configs/VISUALIZATION")

# Load colors
with open(viz_config / "colors.yaml") as f:
    colors = yaml.safe_load(f)

gt_color = colors["semantic"]["--color-ground-truth"]
```

### R

```r
library(yaml)

colors <- yaml::read_yaml("configs/VISUALIZATION/colors.yaml")
gt_color <- colors$semantic$`--color-ground-truth`
```

### Hydra

Colors and styles are loaded as part of the main config:

```bash
python src/viz/generate_figures.py \
    VISUALIZATION=colors,figure_style
```

## Relationship to Code

| Config | Used By |
|--------|---------|
| `colors.yaml` | `src/viz/plot_config.py`, `src/r/theme_foundation_plr.R` |
| `figure_style.yaml` | All figure generation scripts |
| `plot_hyperparam_combos.yaml` | Comparison plots, CD diagrams |
| `demo_subjects.yaml` | Subject trace figures |

## Anti-Hardcoding Rule

**NEVER** hardcode colors or dimensions in visualization code:

```python
# WRONG
color = "#006BA2"

# CORRECT
from src.viz.plot_config import COLORS
color = COLORS["deep_learning"]
```

See `.claude/rules/` for enforcement.

## See Also

- Color definitions: Root `STYLE-GUIDE.md`
- Figure system: `src/r/figure_system/`
- Python viz: `src/viz/`
- Plot config: `src/viz/plot_config.py`

---

**Note**: All colors and styles are centralized here to ensure manuscript consistency.
