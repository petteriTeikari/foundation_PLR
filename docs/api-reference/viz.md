# Visualization Module

Publication-quality figure generation for Foundation PLR.

## Overview

The `viz` module provides Python-based visualization functions for generating figures. For R-based figures, see `src/r/`.

!!! note "Computation vs Visualization"
    The viz module **only visualizes** data from DuckDB. All metric computation happens during extraction (see `scripts/extract_all_configs_to_duckdb.py`).

## Key Modules

| Module | Description |
|--------|-------------|
| `plot_config` | Style setup, colors, save functions |
| `calibration_plot` | STRATOS calibration curves |
| `dca_plot` | Decision curve analysis |
| `cd_diagram` | Critical difference diagrams |
| `factorial_matrix` | Factorial design heatmaps |
| `featurization_comparison` | Handcrafted vs embeddings |

## API Reference

::: src.viz.plot_config
    options:
      show_source: true
      members:
        - setup_style
        - save_figure
        - COLORS
        - get_combo_color

::: src.viz.calibration_plot
    options:
      show_source: true

::: src.viz.dca_plot
    options:
      show_source: true

::: src.viz.cd_diagram
    options:
      show_source: true

::: src.viz.factorial_matrix
    options:
      show_source: true

## Usage Example

```python
from src.viz.plot_config import setup_style, save_figure, COLORS

# Always setup style first
setup_style()

# Create figure using semantic colors
fig, ax = plt.subplots()
ax.plot(x, y, color=COLORS["ground_truth"])

# Save with JSON data for reproducibility
save_figure(fig, "fig_my_analysis", data={"x": x, "y": y})
```

## Color System

Colors are loaded from `configs/VISUALIZATION/combos.yaml`:

```python
from src.viz.plot_config import COLORS

# Available colors
COLORS["ground_truth"]   # #2E5B8C
COLORS["best_ensemble"]  # #932834
COLORS["traditional"]    # #666666
```

## See Also

- [R Figure System](https://github.com/petteriTeikari/foundation_PLR/blob/main/src/r/README.md) - R/ggplot2 figures
- [Figure Registry](https://github.com/petteriTeikari/foundation_PLR/blob/main/configs/VISUALIZATION/figure_registry.yaml) - Figure specifications
