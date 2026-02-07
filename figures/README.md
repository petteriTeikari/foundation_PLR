# Figures (`figures/`)

This directory contains generated figures for the manuscript and supplementary materials.

## Quick Start

```bash
# Generate all figures
python src/viz/generate_all_figures.py

# Generate specific figure
python src/viz/generate_all_figures.py --figure R7

# List available figures
python src/viz/generate_all_figures.py --list
```

## Directory Structure

```
figures/
├── generated/                  # Output directory
│   ├── *.png, *.pdf, *.svg    # Figure files
│   ├── data/                   # JSON data for reproducibility
│   │   └── *.json
│   ├── supplementary/          # Supplementary figures
│   ├── private/                # Private figures (git-ignored)
│   └── outdated-stale/         # Archived figures
│
└── README.md
```

## Figure Naming Convention

| Prefix | Category | Example |
|--------|----------|---------|
| `fig_M*` | Methods figures | `fig_M3_factorial_matrix.png` |
| `fig_R*` | Results figures | `fig_R7_featurization_comparison.png` |
| `fig_C*` | Supplementary | `fig_C3_utility_matrix.png` |
| `cd_*` | CD diagrams | `cd_preprocessing_comparison.png` |

## Main Figures

| ID | File | Description |
|----|------|-------------|
| M3 | `fig_M3_factorial_matrix.*` | Factorial experiment design |
| R7 | `fig_R7_featurization_comparison.*` | Handcrafted vs embeddings |
| R8 | `fig_R8_foundation_model_dashboard.*` | FM performance by task |
| C3 | `fig_C3_utility_matrix.*` | Utility assessment matrix |

## STRATOS-Compliant Figures

| Figure | STRATOS Metric |
|--------|----------------|
| `fig_retained_multi_metric.*` | Discrimination (AUROC, Brier, NB) |
| `fig_calibration_smoothed.*` | Calibration (smoothed curves) |
| `fig_dca_curves.*` | Clinical utility (DCA) |
| `fig_prob_dist_*.*` | Probability distributions |

## JSON Data for Reproducibility

Every figure has accompanying JSON data in `generated/data/`:

```
generated/
├── fig_R7_featurization_comparison.png
└── data/
    └── fig_R7_featurization_comparison.json
```

### JSON Structure

```json
{
  "figure_id": "fig_R7_featurization_comparison",
  "generated_at": "2024-01-22T10:30:00",
  "data": {
    "handcrafted_auroc": 0.830,
    "embedding_auroc": 0.740,
    "methods": ["MOMENT", "TimesNet", "UniTS"],
    "values": [...]
  },
  "stratos_metrics": {
    "auroc": 0.913,
    "auroc_ci_lo": 0.851,
    "auroc_ci_hi": 0.955
  }
}
```

## Privacy

Some figures contain patient-level data and are **excluded from git**:

| Pattern | Privacy |
|---------|---------|
| `fig_subject_traces_*.json` | Private (patient data) |
| `individual_*.json` | Private (per-subject) |
| `private/*.json` | Private (all) |

These patterns are in `.gitignore`.

## Format Outputs

Each figure is saved in multiple formats:

| Format | Use Case |
|--------|----------|
| PNG | Quick preview, presentations |
| PDF | Manuscript submission |
| SVG | Vector editing, web |

## Generation Pipeline

```
configs/VISUALIZATION/figure_registry.yaml → src/viz/*.py → figures/generated/
           ↓                        ↓
    Figure specs           plot_config.py styles
```

## Regenerating Figures

```bash
# Regenerate all
python src/viz/generate_all_figures.py

# Force regeneration (even if unchanged)
python src/viz/generate_all_figures.py --force

# Specific figure
python src/viz/generate_all_figures.py --figure R8
```

## See Also

- [src/viz/README.md](../src/viz/README.md) - Visualization source code
- [configs/VISUALIZATION/figure_registry.yaml](../configs/VISUALIZATION/figure_registry.yaml) - Figure specifications
