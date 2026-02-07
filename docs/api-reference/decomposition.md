# Decomposition Module

PLR waveform decomposition methods for extracting interpretable components.

## Overview

The decomposition module provides methods to decompose PLR signals into meaningful components (e.g., transient, sustained, PIPR). This is useful for understanding the physiological origins of classification features.

## Methods

| Method | Description |
|--------|-------------|
| Template Fitting | Fit canonical PLR template to signal |
| PCA | Principal Component Analysis |
| Rotated PCA | Varimax-rotated PCA for interpretability |
| Sparse PCA | L1-regularized PCA for sparse components |
| GED | Generalized Eigenvalue Decomposition |

## API Reference

::: src.decomposition.template_fitting
    options:
      show_source: true
      members:
        - fit_template
        - TemplateFitter

::: src.decomposition.pca_methods
    options:
      show_source: true
      members:
        - decompose_pca
        - decompose_rotated_pca
        - decompose_sparse_pca

::: src.decomposition.ged
    options:
      show_source: true
      members:
        - decompose_ged

::: src.decomposition.aggregation
    options:
      show_source: true
      members:
        - aggregate_decomposition_results
        - compute_group_statistics

## Usage Example

```python
from src.decomposition import decompose_pca, aggregate_decomposition_results

# Decompose a single signal
components = decompose_pca(plr_signal, n_components=3)

# Aggregate across subjects
group_stats = aggregate_decomposition_results(all_decompositions)
```
