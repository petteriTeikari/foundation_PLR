# Decomposition

PLR waveform decomposition into interpretable components.

## Overview

Provides 5 decomposition methods for pupillary light reflex signals, plus group-level aggregation with bootstrap confidence intervals. Used for understanding the physiological subcomponents (transient, sustained, PIPR) of PLR responses.

## Modules

| Module | Purpose |
|--------|---------|
| `template_fitting.py` | Physiologically-constrained basis functions (transient, sustained, PIPR) |
| `pca_methods.py` | Standard PCA, Rotated PCA (Promax), and Sparse PCA decomposition |
| `ged.py` | Generalized Eigendecomposition for contrast-maximizing components (Cohen 2022) |
| `aggregation.py` | Group-level aggregation across subjects with bootstrap CIs |

## See Also

- Kelbsch et al. 2019, Kawasaki et al. 2002 (template fitting references)
- Cohen MX 2022 NeuroImage (GED reference)
- Bustos 2024 (PCA reference)
