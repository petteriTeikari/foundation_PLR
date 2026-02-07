"""PLR Waveform Decomposition Module.

This module provides 5 decomposition methods for PLR signals:
1. Template Fitting - Physiological basis functions (transient, sustained, PIPR)
2. Standard PCA - Principal Component Analysis
3. Rotated PCA (Promax) - Oblique rotation allowing correlated components
4. GED - Generalized Eigendecomposition (Cohen 2022)
5. Sparse PCA - PCA with L1 penalty for interpretable sparse loadings

Plus aggregation utilities for group-level analysis with bootstrap CIs.
"""

from .aggregation import (
    ComponentTimecourse,
    DecompositionAggregator,
    DecompositionResult,
)
from .ged import GEDDecomposition, GEDResult
from .pca_methods import (
    PCAResult,
    RotatedPCA,
    RotatedPCAResult,
    SparsePCADecomposition,
    SparsePCAResult,
    StandardPCA,
)
from .template_fitting import TemplateFitting, TemplateResult

__all__ = [
    # Decomposition methods
    "TemplateFitting",
    "StandardPCA",
    "RotatedPCA",
    "SparsePCADecomposition",
    "GEDDecomposition",
    # Result dataclasses
    "TemplateResult",
    "PCAResult",
    "RotatedPCAResult",
    "SparsePCAResult",
    "GEDResult",
    # Aggregation
    "DecompositionAggregator",
    "DecompositionResult",
    "ComponentTimecourse",
]
