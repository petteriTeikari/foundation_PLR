"""
Figure Dimension Definitions - Centralized Configuration

This module provides standardized figure dimensions loaded from
configs/VISUALIZATION/figure_registry.yaml.

Usage:
    from src.viz.figure_dimensions import get_dimensions, DIMENSIONS

    # Use preset
    fig, ax = plt.subplots(figsize=get_dimensions("single"))

    # Get specific figure dimensions from registry
    fig, ax = plt.subplots(figsize=get_figure_dimensions("fig_calibration_smoothed"))

CRITICAL: NEVER hardcode figsize tuples like (8, 6). Always use this module.
See: .claude/CLAUDE.md - ANTI-HARDCODING ENFORCEMENT
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

__all__ = [
    "DIMENSIONS",
    "get_dimensions",
    "get_figure_dimensions",
    "get_r_figure_dimensions",
]

# Project root for config file access
PROJECT_ROOT = Path(__file__).parent.parent.parent
REGISTRY_PATH = PROJECT_ROOT / "configs" / "VISUALIZATION" / "figure_registry.yaml"

# ==============================================================================
# Standard Dimension Presets
# ==============================================================================
# These presets match common figure sizes used throughout the codebase.
# When refactoring hardcoded figsizes, map to the appropriate preset.

DIMENSIONS = {
    # Single-column figures (typical for most plots)
    "single": (8, 6),  # Standard single figure
    "single_narrow": (6, 5),  # Narrow single (bar charts)
    "single_wide": (10, 6),  # Slightly wider single
    "single_tall": (8, 8),  # Square-ish single
    # Double-column figures (wide, for multi-panel or comparisons)
    "double": (14, 6),  # Standard double-width
    "double_short": (14, 5),  # Double width, shorter height
    "double_tall": (14, 10),  # Double width, taller
    # Wide figures (12-inch width)
    "wide": (12, 5),  # Wide format
    "wide_tall": (12, 8),  # Wide and taller
    # Square figures
    "square": (8, 8),  # Square
    "square_small": (6, 6),  # Small square (instability plots)
    "square_large": (10, 10),  # Large square
    # Tall figures (for vertical layouts like forest plots)
    "tall": (8, 12),  # Tall single column
    "tall_narrow": (6, 10),  # Narrow and tall
    # Specialized presets
    "matrix": (10, 8),  # For matrix/heatmap visualizations
    "dashboard": (14, 6),  # For multi-panel dashboards
    "cd_diagram": (12, 8),  # For critical difference diagrams
    "calibration": (8, 8),  # For calibration plots (typically square)
    "calibration_small": (7, 7),  # Smaller calibration plot
    "dca": (10, 7),  # For decision curve analysis
    "forest": (10, 12),  # For forest plots
    "raincloud": (12, 10),  # For raincloud plots
    "specification_curve": (16, 12),  # For specification curve analysis (large)
    # Supplementary figure presets
    "supplementary_single": (8, 6),
    "supplementary_wide": (12, 8),
    "supplementary_tall": (10, 14),
}


def get_dimensions(preset: str = "single") -> tuple[float, float]:
    """
    Get figure dimensions by preset name.

    Parameters
    ----------
    preset : str
        Preset name from DIMENSIONS dict. Default is "single".
        Available presets:
        - "single", "single_wide", "single_tall"
        - "double", "double_short", "double_tall"
        - "wide", "wide_tall"
        - "square", "square_large"
        - "tall", "tall_narrow"
        - "matrix", "dashboard", "cd_diagram"
        - "calibration", "dca", "forest", "raincloud"
        - "specification_curve"
        - "supplementary_single", "supplementary_wide", "supplementary_tall"

    Returns
    -------
    tuple[float, float]
        (width, height) in inches

    Examples
    --------
    >>> fig, ax = plt.subplots(figsize=get_dimensions("single"))
    >>> fig, axes = plt.subplots(1, 3, figsize=get_dimensions("double"))
    """
    if preset not in DIMENSIONS:
        import warnings

        warnings.warn(
            f"Unknown preset '{preset}', using 'single'. "
            f"Available presets: {list(DIMENSIONS.keys())}"
        )
        return DIMENSIONS["single"]
    return DIMENSIONS[preset]


@lru_cache(maxsize=1)
def _load_figure_registry() -> dict:
    """Load and cache the figure registry YAML."""
    if not REGISTRY_PATH.exists():
        import warnings

        warnings.warn(f"Figure registry not found at {REGISTRY_PATH}")
        return {}

    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def get_figure_dimensions(figure_name: str) -> Optional[tuple[float, float]]:
    """
    Get dimensions for a specific figure from the registry.

    This looks up the figure in figure_registry.yaml and returns
    the configured dimensions.

    Parameters
    ----------
    figure_name : str
        Figure name as defined in figure_registry.yaml
        (e.g., "fig_calibration_smoothed", "fig_R7_featurization_comparison")

    Returns
    -------
    tuple[float, float] or None
        (width, height) in inches, or None if figure not found

    Examples
    --------
    >>> dims = get_figure_dimensions("fig_calibration_smoothed")
    >>> if dims:
    ...     fig, ax = plt.subplots(figsize=dims)
    """
    registry = _load_figure_registry()

    # Search in main_figures
    main_figs = registry.get("main_figures", {})
    if figure_name in main_figs:
        styling = main_figs[figure_name].get("styling", {})
        width = styling.get("width")
        height = styling.get("height")
        aspect = styling.get("aspect")

        # If explicit width/height, use them
        if width is not None and height is not None:
            return (float(width), float(height))

        # If width preset + aspect ratio
        if isinstance(width, str) and aspect is not None:
            preset_dims = get_dimensions(width)
            return (preset_dims[0], preset_dims[0] * aspect)

        # Just a width preset
        if isinstance(width, str):
            return get_dimensions(width)

    # Search in supplementary_figures
    supp_figs = registry.get("supplementary_figures", {})
    if figure_name in supp_figs:
        styling = supp_figs[figure_name].get("styling", {})
        width = styling.get("width")
        height = styling.get("height")

        if width is not None and height is not None:
            return (float(width), float(height))

    return None


def get_r_figure_dimensions(figure_name: str) -> Optional[tuple[float, float]]:
    """
    Get dimensions for an R-generated figure from the registry.

    R figures have their dimensions defined in the r_figures section
    of figure_registry.yaml.

    Parameters
    ----------
    figure_name : str
        Figure name as defined in figure_registry.yaml r_figures section
        (e.g., "fig_calibration_dca_combined", "fig_instability_combined")

    Returns
    -------
    tuple[float, float] or None
        (width, height) in inches, or None if figure not found

    Examples
    --------
    >>> dims = get_r_figure_dimensions("fig_instability_combined")
    >>> # In R: ggsave(filename, width=dims[1], height=dims[2])
    """
    registry = _load_figure_registry()

    r_figs = registry.get("r_figures", {})
    if figure_name in r_figs:
        styling = r_figs[figure_name].get("styling", {})
        width = styling.get("width")
        height = styling.get("height")

        if width is not None and height is not None:
            return (float(width), float(height))

    return None


def map_hardcoded_to_preset(width: float, height: float) -> str:
    """
    Map a hardcoded figsize to the closest preset.

    Utility function for refactoring - helps identify which preset
    to use when replacing hardcoded dimensions.

    Parameters
    ----------
    width : float
        Figure width in inches
    height : float
        Figure height in inches

    Returns
    -------
    str
        Name of the closest matching preset

    Examples
    --------
    >>> map_hardcoded_to_preset(8, 6)
    'single'
    >>> map_hardcoded_to_preset(14, 6)
    'double'
    """
    target = (width, height)

    # Find closest match by Euclidean distance
    best_preset = "single"
    best_distance = float("inf")

    for preset, dims in DIMENSIONS.items():
        distance = ((dims[0] - target[0]) ** 2 + (dims[1] - target[1]) ** 2) ** 0.5
        if distance < best_distance:
            best_distance = distance
            best_preset = preset

    return best_preset


if __name__ == "__main__":
    # Demo and validation
    print("Available dimension presets:")
    print("=" * 50)
    for name, dims in DIMENSIONS.items():
        print(f"  {name:25s} -> {dims}")

    print("\nRegistry-defined figure dimensions:")
    print("=" * 50)
    registry = _load_figure_registry()

    for section in ["main_figures", "supplementary_figures", "r_figures"]:
        figs = registry.get(section, {})
        if figs:
            print(f"\n{section}:")
            for name in list(figs.keys())[:5]:  # Show first 5
                if section == "r_figures":
                    dims = get_r_figure_dimensions(name)
                else:
                    dims = get_figure_dimensions(name)
                print(f"  {name}: {dims}")
