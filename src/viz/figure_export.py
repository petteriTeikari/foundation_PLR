"""
Figure export utilities for saving figures in multiple formats.

This module provides utilities for exporting matplotlib figures in
PNG, SVG, and EPS formats simultaneously.

NOTE: Prefer using save_figure() from plot_config.py directly.
This module exists for backward compatibility.
"""

from pathlib import Path
from typing import List, Optional

from loguru import logger

from src.viz.plot_config import save_figure


def save_figure_all_formats(
    fig,
    output_path: str,
    dpi: int = 300,
    formats: Optional[List[str]] = None,
) -> str:
    """
    Save a matplotlib figure in multiple formats.

    NOTE: This is a backward-compatibility wrapper around save_figure().
    Prefer using save_figure() from plot_config.py directly.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    output_path : str
        Base output path (with or without extension). Extension will be replaced.
    dpi : int
        Resolution for PNG (default 300 for A4 print quality)
    formats : list, optional
        Formats to save (default: loads from config)

    Returns
    -------
    str
        Base filename (without extension) for logging
    """
    path = Path(output_path)
    base_name = path.stem
    parent = path.parent if path.parent != Path(".") else None

    # Delegate to save_figure from plot_config
    save_figure(fig, base_name, formats=formats, output_dir=parent)

    logger.info(f"Saved: {base_name} (via save_figure)")
    return base_name
