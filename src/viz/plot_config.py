"""
Plot Configuration for Foundation PLR Figures

Style: Neue Haas Grotesk / Helvetica Neue inspired typography
Clean, professional academic visualization aesthetic

Usage:
    from src.viz.plot_config import setup_style, save_figure, COLORS
    setup_style()  # Call before creating figures

# AIDEV-NOTE: This module provides styling and export functionality.
# All viz modules should call setup_style() before creating figures.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = [
    # Style setup
    "setup_style",
    "apply_style",
    "reset_style",
    # Figure utilities
    "save_figure",
    "get_figure_size",
    "get_output_dir",
    "style_axis",
    "add_reference_line",
    "add_benchmark_line",
    "annotate_difference",
    # Color access
    "get_combo_color",
    "get_method_display_name",
    "get_category_display_name",
    "get_category_short_name",
    "get_all_category_display_names",
    # Constants
    "COLORS",
    "COLOR_CYCLE",
    "FIXED_CLASSIFIER",
    "KEY_STATS",  # Deprecated: use get_key_stats() instead
    "get_key_stats",
    # Database
    "get_connection",
    # Config dicts (for advanced customization)
    "FONT_CONFIG",
    "AXES_CONFIG",
    "FIGURE_CONFIG",
    "LINE_CONFIG",
]

# Import config loader for method names and colors
# Uses configs/VISUALIZATION/ (consolidated from old config/ directory)
try:
    from src.viz.config_loader import ConfigurationError, get_config_loader
except ModuleNotFoundError:
    # When running from viz directory
    from config_loader import ConfigurationError, get_config_loader

# ==============================================================================
# Database and Output Paths
# ==============================================================================
# AIDEV-NOTE: Paths are configurable via environment variables:
# - FOUNDATION_PLR_DB_PATH: Path to results database
# - FOUNDATION_PLR_MANUSCRIPT_ROOT: Root of manuscript directory
# - FOUNDATION_PLR_FIGURES_DIR: Output directory for figures


def _find_database() -> Path:
    """Find the DuckDB database file.

    EXPLICIT PATH POLICY: No silent fallback chains!

    Resolution order:
    1. FOUNDATION_PLR_DB_PATH environment variable (explicit override)
    2. Canonical path: outputs/foundation_plr_results.db (project root)

    Raises:
        FileNotFoundError: With explicit instructions if database not found
    """
    import os

    # Check environment variable first (explicit override)
    env_path = os.environ.get("FOUNDATION_PLR_DB_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        raise FileNotFoundError(
            f"FOUNDATION_PLR_DB_PATH set to '{env_path}' but file does not exist.\n"
            "FIX: Check the path or unset the environment variable."
        )

    # Canonical path: data/public/foundation_plr_results.db
    # (Consolidated DB with all STRATOS metrics, curve tables, and predictions)
    project_root = Path(__file__).parent.parent.parent
    canonical_path = project_root / "data" / "public" / "foundation_plr_results.db"

    if canonical_path.exists():
        return canonical_path

    # Clear error with actionable instructions
    raise FileNotFoundError(
        f"Database not found at canonical path:\n"
        f"  {canonical_path}\n\n"
        "FIX: Run the extraction pipeline to create the database:\n"
        "  python scripts/extraction/extract_all_configs_to_duckdb.py\n\n"
        "Or set explicit path via environment variable:\n"
        "  export FOUNDATION_PLR_DB_PATH=/path/to/your/database.db"
    )


def get_connection() -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection to the results database."""
    db_path = _find_database()
    return duckdb.connect(str(db_path), read_only=True)


def _get_manuscript_root() -> Path:
    """Get manuscript root directory."""
    import os

    env_path = os.environ.get("FOUNDATION_PLR_MANUSCRIPT_ROOT")
    if env_path:
        return Path(env_path)

    # Default: relative to project
    project_root = Path(__file__).parent.parent.parent
    return project_root.parent / "manuscripts" / "foundationPLR"


def _get_figures_dir() -> Path:
    """Get output directory for generated figures."""
    import os

    env_path = os.environ.get("FOUNDATION_PLR_FIGURES_DIR")
    if env_path:
        return Path(env_path)

    # Default: within project
    project_root = Path(__file__).parent.parent.parent
    return project_root / "figures" / "generated"


# Lazy initialization - don't create dirs on import
MANUSCRIPT_ROOT = _get_manuscript_root()
FIGURES_DIR = _get_figures_dir()


# ==============================================================================
# Research Design Constants
# ==============================================================================
# Research question: fix classifier, vary preprocessing.
# All standard combos use this classifier (see plot_hyperparam_combos.yaml).
FIXED_CLASSIFIER = "CatBoost"

# ==============================================================================
# Color Palette
# ==============================================================================
# AIDEV-NOTE: This is the canonical source for semantic/general colors in Python viz code.
# All viz scripts MUST use COLORS dict or get_combo_color(), never hardcode hex values.
# For combo-specific colors, use get_combo_color() which resolves via combos YAML color_ref.
# AIDEV-IMMUTABLE: Color palette uses Paul Tol colorblind-safe colors - do not change
# without accessibility review.

COLORS = {
    # Primary palette
    "primary": "#2E5090",  # Navy blue
    "secondary": "#D64045",  # Coral red
    "tertiary": "#45B29D",  # Teal
    "quaternary": "#F5A623",  # Amber
    "quinary": "#7B68EE",  # Medium purple
    # Semantic colors
    "positive": "#45B29D",  # Teal (good results)
    "negative": "#D64045",  # Coral (bad results)
    "neutral": "#4A4A4A",  # Dark gray
    "reference": "#D64045",  # Reference lines
    # Background colors
    "background": "#FAFAFA",
    "grid": "#E0E0E0",
    # Foundation model specific
    "moment": "#2E5090",
    "units": "#45B29D",
    "traditional": "#7B68EE",
    "ensemble": "#F5A623",
    "ground_truth": "#666666",  # Matches combos YAML --color-ground-truth
    "foundation_model": "#0072B2",  # Blue (Paul Tol)
    "deep_learning": "#009E73",  # Teal (Paul Tol)
    # Featurization comparison
    "handcrafted": "#2E5090",  # Navy blue
    "embeddings": "#D64045",  # Coral red
    # Classifier specific
    "catboost": "#2E5090",
    "tabpfn": "#45B29D",
    "xgboost": "#D64045",
    "logreg": "#7B68EE",
    # Semantic for utility matrix
    "good": "#45B29D",  # Teal (positive)
    "bad": "#D64045",  # Coral (negative)
    # Accent colors
    "accent": "#F5A623",  # Amber (highlight)
    "highlight": "#F5A623",
    # Outcome class colors (for probability distributions, etc.)
    "glaucoma": "#E74C3C",  # Glaucoma outcome class
    "control": "#3498DB",  # Control outcome class
    # Light protocol colors (for stimulus visualization)
    "blue_stimulus": "#1f77b4",  # Blue light stimulus
    "red_stimulus": "#d62728",  # Red light stimulus
    "background_light": "#f0f0f0",  # Light background zones
    "blue_zone": "#cce5ff",  # Blue response zone
    "red_zone": "#ffcccc",  # Red response zone
    # CD diagram rank colors
    "cd_rank1": "#2ecc71",  # CD diagram rank 1 (best)
    "cd_rank2": "#3498db",  # CD diagram rank 2
    "cd_rank3": "#e74c3c",  # CD diagram rank 3
    "cd_rank4": "#9b59b6",  # CD diagram rank 4
    "cd_rank5": "#f39c12",  # CD diagram rank 5 (worst)
    # Text colors
    "text_primary": "#333333",  # Primary text
    "text_secondary": "#666666",  # Secondary text, grid lines
    # Background colors (extended)
    "background_neutral": "#F5F5F5",  # Neutral background
    "grid_lines": "#CCCCCC",  # Grid lines (distinct from background grid)
    # Decomposition component colors (Paul Tol colorblind-safe)
    "decomp_component_1": "#E69F00",  # Orange - phasic/PC1
    "decomp_component_2": "#56B4E9",  # Sky blue - sustained/PC2
    "decomp_component_3": "#009E73",  # Bluish green - pipr/PC3
    "decomp_mean_waveform": "#888888",  # Gray - mean waveform
    # Extended color cycle (beyond primary 5)
    "cycle_brown": "#8B4513",  # Saddle brown
    "cycle_seagreen": "#20B2AA",  # Light sea green
}


# ==============================================================================
# Key Statistics (loaded from config - NEVER hardcode!)
# ==============================================================================


def _load_key_stats() -> Dict[str, float]:
    """Load key statistics from config files.

    Sources:
    - Pipeline AUROCs: configs/VISUALIZATION/plot_hyperparam_combos.yaml
    - Featurization: outputs/r_data/featurization_comparison.json
    - Benchmark: configs/defaults.yaml

    Raises:
        ConfigurationError: If required config files are missing
    """
    import json

    import yaml

    stats = {}
    project_root = Path(__file__).parent.parent.parent

    # Load pipeline AUROCs from combos.yaml
    combos_path = (
        project_root / "configs" / "VISUALIZATION" / "plot_hyperparam_combos.yaml"
    )
    if combos_path.exists():
        with open(combos_path) as f:
            combos = yaml.safe_load(f)

        # Extract AUROC values from standard combos
        for combo in combos.get("standard_combos", []):
            combo_id = combo.get("id")
            auroc = combo.get("auroc")
            if combo_id and auroc:
                stats[f"{combo_id}_auroc"] = auroc

    # Load featurization comparison from JSON (if exists)
    feat_json = project_root / "data" / "r_data" / "featurization_comparison.json"
    if feat_json.exists():
        with open(feat_json) as f:
            feat_data = json.load(f)
        for method in feat_data.get("methods", []):
            if method["id"] == "simple1.0":
                stats["handcrafted_mean_auroc"] = method["auroc"]
            elif "embedding" in method["id"].lower():
                stats["embeddings_mean_auroc"] = method["auroc"]
        if "handcrafted_mean_auroc" in stats and "embeddings_mean_auroc" in stats:
            stats["featurization_gap"] = round(
                stats["handcrafted_mean_auroc"] - stats["embeddings_mean_auroc"], 3
            )

    # Benchmark AUROC from defaults.yaml
    defaults_path = project_root / "configs" / "defaults.yaml"
    if defaults_path.exists():
        with open(defaults_path) as f:
            defaults = yaml.safe_load(f)
        benchmark = defaults.get("CLS_EVALUATION", {}).get("benchmark_auroc")
        if benchmark:
            stats["benchmark_auroc"] = benchmark

    # Fallback for benchmark if not in config
    if "benchmark_auroc" not in stats:
        stats["benchmark_auroc"] = 0.93  # Najjar 2021

    return stats


# Lazy loading for KEY_STATS
_key_stats_cache: Optional[Dict[str, float]] = None


def get_key_stats() -> Dict[str, float]:
    """Get key statistics dictionary (lazily loaded from config)."""
    global _key_stats_cache
    if _key_stats_cache is None:
        try:
            _key_stats_cache = _load_key_stats()
        except Exception as e:
            # Fallback for backward compatibility - log warning
            import warnings

            warnings.warn(
                f"Failed to load KEY_STATS from config: {e}. Using empty dict."
            )
            _key_stats_cache = {}
    return _key_stats_cache


# Backward compatibility alias (deprecated - use get_key_stats())
class _LazyKeyStats(dict):
    """Lazy-loading wrapper for KEY_STATS backward compatibility."""

    _loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.update(get_key_stats())
            self._loaded = True

    def __getitem__(self, key: str) -> float:
        self._ensure_loaded()
        return super().__getitem__(key)

    def get(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """Retrieve a key stat value, loading from config on first access.

        Overrides ``dict.get`` to ensure the backing store is populated
        from ``get_key_stats()`` before lookup, providing transparent
        lazy-loading for backward-compatible ``KEY_STATS`` usage.

        Parameters
        ----------
        key : str
            Stat name to look up (e.g. ``"ground_truth_auroc"``).
        default : float or None, optional
            Value returned when *key* is absent. Default is ``None``.

        Returns
        -------
        float or None
            The stat value, or *default* if not found.
        """
        self._ensure_loaded()
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        self._ensure_loaded()
        return super().__contains__(key)


KEY_STATS = _LazyKeyStats()

# Qualitative color cycle for plots
COLOR_CYCLE = [
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["tertiary"],
    COLORS["quaternary"],
    COLORS["quinary"],
    COLORS["cycle_brown"],
    COLORS["cycle_seagreen"],
]


# ==============================================================================
# Typography Configuration
# ==============================================================================

FONT_CONFIG = {
    # Font family (Helvetica Neue fallback chain)
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Helvetica Neue",
        "Helvetica",
        "Arial",
        "Liberation Sans",
        "DejaVu Sans",
        "sans-serif",
    ],
    # Base font size
    "font.size": 10,
    "font.weight": "normal",
    # Title styling (bold, larger)
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 12,
    # Axis labels (regular weight)
    "axes.labelsize": 11,
    "axes.labelweight": "normal",
    "axes.labelpad": 8,
    # Tick labels (smaller)
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    # Legend
    "legend.fontsize": 9,
    "legend.title_fontsize": 10,
    "legend.frameon": False,
    "legend.loc": "best",
    # Math text
    "mathtext.fontset": "dejavusans",
}


# ==============================================================================
# Axes and Grid Configuration
# ==============================================================================
# AIDEV-NOTE: Economist/ggplot2-inspired styling
# Reference: https://altaf-ali.github.io/ggplot_tutorial/challenge.html
# Key features:
# - Light gray background (#F0F0F0)
# - White horizontal grid lines only (no vertical)
# - Minimal spines (left/bottom only, light color)
# - Clean sans-serif typography

AXES_CONFIG = {
    # Background - Economist-style light gray
    "axes.facecolor": "#F0F0F0",  # Light gray (Economist style)
    "figure.facecolor": "white",
    # Spines - minimal, subtle
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.linewidth": 0.5,  # Thinner spines
    "axes.edgecolor": "#AAAAAA",  # Lighter gray spines
    # Grid - white horizontal lines only (Economist style)
    "axes.grid": True,
    "axes.grid.axis": "y",  # Only y-axis grid
    "grid.alpha": 1.0,  # Solid white lines
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "grid.color": "white",  # White grid on gray background
    # Axis formatting
    "axes.formatter.useoffset": False,
    "axes.formatter.use_mathtext": True,
    # Color cycle
    "axes.prop_cycle": plt.cycler(color=COLOR_CYCLE),
}


# ==============================================================================
# Figure and Export Configuration
# ==============================================================================

FIGURE_CONFIG = {
    # Figure size (default for single column)
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    # Tight layout
    "figure.autolayout": False,
    "figure.constrained_layout.use": True,
    # Saving
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
    "savefig.format": "png",
    # PDF/EPS backend
    "pdf.fonttype": 42,  # TrueType fonts in PDF
    "ps.fonttype": 42,
}


# ==============================================================================
# Line and Marker Configuration
# ==============================================================================

LINE_CONFIG = {
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "lines.markeredgewidth": 0.5,
    # Error bars
    "errorbar.capsize": 3,
}


# ==============================================================================
# Style Application Functions
# ==============================================================================


def apply_style(context: str = "paper") -> None:
    """
    Apply the Neue Haas Grotesk inspired style to matplotlib.

    Parameters
    ----------
    context : str
        'paper' - For publication figures (larger fonts)
        'talk' - For presentations (even larger fonts)
        'poster' - For posters (largest fonts)
    """
    # Merge all configurations
    style_dict = {}
    style_dict.update(FONT_CONFIG)
    style_dict.update(AXES_CONFIG)
    style_dict.update(FIGURE_CONFIG)
    style_dict.update(LINE_CONFIG)

    # Context-specific scaling
    scale_factors = {
        "paper": 1.0,
        "talk": 1.3,
        "poster": 1.6,
    }
    scale = scale_factors.get(context, 1.0)

    # Scale font sizes
    font_keys = [
        "font.size",
        "axes.titlesize",
        "axes.labelsize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
        "legend.title_fontsize",
    ]
    for key in font_keys:
        if key in style_dict:
            style_dict[key] = int(style_dict[key] * scale)

    # Apply style
    plt.rcParams.update(style_dict)


def reset_style() -> None:
    """Reset matplotlib to default style."""
    mpl.rcParams.update(mpl.rcParamsDefault)


def get_figure_size(
    width: str = "single", aspect: float = 0.618
) -> Tuple[float, float]:
    """
    Get figure size for different publication contexts.

    Parameters
    ----------
    width : str
        'single' - Single column width (~3.5 inches)
        'double' - Double column width (~7 inches)
        'full' - Full page width (~10 inches)
    aspect : float
        Height/width ratio (default: golden ratio)

    Returns
    -------
    tuple
        (width, height) in inches
    """
    widths = {
        "single": 3.5,
        "double": 7.0,
        "full": 10.0,
    }
    w = widths.get(width, 7.0)
    h = w * aspect
    return (w, h)


def style_axis(
    ax: plt.Axes,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    grid: str = "y",
) -> None:
    """
    Apply consistent styling to an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to style
    title : str, optional
        Axis title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    grid : str
        'x', 'y', 'both', or 'none'
    """
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Grid configuration
    if grid == "none":
        ax.grid(False)
    elif grid == "x":
        ax.grid(True, axis="x", alpha=0.3)
        ax.grid(False, axis="y")
    elif grid == "y":
        ax.grid(True, axis="y", alpha=0.3)
        ax.grid(False, axis="x")
    else:  # 'both'
        ax.grid(True, alpha=0.3)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_reference_line(
    ax: plt.Axes,
    value: float,
    label: str = "Reference",
    color: Optional[str] = None,
    linestyle: str = "--",
) -> None:
    """Add a horizontal reference line to a plot."""
    if color is None:
        color = COLORS["reference"]
    ax.axhline(
        y=value,
        color=color,
        linestyle=linestyle,
        linewidth=1.5,
        alpha=0.7,
        label=label,
        zorder=1,
    )


def annotate_difference(
    ax: plt.Axes, x: float, y1: float, y2: float, text: str, color: Optional[str] = None
) -> None:
    """Add annotation showing difference between two points."""
    if color is None:
        color = COLORS["neutral"]

    # Draw bracket
    ax.annotate(
        "",
        xy=(x, y1),
        xytext=(x, y2),
        arrowprops=dict(arrowstyle="<->", color=color, lw=1),
    )

    # Add text
    mid_y = (y1 + y2) / 2
    ax.annotate(
        text,
        xy=(x, mid_y),
        xytext=(x + 0.1, mid_y),
        fontsize=9,
        color=color,
        va="center",
    )


# ==============================================================================
# Auto-apply on import (optional)
# ==============================================================================

# Uncomment to auto-apply style on import:
# apply_style()

# ==============================================================================
# Alias for backward compatibility
# ==============================================================================

setup_style = apply_style


# ==============================================================================
# Figure Export Functions
# ==============================================================================


def get_output_dir() -> Path:
    """Get the output directory for figures.

    Creates directory on first call (lazy initialization).
    """
    figures_dir = _get_figures_dir()
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def save_figure(
    fig: plt.Figure,
    name: str,
    data: Optional[Dict[str, Any]] = None,
    formats: List[str] = None,
    output_dir: Optional[Path] = None,
    synthetic: Optional[bool] = None,
) -> Path:
    """
    Save figure to multiple formats and optionally save accompanying JSON data.

    Part of the 4-gate isolation architecture. Synthetic figures are automatically
    routed to figures/synthetic/ when synthetic=True or when is_synthetic_mode().

    Parameters
    ----------
    fig : matplotlib.Figure
        The figure to save
    name : str
        Base name for the output file (without extension)
    data : dict, optional
        Data dictionary to save as JSON for reproducibility.
        If synthetic, adds _synthetic_warning=True to the data.
    formats : list, optional
        Output formats. Default loads from config or uses ['png', 'svg'].
        SVG preferred over PDF for vector graphics (infinite scalability).
    output_dir : Path, optional
        Output directory (default: figures/generated/ or figures/synthetic/).
        Auto-detected from data mode if not specified.
    synthetic : bool, optional
        If True, route to synthetic directory. If None, auto-detect from
        is_synthetic_mode() environment variable.

    Returns
    -------
    Path
        Path to the primary output file (PNG)
    """
    # Import data mode utilities for synthetic detection
    from src.utils.data_mode import (
        get_figures_dir_for_mode,
        is_synthetic_mode,
    )

    if formats is None:
        # Load from figure_layouts.yaml - SINGLE SOURCE OF TRUTH
        try:
            import yaml

            project_root = Path(__file__).parent.parent.parent
            layouts_path = (
                project_root / "configs" / "VISUALIZATION" / "figure_layouts.yaml"
            )
            with open(layouts_path) as f:
                layouts_config = yaml.safe_load(f)
            formats = layouts_config.get("output_settings", {}).get("formats", ["png"])
        except Exception:
            formats = ["png"]  # Default: PNG only

    # Auto-detect synthetic mode if not explicitly specified
    if synthetic is None:
        synthetic = is_synthetic_mode()

    # Route to appropriate output directory
    if output_dir is None:
        output_dir = get_figures_dir_for_mode(synthetic=synthetic)
    elif synthetic and "synthetic" not in str(output_dir).lower():
        # User specified a directory but we're in synthetic mode - warn
        import warnings

        warnings.warn(
            f"Synthetic mode detected but output_dir={output_dir} doesn't contain 'synthetic'. "
            "This may cause production contamination. Consider using synthetic=False or "
            "passing a synthetic output directory."
        )

    # Ensure output_dir is a Path (callers may pass str)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save figure in all formats
    primary_path = None
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        if fmt == "png":
            primary_path = path
        print(f"  Saved: {path}")

    # Save JSON data if provided
    if data is not None:
        # Add synthetic warning metadata if in synthetic mode
        if synthetic:
            data = dict(data)  # Make a copy
            data["_synthetic_warning"] = True
            data["_data_source"] = "synthetic"
            data["_do_not_publish"] = True

        data_dir = output_dir / "data"
        data_dir.mkdir(exist_ok=True)
        json_path = data_dir / f"{name}.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Saved: {json_path}")

    return primary_path or output_dir / f"{name}.{formats[0]}"


def add_benchmark_line(
    ax: plt.Axes,
    value: float,
    label: str,
    color: Optional[str] = None,
    linestyle: str = "--",
    linewidth: float = 1.5,
    alpha: float = 0.7,
) -> None:
    """
    Add a horizontal benchmark/reference line to a plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add line to
    value : float
        Y-value for the horizontal line
    label : str
        Label for the line (for legend)
    color : str, optional
        Line color (default: reference color from COLORS)
    linestyle : str
        Line style (default: dashed)
    linewidth : float
        Line width
    alpha : float
        Line transparency
    """
    if color is None:
        color = COLORS["reference"]

    ax.axhline(
        y=value,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        zorder=1,
    )


# ==============================================================================
# Config-Driven Color Access (for migration)
# ==============================================================================


def get_combo_color(combo_id: str) -> str:
    """
    Get the color for a specific combo from config.

    Resolves via plot_hyperparam_combos.yaml: combo.color_ref → color_definitions.

    Parameters
    ----------
    combo_id : str
        Combo identifier (e.g., 'ground_truth', 'best_single_fm')

    Returns
    -------
    str
        Hex color string
    """
    try:
        loader = get_config_loader()
        combos_config = loader.get_combos()
        color_defs = combos_config.get("color_definitions", {})

        # Look up color_ref from combo config
        for combo_type in ["standard_combos", "extended_combos"]:
            for combo in combos_config.get(combo_type, []):
                if combo.get("id") == combo_id:
                    color_ref = combo.get("color_ref")
                    if color_ref and color_ref in color_defs:
                        return color_defs[color_ref]

        # Fallback to COLORS dict (e.g., for non-combo color lookups)
        return COLORS.get(combo_id, COLORS["neutral"])
    except (ConfigurationError, Exception):
        return COLORS.get(combo_id, COLORS["neutral"])


def get_method_display_name(method_name: str, method_type: str) -> str:
    """
    Get the display name for a method from config.

    Parameters
    ----------
    method_name : str
        Method identifier (e.g., 'MOMENT-gt-finetune')
    method_type : str
        Method type ('outlier_detection', 'imputation', 'classifiers')

    Returns
    -------
    str
        Human-readable display name
    """
    try:
        loader = get_config_loader()
        methods = loader.get_methods_config()
        type_methods = methods.get(method_type, {})
        method_info = type_methods.get(method_name, {})
        return method_info.get("display_name", method_name)
    except ConfigurationError:
        return method_name


def get_category_display_name(category_id: str) -> str:
    """
    Get the display name for a category from config.

    Parameters
    ----------
    category_id : str
        Category identifier (e.g., 'ground_truth', 'foundation_model')

    Returns
    -------
    str
        Human-readable display name (e.g., 'Ground Truth', 'Foundation Model')
    """
    try:
        loader = get_config_loader()
        names = loader.get_category_display_names()
        return names.get(category_id, category_id)
    except ConfigurationError:
        # Fallback for backward compatibility
        fallback = {
            "ground_truth": "Ground Truth",
            "foundation_model": "Foundation Model",
            "deep_learning": "Deep Learning",
            "traditional": "Traditional",
            "ensemble": "Ensemble",
        }
        return fallback.get(category_id, category_id)


def get_category_short_name(display_name: str) -> str:
    """
    Get the short name for a category from its display name.

    Parameters
    ----------
    display_name : str
        Category display name (e.g., 'Ground Truth', 'Foundation Model')

    Returns
    -------
    str
        Short name (e.g., 'GT', 'FM')
    """
    try:
        loader = get_config_loader()
        names = loader.get_category_short_names()
        return names.get(display_name, display_name)
    except ConfigurationError:
        # Fallback for backward compatibility
        fallback = {
            "Ground Truth": "GT",
            "Foundation Model": "FM",
            "Deep Learning": "DL",
            "Traditional": "Trad",
            "Ensemble": "Ens",
        }
        return fallback.get(display_name, display_name)


def get_all_category_display_names() -> List[str]:
    """
    Get all category display names in canonical order.

    Returns
    -------
    list
        List of display names in order: Ground Truth, Foundation Model, Deep Learning, Traditional, Ensemble
    """
    try:
        loader = get_config_loader()
        names = loader.get_category_display_names()
        # Return in canonical order
        order = [
            "ground_truth",
            "foundation_model",
            "deep_learning",
            "traditional",
            "ensemble",
        ]
        return [names.get(cat_id, cat_id) for cat_id in order]
    except ConfigurationError:
        # Fallback for backward compatibility
        return [
            "Ground Truth",
            "Foundation Model",
            "Deep Learning",
            "Traditional",
            "Ensemble",
        ]


if __name__ == "__main__":
    # Demo the style
    import numpy as np

    apply_style()

    fig, ax = plt.subplots(figsize=get_figure_size("double"))

    x = np.linspace(0, 10, 100)
    for i, label in enumerate(["Method A", "Method B", "Method C"]):
        y = np.sin(x + i) + np.random.normal(0, 0.1, 100)
        ax.plot(x, y, label=label)

    style_axis(
        ax,
        title="Example Plot with Neue Haas Grotesk Style",
        xlabel="X Axis Label",
        ylabel="Y Axis Label",
    )
    add_reference_line(ax, 0, "Baseline")
    ax.legend()

    save_figure(fig, "style_demo", output_dir=Path("/tmp"))
    print("Style demo saved to /tmp/style_demo.png")
