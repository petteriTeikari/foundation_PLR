"""5×5 PLR Decomposition Grid Figure.

Phase 5 of the decomposition figure pipeline:
Creates a 25-subplot grid showing:
- Rows: 5 decomposition methods (Template, PCA, Rotated PCA, Sparse PCA, GED)
- Columns: 5 preprocessing categories (Ground Truth, FM, DL, Traditional, Ensemble)

Each subplot shows:
- Mean waveform (gray, dashed)
- Component timecourses with CIs
"""

# Import project utilities
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load grid spec from figure registry (single source of truth)
import yaml

from src.decomposition.aggregation import (
    DecompositionAggregator,
    DecompositionMethod,
    DecompositionResult,
    PreprocessingCategory,
)
from src.viz.plot_config import (
    COLORS,
    get_all_category_display_names,
    get_category_short_name,
    save_figure,
    setup_style,
)

_FIGURE_REGISTRY_PATH = (
    Path(__file__).parent.parent.parent
    / "configs"
    / "VISUALIZATION"
    / "figure_registry.yaml"
)


def _load_grid_spec() -> tuple[list[str], list[str]]:
    """Load method and category order from figure registry.

    Returns
    -------
    methods : list[str]
        Decomposition methods (rows)
    categories : list[str]
        Preprocessing categories (columns)

    Raises
    ------
    FileNotFoundError
        If figure_registry.yaml not found
    KeyError
        If fig_decomposition_grid.grid_spec not found
    """
    if not _FIGURE_REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Figure registry not found: {_FIGURE_REGISTRY_PATH}\n"
            "This is required for loading method/category order."
        )

    with open(_FIGURE_REGISTRY_PATH) as f:
        registry = yaml.safe_load(f)

    # Navigate to fig_decomposition_grid.grid_spec
    fig_config = registry.get("supplementary_figures", {}).get(
        "fig_decomposition_grid", {}
    )
    grid_spec = fig_config.get("grid_spec", {})

    methods = grid_spec.get("rows")
    categories = grid_spec.get("columns")

    if not methods or not categories:
        # Fallback to defaults (but warn)
        import warnings

        warnings.warn(
            "grid_spec not found in figure_registry.yaml, using defaults. "
            "Add grid_spec.rows and grid_spec.columns to fig_decomposition_grid entry."
        )
        methods = ["template", "pca", "rotated_pca", "sparse_pca", "ged"]
        # Load category display names from config (not hardcoded)
        categories = get_all_category_display_names()

    return methods, categories


# Load from config (single source of truth)
METHOD_ORDER, CATEGORY_ORDER = _load_grid_spec()

# Display name mappings (these are static, not config-driven)
METHOD_NAMES = {
    "template": "Template Fitting",
    "pca": "Standard PCA",
    "rotated_pca": "Rotated PCA (Promax)",
    "sparse_pca": "Sparse PCA",
    "ged": "GED",
}


def _get_category_short_names() -> dict:
    """Get category short names from config (not hardcoded)."""
    categories = get_all_category_display_names()
    return {cat: get_category_short_name(cat) for cat in categories}


CATEGORY_SHORT = _get_category_short_names()


# Component colors (from COLORS dict - colorblind-friendly Paul Tol palette)
# NOTE: COLORS must be initialized after import, so we use a function
def _get_component_colors() -> dict[str, str]:
    """Get component colors from COLORS dict (no hardcoding)."""
    return {
        # Template fitting components
        "phasic": COLORS["decomp_component_1"],  # Orange
        "sustained": COLORS["decomp_component_2"],  # Sky blue
        "pipr": COLORS["decomp_component_3"],  # Bluish green
        # PCA/GED numbered components
        "1": COLORS["decomp_component_1"],  # Component 1 - Orange
        "2": COLORS["decomp_component_2"],  # Component 2 - Sky blue
        "3": COLORS["decomp_component_3"],  # Component 3 - Bluish green
    }


COMPONENT_COLORS: dict[str, str] = {}  # Initialized lazily


def get_component_color(component_name: str) -> str:
    """Get color for a component based on its name."""
    global COMPONENT_COLORS
    # Lazy initialization to ensure COLORS is available
    if not COMPONENT_COLORS:
        COMPONENT_COLORS = _get_component_colors()

    if component_name in COMPONENT_COLORS:
        return COMPONENT_COLORS[component_name]
    # Extract number from component name (PC1, RC2, GED3, SPC1, etc.)
    for char in reversed(component_name):
        if char.isdigit():
            return COMPONENT_COLORS.get(char, COLORS.get("text_secondary", "#666666"))
    return COLORS.get("text_secondary", "#666666")


def plot_decomposition_subplot(
    ax: plt.Axes,
    result: DecompositionResult,
    show_ylabel: bool = False,
    show_xlabel: bool = False,
    show_legend: bool = False,
) -> None:
    """Plot a single decomposition result in a subplot.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on
    result : DecompositionResult
        Decomposition result with components and CIs
    show_ylabel : bool
        Whether to show y-axis label (leftmost column)
    show_xlabel : bool
        Whether to show x-axis label (bottom row)
    show_legend : bool
        Whether to show legend (top-right subplot)
    """
    time = result.time_vector

    # Plot mean waveform (gray, dashed) - using COLORS dict
    mean_color = COLORS["decomp_mean_waveform"]
    ax.plot(
        time,
        result.mean_waveform,
        color=mean_color,
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label="Mean",
        zorder=1,
    )
    ax.fill_between(
        time,
        result.mean_waveform_ci_lower,
        result.mean_waveform_ci_upper,
        color=mean_color,
        alpha=0.15,
        zorder=0,
    )

    # Plot each component
    for comp in result.components:
        color = get_component_color(comp.name)
        ax.plot(time, comp.mean, color=color, linewidth=1.5, label=comp.name, zorder=2)
        ax.fill_between(
            time, comp.ci_lower, comp.ci_upper, color=color, alpha=0.2, zorder=1
        )

    # Mark stimulus periods (using COLORS dict)
    ax.axvspan(
        15.5, 24.5, color=COLORS["blue_stimulus"], alpha=0.1, zorder=0
    )  # Blue stimulus
    ax.axvspan(
        46.5, 55.5, color=COLORS["red_stimulus"], alpha=0.1, zorder=0
    )  # Red stimulus

    # Formatting
    ax.set_xlim(0, 66)
    ax.axhline(0, color=COLORS["grid_lines"], linewidth=0.5, zorder=0)

    if show_xlabel:
        ax.set_xlabel("Time (s)")
    if show_ylabel:
        ax.set_ylabel("Component\nAmplitude")
    if show_legend:
        ax.legend(loc="upper right", fontsize=6, framealpha=0.9)

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def create_decomposition_grid(
    results: dict[
        tuple[PreprocessingCategory, DecompositionMethod], DecompositionResult
    ],
    figsize: tuple[float, float] = (14, 10),
) -> tuple[plt.Figure, dict]:
    """Create the 5×5 decomposition grid figure.

    Parameters
    ----------
    results : dict
        Results keyed by (category, method) tuples
    figsize : tuple
        Figure size in inches

    Returns
    -------
    fig : Figure
        Matplotlib figure
    data_dict : dict
        JSON-serializable data for reproducibility
    """
    setup_style()

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(5, 5, figure=fig, hspace=0.25, wspace=0.2)

    data_dict = {
        "figure_type": "decomposition_grid",
        "rows": METHOD_ORDER,
        "columns": CATEGORY_ORDER,
        "data": {},
    }

    for row_idx, method in enumerate(METHOD_ORDER):
        for col_idx, category in enumerate(CATEGORY_ORDER):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            key = (category, method)
            if key in results:
                result = results[key]
                plot_decomposition_subplot(
                    ax,
                    result,
                    show_ylabel=(col_idx == 0),
                    show_xlabel=(row_idx == 4),
                    show_legend=(row_idx == 0 and col_idx == 4),
                )

                # Store data for JSON
                data_key = f"{category}__{method}"
                data_dict["data"][data_key] = {
                    "n_subjects": result.n_subjects,
                    "time_vector": result.time_vector.tolist(),
                    "mean_waveform": result.mean_waveform.tolist(),
                    "components": [
                        {
                            "name": c.name,
                            "mean": c.mean.tolist(),
                            "ci_lower": c.ci_lower.tolist(),
                            "ci_upper": c.ci_upper.tolist(),
                        }
                        for c in result.components
                    ],
                }
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

            # Row labels (left side)
            if col_idx == 0:
                ax.set_ylabel(METHOD_NAMES[method], fontsize=9, fontweight="bold")

            # Column labels (top)
            if row_idx == 0:
                ax.set_title(category, fontsize=9, fontweight="bold")

    fig.suptitle(
        "PLR Waveform Decomposition by Preprocessing Method",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    return fig, data_dict


def generate_decomposition_figure(
    db_path: Path,
    output_dir: Path | None = None,
    n_bootstrap: int = 100,
    limit: int | None = None,
) -> Path:
    """Generate the full decomposition figure.

    Parameters
    ----------
    db_path : Path
        Path to preprocessed signals DuckDB
    output_dir : Path, optional
        Output directory (default: figures/generated)
    n_bootstrap : int
        Number of bootstrap iterations for CIs
    limit : int, optional
        Limit subjects per category (for testing)

    Returns
    -------
    output_path : Path
        Path to saved figure
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "figures" / "generated"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create aggregator and compute all decompositions
    aggregator = DecompositionAggregator(
        db_path=db_path, n_bootstrap=n_bootstrap, random_seed=42
    )

    print("Computing decompositions for all categories and methods...")
    results = aggregator.compute_all_decompositions(
        categories=CATEGORY_ORDER, methods=METHOD_ORDER, n_components=3, limit=limit
    )

    # Create figure
    fig, data_dict = create_decomposition_grid(results)

    # Save figure and data
    output_path = save_figure(fig, "fig_decomposition_grid", data=data_dict)

    plt.close(fig)
    print(f"Saved to: {output_path}")

    return output_path


def generate_test_figure() -> Path:
    """Generate figure with synthetic data for testing."""
    setup_style()

    # Create synthetic results
    np.random.seed(42)
    time_vector = np.linspace(0, 66, 200)

    def create_synthetic_result(category: str, method: str) -> DecompositionResult:
        from src.decomposition.aggregation import (
            ComponentTimecourse,
            DecompositionResult,
        )

        # Create synthetic components
        components = []

        # Base waveform shape
        mean_wave = 100 - 15 * np.exp(-np.maximum(0, time_vector - 15.5) / 2)
        mean_wave -= 10 * np.exp(-np.maximum(0, time_vector - 46.5) / 2)
        mean_wave += np.random.randn(len(time_vector)) * 0.5

        if method == "template":
            names = ["phasic", "sustained", "pipr"]
        elif method == "pca":
            names = ["PC1", "PC2", "PC3"]
        elif method == "rotated_pca":
            names = ["RC1", "RC2", "RC3"]
        elif method == "sparse_pca":
            names = ["SPC1", "SPC2", "SPC3"]
        else:  # ged
            names = ["GED1", "GED2", "GED3"]

        for i, name in enumerate(names):
            # Create synthetic component pattern
            phase = i * 10
            comp_mean = (
                10
                * np.exp(-((time_vector - 20 - phase) ** 2) / 50)
                * (1 if i % 2 == 0 else -1)
            )
            comp_mean += np.random.randn(len(time_vector)) * 0.2

            noise = 1 + 0.2 * (hash(category) % 10) / 10  # Category-dependent noise
            ci_width = np.abs(comp_mean) * 0.2 * noise

            components.append(
                ComponentTimecourse(
                    name=name,
                    mean=comp_mean,
                    ci_lower=comp_mean - ci_width,
                    ci_upper=comp_mean + ci_width,
                )
            )

        return DecompositionResult(
            category=category,
            method=method,
            time_vector=time_vector,
            components=components,
            n_subjects=50,
            mean_waveform=mean_wave,
            mean_waveform_ci_lower=mean_wave - 2,
            mean_waveform_ci_upper=mean_wave + 2,
        )

    # Generate all combinations
    results = {}
    for method in METHOD_ORDER:
        for category in CATEGORY_ORDER:
            results[(category, method)] = create_synthetic_result(category, method)

    # Create figure
    fig, data_dict = create_decomposition_grid(results)

    # Mark as synthetic
    data_dict["synthetic"] = True
    data_dict["warning"] = "TEST DATA - DO NOT USE FOR PUBLICATION"

    output_dir = Path(__file__).parent.parent.parent / "figures" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = save_figure(fig, "fig_decomposition_grid_TEST", data=data_dict)
    plt.close(fig)

    print(f"Test figure saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate PLR decomposition grid figure"
    )
    parser.add_argument(
        "--test", action="store_true", help="Generate test figure with synthetic data"
    )
    parser.add_argument("--db", type=Path, help="Path to preprocessed signals DuckDB")
    parser.add_argument(
        "--bootstrap", type=int, default=100, help="Bootstrap iterations"
    )
    parser.add_argument("--limit", type=int, help="Limit subjects (for testing)")

    args = parser.parse_args()

    if args.test:
        generate_test_figure()
    elif args.db:
        generate_decomposition_figure(
            args.db, n_bootstrap=args.bootstrap, limit=args.limit
        )
    else:
        # Default: try standard path
        default_db = (
            Path(__file__).parent.parent.parent
            / "data"
            / "private"
            / "preprocessed_signals_per_subject.db"
        )
        if default_db.exists():
            generate_decomposition_figure(
                default_db, n_bootstrap=args.bootstrap, limit=args.limit
            )
        else:
            print(
                "No DB found. Run with --test for synthetic data or --db to specify path"
            )
