"""
Block 2: Analysis Flow - Statistics, Visualization, and LaTeX Generation.

This flow works from the public DuckDB database and generates:
1. Statistical analyses (variance decomposition, pairwise comparisons)
2. Figures (forest plots, CD diagrams, calibration curves)
3. LaTeX artifacts (tables, numbers.tex)

Gracefully handles missing data (public DB or private traces) with warnings.

Usage:
    python -m src.orchestration.flows.analysis_flow
    python -m src.orchestration.flows.analysis_flow --skip-private
    python -m src.orchestration.flows.analysis_flow --figures-only
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

# Prefect compatibility layer (uses importlib for static-analysis safety)
from src.orchestration._prefect_compat import flow, task


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
VIZ_DIR = PROJECT_ROOT / "src" / "viz"

# Input paths
PUBLIC_DB_PATH = PROJECT_ROOT / "data" / "public" / "foundation_plr_results.db"
PRIVATE_DIR = PROJECT_ROOT / "data" / "private"
DEMO_TRACES_PATH = PRIVATE_DIR / "demo_subjects_traces.pkl"

# Output paths
FIGURES_DIR = PROJECT_ROOT / "figures" / "generated"
FIGURES_DATA_DIR = FIGURES_DIR / "data"
TABLES_DIR = PROJECT_ROOT / "tables" / "generated"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "latex"

# Benchmark reference
NAJJAR_AUROC = 0.93


# ============================================================================
# Data Availability Checks
# ============================================================================


@task(name="check_public_data")
def check_public_data(db_path: Path = PUBLIC_DB_PATH) -> Dict[str, Any]:
    """
    Check if public database exists and has required tables.

    Returns dict with availability status and basic stats.
    """
    result = {
        "available": False,
        "path": str(db_path),
        "tables": [],
        "n_predictions": 0,
        "n_metrics": 0,
        "error": None,
    }

    if not db_path.exists():
        result["error"] = f"Database not found: {db_path}"
        logger.warning(result["error"])
        return result

    try:
        import duckdb

        conn = duckdb.connect(str(db_path), read_only=True)

        # Check tables
        tables = conn.execute("SHOW TABLES").fetchall()
        result["tables"] = [t[0] for t in tables]

        # Count data
        if "predictions" in result["tables"]:
            result["n_predictions"] = conn.execute(
                "SELECT COUNT(*) FROM predictions"
            ).fetchone()[0]

        if "metrics_aggregate" in result["tables"]:
            result["n_metrics"] = conn.execute(
                "SELECT COUNT(*) FROM metrics_aggregate"
            ).fetchone()[0]

        conn.close()
        result["available"] = True
        logger.info(
            f"Public DB available: {result['n_predictions']} predictions, "
            f"{result['n_metrics']} metrics"
        )

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error checking public database: {e}")

    return result


@task(name="check_private_data")
def check_private_data() -> Dict[str, bool]:
    """
    Check availability of private data files.

    Returns dict of {artifact_name: exists}.
    """
    checks = {
        "demo_traces": DEMO_TRACES_PATH.exists(),
        "subject_lookup": (PRIVATE_DIR / "subject_lookup.yaml").exists(),
    }

    for name, exists in checks.items():
        if not exists:
            logger.warning(
                f"Private data '{name}' not found at {PRIVATE_DIR / name}. "
                f"Figures requiring this data will be skipped."
            )
        else:
            logger.info(f"Private data '{name}' available")

    return checks


# ============================================================================
# Figure Generation (via existing viz infrastructure)
# ============================================================================


@task(name="generate_all_figures")
def generate_all_figures(
    public_data_status: Dict[str, Any],
    private_data_status: Dict[str, bool],
    skip_private: bool = False,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate all manuscript figures using existing viz infrastructure.

    Delegates to src/viz/generate_all_figures.py with proper error handling.
    """
    results = {
        "generated": [],
        "skipped": [],
        "failed": [],
    }

    if not public_data_status.get("available"):
        logger.error(
            "Cannot generate figures: public database not available. "
            f"Error: {public_data_status.get('error')}"
        )
        return results

    # Set environment variable for viz modules to find the database
    if db_path is None:
        db_path = PUBLIC_DB_PATH
    os.environ["FOUNDATION_PLR_DB_PATH"] = str(db_path)
    logger.info(f"Set FOUNDATION_PLR_DB_PATH={db_path}")

    # Add viz directory to path for imports
    sys.path.insert(0, str(VIZ_DIR))

    try:
        # Import the figure generation infrastructure
        from plot_config import setup_style

        setup_style()
        logger.info("Matplotlib style configured")
    except Exception as e:
        logger.error(f"Failed to setup plotting style: {e}")
        return results

    # Define figure generators with their requirements
    figure_configs = [
        # Public figures (require only public DB)
        {"name": "R7_featurization", "func": "_generate_r7", "requires_private": False},
        {"name": "R8_dashboard", "func": "_generate_r8", "requires_private": False},
        {"name": "M3_factorial", "func": "_generate_m3", "requires_private": False},
        {"name": "C3_utility", "func": "_generate_c3", "requires_private": False},
        {"name": "CD_diagrams", "func": "_generate_cd", "requires_private": False},
        {"name": "forest_plots", "func": "_generate_forest", "requires_private": False},
        {"name": "heatmap", "func": "_generate_heatmap", "requires_private": False},
        {
            "name": "specification_curve",
            "func": "_generate_spec",
            "requires_private": False,
        },
        # Private figures (require demo traces)
        {"name": "demo_traces", "func": "_generate_traces", "requires_private": True},
    ]

    for config in figure_configs:
        name = config["name"]
        requires_private = config["requires_private"]

        # Skip private figures if requested or data unavailable
        if requires_private:
            if skip_private:
                logger.info(f"Skipping {name} (--skip-private flag)")
                results["skipped"].append(name)
                continue
            if not private_data_status.get("demo_traces"):
                logger.warning(
                    f"Skipping {name}: private demo traces not available. "
                    f"This figure requires data/private/demo_subjects_traces.pkl"
                )
                results["skipped"].append(name)
                continue

        # Try to generate
        try:
            logger.info(f"Generating: {name}")
            func = globals().get(config["func"])
            if func:
                func()
                results["generated"].append(name)
                logger.info(f"  ✓ {name} completed")
            else:
                logger.warning(f"  Generator function not found: {config['func']}")
                results["skipped"].append(name)
        except Exception as e:
            logger.error(f"  ✗ {name} failed: {e}")
            results["failed"].append({"name": name, "error": str(e)})

    return results


# ============================================================================
# Individual Figure Generators (wrappers around existing viz modules)
# ============================================================================


def _generate_r7():
    """Generate Figure R7: Featurization Comparison."""
    try:
        from featurization_comparison import main

        main()
    except ImportError as e:
        logger.warning(f"featurization_comparison module not available: {e}")
    except Exception as e:
        logger.error(f"R7 generation failed: {e}")
        raise


def _generate_r8():
    """Generate Figure R8: Foundation Model Dashboard."""
    try:
        from foundation_model_dashboard import main

        main()
    except ImportError as e:
        logger.warning(f"foundation_model_dashboard module not available: {e}")
    except Exception as e:
        logger.error(f"R8 generation failed: {e}")
        raise


def _generate_m3():
    """Generate Figure M3: Factorial Design Matrix."""
    try:
        from factorial_matrix import main

        main()
    except ImportError as e:
        logger.warning(f"factorial_matrix module not available: {e}")
    except Exception as e:
        logger.error(f"M3 generation failed: {e}")
        raise


def _generate_c3():
    """Generate Figure C3: Foundation Model Utility Matrix."""
    try:
        from utility_matrix import main

        main()
    except ImportError as e:
        logger.warning(f"utility_matrix module not available: {e}")
    except Exception as e:
        logger.error(f"C3 generation failed: {e}")
        raise


def _generate_cd():
    """Generate CD Diagrams."""
    try:
        from cd_diagram_preprocessing import main

        main()
    except ImportError as e:
        logger.warning(f"cd_diagram_preprocessing module not available: {e}")
    except Exception as e:
        logger.error(f"CD generation failed: {e}")
        raise


def _generate_forest():
    """Generate Forest Plots."""
    try:
        # Use the existing generate_all_figures infrastructure
        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt
        from forest_plot import draw_forest_plot
        from plot_config import get_connection, save_figure

        conn = get_connection()

        # Outlier methods
        df = conn.execute("""
            SELECT outlier_method,
                   AVG(auroc) as auroc_mean,
                   MIN(auroc_ci_lower) as auroc_ci_lower,
                   MAX(auroc_ci_upper) as auroc_ci_upper
            FROM essential_metrics
            WHERE classifier = 'CatBoost'
            GROUP BY outlier_method
            ORDER BY auroc_mean DESC
        """).fetchdf()

        if len(df) > 0:
            fig, ax = draw_forest_plot(
                methods=df["outlier_method"].tolist(),
                point_estimates=df["auroc_mean"].tolist(),
                ci_lower=df["auroc_ci_lower"].tolist(),
                ci_upper=df["auroc_ci_upper"].tolist(),
                title="Outlier Detection Methods (CatBoost)",
                xlabel="AUROC",
                reference_line=NAJJAR_AUROC,
            )
            save_figure(fig, "fig02_forest_outlier", data=df.to_dict("list"))
            plt.close(fig)

        conn.close()

    except Exception as e:
        logger.error(f"Forest plot generation failed: {e}")
        raise


def _generate_heatmap():
    """Generate Heatmap."""
    try:
        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt
        from heatmap_sensitivity import heatmap_from_pivot
        from plot_config import get_connection, save_figure

        conn = get_connection()

        df = conn.execute("""
            SELECT outlier_method, imputation_method, AVG(auroc) as auroc_mean
            FROM essential_metrics
            WHERE classifier = 'CatBoost'
            GROUP BY outlier_method, imputation_method
        """).fetchdf()

        if len(df) > 0:
            pivot = df.pivot(
                index="outlier_method", columns="imputation_method", values="auroc_mean"
            )
            fig, ax = heatmap_from_pivot(
                pivot,
                title="AUROC: Outlier × Imputation (CatBoost)",
                cmap="RdYlGn",
                annotate=True,
            )
            save_figure(
                fig,
                "fig05_heatmap_catboost",
                data={
                    "rows": list(pivot.index),
                    "cols": list(pivot.columns),
                    "values": pivot.values.tolist(),
                },
            )
            plt.close(fig)

        conn.close()

    except Exception as e:
        logger.error(f"Heatmap generation failed: {e}")
        raise


def _generate_spec():
    """Generate Specification Curve."""
    try:
        import matplotlib.pyplot as plt
        from plot_config import get_connection, save_figure
        from specification_curve import specification_curve_from_dataframe

        conn = get_connection()
        df = conn.execute("""
            SELECT outlier_method, imputation_method, classifier, auroc
            FROM essential_metrics
            ORDER BY auroc DESC
        """).fetchdf()

        if len(df) > 0:
            fig = specification_curve_from_dataframe(df, y_col="auroc")
            save_figure(
                fig,
                "fig06_specification_curve",
                data={
                    "n_configs": len(df),
                    "auroc_range": [df["auroc"].min(), df["auroc"].max()],
                },
            )
            plt.close(fig)

        conn.close()

    except Exception as e:
        logger.error(f"Specification curve generation failed: {e}")
        raise


def _generate_traces():
    """Generate demo subject trace figures (requires private data)."""
    import pickle

    if not DEMO_TRACES_PATH.exists():
        logger.warning("Demo traces not available, skipping")
        return

    try:
        with open(DEMO_TRACES_PATH, "rb") as f:
            data = pickle.load(f)

        traces = data.get("traces", {})
        logger.info(f"Loaded {len(traces)} demo traces")

        # Use light_protocol_plot for individual traces
        import matplotlib.pyplot as plt
        from light_protocol_plot import plot_subject_trace
        from plot_config import save_figure

        for code, trace_data in traces.items():
            fig = plot_subject_trace(
                time=trace_data["time"],
                pupil_raw=trace_data["pupil_raw"],
                pupil_gt=trace_data["pupil_gt"],
                outlier_mask=trace_data["outlier_mask"],
                title=f"Subject {code}",
            )
            # Note: JSON for individual traces is gitignored
            save_figure(fig, f"fig_subject_trace_{code}")
            plt.close(fig)

    except ImportError as e:
        logger.warning(f"light_protocol_plot not available: {e}")
    except Exception as e:
        logger.error(f"Demo trace generation failed: {e}")
        raise


# ============================================================================
# Statistics Computation
# ============================================================================


@task(name="compute_statistics")
def compute_statistics(
    public_data_status: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute summary statistics for the manuscript.
    """
    stats = {}

    if not public_data_status.get("available"):
        logger.warning("Cannot compute statistics: public database not available")
        return stats

    try:
        import duckdb

        conn = duckdb.connect(str(PUBLIC_DB_PATH), read_only=True)

        # Basic counts
        stats["n_predictions"] = public_data_status["n_predictions"]
        stats["n_metrics"] = public_data_status["n_metrics"]

        # AUROC statistics
        if "metrics_aggregate" in public_data_status["tables"]:
            result = conn.execute("""
                SELECT
                    MIN(AUROC_mean) as min_auroc,
                    MAX(AUROC_mean) as max_auroc,
                    AVG(AUROC_mean) as mean_auroc
                FROM metrics_aggregate
                WHERE AUROC_mean IS NOT NULL
            """).fetchone()

            if result:
                stats["min_auroc"] = result[0]
                stats["max_auroc"] = result[1]
                stats["mean_auroc"] = result[2]

        conn.close()
        logger.info(f"Computed statistics: {stats}")

    except Exception as e:
        logger.error(f"Error computing statistics: {e}")

    return stats


# ============================================================================
# LaTeX Artifacts
# ============================================================================


@task(name="generate_latex_artifacts")
def generate_latex_artifacts(
    stats: Dict[str, Any],
    output_dir: Path = ARTIFACTS_DIR,
) -> List[Path]:
    """
    Generate LaTeX artifacts (numbers.tex).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # Generate numbers.tex
    numbers_path = output_dir / "numbers.tex"

    content = f"""% Auto-generated statistics for inline LaTeX referencing
% Generated: {datetime.now().isoformat()}
% Source: {PUBLIC_DB_PATH}

% Sample sizes
\\providecommand{{\\nPredictions}}{{{stats.get("n_predictions", 0)}}}
\\providecommand{{\\nConfigs}}{{{stats.get("n_metrics", 0)}}}

% AUROC statistics
\\providecommand{{\\minAUROC}}{{{stats.get("min_auroc", 0):.3f}}}
\\providecommand{{\\maxAUROC}}{{{stats.get("max_auroc", 0):.3f}}}
\\providecommand{{\\meanAUROC}}{{{stats.get("mean_auroc", 0):.3f}}}

% Benchmark reference (Najjar et al. 2023)
\\providecommand{{\\najjarAUROC}}{{{NAJJAR_AUROC}}}
"""

    numbers_path.write_text(content)
    generated.append(numbers_path)
    logger.info(f"Generated: {numbers_path}")

    return generated


# ============================================================================
# Report Generation
# ============================================================================


@task(name="generate_report")
def generate_report(
    public_status: Dict[str, Any],
    private_status: Dict[str, bool],
    figure_results: Dict[str, Any],
    stats: Dict[str, Any],
    latex_files: List[Path],
) -> Path:
    """
    Generate analysis report JSON.
    """
    report = {
        "generated_at": datetime.now().isoformat(),
        "public_data": public_status,
        "private_data": private_status,
        "figures": figure_results,
        "statistics": stats,
        "latex_artifacts": [str(p) for p in latex_files],
        "status": "success" if public_status.get("available") else "incomplete",
    }

    # Add warnings
    warnings = []
    if not public_status.get("available"):
        warnings.append("Public database not available - most figures skipped")
    if not private_status.get("demo_traces"):
        warnings.append("Demo traces not available - subject trace figures skipped")
    if figure_results.get("failed"):
        warnings.append(f"{len(figure_results['failed'])} figures failed to generate")

    report["warnings"] = warnings

    report_path = PROJECT_ROOT / "analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Report saved: {report_path}")

    return report_path


# ============================================================================
# Main Flow
# ============================================================================


@flow(name="analysis-flow", log_prints=True)
def analysis_flow(
    db_path: Path = PUBLIC_DB_PATH,
    skip_private_figures: bool = False,
    figures_only: bool = False,
) -> Dict[str, Any]:
    """
    Block 2: Analysis and visualization from public DuckDB.

    Generates statistics, figures, and LaTeX artifacts.
    Gracefully handles missing data with warnings.

    Parameters
    ----------
    db_path : Path
        Path to public DuckDB database
    skip_private_figures : bool
        Skip figures that require private data
    figures_only : bool
        Only generate figures, skip stats and LaTeX
    """
    logger.info("=" * 60)
    logger.info("BLOCK 2: ANALYSIS FLOW")
    logger.info("=" * 60)

    # Step 1: Check data availability
    public_status = check_public_data(db_path)
    private_status = check_private_data()

    # Step 2: Generate figures
    figure_results = generate_all_figures(
        public_status,
        private_status,
        skip_private=skip_private_figures,
        db_path=db_path,
    )

    # Step 3: Compute statistics (unless figures_only)
    stats = {}
    if not figures_only:
        stats = compute_statistics(public_status)

    # Step 4: Generate LaTeX artifacts (unless figures_only)
    latex_files = []
    if not figures_only:
        latex_files = generate_latex_artifacts(stats)

    # Step 5: Generate report
    report_path = generate_report(
        public_status,
        private_status,
        figure_results,
        stats,
        latex_files,
    )

    # Summary
    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"  Figures generated: {len(figure_results.get('generated', []))}")
    logger.info(f"  Figures skipped: {len(figure_results.get('skipped', []))}")
    logger.info(f"  Figures failed: {len(figure_results.get('failed', []))}")
    logger.info(f"  Report: {report_path}")
    logger.info("=" * 60)

    return {
        "public_status": public_status,
        "private_status": private_status,
        "figures": figure_results,
        "statistics": stats,
        "latex_files": latex_files,
        "report": report_path,
    }


# ============================================================================
# CLI Entry Point
# ============================================================================


def main():
    """Run analysis flow from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Block 2: Analysis flow")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=PUBLIC_DB_PATH,
        help="Path to public DuckDB",
    )
    parser.add_argument(
        "--skip-private",
        action="store_true",
        help="Skip figures requiring private data",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Only generate figures, skip stats and LaTeX",
    )

    args = parser.parse_args()

    result = analysis_flow(
        db_path=args.db_path,
        skip_private_figures=args.skip_private,
        figures_only=args.figures_only,
    )

    # Print summary
    print("\nAnalysis complete!")
    print(f"  Figures generated: {len(result['figures'].get('generated', []))}")
    print(f"  Figures skipped: {len(result['figures'].get('skipped', []))}")
    if result["figures"].get("failed"):
        print(f"  Figures failed: {len(result['figures']['failed'])}")
        for f in result["figures"]["failed"]:
            print(f"    - {f['name']}: {f['error']}")


if __name__ == "__main__":
    main()
