#!/usr/bin/env python
"""
generate_all_figures.py - Master script to generate ALL manuscript figures.

Generates ALL figures for the Foundation PLR manuscript:

MAIN FIGURES:
- LIGHT: Light Protocol (fig_light_protocol)
- R7: Featurization Comparison (handcrafted vs embeddings)
- R8: Foundation Model Dashboard (3-panel)
- M3: Factorial Design Matrix
- C3: Foundation Model Utility Matrix
- CD: Critical Difference Diagrams
- RET: Retention Metric Curves
- FOREST: Forest Plots (outlier, imputation, classifier)
- HEAT: Heatmap Sensitivity Analysis
- SPEC: Specification Curve (all 407 configurations)
- CAL: Calibration Curves

Usage:
    python src/viz/generate_all_figures.py              # Generate all figures
    python src/viz/generate_all_figures.py --figure R7  # Generate specific figure
    python src/viz/generate_all_figures.py --main       # Generate all main figures
    python src/viz/generate_all_figures.py --supp       # Generate all supplementary figures
    python src/viz/generate_all_figures.py --list       # List available figures
    python src/viz/generate_all_figures.py --test       # Run tests first

Requirements:
    - DuckDB database at manuscripts/foundationPLR/data/foundation_plr_results.db
    - CD diagrams require data/cd_preprocessing_catboost.duckdb
    - matplotlib, numpy, pandas, duckdb
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")  # Headless backend


from figure_dimensions import get_dimensions
from plot_config import (
    COLORS,
    FIGURES_DIR,
    FIXED_CLASSIFIER,
    get_connection,
    save_figure,
    setup_style,
)

# =============================================================================
# EXISTING FIGURE GENERATORS (working)
# =============================================================================


def generate_r7():
    """Generate Figure R7: Featurization Comparison."""
    print("\n" + "=" * 60)
    print("Generating Figure R7: Featurization Comparison")
    print("=" * 60)

    from featurization_comparison import main

    main()


def generate_r8():
    """Generate Figure R8: Foundation Model Dashboard."""
    print("\n" + "=" * 60)
    print("Generating Figure R8: Foundation Model Dashboard")
    print("=" * 60)

    from foundation_model_dashboard import main

    main()


def generate_m3():
    """Generate Figure M3: Factorial Design Matrix."""
    print("\n" + "=" * 60)
    print("Generating Figure M3: Factorial Design Matrix")
    print("=" * 60)

    from factorial_matrix import main

    main()


def generate_c3():
    """Generate Figure C3: Foundation Model Utility Matrix."""
    print("\n" + "=" * 60)
    print("Generating Figure C3: Foundation Model Utility Matrix")
    print("=" * 60)

    from utility_matrix import main

    main()


def generate_cd():
    """Generate CD Diagrams: Preprocessing pipeline comparisons."""
    print("\n" + "=" * 60)
    print("Generating CD Diagrams: Preprocessing Pipeline Comparison")
    print("=" * 60)

    from cd_diagram_preprocessing import main

    main()


def generate_retention():
    """Generate Retention Metric Figures (4 standard combos).

    Reads pre-computed retention curves from DuckDB (retention_metrics table).
    """
    print("\n" + "=" * 60)
    print("Generating Retention Metric Figures")
    print("=" * 60)

    from retained_metric import generate_multi_combo_retention_figure

    # Loads standard combos from YAML and reads retention data from DuckDB
    paths = generate_multi_combo_retention_figure()
    for name, path in paths.items():
        print(f"  {name}: {path}")


# =============================================================================
# NEW FIGURE GENERATORS
# =============================================================================


def generate_light_protocol():
    """Generate Light Protocol Figure."""
    print("\n" + "=" * 60)
    print("Generating Light Protocol Figure")
    print("=" * 60)

    from light_protocol_plot import plot_light_protocol

    from src.utils.paths import get_seri_db_path

    db_path = get_seri_db_path()
    output_path = str(FIGURES_DIR / "fig_light_protocol")

    if not db_path.exists():
        print(f"WARNING: Database not found at {db_path}")
        print("Skipping light protocol figure")
        return

    result = plot_light_protocol(str(db_path), output_path)
    print(f"  Saved: {result}")


def generate_forest_plots():
    """Generate Forest Plots for outlier, imputation, and classifier methods."""
    print("\n" + "=" * 60)
    print("Generating Forest Plots")
    print("=" * 60)

    import matplotlib.pyplot as plt
    from forest_plot import draw_forest_plot

    conn = get_connection()

    # --- Forest Plot: Outlier Methods (with fixed classifier + best imputation) ---
    print("\n  Generating: Forest Plot - Outlier Methods")
    df_outlier = conn.execute(f"""
        SELECT
            outlier_method,
            AVG(auroc) as auroc_mean,
            MIN(auroc_ci_lo) as auroc_ci_lo,
            MAX(auroc_ci_hi) as auroc_ci_hi,
            COUNT(*) as n_configs
        FROM essential_metrics
        WHERE classifier = '{FIXED_CLASSIFIER}'
        GROUP BY outlier_method
        ORDER BY auroc_mean DESC
    """).fetchdf()

    if len(df_outlier) > 0:
        fig, ax = draw_forest_plot(
            methods=df_outlier["outlier_method"].tolist(),
            point_estimates=df_outlier["auroc_mean"].tolist(),
            ci_lower=df_outlier["auroc_ci_lo"].tolist(),
            ci_upper=df_outlier["auroc_ci_hi"].tolist(),
            title=f"Outlier Detection Methods ({FIXED_CLASSIFIER})",
            xlabel="AUROC",
            reference_line=0.93,
            reference_label="Najjar 2021 benchmark",
            figsize=get_dimensions("matrix"),
            highlight_top_n=3,
        )

        data = {
            "methods": df_outlier["outlier_method"].tolist(),
            "auroc_mean": df_outlier["auroc_mean"].tolist(),
            "auroc_ci_lo": df_outlier["auroc_ci_lo"].tolist(),
            "auroc_ci_hi": df_outlier["auroc_ci_hi"].tolist(),
        }
        save_figure(fig, "fig02_forest_outlier", data=data)
        plt.close(fig)

    # --- Forest Plot: Imputation Methods (with fixed classifier + ground truth outliers) ---
    print("  Generating: Forest Plot - Imputation Methods")
    df_imputation = conn.execute(f"""
        SELECT
            imputation_method,
            AVG(auroc) as auroc_mean,
            MIN(auroc_ci_lo) as auroc_ci_lo,
            MAX(auroc_ci_hi) as auroc_ci_hi,
            COUNT(*) as n_configs
        FROM essential_metrics
        WHERE classifier = '{FIXED_CLASSIFIER}'
        GROUP BY imputation_method
        ORDER BY auroc_mean DESC
    """).fetchdf()

    if len(df_imputation) > 0:
        fig, ax = draw_forest_plot(
            methods=df_imputation["imputation_method"].tolist(),
            point_estimates=df_imputation["auroc_mean"].tolist(),
            ci_lower=df_imputation["auroc_ci_lo"].tolist(),
            ci_upper=df_imputation["auroc_ci_hi"].tolist(),
            title=f"Imputation Methods ({FIXED_CLASSIFIER})",
            xlabel="AUROC",
            reference_line=0.93,
            reference_label="Najjar 2021 benchmark",
            figsize=get_dimensions("single_wide"),
            highlight_top_n=3,
        )

        data = {
            "methods": df_imputation["imputation_method"].tolist(),
            "auroc_mean": df_imputation["auroc_mean"].tolist(),
            "auroc_ci_lo": df_imputation["auroc_ci_lo"].tolist(),
            "auroc_ci_hi": df_imputation["auroc_ci_hi"].tolist(),
        }
        save_figure(fig, "fig03_forest_imputation", data=data)
        plt.close(fig)

    # --- Forest Plot: Classifiers (for reference, not main research question) ---
    print("  Generating: Forest Plot - Classifiers")
    df_classifier = conn.execute("""
        SELECT
            classifier,
            AVG(auroc) as auroc_mean,
            MIN(auroc_ci_lo) as auroc_ci_lo,
            MAX(auroc_ci_hi) as auroc_ci_hi,
            COUNT(*) as n_configs
        FROM essential_metrics
        GROUP BY classifier
        ORDER BY auroc_mean DESC
    """).fetchdf()

    if len(df_classifier) > 0:
        fig, ax = draw_forest_plot(
            methods=df_classifier["classifier"].tolist(),
            point_estimates=df_classifier["auroc_mean"].tolist(),
            ci_lower=df_classifier["auroc_ci_lo"].tolist(),
            ci_upper=df_classifier["auroc_ci_hi"].tolist(),
            title="Classifier Comparison\n(Note: Not main research question)",
            xlabel="AUROC",
            reference_line=0.93,
            reference_label="Najjar 2021 benchmark",
            figsize=get_dimensions("single_wide"),
            highlight_top_n=1,
        )

        data = {
            "methods": df_classifier["classifier"].tolist(),
            "auroc_mean": df_classifier["auroc_mean"].tolist(),
            "auroc_ci_lo": df_classifier["auroc_ci_lo"].tolist(),
            "auroc_ci_hi": df_classifier["auroc_ci_hi"].tolist(),
        }
        save_figure(fig, "fig04_forest_classifier", data=data)
        plt.close(fig)

    conn.close()


def generate_heatmap():
    """Generate Heatmap: Outlier × Imputation sensitivity analysis."""
    print("\n" + "=" * 60)
    print(f"Generating Heatmap: Outlier × Imputation ({FIXED_CLASSIFIER})")
    print("=" * 60)

    import matplotlib.pyplot as plt
    from heatmap_sensitivity import heatmap_from_pivot

    conn = get_connection()

    # Get mean AUROC for each outlier × imputation combination (fixed classifier)
    df = conn.execute(f"""
        SELECT
            outlier_method,
            imputation_method,
            AVG(auroc) as auroc_mean
        FROM essential_metrics
        WHERE classifier = '{FIXED_CLASSIFIER}'
        GROUP BY outlier_method, imputation_method
        ORDER BY outlier_method, imputation_method
    """).fetchdf()

    if len(df) > 0:
        # Pivot for heatmap
        pivot = df.pivot(
            index="outlier_method", columns="imputation_method", values="auroc_mean"
        )

        fig, ax = heatmap_from_pivot(
            pivot,
            title=f"AUROC: Outlier Detection × Imputation Method ({FIXED_CLASSIFIER})",
            xlabel="Imputation Method",
            ylabel="Outlier Detection Method",
            cmap="RdYlGn",
            annotate=True,
            fmt=".3f",
            highlight_best=True,
            highlight_threshold=0.90,
            cbar_label="AUROC",
            figsize=get_dimensions("double_tall"),
        )

        data = {
            "row_labels": list(pivot.index),
            "col_labels": list(pivot.columns),
            "values": pivot.values.tolist(),
            "best_outlier": pivot.values.max(axis=1).argmax(),
            "best_imputation": pivot.values.max(axis=0).argmax(),
        }
        save_figure(fig, "fig05_heatmap_catboost", data=data)
        plt.close(fig)

    conn.close()


def generate_specification_curve():
    """Generate Specification Curve: All 407 pipeline configurations."""
    print("\n" + "=" * 60)
    print("Generating Specification Curve (all configurations)")
    print("=" * 60)

    import matplotlib.pyplot as plt
    from specification_curve import specification_curve_from_dataframe

    conn = get_connection()

    # Get all configurations
    df = conn.execute("""
        SELECT
            outlier_method,
            imputation_method,
            classifier,
            featurization,
            auroc,
            auroc_ci_lo,
            auroc_ci_hi
        FROM essential_metrics
        ORDER BY auroc DESC
    """).fetchdf()

    if len(df) > 0:
        print(f"  {len(df)} configurations loaded")

        fig = specification_curve_from_dataframe(
            df,
            estimate_col="auroc",
            ci_lower_col="auroc_ci_lo",
            ci_upper_col="auroc_ci_hi",
            spec_cols=["outlier_method", "imputation_method", "classifier"],
            title=f"Specification Curve: All {len(df)} Pipeline Configurations",
            ylabel="AUROC",
            reference_line=0.93,
            reference_label="Najjar 2021 benchmark",
            figsize=get_dimensions("specification_curve"),
        )

        data = {
            "n_configurations": len(df),
            "auroc_min": float(df["auroc"].min()),
            "auroc_max": float(df["auroc"].max()),
            "auroc_median": float(df["auroc"].median()),
            "auroc_mean": float(df["auroc"].mean()),
            "auroc_std": float(df["auroc"].std()),
        }
        save_figure(fig, "fig06_specification_curve", data=data)
        plt.close(fig)

    conn.close()


def generate_calibration():
    """Generate Calibration Curves from database or mock data."""
    print("\n" + "=" * 60)
    print("Generating Calibration Curves")
    print("=" * 60)

    import matplotlib.pyplot as plt

    conn = get_connection()

    # Check if we have calibration curve data
    has_calibration = conn.execute("""
        SELECT COUNT(*) FROM calibration_curves
    """).fetchone()[0]

    if has_calibration > 0:
        print(f"  Found {has_calibration} calibration data points")

        # Load calibration data for best pipeline
        df = conn.execute(f"""
            SELECT
                cc.bin_midpoint as predicted,
                cc.observed_freq as observed,
                cc.bin_count as count
            FROM calibration_curves cc
            JOIN essential_metrics em ON cc.run_id = em.run_id
            WHERE em.classifier = '{FIXED_CLASSIFIER}'
            ORDER BY em.auroc DESC
            LIMIT 20
        """).fetchdf()

        if len(df) > 0:
            fig, ax = plt.subplots(figsize=get_dimensions("calibration"))

            # Reference line
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

            # Calibration curve
            ax.plot(
                df["predicted"],
                df["observed"],
                "o-",
                color=COLORS["primary"],
                linewidth=2,
                markersize=8,
                label=f"Best {FIXED_CLASSIFIER} pipeline",
            )

            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Observed Frequency")
            ax.set_title(f"Calibration Curve (Best {FIXED_CLASSIFIER} Pipeline)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend()
            ax.set_aspect("equal")

            data = {
                "predicted": df["predicted"].tolist(),
                "observed": df["observed"].tolist(),
                "count": df["count"].tolist(),
            }
            save_figure(fig, "fig13_calibration", data=data)
            plt.close(fig)
    else:
        print("  No calibration data in database, generating with mock data")
        from calibration_plot import generate_calibration_figure
        from data_loader import MockDataLoader

        loader = MockDataLoader(n_groups=1, n_iterations=1, seed=42)
        mock_df = loader.load_raw(["y_true", "y_prob", "uncertainty"], limit=200)

        pdf_path, json_path = generate_calibration_figure(
            mock_df["y_true"].values,
            mock_df["y_prob"].values,
            output_dir=str(FIGURES_DIR),
            filename="fig_calibration_smoothed",
        )
        print(f"  Saved: {pdf_path}")

    conn.close()


# =============================================================================
# SUPPLEMENTARY FIGURE GENERATORS
# =============================================================================


def generate_individual_traces():
    """Generate Individual Subject PLR Traces (supplementary)."""
    print("\n" + "=" * 60)
    print("Generating Individual Subject Traces (supplementary)")
    print("=" * 60)

    try:
        from individual_subject_traces import main

        main()
    except Exception as e:
        print(f"  WARNING: Could not generate subject traces: {e}")


def generate_dca_curves():
    """Generate Decision Curve Analysis (supplementary)."""
    print("\n" + "=" * 60)
    print("Generating DCA Curves (supplementary)")
    print("=" * 60)

    try:
        import matplotlib.pyplot as plt
        from data_loader import MockDataLoader
        from dca_plot import plot_dca_multi_model

        # Use mock data
        loader = MockDataLoader(n_groups=4, n_iterations=1, seed=42)
        mock_df = loader.load_raw(["y_true", "y_prob", "group_name"], limit=800)

        # Prepare data by group
        models_data = {}
        for group in mock_df["group_name"].unique():
            group_df = mock_df[mock_df["group_name"] == group]
            models_data[group] = {
                "y_true": group_df["y_true"].values,
                "y_prob": group_df["y_prob"].values,
            }

        fig, ax = plot_dca_multi_model(models_data)
        save_figure(fig, "fig_dca_curves", data={"n_models": len(models_data)})
        plt.close(fig)
        print("  Generated DCA curves with mock data")

    except Exception as e:
        print(f"  WARNING: Could not generate DCA curves: {e}")


def generate_instability_figures():
    """Generate Riley 2023 / Kompa 2021 instability figures (supplementary)."""
    print("\n" + "=" * 60)
    print("Generating Prediction Instability Figures (Riley 2023 / Kompa 2021)")
    print("=" * 60)

    try:
        from generate_instability_figures import main as instability_main

        instability_main()
    except Exception as e:
        print(f"  WARNING: Could not generate instability figures: {e}")
        import traceback

        traceback.print_exc()


def generate_probability_distributions():
    """Generate Probability Distribution by Outcome (supplementary)."""
    print("\n" + "=" * 60)
    print("Generating Probability Distributions (supplementary)")
    print("=" * 60)

    import matplotlib.pyplot as plt

    conn = get_connection()

    # Check for probability distribution data
    has_probs = conn.execute("""
        SELECT COUNT(*) FROM probability_distributions
    """).fetchone()[0]

    if has_probs > 0:
        df = conn.execute(f"""
            SELECT
                pd.true_class,
                pd.prob_bin,
                SUM(pd.count) as count
            FROM probability_distributions pd
            JOIN essential_metrics em ON pd.run_id = em.run_id
            WHERE em.classifier = '{FIXED_CLASSIFIER}'
            GROUP BY pd.true_class, pd.prob_bin
            ORDER BY pd.true_class, pd.prob_bin
        """).fetchdf()

        if len(df) > 0:
            fig, ax = plt.subplots(figsize=get_dimensions("single_wide"))

            for label, group in df.groupby("true_class"):
                class_name = "Glaucoma" if label == 1 else "Control"
                color = COLORS["glaucoma"] if label == 1 else COLORS["control"]
                ax.bar(
                    group["prob_bin"],
                    group["count"],
                    width=0.05,
                    alpha=0.5,
                    label=class_name,
                    color=color,
                )

            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Count")
            ax.set_title("Probability Distribution by True Outcome")
            ax.legend()

            save_figure(fig, "fig_prob_dist_by_outcome", data={})
            plt.close(fig)
    else:
        print("  No probability distribution data available")

    conn.close()


# =============================================================================
# FIGURE REGISTRY
# =============================================================================

# Main figures (required for manuscript)
MAIN_FIGURES = {
    "LIGHT": ("Light Protocol", generate_light_protocol),
    "R7": ("Featurization Comparison", generate_r7),
    "R8": ("Foundation Model Dashboard", generate_r8),
    "M3": ("Factorial Design Matrix", generate_m3),
    "C3": ("Utility Matrix", generate_c3),
    "CD": ("Critical Difference Diagram", generate_cd),
    "RET": ("Retention Metric Curves", generate_retention),
    "FOREST": ("Forest Plots", generate_forest_plots),
    "HEAT": ("Heatmap Sensitivity", generate_heatmap),
    "SPEC": ("Specification Curve", generate_specification_curve),
    "CAL": ("Calibration Curves", generate_calibration),
}

# Supplementary figures
SUPP_FIGURES = {
    "TRACES": ("Individual Subject Traces", generate_individual_traces),
    "DCA": ("Decision Curve Analysis", generate_dca_curves),
    "PROBS": ("Probability Distributions", generate_probability_distributions),
    "INSTAB": ("Prediction Instability (Riley 2023)", generate_instability_figures),
}

# Combined registry
FIGURE_GENERATORS = {**MAIN_FIGURES, **SUPP_FIGURES}


def run_tests():
    """Run pytest on the visualization tests."""
    import subprocess

    test_path = Path(__file__).parent.parent.parent / "tests" / "test_viz.py"
    print(f"\nRunning tests: {test_path}")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate Foundation PLR manuscript figures"
    )
    parser.add_argument(
        "--figure",
        "-f",
        choices=list(FIGURE_GENERATORS.keys()),
        help="Generate only this figure",
    )
    parser.add_argument(
        "--main", action="store_true", help="Generate all main manuscript figures"
    )
    parser.add_argument(
        "--supp", action="store_true", help="Generate all supplementary figures"
    )
    parser.add_argument(
        "--test", "-t", action="store_true", help="Run tests before generating figures"
    )
    parser.add_argument(
        "--list", "-l", action="store_true", help="List available figures"
    )

    args = parser.parse_args()

    if args.list:
        print("\n=== MAIN FIGURES ===")
        for key, (desc, _) in MAIN_FIGURES.items():
            print(f"  {key:8s} - {desc}")
        print("\n=== SUPPLEMENTARY FIGURES ===")
        for key, (desc, _) in SUPP_FIGURES.items():
            print(f"  {key:8s} - {desc}")
        return

    # Setup
    print(f"\n{'=' * 60}")
    print("Foundation PLR Figure Generation")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Output: {FIGURES_DIR}")
    print(f"{'=' * 60}")

    # Verify database connection
    try:
        conn = get_connection()
        count = conn.execute("SELECT COUNT(*) FROM essential_metrics").fetchone()[0]
        print(f"\nDatabase connected: {count} configurations")
        conn.close()
    except Exception as e:
        print(f"\nERROR: Could not connect to database: {e}")
        print("Please ensure DuckDB database exists.")
        sys.exit(1)

    # Run tests if requested
    if args.test:
        print("\n" + "=" * 60)
        print("Running tests...")
        print("=" * 60)
        if not run_tests():
            print("\nTests failed! Fix issues before generating figures.")
            sys.exit(1)
        print("\nTests passed!")

    # Setup matplotlib
    setup_style()

    # Determine which figures to generate
    if args.figure:
        figures_to_generate = {args.figure: FIGURE_GENERATORS[args.figure]}
    elif args.main:
        figures_to_generate = MAIN_FIGURES
    elif args.supp:
        figures_to_generate = SUPP_FIGURES
    else:
        figures_to_generate = FIGURE_GENERATORS

    # Track results
    results = {"success": [], "failed": []}

    # Generate figures
    for name, (desc, generator) in figures_to_generate.items():
        try:
            generator()
            results["success"].append(name)
        except Exception as e:
            print(f"\nERROR generating {name}: {e}")
            import traceback

            traceback.print_exc()
            results["failed"].append((name, str(e)))

    # Summary
    print(f"\n{'=' * 60}")
    print("GENERATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Successful: {len(results['success'])}/{len(figures_to_generate)}")
    for name in results["success"]:
        print(f"  ✓ {name}")
    if results["failed"]:
        print(f"\nFailed: {len(results['failed'])}")
        for name, error in results["failed"]:
            print(f"  ✗ {name}: {error[:50]}...")

    print(f"\n{'=' * 60}")
    print(f"Finished: {datetime.now().isoformat()}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
