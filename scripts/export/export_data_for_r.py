#!/usr/bin/env python3
"""
Export DuckDB data to R-friendly formats (CSV/JSON).

Phase 1.1 of ggplot2 visualization migration.
Creates essential_metrics.csv and top10_configs.json for R consumption.

Output Files:
    outputs/r_data/essential_metrics.csv - All 328 config metrics
    outputs/r_data/top10_configs.json - Top-10 CatBoost configs with metadata

Usage:
    python scripts/export_data_for_r.py

Author: Foundation PLR Team
Date: 2026-01-25
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import duckdb

# Configuration - Canonical data locations
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "public" / "foundation_plr_results.db"
OUTPUT_DIR = PROJECT_ROOT / "data" / "r_data"
SHAP_PKL = Path("outputs/shap_summary_top10.pkl")


def compute_file_hash(path: Path) -> str:
    """Compute MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()[:12]


def create_metadata(generator: str, source_db: Path) -> dict:
    """Create standard metadata wrapper for JSON exports."""
    return {
        "created": datetime.now().isoformat(),
        "schema_version": "1.0",
        "generator": generator,
        "r_script": "src/r/figures/<figure_name>.R",
        "data_source": {
            "database": str(source_db),
            "db_hash": compute_file_hash(source_db) if source_db.exists() else "N/A",
            "mlflow_experiment": "253031330985650090",
        },
    }


def export_essential_metrics(conn: duckdb.DuckDBPyConnection) -> None:
    """Export all config metrics to CSV.

    CRITICAL: Only exports HANDCRAFTED FEATURES (featurization containing 'simple').
    Embedding-based features (MOMENT-embedding, etc.) are excluded because they
    underperform by ~9pp and would contaminate downstream analyses if mixed.

    See: CRITICAL-FAILURE-002-mixed-featurization-in-extraction.md
    """
    print("\n1. Exporting essential_metrics.csv...")

    # CRITICAL: Filter to handcrafted features only!
    # Embedding features (MOMENT-embedding, MOMENT-embedding-PCA) have ~9pp lower AUROC
    # and must NOT be mixed with handcrafted feature results.
    # NOTE: Display names are included - these flow from the YAML lookup table
    # (configs/mlflow_registry/display_names.yaml) through extraction to here.
    # NOTE: STRATOS metrics are included - computed during extraction (not here!)
    # See: CRITICAL-FAILURE-003-computation-decoupling-violation.md
    df = conn.execute("""
        SELECT
            run_id,
            outlier_method,
            outlier_display_name,
            imputation_method,
            imputation_display_name,
            classifier,
            classifier_display_name,
            -- Core metrics
            auroc,
            auroc_ci_lo,
            auroc_ci_hi,
            brier,
            n_bootstrap,
            -- STRATOS calibration metrics (Van Calster 2024)
            calibration_slope,
            calibration_intercept,
            o_e_ratio,
            -- STRATOS overall metric
            scaled_brier,
            -- STRATOS clinical utility
            net_benefit_5pct,
            net_benefit_10pct,
            net_benefit_15pct,
            net_benefit_20pct
        FROM essential_metrics
        WHERE featurization LIKE '%simple%'
        ORDER BY auroc DESC
    """).fetchdf()

    # Validate: no duplicate (outlier, imputation, classifier) combinations
    n_total = len(df)
    n_unique = df.groupby(["outlier_method", "imputation_method", "classifier"]).ngroups
    if n_total != n_unique:
        print(f"   WARNING: {n_total - n_unique} duplicate combinations detected!")
        print("   This indicates mixed featurization types - check extraction!")
    else:
        print(f"   Validation: {n_unique} unique combinations (no duplicates)")

    output_path = OUTPUT_DIR / "essential_metrics.csv"
    df.to_csv(output_path, index=False)
    print(f"   Saved: {output_path} ({len(df)} rows)")


def export_top10_configs(conn: duckdb.DuckDBPyConnection) -> None:
    """Export top-10 CatBoost configs to JSON with metadata.

    CRITICAL: Only exports HANDCRAFTED FEATURES (featurization containing 'simple').
    See: CRITICAL-FAILURE-002-mixed-featurization-in-extraction.md
    """
    print("\n2. Exporting top10_configs.json...")

    # Query top-10 CatBoost configs with handcrafted features only
    # NOTE: Don't use top10_catboost view - it doesn't filter by featurization
    # NOTE: Display names are included from the YAML lookup table
    df = conn.execute("""
        SELECT
            ROW_NUMBER() OVER (ORDER BY auroc DESC) as rank,
            run_id,
            outlier_method,
            outlier_display_name,
            imputation_method,
            imputation_display_name,
            auroc,
            auroc_ci_lo,
            auroc_ci_hi
        FROM essential_metrics
        WHERE classifier = 'CatBoost'
          AND outlier_source_known = true
          AND featurization LIKE '%simple%'
        ORDER BY auroc DESC
        LIMIT 10
    """).fetchdf()

    # Convert to list of dicts
    configs = df.to_dict(orient="records")

    # Create output with metadata wrapper
    output = {
        "metadata": create_metadata("scripts/export_data_for_r.py", DB_PATH),
        "data": {
            "n_configs": len(configs),
            "classifier": "CatBoost",
            "selection_criterion": "AUROC mean",
            "configs": configs,
        },
    }

    output_path = OUTPUT_DIR / "top10_configs.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"   Saved: {output_path} ({len(configs)} configs)")


def export_catboost_metrics(conn: duckdb.DuckDBPyConnection) -> None:
    """Export all CatBoost config metrics for raincloud plots.

    CRITICAL: Only exports HANDCRAFTED FEATURES (featurization containing 'simple').
    See: CRITICAL-FAILURE-002-mixed-featurization-in-extraction.md
    """
    print("\n3. Exporting catboost_metrics.json...")

    # CRITICAL: Filter to handcrafted features only!
    # NOTE: Display names are included from the YAML lookup table
    # NOTE: STRATOS metrics computed during extraction, not here!
    df = conn.execute("""
        SELECT
            run_id,
            outlier_method,
            outlier_display_name,
            imputation_method,
            imputation_display_name,
            auroc,
            auroc_ci_lo,
            auroc_ci_hi,
            brier,
            -- STRATOS metrics for raincloud plots
            scaled_brier,
            calibration_slope,
            net_benefit_10pct
        FROM essential_metrics
        WHERE UPPER(classifier) = 'CATBOOST'
          AND featurization LIKE '%simple%'
        ORDER BY auroc DESC
    """).fetchdf()

    output = {
        "metadata": create_metadata("scripts/export_data_for_r.py", DB_PATH),
        "data": {
            "n_configs": len(df),
            "classifier": "CatBoost",
            "metrics": df.to_dict(orient="records"),
        },
    }

    output_path = OUTPUT_DIR / "catboost_metrics.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"   Saved: {output_path} ({len(df)} CatBoost configs)")


def create_data_manifest() -> None:
    """Create data version manifest for reproducibility."""
    print("\n4. Creating DATA_MANIFEST.yaml...")

    manifest = {
        "created": datetime.now().isoformat(),
        "sources": {
            "duckdb": {
                "path": str(DB_PATH),
                "hash": compute_file_hash(DB_PATH) if DB_PATH.exists() else "N/A",
            },
            "shap_summary": {
                "path": str(SHAP_PKL),
                "hash": compute_file_hash(SHAP_PKL) if SHAP_PKL.exists() else "N/A",
            },
        },
        "exports": [
            "essential_metrics.csv",
            "top10_configs.json",
            "catboost_metrics.json",
        ],
        "mlflow_experiment": "253031330985650090",
        "n_bootstrap_iterations": 1000,
    }

    output_path = OUTPUT_DIR / "DATA_MANIFEST.yaml"
    with open(output_path, "w") as f:
        # Simple YAML output without pyyaml dependency
        f.write("# Foundation PLR Data Manifest\n")
        f.write(f"# Generated: {manifest['created']}\n\n")
        f.write(f'created: "{manifest["created"]}"\n\n')
        f.write("sources:\n")
        f.write("  duckdb:\n")
        f.write(f'    path: "{manifest["sources"]["duckdb"]["path"]}"\n')
        f.write(f'    hash: "{manifest["sources"]["duckdb"]["hash"]}"\n')
        f.write("  shap_summary:\n")
        f.write(f'    path: "{manifest["sources"]["shap_summary"]["path"]}"\n')
        f.write(f'    hash: "{manifest["sources"]["shap_summary"]["hash"]}"\n\n')
        f.write("exports:\n")
        for exp in manifest["exports"]:
            f.write(f"  - {exp}\n")
        f.write(f'\nmlflow_experiment: "{manifest["mlflow_experiment"]}"\n')
        f.write(f"n_bootstrap_iterations: {manifest['n_bootstrap_iterations']}\n")

    print(f"   Saved: {output_path}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Phase 1.1: Export DuckDB Data for R")
    print("=" * 60)

    # Check database exists
    if not DB_PATH.exists():
        print(f"ERROR: Database not found: {DB_PATH}")
        print("Run extraction pipeline first.")
        sys.exit(1)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Connect to database
    print(f"\nConnecting to: {DB_PATH}")
    conn = duckdb.connect(str(DB_PATH), read_only=True)

    # Show available tables
    tables = conn.execute("SHOW TABLES").fetchall()
    print(f"Available tables: {[t[0] for t in tables]}")

    # Export data
    export_essential_metrics(conn)
    export_top10_configs(conn)
    export_catboost_metrics(conn)
    create_data_manifest()

    conn.close()

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("Files created:")
    for f in OUTPUT_DIR.glob("*"):
        print(f"  - {f.name}")

    print("\nNext steps:")
    print("  1. Run: python scripts/export_shap_for_r.py")
    print("  2. Run: python scripts/compute_vif.py")
    print("  3. Open R and source: source('src/r/setup.R')")


if __name__ == "__main__":
    main()
