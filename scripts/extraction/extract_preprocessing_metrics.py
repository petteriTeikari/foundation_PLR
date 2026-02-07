#!/usr/bin/env python3
"""
extract_preprocessing_metrics.py - Extract preprocessing metrics for correlation analysis.

This script extracts:
1. Outlier F1 scores from PLR_OutlierDetection experiment
2. Imputation MAE from PLR_Imputation experiment
3. CatBoost AUROC from cd_preprocessing_catboost.duckdb

Output: data/preprocessing_correlation_data.csv with columns:
    - outlier_method
    - imputation_method
    - outlier_f1
    - outlier_f1_ci_lower
    - outlier_f1_ci_upper
    - imputation_mae
    - imputation_mae_ci_lower
    - imputation_mae_ci_upper
    - catboost_auroc
    - catboost_auroc_ci_lower
    - catboost_auroc_ci_upper

Usage:
    python scripts/extract_preprocessing_metrics.py
"""

import argparse

# Paths - use centralized path utilities
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.paths import get_mlruns_dir

MLRUNS_PATH = get_mlruns_dir()
OUTLIER_EXP_ID = "996740926475477194"
IMPUTATION_EXP_ID = "940304421003085572"
CD_DB_PATH = (
    Path(__file__).parent.parent.parent / "data" / "cd_preprocessing_catboost.duckdb"
)
DEFAULT_OUTPUT = (
    Path(__file__).parent.parent.parent / "data" / "preprocessing_correlation_data.csv"
)


def read_mlflow_metric(metric_file: Path) -> float:
    """Read a single MLflow metric file."""
    with open(metric_file) as f:
        line = f.readline().strip()
        # Format: timestamp value step
        parts = line.split()
        if len(parts) >= 2:
            return float(parts[1])
    return np.nan


def extract_outlier_metrics(mlruns_path: Path) -> pd.DataFrame:
    """Extract F1 scores from outlier detection experiments."""
    exp_path = mlruns_path / OUTLIER_EXP_ID
    records = []

    for run_dir in exp_path.iterdir():
        if not run_dir.is_dir():
            continue

        # Get run name from tags
        run_name_file = run_dir / "tags" / "mlflow.runName"
        if not run_name_file.exists():
            continue

        with open(run_name_file) as f:
            run_name = f.read().strip()

        # Parse method name from run name
        # Examples: "MOMENT_finetune_MOMENT-1-large_pupil_orig_imputed" -> "MOMENT-finetune"
        #           "LOF_pupil_orig_imputed" -> "LOF"
        method = run_name.split("_")[0]
        if "zeroshot" in run_name:
            method = method + "-zeroshot"
        elif "finetune" in run_name:
            method = method + "-finetune"

        # Check for pupil_gt vs pupil_orig
        if "pupil_gt" in run_name:
            method = method + "-gt"
        elif "pupil_orig" in run_name:
            method = method + "-orig"

        # Get F1 score from test metrics
        test_metrics = run_dir / "metrics" / "outlier_test"
        if not test_metrics.exists():
            test_metrics = run_dir / "metrics" / "test"

        if test_metrics.exists():
            f1_file = test_metrics / "f1"
            f1_ci_lo_file = test_metrics / "f1_CI_lo"
            f1_ci_hi_file = test_metrics / "f1_CI_hi"

            f1 = read_mlflow_metric(f1_file) if f1_file.exists() else np.nan
            f1_ci_lo = (
                read_mlflow_metric(f1_ci_lo_file) if f1_ci_lo_file.exists() else np.nan
            )
            f1_ci_hi = (
                read_mlflow_metric(f1_ci_hi_file) if f1_ci_hi_file.exists() else np.nan
            )

            if not np.isnan(f1):
                records.append(
                    {
                        "method": method,
                        "run_name": run_name,
                        "f1": f1,
                        "f1_ci_lower": f1_ci_lo,
                        "f1_ci_upper": f1_ci_hi,
                    }
                )

    df = pd.DataFrame(records)

    # Group by method and take the best F1 if multiple runs
    if len(df) > 0:
        df = (
            df.groupby("method")
            .agg(
                {
                    "f1": "max",
                    "f1_ci_lower": "first",
                    "f1_ci_upper": "first",
                }
            )
            .reset_index()
        )

    return df


def extract_imputation_metrics(mlruns_path: Path) -> pd.DataFrame:
    """Extract MAE from imputation experiments."""
    exp_path = mlruns_path / IMPUTATION_EXP_ID
    records = []

    for run_dir in exp_path.iterdir():
        if not run_dir.is_dir():
            continue

        # Get run name from tags
        run_name_file = run_dir / "tags" / "mlflow.runName"
        if not run_name_file.exists():
            continue

        with open(run_name_file) as f:
            run_name = f.read().strip()

        # Parse imputation method from run name
        # First part before first underscore is the method
        method = run_name.split("_")[0]

        # Get MAE from test metrics
        test_metrics = run_dir / "metrics" / "test"
        if test_metrics.exists():
            mae_file = test_metrics / "mae"
            mae_ci_lo_file = test_metrics / "mae_CI_lo"
            mae_ci_hi_file = test_metrics / "mae_CI_hi"

            mae = read_mlflow_metric(mae_file) if mae_file.exists() else np.nan
            mae_ci_lo = (
                read_mlflow_metric(mae_ci_lo_file)
                if mae_ci_lo_file.exists()
                else np.nan
            )
            mae_ci_hi = (
                read_mlflow_metric(mae_ci_hi_file)
                if mae_ci_hi_file.exists()
                else np.nan
            )

            if not np.isnan(mae):
                records.append(
                    {
                        "method": method,
                        "run_name": run_name,
                        "mae": mae,
                        "mae_ci_lower": mae_ci_lo,
                        "mae_ci_upper": mae_ci_hi,
                    }
                )

    df = pd.DataFrame(records)

    # Group by method and take the best (lowest) MAE
    if len(df) > 0:
        df = (
            df.groupby("method")
            .agg(
                {
                    "mae": "min",
                    "mae_ci_lower": "first",
                    "mae_ci_upper": "first",
                }
            )
            .reset_index()
        )

    return df


def extract_catboost_auroc(db_path: Path) -> pd.DataFrame:
    """Extract CatBoost AUROC from CD diagram database."""
    conn = duckdb.connect(str(db_path), read_only=True)

    df = conn.execute("""
        SELECT
            outlier_method,
            imputation_method,
            mean_auroc as auroc,
            -- Approximate CI from bootstrap iterations
            mean_auroc - 1.96 * std_auroc as auroc_ci_lower,
            mean_auroc + 1.96 * std_auroc as auroc_ci_upper
        FROM preprocessing_pivot
        ORDER BY mean_auroc DESC
    """).fetchdf()

    conn.close()
    return df


def create_correlation_dataset(
    outlier_df: pd.DataFrame, imputation_df: pd.DataFrame, auroc_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Join outlier F1, imputation MAE, and AUROC into a single dataset.

    Note: This creates rows for each (outlier, imputation) combination that
    has AUROC data. Not all combinations have matching outlier F1 or imputation MAE.
    """
    # Start with AUROC data (this is our primary dataset)
    df = auroc_df.copy()

    # Add outlier F1 scores
    # Need to match outlier method names
    df["outlier_f1"] = np.nan
    df["outlier_f1_ci_lower"] = np.nan
    df["outlier_f1_ci_upper"] = np.nan

    for _, row in outlier_df.iterrows():
        method = row["method"]
        # Try to match method name in outlier_method column
        mask = df["outlier_method"].str.contains(
            method.replace("-", ".*"), case=False, regex=True
        )
        df.loc[mask, "outlier_f1"] = row["f1"]
        df.loc[mask, "outlier_f1_ci_lower"] = row["f1_ci_lower"]
        df.loc[mask, "outlier_f1_ci_upper"] = row["f1_ci_upper"]

    # Add imputation MAE
    df["imputation_mae"] = np.nan
    df["imputation_mae_ci_lower"] = np.nan
    df["imputation_mae_ci_upper"] = np.nan

    for _, row in imputation_df.iterrows():
        method = row["method"]
        mask = df["imputation_method"].str.upper() == method.upper()
        df.loc[mask, "imputation_mae"] = row["mae"]
        df.loc[mask, "imputation_mae_ci_lower"] = row["mae_ci_lower"]
        df.loc[mask, "imputation_mae_ci_upper"] = row["mae_ci_upper"]

    # Rename AUROC columns for clarity
    df = df.rename(
        columns={
            "auroc": "catboost_auroc",
            "auroc_ci_lower": "catboost_auroc_ci_lower",
            "auroc_ci_upper": "catboost_auroc_ci_upper",
        }
    )

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract preprocessing metrics for correlation"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )

    args = parser.parse_args()

    print("Extracting preprocessing metrics...")

    # Extract outlier F1 scores
    print("\n1. Extracting outlier detection F1 scores...")
    outlier_df = extract_outlier_metrics(MLRUNS_PATH)
    print(f"   Found {len(outlier_df)} outlier methods")
    if len(outlier_df) > 0:
        print(outlier_df[["method", "f1"]].head(10).to_string())

    # Extract imputation MAE
    print("\n2. Extracting imputation MAE...")
    imputation_df = extract_imputation_metrics(MLRUNS_PATH)
    print(f"   Found {len(imputation_df)} imputation methods")
    if len(imputation_df) > 0:
        print(imputation_df[["method", "mae"]].head(10).to_string())

    # Extract CatBoost AUROC
    print("\n3. Extracting CatBoost AUROC...")
    auroc_df = extract_catboost_auroc(CD_DB_PATH)
    print(f"   Found {len(auroc_df)} preprocessing combinations")

    # Create combined dataset
    print("\n4. Creating combined correlation dataset...")
    result_df = create_correlation_dataset(outlier_df, imputation_df, auroc_df)

    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Total rows: {len(result_df)}")
    print(f"Rows with outlier F1: {result_df['outlier_f1'].notna().sum()}")
    print(f"Rows with imputation MAE: {result_df['imputation_mae'].notna().sum()}")
    print(
        f"Rows with both: {(result_df['outlier_f1'].notna() & result_df['imputation_mae'].notna()).sum()}"
    )

    # Correlation analysis
    complete_df = result_df.dropna(
        subset=["outlier_f1", "imputation_mae", "catboost_auroc"]
    )
    if len(complete_df) > 10:
        print("\n=== CORRELATIONS ===")
        corr_f1 = complete_df["catboost_auroc"].corr(complete_df["outlier_f1"])
        corr_mae = complete_df["catboost_auroc"].corr(complete_df["imputation_mae"])
        print(f"Correlation (AUROC vs Outlier F1): r = {corr_f1:.3f}")
        print(f"Correlation (AUROC vs Imputation MAE): r = {corr_mae:.3f}")

    print(f"\n✓ Output saved to: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
