#!/usr/bin/env python3
"""
extract_outlier_difficulty.py - Classify outliers as EASY (blinks) vs HARD (subtle).

Algorithm:
    deviation = |pupil_raw - pupil_gt|
    threshold = 3 * std(pupil_gt)  # per subject
    EASY = outlier_mask AND (deviation > threshold)
    HARD = outlier_mask AND (deviation <= threshold)

EASY outliers are obvious artifacts (blinks, eye closures) that should be easy
to detect. HARD outliers are subtle deviations that were marked by human
annotators but are near the signal - these are more challenging.

Output: data/outlier_difficulty_analysis.csv with columns:
    - subject_code
    - class_label
    - split
    - total_samples
    - total_outliers
    - easy_count
    - hard_count
    - easy_pct
    - hard_pct
    - outlier_pct

Usage:
    python scripts/extract_outlier_difficulty.py
"""

import argparse

# Input database - use centralized path utilities
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.paths import get_seri_db_path

DB_PATH = get_seri_db_path()

# Output path
DEFAULT_OUTPUT = (
    Path(__file__).parent.parent.parent / "data" / "outlier_difficulty_analysis.csv"
)


def classify_outlier_difficulty(df: pd.DataFrame) -> dict:
    """
    Classify outliers in a subject's data as EASY or HARD.

    EASY: |pupil_orig - pupil_gt| > 3 * std(pupil_gt)  -- obvious artifacts (blinks)
    HARD: |pupil_orig - pupil_gt| <= 3 * std(pupil_gt) -- subtle deviations

    Note: pupil_raw is NaN for outliers, so we use pupil_orig (original measurement)
    to calculate deviation from ground truth.

    Args:
        df: DataFrame with pupil_orig, pupil_gt, outlier_mask for one subject

    Returns:
        Dict with easy_count, hard_count, etc.
    """
    # Calculate deviation from ground truth using pupil_orig (not pupil_raw)
    # pupil_raw is NaN for outliers, but pupil_orig has the original measurement
    deviation = np.abs(df["pupil_orig"].values - df["pupil_gt"].values)

    # Calculate threshold based on ground truth std (across all samples for subject)
    gt_std = np.nanstd(df["pupil_gt"].values)
    threshold = 3 * gt_std

    # Get outlier mask (1 = outlier)
    outlier_mask = df["outlier_mask"].values == 1

    # Classify outliers (handle NaN in deviation)
    deviation_valid = ~np.isnan(deviation)
    easy_mask = outlier_mask & deviation_valid & (deviation > threshold)
    hard_mask = outlier_mask & deviation_valid & (deviation <= threshold)

    total_samples = len(df)
    total_outliers = np.sum(outlier_mask)
    easy_count = np.sum(easy_mask)
    hard_count = np.sum(hard_mask)

    return {
        "total_samples": total_samples,
        "total_outliers": int(total_outliers),
        "easy_count": int(easy_count),
        "hard_count": int(hard_count),
        "easy_pct": (easy_count / total_samples * 100) if total_samples > 0 else 0,
        "hard_pct": (hard_count / total_samples * 100) if total_samples > 0 else 0,
        "outlier_pct": (total_outliers / total_samples * 100)
        if total_samples > 0
        else 0,
        "gt_std": gt_std,
        "threshold": threshold,
    }


def extract_outlier_difficulty(db_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Extract easy vs hard outlier classification for all subjects.
    """
    conn = duckdb.connect(str(db_path), read_only=True)

    records = []

    # Process both train and test tables
    for table in ["train", "test"]:
        print(f"\nProcessing {table} table...")

        # Get unique subjects
        subjects = conn.execute(f"""
            SELECT DISTINCT subject_code, class_label, split
            FROM {table}
            WHERE subject_code IS NOT NULL
        """).fetchall()

        for subject_code, class_label, split in subjects:
            # Get data for this subject
            # Note: pupil_raw is NaN for outliers, use pupil_orig for deviation
            df = conn.execute(
                f"""
                SELECT pupil_orig, pupil_gt, outlier_mask
                FROM {table}
                WHERE subject_code = ?
                  AND pupil_orig IS NOT NULL
                  AND pupil_gt IS NOT NULL
            """,
                [subject_code],
            ).fetchdf()

            if len(df) == 0:
                continue

            # Classify outliers
            stats = classify_outlier_difficulty(df)

            records.append(
                {
                    "subject_code": subject_code,
                    "class_label": class_label if class_label else "Unknown",
                    "split": split if split else table,
                    **stats,
                }
            )

    conn.close()

    # Create DataFrame
    result_df = pd.DataFrame(records)

    # Sort by subject code
    result_df = result_df.sort_values("subject_code").reset_index(drop=True)

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    return result_df


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("OUTLIER DIFFICULTY ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nTotal subjects: {len(df)}")

    # By class
    for class_label in df["class_label"].unique():
        class_df = df[df["class_label"] == class_label]
        print(f"\n{class_label}:")
        print(f"  Subjects: {len(class_df)}")
        print(f"  Mean outlier %: {class_df['outlier_pct'].mean():.2f}%")
        print(f"  Mean EASY %: {class_df['easy_pct'].mean():.2f}%")
        print(f"  Mean HARD %: {class_df['hard_pct'].mean():.2f}%")

        total_outliers = class_df["total_outliers"].sum()
        total_easy = class_df["easy_count"].sum()
        total_hard = class_df["hard_count"].sum()

        print(f"  Total outliers: {total_outliers}")
        print(
            f"  Total EASY: {total_easy} ({total_easy / total_outliers * 100:.1f}% of outliers)"
        )
        print(
            f"  Total HARD: {total_hard} ({total_hard / total_outliers * 100:.1f}% of outliers)"
        )

    # Overall
    print("\n--- OVERALL ---")
    total_outliers = df["total_outliers"].sum()
    total_easy = df["easy_count"].sum()
    total_hard = df["hard_count"].sum()
    print(f"Total outliers across all subjects: {total_outliers}")
    print(
        f"EASY outliers (blinks, >3 SD): {total_easy} ({total_easy / total_outliers * 100:.1f}%)"
    )
    print(
        f"HARD outliers (subtle, ≤3 SD): {total_hard} ({total_hard / total_outliers * 100:.1f}%)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract easy vs hard outlier classification"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--db",
        "-d",
        type=Path,
        default=DB_PATH,
        help=f"Input database path (default: {DB_PATH})",
    )

    args = parser.parse_args()

    if not args.db.exists():
        print(f"ERROR: Database not found: {args.db}")
        return 1

    print(f"Extracting outlier difficulty from: {args.db}")
    df = extract_outlier_difficulty(args.db, args.output)

    print_summary(df)

    print(f"\n✓ Output saved to: {args.output}")
    print(f"  Rows: {len(df)}")

    return 0


if __name__ == "__main__":
    exit(main())
