"""
Public API for generating the synthetic PLR demo database.

This module provides a simple interface for generating the SYNTH_PLR_DEMO.db
database that can be used for testing the pipeline without real patient data.

Usage:
    # From command line:
    python -m src.synthetic.demo_dataset

    # From Python:
    from src.synthetic import generate_demo_database
    generate_demo_database()
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.synthetic.database_builder import build_synthetic_database
from src.synthetic.privacy_validator import quick_privacy_check, validate_privacy


# Default paths
DEFAULT_OUTPUT_DB = "data/synthetic/SYNTH_PLR_DEMO.db"
DEFAULT_METADATA = "data/synthetic/generation_params.yaml"
# Real DB path: resolved from DATA.filename_DuckDB in configs/defaults.yaml
# Falls back to parent directory convention (../SERI_PLR_GLAUCOMA.db)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REAL_DB_PATH = str(_PROJECT_ROOT.parent / "SERI_PLR_GLAUCOMA.db")


def generate_demo_database(
    output_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    n_subjects: int = 8,
    validate: bool = True,
    verbose: bool = True,
) -> Path:
    """
    Generate the synthetic PLR demo database.

    Parameters
    ----------
    output_path : str, optional
        Path for output database. Defaults to data/synthetic/SYNTH_PLR_DEMO.db
    metadata_path : str, optional
        Path for generation metadata YAML
    n_subjects : int
        Subjects per label per split (default 8 = 32 total)
    validate : bool
        Run privacy validation after generation
    verbose : bool
        Print progress messages

    Returns
    -------
    output_path : Path
        Path to generated database
    """
    if output_path is None:
        output_path = DEFAULT_OUTPUT_DB
    if metadata_path is None:
        metadata_path = DEFAULT_METADATA

    if verbose:
        print("Generating synthetic PLR database...")
        print(f"  Output: {output_path}")
        print(f"  Subjects per label per split: {n_subjects}")
        print(f"  Total subjects: {n_subjects * 2 * 2}")

    # Generate database
    db_path = build_synthetic_database(
        output_path=output_path,
        n_subjects_per_label_per_split=n_subjects,
        base_seed=42,
        outlier_pct_range=(0.05, 0.35),
        metadata_path=metadata_path,
    )

    if verbose:
        print(f"  Database created: {db_path}")

    # Run privacy validation
    if validate:
        if verbose:
            print("\nRunning privacy validation...")

        # Quick check (always)
        quick_valid = quick_privacy_check(str(db_path))
        if not quick_valid:
            print("  CRITICAL: Subject code validation FAILED!")
            sys.exit(1)

        # Full check if real data available
        real_db = Path(REAL_DB_PATH)
        if real_db.exists():
            is_valid, report = validate_privacy(
                str(db_path), str(real_db), verbose=verbose
            )
            if not is_valid:
                print("\n  CRITICAL: Privacy validation FAILED!")
                print("  Violations:")
                for v in report["violations"]:
                    print(f"    - {v}")
                print("\n  The generated data is TOO SIMILAR to real patient data.")
                print("  This database should NOT be committed to git.")
                sys.exit(1)
        else:
            if verbose:
                print(f"  Note: Real database not found at {real_db}")
                print("  Only subject code validation performed.")
                print("  Run full validation when real data is available.")

        if verbose:
            print("  Privacy validation: PASSED")

    if verbose:
        print(f"\nSuccess! Synthetic database ready at: {db_path}")
        print(f"Metadata saved to: {metadata_path}")

    return db_path


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate synthetic PLR demo database")
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT_DB,
        help=f"Output database path (default: {DEFAULT_OUTPUT_DB})",
    )
    parser.add_argument(
        "--metadata",
        "-m",
        default=DEFAULT_METADATA,
        help=f"Metadata YAML path (default: {DEFAULT_METADATA})",
    )
    parser.add_argument(
        "--n-subjects",
        "-n",
        type=int,
        default=8,
        help="Subjects per label per split (default: 8 = 32 total)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip privacy validation (not recommended)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()

    generate_demo_database(
        output_path=args.output,
        metadata_path=args.metadata,
        n_subjects=args.n_subjects,
        validate=not args.skip_validation,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
