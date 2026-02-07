#!/usr/bin/env python3
"""
Master Orchestration Script for Foundation PLR Results Reproduction.

This script orchestrates the two-block pipeline:
1. Block 1 (Extraction): MLflow → DuckDB with privacy separation
2. Block 2 (Analysis): Statistics, visualization, LaTeX generation

Usage:
    # Full pipeline (requires mlruns access)
    python scripts/reproduce_all_results.py

    # Analysis only (from public DuckDB checkpoint)
    python scripts/reproduce_all_results.py --from-checkpoint

    # Individual blocks
    python scripts/reproduce_all_results.py --extract-only
    python scripts/reproduce_all_results.py --analyze-only

Author: Foundation PLR Team
Date: 2026-01-25
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

# ============================================================================
# Configuration
# ============================================================================
# Use centralized path utilities for portable paths
from src.utils.paths import (
    PROJECT_ROOT,
    get_classification_experiment_id,
    get_mlruns_dir,
    get_results_db_path,
)

MLRUNS_DIR = get_mlruns_dir()
PUBLIC_DB_PATH = get_results_db_path()
PRIVATE_DIR = PROJECT_ROOT / "data" / "private"

CLASSIFICATION_EXP_ID = get_classification_experiment_id()


# ============================================================================
# Logging Setup
# ============================================================================


def setup_logging(verbose: bool = False) -> None:
    """Configure loguru logging."""
    logger.remove()

    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )

    # Also log to file
    log_file = PROJECT_ROOT / "logs" / f"reproduce_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_file, level="DEBUG", rotation="10 MB")


# ============================================================================
# Block Execution
# ============================================================================


def run_extraction(
    mlruns_dir: Path = MLRUNS_DIR,
    experiment_id: str = CLASSIFICATION_EXP_ID,
) -> bool:
    """
    Run Block 1: Extraction flow.

    Requires access to mlruns directory.
    """
    logger.info("=" * 60)
    logger.info("RUNNING BLOCK 1: EXTRACTION")
    logger.info("=" * 60)

    # Check mlruns access
    if not mlruns_dir.exists():
        logger.error(f"MLruns directory not found: {mlruns_dir}")
        logger.error("Block 1 requires access to the original MLflow experiments.")
        return False

    try:
        from orchestration.flows.extraction_flow import extraction_flow

        result = extraction_flow(
            mlruns_dir=mlruns_dir,
            experiment_id=experiment_id,
        )

        logger.info("Block 1 completed successfully")
        logger.info(f"  Public DB: {result.get('public_db')}")
        logger.info(f"  Private lookup: {result.get('private_lookup')}")
        return True

    except Exception as e:
        logger.error(f"Block 1 failed: {e}")
        logger.exception(e)
        return False


def run_analysis(
    db_path: Path = PUBLIC_DB_PATH,
    skip_private: bool = False,
) -> bool:
    """
    Run Block 2: Analysis flow.

    Can run from public DuckDB alone (checkpoint mode).
    """
    logger.info("=" * 60)
    logger.info("RUNNING BLOCK 2: ANALYSIS")
    logger.info("=" * 60)

    # Check database exists
    if not db_path.exists():
        logger.error(f"Public database not found: {db_path}")
        logger.error("Run 'make extract' first, or provide --db-path.")
        return False

    try:
        from orchestration.flows.analysis_flow import analysis_flow

        result = analysis_flow(
            db_path=db_path,
            skip_private_figures=skip_private,
        )

        logger.info("Block 2 completed successfully")
        logger.info(f"  Figures: {len(result.get('figures', []))}")
        logger.info(f"  Tables: {len(result.get('tables', []))}")
        return True

    except Exception as e:
        logger.error(f"Block 2 failed: {e}")
        logger.exception(e)
        return False


# ============================================================================
# Pipeline Modes
# ============================================================================


def run_full_pipeline(args: argparse.Namespace) -> int:
    """Run full pipeline: Block 1 → Block 2."""
    logger.info("Running FULL PIPELINE (Block 1 + Block 2)")

    # Block 1: Extraction
    if not run_extraction(args.mlruns_dir, args.experiment_id):
        logger.error("Block 1 failed - aborting pipeline")
        return 1

    # Block 2: Analysis
    if not run_analysis(args.db_path, args.skip_private):
        logger.error("Block 2 failed")
        return 1

    logger.info("=" * 60)
    logger.info("FULL PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    return 0


def run_from_checkpoint(args: argparse.Namespace) -> int:
    """Run from checkpoint: Block 2 only (using existing public DB)."""
    logger.info("Running FROM CHECKPOINT (Block 2 only)")

    if not args.db_path.exists():
        logger.error(
            f"Checkpoint not found: {args.db_path}\n"
            "Run full pipeline first with: python scripts/reproduce_all_results.py"
        )
        return 1

    if not run_analysis(args.db_path, args.skip_private):
        return 1

    logger.info("=" * 60)
    logger.info("CHECKPOINT PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    return 0


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce all Foundation PLR results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline (requires mlruns access)
    python scripts/reproduce_all_results.py

    # Analysis only from checkpoint
    python scripts/reproduce_all_results.py --from-checkpoint

    # Extract only (generate public DB)
    python scripts/reproduce_all_results.py --extract-only

    # Analyze only (from existing public DB)
    python scripts/reproduce_all_results.py --analyze-only
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--from-checkpoint",
        action="store_true",
        help="Run analysis only from existing public DuckDB",
    )
    mode_group.add_argument(
        "--extract-only",
        action="store_true",
        help="Run extraction only (Block 1)",
    )
    mode_group.add_argument(
        "--analyze-only",
        action="store_true",
        help="Run analysis only (Block 2)",
    )

    # Paths
    parser.add_argument(
        "--mlruns-dir",
        type=Path,
        default=MLRUNS_DIR,
        help="Path to mlruns directory",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=PUBLIC_DB_PATH,
        help="Path to public DuckDB",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=CLASSIFICATION_EXP_ID,
        help="MLflow experiment ID",
    )

    # Options
    parser.add_argument(
        "--skip-private",
        action="store_true",
        help="Skip figures requiring private data",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger.info("Foundation PLR Results Reproduction Pipeline")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Started at: {datetime.now()}")

    # Execute based on mode
    if args.from_checkpoint or args.analyze_only:
        return run_from_checkpoint(args)
    elif args.extract_only:
        success = run_extraction(args.mlruns_dir, args.experiment_id)
        return 0 if success else 1
    else:
        return run_full_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
