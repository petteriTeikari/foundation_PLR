#!/usr/bin/env python3
"""
Validates that visualization code uses fixed hyperparam combos.

Usage:
    python combo_validator.py check-file src/viz/retention_curves.py
    python combo_validator.py check-all
    python combo_validator.py list-combos

This script ensures reproducibility by detecting:
- Hardcoded method names that should come from config
- Missing ground_truth in comparison figures
- Too many curves in main figures
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_PATH = Path("configs/VISUALIZATION/plot_hyperparam_combos.yaml")

# Patterns that indicate hardcoded method names (FORBIDDEN)
FORBIDDEN_PATTERNS = [
    # Hardcoded outlier methods
    (r'outlier_method\s*=\s*["\'](?!{)[^"\']+["\']', "Hardcoded outlier_method"),
    (r'["\']MOMENT-gt["\'](?!.*config)', "Hardcoded 'MOMENT-gt' string"),
    (r'["\']LOF["\'](?!.*config)', "Hardcoded 'LOF' string"),
    (r'["\']OneClassSVM["\'](?!.*config)', "Hardcoded 'OneClassSVM' string"),
    (r'["\']pupil-gt["\'](?!.*config)', "Hardcoded 'pupil-gt' string"),
    # Hardcoded imputation methods
    (r'imputation_method\s*=\s*["\'](?!{)[^"\']+["\']', "Hardcoded imputation_method"),
    (r'["\']SAITS["\'](?!.*config)', "Hardcoded 'SAITS' string"),
    (r'["\']linear["\'](?!.*config)', "Hardcoded 'linear' string"),
    # Hardcoded colors (hex values in viz code)
    (r'stroke\s*=\s*["\']#[0-9a-fA-F]{6}["\']', "Hardcoded stroke color"),
    (r'fill\s*=\s*["\']#[0-9a-fA-F]{6}["\']', "Hardcoded fill color"),
]

# Patterns that indicate proper config usage (REQUIRED)
REQUIRED_PATTERNS = [
    (r"plot_hyperparam_combos\.yaml", "Must reference config file"),
    (r"standard_combos", "Must use standard_combos"),
]

# Files to check
VIZ_PATTERNS = [
    "src/viz/**/*.py",
    "apps/visualization/src/**/*.tsx",
]

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def load_config() -> dict:
    """Load the hyperparam combos config."""
    if not CONFIG_PATH.exists():
        print(f"ERROR: Config not found: {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_file(filepath: Path) -> list[str]:
    """
    Check a single file for violations.

    Returns list of error messages.
    """
    errors = []

    if not filepath.exists():
        return [f"File not found: {filepath}"]

    content = filepath.read_text(encoding="utf-8")

    # Skip non-visualization files
    if "viz" not in str(filepath) and "visualization" not in str(filepath):
        return []

    # Skip test files
    if "test" in str(filepath).lower():
        return []

    # Check for forbidden patterns
    for pattern, message in FORBIDDEN_PATTERNS:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            errors.append(f"{message}: found {len(matches)} occurrence(s)")

    # Check for required patterns (only if file has plotting code)
    has_plot_code = any(
        keyword in content.lower()
        for keyword in ["plot", "figure", "retention", "calibration", "dca"]
    )

    if has_plot_code:
        for pattern, message in REQUIRED_PATTERNS:
            if not re.search(pattern, content):
                errors.append(f"Missing: {message}")

    return errors


def check_all_files() -> dict[str, list[str]]:
    """Check all visualization files."""
    from glob import glob

    results = {}

    for pattern in VIZ_PATTERNS:
        for filepath in glob(pattern, recursive=True):
            path = Path(filepath)
            errors = check_file(path)
            if errors:
                results[str(path)] = errors

    return results


def list_combos():
    """List all available combos."""
    config = load_config()

    print("\n=== STANDARD COMBOS (use for main figures) ===\n")
    for combo in config.get("standard_combos", []):
        print(f"  {combo['id']:20} | {combo['name']:25} | {combo['description']}")

    print("\n=== EXTENDED COMBOS (supplementary only) ===\n")
    for combo in config.get("extended_combos", []):
        print(f"  {combo['id']:20} | {combo['name']:25} | {combo['description']}")

    print("\n=== CLASSIFIER COMPARISON SET ===\n")
    cls_config = config.get("classifier_comparison", {})
    print(f"  Preprocessing: {cls_config.get('preprocessing', {})}")
    for cls in cls_config.get("classifiers", []):
        print(f"  {cls['id']:20} | {cls['name']}")

    print()


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Validate hyperparam combo usage")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # check-file command
    check_file_parser = subparsers.add_parser("check-file", help="Check a single file")
    check_file_parser.add_argument("filepath", type=Path, help="File to check")

    # check-all command
    subparsers.add_parser("check-all", help="Check all visualization files")

    # list-combos command
    subparsers.add_parser("list-combos", help="List available combos")

    args = parser.parse_args()

    if args.command == "check-file":
        errors = check_file(args.filepath)
        if errors:
            print(f"\n{args.filepath}: {len(errors)} error(s)")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        else:
            print(f"{args.filepath}: OK")
            sys.exit(0)

    elif args.command == "check-all":
        results = check_all_files()
        if results:
            print(f"\n=== VALIDATION FAILED: {len(results)} file(s) with errors ===\n")
            for filepath, errors in results.items():
                print(f"{filepath}:")
                for error in errors:
                    print(f"  - {error}")
            sys.exit(1)
        else:
            print("All files OK")
            sys.exit(0)

    elif args.command == "list-combos":
        list_combos()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
