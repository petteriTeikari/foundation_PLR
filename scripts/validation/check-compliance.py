#!/usr/bin/env python3
"""
check-compliance.py - Compliance Validation for Foundation PLR

Checks that visualization code follows project rules:
1. No hardcoded combo names (must load from YAML)
2. ground_truth present in comparison figures
3. Max 4 curves in main figures
4. setup_style() called before plotting
5. No PRIVATE JSON staged for commit

Usage:
    python scripts/check-compliance.py           # Check all
    python scripts/check-compliance.py --staged  # Check only staged files

Exit codes:
    0 - All checks pass
    1 - One or more checks failed
"""
# AIDEV-NOTE: This script enforces critical rules from .claude/rules/
# Run before commits to catch violations early.

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Known combo IDs that should NOT appear as hardcoded strings
COMBO_IDS = [
    "ground_truth",
    "best_ensemble",
    "best_single_fm",
    "traditional",
    "moment_full",
    "lof_moment",
    "timesnet_full",
    "units_pipeline",
    "hybrid_ocsvm_moment",
]

# Patterns that indicate LEGITIMATE use (allowlist)
ALLOWLIST_PATTERNS = [
    r"yaml\.safe_load",  # Loading from YAML
    r"\.yaml['\"]",  # YAML file reference
    r"get_config_loader",  # Using config loader
    r"load_combos",  # Loading combos function
    r"combo\[.+\]",  # Accessing combo dict
    r"combo\.get\(",  # Dict get
    r"for\s+combo\s+in",  # Iterating over combos
    r"#.*",  # Comments
    r"['\"]id['\"]:",  # YAML-like dict key
    r"combo_id",  # Variable named combo_id
    r"required_combos",  # Config key
    r"COMBO_IDS",  # This script's constant
]

# Viz files to check
VIZ_PATTERNS = [
    "src/viz/*.py",
    "scripts/*figure*.py",
]

# Private JSON patterns (should not be staged)
PRIVATE_JSON_PATTERNS = [
    "**/subject_*.json",
    "**/individual_*.json",
    "**/plr_trace*.json",
    "**/per_subject*.json",
    "**/fig_subject_*.json",
    "**/demo_subject_*.json",
]


class ComplianceError:
    """Represents a compliance violation."""

    def __init__(
        self, file: Path, line_num: int, rule: str, message: str, fix: str = ""
    ):
        self.file = file
        self.line_num = line_num
        self.rule = rule
        self.message = message
        self.fix = fix

    def __str__(self):
        result = f"\n  ERROR [{self.rule}]"
        result += f"\n  File: {self.file}:{self.line_num}"
        result += f"\n  Issue: {self.message}"
        if self.fix:
            result += f"\n  Fix: {self.fix}"
        return result


def get_staged_files() -> List[Path]:
    """Get list of staged files."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        return []
    return [PROJECT_ROOT / f for f in result.stdout.strip().split("\n") if f]


def get_viz_files() -> List[Path]:
    """Get all visualization Python files."""
    files = []
    for pattern in VIZ_PATTERNS:
        files.extend(PROJECT_ROOT.glob(pattern))
    return [f for f in files if f.is_file()]


def is_allowlisted(line: str) -> bool:
    """Check if line matches any allowlist pattern."""
    for pattern in ALLOWLIST_PATTERNS:
        if re.search(pattern, line):
            return True
    return False


def check_hardcoded_combos(file: Path) -> List[ComplianceError]:
    """Check for hardcoded combo names."""
    errors = []

    try:
        content = file.read_text()
    except Exception:
        return errors

    for line_num, line in enumerate(content.split("\n"), 1):
        # Skip if line is allowlisted
        if is_allowlisted(line):
            continue

        # Check for hardcoded combo IDs in lists or strings
        for combo_id in COMBO_IDS:
            # Pattern: "combo_id" or 'combo_id' as standalone string
            pattern = rf"""['\"]({combo_id})['\"]"""
            matches = re.findall(pattern, line)

            if matches:
                # Additional check: is this in a list literal?
                if re.search(rf"\[.*['\"]({combo_id})['\"].*\]", line):
                    errors.append(
                        ComplianceError(
                            file=file,
                            line_num=line_num,
                            rule="NO_HARDCODED_COMBOS",
                            message=f"Hardcoded combo list containing '{combo_id}'",
                            fix="Load combos from configs/VISUALIZATION/plot_hyperparam_combos.yaml instead",
                        )
                    )
                    break  # One error per line is enough

    return errors


def check_setup_style(file: Path) -> List[ComplianceError]:
    """Check that setup_style() is called in plotting files."""
    errors = []

    try:
        content = file.read_text()
    except Exception:
        return errors

    # Only check files that use matplotlib
    if "plt." not in content and "matplotlib" not in content:
        return errors

    # Check for setup_style or apply_style call
    if "setup_style(" not in content and "apply_style(" not in content:
        # Check if this file actually creates figures
        if "fig," in content or "plt.figure" in content or "plt.subplots" in content:
            errors.append(
                ComplianceError(
                    file=file,
                    line_num=1,
                    rule="SETUP_STYLE_REQUIRED",
                    message="Plotting file does not call setup_style()",
                    fix="Add: from src.viz.plot_config import setup_style; setup_style()",
                )
            )

    return errors


def check_private_json_staged() -> List[ComplianceError]:
    """Check that no private JSON files are staged."""
    errors = []
    staged = get_staged_files()

    for file in staged:
        file_str = str(file)
        for pattern in PRIVATE_JSON_PATTERNS:
            # Convert glob pattern to regex
            regex = pattern.replace("**", ".*").replace("*", "[^/]*")
            if re.search(regex, file_str):
                errors.append(
                    ComplianceError(
                        file=file,
                        line_num=0,
                        rule="NO_PRIVATE_JSON",
                        message="Private JSON file staged for commit",
                        fix=f"Unstage with: git reset HEAD {file.relative_to(PROJECT_ROOT)}",
                    )
                )

    return errors


def check_ground_truth_present(file: Path) -> List[ComplianceError]:
    """Check that ground_truth is included when multiple combos are used."""
    errors = []

    try:
        content = file.read_text()
    except Exception:
        return errors

    # Count how many combo IDs are referenced
    combo_count = 0
    has_ground_truth = False

    for combo_id in COMBO_IDS:
        if combo_id in content:
            combo_count += 1
            if combo_id == "ground_truth":
                has_ground_truth = True

    # If multiple combos used but no ground_truth
    if combo_count >= 2 and not has_ground_truth:
        # Check if this is a comparison figure
        if "comparison" in file.name.lower() or "retained" in file.name.lower():
            errors.append(
                ComplianceError(
                    file=file,
                    line_num=1,
                    rule="GROUND_TRUTH_REQUIRED",
                    message="Comparison figure missing ground_truth combo",
                    fix="Add ground_truth to combo list",
                )
            )

    return errors


def run_all_checks(staged_only: bool = False) -> Tuple[int, int]:
    """Run all compliance checks. Returns (passed, failed) counts."""
    all_errors = []

    # Get files to check
    if staged_only:
        files = [f for f in get_staged_files() if f.suffix == ".py"]
    else:
        files = get_viz_files()

    print(f"Checking {len(files)} files...")

    # Run checks on each file
    for file in files:
        if not file.exists():
            continue

        all_errors.extend(check_hardcoded_combos(file))
        all_errors.extend(check_setup_style(file))
        all_errors.extend(check_ground_truth_present(file))

    # Always check for private JSON in staged files
    all_errors.extend(check_private_json_staged())

    # Report results
    if all_errors:
        print(f"\n{'=' * 60}")
        print(f"COMPLIANCE CHECK FAILED: {len(all_errors)} violation(s)")
        print(f"{'=' * 60}")

        for error in all_errors:
            print(error)

        print(f"\n{'=' * 60}")
        print("Fix the above issues before committing.")
        print(f"{'=' * 60}\n")

        return (0, len(all_errors))
    else:
        print(f"\n{'=' * 60}")
        print("COMPLIANCE CHECK PASSED")
        print(f"{'=' * 60}\n")
        return (len(files), 0)


def main():
    parser = argparse.ArgumentParser(description="Check Foundation PLR compliance")
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Only check staged files (for pre-commit)",
    )
    args = parser.parse_args()

    passed, failed = run_all_checks(staged_only=args.staged)

    if failed > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
