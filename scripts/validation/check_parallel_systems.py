#!/usr/bin/env python3
"""
Parallel System Detection Script

Prevents creation of parallel directory structures that duplicate
existing canonical paths. For example:
- config/ when configs/ exists (Hydra)
- lib/ when src/ exists
- test/ when tests/ exists

Addresses: GAP-17 from reproducibility-synthesis-double-check.md

Usage:
    python scripts/check_parallel_systems.py           # Check for violations
    python scripts/check_parallel_systems.py --fix     # Report what would be fixed

Exit codes:
    0: No parallel systems detected
    1: Parallel systems found
"""

import argparse
import sys
from pathlib import Path


def find_project_root() -> Path:
    """Find project root by looking for marker files."""
    markers = ["pyproject.toml", "CLAUDE.md", ".git"]
    current = Path.cwd()

    while current != current.parent:
        if any((current / marker).exists() for marker in markers):
            return current
        current = current.parent

    return Path.cwd()


PROJECT_ROOT = find_project_root()

# Canonical paths and their banned alternatives
CANONICAL_PATHS = {
    "configs/": {
        "banned": ["config/", "configuration/", "conf/"],
        "reason": "Use configs/ with Hydra configuration system",
    },
    "src/": {
        "banned": ["lib/", "source/", "code/"],
        "reason": "Use src/ for all source code",
    },
    "tests/": {
        "banned": ["test/", "testing/", "spec/"],
        "reason": "Use tests/ for all test files",
    },
    "scripts/": {
        "banned": ["script/", "bin/", "tools/"],
        "reason": "Use scripts/ for utility scripts",
    },
    "docs/": {
        "banned": ["doc/", "documentation/"],
        "reason": "Use docs/ for documentation",
    },
    "figures/": {
        "banned": ["figs/", "images/", "plots/"],
        "reason": "Use figures/ for generated visualizations",
    },
    "data/": {
        "banned": ["datasets/", "input/"],
        "reason": "Use data/ for data files",
    },
    "apps/": {
        "banned": ["app/", "applications/"],
        "reason": "Use apps/ for application code",
    },
}


def check_parallel_systems() -> list[dict]:
    """
    Check for parallel directory structures.

    Returns list of violations.
    """
    violations = []

    for canonical, config in CANONICAL_PATHS.items():
        canonical_path = PROJECT_ROOT / canonical

        # Only check if canonical exists
        if not canonical_path.exists():
            continue

        for banned in config["banned"]:
            banned_path = PROJECT_ROOT / banned

            if banned_path.exists() and banned_path.is_dir():
                # Check if it has actual content (not empty)
                contents = list(banned_path.iterdir())
                if contents:
                    violations.append(
                        {
                            "canonical": canonical,
                            "banned": banned,
                            "reason": config["reason"],
                            "files_in_banned": len(contents),
                            "example_files": [f.name for f in contents[:3]],
                        }
                    )

    return violations


def check_duplicate_config_files() -> list[dict]:
    """
    Check for duplicate configuration files that should be consolidated.
    """
    duplicates = []

    # Check for duplicate YAML configs
    potential_duplicates = [
        ("configs/VISUALIZATION/colors.yaml", "colors.yaml"),
        ("configs/VISUALIZATION/combos.yaml", "combos.yaml"),
        ("configs/defaults.yaml", "config.yaml"),
    ]

    for canonical, duplicate in potential_duplicates:
        canonical_path = PROJECT_ROOT / canonical
        duplicate_path = PROJECT_ROOT / duplicate

        if canonical_path.exists() and duplicate_path.exists():
            duplicates.append(
                {
                    "canonical": canonical,
                    "duplicate": duplicate,
                    "reason": f"Use {canonical} instead of root-level {duplicate}",
                }
            )

    return duplicates


def format_violations(violations: list[dict], duplicates: list[dict]) -> str:
    """Format violations for display."""
    lines = []

    if violations:
        lines.append("=" * 60)
        lines.append("PARALLEL DIRECTORY VIOLATIONS DETECTED")
        lines.append("=" * 60)
        lines.append("")

        for v in violations:
            lines.append(
                f"VIOLATION: {v['banned']} exists (should use {v['canonical']})"
            )
            lines.append(f"  Reason: {v['reason']}")
            lines.append(f"  Files in banned dir: {v['files_in_banned']}")
            lines.append(f"  Examples: {', '.join(v['example_files'])}")
            lines.append("")

        lines.append("HOW TO FIX:")
        lines.append("-" * 40)
        for v in violations:
            lines.append(f"  mv {v['banned']}* {v['canonical']}")
            lines.append(f"  rmdir {v['banned']}")
        lines.append("")

    if duplicates:
        lines.append("=" * 60)
        lines.append("DUPLICATE CONFIG FILE WARNINGS")
        lines.append("=" * 60)
        lines.append("")

        for d in duplicates:
            lines.append(f"WARNING: {d['duplicate']} duplicates {d['canonical']}")
            lines.append(f"  {d['reason']}")
        lines.append("")

    if not violations and not duplicates:
        lines.append("✅ No parallel systems detected")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Detect parallel directory structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Show commands to fix violations (doesn't execute)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also fail on duplicate config warnings",
    )

    args = parser.parse_args()

    violations = check_parallel_systems()
    duplicates = check_duplicate_config_files()

    print(format_violations(violations, duplicates))

    # Exit code
    if violations:
        return 1
    if args.strict and duplicates:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
