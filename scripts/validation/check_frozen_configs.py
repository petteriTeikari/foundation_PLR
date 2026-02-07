#!/usr/bin/env python3
"""
Frozen config guard for pre-commit.

Prevents modification of experiment configs that have `frozen: true`.
These configs represent published results and should not be changed.

Usage:
    As pre-commit hook (automatic)
    Manual: python scripts/check_frozen_configs.py
"""

import subprocess
import sys
from pathlib import Path

import yaml

# Constants
CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"
EXPERIMENT_DIR = CONFIG_DIR / "experiment"


def get_staged_experiment_configs() -> list[Path]:
    """Get list of staged experiment config files."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=M"],
            capture_output=True,
            text=True,
            check=True,
            cwd=CONFIG_DIR.parent,
        )
        files = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            if line.startswith("configs/experiment/") and line.endswith(".yaml"):
                filepath = CONFIG_DIR.parent / line
                if filepath.exists():
                    files.append(filepath)
        return files
    except subprocess.CalledProcessError:
        return []


def is_frozen(filepath: Path) -> bool:
    """Check if a config file has frozen: true."""
    try:
        content = yaml.safe_load(filepath.read_text())
        if not isinstance(content, dict):
            return False

        # Check top-level frozen field
        if content.get("frozen") is True:
            return True

        # Check experiment.frozen field
        experiment = content.get("experiment", {})
        if isinstance(experiment, dict) and experiment.get("frozen") is True:
            return True

        return False
    except (yaml.YAMLError, Exception):
        return False


def get_original_content(filepath: Path) -> str | None:
    """Get file content from last commit."""
    try:
        rel_path = filepath.relative_to(CONFIG_DIR.parent)
        result = subprocess.run(
            ["git", "show", f"HEAD:{rel_path}"],
            capture_output=True,
            text=True,
            check=True,
            cwd=CONFIG_DIR.parent,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return None


def was_frozen_before(filepath: Path) -> bool:
    """Check if the file was frozen in the previous commit."""
    original = get_original_content(filepath)
    if original is None:
        return False

    try:
        content = yaml.safe_load(original)
        if not isinstance(content, dict):
            return False

        if content.get("frozen") is True:
            return True

        experiment = content.get("experiment", {})
        if isinstance(experiment, dict) and experiment.get("frozen") is True:
            return True

        return False
    except (yaml.YAMLError, Exception):
        return False


def main() -> int:
    """Main entry point."""
    configs = get_staged_experiment_configs()
    if not configs:
        return 0

    violations = []
    for config_path in configs:
        # Check if file WAS frozen (we care about modifying frozen configs)
        if was_frozen_before(config_path):
            rel_path = config_path.relative_to(CONFIG_DIR.parent)
            violations.append(str(rel_path))

    if violations:
        print("ERROR: Attempted to modify frozen experiment configs:")
        for v in violations:
            print(f"  - {v}")
        print()
        print("These configs have `frozen: true` and represent published results.")
        print("If you need to make changes:")
        print("  1. Create a new experiment config (e.g., paper_2026_v2.yaml)")
        print("  2. Or, if this is intentional, temporarily set frozen: false")
        print("     (but this will be flagged in review)")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
