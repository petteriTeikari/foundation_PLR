#!/usr/bin/env python3
"""
Pre-commit hook: Figure Isolation Check

Verifies that figure generation properly routes synthetic data to
figures/synthetic/ and production data to figures/generated/.

Part of the 4-gate isolation architecture.

This hook checks:
1. No production figures in figures/synthetic/
2. No synthetic figures in figures/generated/
3. JSON metadata properly marks synthetic data
4. save_figure() supports synthetic parameter

Exit codes:
- 0: All checks pass
- 1: Isolation violation detected

Usage:
    python scripts/validation/check_figure_isolation.py
    # Or via pre-commit hook
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
FIGURES_GENERATED = PROJECT_ROOT / "figures" / "generated"
FIGURES_SYNTHETIC = PROJECT_ROOT / "figures" / "synthetic"


def check_save_figure_supports_synthetic() -> list[str]:
    """Check that save_figure() supports synthetic parameter."""
    errors = []

    plot_config = PROJECT_ROOT / "src" / "viz" / "plot_config.py"

    if not plot_config.exists():
        errors.append(f"plot_config.py not found: {plot_config}")
        return errors

    content = plot_config.read_text()

    # Check for synthetic parameter in save_figure
    if "def save_figure" not in content:
        errors.append("save_figure() function not found in plot_config.py")
        return errors

    # Check for synthetic parameter
    if "synthetic" not in content or "is_synthetic_mode" not in content:
        errors.append(
            "ISOLATION VIOLATION: save_figure() doesn't support synthetic mode\n"
            "  FIX: Add 'synthetic' parameter and is_synthetic_mode() detection"
        )

    # Check for data_mode import
    if "from src.utils.data_mode import" not in content:
        errors.append(
            "ISOLATION VIOLATION: plot_config.py doesn't import data_mode utilities\n"
            "  FIX: Add 'from src.utils.data_mode import get_figures_dir_for_mode, is_synthetic_mode'"
        )

    return errors


def check_no_synthetic_in_generated() -> list[str]:
    """Check that figures/generated/ doesn't contain synthetic data."""
    errors = []

    if not FIGURES_GENERATED.exists():
        return errors  # No generated figures yet

    # Check JSON files for synthetic markers
    json_files = list(FIGURES_GENERATED.rglob("*.json"))

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Check for synthetic markers
            if data.get("_synthetic_warning") is True:
                errors.append(
                    f"ISOLATION VIOLATION: Synthetic data in production directory\n"
                    f"  File: {json_file}\n"
                    f"  FIX: Move to figures/synthetic/ or regenerate with FOUNDATION_PLR_SYNTHETIC=1"
                )

            if data.get("_data_source") == "synthetic":
                errors.append(
                    f"ISOLATION VIOLATION: Synthetic data source in production directory\n"
                    f"  File: {json_file}\n"
                    f"  FIX: Move to figures/synthetic/ or regenerate"
                )

        except json.JSONDecodeError:
            pass  # Skip invalid JSON
        except Exception as e:
            print(f"  Warning: Could not check {json_file}: {e}")

    # Check for obvious synthetic naming in files
    for file in FIGURES_GENERATED.rglob("*"):
        if file.is_file():
            name_lower = file.name.lower()
            if name_lower.startswith("synth_") or name_lower.startswith("synthetic_"):
                errors.append(
                    f"SUSPICIOUS: Potentially synthetic file in production directory\n"
                    f"  File: {file}\n"
                    f"  FIX: Move to figures/synthetic/ if this is synthetic data"
                )

    return errors


def check_synthetic_json_has_warnings() -> list[str]:
    """Check that synthetic JSON files have proper warning metadata."""
    errors = []

    if not FIGURES_SYNTHETIC.exists():
        return errors  # No synthetic figures yet

    json_files = list(FIGURES_SYNTHETIC.rglob("*.json"))

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Check for required synthetic markers
            if data.get("_synthetic_warning") is not True:
                errors.append(
                    f"MISSING MARKER: Synthetic JSON lacks _synthetic_warning\n"
                    f"  File: {json_file}\n"
                    f"  FIX: Regenerate with synthetic=True or add _synthetic_warning: true"
                )

            if data.get("_do_not_publish") is not True:
                errors.append(
                    f"MISSING MARKER: Synthetic JSON lacks _do_not_publish\n"
                    f"  File: {json_file}\n"
                    f"  FIX: Add _do_not_publish: true to prevent accidental publication"
                )

        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"  Warning: Could not check {json_file}: {e}")

    return errors


def check_gitignore_patterns() -> list[str]:
    """Check that .gitignore includes synthetic directories."""
    errors = []

    gitignore = PROJECT_ROOT / ".gitignore"

    if not gitignore.exists():
        errors.append(".gitignore not found - synthetic directories may be committed!")
        return errors

    content = gitignore.read_text()

    required_patterns = [
        "figures/synthetic",
        "outputs/synthetic",
    ]

    for pattern in required_patterns:
        if pattern not in content:
            errors.append(
                f"MISSING GITIGNORE: Pattern '{pattern}' not in .gitignore\n"
                f"  FIX: Add '{pattern}/' to .gitignore"
            )

    return errors


def main() -> int:
    """Run all figure isolation checks."""
    print("=" * 60)
    print("Figure Isolation Check")
    print("=" * 60)

    all_errors = []

    # Run all checks
    print("\n[1/4] Checking save_figure() supports synthetic mode...")
    errors = check_save_figure_supports_synthetic()
    if errors:
        all_errors.extend(errors)
    else:
        print("  ✓ save_figure() supports synthetic mode")

    print("\n[2/4] Checking figures/generated/ for synthetic contamination...")
    errors = check_no_synthetic_in_generated()
    if errors:
        all_errors.extend(errors)
    else:
        print("  ✓ No synthetic data in generated directory")

    print("\n[3/4] Checking synthetic JSON files have warnings...")
    errors = check_synthetic_json_has_warnings()
    if errors:
        all_errors.extend(errors)
    else:
        print("  ✓ Synthetic JSON files properly marked")

    print("\n[4/4] Checking .gitignore includes synthetic patterns...")
    errors = check_gitignore_patterns()
    if errors:
        all_errors.extend(errors)
    else:
        print("  ✓ Gitignore patterns present")

    # Report results
    if all_errors:
        print("\n" + "=" * 60)
        print("FIGURE ISOLATION VIOLATIONS DETECTED")
        print("=" * 60)
        for error in all_errors:
            print(f"\n❌ {error}")
        print("\n" + "=" * 60)
        return 1

    print("\n" + "=" * 60)
    print("✓ All figure isolation checks passed")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
