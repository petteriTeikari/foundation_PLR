#!/usr/bin/env python3
"""
Deliverables Verification Script

Prevents the "partial execution catastrophe" where infrastructure is built
but 0 actual figures are delivered.

Addresses: GAP-15 from reproducibility-synthesis-double-check.md

Usage:
    python scripts/verify_deliverables.py                    # Check all registered figures
    python scripts/verify_deliverables.py --check fig_name   # Check specific figure
    python scripts/verify_deliverables.py --registry         # Show registered figures
    python scripts/verify_deliverables.py --generated        # Show generated figures

Exit codes:
    0: All requested figures exist
    1: Some figures missing
    2: Configuration error
"""

import argparse
import sys
from pathlib import Path

import yaml


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


def load_figure_registry() -> dict:
    """Load figure registry from YAML."""
    registry_path = PROJECT_ROOT / "configs/VISUALIZATION/figure_registry.yaml"
    if not registry_path.exists():
        print(f"ERROR: figure_registry.yaml not found at {registry_path}")
        sys.exit(2)

    with open(registry_path) as f:
        return yaml.safe_load(f)


def get_registered_figures() -> list[str]:
    """Get list of all registered figure names."""
    registry = load_figure_registry()

    figures = []

    # R figures
    r_figures = registry.get("r_figures", {})
    figures.extend(r_figures.keys())

    # Python figures (if any)
    py_figures = registry.get("python_figures", {})
    figures.extend(py_figures.keys())

    return sorted(set(figures))


def find_figure(fig_name: str) -> list[Path]:
    """Find all instances of a figure in generated directories."""
    search_dirs = [
        PROJECT_ROOT / "figures/generated",
    ]

    found = []
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        # Search for PNG and PDF
        for ext in ["*.png", "*.pdf"]:
            for path in search_dir.rglob(f"{fig_name}{ext[1:]}"):
                found.append(path)

    return found


def verify_figures(figure_names: list[str]) -> tuple[list[str], list[str]]:
    """
    Verify that specified figures exist.

    Returns:
        (found_figures, missing_figures)
    """
    found = []
    missing = []

    for fig_name in figure_names:
        paths = find_figure(fig_name)
        if paths:
            found.append(fig_name)
        else:
            missing.append(fig_name)

    return found, missing


def verify_deliverables(expected: list[str] | None = None) -> bool:
    """
    Main verification function.

    Args:
        expected: List of expected figure names. If None, checks all registered.

    Returns:
        True if all figures exist, False otherwise.
    """
    if expected is None:
        expected = get_registered_figures()

    found, missing = verify_figures(expected)

    print("Deliverables Verification Report")
    print("=" * 50)
    print(f"Expected: {len(expected)} figures")
    print(f"Found:    {len(found)} figures")
    print(f"Missing:  {len(missing)} figures")
    print()

    if missing:
        print("MISSING FIGURES:")
        for fig in missing[:20]:  # Show first 20
            print(f"  - {fig}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        print()
        print("CRITICAL: Not all deliverables present!")
        return False

    print("SUCCESS: All deliverables verified!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Verify figure deliverables exist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--check",
        nargs="+",
        help="Check specific figure names",
    )
    parser.add_argument(
        "--registry",
        action="store_true",
        help="List all registered figures",
    )
    parser.add_argument(
        "--generated",
        action="store_true",
        help="List all generated figures",
    )
    parser.add_argument(
        "--check-registry",
        action="store_true",
        help="Check all figures in registry exist",
    )
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="Only check main figures (not supplementary)",
    )

    args = parser.parse_args()

    if args.registry:
        print("Registered Figures:")
        print("-" * 30)
        for fig in get_registered_figures():
            print(f"  {fig}")
        return 0

    if args.generated:
        print("Generated Figures:")
        print("-" * 30)
        gen_dir = PROJECT_ROOT / "figures/generated"
        if gen_dir.exists():
            for png in sorted(gen_dir.rglob("*.png")):
                print(f"  {png.relative_to(gen_dir)}")
        return 0

    if args.check:
        success = verify_deliverables(args.check)
        return 0 if success else 1

    if args.check_registry or args.main_only:
        # Check subset of registry
        registry = load_figure_registry()
        r_figures = registry.get("r_figures", {})

        if args.main_only:
            # Filter to main figures only
            expected = [
                name
                for name, config in r_figures.items()
                if config.get("category") == "main"
                or "main" in str(config.get("generation", {}).get("output_dir", ""))
            ]
        else:
            expected = list(r_figures.keys())

        success = verify_deliverables(expected)
        return 0 if success else 1

    # Default: check all registered
    success = verify_deliverables()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
