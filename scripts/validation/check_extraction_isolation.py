#!/usr/bin/env python3
"""
Pre-commit hook: Extraction Isolation Check

Verifies that production extraction code properly rejects synthetic runs.
Part of the 4-gate isolation architecture.

This hook checks:
1. Production extraction script imports data_mode validation
2. Production extraction script has synthetic rejection logic
3. No hardcoded 'synth_' or '__SYNTHETIC_' acceptance logic

Exit codes:
- 0: All checks pass
- 1: Isolation violation detected

Usage:
    python scripts/validation/check_extraction_isolation.py
    # Or via pre-commit hook
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def check_extraction_imports_validation() -> list[str]:
    """Check that extraction script imports validation utilities."""
    errors = []

    extraction_script = (
        PROJECT_ROOT / "scripts" / "extraction" / "extract_all_configs_to_duckdb.py"
    )

    if not extraction_script.exists():
        errors.append(f"Production extraction script not found: {extraction_script}")
        return errors

    content = extraction_script.read_text()

    # Check for data_mode import
    if "from src.utils.data_mode import" not in content:
        errors.append(
            f"ISOLATION VIOLATION: {extraction_script.name} does not import from src.utils.data_mode\n"
            "  FIX: Add 'from src.utils.data_mode import validate_run_for_production_extraction'"
        )

    # Check for synthetic validation function usage
    validation_patterns = [
        r"validate_run_for_production_extraction",
        r"is_synthetic_run_name",
        r"is_synthetic_experiment_name",
    ]

    has_validation = any(re.search(p, content) for p in validation_patterns)

    if not has_validation:
        errors.append(
            f"ISOLATION VIOLATION: {extraction_script.name} doesn't use synthetic validation\n"
            "  FIX: Add synthetic run rejection logic using data_mode utilities"
        )

    return errors


def check_no_synthetic_acceptance() -> list[str]:
    """Check that extraction doesn't have logic to ACCEPT synthetic runs."""
    errors = []

    extraction_script = (
        PROJECT_ROOT / "scripts" / "extraction" / "extract_all_configs_to_duckdb.py"
    )

    if not extraction_script.exists():
        return errors  # Already reported above

    content = extraction_script.read_text()

    # Patterns that would indicate accepting synthetic data
    acceptance_patterns = [
        # Don't match comments or string literals
        (r"^\s*include_synthetic\s*=\s*True", "include_synthetic=True"),
        (r"^\s*accept_synthetic\s*=\s*True", "accept_synthetic=True"),
    ]

    for pattern, description in acceptance_patterns:
        if re.search(pattern, content, re.MULTILINE):
            errors.append(
                f"ISOLATION VIOLATION: {extraction_script.name} has synthetic acceptance: {description}\n"
                "  Production extraction must NEVER include synthetic runs."
            )

    return errors


def check_synthetic_script_exists() -> list[str]:
    """Check that dedicated synthetic extraction script exists."""
    errors = []

    synthetic_script = (
        PROJECT_ROOT / "scripts" / "extraction" / "extract_synthetic_to_duckdb.py"
    )

    if not synthetic_script.exists():
        errors.append(
            f"MISSING: Dedicated synthetic extraction script\n"
            f"  Expected: {synthetic_script}\n"
            "  FIX: Create scripts/extraction/extract_synthetic_to_duckdb.py for synthetic data"
        )
        return errors

    content = synthetic_script.read_text()

    # Check that it explicitly handles synthetic data
    if "is_synthetic" not in content.lower() and "synthetic" not in content.lower():
        errors.append(
            f"SUSPICIOUS: {synthetic_script.name} doesn't mention 'synthetic'\n"
            "  The synthetic extraction script should explicitly handle synthetic runs"
        )

    return errors


def check_output_paths_separated() -> list[str]:
    """Check that output paths are properly separated."""
    errors = []

    extraction_script = (
        PROJECT_ROOT / "scripts" / "extraction" / "extract_all_configs_to_duckdb.py"
    )
    synthetic_script = (
        PROJECT_ROOT / "scripts" / "extraction" / "extract_synthetic_to_duckdb.py"
    )

    scripts = []
    if extraction_script.exists():
        scripts.append(("production", extraction_script))
    if synthetic_script.exists():
        scripts.append(("synthetic", synthetic_script))

    for script_type, script_path in scripts:
        content = script_path.read_text()

        if script_type == "production":
            # Production should NOT output to synthetic directory
            if "outputs/synthetic" in content and "reject" not in content.lower():
                errors.append(
                    f"ISOLATION VIOLATION: {script_path.name} outputs to synthetic directory\n"
                    "  Production extraction must output to outputs/ not outputs/synthetic/"
                )

        if script_type == "synthetic":
            # Synthetic should output to synthetic directory
            if (
                "outputs/synthetic" not in content
                and "get_synthetic_output_dir" not in content
            ):
                errors.append(
                    f"SUSPICIOUS: {script_path.name} may not output to synthetic directory\n"
                    "  Synthetic extraction should use outputs/synthetic/"
                )

    return errors


def main() -> int:
    """Run all isolation checks."""
    print("=" * 60)
    print("Extraction Isolation Check")
    print("=" * 60)

    all_errors = []

    # Run all checks
    print("\n[1/4] Checking extraction imports validation utilities...")
    errors = check_extraction_imports_validation()
    if errors:
        all_errors.extend(errors)
    else:
        print("  ✓ Validation utilities imported")

    print("\n[2/4] Checking for synthetic acceptance logic...")
    errors = check_no_synthetic_acceptance()
    if errors:
        all_errors.extend(errors)
    else:
        print("  ✓ No synthetic acceptance found")

    print("\n[3/4] Checking synthetic extraction script exists...")
    errors = check_synthetic_script_exists()
    if errors:
        all_errors.extend(errors)
    else:
        print("  ✓ Synthetic extraction script exists")

    print("\n[4/4] Checking output path separation...")
    errors = check_output_paths_separated()
    if errors:
        all_errors.extend(errors)
    else:
        print("  ✓ Output paths properly separated")

    # Report results
    if all_errors:
        print("\n" + "=" * 60)
        print("EXTRACTION ISOLATION VIOLATIONS DETECTED")
        print("=" * 60)
        for error in all_errors:
            print(f"\n❌ {error}")
        print("\n" + "=" * 60)
        return 1

    print("\n" + "=" * 60)
    print("✓ All extraction isolation checks passed")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
