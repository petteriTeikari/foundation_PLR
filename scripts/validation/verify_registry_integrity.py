#!/usr/bin/env python3
"""
Registry Integrity Verification - Anti-Cheat Quality Gate.

This script verifies that:
1. The canary file counts match the registry YAML counts
2. The registry YAML checksum matches the canary's expected checksum
3. The registry module constants match the canary counts
4. Test file hardcoded values match canary counts and invalid methods

PURPOSE: Prevent Claude Code from temporarily modifying tests to make them pass.
         By requiring FOUR separate files to all agree, any "cheating" attempt
         will be caught because Claude would need to modify all four consistently.

USAGE:
    python scripts/verify_registry_integrity.py          # Verify all
    python scripts/verify_registry_integrity.py --ci     # CI mode (stricter)
    python scripts/verify_registry_integrity.py --fix    # Show what needs fixing

EXIT CODES:
    0 = All integrity checks pass
    1 = Integrity violation detected (CI should FAIL)
    2 = File not found or parse error
"""

import argparse
import ast
import hashlib
import sys
from pathlib import Path

import yaml

# Paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
CANARY_FILE = PROJECT_ROOT / "configs" / "registry_canary.yaml"
REGISTRY_FILE = (
    PROJECT_ROOT / "configs" / "mlflow_registry" / "parameters" / "classification.yaml"
)
REGISTRY_MODULE = PROJECT_ROOT / "src" / "data_io" / "registry.py"
TEST_FILE = PROJECT_ROOT / "tests" / "test_registry.py"


def load_yaml(path: Path) -> dict:
    """Load YAML file with error handling."""
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}")


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hash of file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_constants_from_registry_module(path: Path) -> dict:
    """
    Extract EXPECTED_*_COUNT constants from registry.py using AST parsing.

    Uses Python's ast module for robust parsing - regex is BANNED for code analysis.
    This correctly handles all edge cases (comments, strings, nested scopes).
    """
    content = path.read_text(encoding="utf-8")
    tree = ast.parse(content, filename=str(path))

    constants = {
        "outlier_methods": None,
        "imputation_methods": None,
        "classifiers": None,
    }

    # Map constant names to result keys
    name_map = {
        "EXPECTED_OUTLIER_COUNT": "outlier_methods",
        "EXPECTED_IMPUTATION_COUNT": "imputation_methods",
        "EXPECTED_CLASSIFIER_COUNT": "classifiers",
    }

    # Walk only top-level assignments (not nested in functions/classes)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in name_map:
                    # Extract the integer value
                    if isinstance(node.value, ast.Constant) and isinstance(
                        node.value.value, int
                    ):
                        constants[name_map[target.id]] = node.value.value

    return constants


def extract_invalid_methods_from_tests(path: Path) -> dict:
    """
    Extract INVALID_*_METHODS lists from test file using AST parsing.

    Uses Python's ast module for robust parsing - regex is BANNED for code analysis.
    """
    content = path.read_text(encoding="utf-8")
    tree = ast.parse(content, filename=str(path))

    result = {"outlier": []}

    # Walk only top-level assignments
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "INVALID_OUTLIER_METHODS"
                ):
                    # Extract list elements
                    if isinstance(node.value, ast.List):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(
                                elt.value, str
                            ):
                                result["outlier"].append(elt.value)

    return result


def verify_integrity(_ci_mode: bool = False) -> tuple[bool, list[str]]:
    """
    Verify registry integrity across all sources.

    Returns (success, list_of_errors).
    """
    errors = []

    # Check files exist
    for path, name in [
        (CANARY_FILE, "Canary file"),
        (REGISTRY_FILE, "Registry YAML"),
        (REGISTRY_MODULE, "Registry module"),
        (TEST_FILE, "Test file"),
    ]:
        if not path.exists():
            errors.append(f"FILE NOT FOUND: {name} at {path}")

    if errors:
        return False, errors

    # Load all sources
    try:
        canary = load_yaml(CANARY_FILE)
    except ValueError as e:
        return False, [f"PARSE ERROR: {e}"]

    try:
        registry = load_yaml(REGISTRY_FILE)
    except ValueError as e:
        return False, [f"PARSE ERROR: {e}"]

    module_constants = extract_constants_from_registry_module(REGISTRY_MODULE)
    test_invalid_methods = extract_invalid_methods_from_tests(TEST_FILE)

    # Get values from registry YAML
    registry_outliers = registry["pipeline"]["anomaly_source"]["values"]
    registry_imputations = registry["pipeline"]["imputation_source"]["values"]
    registry_classifiers = registry["pipeline"]["model_name"]["values"]

    # Get values from canary
    canary_counts = canary["expected_counts"]
    canary_invalid_outliers = canary.get("invalid_methods", {}).get("outlier", [])
    expected_checksum = canary.get("source_checksums", {}).get("classification_yaml")

    # VERIFICATION 1: Registry checksum matches canary expectation
    if expected_checksum and expected_checksum != "TO_BE_COMPUTED":
        actual_checksum = compute_sha256(REGISTRY_FILE)
        if actual_checksum != expected_checksum:
            errors.append(
                f"CHECKSUM MISMATCH: Registry YAML has changed!\n"
                f"  Expected: {expected_checksum}\n"
                f"  Actual:   {actual_checksum}\n"
                f"  Update canary file with new checksum if change was intentional."
            )

    # VERIFICATION 2: Canary counts match registry counts
    if len(registry_outliers) != canary_counts["outlier_methods"]:
        errors.append(
            f"CANARY/REGISTRY MISMATCH: Canary expects {canary_counts['outlier_methods']} outliers, "
            f"registry has {len(registry_outliers)}"
        )
    if len(registry_imputations) != canary_counts["imputation_methods"]:
        errors.append(
            f"CANARY/REGISTRY MISMATCH: Canary expects {canary_counts['imputation_methods']} imputations, "
            f"registry has {len(registry_imputations)}"
        )
    if len(registry_classifiers) != canary_counts["classifiers"]:
        errors.append(
            f"CANARY/REGISTRY MISMATCH: Canary expects {canary_counts['classifiers']} classifiers, "
            f"registry has {len(registry_classifiers)}"
        )

    # VERIFICATION 3: Module constants match canary
    if module_constants["outlier_methods"] is None:
        errors.append("PARSE ERROR: EXPECTED_OUTLIER_COUNT not found in registry.py")
    elif module_constants["outlier_methods"] != canary_counts["outlier_methods"]:
        errors.append(
            f"MODULE/CANARY MISMATCH: registry.py EXPECTED_OUTLIER_COUNT={module_constants['outlier_methods']}, "
            f"canary expects {canary_counts['outlier_methods']}"
        )

    if module_constants["imputation_methods"] is None:
        errors.append("PARSE ERROR: EXPECTED_IMPUTATION_COUNT not found in registry.py")
    elif module_constants["imputation_methods"] != canary_counts["imputation_methods"]:
        errors.append(
            f"MODULE/CANARY MISMATCH: registry.py EXPECTED_IMPUTATION_COUNT={module_constants['imputation_methods']}, "
            f"canary expects {canary_counts['imputation_methods']}"
        )

    if module_constants["classifiers"] is None:
        errors.append("PARSE ERROR: EXPECTED_CLASSIFIER_COUNT not found in registry.py")
    elif module_constants["classifiers"] != canary_counts["classifiers"]:
        errors.append(
            f"MODULE/CANARY MISMATCH: registry.py EXPECTED_CLASSIFIER_COUNT={module_constants['classifiers']}, "
            f"canary expects {canary_counts['classifiers']}"
        )

    # VERIFICATION 4: Invalid methods in tests match canary
    if canary_invalid_outliers and test_invalid_methods.get("outlier"):
        canary_set = set(canary_invalid_outliers)
        test_set = set(test_invalid_methods["outlier"])
        if canary_set != test_set:
            missing_in_test = canary_set - test_set
            extra_in_test = test_set - canary_set
            if missing_in_test:
                errors.append(
                    f"TEST/CANARY MISMATCH: Invalid outlier methods in canary but not in tests: {missing_in_test}"
                )
            if extra_in_test:
                errors.append(
                    f"TEST/CANARY MISMATCH: Invalid outlier methods in tests but not in canary: {extra_in_test}"
                )

    # VERIFICATION 5: Invalid methods not in registry
    for invalid_method in canary_invalid_outliers:
        if invalid_method in registry_outliers:
            errors.append(
                f"REGISTRY CONTAMINATION: Invalid method '{invalid_method}' found in registry! "
                f"This should have been removed."
            )

    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="Verify registry integrity")
    parser.add_argument("--ci", action="store_true", help="CI mode (stricter)")
    parser.add_argument("--fix", action="store_true", help="Show what needs fixing")
    args = parser.parse_args()

    print("=" * 60)
    print("Registry Integrity Verification")
    print("=" * 60)
    print()
    print("Checking consistency across:")
    print(f"  1. Canary file:     {CANARY_FILE.relative_to(PROJECT_ROOT)}")
    print(f"  2. Registry YAML:   {REGISTRY_FILE.relative_to(PROJECT_ROOT)}")
    print(f"  3. Registry module: {REGISTRY_MODULE.relative_to(PROJECT_ROOT)}")
    print(f"  4. Test file:       {TEST_FILE.relative_to(PROJECT_ROOT)}")
    print()

    success, errors = verify_integrity(_ci_mode=args.ci)

    if success:
        print("✓ ALL INTEGRITY CHECKS PASSED")
        print()
        canary = load_yaml(CANARY_FILE)
        print("Summary:")
        print(f"  - Outlier methods:    {canary['expected_counts']['outlier_methods']}")
        print(
            f"  - Imputation methods: {canary['expected_counts']['imputation_methods']}"
        )
        print(f"  - Classifiers:        {canary['expected_counts']['classifiers']}")

        # Show checksum status
        expected_checksum = canary.get("source_checksums", {}).get(
            "classification_yaml", ""
        )
        if expected_checksum and expected_checksum != "TO_BE_COMPUTED":
            print("  - Checksum verified:  ✓")
        return 0
    else:
        print("✗ INTEGRITY VIOLATIONS DETECTED")
        print()
        for error in errors:
            print(f"  ERROR: {error}")
        print()

        if args.fix:
            print("To fix these issues:")
            print(
                "  1. Ensure configs/mlflow_registry/parameters/classification.yaml is correct"
            )
            print("  2. Update configs/registry_canary.yaml counts and checksum")
            print("  3. Update src/data_io/registry.py EXPECTED_*_COUNT constants")
            print("  4. Update tests/test_registry.py INVALID_OUTLIER_METHODS list")
            print("  5. Commit ALL changes together")
            print()
            print("  To regenerate checksum:")
            print(
                "    sha256sum configs/mlflow_registry/parameters/classification.yaml"
            )

        # Return 2 for parse errors, 1 for integrity violations
        if any("PARSE ERROR" in e or "FILE NOT FOUND" in e for e in errors):
            return 2
        return 1


if __name__ == "__main__":
    sys.exit(main())
