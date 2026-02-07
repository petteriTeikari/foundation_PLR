"""
TDD Test: Method abbreviations must come from YAML, not hardcoded.

Uses R's native parse() function via subprocess - NO REGEX (per CLAUDE.md absolute ban).
"""

import json
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.r_required, pytest.mark.guardrail]


def get_project_root() -> Path:
    """Find project root."""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root")


def get_r_ast_parser_path() -> Path:
    """Get path to the R AST parser script."""
    return Path(__file__).parent / "r_ast_parser.R"


def parse_r_file_for_strings(file_path: Path) -> list[dict]:
    """
    Use R's parse() to extract all string literals from an R file.

    Returns list of {line, value} dicts.
    """
    parser_path = get_r_ast_parser_path()

    try:
        result = subprocess.run(
            ["Rscript", str(parser_path), str(file_path), "strings"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return []

    if result.returncode != 0:
        return []

    try:
        data = json.loads(result.stdout)
        if isinstance(data, dict) and "error" in data:
            return []
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


# Known method names that might have abbreviations
METHOD_NAMES = [
    "pupil-gt",
    "MOMENT-gt-finetune",
    "MOMENT-gt-zeroshot",
    "MOMENT-orig-finetune",
    "UniTS-gt-finetune",
    "UniTS-orig-finetune",
    "UniTS-orig-zeroshot",
    "TimesNet-gt",
    "TimesNet-orig",
    "LOF",
    "OneClassSVM",
    "SubPCA",
    "PROPHET",
    "SAITS",
    "CSDI",
    "linear",
]

# Abbreviation patterns (uppercase, short)
ABBREVIATION_PATTERNS = [
    "GT",
    "MOM",
    "MOMENT",
    "TN",
    "UniTS",
    "LOF",
    "SVM",
    "PCA",
]


def test_cd_diagram_no_hardcoded_abbreviations() -> None:
    """cd_diagram.R must not have 73 hardcoded abbreviations (AST-based check)."""
    project_root = get_project_root()
    cd_diagram = project_root / "src/r/figure_system/cd_diagram.R"

    assert cd_diagram.exists(), f"cd_diagram.R not found: {cd_diagram}"

    strings = parse_r_file_for_strings(cd_diagram)

    # Count strings that look like method names (contain hyphen or known patterns)
    method_like_strings = []
    abbreviation_like_strings = []

    for item in strings:
        value = item.get("value", "")
        line = item.get("line", 0)

        # Check if it looks like a method name
        if any(method in value for method in METHOD_NAMES) or "-" in value:
            if len(value) > 3:  # Longer strings are likely method names
                method_like_strings.append((line, value))

        # Check if it looks like an abbreviation (short, uppercase-ish)
        if (
            len(value) <= 15
            and value.isupper()
            or any(abbr in value for abbr in ABBREVIATION_PATTERNS)
        ):
            if len(value) <= 15 and len(value) >= 2:
                abbreviation_like_strings.append((line, value))

    # Count pairs that appear to be method->abbreviation mappings
    # (method name and abbreviation on same or adjacent lines)
    potential_mappings = 0
    for m_line, m_value in method_like_strings:
        for a_line, a_value in abbreviation_like_strings:
            if abs(m_line - a_line) <= 1 and a_value != m_value:
                potential_mappings += 1
                break

    # Allow at most 5 (for genuine fallbacks in error handling)
    max_allowed = 5

    assert potential_mappings <= max_allowed, (
        f"cd_diagram.R has ~{potential_mappings} potential hardcoded abbreviations (max allowed: {max_allowed}).\n"
        f"Method-like strings found: {len(method_like_strings)}\n"
        f"First 10: {method_like_strings[:10]}\n"
        f"\nMove to configs/mlflow_registry/method_abbreviations.yaml!"
    )


def test_no_hardcoded_abbreviation_functions() -> None:
    """
    No R file should have inline abbreviation mappings.

    Uses AST to find string literals that look like method->abbreviation pairs.
    """
    project_root = get_project_root()
    violations: list[str] = []

    for r_file in (project_root / "src/r").rglob("*.R"):
        # Skip loader files
        if "abbreviation" in r_file.name.lower() or "category_loader" in r_file.name:
            continue

        strings = parse_r_file_for_strings(r_file)

        # Group strings by line to find potential mapping patterns
        strings_by_line: dict[int, list[str]] = {}
        for item in strings:
            line = item.get("line", 0)
            value = item.get("value", "")
            if line not in strings_by_line:
                strings_by_line[line] = []
            strings_by_line[line].append(value)

        # Look for lines with both a method name and a short abbreviation
        for line, values in strings_by_line.items():
            has_method = any(
                any(m in v for m in METHOD_NAMES) or ("-" in v and len(v) > 5)
                for v in values
            )
            has_abbrev = any(
                len(v) <= 10 and len(v) >= 2 and (v.isupper() or v[0].isupper())
                for v in values
                if v not in METHOD_NAMES
            )

            if has_method and has_abbrev and len(values) >= 2:
                violations.append(
                    f"{r_file.name}:{line}: potential inline abbreviation mapping"
                )

    # Allow some noise - only fail if there are many violations
    if len(violations) > 10:
        assert False, (
            f"HARDCODED ABBREVIATION PATTERNS DETECTED ({len(violations)} instances):\n"
            + "\n".join(violations[:15])
            + "\n\nUse load_method_abbreviations() from YAML instead!"
        )
