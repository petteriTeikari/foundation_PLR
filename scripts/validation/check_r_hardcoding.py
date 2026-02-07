#!/usr/bin/env python3
"""
Pre-commit hook to detect hardcoded values in R figure scripts.

This script enforces the project's anti-hardcoding rules for R files:
1. No hardcoded hex colors (must use color_defs from YAML)
2. No ggsave() calls (must use save_publication_figure())
3. No custom theme definitions (must use theme_foundation_plr())
4. No hardcoded dimensions in save calls

Usage:
    python scripts/check_r_hardcoding.py [file1.R file2.R ...]

Exit codes:
    0: All checks passed
    1: Violations found
"""

import re
import sys
from pathlib import Path

# Patterns to detect violations
VIOLATIONS = {
    "hardcoded_hex_color": {
        "pattern": r'color\s*=\s*"#[0-9A-Fa-f]{6}"',
        "message": 'Hardcoded hex color found. Use color_defs[["--color-xxx"]] instead.',
        "severity": "CRITICAL",
    },
    "hardcoded_hex_fill": {
        "pattern": r'fill\s*=\s*"#[0-9A-Fa-f]{6}"',
        "message": 'Hardcoded hex fill found. Use color_defs[["--color-xxx"]] instead.',
        "severity": "CRITICAL",
    },
    "ggsave_usage": {
        "pattern": r"\bggsave\s*\(",
        "message": "ggsave() used. Use save_publication_figure() instead.",
        "severity": "CRITICAL",
    },
    "custom_theme_definition": {
        "pattern": r"(\w+_theme|theme_\w+)\s*<-\s*function\s*\(",
        "message": "Custom theme function defined. Use theme_foundation_plr() instead.",
        "severity": "WARNING",
        "exclude_pattern": r"theme_foundation_plr",  # Allow the official theme
    },
    "hardcoded_dpi": {
        "pattern": r"dpi\s*=\s*\d+",
        "message": "Hardcoded DPI. Load from figure_layouts.yaml instead.",
        "severity": "WARNING",
    },
    "raw_color_names": {
        # Common color names that should use palette
        "pattern": r'color\s*=\s*"(gray\d*|grey\d*|black|white|red|blue|green)"',
        "message": 'Raw color name used. Use color_defs[["--color-xxx"]] instead.',
        "severity": "WARNING",
    },
    "hardcoded_dimensions": {
        # Dimensions in save_publication_figure() should come from YAML
        "pattern": r"save_publication_figure\s*\([^)]*\b(width|height)\s*=\s*\d+",
        "message": "Hardcoded dimensions in save_publication_figure(). Dimensions should come from figure_registry.yaml.",
        "severity": "CRITICAL",
    },
}

# Files/patterns to exclude from checking
EXCLUDE_PATTERNS = [
    r"_TEMPLATE\.R$",  # Template files can have examples
    r"test_.*\.R$",  # Test files may need hardcoded values
    r"common\.R$",  # Utility files
    r"config_loader\.R$",  # Config system itself
    r"color_palettes\.R$",  # Color definitions file
    r"theme_foundation_plr\.R$",  # Theme definition file
    r"save_figure\.R$",  # Figure system implementation (uses ggsave internally)
    r"figure_system/.*\.R$",  # All figure system infrastructure files
]


def should_exclude(filepath: str) -> bool:
    """Check if file should be excluded from validation."""
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, filepath):
            return True
    return False


def check_file(filepath: Path) -> list[dict]:
    """Check a single R file for violations."""
    if should_exclude(str(filepath)):
        return []

    violations = []
    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return [
            {"line": 0, "type": "read_error", "message": str(e), "severity": "ERROR"}
        ]

    lines = content.split("\n")

    for line_num, line in enumerate(lines, 1):
        # Skip comments
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        for violation_type, config in VIOLATIONS.items():
            pattern = config["pattern"]
            exclude = config.get("exclude_pattern")

            matches = re.finditer(pattern, line, re.IGNORECASE)
            for match in matches:
                # Check if this match should be excluded
                if exclude and re.search(exclude, line, re.IGNORECASE):
                    continue

                violations.append(
                    {
                        "line": line_num,
                        "type": violation_type,
                        "message": config["message"],
                        "severity": config["severity"],
                        "match": match.group(0),
                        "context": line.strip()[:80],
                    }
                )

    return violations


def format_violation(filepath: Path, violation: dict) -> str:
    """Format a violation for display."""
    severity = violation["severity"]
    line = violation["line"]
    msg = violation["message"]
    match = violation.get("match", "")
    context = violation.get("context", "")

    if severity == "CRITICAL":
        prefix = "❌ CRITICAL"
    elif severity == "WARNING":
        prefix = "⚠️  WARNING"
    else:
        prefix = "ℹ️  INFO"

    return f"{prefix}: {filepath}:{line}\n    {msg}\n    Found: {match}\n    Line: {context}\n"


def main(files: list[str]) -> int:
    """Main entry point."""
    all_violations = []

    for filepath_str in files:
        filepath = Path(filepath_str)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            continue

        if not filepath.suffix.lower() == ".r":
            continue

        violations = check_file(filepath)
        for v in violations:
            all_violations.append((filepath, v))

    # Report results
    if not all_violations:
        if files:
            print("✅ R hardcoding check passed - no violations found")
        return 0

    # Group by severity
    critical = [(f, v) for f, v in all_violations if v["severity"] == "CRITICAL"]
    warnings = [(f, v) for f, v in all_violations if v["severity"] == "WARNING"]

    print("=" * 70)
    print("R HARDCODING VIOLATIONS DETECTED")
    print("=" * 70)
    print()

    if critical:
        print(f"CRITICAL VIOLATIONS ({len(critical)}):")
        print("-" * 40)
        for filepath, v in critical:
            print(format_violation(filepath, v))

    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        print("-" * 40)
        for filepath, v in warnings:
            print(format_violation(filepath, v))

    print("=" * 70)
    print("HOW TO FIX:")
    print("=" * 70)
    print(
        """
1. COLORS: Replace hardcoded hex with YAML lookup:
   WRONG:  color = "#006BA2"
   RIGHT:  color = color_defs[["--color-primary"]]

2. SAVE: Replace ggsave() with save_publication_figure():
   WRONG:  ggsave("fig.png", p, width = 10, dpi = 150)
   RIGHT:  save_publication_figure(p, "fig_name")

3. THEME: Use the project theme:
   WRONG:  my_theme <- function() { theme_minimal() + ... }
   RIGHT:  source("src/r/theme_foundation_plr.R")
           p + theme_foundation_plr()

See: .claude/docs/meta-learnings/CRITICAL-FAILURE-004-r-figure-hardcoding.md
"""
    )

    # Fail on critical violations only
    if critical:
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # If no args, check all R files in src/r/figures/
        r_files = list(Path("src/r/figures").glob("*.R"))
        sys.exit(main([str(f) for f in r_files]))
    else:
        sys.exit(main(sys.argv[1:]))
