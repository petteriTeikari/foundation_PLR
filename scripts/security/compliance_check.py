#!/usr/bin/env python3
"""
Foundation PLR - Security Compliance Check

Standalone script (stdlib only) that validates data privacy, secrets hygiene,
and quality gate infrastructure. Designed for CI via GitHub Actions.

Usage:
    uv run python scripts/security/compliance_check.py --validate
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Raw subject ID pattern - only Hxxx/Gxxx codes are allowed in public files
# Excludes Pylint/ruff rule codes (PLR0912, PLR0913, etc.) which use the same prefix
RAW_SUBJECT_ID_RE = re.compile(r"PLR\d{4}")
# Known linter rule codes that match PLR\d{4} but are NOT subject IDs
LINTER_RULE_CODES = frozenset(
    {
        "PLR0904",
        "PLR0911",
        "PLR0912",
        "PLR0913",
        "PLR0914",
        "PLR0915",
        "PLR0916",
        "PLR0917",
        "PLR1702",
        "PLR1704",
        "PLR2004",
        "PLR5501",
        "PLR6201",
        "PLR6301",
    }
)

# Common secret patterns
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AWS Access Key", re.compile(r"AKIA[0-9A-Z]{16}")),
    (
        "Generic API Key",
        re.compile(
            r"""(?:api[_-]?key|apikey)\s*[:=]\s*['"][A-Za-z0-9_\-]{20,}['"]""",
            re.IGNORECASE,
        ),
    ),
    (
        "Generic Secret",
        re.compile(
            r"""(?:secret|password|passwd|pwd)\s*[:=]\s*['"][^'"]{8,}['"]""",
            re.IGNORECASE,
        ),
    ),
    (
        "Private Key Header",
        re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----"),
    ),
    ("GitHub Token", re.compile(r"gh[pousr]_[A-Za-z0-9_]{36,}")),
]

# File extensions to scan
SCAN_EXTENSIONS = {
    ".py",
    ".yaml",
    ".yml",
    ".md",
    ".json",
    ".toml",
    ".cfg",
    ".ini",
    ".sh",
}

# Paths allowed to contain raw PLR IDs (internal data processing, configs, tests, tools)
# These files need original subject codes for data pipeline operations.
# The privacy boundary is enforced by gitignoring subject-level JSON outputs.
PLR_ID_ALLOWLIST_PREFIXES = (
    "configs/",  # Internal config (demo_subjects, registry)
    "src/data_io/",  # Data I/O with re-anonymization
    "src/extraction/",  # MLflow extraction (internal)
    "src/metrics/",  # Metric evaluation (internal)
    "src/orchestration/",  # Pipeline orchestration
    "src/tools/",  # Legacy ground-truth tooling
    "src/preprocessing/",  # Signal preprocessing
    "src/classification/",  # Vendored classifiers (TabPFN) + linter noqa comments
    "tests/",  # Test fixtures
    "scripts/",  # Validation/extraction scripts
    "docs/",  # Internal documentation and figure plans
    ".claude/",  # Claude instructions
)

# Quality gate scripts that must exist
REQUIRED_QUALITY_GATES = [
    "scripts/validation/check_computation_decoupling.py",
    "scripts/validation/verify_registry_integrity.py",
    "scripts/validation/check_r_hardcoding.py",
    "scripts/validation/validate_python_hardcoding.py",
    "scripts/validation/check_extraction_isolation.py",
    "scripts/validation/check_frozen_configs.py",
]

# Gitignore patterns that must be present for privacy
REQUIRED_GITIGNORE_PATTERNS = [
    "data/private/",
    "**/subject_lookup.yaml",
    "**/subject_lookup.json",
    "**/*_lookup_table.*",
    "**/subject_traces*.json",
    "**/individual_*.json",
]


def get_tracked_files() -> list[Path]:
    """Get list of git-tracked files."""
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        return []
    return [PROJECT_ROOT / f for f in result.stdout.strip().splitlines() if f]


def check_data_privacy(tracked_files: list[Path]) -> dict:
    """Scan tracked files for raw subject IDs (PLRxxxx)."""
    findings: list[dict] = []
    files_scanned = 0

    for fpath in tracked_files:
        if fpath.suffix not in SCAN_EXTENSIONS:
            continue
        rel = fpath.relative_to(PROJECT_ROOT)
        rel_str = str(rel)

        # Skip this script itself
        if rel_str.startswith("scripts/security/"):
            continue

        # Files in allowlisted paths may legitimately use internal PLR codes
        is_allowlisted = rel_str.startswith(PLR_ID_ALLOWLIST_PREFIXES)

        files_scanned += 1
        try:
            content = fpath.read_text(errors="replace")
        except OSError:
            continue

        for match in RAW_SUBJECT_ID_RE.finditer(content):
            # Skip known linter rule codes (PLR0912, PLR0913, etc.)
            if match.group() in LINTER_RULE_CODES:
                continue
            line_num = content[: match.start()].count("\n") + 1
            if is_allowlisted:
                # Track as INFO for awareness, but don't fail the check
                severity = "INFO"
                msg = f"Raw subject ID '{match.group()}' in allowlisted internal file"
            else:
                severity = "CRITICAL"
                msg = (
                    f"Raw subject ID '{match.group()}' found - use Hxxx/Gxxx codes only"
                )
            findings.append(
                {
                    "file": rel_str,
                    "line": line_num,
                    "pattern": match.group(),
                    "severity": severity,
                    "message": msg,
                }
            )

    critical = [f for f in findings if f["severity"] == "CRITICAL"]
    return {
        "check": "data_privacy",
        "description": "Scan for raw subject IDs (PLRxxxx) in public-facing files",
        "framework": "GDPR / Data Privacy",
        "passed": len(critical) == 0,
        "files_scanned": files_scanned,
        "findings": critical,  # Only report critical (non-allowlisted) findings
        "info_count": len(findings) - len(critical),  # Count of allowlisted occurrences
    }


def check_private_data_isolation() -> dict:
    """Verify data/private/ is gitignored and no private patterns are tracked."""
    findings: list[dict] = []

    # Check that data/private/ directory contents are not tracked
    result = subprocess.run(
        ["git", "ls-files", "data/private/"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    tracked_private = [f for f in result.stdout.strip().splitlines() if f]
    if tracked_private:
        for f in tracked_private:
            findings.append(
                {
                    "file": f,
                    "severity": "CRITICAL",
                    "message": "File in data/private/ is tracked by git",
                }
            )

    # Check that subject lookup files are not tracked
    for pattern in ["**/subject_lookup.yaml", "**/subject_lookup.json"]:
        result = subprocess.run(
            ["git", "ls-files", pattern],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        tracked = [f for f in result.stdout.strip().splitlines() if f]
        for f in tracked:
            findings.append(
                {
                    "file": f,
                    "severity": "CRITICAL",
                    "message": "Subject lookup file is tracked by git",
                }
            )

    return {
        "check": "private_data_isolation",
        "description": "Verify data/private/ is gitignored and no private patterns tracked",
        "framework": "GDPR / Data Privacy",
        "passed": len(findings) == 0,
        "findings": findings,
    }


def check_secrets(tracked_files: list[Path]) -> dict:
    """Scan tracked files for common secret patterns."""
    findings: list[dict] = []
    files_scanned = 0

    for fpath in tracked_files:
        if fpath.suffix not in SCAN_EXTENSIONS:
            continue
        # Skip this script
        rel = fpath.relative_to(PROJECT_ROOT)
        if str(rel).startswith("scripts/security/"):
            continue

        files_scanned += 1
        try:
            content = fpath.read_text(errors="replace")
        except OSError:
            continue

        for name, pattern in SECRET_PATTERNS:
            for match in pattern.finditer(content):
                line_num = content[: match.start()].count("\n") + 1
                findings.append(
                    {
                        "file": str(rel),
                        "line": line_num,
                        "pattern_name": name,
                        "severity": "CRITICAL",
                        "message": f"Potential {name} detected",
                    }
                )

    return {
        "check": "secrets_scan",
        "description": "Check for API keys, tokens, and passwords in tracked files",
        "framework": "Security Hygiene",
        "passed": len(findings) == 0,
        "files_scanned": files_scanned,
        "findings": findings,
    }


def check_gitignore_coverage() -> dict:
    """Confirm critical privacy patterns exist in .gitignore."""
    findings: list[dict] = []
    gitignore_path = PROJECT_ROOT / ".gitignore"

    if not gitignore_path.exists():
        findings.append(
            {
                "severity": "CRITICAL",
                "message": ".gitignore file not found",
            }
        )
        return {
            "check": "gitignore_coverage",
            "description": "Verify critical privacy patterns in .gitignore",
            "framework": "Defense in Depth",
            "passed": False,
            "findings": findings,
        }

    content = gitignore_path.read_text()

    for pattern in REQUIRED_GITIGNORE_PATTERNS:
        if pattern not in content:
            findings.append(
                {
                    "pattern": pattern,
                    "severity": "HIGH",
                    "message": f"Required gitignore pattern '{pattern}' not found",
                }
            )

    return {
        "check": "gitignore_coverage",
        "description": "Verify critical privacy patterns in .gitignore",
        "framework": "Defense in Depth",
        "passed": len(findings) == 0,
        "findings": findings,
    }


def check_quality_gates() -> dict:
    """Verify key validation scripts exist."""
    findings: list[dict] = []

    for script_path in REQUIRED_QUALITY_GATES:
        full_path = PROJECT_ROOT / script_path
        if not full_path.exists():
            findings.append(
                {
                    "script": script_path,
                    "severity": "HIGH",
                    "message": f"Required quality gate script missing: {script_path}",
                }
            )

    return {
        "check": "quality_gates_present",
        "description": "Verify key validation/compliance scripts exist",
        "framework": "Research Integrity",
        "passed": len(findings) == 0,
        "scripts_checked": len(REQUIRED_QUALITY_GATES),
        "findings": findings,
    }


def run_all_checks() -> dict:
    """Run all compliance checks and return structured report."""
    tracked_files = get_tracked_files()

    checks = [
        check_data_privacy(tracked_files),
        check_private_data_isolation(),
        check_secrets(tracked_files),
        check_gitignore_coverage(),
        check_quality_gates(),
    ]

    total_findings = sum(len(c["findings"]) for c in checks)
    critical_findings = sum(
        1 for c in checks for f in c["findings"] if f.get("severity") == "CRITICAL"
    )
    passed_checks = sum(1 for c in checks if c["passed"])
    total_checks = len(checks)
    compliance_pct = (passed_checks / total_checks) * 100 if total_checks > 0 else 0

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_compliance": compliance_pct,
        "critical_findings": critical_findings,
        "total_findings": total_findings,
        "checks_passed": passed_checks,
        "checks_total": total_checks,
        "frameworks": {
            "data_privacy": {
                "compliance_percentage": compliance_pct,
                "implemented_controls": passed_checks,
                "not_implemented_controls": total_checks - passed_checks,
            },
        },
        "validation_details": {c["check"]: c for c in checks},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Foundation PLR Compliance Check")
    parser.add_argument(
        "--validate", action="store_true", help="Run all compliance checks"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON report to file (default: stdout)",
    )
    args = parser.parse_args()

    if not args.validate:
        parser.print_help()
        return 0

    report = run_all_checks()
    report_json = json.dumps(report, indent=2)

    if args.output:
        args.output.write_text(report_json)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report_json)

    # Log summary to stderr
    print(
        f"\nCompliance: {report['overall_compliance']:.0f}% "
        f"({report['checks_passed']}/{report['checks_total']} checks passed, "
        f"{report['critical_findings']} critical findings)",
        file=sys.stderr,
    )

    return 1 if report["critical_findings"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
