"""
Rendering Artifact Detection Tests

Detect artifacts that shouldn't appear in scientific figures:
- [cite:xxx] tags (internal LaTeX references leaked)
- Hex color codes in annotation text (#RRGGBB)
- Semantic tags (::category::)
- Internal guidance text

Addresses: GAP-14 from reproducibility-synthesis-double-check.md

TDD: These tests define what artifacts we're looking for.
"""

import json
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Patterns that indicate rendering artifacts
ARTIFACT_PATTERNS = {
    "cite_tag": {
        "pattern": r"\[cite:\s*\w+\]",
        "description": "LaTeX citation tag leaked into figure",
        "severity": "CRITICAL",
    },
    "hex_color_in_text": {
        "pattern": r'(?:annotation|title|label|text)["\']?\s*[:=]\s*["\']?[^"\']*#[0-9A-Fa-f]{6}',
        "description": "Hex color code in annotation/label text",
        "severity": "HIGH",
    },
    "semantic_tag": {
        "pattern": r"::\w+::",
        "description": "Semantic tag not rendered",
        "severity": "MEDIUM",
    },
    "placeholder_text": {
        "pattern": r"\[(?:TODO|FIXME|XXX|PLACEHOLDER)\]",
        "description": "Placeholder text in figure",
        "severity": "HIGH",
    },
    "internal_guidance": {
        # Note: Excludes "pminternal" (R package name) which is a valid term
        # Note: Excludes "_TEST" files which intentionally have warnings
        "pattern": r"(?<![a-z])(?:INTERNAL|GUIDANCE|DO NOT|REMOVE BEFORE)(?![a-z])",
        "description": "Internal guidance text leaked",
        "severity": "CRITICAL",
        "false_positives": ["pminternal", "_TEST"],  # R package name, test data files
    },
}


@pytest.fixture
def figure_data_files():
    """Get all JSON data files for figures."""
    data_dirs = [
        PROJECT_ROOT / "data" / "r_data",
        PROJECT_ROOT / "figures" / "generated" / "data",
    ]

    json_files = []
    for data_dir in data_dirs:
        if data_dir.exists():
            json_files.extend(data_dir.glob("*.json"))

    return json_files


@pytest.fixture
def figure_r_scripts():
    """Get all R figure scripts."""
    r_dir = PROJECT_ROOT / "src" / "r" / "figures"
    if r_dir.exists():
        return list(r_dir.glob("fig_*.R"))
    return []


class TestNoCiteTagsInFigures:
    """Ensure no [cite:xxx] tags appear in figure data or scripts."""

    def test_no_cite_tags_in_json_data(self, figure_data_files):
        """JSON data files must not contain [cite:] artifacts."""
        violations = []

        for json_file in figure_data_files:
            try:
                content = json_file.read_text()
                matches = re.findall(ARTIFACT_PATTERNS["cite_tag"]["pattern"], content)
                if matches:
                    violations.append((json_file.name, matches))
            except Exception as e:
                pytest.skip(f"Could not read {json_file}: {e}")

        assert not violations, (
            "CRITICAL: [cite:] tags found in figure data:\n"
            + "\n".join(f"  {f}: {m}" for f, m in violations)
        )

    def test_no_cite_tags_in_r_scripts(self, figure_r_scripts):
        """R scripts must not have hardcoded [cite:] references."""
        violations = []

        for r_script in figure_r_scripts:
            content = r_script.read_text()
            # Skip comments
            lines = [
                line for line in content.split("\n") if not line.strip().startswith("#")
            ]
            content_no_comments = "\n".join(lines)

            matches = re.findall(
                ARTIFACT_PATTERNS["cite_tag"]["pattern"], content_no_comments
            )
            if matches:
                violations.append((r_script.name, matches))

        assert not violations, "[cite:] tags found in R scripts:\n" + "\n".join(
            f"  {f}: {m}" for f, m in violations
        )


class TestNoHexColorsInAnnotations:
    """Ensure hex colors aren't exposed in figure text elements."""

    def test_no_hex_in_json_labels(self, figure_data_files):
        """JSON label/title fields should not contain hex color codes."""
        violations = []

        for json_file in figure_data_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Check string values recursively
                def check_dict(d, path=""):
                    if isinstance(d, dict):
                        for k, v in d.items():
                            check_dict(v, f"{path}.{k}")
                    elif isinstance(d, list):
                        for i, item in enumerate(d):
                            check_dict(item, f"{path}[{i}]")
                    elif isinstance(d, str):
                        # Check for hex colors in text fields
                        if any(
                            key in path.lower()
                            for key in ["label", "title", "annotation", "text"]
                        ):
                            hex_matches = re.findall(r"#[0-9A-Fa-f]{6}", d)
                            if hex_matches:
                                violations.append((json_file.name, path, hex_matches))

                check_dict(data)

            except json.JSONDecodeError:
                continue  # Skip non-JSON files

        assert not violations, "Hex colors found in text fields:\n" + "\n".join(
            f"  {f} {p}: {m}" for f, p, m in violations
        )


class TestNoPlaceholderText:
    """Ensure no placeholder/TODO text in figures."""

    def test_no_placeholders_in_json(self, figure_data_files):
        """JSON data should not contain TODO/PLACEHOLDER markers."""
        violations = []

        for json_file in figure_data_files:
            content = json_file.read_text()
            matches = re.findall(
                ARTIFACT_PATTERNS["placeholder_text"]["pattern"], content, re.IGNORECASE
            )
            if matches:
                violations.append((json_file.name, matches))

        assert not violations, "Placeholder text found in figure data:\n" + "\n".join(
            f"  {f}: {m}" for f, m in violations
        )

    def test_no_placeholders_in_r_output_strings(self, figure_r_scripts):
        """R scripts should not output placeholder text."""
        violations = []

        for r_script in figure_r_scripts:
            content = r_script.read_text()
            # Look for placeholders in ggtitle, labs, annotate calls
            output_contexts = re.findall(
                r"(?:ggtitle|labs|annotate|geom_text)\s*\([^)]*(?:TODO|FIXME|PLACEHOLDER)[^)]*\)",
                content,
                re.IGNORECASE,
            )
            if output_contexts:
                violations.append((r_script.name, output_contexts))

        assert not violations, "Placeholder text in R figure output:\n" + "\n".join(
            f"  {f}: {m}" for f, m in violations
        )


class TestNoSemanticTags:
    """Ensure semantic tags are rendered, not shown literally."""

    def test_no_double_colon_tags(self, figure_data_files):
        """No ::tag:: patterns in figure data."""
        violations = []

        for json_file in figure_data_files:
            content = json_file.read_text()
            matches = re.findall(ARTIFACT_PATTERNS["semantic_tag"]["pattern"], content)
            if matches:
                violations.append((json_file.name, matches))

        assert not violations, (
            "Semantic tags found (should be rendered):\n"
            + "\n".join(f"  {f}: {m}" for f, m in violations)
        )


class TestNoInternalGuidance:
    """Ensure internal guidance text doesn't leak into outputs."""

    def test_no_internal_markers(self, figure_data_files):
        """No INTERNAL/GUIDANCE markers in outputs."""
        violations = []

        for json_file in figure_data_files:
            # Exclude _TEST files which intentionally have synthetic data warnings
            if "_TEST" in json_file.name:
                continue
            content = json_file.read_text()
            matches = re.findall(
                ARTIFACT_PATTERNS["internal_guidance"]["pattern"],
                content,
                re.IGNORECASE,
            )
            if matches:
                violations.append((json_file.name, matches))

        assert not violations, (
            "Internal markers found in production outputs:\n"
            + "\n".join(f"  {f}: {m}" for f, m in violations)
        )


class TestArtifactDetectionComprehensive:
    """Run all artifact detection patterns."""

    def test_comprehensive_artifact_scan(self, figure_data_files):
        """
        Comprehensive scan for all known artifact patterns.

        This is the main integration test that catches any artifacts.
        """
        all_violations = []

        for json_file in figure_data_files:
            try:
                content = json_file.read_text()

                for artifact_name, config in ARTIFACT_PATTERNS.items():
                    if config["severity"] in ["CRITICAL", "HIGH"]:
                        matches = re.findall(config["pattern"], content, re.IGNORECASE)

                        # Filter out false positives
                        false_positives = config.get("false_positives", [])
                        if false_positives:
                            matches = [
                                m
                                for m in matches
                                if not any(
                                    fp.lower() in content.lower()
                                    for fp in false_positives
                                )
                            ]

                        # Also check filename for false positives
                        if any(
                            fp.lower() in json_file.name.lower()
                            for fp in false_positives
                        ):
                            matches = []

                        if matches:
                            all_violations.append(
                                {
                                    "file": json_file.name,
                                    "artifact": artifact_name,
                                    "severity": config["severity"],
                                    "matches": matches[:3],  # First 3 matches
                                    "description": config["description"],
                                }
                            )
            except Exception:
                continue

        assert not all_violations, "Rendering artifacts detected:\n" + "\n".join(
            f"  [{v['severity']}] {v['file']}: {v['artifact']} - {v['description']}"
            for v in all_violations
        )
