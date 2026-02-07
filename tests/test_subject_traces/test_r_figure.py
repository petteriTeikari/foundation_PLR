# tests/test_subject_traces/test_r_figure.py
"""
TDD tests for R subject traces figure.
Run: pytest tests/test_subject_traces/test_r_figure.py -v

IMPORTANT: NO REGEX ALLOWED for code analysis.
Use simple string matching or R's own parser.
See: .claude/docs/meta-learnings/VIOLATION-002-regex-in-test-despite-ban.md
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.r_required

PROJECT_ROOT = Path(__file__).parents[2]
R_SCRIPT = PROJECT_ROOT / "src" / "r" / "figures" / "fig_subject_traces.R"


class TestRScriptHardcoding:
    """Verify R script follows no-hardcoding rules.

    NOTE: These checks use simple string matching, NOT regex.
    Regex is BANNED for code analysis per CLAUDE.md.
    """

    @pytest.fixture
    def r_lines(self):
        """Return R script as list of (line_number, line_content) tuples."""
        return list(enumerate(R_SCRIPT.read_text().split("\n"), 1))

    @pytest.fixture
    def r_code(self):
        return R_SCRIPT.read_text()

    def test_no_hardcoded_hex_colors(self, r_lines):
        """R script must not use hardcoded hex colors (except in comments).

        Uses simple string matching (no regex).
        """
        violations = []

        # Patterns that indicate hardcoded colors (simple string checks)
        bad_patterns = [
            'color = "#',
            "color = '#",
            'color= "#',
            "color= '#",
            'fill = "#',
            "fill = '#",
            'fill= "#',
            "fill= '#",
        ]

        for line_num, line in r_lines:
            # Skip comment lines
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # Check for hardcoded color patterns
            for pattern in bad_patterns:
                if pattern in line:
                    violations.append(f"Line {line_num}: {stripped}")
                    break

        assert not violations, "Hardcoded hex colors found:\n" + "\n".join(violations)

    def test_no_ggsave(self, r_lines):
        """R script must use save_publication_figure, not ggsave."""
        violations = []

        for line_num, line in r_lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "ggsave(" in line:
                violations.append(f"Line {line_num}: {stripped}")

        assert not violations, (
            "ggsave() found (use save_publication_figure):\n" + "\n".join(violations)
        )

    def test_uses_color_defs(self, r_code):
        """R script must load colors from YAML via color_defs."""
        assert "color_defs <- load_color_definitions()" in r_code, (
            "R script must load colors using load_color_definitions()"
        )

        # Should reference color_defs for actual color usage
        assert "color_defs[[" in r_code, (
            "R script must use color_defs[['--color-name']] for colors"
        )

    def test_uses_config_for_dimensions(self, r_code):
        """R script must get dimensions from config, not hardcoded."""
        # Should reference cfg$dimensions
        assert "cfg$dimensions$width" in r_code or "cfg$dimensions" in r_code, (
            "R script must get width from cfg$dimensions"
        )

    def test_graceful_degradation(self, r_code):
        """R script must handle missing data gracefully."""
        # Should check for NULL and exit cleanly
        assert "is.null(data)" in r_code, "R script must check if data is NULL"

        # Should not use stop() for missing data
        assert 'quit(save = "no", status = 0)' in r_code, (
            "R script must exit cleanly (status=0) when data unavailable"
        )

    def test_no_hardcoded_numeric_dimensions(self, r_lines):
        """Check that width/height are not hardcoded in save calls."""
        violations = []

        for line_num, line in r_lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # Check for save_publication_figure with hardcoded numbers
            # This is a simple heuristic check
            if "save_publication_figure(" in line:
                # The dimensions should come from cfg, not literal numbers
                # If we see patterns like "width = 14" that's hardcoded
                if "width = 1" in line or "height = 1" in line:
                    # But cfg$dimensions$width is OK
                    if "cfg$dimensions" not in line:
                        violations.append(f"Line {line_num}: {stripped}")

        assert not violations, "Hardcoded dimensions found:\n" + "\n".join(violations)


class TestFigureOutputs:
    """Test that generated figures exist and are valid."""

    @pytest.fixture
    def output_dir(self):
        return PROJECT_ROOT / "figures" / "generated" / "ggplot2" / "supplementary"

    def test_control_figure_exists(self, output_dir):
        """Control figure should exist after generation."""
        fig_path = output_dir / "fig_subject_traces_control.png"
        assert fig_path.exists(), f"Missing: {fig_path}. Run: make analyze"

        # Check file size is reasonable (> 100KB for a real figure)
        assert fig_path.stat().st_size > 100_000, (
            f"Figure too small ({fig_path.stat().st_size} bytes)"
        )

    def test_glaucoma_figure_exists(self, output_dir):
        """Glaucoma figure should exist after generation."""
        fig_path = output_dir / "fig_subject_traces_glaucoma.png"
        assert fig_path.exists(), f"Missing: {fig_path}. Run: make analyze"

        assert fig_path.stat().st_size > 100_000, (
            f"Figure too small ({fig_path.stat().st_size} bytes)"
        )


class TestRSyntaxValidation:
    """Use R's own parser to validate R code syntax."""

    def test_r_syntax_valid(self):
        """R script should parse without syntax errors."""
        import subprocess

        result = subprocess.run(
            ["Rscript", "-e", f"parse('{R_SCRIPT}')"], capture_output=True, text=True
        )

        assert result.returncode == 0, f"R syntax error:\n{result.stderr}"
