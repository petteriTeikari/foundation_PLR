#!/usr/bin/env python3
"""
Tests for R figure script compliance with anti-hardcoding rules.

These tests enforce CRITICAL-FAILURE-004 prevention:
- No hardcoded hex colors (#RRGGBB)
- No raw color names (gray, black, white, red, blue)
- No ggsave() calls (use save_publication_figure)
- No custom theme definitions (use theme_foundation_plr)
- No hardcoded dimensions in save calls

Severity levels (matching pre-commit hook):
- CRITICAL: Blocks commit - hex colors, ggsave()
- WARNING: Allows commit but logged - raw colors, custom themes, DPI

See: .claude/docs/meta-learnings/CRITICAL-FAILURE-004-r-figure-hardcoding.md
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
R_FIGURES_DIR = PROJECT_ROOT / "src" / "r" / "figures"
CHECKER_SCRIPT = PROJECT_ROOT / "scripts" / "validation" / "check_r_hardcoding.py"

# Known utility/infrastructure files to exclude from checks
EXCLUDE_PATTERNS = [
    "_TEMPLATE.R",
    "generate_all_r_figures.R",
    "common.R",
    "save_figure.R",  # Figure system implementation
    "config_loader.R",  # Config system
    "color_palettes.R",  # Color definitions
    "theme_foundation_plr.R",  # Theme definitions
]


def get_r_figure_files() -> list[Path]:
    """Get all R figure files (excluding templates and utilities)."""
    if not R_FIGURES_DIR.exists():
        return []

    files = list(R_FIGURES_DIR.glob("*.R"))
    return [f for f in files if not any(p in f.name for p in EXCLUDE_PATTERNS)]


def check_file_for_pattern(
    filepath: Path, pattern: str, skip_comments: bool = True
) -> list[tuple[int, str]]:
    """Check a file for a regex pattern, returning (line_num, line) tuples."""
    matches = []
    content = filepath.read_text(encoding="utf-8")
    lines = content.split("\n")

    for line_num, line in enumerate(lines, 1):
        if skip_comments and line.strip().startswith("#"):
            continue
        if re.search(pattern, line, re.IGNORECASE):
            matches.append((line_num, line.strip()[:80]))

    return matches


class TestRHardcodingCompliance:
    """Test suite for R figure hardcoding compliance."""

    def test_checker_script_exists(self):
        """The R hardcoding checker script must exist."""
        assert CHECKER_SCRIPT.exists(), (
            f"Missing: {CHECKER_SCRIPT}\n"
            "Create the checker script to enforce anti-hardcoding rules."
        )

    def test_checker_script_runs(self):
        """The checker script must be runnable."""
        assert CHECKER_SCRIPT.exists(), f"Missing: {CHECKER_SCRIPT}"

        result = subprocess.run(
            [sys.executable, str(CHECKER_SCRIPT), "--help"],
            capture_output=True,
            text=True,
        )
        # Script should run without error
        assert result.returncode in [0, 1], f"Checker failed: {result.stderr}"

    def test_no_hardcoded_hex_colors(self):
        """CRITICAL: R figure scripts must not contain hardcoded hex colors."""
        files = get_r_figure_files()
        assert files, "No R figure files found in src/r/figures/"

        violations = []
        # Pattern: color/fill/colour = "#RRGGBB" (direct assignment)
        hex_pattern = r'(color|fill|colour)\s*=\s*"#[0-9A-Fa-f]{6}"'

        for filepath in files:
            for line_num, line in check_file_for_pattern(filepath, hex_pattern):
                violations.append(f"{filepath.name}:{line_num}: {line}")

        if violations:
            pytest.fail(
                "CRITICAL: Hardcoded hex colors found!\n"
                'Use color_defs[["--color-xxx"]] instead.\n\n'
                "Violations:\n" + "\n".join(violations[:10])
            )

    def test_no_raw_color_names(self):
        """WARNING: R figure scripts should not use raw color names."""
        files = get_r_figure_files()
        assert files, "No R figure files found in src/r/figures/"

        violations = []
        # Pattern: common color names in color/fill assignments
        raw_color_pattern = (
            r'(color|fill|colour)\s*=\s*"(gray\d*|grey\d*|black|white|red|blue|green)"'
        )

        for filepath in files:
            for line_num, line in check_file_for_pattern(filepath, raw_color_pattern):
                violations.append(f"{filepath.name}:{line_num}: {line}")

        if violations:
            # This is a WARNING, not CRITICAL - using pytest.warns would be better
            # but for consistency, we fail but note it's a warning
            pytest.fail(
                "WARNING: Raw color names used instead of color_defs!\n"
                'Use color_defs[["--color-xxx"]] instead of "gray50", "black", etc.\n\n'
                "Violations:\n" + "\n".join(violations[:10])
            )

    def test_no_hex_colors_in_scale_vectors(self):
        """CRITICAL: Hex colors in scale_*_manual() vectors must use color_defs."""
        files = get_r_figure_files()
        assert files, "No R figure files found in src/r/figures/"

        violations = []
        # Pattern: hex colors embedded in c() vectors within scale functions
        # This catches: scale_color_manual(values = c("A" = "#006BA2", ...))
        hex_in_vector_pattern = r'=\s*"#[0-9A-Fa-f]{6}"'

        for filepath in files:
            content = filepath.read_text(encoding="utf-8")
            lines = content.split("\n")

            in_scale_function = False
            paren_depth = 0

            for line_num, line in enumerate(lines, 1):
                if line.strip().startswith("#"):
                    continue

                # Track if we're inside a scale_*_manual call
                if re.search(r"scale_\w+_manual\s*\(", line):
                    in_scale_function = True
                    paren_depth = line.count("(") - line.count(")")

                if in_scale_function:
                    paren_depth += line.count("(") - line.count(")")
                    if paren_depth <= 0:
                        in_scale_function = False

                    # Check for hex colors inside the scale function
                    if re.search(hex_in_vector_pattern, line):
                        # Exclude lines that use color_defs
                        if "color_defs" not in line:
                            violations.append(
                                f"{filepath.name}:{line_num}: {line.strip()[:80]}"
                            )

        if violations:
            pytest.fail(
                "CRITICAL: Hardcoded hex colors in scale_*_manual() vectors!\n"
                'Use color_defs[["--color-xxx"]] in scale functions.\n\n'
                "Violations:\n" + "\n".join(violations[:10])
            )

    def test_no_ggsave_usage(self):
        """CRITICAL: R figure scripts must use save_publication_figure(), not ggsave()."""
        files = get_r_figure_files()
        assert files, "No R figure files found in src/r/figures/"

        violations = []
        ggsave_pattern = r"\bggsave\s*\("

        for filepath in files:
            for line_num, line in check_file_for_pattern(filepath, ggsave_pattern):
                violations.append(f"{filepath.name}:{line_num}: {line}")

        if violations:
            pytest.fail(
                "CRITICAL: ggsave() used instead of save_publication_figure()!\n"
                'Use: save_publication_figure(plot, "fig_name")\n\n'
                "Violations:\n" + "\n".join(violations[:10])
            )

    def test_no_custom_theme_definitions(self):
        """WARNING: R figure scripts must not define custom themes."""
        files = get_r_figure_files()
        assert files, "No R figure files found in src/r/figures/"

        violations = []
        # Pattern: name_theme or theme_name <- function (excluding theme_foundation_plr)
        theme_pattern = r"(\w+_theme|theme_\w+)\s*<-\s*function\s*\("

        for filepath in files:
            for line_num, line in check_file_for_pattern(filepath, theme_pattern):
                if "theme_foundation_plr" not in line:
                    violations.append(f"{filepath.name}:{line_num}: {line}")

        if violations:
            pytest.fail(
                "WARNING: Custom theme functions defined!\n"
                "Use: source('src/r/theme_foundation_plr.R'); p + theme_foundation_plr()\n\n"
                "Violations:\n" + "\n".join(violations[:10])
            )

    def test_loads_color_definitions(self):
        """R figure scripts using colors should load color_defs from YAML."""
        files = get_r_figure_files()
        assert files, "No R figure files found in src/r/figures/"

        missing_loader = []
        for filepath in files:
            content = filepath.read_text(encoding="utf-8")

            # Check for color_defs loading
            has_color_loader = (
                "load_color_definitions" in content
                or "color_defs <-" in content
                or "color_defs=" in content
                or "ECONOMIST_PALETTE" in content  # Alternative pattern
                or "ECONOMIST_COLORS" in content  # theme_foundation_plr pattern
            )

            # Only flag if file uses colors at all (not just mentions them in comments)
            # Check for actual color assignments, not just the word "color"
            uses_colors = bool(
                re.search(
                    r'(color|fill|colour)\s*=\s*["\']?[^,\s\)]+',
                    content,
                    re.IGNORECASE,
                )
            )

            if uses_colors and not has_color_loader:
                missing_loader.append(filepath.name)

        if missing_loader:
            pytest.fail(
                "R figure scripts using colors should load color_defs from YAML.\n"
                "Add: color_defs <- load_color_definitions()\n\n"
                f"Missing in: {', '.join(missing_loader[:5])}"
            )

    def test_uses_figure_system_for_save(self):
        """R figure scripts saving figures should use save_publication_figure()."""
        files = get_r_figure_files()
        assert files, "No R figure files found in src/r/figures/"

        missing_system = []
        for filepath in files:
            content = filepath.read_text(encoding="utf-8")

            # Check if file saves figures (ggsave, png, pdf, or figure system)
            saves_figures = bool(
                re.search(r"\b(ggsave|save_publication_figure|png|pdf)\s*\(", content)
            )

            # Check if using figure system
            uses_system = "save_publication_figure" in content

            # Only flag if file saves but doesn't use system
            # Allow png()/pdf() if they have exception comment
            if saves_figures and not uses_system:
                # Check for documented exception
                if "# figure-system-exception" not in content:
                    missing_system.append(filepath.name)

        if missing_system:
            pytest.fail(
                "R figure scripts should use save_publication_figure() from figure system.\n"
                "Add: source('src/r/figure_system/save_figure.R')\n"
                "Use: save_publication_figure(plot, 'fig_name')\n\n"
                f"Missing in: {', '.join(missing_system[:5])}"
            )

    def test_sources_figure_system(self):
        """R figure scripts should source the figure system files."""
        files = get_r_figure_files()
        assert files, "No R figure files found in src/r/figures/"

        missing_sources = []
        required_sources = [
            "config_loader.R",
            "save_figure.R",
            "theme_foundation_plr.R",
        ]

        for filepath in files:
            content = filepath.read_text(encoding="utf-8")

            # Skip if file doesn't save figures (utility scripts)
            if not re.search(r"\b(ggsave|save_publication_figure)\s*\(", content):
                continue

            # Check for required sources
            missing = []
            for source in required_sources:
                if source not in content:
                    missing.append(source)

            if missing:
                missing_sources.append(f"{filepath.name}: missing {', '.join(missing)}")

        if missing_sources:
            pytest.fail(
                "R figure scripts should source the figure system files.\n"
                "Add to header:\n"
                '  source(file.path(PROJECT_ROOT, "src/r/figure_system/config_loader.R"))\n'
                '  source(file.path(PROJECT_ROOT, "src/r/figure_system/save_figure.R"))\n'
                '  source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))\n\n'
                "Issues:\n" + "\n".join(missing_sources[:5])
            )

    def test_no_hardcoded_dimensions_in_save(self):
        """WARNING: save_publication_figure() calls should not have hardcoded dimensions.

        Dimensions should come from figure_registry.yaml or figure_layouts.yaml,
        not be hardcoded as width = 14, height = 7 in the script.
        """
        files = get_r_figure_files()
        assert files, "No R figure files found in src/r/figures/"

        violations = []
        # Pattern: save_publication_figure(..., width = <number>, ...)
        dimension_pattern = (
            r"save_publication_figure\s*\([^)]*\b(width|height)\s*=\s*\d+"
        )

        for filepath in files:
            content = filepath.read_text(encoding="utf-8")
            lines = content.split("\n")

            for line_num, line in enumerate(lines, 1):
                if line.strip().startswith("#"):
                    continue
                if re.search(dimension_pattern, line, re.IGNORECASE):
                    violations.append(
                        f"{filepath.name}:{line_num}: {line.strip()[:80]}"
                    )

        if violations:
            pytest.fail(
                "WARNING: Hardcoded dimensions in save_publication_figure() calls!\n"
                "Dimensions should come from figure_registry.yaml or figure_layouts.yaml.\n"
                "The figure system should look up dimensions by figure name.\n\n"
                f"Violations ({len(violations)} total):\n" + "\n".join(violations[:15])
            )

    def test_no_hardcoded_dpi(self):
        """WARNING: DPI values should come from figure_layouts.yaml, not hardcoded."""
        files = get_r_figure_files()
        assert files, "No R figure files found in src/r/figures/"

        violations = []
        # Pattern: dpi = <number> (outside of comments)
        dpi_pattern = r"\bdpi\s*=\s*\d+"

        for filepath in files:
            for line_num, line in check_file_for_pattern(filepath, dpi_pattern):
                violations.append(f"{filepath.name}:{line_num}: {line}")

        if violations:
            pytest.fail(
                "WARNING: Hardcoded DPI values found!\n"
                "DPI should come from figure_layouts.yaml output_settings.dpi.\n\n"
                f"Violations ({len(violations)} total):\n" + "\n".join(violations[:10])
            )


class TestRTemplateCompliance:
    """Test that R template exists and is compliant."""

    def test_template_exists(self):
        """An R figure template should exist."""
        template_path = R_FIGURES_DIR / "_TEMPLATE.R"
        assert template_path.exists(), (
            f"Template not found at {template_path}\n"
            "Create a template with the mandatory header pattern."
        )

    def test_template_has_mandatory_header(self):
        """Template should have the mandatory config loading header."""
        template_path = R_FIGURES_DIR / "_TEMPLATE.R"
        assert template_path.exists(), f"Missing: {template_path}"

        content = template_path.read_text()

        required_patterns = [
            "PROJECT_ROOT",
            "config_loader.R",
            "save_figure.R",
            "theme_foundation_plr.R",
            "load_color_definitions",
            "MANDATORY HEADER",
        ]

        missing = [p for p in required_patterns if p not in content]
        if missing:
            pytest.fail(f"Template missing required patterns: {missing}")


class TestYAMLConfigLoading:
    """Test that R figure system correctly loads config from YAML."""

    def test_save_figure_has_dimension_loader(self):
        """save_figure.R must have the .load_figure_dimensions() function."""
        save_figure_path = (
            PROJECT_ROOT / "src" / "r" / "figure_system" / "save_figure.R"
        )
        assert save_figure_path.exists(), f"Missing: {save_figure_path}"

        content = save_figure_path.read_text()

        assert ".load_figure_dimensions" in content, (
            "save_figure.R must define .load_figure_dimensions() function\n"
            "This function loads dimensions from figure_registry.yaml"
        )

    def test_dimension_loader_reads_registry(self):
        """The dimension loader must read from figure_registry.yaml."""
        save_figure_path = (
            PROJECT_ROOT / "src" / "r" / "figure_system" / "save_figure.R"
        )
        assert save_figure_path.exists(), f"Missing: {save_figure_path}"

        content = save_figure_path.read_text()

        assert "figure_registry.yaml" in content, (
            "save_figure.R must reference figure_registry.yaml\n"
            "Dimensions should come from: configs/VISUALIZATION/figure_registry.yaml"
        )

    def test_figure_registry_has_r_figures_section(self):
        """figure_registry.yaml must have r_figures section with dimensions."""
        import yaml

        registry_path = (
            PROJECT_ROOT / "configs" / "VISUALIZATION" / "figure_registry.yaml"
        )
        assert registry_path.exists(), f"Missing: {registry_path}"

        with open(registry_path) as f:
            registry = yaml.safe_load(f)

        assert "r_figures" in registry, (
            "figure_registry.yaml must have 'r_figures' section\n"
            "This section defines dimensions for R-generated figures"
        )

        r_figures = registry["r_figures"]
        assert len(r_figures) > 0, "r_figures section should not be empty"

        # Check that at least some figures have width/height
        figures_with_dims = 0
        for fig_name, fig_config in r_figures.items():
            if "styling" in fig_config:
                styling = fig_config["styling"]
                if "width" in styling and "height" in styling:
                    figures_with_dims += 1

        assert figures_with_dims > 0, (
            "r_figures should have at least some figures with width/height defined"
        )

    def test_save_publication_figure_uses_registry_dimensions(self):
        """save_publication_figure() should auto-load dimensions from registry."""
        save_figure_path = (
            PROJECT_ROOT / "src" / "r" / "figure_system" / "save_figure.R"
        )
        assert save_figure_path.exists(), f"Missing: {save_figure_path}"

        content = save_figure_path.read_text()

        # Check that save_publication_figure calls the dimension loader
        # Look for the pattern where it loads dimensions when width/height are NULL
        has_auto_load = (
            "is.null(width)" in content and ".load_figure_dimensions" in content
        )

        assert has_auto_load, (
            "save_publication_figure() should auto-load dimensions when width/height are NULL.\n"
            "Pattern expected:\n"
            "  if (is.null(width) || is.null(height)) {\n"
            "    registry_dims <- .load_figure_dimensions(filename)\n"
            "    ..."
        )

    def test_all_r_figure_scripts_registered(self):
        """All R figure scripts should have corresponding entries in figure_registry.yaml."""
        import yaml

        registry_path = (
            PROJECT_ROOT / "configs" / "VISUALIZATION" / "figure_registry.yaml"
        )
        assert registry_path.exists(), f"Missing: {registry_path}"

        with open(registry_path) as f:
            registry = yaml.safe_load(f)

        r_figures = registry.get("r_figures", {})
        registered_names = set(r_figures.keys())

        # Get figure names from scripts (extract from save_publication_figure calls)
        unregistered = []
        files = get_r_figure_files()

        for filepath in files:
            content = filepath.read_text()
            # Find save_publication_figure calls and extract figure names
            # Pattern: save_publication_figure(plot_object, "fig_name" - must start with "fig_"
            matches = re.findall(
                r'save_publication_figure\s*\(\s*\w+\s*,\s*"(fig_[^"]+)"', content
            )
            for fig_name in matches:
                if fig_name not in registered_names:
                    unregistered.append(f"{filepath.name}: {fig_name}")

        if unregistered:
            pytest.fail(
                "Some figures are not registered in figure_registry.yaml:\n"
                + "\n".join(unregistered[:10])
                + "\n\nAdd entries to configs/VISUALIZATION/figure_registry.yaml r_figures section"
            )


class TestCheckerPrecommitSync:
    """Test that checker script and pre-commit are in sync."""

    def test_checker_catches_all_critical_violations(self):
        """Verify checker catches critical violations on test files."""
        assert CHECKER_SCRIPT.exists(), f"Missing: {CHECKER_SCRIPT}"

        # Create a temp file with known violations
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
            f.write("""
# Test file with violations
library(ggplot2)
p <- ggplot(data) +
  geom_point(color = "#006BA2") +
  theme_minimal()
ggsave("test.png", p)
""")
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, str(CHECKER_SCRIPT), temp_path],
                capture_output=True,
                text=True,
            )
            # Should exit with code 1 (critical violations found)
            assert result.returncode == 1, (
                f"Checker should fail on critical violations.\n"
                f"Exit code: {result.returncode}\n"
                f"Output: {result.stdout}"
            )
            # Should mention the violations
            assert "CRITICAL" in result.stdout, (
                f"Checker output should mention CRITICAL violations.\n"
                f"Output: {result.stdout}"
            )
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
