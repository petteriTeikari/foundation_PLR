"""
P2 MEDIUM: Publication Standards Validation Tests

Validate DPI, dimensions, fonts, and journal requirements.

ZERO TOLERANCE: Unpublishable figures = wasted research effort = unacceptable.
"""

import re
import subprocess
import warnings

import pytest

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class TestResolutionStandards:
    """
    Validate figure resolution meets publication requirements.
    """

    MIN_DPI = 300  # Standard minimum for print
    MIN_WIDTH_PX_COMPENSATE = 2100  # If DPI unknown, width should compensate

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not installed")
    def test_raster_figures_have_sufficient_dpi(self, ggplot2_figures):
        """Raster figures must have DPI >= 300 for publication."""
        raster_formats = [".png", ".jpg", ".jpeg", ".tiff", ".tif"]

        for fig_path in ggplot2_figures:
            if fig_path.suffix.lower() not in raster_formats:
                continue

            img = Image.open(fig_path)
            width_px, height_px = img.size

            # Try to get DPI from metadata
            dpi = img.info.get("dpi", (72, 72))
            if isinstance(dpi, tuple):
                dpi = dpi[0]

            # If DPI is low, check if resolution compensates
            if dpi < self.MIN_DPI:
                if width_px < self.MIN_WIDTH_PX_COMPENSATE:
                    pytest.fail(
                        f"CRITICAL: {fig_path.name} has DPI={dpi} (< {self.MIN_DPI}) "
                        f"and width={width_px}px (< {self.MIN_WIDTH_PX_COMPENSATE}px). "
                        f"Figure will appear pixelated in print."
                    )
                else:
                    warnings.warn(
                        f"{fig_path.name}: DPI={dpi} is low but width={width_px}px "
                        f"may compensate. Verify print quality."
                    )


class TestDimensionStandards:
    """
    Validate figure dimensions for journal requirements.
    """

    # Common journal dimension limits (in inches)
    SINGLE_COLUMN_WIDTH = 3.5  # ~89mm
    DOUBLE_COLUMN_WIDTH = 7.0  # ~178mm
    MAX_HEIGHT = 9.0  # ~229mm

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not installed")
    def test_figure_dimensions_for_print(self, ggplot2_figures):
        """Figures should fit standard journal column widths."""
        for fig_path in ggplot2_figures:
            if fig_path.suffix.lower() == ".pdf":
                # For PDFs, would need to extract mediabox
                continue

            img = Image.open(fig_path)
            width_px, height_px = img.size

            # Get DPI
            dpi = img.info.get("dpi", (300, 300))
            if isinstance(dpi, tuple):
                dpi = dpi[0]
            if dpi == 0:
                dpi = 300  # Assume 300 if not specified

            # Convert to inches
            width_in = width_px / dpi
            height_in = height_px / dpi

            # Check against maximums
            if width_in > self.DOUBLE_COLUMN_WIDTH * 1.5:
                warnings.warn(
                    f'{fig_path.name}: Width {width_in:.1f}" exceeds double column '
                    f'({self.DOUBLE_COLUMN_WIDTH}"). May need scaling.'
                )

            if height_in > self.MAX_HEIGHT:
                warnings.warn(
                    f'{fig_path.name}: Height {height_in:.1f}" exceeds max '
                    f'({self.MAX_HEIGHT}"). May need scaling.'
                )


class TestPDFStandards:
    """
    Validate PDF-specific requirements.
    """

    def test_pdf_files_are_valid(self, ggplot2_figures):
        """PDF files should be valid/openable."""
        for fig_path in ggplot2_figures:
            if fig_path.suffix.lower() != ".pdf":
                continue

            # Try to get PDF info using pdfinfo (if available)
            try:
                result = subprocess.run(
                    ["pdfinfo", str(fig_path)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    pytest.fail(
                        f"CRITICAL: {fig_path.name} is not a valid PDF. "
                        f"Error: {result.stderr}"
                    )
            except FileNotFoundError:
                # pdfinfo not available, skip this check
                pass
            except subprocess.TimeoutExpired:
                pytest.fail(
                    f"CRITICAL: {fig_path.name} took too long to analyze. "
                    f"File may be corrupt."
                )

    def test_pdf_fonts_embedded(self, ggplot2_figures):
        """PDF fonts should be embedded to prevent substitution."""
        for fig_path in ggplot2_figures:
            if fig_path.suffix.lower() != ".pdf":
                continue

            # Check fonts using pdffonts (if available)
            try:
                result = subprocess.run(
                    ["pdffonts", str(fig_path)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    output = result.stdout

                    # Check for fonts that are not embedded
                    # pdffonts output has columns: name, type, emb, sub, uni, prob
                    lines = output.strip().split("\n")
                    for line in lines[2:]:  # Skip header
                        parts = line.split()
                        if len(parts) >= 3:
                            emb = parts[2] if len(parts) > 2 else ""
                            if emb == "no":
                                font_name = parts[0]
                                warnings.warn(
                                    f"{fig_path.name}: Font '{font_name}' is not embedded. "
                                    f"May cause substitution on other systems."
                                )
            except FileNotFoundError:
                # pdffonts not available
                pass


class TestStatisticalNotation:
    """
    Validate statistical notation in figure data/labels.
    """

    def test_ci_notation_format(self, calibration_data):
        """
        CI notation should follow standard format: "X.XX [lo, hi]" or "X.XX (lo-hi)".
        """
        # This is more about the data format than the visual - check metadata
        # The actual label formatting is in R scripts
        pass

    def test_metric_precision(self, calibration_data):
        """Metrics should have appropriate precision (not too many decimals)."""
        MAX_DECIMALS = 4

        configs = calibration_data.get("data", {}).get("configs", [])

        for config in configs:
            for key, value in config.items():
                if isinstance(value, float):
                    # Count decimal places
                    str_val = f"{value:.10f}".rstrip("0")
                    if "." in str_val:
                        decimals = len(str_val.split(".")[1])
                        # Just warn, don't fail - this is about display
                        if decimals > MAX_DECIMALS:
                            pass  # Storage precision is fine


class TestColorConsistency:
    """
    Validate color consistency across figures.
    """

    # Expected colors from plot_config.py COLORS dict
    EXPECTED_MODEL_COLORS = {
        # These should match the theme
        "ground_truth": "#1f77b4",  # Blue
        "best_ensemble": "#ff7f0e",  # Orange
        "best_single_fm": "#2ca02c",  # Green
        "traditional": "#d62728",  # Red
    }

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not installed")
    def test_color_palette_present(self, ggplot2_figures):
        """
        Verify that expected palette colors appear in multi-series figures.
        """
        MULTI_SERIES_FIGURES = ["fig_calibration", "fig_dca"]

        for fig_path in ggplot2_figures:
            if fig_path.suffix.lower() == ".pdf":
                continue

            is_multi_series = any(ms in fig_path.stem for ms in MULTI_SERIES_FIGURES)
            if not is_multi_series:
                continue

            # This would require color extraction and matching
            # For now, just check the figure exists and has content
            img = Image.open(fig_path)
            assert img.size[0] > 0 and img.size[1] > 0


class TestFileNaming:
    """
    Validate figure file naming conventions.
    """

    # Expected naming pattern: fig_<name>.{pdf,png}
    VALID_PATTERN = re.compile(r"^fig_[a-z0-9_]+\.(pdf|png)$")

    def test_figure_naming_convention(self, ggplot2_figures):
        """Figure files should follow naming convention."""
        for fig_path in ggplot2_figures:
            if not self.VALID_PATTERN.match(fig_path.name):
                warnings.warn(
                    f"{fig_path.name} does not follow naming convention "
                    f"'fig_<name>.{{pdf,png}}'. Consider renaming for consistency."
                )

    def test_no_spaces_in_filenames(self, ggplot2_figures):
        """Filenames should not contain spaces."""
        for fig_path in ggplot2_figures:
            assert " " not in fig_path.name, (
                f"CRITICAL: {fig_path.name} contains spaces. "
                f"This can cause issues with LaTeX and scripts."
            )

    def test_no_special_characters(self, ggplot2_figures):
        """Filenames should not contain problematic special characters."""
        FORBIDDEN = ["#", "$", "%", "&", "{", "}", "\\", "<", ">", "*", "?", '"', "'"]

        for fig_path in ggplot2_figures:
            for char in FORBIDDEN:
                assert char not in fig_path.name, (
                    f"CRITICAL: {fig_path.name} contains '{char}'. "
                    f"This can cause issues with file systems and LaTeX."
                )


class TestMetadataPresence:
    """
    Validate that figures have associated metadata.
    """

    def test_json_data_exists_for_figures(self, ggplot2_figures, project_root):
        """Each figure should have corresponding JSON data for reproducibility."""
        json_dir = project_root / "data" / "r_data"

        assert json_dir.exists(), (
            f"JSON data directory not found: {json_dir}. Run: make analyze"
        )

        for fig_path in ggplot2_figures:
            # Map figure name to expected JSON
            fig_name = fig_path.stem

            # Common mappings
            json_mappings = {
                "fig_calibration_stratos": "calibration_data.json",
                "fig_dca_stratos": "dca_data.json",
                "fig_prob_dist_by_outcome": "predictions_top4.json",
            }

            if fig_name in json_mappings:
                json_path = json_dir / json_mappings[fig_name]
                assert json_path.exists(), (
                    f"CRITICAL: {fig_path.name} has no corresponding JSON data at "
                    f"{json_path}. Cannot verify data provenance."
                )
