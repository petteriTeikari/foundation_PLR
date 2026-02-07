"""
P1 HIGH: Visual Rendering Validation Tests

Detect rendering failures, invisible elements, and visual anomalies.

ZERO TOLERANCE: If it's not visible, it's not communicated. Invisible data = failed science.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

try:
    import imagehash
    from PIL import Image

    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False


class TestMultiSeriesVisibility:
    """
    CRITICAL: Detect when multiple series in a figure are actually identical/overlapping.

    This would have caught CRITICAL-FAILURE-001 visually.
    """

    @pytest.mark.skipif(not IMAGEHASH_AVAILABLE, reason="imagehash not installed")
    def test_figure_has_multiple_distinct_colors(self, ggplot2_figures):
        """
        Multi-series figures should have distinct colors for each series.
        Uses color analysis to find dominant colors.
        """
        # Figures that should have multiple series
        MULTI_SERIES_FIGURES = [
            "fig_calibration",
            "fig_dca",
            "fig_prob_dist",
            "fig_retained",
        ]

        for fig_path in ggplot2_figures:
            # Check if this is a multi-series figure
            is_multi_series = any(ms in fig_path.stem for ms in MULTI_SERIES_FIGURES)

            if not is_multi_series:
                continue

            if fig_path.suffix.lower() == ".pdf":
                # Skip PDFs for now - need pdf2image
                continue

            self._check_color_diversity(fig_path, expected_series=4)

    def _check_color_diversity(self, fig_path: Path, expected_series: int):
        """Check that a figure has sufficient color diversity."""
        img = Image.open(fig_path).convert("RGB")
        pixels = np.array(img).reshape(-1, 3)

        # Quantize colors to reduce noise (32 levels per channel = ~32k colors)
        quantized = (pixels // 8) * 8

        # Find unique colors excluding near-white and near-gray backgrounds
        unique_colors = set()
        for pixel in quantized:
            # Skip white/light gray backgrounds
            if all(c > 220 for c in pixel):
                continue
            # Skip pure grays (gridlines, axis)
            if abs(pixel[0] - pixel[1]) < 10 and abs(pixel[1] - pixel[2]) < 10:
                if pixel[0] > 100:  # Light gray
                    continue
            unique_colors.add(tuple(pixel))

        # Filter to find "line" colors (colors with significant presence)
        # This is a heuristic - real lines should have multiple pixels
        color_counts = {}
        for pixel in quantized:
            key = tuple(pixel)
            color_counts[key] = color_counts.get(key, 0) + 1

        # Find colors that appear at least 0.1% of the time
        min_pixels = len(pixels) * 0.001
        significant_colors = {
            c
            for c, count in color_counts.items()
            if count > min_pixels
            and not all(v > 220 for v in c)  # Not white
            and not (
                abs(c[0] - c[1]) < 10 and abs(c[1] - c[2]) < 10 and c[0] > 100
            )  # Not light gray
        }

        # Should have at least expected_series distinct significant colors
        if len(significant_colors) < expected_series:
            warnings.warn(
                f"{fig_path.name}: Found only {len(significant_colors)} distinct "
                f"significant colors, expected at least {expected_series} for "
                f"multi-series figure. Curves may be overlapping."
            )


class TestFigureIntegrity:
    """
    Basic figure integrity checks.

    These tests catch blank/empty/corrupt figures that would indicate
    rendering failures or data issues.

    Addresses: GAP-03 (blank figure detection)
    """

    @pytest.mark.skipif(not IMAGEHASH_AVAILABLE, reason="imagehash not installed")
    def test_figure_not_empty(self, ggplot2_figures):
        """Figures should not be blank/white."""
        for fig_path in ggplot2_figures:
            if fig_path.suffix.lower() == ".pdf":
                continue

            img = Image.open(fig_path).convert("RGB")
            pixels = np.array(img)

            # Check if image is nearly all white
            white_ratio = np.mean(pixels > 250)
            assert white_ratio < 0.95, (
                f"CRITICAL: {fig_path.name} appears to be blank "
                f"({white_ratio:.1%} white pixels)"
            )

    @pytest.mark.skipif(not IMAGEHASH_AVAILABLE, reason="imagehash not installed")
    def test_figure_not_all_black(self, ggplot2_figures):
        """Figures should not be all black (rendering failure)."""
        for fig_path in ggplot2_figures:
            if fig_path.suffix.lower() == ".pdf":
                continue

            img = Image.open(fig_path).convert("RGB")
            pixels = np.array(img)

            # Check if image is nearly all black
            black_ratio = np.mean(pixels < 5)
            assert black_ratio < 0.95, (
                f"CRITICAL: {fig_path.name} appears to be all black "
                f"({black_ratio:.1%} black pixels). Rendering failure?"
            )

    @pytest.mark.skipif(not IMAGEHASH_AVAILABLE, reason="imagehash not installed")
    def test_figure_has_content_variance(self, ggplot2_figures):
        """
        Figures must have sufficient pixel variance to indicate actual content.

        A completely uniform or near-uniform image indicates rendering failure.
        Minimum variance threshold of 100 catches blank figures that might
        have subtle non-white backgrounds.

        Addresses: GAP-03 (blank figure detection with pixel variance)
        """
        MIN_VARIANCE = 100  # Minimum pixel variance threshold

        for fig_path in ggplot2_figures:
            if fig_path.suffix.lower() == ".pdf":
                continue

            img = Image.open(fig_path).convert("RGB")
            pixels = np.array(img, dtype=np.float64)

            # Calculate variance across all channels
            variance = np.var(pixels)

            assert variance > MIN_VARIANCE, (
                f"CRITICAL: {fig_path.name} has insufficient pixel variance "
                f"({variance:.1f}, minimum: {MIN_VARIANCE}). "
                f"Figure may be blank or have failed to render content."
            )

    def test_figure_file_size(self, ggplot2_figures):
        """
        Figures should have reasonable file sizes.

        Minimum 10KB ensures the file has actual content.
        Maximum 100MB prevents bloated files.
        """
        MIN_SIZE = 10_000  # 10KB minimum (increased from 1KB for better detection)
        MAX_SIZE = 100_000_000  # 100MB maximum

        for fig_path in ggplot2_figures:
            size = fig_path.stat().st_size

            assert size >= MIN_SIZE, (
                f"CRITICAL: {fig_path.name} is only {size} bytes ({size / 1000:.1f} KB). "
                f"File may be corrupt or empty. Minimum: {MIN_SIZE / 1000:.0f} KB."
            )

            assert size <= MAX_SIZE, (
                f"CRITICAL: {fig_path.name} is {size / 1e6:.1f} MB. "
                f"File may be bloated or corrupt."
            )


class TestFigureDimensions:
    """
    Validate figure dimensions and aspect ratios.
    """

    @pytest.mark.skipif(not IMAGEHASH_AVAILABLE, reason="imagehash not installed")
    def test_figure_dimensions_reasonable(self, ggplot2_figures):
        """Figures should have reasonable dimensions."""
        MIN_DIM = 100  # pixels
        MAX_DIM = 10000  # pixels

        for fig_path in ggplot2_figures:
            if fig_path.suffix.lower() == ".pdf":
                continue

            img = Image.open(fig_path)
            width, height = img.size

            assert width >= MIN_DIM and height >= MIN_DIM, (
                f"CRITICAL: {fig_path.name} has tiny dimensions "
                f"({width}x{height}). May be corrupt."
            )

            assert width <= MAX_DIM and height <= MAX_DIM, (
                f"CRITICAL: {fig_path.name} has huge dimensions "
                f"({width}x{height}). May cause memory issues."
            )

    @pytest.mark.skipif(not IMAGEHASH_AVAILABLE, reason="imagehash not installed")
    def test_aspect_ratio_reasonable(self, ggplot2_figures):
        """Figures should have reasonable aspect ratios."""
        MIN_RATIO = 0.2  # Width/height
        MAX_RATIO = 5.0

        for fig_path in ggplot2_figures:
            if fig_path.suffix.lower() == ".pdf":
                continue

            img = Image.open(fig_path)
            width, height = img.size
            ratio = width / height

            assert MIN_RATIO < ratio < MAX_RATIO, (
                f"CRITICAL: {fig_path.name} has extreme aspect ratio "
                f"{ratio:.2f}. Expected between {MIN_RATIO} and {MAX_RATIO}."
            )


class TestVisualRegression:
    """
    Visual regression testing against golden files.
    """

    HASH_THRESHOLD = 15  # Max Hamming distance for perceptual hash

    @pytest.mark.skipif(not IMAGEHASH_AVAILABLE, reason="imagehash not installed")
    def test_figures_match_golden(self, ggplot2_figures, golden_dir):
        """
        Compare figures to golden references.
        Skip if no golden file exists (first run).
        """
        for fig_path in ggplot2_figures:
            if fig_path.suffix.lower() == ".pdf":
                continue

            golden_path = golden_dir / fig_path.name

            if not golden_path.exists():
                # No golden file yet - this is expected on first run
                # To create golden files: copy current figures to golden_dir
                continue

            # Compare using perceptual hash
            gen_hash = imagehash.phash(Image.open(fig_path))
            gold_hash = imagehash.phash(Image.open(golden_path))

            distance = gen_hash - gold_hash

            assert distance <= self.HASH_THRESHOLD, (
                f"CRITICAL: {fig_path.name} differs from golden by {distance} "
                f"(threshold: {self.HASH_THRESHOLD}). Visual regression detected. "
                f"If intentional, update golden file."
            )

    @pytest.fixture
    def update_golden(self, request):
        """Fixture to enable golden file updates via --update-golden flag."""
        return request.config.getoption("--update-golden", default=False)


class TestLegendSize:
    """
    Validate legend sizes don't dominate the figure.

    Oversized legends reduce the plot area and obscure data.
    Addresses: GAP-04 (legend size validation)
    """

    MAX_LEGEND_WIDTH_RATIO = 0.30  # Legend shouldn't exceed 30% of figure width
    MAX_LEGEND_HEIGHT_RATIO = 0.40  # Legend shouldn't exceed 40% of figure height

    @pytest.mark.skipif(not IMAGEHASH_AVAILABLE, reason="imagehash not installed")
    def test_legend_not_oversized(self, ggplot2_figures):
        """
        Legend regions should not dominate the figure.

        This uses a heuristic: detect contiguous non-plot regions on the right
        side of the figure that might be legends.
        """
        for fig_path in ggplot2_figures:
            if fig_path.suffix.lower() == ".pdf":
                continue

            img = Image.open(fig_path).convert("RGB")
            width, height = img.size
            pixels = np.array(img)

            # Heuristic: Legend is usually on the right side
            # Check the rightmost 40% of the figure for legend-like regions
            right_portion = pixels[:, int(width * 0.6) :, :]

            # Look for text-like regions (not white background, not pure plot colors)
            # Legend typically has small text markers and lines
            right_gray = np.mean(right_portion, axis=2)

            # Count columns that have content (not all white)
            content_cols = np.sum(
                right_gray < 245, axis=0
            )  # Count non-white rows per column

            # Find contiguous legend-like region (columns with some but not too much content)
            legend_cols = 0
            for col_content in content_cols:
                # Legend has sparse content (text, lines) - between 5-50% of height
                content_ratio = col_content / height
                if 0.05 < content_ratio < 0.5:
                    legend_cols += 1

            # Estimate legend width
            legend_width_ratio = legend_cols / width if width > 0 else 0

            if legend_width_ratio > self.MAX_LEGEND_WIDTH_RATIO:
                warnings.warn(
                    f"{fig_path.name}: Potential oversized legend detected. "
                    f"Estimated legend width: {legend_width_ratio:.1%} of figure "
                    f"(threshold: {self.MAX_LEGEND_WIDTH_RATIO:.0%}). "
                    f"Consider repositioning or simplifying the legend."
                )


class TestEdgeClipping:
    """
    Detect content clipped at figure edges.
    """

    @pytest.mark.skipif(not IMAGEHASH_AVAILABLE, reason="imagehash not installed")
    def test_no_edge_clipping(self, ggplot2_figures):
        """
        Content should not be cut off at figure edges.
        Check for non-white pixels at borders.
        """
        BORDER_WIDTH = 5  # pixels

        for fig_path in ggplot2_figures:
            if fig_path.suffix.lower() == ".pdf":
                continue

            img = Image.open(fig_path).convert("RGB")
            pixels = np.array(img)
            h, w = pixels.shape[:2]

            # Check borders for non-white content
            borders = {
                "top": pixels[:BORDER_WIDTH, :, :],
                "bottom": pixels[-BORDER_WIDTH:, :, :],
                "left": pixels[:, :BORDER_WIDTH, :],
                "right": pixels[:, -BORDER_WIDTH:, :],
            }

            for edge, border_pixels in borders.items():
                # Calculate ratio of dark pixels in border
                dark_ratio = np.mean(border_pixels < 200)

                if dark_ratio > 0.1:  # More than 10% dark pixels at edge
                    warnings.warn(
                        f"{fig_path.name}: Significant content at {edge} edge "
                        f"({dark_ratio:.1%} dark pixels). "
                        f"Content may be clipped."
                    )


# Pytest hooks for custom options
def pytest_addoption(parser):
    """Add custom command line options."""
    try:
        parser.addoption(
            "--update-golden",
            action="store_true",
            default=False,
            help="Update golden files with current figures",
        )
    except ValueError:
        # Option already added
        pass
