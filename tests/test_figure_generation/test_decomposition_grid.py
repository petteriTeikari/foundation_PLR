"""End-to-end tests for decomposition grid figure generation.

Tests the full pipeline from data loading to figure output:
1. Figure generates without errors
2. Output has correct structure (5×5 grid)
3. JSON data is saved correctly
4. No synthetic data markers in production output
5. Publication standards are met (DPI, dimensions)
"""

import json
import pytest
from pathlib import Path
from PIL import Image
import numpy as np

# Test paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures" / "generated"
TEST_FIGURE = FIGURES_DIR / "fig_decomposition_grid_TEST.png"
TEST_JSON = FIGURES_DIR / "data" / "fig_decomposition_grid_TEST.json"


class TestDecompositionGridFigureGeneration:
    """E2E tests for decomposition grid figure generation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Generate test figure before tests."""
        import subprocess

        # Generate test figure with synthetic data
        result = subprocess.run(
            ["uv", "run", "python", "src/viz/fig_decomposition_grid.py", "--test"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        if result.returncode != 0:
            pytest.fail(f"Figure generation failed: {result.stderr}")

    def test_figure_file_created(self):
        """Test figure PNG file is created."""
        assert TEST_FIGURE.exists(), f"Figure not found: {TEST_FIGURE}"

    def test_json_file_created(self):
        """Test JSON data file is created alongside figure."""
        assert TEST_JSON.exists(), f"JSON data not found: {TEST_JSON}"

    def test_figure_dimensions_reasonable(self):
        """Test figure has reasonable dimensions (not blank/corrupt)."""
        img = Image.open(TEST_FIGURE)

        # At 100 DPI (test mode), 14x10 inches = 1400x1000 pixels
        # Allow some variation for different DPI settings
        assert img.width >= 1000, f"Figure too narrow: {img.width}px"
        assert img.height >= 700, f"Figure too short: {img.height}px"

    def test_json_has_all_25_cells(self):
        """Test JSON contains data for all 5×5=25 grid cells."""
        with open(TEST_JSON) as f:
            data = json.load(f)

        # Should have 'data' key with cell data
        assert "data" in data, "JSON missing 'data' key"

        # Should have 25 cells (5 methods × 5 categories)
        cells = data["data"]
        assert len(cells) == 25, f"Expected 25 cells, got {len(cells)}"

    def test_json_has_row_column_metadata(self):
        """Test JSON includes row/column metadata."""
        with open(TEST_JSON) as f:
            data = json.load(f)

        assert "rows" in data, "JSON missing 'rows' metadata"
        assert "columns" in data, "JSON missing 'columns' metadata"

        assert len(data["rows"]) == 5, f"Expected 5 rows, got {len(data['rows'])}"
        assert len(data["columns"]) == 5, (
            f"Expected 5 columns, got {len(data['columns'])}"
        )

    def test_json_marks_synthetic_data(self):
        """Test JSON correctly marks test data as synthetic."""
        with open(TEST_JSON) as f:
            data = json.load(f)

        # Test figure should have synthetic marker
        assert data.get("synthetic") is True, "Test data should be marked as synthetic"
        assert "warning" in data, "Test data should have warning about synthetic data"


class TestDecompositionGridNoHardcoding:
    """Tests to verify no hardcoded values in figure generation."""

    def test_no_hardcoded_hex_colors_in_main_plotting(self):
        """Verify main plotting code uses COLORS dict, not hardcoded hex."""
        fig_script = PROJECT_ROOT / "src" / "viz" / "fig_decomposition_grid.py"

        with open(fig_script) as f:
            content = f.read()

        # Split into lines and check non-comment, non-fallback lines
        import re

        lines_with_hex = []
        for i, line in enumerate(content.split("\n"), 1):
            # Skip comments and docstrings
            stripped = line.strip()
            if (
                stripped.startswith("#")
                or stripped.startswith('"""')
                or stripped.startswith("'''")
            ):
                continue

            # Skip fallback patterns like: COLORS.get("key", "#666666")
            if "COLORS.get(" in line and "#" in line:
                continue

            # Check for hardcoded hex colors (not in string literals for config)
            # Pattern: #RRGGBB not inside quotes preceded by COLORS
            if re.search(r'[^"\']\s*#[0-9A-Fa-f]{6}', line):
                # Additional check: is this in a dict assignment that's clearly config?
                if 'COLORS["' not in line and "COLORS['" not in line:
                    lines_with_hex.append((i, line.strip()))

        # Should have no hardcoded hex colors in plotting code
        # (the fallback defaults are acceptable)
        problem_lines = [
            (ln, code)
            for ln, code in lines_with_hex
            if "fallback" not in code.lower() and ".get(" not in code
        ]

        assert len(problem_lines) == 0, "Found hardcoded hex colors:\n" + "\n".join(
            f"  Line {ln}: {code}" for ln, code in problem_lines
        )

    def test_colors_imported_from_plot_config(self):
        """Verify COLORS is imported from plot_config."""
        fig_script = PROJECT_ROOT / "src" / "viz" / "fig_decomposition_grid.py"

        with open(fig_script) as f:
            content = f.read()

        assert "from src.viz.plot_config import" in content
        assert "COLORS" in content


class TestDecompositionGridPublicationStandards:
    """Tests for publication-quality figure standards."""

    def test_figure_is_not_blank(self):
        """Test figure is not blank (has visible content)."""
        img = Image.open(TEST_FIGURE)
        img_array = np.array(img)

        # Check that not all pixels are the same (would indicate blank figure)
        if len(img_array.shape) == 3:
            # Color image
            std_per_channel = [
                img_array[:, :, i].std() for i in range(img_array.shape[2])
            ]
            assert max(std_per_channel) > 10, (
                "Figure appears to be blank (low variance)"
            )
        else:
            # Grayscale
            assert img_array.std() > 10, "Figure appears to be blank (low variance)"

    def test_json_components_have_arrays(self):
        """Test each cell's components have proper array data."""
        with open(TEST_JSON) as f:
            data = json.load(f)

        for cell_key, cell_data in data["data"].items():
            assert "components" in cell_data, f"Cell {cell_key} missing 'components'"
            assert "time_vector" in cell_data, f"Cell {cell_key} missing 'time_vector'"
            assert "mean_waveform" in cell_data, (
                f"Cell {cell_key} missing 'mean_waveform'"
            )

            # Check components have proper structure
            for comp in cell_data["components"]:
                assert "name" in comp, f"Component in {cell_key} missing 'name'"
                assert "mean" in comp, f"Component in {cell_key} missing 'mean'"
                assert "ci_lower" in comp, f"Component in {cell_key} missing 'ci_lower'"
                assert "ci_upper" in comp, f"Component in {cell_key} missing 'ci_upper'"

                # Verify arrays have reasonable length
                assert len(comp["mean"]) > 50, (
                    f"Component {comp['name']} in {cell_key} has too few points"
                )


class TestDecompositionGridDataProvenance:
    """Tests for data provenance (CRITICAL-FAILURE-001 prevention)."""

    def test_different_categories_have_different_data(self):
        """Verify different preprocessing categories produce different waveforms."""
        with open(TEST_JSON) as f:
            data = json.load(f)

        # Get mean waveforms for each category (using first method as representative)
        method = data["rows"][0]  # e.g., "template"
        category_waveforms = {}

        for cell_key, cell_data in data["data"].items():
            # Cell keys are like "Ground Truth__template"
            parts = cell_key.split("__")
            if len(parts) == 2:
                category, cell_method = parts
                if cell_method == method:
                    category_waveforms[category] = np.array(cell_data["mean_waveform"])

        # Check pairwise correlations
        categories = list(category_waveforms.keys())
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i + 1 :]:
                corr = np.corrcoef(category_waveforms[cat1], category_waveforms[cat2])[
                    0, 1
                ]

                # For synthetic TEST data, correlations will be high but not perfect
                # For real data, we expect more variation
                # This test mainly catches the bug where all categories are identical
                assert corr < 0.9999, (
                    f"Categories {cat1} and {cat2} have suspiciously identical waveforms "
                    f"(correlation={corr:.6f}). Possible synthetic data issue."
                )

    def test_bootstrap_cis_have_positive_width(self):
        """Verify bootstrap CIs have positive width (not degenerate)."""
        with open(TEST_JSON) as f:
            data = json.load(f)

        for cell_key, cell_data in data["data"].items():
            for comp in cell_data["components"]:
                ci_lower = np.array(comp["ci_lower"])
                ci_upper = np.array(comp["ci_upper"])
                ci_width = ci_upper - ci_lower

                # CI width should be positive (with some tolerance for numerical precision)
                assert np.mean(ci_width) > 0, (
                    f"Component {comp['name']} in {cell_key} has zero CI width. "
                    "Bootstrap may not be working correctly."
                )


# Production figure tests
class TestProductionDecompositionGrid:
    """Tests for production figure with real data."""

    PROD_FIGURE = FIGURES_DIR / "fig_decomposition_grid.png"
    PROD_JSON = FIGURES_DIR / "data" / "fig_decomposition_grid.json"

    @pytest.fixture(autouse=True)
    def _require_production_data(self):
        """Skip all tests in this class if production data is not available."""
        if not self.PROD_FIGURE.exists():
            pytest.skip(
                f"Production data not found: {self.PROD_FIGURE}. Run: make analyze"
            )

    def test_production_json_not_synthetic(self):
        """Production JSON should NOT have synthetic marker."""
        if not self.PROD_JSON.exists():
            pytest.skip(
                f"Production data not found: {self.PROD_JSON}. Run: make analyze"
            )

        with open(self.PROD_JSON) as f:
            data = json.load(f)

        assert data.get("synthetic") is not True, (
            "Production figure marked as synthetic! "
            "This is a CRITICAL failure - using fake data in publication."
        )

    def test_production_figure_meets_dpi_requirements(self):
        """Production figure should have DPI >= 300."""

        img = Image.open(self.PROD_FIGURE)
        dpi = img.info.get("dpi", (72, 72))

        assert round(dpi[0]) >= 300, f"DPI too low for publication: {dpi[0]}"
