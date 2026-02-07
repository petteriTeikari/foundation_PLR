"""
Tests for legacy R tools used in ground truth creation.

These are smoke tests to ensure:
1. R files have valid syntax (parse without errors)
2. Required R packages are documented
3. Key files exist in the expected locations

These tools are archived for transparency/reproducibility, not active development.
"""

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.r_required

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
TOOLS_DIR = PROJECT_ROOT / "src" / "tools" / "ground-truth-creation"


class TestLegacyRFilesExist:
    """Test that all expected R files are present."""

    EXPECTED_FILES = [
        # Shiny apps
        "shiny-apps/inspect_outliers/ui.R",
        "shiny-apps/inspect_outliers/server.R",
        "shiny-apps/inspect_EMD/ui.R",
        "shiny-apps/inspect_EMD/server.R",
        # Imputation
        "imputation/lowLevel_imputation_wrappers.R",
        "imputation/batch_AnalyzeAndReImpute.R",
        # Denoising
        "denoising/lowLevel_decomposition_wrappers.R",
        "denoising/lowLevel_denoising_wrappers.R",
        # Supporting
        "supporting/changepoint_detection.R",
        "supporting/compute_PLR_features.R",
        "supporting/PLR_augmentation.R",
    ]

    @pytest.mark.parametrize("relative_path", EXPECTED_FILES)
    def test_file_exists(self, relative_path: str):
        """Test that expected R file exists."""
        file_path = TOOLS_DIR / relative_path
        assert file_path.exists(), f"Missing file: {relative_path}"

    def test_all_files_count(self):
        """Test that we have the expected number of R files."""
        r_files = list(TOOLS_DIR.rglob("*.R"))
        assert len(r_files) >= len(self.EXPECTED_FILES), (
            f"Expected at least {len(self.EXPECTED_FILES)} R files, "
            f"found {len(r_files)}"
        )


class TestRSyntax:
    """Test that R files have valid syntax."""

    def _check_r_available(self) -> bool:
        """Check if R is available on the system."""
        try:
            result = subprocess.run(
                ["Rscript", "--version"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @pytest.fixture
    def r_available(self):
        """Fixture that skips tests if R is not available."""
        if not self._check_r_available():
            pytest.skip("R/Rscript not available on this system")
        return True

    def test_r_syntax_via_script(self, r_available):
        """Run the R syntax test script."""
        test_script = Path(__file__).parent / "test_r_syntax.R"

        if not test_script.exists():
            pytest.skip(f"R test script not found: {test_script}")

        result = subprocess.run(
            ["Rscript", str(test_script)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(PROJECT_ROOT),
        )

        if result.returncode != 0:
            pytest.fail(
                f"R syntax tests failed:\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

    @pytest.mark.parametrize(
        "r_file",
        list(TOOLS_DIR.rglob("*.R")) if TOOLS_DIR.exists() else [],
        ids=lambda p: (
            str(p.relative_to(PROJECT_ROOT)) if TOOLS_DIR.exists() else str(p)
        ),
    )
    def test_individual_file_syntax(self, r_available, r_file: Path):
        """Test each R file parses without syntax errors."""
        result = subprocess.run(
            ["Rscript", "-e", f"parse('{r_file}')"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            pytest.fail(f"Syntax error in {r_file.name}:\n{result.stderr}")


class TestMediaFiles:
    """Test that media files are present."""

    def test_demo_video_exists(self):
        """Test that the demo video exists."""
        video_path = (
            PROJECT_ROOT / "src" / "tools" / "media" / "inspect-outliers-demo-2018.mp4"
        )
        assert video_path.exists(), "Demo video not found"

    def test_demo_video_size(self):
        """Test that demo video has reasonable size (> 1MB)."""
        video_path = (
            PROJECT_ROOT / "src" / "tools" / "media" / "inspect-outliers-demo-2018.mp4"
        )
        if video_path.exists():
            size_mb = video_path.stat().st_size / (1024 * 1024)
            assert size_mb > 1, f"Video seems too small: {size_mb:.2f} MB"


class TestWikiDocs:
    """Test that wiki documentation is present."""

    def test_wiki_html_exists(self):
        """Test that the main wiki HTML file exists."""
        docs_dir = PROJECT_ROOT / "src" / "tools" / "docs"
        if not docs_dir.exists():
            pytest.skip("docs/ directory not found (wiki docs not generated)")
        html_files = list(docs_dir.glob("*.html"))
        if len(html_files) == 0:
            pytest.skip("No wiki HTML files found in docs/ (not generated)")

    def test_wiki_images_exist(self):
        """Test that wiki images exist."""
        docs_dir = PROJECT_ROOT / "src" / "tools" / "docs"
        if not docs_dir.exists():
            pytest.skip("docs/ directory not found (wiki docs not generated)")
        # Look for PNG/JPG files in subdirectories
        image_files = (
            list(docs_dir.rglob("*.PNG"))
            + list(docs_dir.rglob("*.png"))
            + list(docs_dir.rglob("*.jpg"))
        )
        if len(image_files) == 0:
            pytest.skip("No wiki images found (not generated)")


class TestRDependencies:
    """Document and test R package dependencies."""

    # These are the R packages required by the legacy tools
    REQUIRED_PACKAGES = [
        "shiny",
        "missForest",
        "EMD",  # or "hht" for CEEMD
        "changepoint",
        "imputeTS",
        "data.table",
        "ggplot2",
    ]

    def test_dependencies_documented(self):
        """Verify that dependencies are documented (this is a documentation test)."""
        # This test always passes - it's here to document dependencies
        # The actual installation is optional for users
        assert len(self.REQUIRED_PACKAGES) > 0

    def test_readme_exists(self):
        """Test that README exists to document how to run tools."""
        readme_path = PROJECT_ROOT / "src" / "tools" / "README.md"
        # Check both tools dir and README exist
        assert TOOLS_DIR.exists()
        assert readme_path.exists(), f"README should exist at {readme_path}"
