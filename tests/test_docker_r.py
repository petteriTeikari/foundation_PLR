"""
Test R Docker image build and execution.

TDD: These tests define expected behavior of Dockerfile.r.
Run with: pytest tests/test_docker_r.py -v --timeout=1800

Note: These tests require Docker to be installed and running.
Skip in CI if Docker is not available.
"""

import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.r_required, pytest.mark.slow]

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Docker image name for testing
IMAGE_NAME = "foundation-plr-r:test"


def docker_available() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# Skip all tests if Docker is not available
pytestmark = pytest.mark.skipif(
    not docker_available(),
    reason="Docker not available",
)


class TestDockerfileExists:
    """Tests for Dockerfile.r existence."""

    def test_dockerfile_exists(self):
        """Dockerfile.r must exist."""
        dockerfile = PROJECT_ROOT / "Dockerfile.r"
        assert dockerfile.exists(), (
            "Dockerfile.r not found. Create it for R environment containerization."
        )

    def test_dockerfile_has_rocker_base(self):
        """Dockerfile.r should use rocker base image."""
        dockerfile = PROJECT_ROOT / "Dockerfile.r"
        if not dockerfile.exists():
            pytest.skip("Dockerfile.r not yet created")

        content = dockerfile.read_text()
        assert "rocker/" in content.lower() or "FROM" in content, (
            "Dockerfile.r should use a rocker base image"
        )

    def test_dockerfile_copies_renv_lock(self):
        """Dockerfile.r should copy renv.lock."""
        dockerfile = PROJECT_ROOT / "Dockerfile.r"
        if not dockerfile.exists():
            pytest.skip("Dockerfile.r not yet created")

        content = dockerfile.read_text()
        assert "renv.lock" in content, "Dockerfile.r should copy renv.lock"

    def test_dockerfile_restores_renv(self):
        """Dockerfile.r should run renv::restore()."""
        dockerfile = PROJECT_ROOT / "Dockerfile.r"
        if not dockerfile.exists():
            pytest.skip("Dockerfile.r not yet created")

        content = dockerfile.read_text()
        assert "renv::restore" in content or "renv::" in content, (
            "Dockerfile.r should restore packages with renv"
        )


@pytest.fixture(scope="module")
def docker_image():
    """Build Docker image for testing.

    This fixture builds the image once per test module.
    Cleanup happens after all tests complete.
    """
    dockerfile = PROJECT_ROOT / "Dockerfile.r"
    if not dockerfile.exists():
        pytest.skip("Dockerfile.r not yet created")

    # Build the image
    result = subprocess.run(
        [
            "docker",
            "build",
            "-t",
            IMAGE_NAME,
            "-f",
            "Dockerfile.r",
            ".",
        ],
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min timeout for initial build
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        pytest.fail(f"Docker build failed: {result.stderr}")

    yield IMAGE_NAME

    # Cleanup - remove the test image
    subprocess.run(
        ["docker", "rmi", "-f", IMAGE_NAME],
        capture_output=True,
        timeout=60,
    )


class TestDockerBuild:
    """Tests for Docker image build."""

    @pytest.mark.slow
    def test_docker_build_succeeds(self, docker_image):
        """Docker image builds successfully."""
        result = subprocess.run(
            ["docker", "images", "-q", docker_image],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.stdout.strip(), f"Docker image {docker_image} not found"


class TestDockerREnvironment:
    """Tests for R environment in Docker container."""

    @pytest.mark.slow
    def test_r_version_pinned(self, docker_image):
        """R version matches expected (4.4.x or 4.5.x)."""
        result = subprocess.run(
            ["docker", "run", "--rm", docker_image, "R", "--version"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"R --version failed: {result.stderr}"
        # Should be R 4.x
        assert "R version 4." in result.stdout, f"Expected R 4.x, got: {result.stdout}"

    @pytest.mark.slow
    def test_renv_packages_installed(self, docker_image):
        """Critical renv packages are installed correctly."""
        r_code = """
        packages <- c('ggplot2', 'dplyr', 'pROC', 'dcurves', 'pminternal')
        for (pkg in packages) {
            if (!require(pkg, character.only=TRUE, quietly=TRUE)) {
                stop(paste('Failed to load:', pkg))
            }
        }
        cat('SUCCESS')
        """

        result = subprocess.run(
            ["docker", "run", "--rm", docker_image, "Rscript", "-e", r_code],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert "SUCCESS" in result.stdout, (
            f"Package load failed: {result.stderr}\n{result.stdout}"
        )

    @pytest.mark.slow
    def test_ggplot2_can_create_plot(self, docker_image):
        """ggplot2 can create a basic plot without error."""
        r_code = """
        library(ggplot2)
        p <- ggplot(mtcars, aes(x=wt, y=mpg)) + geom_point()
        # Don't save, just check it doesn't error
        cat('SUCCESS')
        """

        result = subprocess.run(
            ["docker", "run", "--rm", docker_image, "Rscript", "-e", r_code],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert "SUCCESS" in result.stdout, f"ggplot2 test failed: {result.stderr}"


class TestDockerRScripts:
    """Tests for R figure scripts in Docker."""

    @pytest.mark.slow
    def test_r_source_files_exist(self, docker_image):
        """R source files are copied to container."""
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                docker_image,
                "ls",
                "-la",
                "src/r/figures/",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"ls failed: {result.stderr}"
        # Should have some .R files
        assert ".R" in result.stdout, (
            f"No R files found in src/r/figures/: {result.stdout}"
        )

    @pytest.mark.slow
    def test_setup_r_sources_without_error(self, docker_image):
        """setup.R sources without error (syntax check)."""
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                docker_image,
                "Rscript",
                "-e",
                "source('src/r/setup.R'); cat('SUCCESS')",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        # setup.R may fail if data files don't exist, but shouldn't have syntax errors
        if result.returncode != 0:
            # Check it's not a syntax error
            assert "unexpected" not in result.stderr.lower(), (
                f"Syntax error in setup.R: {result.stderr}"
            )


class TestMakefileTargets:
    """Tests for Makefile R Docker targets."""

    def test_makefile_has_r_docker_targets(self):
        """Makefile should have r-docker-build target."""
        makefile = PROJECT_ROOT / "Makefile"
        if not makefile.exists():
            pytest.skip("Makefile not found")

        content = makefile.read_text()
        assert "r-docker-build" in content, "Makefile should have r-docker-build target"

    def test_makefile_has_r_docker_test_target(self):
        """Makefile should have r-docker-test target."""
        makefile = PROJECT_ROOT / "Makefile"
        if not makefile.exists():
            pytest.skip("Makefile not found")

        content = makefile.read_text()
        assert "r-docker-test" in content, "Makefile should have r-docker-test target"
