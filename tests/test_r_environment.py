"""
Test R environment setup with renv.

TDD: These tests are written BEFORE renv initialization to define
the expected state of the R environment.

Run with: pytest tests/test_r_environment.py -v
"""

import json
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.r_required

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


class TestRenvLockExists:
    """Tests for renv.lock file existence and validity."""

    def test_renv_lock_exists(self):
        """renv.lock must exist after initialization."""
        renv_lock = PROJECT_ROOT / "renv.lock"
        assert renv_lock.exists(), (
            'renv.lock not found. Run: Rscript -e "renv::init(); renv::snapshot()"'
        )

    def test_renv_lock_valid_json(self):
        """renv.lock must be valid JSON."""
        renv_lock = PROJECT_ROOT / "renv.lock"
        assert renv_lock.exists(), (
            "renv.lock missing. Run: Rscript -e 'renv::init(); renv::snapshot()'"
        )

        with open(renv_lock) as f:
            data = json.load(f)

        assert "R" in data, "Missing R section in renv.lock"
        assert "Packages" in data, "Missing Packages section in renv.lock"

    def test_renv_lock_has_r_version(self):
        """renv.lock must specify R version."""
        renv_lock = PROJECT_ROOT / "renv.lock"
        assert renv_lock.exists(), (
            "renv.lock missing. Run: Rscript -e 'renv::init(); renv::snapshot()'"
        )

        with open(renv_lock) as f:
            data = json.load(f)

        r_section = data.get("R", {})
        assert "Version" in r_section, "R version not specified in renv.lock"
        # Should be R 4.x
        version = r_section["Version"]
        assert version.startswith("4."), f"Expected R 4.x, got {version}"


class TestCriticalPackages:
    """Tests for critical R packages required for STRATOS compliance."""

    # Critical packages with optional version pinning
    # None means any version is acceptable
    CRITICAL_PACKAGES = {
        # Visualization
        "ggplot2": None,
        "dplyr": None,
        "tidyr": None,
        "scales": None,
        "patchwork": None,
        # Statistics (STRATOS compliance)
        "pROC": None,  # ROC curves
        "dcurves": None,  # Decision curve analysis
        # Model validation (Riley 2023)
        "pminternal": None,  # Internal validation, bootstrap stability
    }

    def test_critical_packages_in_lockfile(self):
        """All critical packages must be in renv.lock."""
        renv_lock = PROJECT_ROOT / "renv.lock"
        assert renv_lock.exists(), (
            "renv.lock missing. Run: Rscript -e 'renv::init(); renv::snapshot()'"
        )

        with open(renv_lock) as f:
            data = json.load(f)

        packages = data.get("Packages", {})

        missing = []
        for pkg in self.CRITICAL_PACKAGES:
            if pkg not in packages:
                missing.append(pkg)

        assert not missing, f"Critical packages missing from renv.lock: {missing}"

    def test_package_versions_if_pinned(self):
        """Pinned packages must have correct version."""
        pinned = {p: v for p, v in self.CRITICAL_PACKAGES.items() if v is not None}
        if not pinned:
            return  # No versions pinned — nothing to check

        renv_lock = PROJECT_ROOT / "renv.lock"
        assert renv_lock.exists(), "renv.lock not found"

        with open(renv_lock) as f:
            data = json.load(f)

        packages = data.get("Packages", {})
        mismatches = []
        for package, expected_version in pinned.items():
            if package not in packages:
                mismatches.append(
                    f"{package}: not installed (expected {expected_version})"
                )
                continue
            actual_version = packages[package].get("Version", "")
            if actual_version != expected_version:
                mismatches.append(
                    f"{package}: {actual_version} (expected {expected_version})"
                )

        assert not mismatches, f"Version mismatches: {mismatches}"


class TestRenvFiles:
    """Tests for renv supporting files."""

    def test_rprofile_exists(self):
        """.Rprofile must exist for renv activation."""
        rprofile = PROJECT_ROOT / ".Rprofile"
        assert rprofile.exists(), ".Rprofile not found"

    def test_rprofile_activates_renv(self):
        """.Rprofile must source renv/activate.R."""
        rprofile = PROJECT_ROOT / ".Rprofile"
        assert rprofile.exists(), ".Rprofile missing"

        content = rprofile.read_text()
        assert "renv/activate.R" in content, (
            ".Rprofile does not activate renv. "
            'Should contain: source("renv/activate.R")'
        )

    def test_renv_activate_exists(self):
        """renv/activate.R must exist."""
        activate = PROJECT_ROOT / "renv" / "activate.R"
        assert activate.exists(), "renv/activate.R not found"

    def test_renv_settings_exists(self):
        """renv/settings.json must exist."""
        settings = PROJECT_ROOT / "renv" / "settings.json"
        # settings.json is optional but recommended
        assert settings.exists(), "renv/settings.json missing"

        with open(settings) as f:
            data = json.load(f)
        # Should be valid JSON
        assert isinstance(data, dict)


class TestRenvRestore:
    """Tests for renv::restore() functionality."""

    @pytest.mark.slow
    def test_renv_restore_succeeds(self):
        """renv::restore() must complete without errors."""
        renv_lock = PROJECT_ROOT / "renv.lock"
        assert renv_lock.exists(), (
            "renv.lock missing. Run: Rscript -e 'renv::init(); renv::snapshot()'"
        )

        result = subprocess.run(
            [
                "Rscript",
                "-e",
                "renv::restore(prompt=FALSE, clean=FALSE)",
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=PROJECT_ROOT,
        )

        # Check for success
        assert result.returncode == 0, f"renv::restore failed: {result.stderr}"

    @pytest.mark.slow
    def test_critical_packages_load(self):
        """Critical packages must load without error."""
        renv_lock = PROJECT_ROOT / "renv.lock"
        assert renv_lock.exists(), (
            "renv.lock missing. Run: Rscript -e 'renv::init(); renv::snapshot()'"
        )

        # R code to load critical packages
        r_code = """
        packages <- c("ggplot2", "dplyr", "pROC", "dcurves", "pminternal")
        for (pkg in packages) {
            if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
                stop(paste("Failed to load:", pkg))
            }
        }
        cat("SUCCESS\\n")
        """

        result = subprocess.run(
            ["Rscript", "-e", r_code],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=PROJECT_ROOT,
        )

        assert "SUCCESS" in result.stdout, (
            f"Failed to load critical packages: {result.stderr}"
        )


class TestGitignore:
    """Tests for proper renv gitignore setup."""

    def test_renv_library_ignored(self):
        """renv/library/ should be in .gitignore."""
        gitignore = PROJECT_ROOT / ".gitignore"
        assert gitignore.exists(), ".gitignore missing"

        content = gitignore.read_text()
        # Check for renv library ignore patterns
        renv_patterns = ["renv/library", "renv/local", "renv/cellar", "renv/staging"]
        found_any = any(pattern in content for pattern in renv_patterns)
        assert found_any, (
            "renv/library/ not found in .gitignore. "
            "Run: echo 'renv/library/' >> .gitignore"
        )
