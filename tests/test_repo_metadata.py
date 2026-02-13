"""
Guardrail tests for repository metadata correctness.

Verifies that pyproject.toml, DATA_MANIFEST.yaml, and CITATION.cff
have no placeholder values and accurately reflect the repo state.

Part of: TRIPOD-Code repo housekeeping (T2, T4, T7)
"""

import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent


@pytest.mark.guardrail
class TestPyprojectMetadata:
    """Verify pyproject.toml has no placeholder values."""

    def test_no_placeholder_description(self):
        """pyproject.toml must not have the default 'Add your description here'."""
        toml_path = REPO_ROOT / "pyproject.toml"
        content = toml_path.read_text()
        assert "Add your description here" not in content, (
            "pyproject.toml still has placeholder description. "
            "Replace with actual project description."
        )

    def test_description_is_meaningful(self):
        """pyproject.toml description should mention the project domain."""
        toml_path = REPO_ROOT / "pyproject.toml"
        content = toml_path.read_text()
        # Should mention key terms from the project
        assert any(
            term in content.lower()
            for term in ["pupillary", "plr", "glaucoma", "foundation model"]
        ), "pyproject.toml description should reference the project domain"


@pytest.mark.guardrail
class TestDataManifestAccuracy:
    """Verify DATA_MANIFEST.yaml reflects actual git tracking state."""

    def test_git_tracked_fields_match_reality(self):
        """DATA_MANIFEST.yaml git_tracked fields must match actual git state."""
        git_dir = REPO_ROOT / ".git"
        if not git_dir.exists():
            pytest.skip("No .git directory (e.g., Docker build)")

        manifest_path = REPO_ROOT / "data" / "public" / "DATA_MANIFEST.yaml"
        if not manifest_path.exists():
            pytest.skip("DATA_MANIFEST.yaml not found")

        manifest = yaml.safe_load(manifest_path.read_text())
        files = manifest.get("files", {})

        for filename, meta in files.items():
            claimed_tracked = meta.get("git_tracked", False)
            filepath = REPO_ROOT / "data" / "public" / filename

            if not filepath.exists():
                continue  # File doesn't exist, can't verify

            # Check actual git tracking state
            result = subprocess.run(
                ["git", "ls-files", f"data/public/{filename}"],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )
            actually_tracked = bool(result.stdout.strip())

            assert claimed_tracked == actually_tracked, (
                f"DATA_MANIFEST.yaml says {filename} git_tracked={claimed_tracked}, "
                f"but actual git state is tracked={actually_tracked}. "
                f"Update the manifest to match reality."
            )


@pytest.mark.guardrail
class TestCitationCff:
    """Verify CITATION.cff has required fields."""

    def test_has_version_field(self):
        """CITATION.cff must have a version field."""
        cff_path = REPO_ROOT / "CITATION.cff"
        if not cff_path.exists():
            pytest.skip("CITATION.cff not found")

        content = yaml.safe_load(cff_path.read_text())
        assert "version" in content, (
            "CITATION.cff missing 'version' field. "
            "Required for GitHub 'Cite this repository' feature."
        )

    def test_has_date_released_field(self):
        """CITATION.cff must have a date-released field."""
        cff_path = REPO_ROOT / "CITATION.cff"
        if not cff_path.exists():
            pytest.skip("CITATION.cff not found")

        content = yaml.safe_load(cff_path.read_text())
        assert "date-released" in content, (
            "CITATION.cff missing 'date-released' field. "
            "Required for GitHub 'Cite this repository' feature."
        )
