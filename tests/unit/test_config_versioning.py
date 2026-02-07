"""
Unit tests for config versioning system.

TDD: Tests written first, implementation follows.

Tests verify:
1. All config files have version fields
2. Version format is semantic (MAJOR.MINOR.PATCH)
3. Content hashes are valid
4. Manifest generation and verification works
"""

import hashlib
import re
from pathlib import Path

import pytest
import yaml


def get_configs_dir() -> Path:
    """Get the configs directory path."""
    return Path(__file__).parent.parent.parent / "configs"


def get_all_config_files() -> list[Path]:
    """Get all YAML config files in configs/."""
    configs_dir = get_configs_dir()
    return sorted(configs_dir.rglob("*.yaml"))


# Files exempt from version requirement (meta files, archives)
EXEMPT_FILES = {
    "_version_manifest.yaml",
    "FROZEN_2026-02.yaml",
}

# Directories exempt from version requirement
EXEMPT_DIRS = {
    "archived",
    "schemas",
}


def is_exempt(path: Path) -> bool:
    """Check if a file is exempt from version requirements."""
    if path.name in EXEMPT_FILES:
        return True
    if any(part in EXEMPT_DIRS for part in path.parts):
        return True
    return False


class TestConfigVersionField:
    """Test that all configs have proper version fields."""

    @pytest.mark.unit
    def test_all_configs_have_version_field(self):
        """All YAML configs must have _version or VERSION field."""
        configs = get_all_config_files()
        missing_version = []

        for config_path in configs:
            if is_exempt(config_path):
                continue

            try:
                content = yaml.safe_load(config_path.read_text())
                if content is None:
                    continue  # Empty file
                if not isinstance(content, dict):
                    continue  # Not a dict (e.g., just a list)

                has_version = (
                    "_version" in content
                    or "VERSION" in content
                    or "version" in content
                )
                if not has_version:
                    rel_path = config_path.relative_to(get_configs_dir())
                    missing_version.append(str(rel_path))
            except yaml.YAMLError:
                pass  # Invalid YAML handled elsewhere

        assert not missing_version, "Config files missing version field:\n" + "\n".join(
            f"  - {f}" for f in missing_version
        )

    @pytest.mark.unit
    def test_version_format_semantic(self):
        """Version must be semantic (MAJOR.MINOR.PATCH or MAJOR.MINOR)."""
        pattern = re.compile(r"^\d+\.\d+(\.\d+)?$")
        configs = get_all_config_files()
        invalid_versions = []

        for config_path in configs:
            if is_exempt(config_path):
                continue

            try:
                content = yaml.safe_load(config_path.read_text())
                if content is None or not isinstance(content, dict):
                    continue

                version = (
                    content.get("_version")
                    or content.get("VERSION")
                    or content.get("version")
                )
                if version is None:
                    continue  # Handled by other test

                version_str = str(version)
                if not pattern.match(version_str):
                    rel_path = config_path.relative_to(get_configs_dir())
                    invalid_versions.append(f"{rel_path}: {version_str}")
            except yaml.YAMLError:
                pass

        assert not invalid_versions, (
            "Config files with invalid version format:\n"
            + "\n".join(f"  - {f}" for f in invalid_versions)
        )


class TestContentHash:
    """Test content hash functionality."""

    @pytest.mark.unit
    def test_content_hash_format(self):
        """Content hash must be 12-char hex if present."""
        pattern = re.compile(r"^[a-f0-9]{12}$")
        configs = get_all_config_files()
        invalid_hashes = []

        for config_path in configs:
            if is_exempt(config_path):
                continue

            try:
                content = yaml.safe_load(config_path.read_text())
                if content is None or not isinstance(content, dict):
                    continue

                content_hash = content.get("_content_hash")
                if content_hash is None:
                    continue  # Hash is optional initially

                if not pattern.match(str(content_hash)):
                    rel_path = config_path.relative_to(get_configs_dir())
                    invalid_hashes.append(f"{rel_path}: {content_hash}")
            except yaml.YAMLError:
                pass

        assert not invalid_hashes, (
            "Config files with invalid content hash format:\n"
            + "\n".join(f"  - {f}" for f in invalid_hashes)
        )

    @pytest.mark.unit
    def test_content_hash_matches_content(self):
        """If content hash present, it must match actual content."""
        configs = get_all_config_files()
        mismatches = []

        for config_path in configs:
            if is_exempt(config_path):
                continue

            try:
                raw_content = config_path.read_text()
                content = yaml.safe_load(raw_content)
                if content is None or not isinstance(content, dict):
                    continue

                declared_hash = content.get("_content_hash")
                if declared_hash is None:
                    continue

                # Compute hash excluding version and hash lines
                lines = raw_content.splitlines()
                filtered_lines = [
                    line
                    for line in lines
                    if not line.strip().startswith("_version:")
                    and not line.strip().startswith("VERSION:")
                    and not line.strip().startswith("version:")
                    and not line.strip().startswith("_content_hash:")
                ]
                clean_content = "\n".join(filtered_lines)
                actual_hash = hashlib.sha256(clean_content.encode()).hexdigest()[:12]

                if declared_hash != actual_hash:
                    rel_path = config_path.relative_to(get_configs_dir())
                    mismatches.append(
                        f"{rel_path}: declared={declared_hash}, actual={actual_hash}"
                    )
            except yaml.YAMLError:
                pass

        assert not mismatches, (
            "Config files with mismatched content hash:\n"
            + "\n".join(f"  - {f}" for f in mismatches)
        )


class TestVersionManifest:
    """Test version manifest generation and verification."""

    @pytest.mark.unit
    def test_manifest_exists(self):
        """Version manifest should exist."""
        manifest_path = get_configs_dir() / "_version_manifest.yaml"
        assert manifest_path.exists(), (
            f"Version manifest not found at {manifest_path}. "
            "Run `python scripts/config_versioning.py generate` to create it."
        )

    @pytest.mark.unit
    def test_manifest_has_required_fields(self):
        """Manifest must have required metadata fields."""
        manifest_path = get_configs_dir() / "_version_manifest.yaml"
        if not manifest_path.exists():
            pytest.skip("Manifest not yet generated")

        manifest = yaml.safe_load(manifest_path.read_text())

        required_fields = ["manifest_version", "generated_at", "git_head", "configs"]
        missing = [f for f in required_fields if f not in manifest]

        assert not missing, f"Manifest missing required fields: {missing}"

    @pytest.mark.unit
    def test_manifest_covers_all_configs(self):
        """Manifest should cover all config files."""
        manifest_path = get_configs_dir() / "_version_manifest.yaml"
        if not manifest_path.exists():
            pytest.skip("Manifest not yet generated")

        manifest = yaml.safe_load(manifest_path.read_text())
        configs_in_manifest = set(manifest.get("configs", {}).keys())

        # Get all actual configs
        configs_dir = get_configs_dir()
        actual_configs = set()
        for config_path in get_all_config_files():
            if is_exempt(config_path):
                continue
            rel_path = str(config_path.relative_to(configs_dir))
            actual_configs.add(rel_path)

        missing_from_manifest = actual_configs - configs_in_manifest
        assert not missing_from_manifest, "Configs not in manifest:\n" + "\n".join(
            f"  - {f}" for f in sorted(missing_from_manifest)
        )


class TestFullVersionFormat:
    """Test full version format (semantic + hash)."""

    @pytest.mark.unit
    def test_full_version_format(self):
        """Full version should be MAJOR.MINOR.PATCH-HASH."""
        manifest_path = get_configs_dir() / "_version_manifest.yaml"
        if not manifest_path.exists():
            pytest.skip("Manifest not yet generated")

        manifest = yaml.safe_load(manifest_path.read_text())
        pattern = re.compile(r"^\d+\.\d+\.\d+-[a-f0-9]{7}$")

        invalid_full_versions = []
        for config_path, info in manifest.get("configs", {}).items():
            full_version = info.get("full_version", "")
            if not pattern.match(full_version):
                invalid_full_versions.append(f"{config_path}: {full_version}")

        assert not invalid_full_versions, (
            "Configs with invalid full version format:\n"
            + "\n".join(f"  - {f}" for f in invalid_full_versions)
        )
