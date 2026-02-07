#!/usr/bin/env python3
"""
Config versioning system for Foundation PLR.

Provides:
- Manifest generation: Track all config file versions and hashes
- Manifest verification: Ensure configs match declared versions
- Content hashing: Cryptographic verification of config content

Usage:
    python scripts/config_versioning.py generate  # Generate/update manifest
    python scripts/config_versioning.py verify    # Verify configs match manifest
    python scripts/config_versioning.py show      # Show all config versions

The manifest file (configs/_version_manifest.yaml) tracks:
- Semantic version (_version field in each config)
- Content hash (SHA256 of content excluding version/hash lines)
- Git commit that last modified the file
- Full version (semantic-hash format: v1.0.0-8f3a7bc)
"""

import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Constants
CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"
MANIFEST_PATH = CONFIG_DIR / "_version_manifest.yaml"

# Files/directories exempt from versioning
EXEMPT_FILES = {"_version_manifest.yaml", "FROZEN_2026-02.yaml"}
EXEMPT_DIRS = {"archived", "schemas"}

# Patterns for extracting/excluding version info
VERSION_KEYS = {"_version", "VERSION", "version"}
HASH_KEY = "_content_hash"


def is_exempt(filepath: Path) -> bool:
    """Check if a file is exempt from version requirements."""
    if filepath.name in EXEMPT_FILES:
        return True
    if any(part in EXEMPT_DIRS for part in filepath.parts):
        return True
    return False


def get_all_config_files() -> list[Path]:
    """Get all YAML config files, excluding exempt ones."""
    return sorted(
        f for f in CONFIG_DIR.rglob("*.yaml") if not is_exempt(f) and f.is_file()
    )


def compute_content_hash(filepath: Path) -> str:
    """
    Compute SHA256 hash of config content.

    Excludes version and hash lines so that updating these
    doesn't change the content hash.
    """
    content = filepath.read_text()
    lines = content.splitlines()

    # Filter out version and hash lines
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip version lines
        if any(stripped.startswith(f"{key}:") for key in VERSION_KEYS):
            continue
        # Skip hash line
        if stripped.startswith(f"{HASH_KEY}:"):
            continue
        filtered_lines.append(line)

    clean_content = "\n".join(filtered_lines)
    return hashlib.sha256(clean_content.encode()).hexdigest()[:12]


def get_git_last_modified_commit(filepath: Path) -> str:
    """Get short SHA of last commit that modified this file."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", str(filepath)],
            capture_output=True,
            text=True,
            check=True,
            cwd=CONFIG_DIR.parent,
        )
        sha = result.stdout.strip()
        return sha if sha else "uncommitted"
    except subprocess.CalledProcessError:
        return "unknown"


def get_git_head() -> str:
    """Get current HEAD commit short SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=CONFIG_DIR.parent,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def get_semantic_version(filepath: Path) -> str:
    """Extract semantic version from config file."""
    try:
        content = yaml.safe_load(filepath.read_text())
        if not isinstance(content, dict):
            return "0.0.0"

        for key in VERSION_KEYS:
            if key in content:
                return str(content[key])
        return "0.0.0"
    except (yaml.YAMLError, Exception):
        return "0.0.0"


def generate_manifest() -> dict[str, Any]:
    """Generate complete version manifest for all configs."""
    configs = get_all_config_files()

    manifest = {
        "manifest_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_head": get_git_head(),
        "total_configs": len(configs),
        "configs": {},
    }

    for config_path in configs:
        rel_path = str(config_path.relative_to(CONFIG_DIR))
        semantic = get_semantic_version(config_path)
        content_hash = compute_content_hash(config_path)
        git_commit = get_git_last_modified_commit(config_path)

        manifest["configs"][rel_path] = {
            "semantic": semantic,
            "content_hash": content_hash,
            "git_commit": git_commit,
            "full_version": f"{semantic}-{content_hash[:7]}",
        }

    return manifest


def save_manifest(manifest: dict[str, Any]) -> None:
    """Save manifest to YAML file."""
    MANIFEST_PATH.write_text(
        yaml.dump(
            manifest, sort_keys=False, default_flow_style=False, allow_unicode=True
        )
    )


def load_manifest() -> dict[str, Any] | None:
    """Load existing manifest or return None."""
    if not MANIFEST_PATH.exists():
        return None
    return yaml.safe_load(MANIFEST_PATH.read_text())


def verify_manifest() -> list[str]:
    """
    Verify current configs match saved manifest.

    Returns list of discrepancies (empty if all match).
    """
    saved = load_manifest()
    if saved is None:
        return ["Manifest does not exist. Run 'generate' first."]

    current = generate_manifest()
    discrepancies = []

    # Check for changed or new configs
    for path, info in current["configs"].items():
        if path not in saved["configs"]:
            discrepancies.append(f"NEW: {path} (not in manifest)")
        elif saved["configs"][path]["content_hash"] != info["content_hash"]:
            old_ver = saved["configs"][path]["full_version"]
            new_ver = info["full_version"]
            discrepancies.append(f"CHANGED: {path} ({old_ver} -> {new_ver})")

    # Check for deleted configs
    for path in saved["configs"]:
        if path not in current["configs"]:
            discrepancies.append(f"DELETED: {path} (in manifest but not on disk)")

    return discrepancies


def add_version_to_config(filepath: Path, version: str = "1.0.0") -> bool:
    """
    Add _version field to a config file if missing.

    Returns True if modified, False otherwise.
    """
    try:
        content = filepath.read_text()
        data = yaml.safe_load(content)

        if not isinstance(data, dict):
            return False

        # Check if version already exists
        if any(key in data for key in VERSION_KEYS):
            return False

        # Add version at the beginning
        lines = content.splitlines()
        new_lines = [f'_version: "{version}"'] + lines
        filepath.write_text("\n".join(new_lines) + "\n")
        return True
    except Exception:
        return False


def show_versions() -> None:
    """Print all config versions in a readable format."""
    manifest = generate_manifest()
    print(f"Config versions (HEAD: {manifest['git_head']})")
    print(f"Generated: {manifest['generated_at']}")
    print(f"Total: {manifest['total_configs']} configs")
    print("-" * 70)

    for path, info in sorted(manifest["configs"].items()):
        print(f"{info['full_version']:20s} {path}")


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/config_versioning.py <command>")
        print("Commands:")
        print("  generate  - Generate/update version manifest")
        print("  verify    - Verify configs match manifest")
        print("  show      - Show all config versions")
        print("  add-versions - Add _version field to all configs missing it")
        return 1

    command = sys.argv[1]

    if command == "generate":
        manifest = generate_manifest()
        save_manifest(manifest)
        print(f"Generated manifest with {manifest['total_configs']} configs")
        print(f"Saved to: {MANIFEST_PATH}")
        return 0

    elif command == "verify":
        discrepancies = verify_manifest()
        if discrepancies:
            print("Config verification FAILED:")
            for d in discrepancies:
                print(f"  - {d}")
            return 1
        else:
            print("All configs match manifest.")
            return 0

    elif command == "show":
        show_versions()
        return 0

    elif command == "add-versions":
        configs = get_all_config_files()
        modified = 0
        for config_path in configs:
            if add_version_to_config(config_path):
                rel_path = config_path.relative_to(CONFIG_DIR)
                print(f"Added _version to: {rel_path}")
                modified += 1
        print(f"\nModified {modified} files")
        return 0

    else:
        print(f"Unknown command: {command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
