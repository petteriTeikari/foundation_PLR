#!/usr/bin/env python3
"""
Auto-version config files on content change.

This pre-commit hook:
1. Detects staged config files with content changes
2. Auto-bumps the patch version (1.0.0 -> 1.0.1)
3. Updates the content hash
4. Re-stages the modified files

Usage:
    As pre-commit hook (automatic)
    Manual: python scripts/auto_version_configs.py

This ensures version numbers are always updated when content changes,
eliminating the need for manual version bumping.
"""

import hashlib
import re
import subprocess
import sys
from pathlib import Path

# Constants
CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"

# Patterns
VERSION_PATTERN = re.compile(
    r'^(_version|VERSION|version):\s*["\']?(\d+\.\d+\.\d+)["\']?', re.MULTILINE
)
HASH_PATTERN = re.compile(r'^_content_hash:\s*["\']?([a-f0-9]+)["\']?', re.MULTILINE)

# Files/directories exempt from auto-versioning
EXEMPT_FILES = {"_version_manifest.yaml", "FROZEN_2026-02.yaml"}
EXEMPT_DIRS = {"archived", "schemas"}


def is_exempt(filepath: Path) -> bool:
    """Check if a file is exempt from auto-versioning."""
    if filepath.name in EXEMPT_FILES:
        return True
    if any(part in EXEMPT_DIRS for part in filepath.parts):
        return True
    return False


def get_staged_configs() -> list[Path]:
    """Get list of staged YAML config files."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True,
            text=True,
            check=True,
            cwd=CONFIG_DIR.parent,
        )
        files = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            if line.startswith("configs/") and line.endswith(".yaml"):
                filepath = CONFIG_DIR.parent / line
                if filepath.exists() and not is_exempt(filepath):
                    files.append(filepath)
        return files
    except subprocess.CalledProcessError:
        return []


def get_original_content(filepath: Path) -> str | None:
    """Get file content from last commit (HEAD)."""
    try:
        rel_path = filepath.relative_to(CONFIG_DIR.parent)
        result = subprocess.run(
            ["git", "show", f"HEAD:{rel_path}"],
            capture_output=True,
            text=True,
            check=True,
            cwd=CONFIG_DIR.parent,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return None  # New file


def compute_content_hash(content: str) -> str:
    """Compute hash of content excluding version/hash lines."""
    lines = content.splitlines()
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("_version:", "VERSION:", "version:", "_content_hash:")):
            continue
        filtered_lines.append(line)
    clean = "\n".join(filtered_lines)
    return hashlib.sha256(clean.encode()).hexdigest()[:12]


def bump_patch_version(version: str) -> str:
    """Increment patch version: 1.0.0 -> 1.0.1"""
    parts = version.split(".")
    if len(parts) == 3:
        major, minor, patch = parts
        return f"{major}.{minor}.{int(patch) + 1}"
    elif len(parts) == 2:
        major, minor = parts
        return f"{major}.{minor}.1"
    return version


def process_config(filepath: Path) -> bool:
    """
    Process a single config file.

    Returns True if file was modified, False otherwise.
    """
    current_content = filepath.read_text()
    original_content = get_original_content(filepath)

    # Compute hashes of meaningful content
    current_hash = compute_content_hash(current_content)
    original_hash = compute_content_hash(original_content) if original_content else None

    # No content change (only version/hash changed, or no real change)
    if current_hash == original_hash:
        return False

    # Content changed - need to bump version
    version_match = VERSION_PATTERN.search(current_content)
    if not version_match:
        # No version field - skip (will be caught by other validation)
        return False

    version_key = version_match.group(1)
    old_version = version_match.group(2)
    new_version = bump_patch_version(old_version)

    # Check if version was already bumped by user
    if original_content:
        old_version_match = VERSION_PATTERN.search(original_content)
        if old_version_match and old_version_match.group(2) != old_version:
            # User already bumped version manually - respect their change
            # But still update hash if needed
            pass
        else:
            # Auto-bump version
            new_content = VERSION_PATTERN.sub(
                f'{version_key}: "{new_version}"', current_content
            )
            current_content = new_content

    # Update or add content hash
    if HASH_PATTERN.search(current_content):
        # Update existing hash
        current_content = HASH_PATTERN.sub(
            f'_content_hash: "{current_hash}"', current_content
        )
    else:
        # Add hash after version line
        lines = current_content.splitlines()
        new_lines = []
        hash_added = False
        for line in lines:
            new_lines.append(line)
            if not hash_added and line.strip().startswith(
                ("_version:", "VERSION:", "version:")
            ):
                new_lines.append(f'_content_hash: "{current_hash}"')
                hash_added = True
        if not hash_added:
            # Prepend if no version line found
            new_lines.insert(0, f'_content_hash: "{current_hash}"')
        current_content = "\n".join(new_lines)
        if not current_content.endswith("\n"):
            current_content += "\n"

    # Write updated content
    filepath.write_text(current_content)

    # Re-stage the modified file
    subprocess.run(
        ["git", "add", str(filepath)],
        check=True,
        cwd=CONFIG_DIR.parent,
    )

    rel_path = filepath.relative_to(CONFIG_DIR.parent)
    print(
        f"Auto-versioned: {rel_path} ({old_version} -> {new_version}, hash: {current_hash})"
    )
    return True


def main() -> int:
    """Main entry point for pre-commit hook."""
    configs = get_staged_configs()
    if not configs:
        return 0

    modified = 0
    for config_path in configs:
        try:
            if process_config(config_path):
                modified += 1
        except Exception as e:
            print(f"Warning: Failed to process {config_path}: {e}", file=sys.stderr)

    if modified:
        print(f"\n{modified} config(s) auto-versioned and re-staged")

    return 0


if __name__ == "__main__":
    sys.exit(main())
