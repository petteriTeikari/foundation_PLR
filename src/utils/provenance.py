"""
Provenance utilities for JSON exports.

Provides standard metadata format for reproducibility.
All JSON exports should use create_metadata() to ensure
consistent provenance tracking.
"""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def compute_file_hash(path: Path, algorithm: str = "md5", length: int = 12) -> str:
    """Compute truncated hash of a file.

    Args:
        path: Path to the file
        algorithm: Hash algorithm (md5, sha256)
        length: Number of hex characters to return

    Returns:
        Truncated hex digest
    """
    if algorithm == "md5":
        hasher = hashlib.md5()
    elif algorithm == "sha256":
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()[:length]


def create_metadata(
    generator: str,
    database_path: Optional[Path] = None,
    table: Optional[str] = None,
    query: Optional[str] = None,
    description: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create standard metadata dict for JSON exports.

    Args:
        generator: Name of the generating script (e.g., 'scripts/export_foo.py')
        database_path: Path to source database (will compute hash)
        table: Source table name
        query: SQL query used (optional)
        description: Human-readable description
        extra: Additional metadata fields

    Returns:
        Metadata dict with provenance information
    """
    metadata = {
        "created": datetime.now().isoformat(),
        "schema_version": "1.0",
        "generator": generator,
    }

    if description:
        metadata["description"] = description

    if database_path:
        data_source = {
            "database": str(database_path),
        }
        if database_path.exists():
            data_source["db_hash"] = compute_file_hash(database_path)
        if table:
            data_source["table"] = table
        if query:
            data_source["query"] = query
        metadata["data_source"] = data_source

    if extra:
        metadata.update(extra)

    return metadata


def add_provenance_to_export(
    data: Dict[str, Any],
    generator: str,
    database_path: Optional[Path] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Add provenance metadata to an export dict.

    If 'metadata' key exists, merges with existing.
    Otherwise, creates new metadata section.

    Args:
        data: Export data dict
        generator: Generating script name
        database_path: Source database path
        **kwargs: Additional args for create_metadata()

    Returns:
        Data dict with metadata added/updated
    """
    new_metadata = create_metadata(
        generator=generator,
        database_path=database_path,
        **kwargs,
    )

    if "metadata" in data:
        # Merge with existing, but ensure db_hash is added
        existing = data["metadata"]
        if "data_source" in new_metadata and isinstance(
            existing.get("data_source"), str
        ):
            # Convert string data_source to dict
            existing["data_source"] = new_metadata["data_source"]
        elif "data_source" in new_metadata:
            existing["data_source"] = new_metadata["data_source"]
        data["metadata"] = existing
    else:
        data["metadata"] = new_metadata

    return data
