#!/usr/bin/env python
"""
Fix JSON provenance metadata.

Adds db_hash to JSON files that are missing it.
Run after generating exports to ensure all have proper provenance.
"""

import hashlib
import json
from pathlib import Path


def compute_db_hash(db_path: Path) -> str:
    """Compute truncated MD5 hash of database."""
    with open(db_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


def fix_json_provenance(json_path: Path, db_path: Path) -> bool:
    """Add db_hash to JSON file if missing.

    Returns True if file was modified.
    """
    with open(json_path) as f:
        data = json.load(f)

    if "metadata" not in data:
        print(f"  SKIP: {json_path.name} - no metadata section")
        return False

    metadata = data["metadata"]
    data_source = metadata.get("data_source")

    # Check if already has db_hash
    if isinstance(data_source, dict) and "db_hash" in data_source:
        print(f"  OK: {json_path.name}")
        return False

    # Need to add db_hash
    db_hash = compute_db_hash(db_path)

    if isinstance(data_source, str):
        # Convert string to dict
        metadata["data_source"] = {
            "database": str(db_path),
            "db_hash": db_hash,
            "note": data_source,  # Preserve original value
        }
    elif isinstance(data_source, dict):
        # Add hash to existing dict
        data_source["db_hash"] = db_hash
        if "database" not in data_source:
            data_source["database"] = str(db_path)
    else:
        # No data_source - create new
        metadata["data_source"] = {
            "database": str(db_path),
            "db_hash": db_hash,
        }

    # Write back
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  FIXED: {json_path.name}")
    return True


def main():
    """Fix provenance for all JSON files in outputs/r_data/."""
    project_root = Path(__file__).parent.parent.parent
    r_data_dir = project_root / "outputs" / "r_data"

    # Default database path
    db_path = project_root / "outputs" / "foundation_plr_results.db"
    if not db_path.exists():
        print(f"WARNING: Database not found at {db_path}")
        # Try STRATOS DB
        db_path = project_root / "outputs" / "foundation_plr_results_stratos.db"
        if not db_path.exists():
            print("ERROR: No database found to compute hash from")
            return

    print(f"Using database: {db_path}")
    print(f"Database hash: {compute_db_hash(db_path)}")
    print()

    fixed_count = 0
    for json_file in sorted(r_data_dir.glob("*.json")):
        if fix_json_provenance(json_file, db_path):
            fixed_count += 1

    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()
