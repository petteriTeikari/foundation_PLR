#!/usr/bin/env python3
"""
Export subject traces from DuckDB to JSON for R visualization.

This script:
1. Reads demo subject configuration from YAML
2. Extracts trace data from SERI_PLR_GLAUCOMA.db
3. Anonymizes subject codes (PLRxxxx -> Hxxx/Gxxx)
4. Exports to JSON for R consumption

Usage:
    uv run python scripts/export_subject_traces_for_r.py

Output:
    data/r_data/subject_traces.json
"""

import json
import math
from datetime import datetime
from pathlib import Path

import duckdb
import yaml


def find_project_root() -> Path:
    """Find project root by looking for marker files."""
    markers = ["pyproject.toml", "CLAUDE.md", ".git"]
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if any((current / m).exists() for m in markers):
            return current
        current = current.parent
    raise RuntimeError("Could not find project root")


PROJECT_ROOT = find_project_root()


def load_demo_subjects_config() -> dict:
    """Load demo subjects configuration from YAML."""
    config_path = PROJECT_ROOT / "configs" / "VISUALIZATION" / "demo_subjects.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_subject_lookup() -> dict:
    """Load subject code to anonymized ID mapping.

    Returns mapping of PLRxxxx -> Hxxx/Gxxx
    """
    lookup_path = PROJECT_ROOT / "data" / "private" / "subject_lookup.yaml"
    if not lookup_path.exists():
        # Generate lookup on the fly based on config
        print("  Note: subject_lookup.yaml not found, generating from config")
        return {}

    with open(lookup_path) as f:
        return yaml.safe_load(f)


def get_all_demo_subjects(config: dict) -> list[dict]:
    """Extract flat list of demo subjects from nested config structure."""
    subjects = []

    # Counter for anonymized IDs
    control_count = 0
    glaucoma_count = 0

    for class_label in ["control", "glaucoma"]:
        class_data = config["demo_subjects"].get(class_label, {})
        for outlier_level in ["high_outlier", "average_outlier", "low_outlier"]:
            level_subjects = class_data.get(outlier_level, [])
            for subj in level_subjects:
                # Generate anonymized ID
                if class_label == "control":
                    control_count += 1
                    anon_id = f"H{control_count:03d}"
                else:
                    glaucoma_count += 1
                    anon_id = f"G{glaucoma_count:03d}"

                subjects.append(
                    {
                        "plr_code": subj["code"],
                        "anon_id": anon_id,
                        "class_label": class_label,
                        "outlier_pct": subj["outlier_pct"],
                        "note": subj["note"],
                        "outlier_level": outlier_level,
                    }
                )

    return subjects


def extract_subject_data(conn: duckdb.DuckDBPyConnection, plr_code: str) -> dict:
    """Extract trace data for a single subject from DuckDB."""
    query = """
    SELECT
        time,
        pupil_orig,
        pupil_gt,
        outlier_mask,
        Blue as blue_stimulus,
        Red as red_stimulus
    FROM train
    WHERE subject_code = ?
    ORDER BY time
    """
    df = conn.execute(query, [plr_code]).fetchdf()

    if len(df) == 0:
        # Try test table
        query = query.replace("FROM train", "FROM test")
        df = conn.execute(query, [plr_code]).fetchdf()

    if len(df) == 0:
        raise ValueError(f"No data found for subject {plr_code}")

    # Convert to lists, handling NaN -> None for JSON
    def to_list(series):
        return [
            None if (isinstance(v, float) and math.isnan(v)) else float(v)
            for v in series.values
        ]

    return {
        "time": to_list(df["time"]),
        "pupil_orig": to_list(df["pupil_orig"]),  # Original signal (show in gray)
        "pupil_gt": to_list(df["pupil_gt"]),  # Ground truth (show in black)
        "outlier_mask": [int(v) for v in df["outlier_mask"].values],
        "blue_stimulus": to_list(df["blue_stimulus"]),
        "red_stimulus": to_list(df["red_stimulus"]),
        "n_timepoints": len(df),
    }


def calculate_global_y_range(subjects_data: list[dict]) -> dict:
    """Return fixed y-axis range for consistent visualization.

    Uses fixed range [-60, 10] (percent change from baseline) for
    publication-quality figures with consistent visual comparison.
    """
    # Fixed range for publication figures (percent change from baseline)
    y_min = -70.0
    y_max = 10.0

    # Still calculate actual data ranges for reference
    all_orig = []
    all_gt = []

    for subj in subjects_data:
        all_orig.extend([v for v in subj["pupil_orig"] if v is not None])
        all_gt.extend([v for v in subj["pupil_gt"] if v is not None])

    return {
        "y_min": y_min,
        "y_max": y_max,
        "orig_range": [round(min(all_orig), 1), round(max(all_orig), 1)],
        "gt_range": [round(min(all_gt), 1), round(max(all_gt), 1)],
    }


def detect_light_protocol_timing(subjects_data: list[dict]) -> dict:
    """Detect light protocol timing from actual stimulus data.

    Returns approximate start/end times for blue and red stimuli.
    """
    # Use first subject's data to detect timing
    subj = subjects_data[0]
    time = subj["time"]
    blue = subj["blue_stimulus"]
    red = subj["red_stimulus"]

    # Find when blue is active (non-zero)
    blue_active_times = [t for t, b in zip(time, blue) if b is not None and b > 0]
    # Find when red is active (non-zero)
    red_active_times = [t for t, r in zip(time, red) if r is not None and r > 0]

    protocol = {
        "blue_start": min(blue_active_times) if blue_active_times else 19.0,
        "blue_end": max(blue_active_times) if blue_active_times else 29.0,
        "red_start": min(red_active_times) if red_active_times else 49.0,
        "red_end": max(red_active_times) if red_active_times else 59.0,
        "wavelength_blue_nm": 469,
        "wavelength_red_nm": 640,
    }

    return protocol


def main():
    """Main extraction function."""
    print("=" * 60)
    print("Exporting Subject Traces for R Visualization")
    print("=" * 60)

    # Load configuration
    print("\n1. Loading configuration...")
    config = load_demo_subjects_config()
    demo_subjects = get_all_demo_subjects(config)
    print(f"   Found {len(demo_subjects)} demo subjects")

    # Connect to database
    db_path = PROJECT_ROOT.parent / "SERI_PLR_GLAUCOMA.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    print(f"\n2. Connecting to database: {db_path}")
    conn = duckdb.connect(str(db_path), read_only=True)

    # Extract data for each subject
    print("\n3. Extracting subject data...")
    subjects_output = []

    for subj in demo_subjects:
        print(
            f"   {subj['plr_code']} -> {subj['anon_id']} ({subj['class_label']}, {subj['outlier_pct']:.1f}%)"
        )

        data = extract_subject_data(conn, subj["plr_code"])

        subjects_output.append(
            {
                "subject_id": subj["anon_id"],
                "class_label": subj["class_label"],
                "outlier_pct": subj["outlier_pct"],
                "note": subj["note"],
                "outlier_level": subj["outlier_level"],
                **data,
            }
        )

    conn.close()

    # Detect light protocol timing
    print("\n4. Detecting light protocol timing...")
    light_protocol = detect_light_protocol_timing(subjects_output)
    print(
        f"   Blue: {light_protocol['blue_start']:.1f}s - {light_protocol['blue_end']:.1f}s"
    )
    print(
        f"   Red:  {light_protocol['red_start']:.1f}s - {light_protocol['red_end']:.1f}s"
    )

    # Calculate global y-axis range for consistent visualization
    print("\n5. Calculating global y-axis range...")
    y_range = calculate_global_y_range(subjects_output)
    print(f"   Global range: [{y_range['y_min']:.1f}, {y_range['y_max']:.1f}]")
    print(f"   Orig range:   {y_range['orig_range']}")
    print(f"   GT range:     {y_range['gt_range']}")

    # Build output JSON
    output = {
        "metadata": {
            "created": datetime.utcnow().isoformat() + "Z",
            "generator": "scripts/export_subject_traces_for_r.py",
            "source_db": "SERI_PLR_GLAUCOMA.db",
            "n_subjects": len(subjects_output),
            "n_control": sum(
                1 for s in subjects_output if s["class_label"] == "control"
            ),
            "n_glaucoma": sum(
                1 for s in subjects_output if s["class_label"] == "glaucoma"
            ),
            "light_protocol": light_protocol,
            "y_axis_range": y_range,
        },
        "subjects": subjects_output,
    }

    # Write output
    output_dir = PROJECT_ROOT / "data" / "r_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "subject_traces.json"

    print(f"\n6. Writing output to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Verify
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   File size: {file_size_mb:.2f} MB")

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
