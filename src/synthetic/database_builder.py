"""
DuckDB database builder for synthetic PLR data.

Assembles synthetic subjects into a DuckDB database matching the schema
of SERI_PLR_GLAUCOMA.db with tables 'train' and 'test'.
"""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import numpy as np
import yaml

from src.synthetic.artifact_injection import create_imputed_signals, inject_artifacts
from src.synthetic.plr_generator import generate_light_stimuli, generate_plr_curve


def generate_subject_data(
    subject_code: str,
    class_label: str,
    split: str,
    seed: int,
    outlier_pct: float,
) -> Dict[str, np.ndarray]:
    """
    Generate all data for a single synthetic subject.

    Parameters
    ----------
    subject_code : str
        Subject identifier (e.g., "SYNTH_H001")
    class_label : str
        "control" or "glaucoma"
    split : str
        "train" or "test"
    seed : int
        Random seed for this subject
    outlier_pct : float
        Outlier percentage (0.02 to 0.40)

    Returns
    -------
    data : dict
        Dictionary with all columns for this subject
    """
    rng = np.random.default_rng(seed)

    # Generate PLR curve
    time, pupil_gt = generate_plr_curve(class_label, seed)
    n = len(time)

    # Inject artifacts
    pupil_orig, pupil_raw, outlier_mask = inject_artifacts(pupil_gt, outlier_pct, seed)

    # Create imputed versions
    pupil_orig_imputed, pupil_raw_imputed, imputation_mask = create_imputed_signals(
        pupil_orig, pupil_raw, outlier_mask
    )

    # Generate light stimuli
    red, blue, light_stimuli = generate_light_stimuli(n)

    # Generate age (40-80 years, glaucoma tends older)
    if class_label == "control":
        age = int(rng.uniform(40, 70))
    else:
        age = int(rng.uniform(50, 80))

    # Count outliers
    no_outliers = int(np.sum(outlier_mask))

    return {
        "time": time,
        "pupil_orig": pupil_orig,
        "pupil_raw": pupil_raw,
        "pupil_gt": pupil_gt,
        "Red": red,
        "Blue": blue,
        "time_orig": time.copy(),
        "subject_code": np.full(n, subject_code, dtype=object),
        "no_outliers": np.full(n, no_outliers, dtype=np.int64),
        "Age": np.full(n, age, dtype=np.int64),
        "class_label": np.full(n, class_label, dtype=object),
        "light_stimuli": light_stimuli,
        "pupil_orig_imputed": pupil_orig_imputed,
        "outlier_mask": outlier_mask,
        "pupil_raw_imputed": pupil_raw_imputed,
        "imputation_mask": imputation_mask,
        "split": np.full(n, split, dtype=object),
    }


def build_synthetic_database(
    output_path: str,
    n_subjects_per_label_per_split: int = 8,
    base_seed: int = 42,
    outlier_pct_range: tuple = (0.05, 0.35),
    metadata_path: Optional[str] = None,
) -> Path:
    """
    Build a complete synthetic PLR database.

    Parameters
    ----------
    output_path : str
        Path for output DuckDB file
    n_subjects_per_label_per_split : int
        Number of subjects per (label, split) combination
        Total = n × 2 labels × 2 splits = 4n subjects
    base_seed : int
        Base random seed for reproducibility
    outlier_pct_range : tuple
        Range of outlier percentages to sample from
    metadata_path : str, optional
        Path to save generation metadata YAML

    Returns
    -------
    output_path : Path
        Path to created database
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing database
    if output_path.exists():
        output_path.unlink()

    conn = duckdb.connect(str(output_path))

    rng = np.random.default_rng(base_seed)

    # Track all subjects for metadata
    subjects_metadata: List[Dict] = []

    # Generate subjects for each (label, split) combination
    subject_counter = {"control": 0, "glaucoma": 0}

    for split in ["train", "test"]:
        all_data: Dict[str, List] = {
            col: []
            for col in [
                "time",
                "pupil_orig",
                "pupil_raw",
                "pupil_gt",
                "Red",
                "Blue",
                "time_orig",
                "subject_code",
                "no_outliers",
                "Age",
                "class_label",
                "light_stimuli",
                "pupil_orig_imputed",
                "outlier_mask",
                "pupil_raw_imputed",
                "imputation_mask",
                "split",
            ]
        }

        for class_label in ["control", "glaucoma"]:
            for i in range(n_subjects_per_label_per_split):
                # Generate subject code
                subject_counter[class_label] += 1
                prefix = "H" if class_label == "control" else "G"
                idx = subject_counter[class_label]
                subject_code = f"SYNTH_{prefix}{idx:03d}"

                # Generate unique seed for this subject
                subject_seed = base_seed + hash(subject_code) % (2**31)

                # Sample outlier percentage
                outlier_pct = rng.uniform(*outlier_pct_range)

                # Generate subject data
                subject_data = generate_subject_data(
                    subject_code=subject_code,
                    class_label=class_label,
                    split=split,
                    seed=subject_seed,
                    outlier_pct=outlier_pct,
                )

                # Append to all_data
                for col, values in subject_data.items():
                    all_data[col].extend(values)

                # Track metadata
                subjects_metadata.append(
                    {
                        "subject_code": subject_code,
                        "class_label": class_label,
                        "split": split,
                        "seed": int(subject_seed),
                        "outlier_pct": float(outlier_pct),
                        "n_outliers": int(np.sum(subject_data["outlier_mask"])),
                    }
                )

        # Convert to numpy arrays
        data_arrays = {}
        for col, values in all_data.items():
            if col in ["subject_code", "class_label", "split"]:
                data_arrays[col] = np.array(values, dtype=str)
            elif col == "imputation_mask":
                data_arrays[col] = np.array(values, dtype=bool)
            elif col in ["outlier_mask", "no_outliers", "Age"]:
                data_arrays[col] = np.array(values, dtype=np.int64)
            else:
                data_arrays[col] = np.array(values, dtype=np.float64)

        # Create DataFrame-like structure for insertion
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {split} (
                time DOUBLE,
                pupil_orig DOUBLE,
                pupil_raw DOUBLE,
                pupil_gt DOUBLE,
                Red DOUBLE,
                Blue DOUBLE,
                time_orig DOUBLE,
                subject_code VARCHAR,
                no_outliers BIGINT,
                Age BIGINT,
                class_label VARCHAR,
                light_stimuli DOUBLE,
                pupil_orig_imputed DOUBLE,
                outlier_mask INTEGER,
                pupil_raw_imputed DOUBLE,
                imputation_mask BOOLEAN,
                split VARCHAR
            )
        """)

        # Insert data row by row (DuckDB handles this efficiently)
        n_rows = len(data_arrays["time"])
        for i in range(n_rows):
            row_values = [
                data_arrays["time"][i],
                data_arrays["pupil_orig"][i],
                data_arrays["pupil_raw"][i],
                data_arrays["pupil_gt"][i],
                data_arrays["Red"][i],
                data_arrays["Blue"][i],
                data_arrays["time_orig"][i],
                data_arrays["subject_code"][i],
                int(data_arrays["no_outliers"][i]),
                int(data_arrays["Age"][i]),
                data_arrays["class_label"][i],
                data_arrays["light_stimuli"][i],
                data_arrays["pupil_orig_imputed"][i],
                int(data_arrays["outlier_mask"][i]),
                data_arrays["pupil_raw_imputed"][i],
                bool(data_arrays["imputation_mask"][i]),
                data_arrays["split"][i],
            ]

            # Handle NaN for pupil_raw
            if np.isnan(row_values[2]):
                row_values[2] = None

            conn.execute(
                f"""
                INSERT INTO {split} VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """,
                row_values,
            )

    conn.close()

    # Save metadata
    if metadata_path:
        metadata = {
            "generation_timestamp": datetime.now().isoformat(),
            "base_seed": base_seed,
            "n_subjects_per_label_per_split": n_subjects_per_label_per_split,
            "total_subjects": len(subjects_metadata),
            "outlier_pct_range": list(outlier_pct_range),
            "database_path": str(output_path),
            "database_sha256": _compute_file_hash(output_path),
            "subjects": subjects_metadata,
        }

        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    return output_path


def _compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()
