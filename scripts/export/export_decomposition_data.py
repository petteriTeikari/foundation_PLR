#!/usr/bin/env python3
"""Export decomposition grid data to JSON for R/ggplot2 visualization.

This script computes PLR waveform decomposition for all 25 combinations
(5 methods × 5 preprocessing categories) and exports to JSON format
for the R figure script.

Output: data/r_data/decomposition_grid_data.json
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.decomposition.aggregation import (  # noqa: E402
    DecompositionAggregator,
    get_preprocessing_categories,
)
from src.utils.paths import get_preprocessed_signals_db_path  # noqa: E402


def export_decomposition_data(
    output_path: Path | None = None,
    n_bootstrap: int = 100,
    limit: int | None = None,
) -> Path:
    """Export decomposition grid data to JSON.

    Parameters
    ----------
    output_path : Path, optional
        Output JSON path. Default: data/r_data/decomposition_grid_data.json
    n_bootstrap : int
        Number of bootstrap iterations for CIs
    limit : int, optional
        Limit subjects per category (for testing)

    Returns
    -------
    Path
        Path to exported JSON file
    """
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "r_data" / "decomposition_grid_data.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get database path
    db_path = get_preprocessed_signals_db_path()
    if not db_path.exists():
        raise FileNotFoundError(
            f"Preprocessed signals DB not found: {db_path}\n"
            "Run Phase 2 extraction first."
        )

    print(f"Loading from: {db_path}")
    print(f"Bootstrap iterations: {n_bootstrap}")

    # Create aggregator
    aggregator = DecompositionAggregator(
        db_path=db_path,
        n_bootstrap=n_bootstrap,
        random_seed=42,
    )

    # Define methods and categories
    methods = ["template", "pca", "rotated_pca", "sparse_pca", "ged"]
    categories = get_preprocessing_categories()

    print(f"Methods: {methods}")
    print(f"Categories: {categories}")

    # Compute all decompositions
    print("\nComputing decompositions...")
    results = aggregator.compute_all_decompositions(
        categories=categories,
        methods=methods,
        n_components=3,
        limit=limit,
    )

    # Convert to JSON-serializable format
    data = {
        "metadata": {
            "generator": "export_decomposition_data.py",
            "n_bootstrap": n_bootstrap,
            "methods": methods,
            "categories": categories,
            "db_path": str(db_path),
        },
        "data": {},
    }

    for (category, method), result in results.items():
        key = f"{category}__{method}"
        data["data"][key] = {
            "category": category,
            "method": method,
            "n_subjects": result.n_subjects,
            "time_vector": result.time_vector.tolist(),
            "mean_waveform": result.mean_waveform.tolist(),
            "mean_waveform_ci_lower": result.mean_waveform_ci_lower.tolist(),
            "mean_waveform_ci_upper": result.mean_waveform_ci_upper.tolist(),
            "components": [
                {
                    "name": c.name,
                    "mean": c.mean.tolist(),
                    "ci_lower": c.ci_lower.tolist(),
                    "ci_upper": c.ci_upper.tolist(),
                }
                for c in result.components
            ],
        }

    # Write JSON
    print(f"\nWriting to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    # Report statistics
    print("\nExport complete:")
    print(f"  Total combinations: {len(data['data'])}")
    for key, item in data["data"].items():
        print(f"  {key}: n={item['n_subjects']}")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export decomposition data to JSON")
    parser.add_argument(
        "--bootstrap", type=int, default=100, help="Bootstrap iterations"
    )
    parser.add_argument("--limit", type=int, help="Limit subjects (for testing)")
    parser.add_argument("--output", type=Path, help="Output JSON path")

    args = parser.parse_args()

    output = export_decomposition_data(
        output_path=args.output,
        n_bootstrap=args.bootstrap,
        limit=args.limit,
    )
    print(f"\nOutput: {output}")
