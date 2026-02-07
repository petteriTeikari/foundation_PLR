#!/usr/bin/env python3
"""
Extract Top Models and Curves from MLflow for Foundation PLR Analysis.

This script extracts the top-performing models and key comparison curves from
MLflow experiment runs, saving them as a consolidated .pkl file for easier
subsequent analysis (SHAP, feature importance, VIF, etc.).

Usage
-----
    python extract_top_models_from_mlflow.py --experiment-id 253031330985650090 \
        --output models_subset.pkl --top-n 10

    # Or with defaults:
    python extract_top_models_from_mlflow.py

Author: Foundation PLR Team
Date: 2026-01-24
"""

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data_io.registry import validate_imputation_method, validate_outlier_method


@dataclass
class ModelInfo:
    """
    Container for extracted model information.

    Attributes
    ----------
    run_id : str
        MLflow run ID.
    model_name : str
        Full model filename stem.
    featurization : str
        Featurization method used (e.g., 'simple1.0', 'MOMENT-embedding').
    imputation : str
        Imputation method used (e.g., 'SAITS', 'CSDI').
    outlier_detection : str
        Outlier detection method used (e.g., 'LOF', 'UniTS-zeroshot').
    auroc : float
        Area under ROC curve on test set.
    model_path : Path
        Path to the pickled model file.
    metrics_path : Optional[Path]
        Path to metrics pickle file, if available.
    features_path : Optional[Path]
        Path to features/data arrays pickle file, if available.
    """

    run_id: str
    model_name: str
    featurization: str
    imputation: str
    outlier_detection: str
    auroc: float
    model_path: Path
    metrics_path: Optional[Path]
    features_path: Optional[Path]


def find_mlflow_tracking_dir() -> Path:
    """
    Find the MLflow tracking directory.

    Searches common locations for the mlruns directory.

    Returns
    -------
    Path
        Path to the mlruns directory.

    Raises
    ------
    FileNotFoundError
        If no mlruns directory is found.
    """
    # Use centralized path utility first, then fallback candidates
    from src.utils.paths import get_mlruns_dir

    try:
        mlruns = get_mlruns_dir()
        if mlruns.exists():
            return mlruns
    except Exception:
        pass

    candidates = [
        Path.home() / "mlruns",
        Path.cwd() / "mlruns",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Could not find mlruns directory")


def parse_model_filename(filename: str) -> Dict[str, str]:
    """
    Parse model filename to extract configuration details.

    Parameters
    ----------
    filename : str
        Model pickle filename.
        Example: model_CATBOOST_eval-auc__simple1.0__SAITS__pupil-gt.pickle

    Returns
    -------
    Dict[str, str]
        Dictionary with keys:
        - 'classifier': Classifier type (e.g., 'CATBOOST')
        - 'eval_metric': Evaluation metric (e.g., 'eval-auc')
        - 'featurization': Featurization method (e.g., 'simple1.0')
        - 'imputation': Imputation method (e.g., 'SAITS')
        - 'outlier': Outlier detection method (e.g., 'pupil-gt')
    """
    # Remove prefix and suffix
    name = filename.replace("model_", "").replace(".pickle", "")
    parts = name.split("__")

    result = {
        "classifier": "",
        "eval_metric": "",
        "featurization": "",
        "imputation": "",
        "outlier": "",
    }

    if len(parts) >= 1:
        # First part: CATBOOST_eval-auc
        classifier_parts = parts[0].split("_")
        result["classifier"] = classifier_parts[0]
        if len(classifier_parts) > 1:
            result["eval_metric"] = "_".join(classifier_parts[1:])

    if len(parts) >= 2:
        result["featurization"] = parts[1]

    if len(parts) >= 3:
        result["imputation"] = parts[2]

    if len(parts) >= 4:
        result["outlier"] = parts[3]

    return result


def get_run_metrics(run_dir: Path) -> Dict[str, float]:
    """
    Extract metrics from a run directory.

    MLflow stores metrics in subdirectories: metrics/test/, metrics/train/.
    This function reads the last value from each metric file.

    Parameters
    ----------
    run_dir : Path
        Path to the MLflow run directory.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping metric names to their values.
        Keys are prefixed with 'test_' or 'train_' accordingly.
    """
    metrics = {}
    metrics_base = run_dir / "metrics"

    if not metrics_base.exists():
        return metrics

    # Check for test metrics specifically (what we need for AUROC)
    test_metrics_dir = metrics_base / "test"
    if test_metrics_dir.exists():
        for metric_file in test_metrics_dir.iterdir():
            if metric_file.is_file():
                try:
                    with open(metric_file, "r") as f:
                        # MLflow metrics format: timestamp value step
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            parts = last_line.split()
                            if len(parts) >= 2:
                                metrics[f"test_{metric_file.name}"] = float(parts[1])
                except (ValueError, IndexError):
                    pass

    # Also check train metrics
    train_metrics_dir = metrics_base / "train"
    if train_metrics_dir.exists():
        for metric_file in train_metrics_dir.iterdir():
            if metric_file.is_file():
                try:
                    with open(metric_file, "r") as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            parts = last_line.split()
                            if len(parts) >= 2:
                                metrics[f"train_{metric_file.name}"] = float(parts[1])
                except (ValueError, IndexError):
                    pass

    return metrics


def scan_experiment_runs(
    mlruns_dir: Path, experiment_id: str, classifier_filter: Optional[str] = None
) -> List[ModelInfo]:
    """
    Scan all runs in an experiment and extract model information.

    Parameters
    ----------
    mlruns_dir : Path
        Path to mlruns directory.
    experiment_id : str
        MLflow experiment ID.
    classifier_filter : Optional[str], optional
        Filter for classifier type (e.g., 'CATBOOST'), by default None.

    Returns
    -------
    List[ModelInfo]
        List of ModelInfo objects for all discovered models.

    Raises
    ------
    FileNotFoundError
        If the experiment directory does not exist.
    """
    experiment_dir = mlruns_dir / experiment_id

    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    models = []

    for run_dir in experiment_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue

        artifacts_dir = run_dir / "artifacts"
        model_dir = artifacts_dir / "model"

        if not model_dir.exists():
            continue

        # Find model pickle files
        for model_file in model_dir.glob("*.pickle"):
            config = parse_model_filename(model_file.name)

            if classifier_filter and config["classifier"] != classifier_filter:
                continue

            # Validate outlier and imputation methods against registry
            # Skip models with invalid methods (e.g., orphan runs, test experiments)
            if config["outlier"] and not validate_outlier_method(config["outlier"]):
                continue
            if config["imputation"] and not validate_imputation_method(
                config["imputation"]
            ):
                continue

            # Get metrics
            metrics = get_run_metrics(run_dir)
            auroc = metrics.get("test_AUROC", metrics.get("AUROC", 0.0))

            # Find associated files
            features_path = None
            features_dir = artifacts_dir / "dict_arrays"
            if features_dir.exists():
                for fp in features_dir.glob("*.pickle"):
                    features_path = fp
                    break

            metrics_pickle = None
            metrics_pkl_dir = artifacts_dir / "metrics"
            if metrics_pkl_dir.exists():
                for mp in metrics_pkl_dir.glob("*.pickle"):
                    metrics_pickle = mp
                    break

            models.append(
                ModelInfo(
                    run_id=run_dir.name,
                    model_name=model_file.stem,
                    featurization=config["featurization"],
                    imputation=config["imputation"],
                    outlier_detection=config["outlier"],
                    auroc=auroc,
                    model_path=model_file,
                    metrics_path=metrics_pickle,
                    features_path=features_path,
                )
            )

    return models


def get_top_models(
    models: List[ModelInfo], top_n: int = 10, diverse: bool = True
) -> List[ModelInfo]:
    """
    Get top N models by AUROC.

    Parameters
    ----------
    models : List[ModelInfo]
        List of ModelInfo objects to select from.
    top_n : int, optional
        Number of top models to return, by default 10.
    diverse : bool, optional
        If True, ensure diversity in configurations, by default True.

    Returns
    -------
    List[ModelInfo]
        List of top N ModelInfo objects sorted by AUROC.
    """
    # Sort by AUROC descending
    sorted_models = sorted(models, key=lambda x: x.auroc, reverse=True)

    if not diverse:
        return sorted_models[:top_n]

    # Ensure diversity: at least one from each classifier type if available
    selected = []
    seen_classifiers = set()
    seen_configs = set()

    for model in sorted_models:
        config_key = (model.featurization, model.imputation, model.outlier_detection)
        classifier = (
            model.model_name.split("_")[1] if "_" in model.model_name else "unknown"
        )

        if len(selected) < top_n:
            # First pass: prioritize diversity
            if classifier not in seen_classifiers or config_key not in seen_configs:
                selected.append(model)
                seen_classifiers.add(classifier)
                seen_configs.add(config_key)

    # Fill remaining slots with highest AUROC
    for model in sorted_models:
        if len(selected) >= top_n:
            break
        if model not in selected:
            selected.append(model)

    return selected[:top_n]


def _build_comparison_curves() -> Dict[str, List[str]]:
    """
    Build comparison curves from registry (single source of truth).

    Returns curated selections for figures, using registry categories.
    """
    from src.data_io.registry import (
        get_imputation_categories,
        get_outlier_categories,
        get_valid_classifiers,
    )

    outlier_cats = get_outlier_categories()
    imputation_cats = get_imputation_categories()
    classifiers = get_valid_classifiers()

    # Build selections from registry categories
    # Note: We select representative methods from each category for figure readability
    return {
        "outlier_detection": (
            outlier_cats.get("foundation_model", [])[:1]  # Best FM
            + outlier_cats.get("traditional", [])[:2]  # Traditional baselines
            + outlier_cats.get("ground_truth", [])  # Ground truth reference
        ),
        "imputation": (
            imputation_cats.get("foundation_model", [])[:1]  # FM
            + imputation_cats.get("deep_learning", [])[:2]  # DL baselines
            + imputation_cats.get("ground_truth", [])  # Ground truth reference
        ),
        "featurization": [
            "simple1.0",  # Handcrafted features (canonical name)
            "MOMENT-embedding",  # FM embeddings
            "MOMENT-embedding-PCA",  # FM embeddings with PCA
        ],
        "classifier": list(classifiers)[:4],  # Top 4 from registry
    }


# Build curves at module load time (uses registry)
HARDCODED_CURVES = _build_comparison_curves()


def extract_hardcoded_models(
    models: List[ModelInfo], curves: Dict[str, List[str]] = HARDCODED_CURVES
) -> Dict[str, List[ModelInfo]]:
    """
    Extract models matching the hard-coded comparison curves.

    Parameters
    ----------
    models : List[ModelInfo]
        List of all available models.
    curves : Dict[str, List[str]], optional
        Dictionary mapping curve categories to method names, by default HARDCODED_CURVES.

    Returns
    -------
    Dict[str, List[ModelInfo]]
        Dictionary mapping curve category to list of matching models.
    """
    extracted = {category: [] for category in curves}

    for model in models:
        # Check outlier detection
        for od_name in curves.get("outlier_detection", []):
            if od_name.lower() in model.outlier_detection.lower():
                extracted["outlier_detection"].append(model)
                break

        # Check imputation
        for imp_name in curves.get("imputation", []):
            if imp_name.lower() in model.imputation.lower():
                extracted["imputation"].append(model)
                break

        # Check featurization
        for feat_name in curves.get("featurization", []):
            if feat_name.lower() in model.featurization.lower():
                extracted["featurization"].append(model)
                break

        # Check classifier
        for clf_name in curves.get("classifier", []):
            if clf_name in model.model_name:
                extracted["classifier"].append(model)
                break

    return extracted


def load_model_artifacts(model_info: ModelInfo) -> Dict[str, Any]:
    """
    Load the actual model and associated artifacts.

    Parameters
    ----------
    model_info : ModelInfo
        Model information including file paths.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'model_info': Basic model metadata
        - 'model': Loaded model object (if successful)
        - 'data_arrays': Loaded feature arrays (if available)
        - 'metrics': Loaded metrics (if available)
        - '*_error': Error messages for failed loads
    """
    artifacts = {
        "model_info": {
            "run_id": model_info.run_id,
            "model_name": model_info.model_name,
            "featurization": model_info.featurization,
            "imputation": model_info.imputation,
            "outlier_detection": model_info.outlier_detection,
            "auroc": model_info.auroc,
        }
    }

    # Load model
    if model_info.model_path.exists():
        try:
            with open(model_info.model_path, "rb") as f:
                artifacts["model"] = pickle.load(f)
        except Exception as e:
            artifacts["model_error"] = str(e)

    # Load features/data arrays
    if model_info.features_path and model_info.features_path.exists():
        try:
            with open(model_info.features_path, "rb") as f:
                artifacts["data_arrays"] = pickle.load(f)
        except Exception as e:
            artifacts["data_arrays_error"] = str(e)

    # Load metrics
    if model_info.metrics_path and model_info.metrics_path.exists():
        try:
            with open(model_info.metrics_path, "rb") as f:
                artifacts["metrics"] = pickle.load(f)
        except Exception as e:
            artifacts["metrics_error"] = str(e)

    return artifacts


def save_model_subset(
    output_path: Path,
    top_models: List[Dict[str, Any]],
    hardcoded_models: Dict[str, List[Dict[str, Any]]],
    metadata: Dict[str, Any],
) -> None:
    """
    Save the extracted models and metadata to a single pickle file.

    Parameters
    ----------
    output_path : Path
        Path for the output pickle file.
    top_models : List[Dict[str, Any]]
        List of top model artifacts/metadata.
    hardcoded_models : Dict[str, List[Dict[str, Any]]]
        Dictionary of hardcoded comparison model artifacts.
    metadata : Dict[str, Any]
        Extraction metadata (date, experiment ID, etc.).
    """
    output_data = {
        "metadata": metadata,
        "top_models": top_models,
        "hardcoded_comparisons": hardcoded_models,
    }

    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)

    print(f"Saved model subset to: {output_path}")
    print(f"  - Top models: {len(top_models)}")
    for category, models in hardcoded_models.items():
        print(f"  - {category}: {len(models)} models")


def main():
    """Main entry point for the extraction script."""
    parser = argparse.ArgumentParser(
        description="Extract top models and comparison curves from MLflow"
    )
    parser.add_argument(
        "--mlruns-dir",
        type=Path,
        default=None,
        help="Path to mlruns directory (auto-detected if not specified)",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="253031330985650090",
        help="MLflow experiment ID",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models_subset.pkl"),
        help="Output pickle file path",
    )
    parser.add_argument(
        "--top-n", type=int, default=10, help="Number of top models to extract"
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default=None,
        choices=["CATBOOST", "XGBOOST", "TabPFN", "TabM", "ensemble"],
        help="Filter by classifier type",
    )
    parser.add_argument(
        "--load-artifacts",
        action="store_true",
        help="Load full model artifacts (large memory requirement)",
    )
    parser.add_argument(
        "--list-only", action="store_true", help="List models without saving"
    )

    args = parser.parse_args()

    # Find mlruns directory
    if args.mlruns_dir:
        mlruns_dir = args.mlruns_dir
    else:
        mlruns_dir = find_mlflow_tracking_dir()

    print(f"Using MLflow tracking directory: {mlruns_dir}")
    print(f"Experiment ID: {args.experiment_id}")

    # Scan runs
    print("Scanning experiment runs...")
    all_models = scan_experiment_runs(
        mlruns_dir, args.experiment_id, classifier_filter=args.classifier
    )
    print(f"Found {len(all_models)} models")

    # Get top models
    top_models = get_top_models(all_models, top_n=args.top_n, diverse=True)
    print(f"\nTop {args.top_n} models by AUROC:")
    for i, model in enumerate(top_models, 1):
        print(f"  {i}. AUROC={model.auroc:.4f} | {model.model_name[:60]}...")

    # Get hardcoded comparison models
    hardcoded = extract_hardcoded_models(all_models)
    print("\nHardcoded comparison curves:")
    for category, models in hardcoded.items():
        print(f"  {category}: {len(models)} models")

    if args.list_only:
        return

    # Load artifacts if requested
    if args.load_artifacts:
        print("\nLoading model artifacts (this may take a while)...")
        top_artifacts = [load_model_artifacts(m) for m in top_models]
        hardcoded_artifacts = {
            cat: [load_model_artifacts(m) for m in models]
            for cat, models in hardcoded.items()
        }
    else:
        # Just save paths and metadata
        top_artifacts = [
            {
                "model_info": {
                    "run_id": m.run_id,
                    "model_name": m.model_name,
                    "featurization": m.featurization,
                    "imputation": m.imputation,
                    "outlier_detection": m.outlier_detection,
                    "auroc": m.auroc,
                },
                "model_path": str(m.model_path),
                "metrics_path": str(m.metrics_path) if m.metrics_path else None,
                "features_path": str(m.features_path) if m.features_path else None,
            }
            for m in top_models
        ]
        hardcoded_artifacts = {
            cat: [
                {
                    "model_info": {
                        "run_id": m.run_id,
                        "model_name": m.model_name,
                        "auroc": m.auroc,
                    },
                    "model_path": str(m.model_path),
                }
                for m in models
            ]
            for cat, models in hardcoded.items()
        }

    # Prepare metadata
    metadata = {
        "extraction_date": "2026-01-24",
        "mlruns_dir": str(mlruns_dir),
        "experiment_id": args.experiment_id,
        "top_n": args.top_n,
        "classifier_filter": args.classifier,
        "total_models_scanned": len(all_models),
        "artifacts_loaded": args.load_artifacts,
        "hardcoded_curves": HARDCODED_CURVES,
    }

    # Save
    save_model_subset(args.output, top_artifacts, hardcoded_artifacts, metadata)

    # Print usage instructions
    print("\n" + "=" * 60)
    print("USAGE INSTRUCTIONS")
    print("=" * 60)
    print("""
To load and use the extracted models:

```python
import pickle

# Load the subset
with open('models_subset.pkl', 'rb') as f:
    data = pickle.load(f)

# Access top models
for model in data['top_models']:
    print(model['model_info']['model_name'])
    print(f"  AUROC: {model['model_info']['auroc']}")

    # Load actual model if needed
    with open(model['model_path'], 'rb') as mf:
        clf = pickle.load(mf)

    # SHAP analysis
    import shap
    explainer = shap.TreeExplainer(clf)
    # ... continue with analysis

# Access comparison curves
for model in data['hardcoded_comparisons']['outlier_detection']:
    print(model['model_info']['model_name'])
```
""")


if __name__ == "__main__":
    main()
