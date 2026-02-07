"""
Experiment Configuration Models for Foundation PLR.

Pydantic models for validating and loading experiment configurations.
These ensure type safety and catch configuration errors early.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, field_validator

from loguru import logger


# =============================================================================
# Sub-Configuration Models
# =============================================================================


class DataConfig(BaseModel):
    """Configuration for data sources."""

    name: str = Field(description="Human-readable dataset name")
    version: str = Field(description="Dataset version")
    source: str = Field(description="Data source reference")

    class Config:
        extra = "allow"  # Allow additional fields


class CombosConfig(BaseModel):
    """Configuration for hyperparameter combinations."""

    standard: Dict[str, Any] = Field(
        default_factory=dict, description="Standard combos for main figures"
    )
    extended: Dict[str, Any] = Field(
        default_factory=dict, description="Extended combos for supplementary figures"
    )
    visualization: Dict[str, Any] = Field(
        default_factory=dict, description="Visualization settings"
    )

    class Config:
        extra = "allow"


class SubjectsConfig(BaseModel):
    """Configuration for demo subjects."""

    control: Dict[str, List[str]] = Field(
        default_factory=dict, description="Control subject codes by outlier level"
    )
    glaucoma: Dict[str, List[str]] = Field(
        default_factory=dict, description="Glaucoma subject codes by outlier level"
    )
    selection_criteria: Dict[str, Any] = Field(
        default_factory=dict, description="Criteria used for subject selection"
    )

    class Config:
        extra = "allow"


class FiguresConfig(BaseModel):
    """Configuration for figure generation."""

    quality: str = Field(default="draft", description="Quality level")
    formats: Dict[str, Any] = Field(default_factory=dict)
    dimensions: Dict[str, Any] = Field(default_factory=dict)
    typography: Dict[str, Any] = Field(default_factory=dict)
    colors: Dict[str, str] = Field(default_factory=dict)
    export: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class MLflowConfig(BaseModel):
    """Configuration for MLflow tracking."""

    environment: str = Field(default="production")
    tracking: Dict[str, Any] = Field(default_factory=dict)
    experiments: Dict[str, Any] = Field(default_factory=dict)
    logging: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    extraction: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class ExperimentMetadata(BaseModel):
    """Metadata about the experiment."""

    publication_status: Optional[str] = None
    manuscript_repo: Optional[str] = None
    doi: Optional[str] = None
    publication_date: Optional[str] = None
    authors: Optional[List[str]] = None
    purpose: Optional[str] = None
    data_source: Optional[str] = None

    class Config:
        extra = "allow"


class ReproducibilityConfig(BaseModel):
    """Configuration for reproducibility guarantees."""

    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    python_version: Optional[str] = None
    uv_lock_sha256: Optional[str] = None
    data_checksums: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class FactorialDesign(BaseModel):
    """Configuration for factorial experiment design."""

    outlier_methods: int = Field(default=11, description="Number of outlier methods")
    imputation_methods: int = Field(
        default=8, description="Number of imputation methods"
    )
    classifiers: int = Field(default=5, description="Number of classifiers")
    featurization_methods: int = Field(
        default=2, description="Number of featurization methods"
    )

    @field_validator("outlier_methods")
    @classmethod
    def validate_outlier_count(cls, v: int) -> int:
        """Validate outlier method count matches registry."""
        if v != 11:
            logger.warning(
                f"Outlier method count ({v}) differs from registry standard (11). "
                "Ensure this is intentional."
            )
        return v


class ExpectedResults(BaseModel):
    """Expected results for validation."""

    ground_truth_auroc: Optional[float] = None
    best_ensemble_auroc: Optional[float] = None
    n_bootstrap_iterations: Optional[int] = None
    expected_auroc_range: Optional[List[float]] = None
    note: Optional[str] = None

    class Config:
        extra = "allow"


# =============================================================================
# Main Experiment Configuration
# =============================================================================


class ExperimentConfig(BaseModel):
    """
    Complete experiment configuration.

    This is the top-level model that validates an entire experiment setup.
    """

    # Core experiment info
    experiment: Dict[str, Any] = Field(
        description="Experiment name, version, and settings"
    )

    # Optional composed configs (loaded from defaults)
    data: Optional[DataConfig] = None
    combos: Optional[CombosConfig] = None
    subjects: Optional[SubjectsConfig] = None
    figures: Optional[FiguresConfig] = None
    mlflow: Optional[MLflowConfig] = None

    # Metadata
    metadata: Optional[ExperimentMetadata] = None
    reproducibility: Optional[ReproducibilityConfig] = None
    factorial_design: Optional[FactorialDesign] = None
    expected_results: Optional[ExpectedResults] = None
    pipeline: Optional[Dict[str, Any]] = None

    # Hydra defaults (not validated)
    defaults: Optional[List[Any]] = None

    class Config:
        extra = "allow"  # Allow Hydra-specific fields

    @property
    def name(self) -> str:
        """Get experiment name."""
        return self.experiment.get("name", "unnamed")

    @property
    def version(self) -> str:
        """Get experiment version."""
        return self.experiment.get("version", "0.0.0")

    @property
    def is_frozen(self) -> bool:
        """Check if experiment is frozen (should not be modified)."""
        return self.experiment.get("frozen", False)


# =============================================================================
# Loading Functions
# =============================================================================


def get_config_dir() -> Path:
    """Get the configs directory path."""
    return Path(__file__).parent.parent.parent / "configs"


def load_experiment(name: str) -> ExperimentConfig:
    """
    Load and validate an experiment configuration.

    Args:
        name: Experiment name (e.g., "paper_2026", "synthetic")
              Can be a bare name or include .yaml extension

    Returns:
        Validated ExperimentConfig object

    Raises:
        FileNotFoundError: If experiment config doesn't exist
        ValidationError: If config fails validation
    """
    config_dir = get_config_dir()

    # Remove .yaml extension if present
    if name.endswith(".yaml"):
        name = name[:-5]

    # Find experiment file
    exp_path = config_dir / "experiment" / f"{name}.yaml"
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {exp_path}")

    logger.info(f"Loading experiment config: {name}")

    # Load with OmegaConf (handles Hydra features)
    cfg = OmegaConf.load(exp_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Validate with Pydantic
    experiment = ExperimentConfig(**cfg_dict)

    logger.info(f"Loaded experiment '{experiment.name}' v{experiment.version}")

    return experiment


def validate_experiment_config(config_path: Union[str, Path]) -> bool:
    """
    Validate an experiment configuration file.

    Args:
        config_path: Path to the experiment config YAML

    Returns:
        True if valid, raises exception otherwise

    Raises:
        FileNotFoundError: If file doesn't exist
        ValidationError: If validation fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load and validate
    cfg = OmegaConf.load(config_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # This will raise ValidationError if invalid
    ExperimentConfig(**cfg_dict)

    logger.info(f"Config validated: {config_path}")
    return True


def list_experiments() -> List[str]:
    """
    List available experiment configurations.

    Returns:
        List of experiment names (without .yaml extension)
    """
    config_dir = get_config_dir()
    exp_dir = config_dir / "experiment"

    if not exp_dir.exists():
        return []

    experiments = []
    for f in exp_dir.glob("*.yaml"):
        if f.name != "README.md":
            experiments.append(f.stem)

    return sorted(experiments)
