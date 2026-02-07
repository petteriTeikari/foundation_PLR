"""
Synthetic PLR data generation module for privacy-safe testing.

This module generates synthetic pupillary light reflex (PLR) data that matches
the schema of the real SERI_PLR_GLAUCOMA.db but contains NO patient-identifiable
information. The data is suitable for:

1. Public distribution with the repository
2. CI/CD pipeline testing
3. Demonstration of the analysis pipeline

CRITICAL: The synthetic data must pass privacy validation tests:
- Pearson correlation with any real subject < 0.60
- DTW normalized distance from any real subject > 0.15
- No shared subject codes with real data

Usage:
    from src.synthetic import generate_demo_database
    generate_demo_database(output_path="data/synthetic/SYNTH_PLR_DEMO.db")
"""

# Use lazy imports to avoid circular import issues when running as __main__


def generate_demo_database(*args, **kwargs):
    """Generate the synthetic PLR demo database. See demo_dataset module."""
    from src.synthetic.demo_dataset import generate_demo_database as _gen

    return _gen(*args, **kwargs)


def generate_plr_curve(*args, **kwargs):
    """Generate a synthetic PLR curve. See plr_generator module."""
    from src.synthetic.plr_generator import generate_plr_curve as _gen

    return _gen(*args, **kwargs)


def inject_artifacts(*args, **kwargs):
    """Inject artifacts into PLR signal. See artifact_injection module."""
    from src.synthetic.artifact_injection import inject_artifacts as _inj

    return _inj(*args, **kwargs)


def build_synthetic_database(*args, **kwargs):
    """Build synthetic DuckDB database. See database_builder module."""
    from src.synthetic.database_builder import build_synthetic_database as _build

    return _build(*args, **kwargs)


def validate_privacy(*args, **kwargs):
    """Validate privacy of synthetic data. See privacy_validator module."""
    from src.synthetic.privacy_validator import validate_privacy as _val

    return _val(*args, **kwargs)


__all__ = [
    "generate_demo_database",
    "generate_plr_curve",
    "inject_artifacts",
    "build_synthetic_database",
    "validate_privacy",
]
