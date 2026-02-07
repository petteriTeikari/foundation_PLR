"""
Prefect flows for Foundation PLR pipeline orchestration.

Two main flows:
1. extraction_flow: MLflow → DuckDB extraction with privacy separation
2. analysis_flow: Statistics, visualization, and LaTeX generation

Architecture:
    Block 1 (Extraction): Requires mlruns access, produces:
        - PUBLIC: data/public/foundation_plr_results.db
        - PRIVATE: data/private/subject_lookup.yaml
        - PRIVATE: data/private/demo_subjects_traces.pkl

    Block 2 (Analysis): Works from public DB, produces:
        - figures/generated/*.png
        - figures/generated/data/*.json
        - tables/generated/*.tex

Usage:
    # Full pipeline
    make reproduce

    # From checkpoint (public DB exists)
    make reproduce-from-checkpoint
"""

__all__ = ["extraction_flow", "analysis_flow"]


def __getattr__(name: str):
    """Lazy imports — flows are Prefect entry points, not library APIs.

    This avoids eager imports of heavy dependencies (prefect, mlflow, catboost)
    when the package is scanned by static analysis tools like griffe/mkdocstrings.
    """
    if name == "extraction_flow":
        from .extraction_flow import extraction_flow

        return extraction_flow
    if name == "analysis_flow":
        from .analysis_flow import analysis_flow

        return analysis_flow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
