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

try:
    from .analysis_flow import analysis_flow
    from .extraction_flow import extraction_flow

    __all__ = ["extraction_flow", "analysis_flow"]
except ImportError:
    # Allow import even if dependencies are missing
    __all__ = []
