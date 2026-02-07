# GitHub Copilot Instructions for Foundation PLR

## Repository Overview

This repository evaluates foundation models for preprocessing biosignals (Pupillary Light Reflex) for glaucoma screening.

## Research Focus

**Question**: How do preprocessing choices (outlier detection → imputation) affect downstream prediction quality using handcrafted physiological features?

- **NOT** about comparing classifiers - fix CatBoost, vary preprocessing
- **NOT** AUROC-only - use ALL STRATOS-compliant metrics (discrimination + calibration + clinical utility)

## Coding Standards

### Python
- Use `uv` for package management (pip/conda BANNED)
- Python 3.11+ with modern type hints (`list[str]` not `List[str]`)
- All values from `configs/` YAML files - no hardcoding

### Registry is Single Source of Truth
- 11 outlier methods (exactly)
- 8 imputation methods
- 5 classifiers
- See `configs/mlflow_registry/parameters/classification.yaml`

### Computation Decoupling
- `scripts/extract_*.py` - compute and store in DuckDB
- `src/viz/` - ONLY read from DuckDB, never compute

### Figure Standards
- Use `save_publication_figure()` not `ggsave()` or `plt.savefig()`
- Load colors from `configs/VISUALIZATION/colors.yaml`
- Include JSON data sidecars for reproducibility

## Key Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Comprehensive project context |
| `AGENTS.md` | Multi-LLM support guide |
| `configs/defaults.yaml` | Central configuration |
| `src/data_io/registry.py` | Method name validation |
| `src/viz/plot_config.py` | Visualization standards |

## Common Patterns

```python
# Load from registry - CORRECT
from src.data_io.registry import get_valid_outlier_methods
valid_methods = get_valid_outlier_methods()

# Load colors - CORRECT
from src.viz.plot_config import COLORS, get_combo_color
color = get_combo_color("ground_truth")

# Never do this - BANNED
methods = run_name.split("__")[3]  # Parsing run names
color = "#0072B2"  # Hardcoded hex
```
