# AGENTS.md - Universal LLM Agent Instructions

> **For all AI coding assistants**: Claude, Codex, GitHub Copilot, Cursor, Gemini, and others.

This file provides essential rules that apply to **all agents** working with this codebase. For comprehensive Claude-specific instructions, see `CLAUDE.md`.

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Run tests before any changes
uv run pytest tests/ -v

# 3. Check pre-commit hooks
pre-commit run --all-files
```

## Critical Rules (ALL AGENTS)

### 1. Registry is Single Source of Truth

**Method counts are FIXED:**
- **11 outlier methods** (NOT 15, NOT 17)
- **8 imputation methods**
- **5 classifiers**

```python
# CORRECT - always use registry
from src.data_io.registry import get_valid_outlier_methods
methods = get_valid_outlier_methods()  # Returns exactly 11

# WRONG - never parse MLflow runs
methods = run_name.split("__")[3]  # BANNED
```

**Registry location**: `configs/mlflow_registry/parameters/classification.yaml`

### 2. No Hardcoding

| WRONG | CORRECT |
|-------|---------|
| `color = "#006BA2"` | `color = COLORS["name"]` |
| `ggsave("file.png")` | `save_publication_figure(p, "name")` |
| `width = 10` | Load from config |

### 3. Fix Issues at Source

```
MLflow → DuckDB → CSV → R → Figure
  ↑
FIX HERE (extraction layer, not visualization)
```

### 4. Figure QA is Mandatory

```bash
# Run before committing ANY figure
pytest tests/test_figure_qa/ -v
```

**Zero tolerance for:**
- Synthetic data in scientific figures
- Identical model predictions (suggests fake data)
- Wrong data sources

### 5. Use Existing Implementations

| Method | Use This | NOT This |
|--------|----------|----------|
| Calibration metrics | `src/stats/calibration_extended.py` | Reimplement |
| Net Benefit | `src/stats/clinical_utility.py` | Reimplement |
| pminternal | R package via rpy2 | Python reimplementation |

## R Figure System

```r
# MANDATORY pattern for R figures
source(file.path(PROJECT_ROOT, "src/r/figure_system/config_loader.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/save_figure.R"))
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))

color_defs <- load_color_definitions()

p <- ggplot(...) +
  geom_point(color = color_defs[["--color-primary"]]) +
  theme_foundation_plr()

save_publication_figure(p, "fig_name")
```

## Python Figure System

```python
from src.viz.plot_config import setup_style, save_figure, COLORS

setup_style()

fig, ax = plt.subplots()
ax.plot(x, y, color=COLORS["ground_truth"])

save_figure(fig, "fig_name", data=data_dict)
```

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `configs/` | All YAML configuration |
| `configs/mlflow_registry/` | **Single source of truth** for parameters |
| `src/viz/` | Python visualization |
| `src/r/figures/` | R/ggplot2 figures |
| `tests/` | Test suite |

## Research Context

This repo studies **preprocessing effects on downstream classification**, NOT classifier comparison.

- **Fix classifier**: CatBoost (established as best)
- **Vary preprocessing**: outlier detection × imputation
- **Measure**: ALL STRATOS metrics (not just AUROC)

## Agent-Specific Notes

### Claude Code
- Full rules in `CLAUDE.md` and `.claude/` directory
- Uses auto-context rules from `.claude/rules/`
- Has skill commands (`/commit`, `/figures`, etc.)

### GitHub Copilot
- See `.github/copilot-instructions.md` (if exists)
- Focus on docstrings and type hints
- Respect pre-commit hooks

### Cursor
- Load `.cursor/` settings (if exists)
- Same rules as other agents apply

### Codex / ChatGPT
- Read `CLAUDE.md` for comprehensive rules
- All critical rules above apply equally

### Gemini
- Used for Nano Banana Pro figure generation
- See `docs/repo-figures/PROMPTING-INSTRUCTIONS.md`

## Before Committing

```bash
# 1. Run tests
uv run pytest tests/ -v

# 2. Check pre-commit
pre-commit run --all-files

# 3. Verify method counts
python -c "from src.data_io.registry import get_valid_outlier_methods; print(len(get_valid_outlier_methods()))"
# Must print: 11
```

## References

- **Detailed rules**: `CLAUDE.md`
- **Architecture**: `ARCHITECTURE.md`
- **Figure system**: `src/r/README.md`
- **API docs**: `docs/api-reference/`
