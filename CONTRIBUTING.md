# Contributing to Foundation PLR

Thank you for your interest in contributing to Foundation PLR! This document provides guidelines for contributing to the project.

## Quick Start for Contributors

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand the pipeline
2. Set up your development environment (see below)
3. Pick an issue or propose a new feature
4. Submit a pull request

## Development Environment Setup

### Prerequisites

- Python 3.11+
- [uv](https://astral.sh/blog/uv) (package manager) - **conda/pip are banned**
- R 4.4+ (only for pminternal stability analysis)
- Node.js 20 LTS (for visualization)

### Setup

```bash
# Clone the repository
git clone [repository-url]
cd foundation_PLR

# Run the setup script (recommended)
./scripts/infra/setup-dev-environment.sh

# Or manually:
uv venv --python 3.11
uv sync
source .venv/bin/activate

# Install pre-commit hooks
pre-commit install
```

### Package Management Rules

| DO | DON'T |
|----|-------|
| `uv pip install package` | `pip install package` |
| `uv sync` | `conda install package` |
| R: `install.packages("pkg")` | `conda install r-pkg` |

**Why uv?** 10-100x faster, better dependency resolution, reproducible.

## Code Style

### Python

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check code
ruff check src/

# Format code
ruff format src/

# Fix auto-fixable issues
ruff check --fix src/
```

### Docstrings

Use **NumPy-style docstrings** for all public functions:

```python
def compute_auroc(y_true, y_prob):
    """
    Compute Area Under the ROC Curve.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        AUROC score between 0 and 1.

    Examples
    --------
    >>> compute_auroc([0, 1, 1], [0.1, 0.8, 0.9])
    1.0

    Notes
    -----
    AUROC is a semi-proper scoring rule per Van Calster 2024.
    """
    ...
```

### Configuration

- All configuration lives in `configs/` using Hydra
- Never hardcode values that should be configurable
- Use `configs/VISUALIZATION/plot_hyperparam_combos.yaml` for pipeline combinations

### Pre-Commit Quality Gates

Every `git commit` runs automated quality gates. If any hook fails, the commit is blocked until the violation is fixed.

[![Pre-commit hook enforcement matrix showing 5 quality gates for the Foundation PLR codebase: ruff linting, ruff formatting, R hardcoding checks (hex colors, ggsave, hardcoded dimensions), computation decoupling enforcement (banning sklearn imports in visualization code), and renv lockfile synchronization](docs/repo-figures/assets/fig-repo-58-precommit-enforcement-matrix.jpg)](docs/repo-figures/assets/fig-repo-58-precommit-enforcement-matrix.jpg)

*Figure: Pre-Commit Quality Gates — what each hook catches and how to fix common violations. See [.pre-commit-config.yaml](.pre-commit-config.yaml) for hook definitions and [enforcement plan](docs/repo-figures/figure-plans/fig-repo-58-precommit-enforcement-matrix.md) for details.*

## Adding New Methods

### Adding an Outlier Detection Method

1. Create a new file in `src/anomaly_detection/`:
   ```python
   # src/anomaly_detection/outlier_mymethod.py
   from typing import Any
   import numpy as np

   def detect_outliers_mymethod(
       signal: np.ndarray,
       **kwargs: Any
   ) -> np.ndarray:
       """
       Detect outliers using MyMethod.

       Parameters
       ----------
       signal : np.ndarray
           Input PLR signal of shape (n_timepoints,).
       **kwargs : Any
           Method-specific parameters.

       Returns
       -------
       np.ndarray
           Boolean mask where True indicates outlier.
       """
       # Implementation
       ...
   ```

2. Register in `configs/OUTLIER_MODELS/`:
   ```yaml
   # configs/OUTLIER_MODELS/mymethod.yaml
   name: MyMethod
   enabled: true
   params:
     threshold: 0.5
   ```

3. Add to the flow in `src/anomaly_detection/flow_anomaly_detection.py`

### Adding an Imputation Method

1. Create implementation in `src/imputation/`
2. Register in `configs/MODELS/`
3. Add to `src/imputation/flow_imputation.py`

### Adding a Figure

1. Add specification to `configs/VISUALIZATION/figure_registry.yaml`:
   ```yaml
   fig_my_figure:
     description: "Description of my figure"
     module: my_figure_module
     function: create_my_figure
     combos_required: true
     json_privacy: public
   ```

2. Implement in `src/viz/`:
   ```python
   # src/viz/my_figure_module.py
   from .plot_config import setup_style, save_figure, COLORS

   def create_my_figure(data, combos):
       setup_style()
       fig, ax = plt.subplots()
       # ... plotting code ...
       data_dict = {"values": [...]}  # For JSON export
       return fig, data_dict
   ```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_stats.py

# Run with coverage
pytest --cov=src tests/
```

### Test Requirements

- All new functions should have tests
- Tests should be in `tests/` mirroring the `src/` structure
- Use pytest fixtures for common setup

### Docker Tests

Some tests require Docker (for isolated R environments):

```bash
# Run Docker-dependent tests
pytest tests/ -m docker
```

## Pull Request Process

### Before Submitting

1. **Run tests**: `pytest tests/`
2. **Run linter**: `ruff check src/`
3. **Format code**: `ruff format src/`
4. **Update docstrings** for any new/modified functions
5. **Update documentation** if behavior changed

### PR Guidelines

- Keep PRs focused on a single change
- Write clear commit messages
- Reference related issues
- Include tests for new functionality

### Commit Message Format

```
<type>: <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

Example:
```
feat: Add TimesNet outlier detection method

Implements TimesNet-based anomaly detection for PLR signals.
Integrates with existing flow_anomaly_detection pipeline.

Closes #42
```

## Research Context (Important!)

### What This Project IS About

- **Preprocessing sensitivity**: How outlier detection and imputation affect classification
- **Foundation model utility**: Can MOMENT/UniTS/TimesNet replace traditional methods?
- **STRATOS compliance**: Report ALL metrics (discrimination, calibration, clinical utility)

### What This Project Is NOT About

- Comparing classifiers (CatBoost is fixed as best)
- Maximizing AUROC alone (use ALL STRATOS metrics)
- Generic ML benchmarking

### Key References

- Van Calster et al. 2024 (STRATOS metrics)
- Riley 2023 (pminternal stability)
- Najjar et al. 2023 (source data)

## Vendored Code

The following directories contain vendored third-party code and should NOT be modified:

- `src/anomaly_detection/extra_eval/TSB_AD/`
- `src/imputation/pypots/`
- `src/imputation/nuwats/`
- `src/classification/tabpfn/`
- `src/classification/tabpfn_v1/`

## Getting Help

- Check existing documentation in `docs/`
- Review [ARCHITECTURE.md](ARCHITECTURE.md) for pipeline understanding
- Read [.claude/README.md](.claude/README.md) for AI assistant context
- Open an issue for bugs or feature requests

## Code of Conduct

Be respectful and constructive. Focus on technical merit and the research question.
