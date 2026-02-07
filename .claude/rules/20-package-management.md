# RULE: Package Management - uv Only

## Python: uv (MANDATORY)

```bash
# Preferred (adds to pyproject.toml):
uv add package_name
uv add --dev package_name
uv sync

# Acceptable (direct install):
uv pip install package_name
```

## BANNED - No Exceptions

| BANNED | Use Instead |
|--------|-------------|
| `pip install X` | `uv pip install X` or `uv add X` |
| `conda install X` | `uv add X` |
| `mamba install X` | `uv add X` |
| `conda activate` | `source .venv/bin/activate` |
| `conda install r-base` | System R from CRAN (sudo apt/brew) |
| `conda install r-X` | R's `install.packages("X")` |

## R Packages

Install via R's `install.packages()` from CRAN. Requires system R >= 4.4.

## TypeScript/JS

Use `npm install package_name`.

## Why uv add over uv pip install

`uv add` writes to pyproject.toml for reproducibility. Other developers get the dependency automatically. Version locked in lockfile.
