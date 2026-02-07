# Scripts (`scripts/`)

Utility scripts for development, setup, and analysis.

## Available Scripts

| Script | Purpose |
|--------|---------|
| `setup-dev-environment.sh` | Install all development dependencies |
| `check-compliance.py` | Validate STRATOS compliance |
| `extract_cd_diagram_data.py` | Extract data for CD diagrams |
| `figure_and_stats_generation.py` | Generate figures and statistics |
| `pre-commit` | Pre-commit hook configuration |

## Setup Script

### `setup-dev-environment.sh`

**One command to set up everything:**

```bash
./scripts/setup-dev-environment.sh
```

**What it installs:**
- Python 3.11 via pyenv
- uv package manager
- Node.js 20 LTS
- R 4.4+ with pminternal
- Docker
- ruff, pre-commit hooks
- Project dependencies

**Supported platforms:**
- Ubuntu/Debian
- macOS (via Homebrew)
- Windows (Git Bash)

### Usage

```bash
# Full setup
./scripts/setup-dev-environment.sh

# Check what would be installed
./scripts/setup-dev-environment.sh --dry-run
```

## Analysis Scripts

### `check-compliance.py`

Validates that code follows project standards:

```bash
python scripts/check-compliance.py
```

**Checks:**
- STRATOS metrics coverage
- Configuration validity
- Docstring coverage
- Code style (ruff)

### `extract_cd_diagram_data.py`

Extracts data for Critical Difference diagrams:

```bash
python scripts/extract_cd_diagram_data.py
```

**Output:** `scripts/plr_results.duckdb`

### `figure_and_stats_generation.py`

Comprehensive figure and statistics generation:

```bash
python scripts/figure_and_stats_generation.py
```

## Pre-commit Hook

### `pre-commit`

Git pre-commit hook for code quality:

```bash
# Install (automatic with setup script)
cp scripts/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Or use pre-commit framework
pre-commit install
```

**What it runs:**
- ruff check
- ruff format --check
- Basic tests

## Adding New Scripts

1. Create script in `scripts/`
2. Add shebang line (e.g., `#!/usr/bin/env python3`)
3. Make executable: `chmod +x scripts/my_script.py`
4. Add description to this README
5. Add docstring explaining usage

### Script Template

```python
#!/usr/bin/env python3
"""
Script description.

Usage:
    python scripts/my_script.py [options]

Arguments:
    --option    Description of option
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--option", help="Description")
    args = parser.parse_args()

    # Implementation
    ...


if __name__ == "__main__":
    main()
```
