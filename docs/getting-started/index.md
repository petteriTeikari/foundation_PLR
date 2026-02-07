# Getting Started

Welcome to Foundation PLR! This section will help you get up and running quickly.

## Prerequisites

- Python 3.11 or higher
- UV package manager (conda is not supported)
- Git

## Quick Installation

```bash
# Clone the repository
git clone https://github.com/petteriTeikari/foundation_PLR.git
cd foundation-PLR/foundation_PLR

# Install dependencies with UV
uv sync
```

## Next Steps

1. **[Installation](installation.md)** - Detailed setup instructions
2. **[Quick Start](quickstart.md)** - Run your first experiment
3. **[Configuration](configuration.md)** - Understand Hydra configs
4. **[Concepts for Researchers](../concepts-for-researchers.md)** - ELI5 explanations of tools

## What's in the Box

| Component | Count | Description |
|-----------|-------|-------------|
| Outlier methods | 11 | LOF, MOMENT, UniTS, TimesNet, etc. |
| Imputation methods | 8 | SAITS, CSDI, MOMENT, linear, etc. |
| Classifiers | 5 | CatBoost (recommended), XGBoost, etc. |
