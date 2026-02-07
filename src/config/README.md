# Config

Typed configuration loading and validation.

## Overview

Provides Pydantic models for experiment configuration validation and a YAML loader with caching and immutable access. Ensures type safety and catches configuration errors early, complementing the Hydra-based configuration in `configs/`.

## Modules

| Module | Purpose |
|--------|---------|
| `experiment.py` | Pydantic models for experiment configuration validation |
| `loader.py` | YAML configuration loader with caching and immutable access |

## See Also

- `configs/` -- Hydra YAML configuration files loaded by this module
