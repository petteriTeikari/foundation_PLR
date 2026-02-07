#!/usr/bin/env python
"""Validate all experiment configurations."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import validate_experiment_config


def main():
    config_dir = Path("configs/experiment")
    errors = []

    for f in sorted(config_dir.glob("*.yaml")):
        try:
            validate_experiment_config(f)
            print(f"✓ {f.name}")
        except Exception as e:
            errors.append(f"{f.name}: {e}")
            print(f"✗ {f.name}: {e}")

    if errors:
        print(f"\n{len(errors)} experiment(s) failed validation")
        sys.exit(1)
    else:
        print("\nAll experiments valid!")


if __name__ == "__main__":
    main()
