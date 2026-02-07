# /validate - Compliance Validation

Run compliance checks for Foundation PLR codebase.

## Usage

- `/validate` - Run all compliance checks
- `/validate figures` - Validate figure outputs
- `/validate combos` - Check for hardcoded combo names
- `/validate privacy` - Check for PRIVATE JSON files staged for commit

## Checks Performed

1. **No hardcoded combos** - Combo names must come from `configs/VISUALIZATION/plot_hyperparam_combos.yaml`
2. **Ground truth present** - All comparison figures must include ground_truth combo
3. **Max curves** - Main figures: 4, Supplementary: 6
4. **No hardcoded colors** - Must use `plot_config.COLORS`
5. **setup_style() called** - All plotting scripts must call this
6. **PRIVATE JSON not staged** - Subject-level data must not be committed

## Commands

```bash
# Run compliance check (when it exists)
python scripts/check-compliance.py

# Validate figures exist
python scripts/validate_figures.py

# Check for staged private files
git diff --cached --name-only | grep -E "(subject|individual|plr_trace)"
```

## What To Do When Validation Fails

1. **Hardcoded combo**: Replace with YAML load
   ```python
   # Wrong
   combos = ["ground_truth", "best_ensemble"]

   # Right
   import yaml
   cfg = yaml.safe_load(open("configs/VISUALIZATION/plot_hyperparam_combos.yaml"))
   combos = cfg["standard_combos"]
   ```

2. **Missing ground_truth**: Add to combo list
3. **Too many curves**: Split into multiple figures
4. **PRIVATE JSON staged**: Unstage with `git reset HEAD <file>`
