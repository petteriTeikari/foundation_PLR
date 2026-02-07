# Agent 2: Fix Proposal Generation

## Mandate

For each issue from Agent 1, propose a specific, minimal fix. Group by category for batch approval.

## Fix Templates by Category

### Dead Code → Deletion

```yaml
fix:
  issue_id: D001
  action: delete_function
  file: src/utils/old_helper.py
  lines: 45-67
  risk: LOW  # No callers found
  preview: |
    # DELETE: function `unused_function` (lines 45-67)
    # Evidence: 0 callers across 305 files
    # Last modified: git blame shows 2025-12-15
```

**Safety check before proposing deletion:**
- Confirm 0 callers (Agent 1 already checked)
- Check if exported in `__init__.py` (if yes, DON'T delete — it's public API)
- Check git blame: recently added code is less likely truly dead
- Check if it's a callback/hook (registered dynamically)

### Duplicate Code → Consolidation

```yaml
fix:
  issue_id: DUP001
  action: consolidate
  source_files:
    - src/viz/plot_a.py:load_config (lines 10-25)
    - src/viz/plot_b.py:load_config (lines 8-22)
  target: src/viz/common.py:load_config
  risk: MEDIUM  # Requires updating imports
  preview: |
    # MOVE shared function to src/viz/common.py
    # UPDATE imports in plot_a.py and plot_b.py
```

### Hardcoding → Config Loading

```yaml
fix:
  issue_id: H001
  action: replace_hardcoded
  file: src/viz/plot_something.py
  line: 23
  old: 'color="#006BA2"'
  new: 'color=COLORS["primary"]'
  config_source: configs/VISUALIZATION/colors.yaml
  risk: LOW
  preview: |
    # BEFORE: ax.plot(x, y, color="#006BA2")
    # AFTER:  ax.plot(x, y, color=COLORS["primary"])
    # Requires: from src.viz.plot_config import COLORS
```

For method names:
```yaml
fix:
  old: 'if method == "LOF":'
  new: 'if method in get_valid_outlier_methods():'
  requires_import: 'from src.data_io.registry import get_valid_outlier_methods'
```

For numeric constants:
```yaml
fix:
  old: 'prevalence = 0.0354'
  new: 'prevalence = cfg.CLS_EVALUATION.glaucoma_params.prevalence'
  requires: 'Hydra config or OmegaConf.load()'
```

### Dead Config → Removal or Documentation

```yaml
fix:
  issue_id: C001
  action: comment_or_remove
  file: configs/defaults.yaml
  key: UNUSED_PARAM.some_value
  risk: LOW
  recommendation: |
    # Option A: Remove if confirmed unused
    # Option B: Add comment "# TODO: Wire up to code or remove"
```

### Banned Pattern → Replacement

```yaml
fix:
  issue_id: B001
  action: replace_pattern
  file: src/viz/calibration.py
  line: 5
  old: 'from sklearn.metrics import brier_score_loss'
  new: '# Brier score comes from DuckDB, not computed here'
  risk: HIGH  # Must verify DuckDB has the column
```

## Grouping for Batch Approval

Present fixes in categories, each approved independently:

1. **CRITICAL fixes** (hardcoding, banned patterns) → Approve first
2. **Dead code deletions** (HIGH confidence only) → Approve second
3. **Duplicate consolidation** → Approve third
4. **Dead config cleanup** → Approve last

## Output Format

```yaml
fix_proposals:
  critical:
    count: 5
    fixes: [...]
    estimated_risk: LOW

  dead_code:
    count: 12
    fixes: [...]
    estimated_risk: MEDIUM

  duplicates:
    count: 3
    fixes: [...]
    estimated_risk: MEDIUM

  dead_config:
    count: 8
    fixes: [...]
    estimated_risk: LOW

total_fixes: 28
total_lines_affected: 450
```

## Constraints

- Every fix must be minimal (change only what's needed)
- Never introduce new dependencies
- Never change function signatures (could break callers)
- Dead code deletion: only HIGH confidence from Agent 1
- Always show before/after preview
- Group by risk level within each category
