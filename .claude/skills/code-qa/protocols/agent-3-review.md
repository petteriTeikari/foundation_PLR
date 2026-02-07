# Agent 3: Review & Quality Gate

## Mandate

Review all fix proposals from Agent 2 for correctness and safety. Reject false positives.

## Review Checklist Per Category

### Dead Code Deletion Review

For each proposed deletion, verify:

1. **Is it truly uncalled?**
   - Check for dynamic dispatch: `getattr(obj, name)`, `globals()[name]`
   - Check for decorator registration: `@register`, `@app.route`
   - Check for callback patterns: `callback=function_name`
   - Check for `__all__` exports in `__init__.py`
   - Check for string-based imports: `importlib.import_module`

2. **Is it called from outside src/?**
   - Check `tests/` for usage (test-only code is still valid)
   - Check `scripts/` for CLI usage
   - Check `Makefile` for targets
   - Check Jupyter notebooks (`.ipynb`)

3. **Is it a public API?**
   - Exported in `__init__.py` → DO NOT DELETE
   - Has docstring suggesting public use → FLAG for human review
   - Part of a Protocol/ABC → DO NOT DELETE (interface contract)

**Verdict**: APPROVE only if all 3 checks pass.

### Hardcoding Fix Review

For each proposed config replacement:

1. **Does the config key actually exist?**
   - Load the referenced YAML file
   - Verify the key path exists
   - Verify the value matches the hardcoded one

2. **Is the import available?**
   - Verify `COLORS`, `setup_style`, etc. are importable
   - Verify the config loader works in the target module

3. **Is this truly hardcoding or intentional?**
   - Constants in test fixtures → ACCEPTABLE (not hardcoding)
   - Default values in function signatures → REVIEW CAREFULLY
   - Constants in `__main__` blocks → LOWER PRIORITY
   - Canonical definitions (COLORS dict, color_palettes.R) → EXCLUDED (these ARE the source)
   - `.get()` fallbacks that duplicate the primary value → HIGH (masks broken config)
   - Legacy code in `src/tools/` → MEDIUM (not active analysis code)

4. **Is this acknowledged tech debt?**
   - Has `# noqa:` or `# nolint:` suppression comment → Track as "acknowledged debt"
   - Has `# DEPRECATED` marker → Track separately, verify if code is actually unreachable
   - Has `# TODO` marker → Work in progress, defer

5. **Is this a dual source of truth?**
   - Same concept (e.g., color name) defined with DIFFERENT values in Python and YAML/R
   - Requires structural fix (unify sources), not simple replacement
   - Mark as ARCHITECTURE_SMELL with both conflicting locations

### Duplicate Code Review

1. **Are the functions TRULY identical?**
   - Compare normalized AST, not just text
   - Different default arguments → NOT duplicate
   - Different error handling → NOT duplicate

2. **Is consolidation safe?**
   - Do both callers expect the same interface?
   - Would moving create a circular import?
   - Is the target module (`common.py`, `utils.py`) appropriate?

### Dead Config Review

1. **Could the key be used dynamically?**
   - Hydra interpolation: `${other.key}` referencing this one
   - OmegaConf resolver: computed from this value
   - Used in Hydra overrides from CLI

2. **Is it used by external tools?**
   - MLflow expects certain parameter names
   - Docker compose references config values
   - GitHub Actions may reference config

## Verdict Categories

| Verdict | Meaning | Action |
|---------|---------|--------|
| **APPROVE** | Fix is correct and safe | Include in batch |
| **REJECT** | False positive or unsafe | Remove from batch |
| **REVISE** | Fix direction is right but implementation needs adjustment | Return to Agent 2 |
| **DEFER** | Uncertain, needs human expertise | Flag for human decision |

## Output Format

```yaml
reviews:
  - proposal_id: D001
    verdict: APPROVE
    reason: "Confirmed 0 callers, not in __init__.py, not dynamically dispatched"

  - proposal_id: H005
    verdict: REJECT
    reason: "This is a test fixture default, not production hardcoding"

  - proposal_id: DUP002
    verdict: REVISE
    reason: "Functions differ in error handling. Consolidate the common part only."
    suggested_revision: "Extract only the config-loading prefix (lines 10-15)"

summary:
  approved: 22
  rejected: 4
  revised: 2
  deferred: 0
```

## Constraints

- When in doubt, REJECT (false negatives are safer than false positives)
- Never approve deletion of `__init__.py` exports
- Never approve deletion of functions with `# TODO` or `# FIXME` comments (work in progress)
- Flag any fix that touches > 50 lines as HIGH RISK for human attention
