# GitHub Issue: Comprehensive Test Documentation

**Issue Type**: Enhancement
**Priority**: High
**Labels**: documentation, testing, developer-experience

## Summary

Create comprehensive documentation for all tests in this project so new developers can understand:
1. What each test validates
2. Common failure scenarios and their fixes
3. How to debug failing tests

## Background

Currently, test failures can be cryptic for new developers. For example:
- `validate_r_figures.R` fails with "Missing source(...config_loader.R)" but doesn't explain WHY this is required
- `validate_python_hardcoding.py` reports "Hardcoded color '#FF0000'" but doesn't guide the developer on the correct fix

## Requirements

### 1. Add Intuitive Error Messages

Each validation script should wrap checks in try/except with explanatory messages:

```python
# BEFORE (cryptic)
if not re.match(r'^#[0-9A-Fa-f]{6}$', value):
    raise ValueError(f"Invalid color: {value}")

# AFTER (helpful)
try:
    if re.match(r'^#[0-9A-Fa-f]{6}$', value):
        raise HardcodedColorError(
            f"Line {node.lineno}: Found hardcoded hex color '{value}'\n"
            f"\n"
            f"FIX: Replace with a semantic color reference:\n"
            f"  - Python: Use COLORS['semantic_name'] from plot_config.py\n"
            f"  - R: Use resolve_color('--color-ref', color_defs)\n"
            f"\n"
            f"Available colors are defined in:\n"
            f"  - configs/VISUALIZATION/figure_colors.yaml\n"
            f"\n"
            f"See: .claude/CLAUDE.md (Anti-Hardcoding section) for full guidelines"
        )
except Exception as e:
    log_failure_with_context(e)
```

### 2. Test Documentation Template

For each test file, create a corresponding `.md` documentation file:

```
tests/
├── test_figure_qa/
│   ├── test_data_provenance.py
│   ├── test_data_provenance.md  # NEW: Documentation
│   ├── test_visual_elements.py
│   ├── test_visual_elements.md  # NEW: Documentation
│   ...
```

Each documentation file should include:

```markdown
# Test: test_data_provenance.py

## Purpose
Validates that all figure data comes from real experimental results, not synthetic data.

## What It Checks
1. JSON data files reference valid database sources
2. Data hashes match expected values
3. No synthetic/fake data patterns detected

## Common Failures and Fixes

### Failure: "Synthetic data detected"
**Cause**: Figure JSON contains data generated with `np.random` instead of real predictions.
**Fix**: Use `load_bootstrap_predictions()` from `src/data_io/` to load real data.
**Reference**: See CRITICAL-FAILURE-001 in `.claude/docs/meta-learnings/`

### Failure: "Database hash mismatch"
**Cause**: The DuckDB database has been modified since data was exported.
**Fix**: Re-run `make extract` to regenerate the database, then re-export figure data.

## Running This Test
```bash
pytest tests/test_figure_qa/test_data_provenance.py -v
```

## Related Files
- `src/viz/figure_data.py` - Figure data export functions
- `configs/VISUALIZATION/figure_registry.yaml` - Figure specifications
```

### 3. Validator Scripts Improvements

#### `scripts/validate_r_figures.R`

Add comprehensive error messages:

```r
check_sources_config_loader <- function(content, filename) {
  if (!grepl("source.*config_loader\\.R", content)) {
    return(paste0(
      filename, ": Missing 'source(...config_loader.R)'\n",
      "\n",
      "  WHY: config_loader.R provides:\n",
      "    - load_color_definitions() for YAML color loading\n",
      "    - validate_data_source() for data provenance tracking\n",
      "    - resolve_color() for semantic color resolution\n",
      "\n",
      "  FIX: Add this to the top of your script:\n",
      "    source(file.path(PROJECT_ROOT, 'src/r/figure_system/config_loader.R'))\n",
      "\n",
      "  See: .claude/CLAUDE.md (MANDATORY Pattern for R) for full example"
    ))
  }
  NULL
}
```

#### `scripts/validate_python_hardcoding.py`

Add contextual fix suggestions:

```python
class HardcodingDetector(ast.NodeVisitor):
    """Detect hardcoded values that should come from config."""

    ERROR_MESSAGES = {
        'hex_color': (
            "Hardcoded color '{value}' found on line {line}\n"
            "\n"
            "FIX: Replace with semantic color from COLORS dict:\n"
            "    from src.viz.plot_config import COLORS\n"
            "    color = COLORS['primary']  # instead of '#006BA2'\n"
            "\n"
            "Available colors: {available_colors}\n"
            "Defined in: configs/VISUALIZATION/figure_colors.yaml"
        ),
        'savefig': (
            "Direct {obj}.savefig() call on line {line}\n"
            "\n"
            "FIX: Use save_figure() which handles:\n"
            "    - JSON data export for reproducibility\n"
            "    - Proper directory routing\n"
            "    - DPI and format settings\n"
            "\n"
            "Example:\n"
            "    from src.viz.plot_config import save_figure\n"
            "    save_figure(fig, 'figure_name', data=data_dict)"
        ),
        'hardcoded_path': (
            "Hardcoded path pattern on line {line}\n"
            "\n"
            "FIX: Use the figure system for automatic routing:\n"
            "    save_figure(fig, 'name')  # Routes to correct directory\n"
            "\n"
            "Directories are defined in:\n"
            "    configs/VISUALIZATION/figure_layouts.yaml"
        )
    }
```

### 4. Pre-commit Hook Improvements

Add helpful messages to pre-commit output:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: validate-r-figures
        name: Validate R figure scripts
        entry: bash -c 'Rscript scripts/validate_r_figures.R || (echo ""; echo "=== HELP ==="; echo "See docs/planning/test-documentation.md for common fixes"; exit 1)'
        language: system
        files: 'src/r/figures/fig.*\.R$'
```

## Acceptance Criteria

- [ ] All validation scripts provide actionable error messages
- [ ] Each test file has corresponding documentation
- [ ] Common failure scenarios are documented with fixes
- [ ] Pre-commit hooks provide help links on failure
- [ ] New developer can fix their first test failure within 5 minutes using documentation

## Related

- `.claude/docs/meta-learnings/CRITICAL-FAILURE-002-hardcoding-despite-existing-systems.md`
- `docs/planning/hardcoding-guardrails-improvement.md`
- `CONTRIBUTING.md` (to be updated with testing guide)

## Notes

This documentation should be auto-context loaded for Claude Code sessions working on tests so the AI assistant can provide accurate fix suggestions.
