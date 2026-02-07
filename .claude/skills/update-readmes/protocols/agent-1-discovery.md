# Agent 1: Discovery & Staleness Detection

## Mandate

Scan all documentation files and produce a structured issue list. READ-ONLY -- no modifications.

## Scan Workflow

### Phase 1: README Inventory

Collect all project-level README.md files:

```bash
find . -name "README.md" \
  -not -path "*/node_modules/*" \
  -not -path "*/.venv/*" \
  -not -path "*/renv/library/*" \
  -not -path "*/.pytest_cache/*" \
  -not -path "*/archived/*" | sort
```

For each README, record:
- Path
- Last modified date
- Word count
- Number of internal links
- Number of image references

### Phase 2: Cross-Reference Validation

For each `[text](path)` link in every README:

1. Resolve relative path from the README's directory
2. Check if target file exists
3. If target is another README, verify it's not empty
4. Record broken links with source file:line

For each `![alt](path)` image reference:
1. Resolve path
2. Check if image file exists in `docs/repo-figures/assets/` or `docs/repo-figures/generated/`
3. Record broken image refs

### Phase 3: Config Table Validation

For each README in `configs/*/`:

1. Parse the actual YAML files in that directory
2. Extract parameter names, types, default values
3. Compare against any parameter tables in the README
4. Flag mismatches:
   - Parameters in YAML but not in README table
   - Parameters in README table but deleted from YAML
   - Default values that have changed

For registry README (`configs/mlflow_registry/README.md`):
- Verify method counts match actual YAML lists (11 outlier, 8 imputation, 5 classifier)
- Verify method names match

### Phase 4: Module Coverage

For each directory in `src/`:

1. List all `.py` files (excluding `__pycache__`)
2. Check if the directory's README mentions each module
3. Flag new modules not documented in README
4. Flag README entries for deleted modules

### Phase 5: Docstring Audit

For each `.py` file in `src/` (by priority tier):

```python
import ast

tree = ast.parse(source)

for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        # Skip private functions (start with _) unless they're __init__
        if node.name.startswith('_') and node.name != '__init__':
            continue

        docstring = ast.get_docstring(node)

        if docstring is None:
            report_missing(node.name, node.lineno)
        else:
            # Check signature match
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                actual_params = [a.arg for a in node.args.args if a.arg != 'self']
                documented_params = extract_params_from_docstring(docstring)

                missing = set(actual_params) - set(documented_params)
                extra = set(documented_params) - set(actual_params)

                if missing:
                    report_drift(node.name, "missing params", missing)
                if extra:
                    report_drift(node.name, "extra params", extra)
```

### Phase 6: Figure Plan Gap Analysis

1. List all existing figure plans in `docs/repo-figures/figure-plans/`
2. List all plans-TODO in `docs/repo-figures/plans-TODO/`
3. List all generated assets in `docs/repo-figures/assets/`
4. Cross-reference:
   - Plans without corresponding assets = pending generation
   - Architecture features without any plan = gap
5. Identify new features since last figure plan batch:
   - New DuckDB tables (retention_metrics, cohort_metrics, etc.)
   - New Prefect flows
   - New pre-commit hooks
   - Two-block architecture refinements
   - Computation decoupling enforcement

## Output Format

```json
{
  "scan_timestamp": "ISO-8601",
  "summary": {
    "total_readmes": 47,
    "broken_links": 3,
    "stale_config_tables": 5,
    "undocumented_modules": 8,
    "missing_docstrings": 42,
    "docstring_drift": 7,
    "figure_gaps": 4
  },
  "issues": [
    {
      "id": "README-001",
      "category": "broken_link",
      "severity": "HIGH",
      "file": "src/viz/README.md",
      "line": 15,
      "description": "Link to src/viz/calibration.py but file is now calibration_plot.py",
      "suggested_fix": "Update link to calibration_plot.py"
    }
  ]
}
```

## Severity Levels

| Level | Description | Examples |
|-------|-------------|---------|
| **CRITICAL** | Factually wrong information | Wrong method count, wrong AUROC value |
| **HIGH** | Broken links, missing modules, stale tables | Dead cross-references |
| **MEDIUM** | Missing docstrings on P0/P1 modules | Public API without docs |
| **LOW** | Missing docstrings on P2/P3, cosmetic issues | Support code, formatting |
| **INFO** | Figure gaps, enhancement opportunities | New architecture not documented |

## Exclusions

- `node_modules/`, `.venv/`, `renv/library/`, `.pytest_cache/`, `archived/` -- third-party
- `__pycache__/` directories
- Private functions (start with `_`, except `__init__`)
- Test files (`tests/`) -- docstrings not required
- Generated files (`outputs/`, `figures/generated/`)
