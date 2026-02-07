# Agent 1: Discovery & Analysis

## Mandate

Scan all source files and produce a structured issue list. READ-ONLY — no modifications.

## Scan Categories

### 1. Dead Code (Python)

Use AST parsing (NOT grep) to find:

```python
import ast, os

# For each .py file in src/:
tree = ast.parse(source)

# Collect all defined functions/classes
defined = {node.name for node in ast.walk(tree)
           if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}

# Collect all Name references across ALL files
# A function is "dead" if defined but never referenced anywhere
```

**Cross-reference**: For each defined name, search ALL other files for usage:
- Direct calls: `function_name(`
- Imports: `from module import function_name`
- Attribute access: `obj.function_name`
- String references: `"function_name"` (dynamic dispatch safety)

**Confidence levels**:
- HIGH: 0 references found anywhere → likely dead
- MEDIUM: Only referenced in same file → possibly dead
- LOW: Referenced in test files only → might be test-only utility

Only report HIGH confidence as actionable.

**Additional sources to check before flagging dead code (reviewer recommendations):**
- Jupyter notebooks (`.ipynb`): Extract code cells and search for function usage
- `Makefile` targets: Parse for Hydra CLI overrides (`+key=value`, `key=value`)
- `scripts/` shell scripts: Parse for `python -m module` invocations
- Skip functions with decorators like `@register`, `@app.route`, `@hydra.main`
- Skip functions in files matching `*_registry.py`, `*_catalog.py`

### 2. Dead Code (R)

For each `.R` file:
- Find function definitions: `function_name <- function(`
- Search all other R files + Python subprocess calls for usage
- Check `source()` chains

### 3. Duplicate Code

For functions > 10 lines, compute similarity:
- Extract function body (AST)
- Normalize: strip comments, normalize whitespace, replace variable names
- Compare across files
- Flag pairs with > 80% token overlap

### 4. Hardcoding Violations

Scan for patterns (using AST for Python, regex only for non-structured content):

| Pattern | Where to Look | What It Means |
|---------|---------------|---------------|
| `#[0-9a-fA-F]{6}` | Python/R strings | Hardcoded color |
| Literal paths containing `/home/` | Python/R strings | Hardcoded absolute path |
| `"LOF"`, `"MOMENT-gt"`, etc. | Python/R outside configs/ | Hardcoded method name |
| `dpi=100`, `width=14` | Python/R | Hardcoded dimension |
| `n_iterations=1000` | Python/R | Hardcoded config value |
| `0.0354` (prevalence) | Python/R | Hardcoded config value |

**Exclusions**: Config files themselves, test fixtures, comments, docstrings.

**Severity Sub-Categories (from Iteration 1 learnings):**

| Sub-Type | Severity | Example |
|----------|----------|---------|
| Inline literal in active code | CRITICAL | `color = "#006BA2"` in a plot function |
| Function default matching config | HIGH | `n_bootstrap: int = 1000` (matches CLS_EVALUATION.BOOTSTRAP.n_iterations) |
| Fallback in `.get()` call | HIGH | `COLORS.get('key', '#666666')` — masks broken config |
| Canonical definition file | EXCLUDED | COLORS dict in `plot_config.py`, `color_palettes.R` — these ARE the source |
| Legacy tooling (`src/tools/`) | MEDIUM | Old preprocessing scripts with `/home/petteri/` paths |
| Demo/test function at EOF | LOW | `plt.savefig("/tmp/style_demo.png")` in `__main__` block |

**Acknowledged Tech Debt Detection:**
- Scan for `# noqa:`, `# nolint:`, `# DEPRECATED`, `# TODO` suppression comments
- Report these SEPARATELY as "acknowledged debt" — not new violations
- Track count of suppressed issues per file for tech debt dashboard

**Dual Source of Truth Detection (Architecture Smell):**
- When the SAME semantic concept (e.g., `ground_truth` color) is defined with DIFFERENT values in multiple files → flag as ARCHITECTURE_SMELL, not simple hardcoding
- Example: Python COLORS dict says `ground_truth = "#4A4A4A"` but `colors.yaml` says `ground_truth = "#666666"`
- These require structural fixes (unify source), not simple replacements

### 5. Config Decoupling (Bidirectional)

**Code → Config** (hardcoded values that should be in YAML):
- For each literal found in code, check if an equivalent exists in `configs/*.yaml`
- If yes → the code should be loading it, not hardcoding it

**Config → Code** (dead config parameters):
- For each key in every YAML file under `configs/`:
  - Search all Python/R files for that key name
  - If 0 references → potentially dead config
  - **Hydra interpolation check**: Parse YAML for `${other.key}` references — these count as usage
  - **OmegaConf resolvers**: `oc.env:VAR`, `oc.select:key`, `oc.deprecated:key` count as usage
  - **MLflow logging**: Keys passed to `mlflow.log_param()` / `log_params()` are NOT dead

### 6. Banned Patterns

Check all Python files for:
- `from sklearn.metrics import` in `src/viz/` (computation decoupling violation)
- `from src.stats.` in `src/viz/` (computation decoupling violation)
- `import re` used for YAML/Python/JSON parsing
- Subprocess calls to `grep`, `sed`, `awk` for structured data

Check all R files for:
- `ggsave(` (should use `save_publication_figure()`)
- Hex color literals in `geom_*()` calls
- Custom theme definitions (should use `theme_foundation_plr()`)

## Output Format

```yaml
issues:
  - id: D001
    category: dead_code
    severity: HIGH
    file: src/utils/old_helper.py
    line: 45
    name: unused_function
    evidence: "0 callers found across 305 Python files"
    confidence: HIGH

  - id: H001
    category: hardcoding
    severity: CRITICAL
    file: src/viz/plot_something.py
    line: 23
    pattern: 'color="#006BA2"'
    config_equivalent: "configs/VISUALIZATION/colors.yaml → primary"
    confidence: HIGH

  - id: C001
    category: dead_config
    severity: MEDIUM
    file: configs/defaults.yaml
    key: "UNUSED_PARAM.some_value"
    evidence: "0 references in any Python/R file"
    confidence: MEDIUM

  - id: A001
    category: architecture_smell
    severity: HIGH
    file: src/viz/plot_config.py
    line: 166
    pattern: 'COLORS["ground_truth"] = "#4A4A4A"'
    conflict_with: "configs/VISUALIZATION/colors.yaml → ground_truth = #666666"
    confidence: HIGH

summary:
  total_issues: 0
  by_severity: { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 }
  by_category: { dead_code: 0, duplicate: 0, hardcoding: 0, dead_config: 0, banned_pattern: 0, architecture_smell: 0 }
  acknowledged_debt: { noqa: 0, deprecated: 0, nolint: 0 }
  files_scanned: { python: 305, r: 56, yaml: 72 }
```

## Constraints

- Use `ast.parse()` for Python analysis, NEVER grep for code patterns
- Use proper YAML parser for config analysis
- Report file:line for every issue
- Include evidence string explaining WHY it's flagged
- Exclude `__pycache__`, `.git`, `node_modules`, `.venv`
- Exclude test files from dead code analysis (they're consumers, not producers)
- Exclude canonical definition files from hardcoding violations (see Severity Sub-Categories)
- Separate acknowledged tech debt (`# noqa`, `# nolint`, `# DEPRECATED`) from new findings
- Flag dual source of truth as ARCHITECTURE_SMELL with both conflicting locations
- Treat `src/tools/` as legacy (MEDIUM severity) vs `src/viz/`, `src/stats/` as active (CRITICAL)
