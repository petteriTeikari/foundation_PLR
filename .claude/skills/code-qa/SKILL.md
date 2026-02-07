# /code-qa - Codebase Quality Assurance with Self-Correction

Iterative codebase QA using the Ralph Wiggum self-correction pattern. Finds and fixes dead code, duplicates, hardcoded values, suboptimal patterns, and config decoupling violations.

## Usage

- `/code-qa` - Full QA run (discovery → proposals → approval → fix)
- `/code-qa report` - Discovery only, no fixes (dry-run)
- `/code-qa scope=tier1` - Tier 1: `src/viz/`, `src/stats/`, `src/data_io/`, `configs/` (~60 files, always fits context)
- `/code-qa scope=tier2` - Tier 2: Tier 1 + `src/orchestration/`, `src/extraction/` (~100 files)
- `/code-qa scope=full` - All `src/` files (batched in 3 passes by directory)

## What This Checks

### 1. Dead Code Detection
- Unreferenced functions/classes (no callers in the codebase)
- Unused imports
- Commented-out code blocks (> 5 lines)
- Files with no imports from anywhere
- **Safety**: Only flags code with HIGH confidence (no callers found via AST + grep)

### 2. Duplicate Code Detection
- Near-identical functions across files (> 80% similarity)
- Copy-pasted utility patterns that should be consolidated
- Duplicate config loading boilerplate

### 3. Hardcoding Violations (ZERO TOLERANCE)
- Hex color codes (`#RRGGBB`) outside `configs/`
- Literal file paths outside `configs/`
- Method names (outlier/imputation/classifier) not loaded from registry
- Numeric constants that exist in `configs/defaults.yaml`
- DPI, dimensions, font sizes not from config

### 4. Config Decoupling (Bidirectional)
- **Code → Config**: Values hardcoded in code that should come from YAML
- **Config → Code**: Parameters in YAML that no code ever loads (dead config)
- **Cross-reference**: Every `yaml.safe_load()` / `OmegaConf.load()` path verified

### 5. Suboptimal Patterns
- `grep`/`sed`/`awk` used for structured data (BANNED)
- `import re` for parsing Python/YAML/JSON (BANNED)
- Metric computation in `src/viz/` (BANNED - extraction only)
- `ggsave()` instead of `save_publication_figure()` in R
- Missing `setup_style()` calls before matplotlib
- `plt.savefig()` instead of `save_figure()`

### 6. Architecture Smells (added Iteration 1)
- **Dual source of truth**: Same concept defined with different values in Python and YAML/R
- **Acknowledged tech debt**: `# noqa`, `# nolint`, `# DEPRECATED` suppressions tracked separately
- **Legacy vs active code**: `src/tools/` (legacy, MEDIUM) vs `src/viz/` (active, CRITICAL)

## Architecture (Ralph Wiggum Pattern)

```
Iteration N (max 3):
  Agent 1: Discovery (read-only scan)
    - AST-parse all Python files for dead code, imports, patterns
    - Grep for hardcoded values, banned patterns
    - Cross-reference config files bidirectionally
    - Output: structured issue list with severity + file:line

  Agent 2: Fix Proposal (generate changes)
    - For each issue, propose a specific fix
    - Group fixes by category for batch approval
    - Estimate risk level per fix

  Agent 3: Review (quality gate)
    - Review proposals for false positives
    - Verify dead code is TRULY dead (not called via dynamic dispatch)
    - Ensure config replacements are correct
    - Output: approved/rejected per proposal

  Human Approval: Show categorized diff, get sign-off per category
  Execute: Apply approved fixes
  Git Checkpoint: Commit after each category
  Validate: Run tests, check nothing broke
```

## Severity Levels

| Level | Description | Action |
|-------|-------------|--------|
| **CRITICAL** | Inline hardcoded values in active code, banned patterns, metric computation in viz | Fix immediately |
| **HIGH** | Function defaults matching config, dead code (confirmed), duplicate functions, dual source of truth, `.get()` fallbacks | Fix in this iteration |
| **MEDIUM** | Legacy code violations (`src/tools/`), unused config params, suboptimal patterns | Fix if time permits |
| **LOW** | Demo/test function issues, style issues, stub config | Report only |
| **ACKNOWLEDGED** | Issues with `# noqa`/`# nolint`/`# DEPRECATED` markers | Track as tech debt |

## Scope (Tiered Scanning)

| Tier | Directories | Files | Use When |
|------|-------------|-------|----------|
| **Tier 1** | `src/viz/`, `src/stats/`, `src/data_io/`, `configs/` | ~60 | Default first pass, always fits context |
| **Tier 2** | Tier 1 + `src/orchestration/`, `src/extraction/`, `src/r/` | ~100 | Critical path scan |
| **Full** | All `src/` + `configs/` | ~360 | Batched in 3 passes by directory |

Config audit (72 YAML files) runs at ALL tiers. Jupyter notebooks + Makefile always checked for dead code refs.

## Pre-Flight Checks

Before scanning, verify:
1. `pytest tests/` passes (don't QA a broken codebase)
2. Registry files exist: `configs/mlflow_registry/parameters/classification.yaml`
3. Git working tree is clean (for clean checkpoints)

## Safety Constraints

- NEVER remove code that might be called via dynamic dispatch, decorators, or string-based imports
- NEVER remove `__init__.py` exports (other repos may depend on them)
- NEVER modify test files (only source code)
- Dead code removal requires: 0 callers via AST + 0 grep matches + human approval
- Config removal requires: 0 loaders found + human verification of YAML key

## Files

- `protocols/agent-1-discovery.md` - Read-only scanning agent
- `protocols/agent-2-fix-proposal.md` - Change generation agent
- `protocols/agent-3-review.md` - Quality gate agent
- `protocols/convergence.md` - When to stop
- `reference/banned-patterns.yaml` - Patterns to flag
- `reference/config-mapping.md` - Config key → code loader mapping
- `state/qa-state.json` - Progress tracking between iterations
