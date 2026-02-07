# fig-repo-58: Pre-Commit Hook Enforcement Matrix

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-58 |
| **Title** | Pre-Commit Quality Gates: What Each Hook Catches |
| **Complexity Level** | L2 |
| **Target Persona** | ML Engineer / Research Scientist |
| **Location** | `CONTRIBUTING.md`, `docs/tutorials/adding-new-methods.md` |
| **Priority** | P1 (Critical) |

## Purpose

Show developers exactly what each pre-commit hook enforces, what violations it catches, and how to fix failures. This prevents "why did my commit fail?" confusion and documents the quality gate architecture.

## Content Specification

### Panel 1: Hook Overview (Matrix)

```
┌──────────────────────────────────────────────────────────────────────┐
│                     PRE-COMMIT QUALITY GATES                         │
├──────────────┬──────────────┬────────────┬──────────────────────────┤
│ Hook         │ Language     │ What       │ What It Catches          │
├──────────────┼──────────────┼────────────┼──────────────────────────┤
│ ruff         │ Python       │ Lint       │ Unused imports, style    │
│ ruff-format  │ Python       │ Format     │ Black-compatible format  │
│ check_r_     │ R            │ Hardcoding │ Hex colors, ggsave(),    │
│ hardcoding   │              │            │ custom themes, hardcoded │
│              │              │            │ width/height             │
│ check_comp_  │ Python       │ Decoupling │ sklearn/stats imports    │
│ decoupling   │ (src/viz/)   │            │ in visualization code    │
│ renv-sync    │ R            │ Lock       │ renv.lock vs installed   │
└──────────────┴──────────────┴────────────┴──────────────────────────┘
```

### Panel 2: Enforcement Flow

```
Developer writes code
  │
  ▼
git commit
  │
  ▼
Pre-commit hooks run (in order):
  1. ruff check → Fix lint issues
  2. ruff format → Auto-format
  3. check_r_hardcoding.py → Reject hardcoded R values
  4. check_computation_decoupling.py → Reject sklearn in viz
  5. renv-sync-check → Verify R lockfile
  │
  ├─ ALL PASS → Commit created ✓
  └─ ANY FAIL → Commit blocked ✗
       │
       ▼
     Developer reads error message
     Fixes violation
     Tries git commit again
```

### Panel 3: Violation Examples & Fixes

| Violation | Hook | Error Message | Fix |
|-----------|------|---------------|-----|
| `color = "#006BA2"` in R | check_r_hardcoding | "Hardcoded hex color" | `color_defs[["--color-primary"]]` |
| `ggsave(path)` in R | check_r_hardcoding | "Use save_publication_figure()" | `save_publication_figure(plot, "name")` |
| `from sklearn.metrics import roc_auc_score` in viz | check_comp_decoupling | "Banned import in viz code" | Read from DuckDB instead |
| `width = 14` in R | check_r_hardcoding | "Hardcoded dimension" | Load from config YAML |

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `.pre-commit-config.yaml` | Hook definitions and versions |
| `scripts/check_r_hardcoding.py` | R hardcoding checker implementation |
| `scripts/check_computation_decoupling.py` | Computation decoupling checker |

## Code Paths

| Module | Role |
|--------|------|
| `scripts/check_r_hardcoding.py` | Scans R files for banned patterns |
| `scripts/check_computation_decoupling.py` | Scans src/viz/ for banned imports |
| `.pre-commit-config.yaml` | Orchestrates all hooks |

## Extension Guide

To add a new pre-commit hook:
1. Create checker script in `scripts/`
2. Add entry to `.pre-commit-config.yaml`
3. Add corresponding test in `tests/test_guardrails/`
4. Document in this figure plan and CONTRIBUTING.md

Note: Performance comparisons are in the manuscript, not this repository.
