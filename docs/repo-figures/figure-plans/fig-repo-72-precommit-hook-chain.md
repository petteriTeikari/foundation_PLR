# fig-repo-72: Pre-commit Hook Execution Chain

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-72 |
| **Title** | Pre-commit: 7 Hooks That Guard Every Commit |
| **Complexity Level** | L3 |
| **Target Persona** | ML Engineer / Research Scientist |
| **Location** | `docs/explanation/quality-gates.md`, root `CONTRIBUTING.md` |
| **Priority** | P1 (Critical) |

## Purpose

Show the 7 pre-commit hooks that run on every `git commit`, what each one catches, which files trigger each hook, and what to do when a hook fails. Developers need to understand that hooks are conditional (only some run on every commit) and how to fix failures without corrupting the previous commit.

## Key Message

7 pre-commit hooks run in sequence on `git commit`. Each guards against a specific class of violation: style (ruff), registry integrity, R hardcoding, computation decoupling, renv sync, and synthetic data isolation. When a hook fails, the commit is NOT created -- fix, re-stage, and create a NEW commit.

## Content Specification

### Panel 1: Hook Execution Chain

```
git commit -m "my changes"
  │
  │  Pre-commit framework checks which hooks apply
  │  based on staged files matching each hook's
  │  'files:' or 'types:' pattern
  │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  HOOK 1: ruff (format)                                               │
│  ─────────────────────                                               │
│  Source: astral-sh/ruff-pre-commit v0.9.1                            │
│  Triggers: ALL staged .py files                                      │
│  Action: Auto-formats Python code                                    │
│  On fail: Files modified in-place, re-stage with git add             │
│  Catches: Inconsistent formatting, trailing whitespace               │
├─────────────────────────────────────────────────────────────────────┤
│  HOOK 2: ruff (lint)                                                 │
│  ────────────────────                                                │
│  Source: astral-sh/ruff-pre-commit v0.9.1                            │
│  Args: --fix --config=pyproject.toml                                 │
│  Triggers: ALL staged .py files                                      │
│  Action: Lints + auto-fixes where possible                           │
│  On fail: Some fixes auto-applied, some need manual fix              │
│  Catches: Unused imports, style violations, common errors            │
├─────────────────────────────────────────────────────────────────────┤
│  HOOK 3: registry-integrity                                          │
│  ──────────────────────────                                          │
│  Source: local (scripts/verify_registry_integrity.py)                │
│  Triggers: ONLY when staged files match:                             │
│    configs/mlflow_registry/ | configs/registry_canary.yaml |         │
│    src/data_io/registry.py | tests/test_registry.py                  │
│  Action: Verifies all 5 registry layers agree on 11/8/5 counts      │
│  On fail: Update ALL 5 layers to match                               │
│  Catches: Method count tampering, out-of-sync registry sources       │
├─────────────────────────────────────────────────────────────────────┤
│  HOOK 4: registry-validation                                         │
│  ───────────────────────────                                         │
│  Source: local (pytest tests/test_registry.py)                       │
│  Triggers: Same file pattern as registry-integrity                   │
│  Action: Runs registry test suite                                    │
│  On fail: Registry YAML vs code mismatch, fix and re-stage           │
│  Catches: Registry YAML definitions not matching Python constants    │
├─────────────────────────────────────────────────────────────────────┤
│  HOOK 5: r-hardcoding-check                                         │
│  ───────────────────────────                                         │
│  Source: local (scripts/check_r_hardcoding.py)                       │
│  Triggers: ALL staged .R files                                       │
│  Action: Scans for banned patterns in R code                         │
│  On fail: Replace hardcoded values with figure system functions      │
│  Catches: Hex colors (#RRGGBB), ggsave(), custom themes in R        │
│  Correct: load_color_definitions(), save_publication_figure(),       │
│           theme_foundation_plr()                                     │
├─────────────────────────────────────────────────────────────────────┤
│  HOOK 6: computation-decoupling                                      │
│  ──────────────────────────────                                      │
│  Source: local (scripts/check_computation_decoupling.py)             │
│  Triggers: ONLY staged .py files in src/viz/                         │
│  Action: Scans for banned imports in visualization code              │
│  On fail: Remove banned import, read from DuckDB instead             │
│  Catches: sklearn.metrics, scipy.stats, src/stats/* in src/viz/     │
│  Correct: conn.execute("SELECT auroc FROM essential_metrics")        │
├─────────────────────────────────────────────────────────────────────┤
│  HOOK 7: renv-sync-check                                            │
│  ────────────────────────                                            │
│  Source: local (Rscript scripts/check_renv_sync.R)                   │
│  Triggers: Staged .R files OR renv.lock OR src/r/ changes            │
│  Action: Verifies renv.lock is in sync with R deps                   │
│  On fail: Run renv::snapshot() to update lockfile                    │
│  KNOWN ISSUE: Pre-existing failure, bypass with                      │
│    SKIP=renv-sync-check git commit -m "..."                          │
├─────────────────────────────────────────────────────────────────────┤
│  HOOK 8: extraction-isolation-check                                  │
│  ──────────────────────────────────                                  │
│  Source: local (scripts/check_extraction_isolation.py)               │
│  Triggers: scripts/extract*.py | src/utils/data_mode.py |           │
│            configs/data_isolation.yaml                                │
│  Action: Ensures synthetic data never contaminates production paths  │
│  On fail: Fix data mode logic in extraction scripts                  │
│  Catches: Synthetic DB paths in production extraction code           │
├─────────────────────────────────────────────────────────────────────┤
│  HOOK 9: figure-isolation-check                                      │
│  ──────────────────────────────                                      │
│  Source: local (scripts/check_figure_isolation.py)                   │
│  Triggers: src/viz/ | figures/ | configs/data_isolation.yaml         │
│  Action: Ensures synthetic data never appears in generated figures   │
│  On fail: Remove synthetic data references from figure code          │
│  Catches: Synthetic data paths in figure generation or output dirs   │
└─────────────────────────────────────────────────────────────────────┘
  │
  ├── ALL PASS → Commit created successfully
  │
  └── ANY FAIL → Commit NOT created
       │
       ├── Fix the issue
       ├── git add <fixed files>     ← RE-STAGE
       └── git commit -m "message"   ← NEW COMMIT (never --amend!)
```

### Panel 2: Hook Trigger Matrix

```
WHICH HOOKS RUN ON WHICH FILES?

Staged File Pattern          Hooks That Run
───────────────────────     ────────────────────────────
*.py (any Python file)      ruff-format, ruff-lint
src/viz/*.py                ruff-format, ruff-lint,
                            computation-decoupling
*.R (any R file)            r-hardcoding-check,
                            renv-sync-check
configs/mlflow_registry/*   registry-integrity,
                            registry-validation
src/data_io/registry.py     registry-integrity,
                            registry-validation
tests/test_registry.py      registry-integrity,
                            registry-validation
configs/registry_canary.yaml registry-integrity,
                            registry-validation
renv.lock                   renv-sync-check
src/r/*                     renv-sync-check
scripts/extract*.py         extraction-isolation-check
src/utils/data_mode.py      extraction-isolation-check
configs/data_isolation.yaml extraction-isolation-check,
                            figure-isolation-check
src/viz/*                   figure-isolation-check,
                            computation-decoupling
figures/*                   figure-isolation-check
```

### Panel 3: Failure Recovery

```
WHEN A HOOK FAILS
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  git commit -m "my changes"                                          │
│    │                                                                 │
│    ├── Hook passes → Commit created                                  │
│    │                                                                 │
│    └── Hook FAILS → Commit NOT created                               │
│         │                                                            │
│         ├── ruff auto-fixed files?                                   │
│         │   └── git add <auto-fixed files> → git commit again        │
│         │                                                            │
│         ├── registry mismatch?                                       │
│         │   └── Update all 5 registry layers → git add → commit      │
│         │                                                            │
│         ├── computation decoupling?                                   │
│         │   └── Remove banned import from src/viz/ → git add → commit│
│         │                                                            │
│         ├── r-hardcoding?                                            │
│         │   └── Use load_color_definitions() → git add → commit      │
│         │                                                            │
│         └── renv-sync-check (known issue)?                           │
│             └── SKIP=renv-sync-check git commit -m "message"         │
│                                                                      │
│  CRITICAL: Never use --amend after hook failure!                     │
│  The failed commit does NOT exist. --amend would modify              │
│  the PREVIOUS (unrelated) commit.                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Panel 4: CRITICAL-FAILURE Lineage

```
EACH HOOK TRACES TO A CRITICAL-FAILURE

Hook                         Prevents Recurrence Of
──────────────────────      ────────────────────────────────
ruff (format + lint)        General code quality
registry-integrity          CRITICAL-FAILURE-002 (mixed featurization)
registry-validation         CRITICAL-FAILURE-002
r-hardcoding-check          CRITICAL-FAILURE-004 (hardcoded values)
computation-decoupling      CRITICAL-FAILURE-003 (compute in viz)
renv-sync-check             R4R reproducibility concern
extraction-isolation-check  CRITICAL-FAILURE-001 (synthetic in prod)
figure-isolation-check      CRITICAL-FAILURE-001 (synthetic in figures)
```

## Spatial Anchors

```yaml
layout_flow: "Top-to-bottom sequential chain with branching failure recovery"
spatial_anchors:
  git_commit:
    x: 0.5
    y: 0.02
    content: "git commit triggers pre-commit framework"
  ruff_hooks:
    x: 0.5
    y: 0.1
    content: "Hooks 1-2: ruff format + lint (all .py files)"
  registry_hooks:
    x: 0.5
    y: 0.25
    content: "Hooks 3-4: registry integrity + validation (conditional)"
  r_hook:
    x: 0.5
    y: 0.38
    content: "Hook 5: r-hardcoding-check (all .R files)"
  decoupling_hook:
    x: 0.5
    y: 0.48
    content: "Hook 6: computation-decoupling (src/viz/ only)"
  isolation_hooks:
    x: 0.5
    y: 0.6
    content: "Hooks 7-9: renv-sync, extraction/figure isolation"
  outcome:
    x: 0.5
    y: 0.75
    content: "All pass → commit created; any fail → fix, re-stage, new commit"
  failure_recovery:
    x: 0.5
    y: 0.9
    content: "Failure recovery flowchart"
```

## Content Elements

### Key Structures
| Name | Role (Semantic Tag) | Description |
|------|---------------------|-------------|
| ruff format | `traditional_method` | Auto-format Python code |
| ruff lint | `traditional_method` | Lint + auto-fix Python code |
| registry-integrity | `primary_pathway` | 5-layer registry agreement check |
| registry-validation | `primary_pathway` | pytest registry assertions |
| r-hardcoding-check | `primary_pathway` | R code hardcoding scanner |
| computation-decoupling | `primary_pathway` | Import ban enforcement for src/viz/ |
| renv-sync-check | `secondary_pathway` | R lockfile sync (known bypass) |
| extraction-isolation | `primary_pathway` | Synthetic-production boundary guard |
| figure-isolation | `primary_pathway` | Synthetic data in figures guard |

### Relationships/Connections
| From | To | Type | Label |
|------|-----|------|-------|
| git commit | ruff format | Sequential | "first" |
| ruff format | ruff lint | Sequential | "then" |
| ruff lint | registry-integrity | Sequential | "then (if files match)" |
| registry-integrity | registry-validation | Sequential | "then" |
| registry-validation | r-hardcoding-check | Sequential | "then (if .R files)" |
| r-hardcoding-check | computation-decoupling | Sequential | "then (if src/viz/)" |
| computation-decoupling | renv-sync-check | Sequential | "then" |
| renv-sync-check | extraction-isolation | Sequential | "then" |
| extraction-isolation | figure-isolation | Sequential | "then" |
| Any hook failure | Fix + re-stage | Branch | "commit NOT created" |

### Callout Boxes
| Title | Content | Location |
|-------|---------|----------|
| "CONDITIONAL" | Most hooks only run when matching files are staged (files: pattern) | Top-right |
| "NEVER --amend" | After hook failure, the commit does not exist. --amend modifies the previous commit. | Bottom |
| "KNOWN BYPASS" | SKIP=renv-sync-check for pre-existing renv failure | Next to renv hook |

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `.pre-commit-config.yaml` | All 9 hook definitions (2 ruff + 7 local) |
| `pyproject.toml` | ruff configuration (rules, line length, etc.) |
| `scripts/verify_registry_integrity.py` | Registry integrity check script |
| `scripts/check_r_hardcoding.py` | R hardcoding detection script |
| `scripts/check_computation_decoupling.py` | Import ban enforcement script |
| `scripts/check_renv_sync.R` | renv lockfile sync check |
| `scripts/check_extraction_isolation.py` | Extraction isolation guard |
| `scripts/check_figure_isolation.py` | Figure isolation guard |

## Code Paths

| Module | Role |
|--------|------|
| `scripts/verify_registry_integrity.py` | Checks canary, YAML, Python, tests agree on 11/8/5 |
| `scripts/check_r_hardcoding.py` | Scans R files for hex colors, ggsave(), custom themes |
| `scripts/check_computation_decoupling.py` | Scans src/viz/ for banned sklearn/scipy imports |
| `scripts/check_renv_sync.R` | Compares installed R packages against renv.lock |
| `scripts/check_extraction_isolation.py` | Validates synthetic data stays in synthetic paths |
| `scripts/check_figure_isolation.py` | Validates figures are generated from production data |
| `configs/registry_canary.yaml` | Reference counts for registry integrity |
| `configs/data_isolation.yaml` | Data isolation boundary definitions |

## Extension Guide

To add a new pre-commit hook:
1. Create validation script in `scripts/check_<name>.py`
2. Add hook definition in `.pre-commit-config.yaml` under `- repo: local`
3. Set `files:` pattern to limit when the hook runs (avoid `always_run: true`)
4. Set `pass_filenames: true` if the script checks individual files, `false` for project-wide checks
5. Set `types: [python]` or `types: [r]` to filter by file type
6. Set `stages: [pre-commit]`
7. Add corresponding CI job step in `.github/workflows/ci.yml` quality-gates
8. Document the corresponding CRITICAL-FAILURE it prevents
9. Update this figure plan with the new hook

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-72",
    "title": "Pre-commit: 7 Hooks That Guard Every Commit"
  },
  "content_architecture": {
    "primary_message": "9 pre-commit hooks run sequentially on git commit: ruff (2), registry (2), R hardcoding, computation decoupling, renv sync, and 2 data isolation checks. When any hook fails, the commit is NOT created.",
    "layout_flow": "Top-to-bottom sequential chain with conditional branching",
    "spatial_anchors": {
      "git_commit": {"x": 0.5, "y": 0.02},
      "ruff_hooks": {"x": 0.5, "y": 0.1},
      "registry_hooks": {"x": 0.5, "y": 0.25},
      "r_hook": {"x": 0.5, "y": 0.38},
      "decoupling_hook": {"x": 0.5, "y": 0.48},
      "isolation_hooks": {"x": 0.5, "y": 0.6},
      "outcome": {"x": 0.5, "y": 0.75},
      "failure_recovery": {"x": 0.5, "y": 0.9}
    },
    "key_structures": [
      {
        "name": "ruff (format + lint)",
        "role": "traditional_method",
        "is_highlighted": false,
        "labels": ["All .py files", "Auto-fix + lint", "astral-sh v0.9.1"]
      },
      {
        "name": "registry-integrity",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Conditional: registry files only", "Checks 11/8/5 counts", "5-layer agreement"]
      },
      {
        "name": "registry-validation",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Conditional: registry files only", "pytest test_registry.py"]
      },
      {
        "name": "r-hardcoding-check",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["All .R files", "No hex colors", "No ggsave()"]
      },
      {
        "name": "computation-decoupling",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["src/viz/ .py only", "No sklearn imports", "CRITICAL-FAILURE-003"]
      },
      {
        "name": "renv-sync-check",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": [".R or renv.lock", "Known bypass: SKIP=renv-sync-check"]
      },
      {
        "name": "extraction-isolation",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Extraction scripts", "No synthetic in production"]
      },
      {
        "name": "figure-isolation",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["src/viz/ + figures/", "No synthetic in figures"]
      }
    ],
    "callout_boxes": [
      {"heading": "CONDITIONAL EXECUTION", "body_text": "Most hooks only run when staged files match their files: pattern. Ruff runs on all .py; registry hooks only on registry-related files."},
      {"heading": "NEVER --AMEND", "body_text": "After hook failure, the commit does NOT exist. Using --amend would modify the previous (unrelated) commit, potentially destroying work."},
      {"heading": "KNOWN BYPASS", "body_text": "renv-sync-check has a pre-existing failure. Use SKIP=renv-sync-check git commit -m 'message' to bypass."}
    ]
  }
}
```

## Alt Text

Sequential chain of 9 pre-commit hooks from ruff through data isolation, with conditional triggers and failure recovery branching to fix-re-stage-new-commit.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
