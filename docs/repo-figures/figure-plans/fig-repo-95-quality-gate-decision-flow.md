# fig-repo-95: When a Pre-commit Hook Fails: Fix, Stage, Commit (Never Amend)

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-95 |
| **Title** | When a Pre-commit Hook Fails: Fix, Stage, Commit (Never Amend) |
| **Complexity Level** | L2 |
| **Target Persona** | All (especially new contributors) |
| **Location** | `CONTRIBUTING.md`, `docs/explanation/pre-commit-hooks.md` |
| **Priority** | P2 (High) |

## Purpose

Prevent the most common git mistake after a pre-commit hook failure: using `--amend` on a commit that never existed. When a hook fails, the commit is NOT created. Using `--amend` at that point modifies the PREVIOUS commit, potentially destroying unrelated work. This flowchart makes the correct recovery path unmistakable.

## Key Message

When a pre-commit hook fails, the commit did NOT happen. Fix the issue, re-stage with `git add`, then create a NEW commit. Never use `--amend` because it would modify the previous (unrelated) commit.

## Content Specification

### Panel 1: The Fork -- Pass vs Fail

```
git commit -m "my changes"
  |
  +---> Pre-commit hooks run (7 hooks in sequence)
        |
        +---------- ALL PASS ----------+
        |                               |
        v                               |
  Commit CREATED                        |
  (SHA exists, visible in git log)      |
  DONE                                  |
        |                               |
        +---------- ANY FAILS ---------+
                                        |
                                        v
                                  Commit NOT CREATED
                                  (no SHA, nothing in git log)
                                  |
                                  v
                            Continue to Panel 2...
```

### Panel 2: Recovery Flow (The Correct Path)

```
Hook FAILED --> Commit NOT created
  |
  v
Step 1: READ the error message
  |
  +-- "ruff" or "ruff-format"?
  |   |
  |   +-- Files were AUTO-FIXED by ruff
  |       Just re-stage: git add <auto-fixed files>
  |
  +-- "registry-integrity"?
  |   |
  |   +-- Method counts (11/8/5) don't match across sources
  |       Update ALL 5 layers:
  |         1. configs/registry_canary.yaml
  |         2. configs/mlflow_registry/parameters/classification.yaml
  |         3. src/data_io/registry.py (EXPECTED_*_COUNT)
  |         4. tests/test_registry.py (assert statements)
  |         5. .pre-commit-config.yaml (hook args)
  |
  +-- "computation-decoupling"?
  |   |
  |   +-- Banned import found in src/viz/*.py
  |       Remove the import (sklearn, scipy.stats, src/stats/*)
  |       Replace with DuckDB SELECT query
  |
  +-- "r-hardcoding-check"?
  |   |
  |   +-- Hex color or ggsave() found in R code
  |       Replace hex with load_color_definitions()
  |       Replace ggsave() with save_publication_figure()
  |
  +-- "renv-sync-check"?
      |
      +-- Known pre-existing failure (not your fault)
          Bypass: SKIP=renv-sync-check git commit -m "my changes"
  |
  v
Step 2: FIX the issue (per above)
  |
  v
Step 3: RE-STAGE fixed files
  git add <fixed files>
  |
  v
Step 4: Create a NEW commit
  git commit -m "my changes"      <-- NEW commit (correct!)
  git commit --amend               <-- WRONG! (see Panel 3)
```

### Panel 3: Why NOT --amend? (Critical Warning)

```
+========================================================================+
||                                                                        ||
||  WARNING: The --amend Trap                                             ||
||                                                                        ||
||  Timeline of commits:                                                  ||
||                                                                        ||
||  ... --> [abc123] "previous work"  --> (your failed attempt)           ||
||                                         ^                              ||
||                                         |                              ||
||                                    This commit DOES NOT EXIST          ||
||                                    (hook failure = no commit)          ||
||                                                                        ||
||  If you run: git commit --amend -m "my changes"                       ||
||                                                                        ||
||  ... --> [def456] "my changes"     (OVERWRITES "previous work"!)      ||
||                                                                        ||
||  The PREVIOUS commit (abc123) is DESTROYED.                           ||
||  "previous work" is LOST.                                             ||
||  Your colleague's changes from that commit: GONE.                     ||
||                                                                        ||
||  CORRECT: git commit -m "my changes"  (creates NEW commit)            ||
||                                                                        ||
||  ... --> [abc123] "previous work"  --> [ghi789] "my changes"          ||
||          (preserved!)                   (new, separate commit)         ||
||                                                                        ||
+========================================================================+
```

### Panel 4: The 7 Hooks in Execution Order

```
git commit triggers hooks in sequence:

  1. ruff           --> Style + lint (auto-fixes, re-stage needed)
  2. ruff-format    --> Code formatting (auto-fixes, re-stage needed)
  3. registry-      --> Method counts 11/8/5 match across 5 sources
     integrity
  4. registry-      --> pytest runs test_registry.py
     validation
  5. r-hardcoding-  --> No hex colors or ggsave() in .R files
     check
  6. computation-   --> No sklearn/scipy imports in src/viz/
     decoupling
  7. extraction-    --> No synthetic data in production paths
     isolation
  8. figure-        --> No synthetic data in figures/generated/
     isolation

  (renv-sync-check also exists but has known bypass: SKIP=renv-sync-check)

  FIRST failure stops the chain. Fix it, re-stage, try again.
```

## Spatial Anchors

```yaml
layout_flow: "Top-down: pass/fail fork, recovery flow, warning box, hook list"
spatial_anchors:
  fork:
    x: 0.5
    y: 0.15
    content: "git commit leads to pass (done) or fail (continue)"
  recovery:
    x: 0.5
    y: 0.4
    content: "Read error, fix, re-stage, NEW commit"
  warning:
    x: 0.5
    y: 0.7
    content: "Why --amend destroys previous commit"
  hook_list:
    x: 0.5
    y: 0.9
    content: "7 hooks in execution order"
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `.pre-commit-config.yaml` | All 7+ hooks with IDs, trigger patterns, and commands |
| `pyproject.toml` | ruff configuration (rules, line length, target version) |
| `configs/registry_canary.yaml` | Expected method counts (anti-cheat layer 1) |
| `configs/mlflow_registry/parameters/classification.yaml` | Registry definitions (anti-cheat layer 2) |

## Code Paths

| Module | Role |
|--------|------|
| `.pre-commit-config.yaml` | Hook definitions (id, entry, files patterns) |
| `scripts/verify_registry_integrity.py` | Registry integrity verification script |
| `scripts/check_computation_decoupling.py` | Scans `src/viz/` for banned imports |
| `scripts/check_r_hardcoding.py` | Scans `.R` files for hex colors and `ggsave()` |
| `scripts/check_extraction_isolation.py` | Verifies synthetic data stays isolated |
| `scripts/check_figure_isolation.py` | Verifies figures come from real data only |
| `src/data_io/registry.py` | `EXPECTED_OUTLIER_COUNT`, `EXPECTED_IMPUTATION_COUNT`, `EXPECTED_CLASSIFIER_COUNT` |

## Extension Guide

To add a new pre-commit hook:
1. Add hook definition to `.pre-commit-config.yaml` with unique `id`
2. Create enforcement script in `scripts/check_*.py`
3. Set `files:` pattern to limit scope (avoid running on every file)
4. Document the hook in this figure (Panel 4) and in `CONTRIBUTING.md`
5. Add bypass instructions if the hook has known false positives

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-95",
    "title": "When a Pre-commit Hook Fails: Fix, Stage, Commit (Never Amend)"
  },
  "content_architecture": {
    "primary_message": "When a pre-commit hook fails, the commit did NOT happen. Fix, re-stage, then create a NEW commit. Never --amend.",
    "layout_flow": "Top-down: pass/fail fork, recovery flow, critical warning, hook list",
    "spatial_anchors": {
      "fork": {"x": 0.5, "y": 0.15},
      "recovery": {"x": 0.5, "y": 0.4},
      "warning": {"x": 0.5, "y": 0.7},
      "hook_list": {"x": 0.5, "y": 0.9}
    },
    "key_structures": [
      {
        "name": "Pass/Fail Fork",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["ALL PASS -> Commit created", "ANY FAILS -> Commit NOT created"]
      },
      {
        "name": "Recovery Flow",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Read error", "Fix issue", "git add", "git commit (NEW)"]
      },
      {
        "name": "Amend Warning",
        "role": "abnormal_warning",
        "is_highlighted": true,
        "labels": ["--amend DESTROYS previous commit", "Previous work LOST"]
      },
      {
        "name": "Hook Execution Chain",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["ruff", "registry", "r-hardcoding", "computation-decoupling", "isolation"]
      }
    ],
    "callout_boxes": [
      {"heading": "CRITICAL", "body_text": "The failed commit DOES NOT EXIST. --amend modifies the PREVIOUS commit, destroying unrelated work."},
      {"heading": "CORRECT PATTERN", "body_text": "Fix -> git add <files> -> git commit -m 'message' (NEW commit, not amend)."}
    ]
  }
}
```

## Alt Text

Flowchart showing the correct recovery from a pre-commit hook failure: fix the issue, re-stage files, create a new commit. Warning box explains why --amend destroys the previous commit.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
