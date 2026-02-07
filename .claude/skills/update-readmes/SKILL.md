# /update-readmes - Repository Documentation Auto-Update

Update all README.md files, audit docstrings, and plan Nano Banana Pro figures. Uses the Ralph Wiggum iterated self-correction pattern.

## Usage

- `/update-readmes` - Full run (discovery + generation + validation)
- `/update-readmes report` - Discovery only, no edits (dry-run)
- `/update-readmes scope=readmes` - Only update README.md files
- `/update-readmes scope=docstrings` - Only audit/fix docstrings
- `/update-readmes scope=figures` - Only plan new Nano Banana Pro figures

## What This Does

### 1. README Auto-Update

Scans all project-level `README.md` files (excluding `node_modules/`, `.venv/`, `renv/`, `.pytest_cache/`, `archived/`) and detects:

- **Stale references**: Links to renamed/deleted files
- **Outdated counts**: Method counts, config counts, table rows that don't match actual YAML contents
- **Missing modules**: New source files or directories without README entries
- **Broken cross-references**: Internal `[text](path)` links that don't resolve
- **Missing figure thumbnails**: Inline image refs pointing to nonexistent assets
- **Config table drift**: README tables that don't match actual YAML file contents

Target directories (47+ READMEs):
- `README.md` (root)
- `src/` and all subdirectories
- `configs/` and all subdirectories
- `docs/`, `tests/`, `scripts/`, `apps/`, `figures/`, `data/`, `outputs/`, `notebooks/`
- `.claude/`

### 2. Docstring Audit

Scans Python modules for missing/stale NumPy-style docstrings:

| Priority | Directories | Reason |
|----------|-------------|--------|
| P0 | `src/data_io/`, `src/viz/`, `src/stats/` | Public API surface |
| P1 | `src/orchestration/`, `src/extraction/` | Pipeline-critical |
| P2 | `src/classification/`, `src/featurization/`, `src/imputation/` | Domain code |
| P3 | `src/utils/`, `src/log_helpers/`, `src/tools/` | Support code |

Checks:
- Missing docstrings on public functions/classes
- Signature drift (params in code but not in docstring, or vice versa)
- Missing return type documentation
- Missing `Raises` section for functions that raise exceptions
- Docstrings that reference removed parameters

### 3. Nano Banana Pro Figure Plans

Identifies gaps in documentation figure coverage:

- Scan `docs/repo-figures/plans-TODO/` for pending plans
- Cross-reference `docs/repo-figures/assets/` for generated images
- Identify new architectural features that lack visual documentation
- Follow progressive disclosure pattern (Level 1 visual -> Level 2 README -> Level 3 YAML comments)
- **Code architecture ONLY** - never results/performance (per `docs/repo-figures/CLAUDE.md`)

## Architecture (Ralph Wiggum Pattern)

```
Iteration N (max 3):
  Agent 1: Discovery (read-only scan)
    - Find all READMEs, Python modules, figure plans
    - Detect stale content, missing docstrings, figure gaps
    - Cross-reference internal links
    - Output: structured issue list with file paths and severity

  Agent 2: Generation (create/update content)
    - Update stale README sections
    - Generate config tables from actual YAML contents
    - Add missing docstrings (NumPy style)
    - Create figure plan files for identified gaps
    - Output: list of files modified/created

  Human Approval: Show categorized diff, get sign-off
  Git Checkpoint: Commit after each approved iteration
  Validation: All cross-references resolve, no broken links
  Converge: No new issues found? Stop.
```

## Scope Tiers

| Tier | Directories | READMEs | Use When |
|------|-------------|---------|----------|
| **readmes** | All README.md files | ~47 | Quick doc refresh |
| **docstrings** | `src/` Python modules | ~200 files | Pre-MkDocs prep |
| **figures** | `docs/repo-figures/` | Plans only | New feature documented |
| **full** (default) | All of the above | Everything | Publication prep |

## Pre-Flight Checks

Before running, verify:
1. `pytest tests/` passes (don't document broken code)
2. Git working tree is clean (for clean checkpoints)
3. `configs/` YAML files parse without errors

## Safety Constraints

- NEVER modify source code (only README.md, docstrings, figure plans)
- NEVER add docstrings to test files
- NEVER create figure plans for results/performance (code architecture ONLY)
- NEVER duplicate content between README levels (single source of truth)
- NEVER remove existing accurate content
- Docstring additions must match actual function signatures exactly
- Config tables generated from actual YAML parsing, never manual counts

## Convergence Criteria

Stop iterating when:
- All internal cross-references resolve
- All config README tables match actual YAML contents
- P0 + P1 docstring coverage >= 90%
- No stale references detected
- All plans-TODO have corresponding figure plans

## Relationship to GH#4 (MkDocs)

This skill generates CONTENT. GH#4 handles DEPLOYMENT:
- Updated READMEs -> MkDocs nav reflects reality
- Complete docstrings -> mkdocstrings extracts API reference
- Figure plans -> docs site has visual assets

## Files

- `protocols/agent-1-discovery.md` - Read-only scanning agent
- `protocols/agent-2-generation.md` - Content generation agent
- `reference/readme-template.md` - Standard README structure
- `reference/docstring-standards.md` - NumPy docstring format
- `reference/figure-plan-checklist.yaml` - Figure plan completeness checks
- `state/update-state.json` - Progress tracking between iterations
