# FAILURE REPORT #007: Ruff Version Skew — CI Lint Fails Despite Clean Pre-commit

**Severity**: HIGH — CI pipeline broken on first deployment
**Date**: 2026-02-07
**Discovered by**: PR #42 CI run (GitHub Actions run 21778416887)
**Root cause**: Pre-commit pins ruff v0.9.1; CI runs `uvx ruff` which pulls v0.14.13 (latest). 5+ months of ruff releases = different formatting rules.

---

## Summary

106 out of 557 Python files failed `ruff format --check` on CI despite every file passing local pre-commit hooks before push. The developer (correctly) said: "I have no idea how would this ever fail when pushing from a machine with precommit installed."

The answer: **pre-commit and CI were running different ruff versions**.

## What Happened

### The Configuration

| System | Ruff Version | Source |
|--------|-------------|--------|
| Pre-commit (local) | `v0.9.1` | `.pre-commit-config.yaml` line 5: `rev: v0.9.1` |
| CI lint job (remote) | `v0.14.13` | `ci.yml` line 78: `uvx ruff format --check` (unpinned = latest) |

### The Version Gap

- `v0.9.1` was released ~August 2025
- `v0.14.13` is the latest as of February 2026
- 5+ months of ruff development = **new formatting rules, changed defaults, new style decisions**
- Example: ruff v0.10+ changed trailing comma handling, string quote normalization, and multi-line expression formatting

### The Failure Cascade

1. Developer writes code
2. Pre-commit runs `ruff-format` (v0.9.1) → formats to v0.9.1 style → **PASS**
3. `git commit` succeeds (pre-commit passed)
4. `git push` triggers CI
5. CI runs `uvx ruff format --check` (v0.14.13) → sees files not in v0.14.13 style → **FAIL**
6. 106 files reported as "would be reformatted"

### Why Cherry-Pick Made It Worse

During the 26-PR decomposition, the `execute_pr.sh` script ran `uvx ruff format $PY_FILES` (v0.14.13) on each PR's files before commit. Then pre-commit ran `ruff-format` (v0.9.1) and reformatted them back. The last formatter to run wins — and that was the pre-commit hook (v0.9.1). So all committed files have v0.9.1 formatting, but CI expects v0.14.13 formatting.

## The Fundamental Error

### Error A: Unpinned Tool Version in CI

```yaml
# ci.yml — THE BUG
- run: uvx ruff format --check src/ tests/ scripts/
#       ^^^ uvx pulls LATEST version, not the project's pinned version
```

`uvx` is pip-like: it installs the latest version every time. This creates a **moving target** — CI results change even when code doesn't.

### Error B: Version Pin Only in Pre-commit

The project pinned ruff in `.pre-commit-config.yaml` but forgot to propagate that pin to CI. These are two independent systems with no shared version source.

### Error C: No Version Consistency Check

No test or CI step verifies that pre-commit and CI use the same tool versions. The version-consistency pattern from CRITICAL-FAILURE-002 (R version mismatch) was not applied to Python tooling.

## Why This Is Wrong

### From This Project's Own Meta-learnings

CRITICAL-FAILURE-002 established: **"Version numbers are CONSTRAINTS, not suggestions — match exactly or fail."** That lesson was about R versions in Docker. The same principle applies to Python tooling versions in CI.

### The Reproducibility Argument

If CI uses a different formatter version than developers:
1. Developers can't reproduce CI failures locally
2. "Fix lint" commits change formatting, not logic — noise in git history
3. CI becomes non-deterministic (upgrades to ruff can break CI at any time)

## What Should Have Been Done

### Option 1 (CORRECT): Pin ruff version in CI to match pre-commit

```yaml
# ci.yml — FIXED
- run: uvx ruff@0.9.1 check src/ tests/ scripts/
- run: uvx ruff@0.9.1 format --check src/ tests/ scripts/
```

### Option 2 (ALSO CORRECT): Update both to the same latest version

```yaml
# .pre-commit-config.yaml
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.14.13  # Updated to match CI

# ci.yml
- run: uvx ruff@0.14.13 format --check src/ tests/ scripts/
```

### Option 3 (BEST): Single source of truth for ruff version

Define the version once (e.g., in `pyproject.toml` `[tool.ruff]` or a `RUFF_VERSION` env var) and reference it from both pre-commit and CI. This is the DRY principle applied to tool versions.

## Resolution

1. Updated both pre-commit and CI to use the same ruff version
2. Ran `uvx ruff format src/ tests/ scripts/` to normalize all files
3. Added this failure report

## Prevention

### Self-Check Question (Add to Workflow)

Before creating ANY CI step that runs a tool:
1. "Is this tool also in `.pre-commit-config.yaml`?" → **PIN THE SAME VERSION**
2. "Am I using `uvx <tool>` without a version pin?" → **ADD `@version`**
3. "Will this CI step produce different results tomorrow?" → **PIN IT**

### General Rule

**Every tool version in CI must trace to a pinned source.** Acceptable sources:
- `pyproject.toml` (e.g., `[tool.ruff]`)
- `.pre-commit-config.yaml` (e.g., `rev: v0.9.1`)
- Lockfiles (e.g., `uv.lock`)

Unacceptable: `uvx <tool>` (unpinned), `pip install <tool>` (unpinned), `npx <tool>` (unpinned).

---

**This failure demonstrates a class of CI bug: "works on my machine" caused by version skew between local and remote tooling. The fix is simple — pin the same version everywhere — but the bug is invisible until first CI deployment.**
