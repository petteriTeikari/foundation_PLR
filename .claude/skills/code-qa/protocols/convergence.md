# Convergence Protocol

## When to Stop Iterating

### Success (CONVERGED)

- 0 CRITICAL issues remaining
- 0 HIGH issues remaining
- All approved fixes applied and tests pass

### Plateau (STOP)

- New iteration finds < 3 new issues beyond previous iteration
- 2 consecutive iterations with no CRITICAL/HIGH findings

### Safety Stop

- Tests fail after applying fixes → REVERT last batch, STOP
- Human rejects a full category → skip that category, continue others
- > 100 issues found in single scan → report only, don't propose fixes (needs architectural review)

## Max Iterations

**3 iterations maximum.** Typically:
- Iteration 1: Find bulk issues (hardcoding, dead code)
- Iteration 2: Find issues exposed by iteration 1 fixes (new dead code after consolidation)
- Iteration 3: Final sweep, typically finds < 5 issues

## Per-Iteration Metrics

```json
{
  "iteration": 1,
  "issues_found": { "CRITICAL": 5, "HIGH": 12, "MEDIUM": 20, "LOW": 8 },
  "fixes_proposed": 37,
  "fixes_approved": 33,
  "fixes_applied": 33,
  "tests_pass": true,
  "files_modified": 15,
  "lines_removed": 230,
  "lines_added": 45,
  "status": "continue"
}
```

## Git Checkpoint Strategy

After each approved category:
```bash
git add <modified files>
git commit -m "refactor(code-qa): <category> fixes - iteration N"
```

Categories committed separately for easy revert:
1. `refactor(code-qa): Remove hardcoded values`
2. `refactor(code-qa): Remove dead code`
3. `refactor(code-qa): Consolidate duplicates`
4. `refactor(code-qa): Clean up dead config`
