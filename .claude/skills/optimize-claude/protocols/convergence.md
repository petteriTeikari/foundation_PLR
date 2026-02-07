# Convergence Protocol

## When to Stop Iterating

### Success (CONVERGED)

All targets met:
- Root CLAUDE.md < 20,000 chars
- .claude/CLAUDE.md < 10,000 chars
- Total always-loaded < 35,000 chars
- All 14 critical rules present (binary check)

### Plateau (STOP - diminishing returns)

- Char reduction < 5% between consecutive iterations
- 2 consecutive plateaus → stop

### Safety Stop (REVERT + STOP)

- Any critical rule missing after validation → revert to last good state
- User rejects diff → keep previous iteration's state

## Max Iterations

**3 iterations maximum.** If not converged after 3, stop and report remaining gaps.

## Decision Tree

```
After each iteration:
  ├── All targets met? → DONE
  ├── Critical rule missing? → REVERT to last commit, STOP
  ├── User rejected? → STOP, keep current state
  ├── Delta < 5%? → INCREMENT plateau counter
  │   ├── 2 plateaus? → STOP
  │   └── < 2 plateaus? → CONTINUE
  └── Iteration < 3? → CONTINUE
```

## Metrics to Track Per Iteration

```json
{
  "iteration": 1,
  "chars": {
    "CLAUDE.md": { "before": 42191, "after": 19500 },
    ".claude/CLAUDE.md": { "before": 21537, "after": 9800 },
    "total_always_loaded": { "before": 70089, "after": 34500 }
  },
  "delta_pct": 51,
  "critical_rules_check": "14/14 PASS",
  "status": "continue"
}
```
