# /optimize-claude - CLAUDE.md Hierarchy Optimization

Optimize the CLAUDE.md instruction hierarchy to reduce token bloat while preserving all critical rules.

## Usage

- `/optimize-claude` - Run full optimization (discovery + compression + validation)
- `/optimize-claude check` - Measure current char counts and redundancy only

## What This Does

Uses the Ralph Wiggum iterated self-correction pattern with 2 serial agents:

1. **Agent 1 (Compression)**: Deduplicates, compresses prose to tables, moves domain-specific content out of always-loaded files
2. **Agent 2 (Clarity)**: Reviews compression for LLM instruction effectiveness, restructures for front-loaded critical rules

Each iteration requires human approval before edits are applied. Git checkpoints enable rollback.

## Targets

| Metric | Target |
|--------|--------|
| Root CLAUDE.md | < 20,000 chars |
| .claude/CLAUDE.md | < 10,000 chars |
| Total always-loaded | < 35,000 chars |
| Critical rules preserved | 100% (14 rules, binary check) |

## Activation

Before optimizing, run the ACTIVATION-CHECKLIST:
1. Discover all CLAUDE.md files
2. Measure baseline char counts
3. Snapshot critical rules
4. Git tag baseline

## Protocol

```
Iteration N (max 3):
  Agent 1: Compress & deduplicate → structured proposals
  Agent 2: Review clarity & architecture → revised proposals
  Main: Synthesize, show diff, get human approval
  Execute: Apply edits, git commit
  Validate: Check all 14 critical rules
  Converge: Targets met? Plateau? Continue?
```

## Files

- `ACTIVATION-CHECKLIST.md` - Pre-execution discovery
- `protocols/agent-1-compression.md` - Compression agent prompt
- `protocols/agent-2-clarity.md` - Clarity agent prompt
- `protocols/convergence.md` - When to stop
- `protocols/validation.md` - Post-edit checks
- `reference/hierarchy-levels.md` - Content placement guide
- `reference/critical-rules-checklist.yaml` - Must-preserve rules
- `reference/llm-optimization-heuristics.md` - Writing effective LLM instructions
