# Activation Checklist

Run these steps BEFORE any optimization iteration.

## Step 1: Discover All CLAUDE.md Files

```bash
find /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR \
  -name "CLAUDE.md" -o -name "GUARDRAILS.md" -o -name "auto-context.yaml" \
  | sort
```

Also check `.claude/rules/*.md` for always-loaded rule files.

## Step 2: Measure Baseline Char Counts

For each discovered file, record:

```bash
wc -c <file>
```

Create baseline state in `state/optimization-state.json`:

```json
{
  "iteration": 0,
  "baseline": {
    "CLAUDE.md": { "chars": 0, "path": "CLAUDE.md" },
    ".claude/CLAUDE.md": { "chars": 0, "path": ".claude/CLAUDE.md" },
    ".claude/rules/00-research-question.md": { "chars": 0 },
    ".claude/rules/05-registry-source-of-truth.md": { "chars": 0 },
    ".claude/rules/10-figures.md": { "chars": 0 },
    ".claude/GUARDRAILS.md": { "chars": 0 }
  },
  "always_loaded_total": 0,
  "history": []
}
```

## Step 3: Snapshot Critical Rules

For each rule in `reference/critical-rules-checklist.yaml`:
- Grep all always-loaded files for each keyword
- Record which files contain each rule
- Flag rules that appear in 3+ files (redundancy)
- Flag rules that appear in 0 files (MISSING - critical failure)

## Step 4: Git Checkpoint

```bash
git tag optimize-claude-baseline
```

## Step 5: Identify Redundancy

Compare content across files. For each concept, identify:
- **Canonical location** (where it SHOULD live)
- **Duplicate locations** (where it's repeated)
- **Compression opportunity** (verbose prose → table/imperative)

Output: structured list of redundancies with proposed canonical locations.

## Completion

Once all steps are done, proceed to the first iteration using:
1. `protocols/agent-1-compression.md`
2. `protocols/agent-2-clarity.md`
