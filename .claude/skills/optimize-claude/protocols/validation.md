# Validation Protocol

## Post-Edit Validation (MANDATORY after every edit)

### Step 1: Critical Rules Check

For each rule in `reference/critical-rules-checklist.yaml`:

1. Grep all always-loaded files for each keyword
2. If ANY rule has 0 keyword matches across ALL files → **FAIL**
3. If FAIL → revert to last git commit, stop iteration

```bash
# Example check for rule R01 (Research question)
grep -rl "preprocessing" CLAUDE.md .claude/CLAUDE.md .claude/rules/*.md
grep -rl "NOT comparing classifiers" CLAUDE.md .claude/CLAUDE.md .claude/rules/*.md
grep -rl "CatBoost fixed" CLAUDE.md .claude/CLAUDE.md .claude/rules/*.md
```

All 3 keywords must appear in at least 1 always-loaded file.

### Step 2: Char Count Verification

```bash
wc -c CLAUDE.md
wc -c .claude/CLAUDE.md
wc -c .claude/rules/*.md
wc -c .claude/GUARDRAILS.md
```

Sum all always-loaded files. Compare against targets.

### Step 3: Cross-Reference Integrity

For each cross-reference in the edited files:
- Verify the referenced file exists
- Verify the referenced section/content exists in that file
- No broken references allowed

### Step 4: Redundancy Check

For each critical rule, count how many files contain it:
- 0 files → **CRITICAL FAILURE** (rule lost)
- 1 file → **OK** (canonical)
- 2 files → **ACCEPTABLE** (brief mention + canonical)
- 3+ files → **STILL REDUNDANT** (needs more dedup)

Target: no rule in 3+ files.

## Validation Report Format

```
VALIDATION REPORT - Iteration N
================================
Critical Rules: 14/14 PASS
Char Counts:
  CLAUDE.md: 19,500 (target: <20,000) ✅
  .claude/CLAUDE.md: 9,800 (target: <10,000) ✅
  Total always-loaded: 34,500 (target: <35,000) ✅
Cross-references: 12/12 valid ✅
Max redundancy: 2 files (target: <3) ✅

STATUS: PASS
```
