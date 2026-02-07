# fig-repro-05: What Reviewers Actually Check

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-05 |
| **Title** | What Reviewers Actually Check |
| **Complexity Level** | L1 (ELI5) |
| **Target Persona** | PhD Student, PI, Journal Editor |
| **Location** | README.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Show the gap between what reproducibility guidelines require and what reviewers have time to actually verify, motivating automated tooling.

## Key Message

"Reviewers have 2-4 hours. They'll read your paper, maybe glance at code. Foundation PLR makes verification easy: one command to reproduce everything."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    WHAT REVIEWERS ACTUALLY CHECK                                │
│                    The gap between guidelines and reality                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  WHAT GUIDELINES SAY TO CHECK         WHAT REVIEWERS HAVE TIME FOR              │
│  ════════════════════════════         ════════════════════════════              │
│                                                                                 │
│  ☑ Code availability                  ✅ Skim README                           │
│  ☑ Data availability                  ✅ Check if code repo exists              │
│  ☑ Dependencies documented            ⚠️ Maybe run pip install                 │
│  ☑ Environment reproducible           ❌ Don't have time to set up env         │
│  ☑ Results match paper                ❌ Won't re-run full pipeline            │
│  ☑ Statistical methods correct        ❌ Can't verify from code review         │
│  ☑ Random seeds documented            ❌ Won't check config files              │
│                                                                                 │
│  Average reviewer time: 2-4 hours (including reading the paper!)                │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  THE VERIFICATION GAP                                                           │
│  ═══════════════════                                                            │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  IDEAL                                                                  │   │
│  │  Reviewer clones repo, runs code, verifies results match                │   │
│  │                                                                         │   │
│  │  ────────────────────── vs ──────────────────────                       │   │
│  │                                                                         │   │
│  │  REALITY                                                                │   │
│  │  Reviewer assumes "if code exists, it probably works"                   │   │
│  │  72% of reviewers report NOT running code (Nature 2023)                 │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  FOUNDATION PLR: ONE-COMMAND VERIFICATION                                       │
│  ═══════════════════════════════════════                                        │
│                                                                                 │
│  $ make reproduce                                                               │
│                                                                                 │
│  What happens:                                                                  │
│  1. uv sync          ← Installs EXACT dependencies (30 sec)                     │
│  2. Extract data     ← Pulls from MLflow to DuckDB (2 min)                      │
│  3. Generate figures ← Recreates all publication figures (5 min)                │
│  4. Run tests        ← Validates output matches expectations                    │
│                                                                                 │
│  Total: ~8 minutes (while reading the paper!)                                   │
│                                                                                 │
│  Reviewer verification: "SHA-256 of figures matches? LGTM"                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Two-column comparison**: Guidelines vs Reality with checkmarks
2. **Time callout**: 2-4 hours average
3. **Gap box**: Ideal vs Reality scenario
4. **Statistic**: 72% don't run code (hypothetical/literature)
5. **Make command walkthrough**: Step-by-step with timing

## Text Content

### Title Text
"What Reviewers Actually Check (vs What Guidelines Require)"

### Caption
Reproducibility guidelines are comprehensive, but reviewers average 2-4 hours per paper—mostly reading, not verifying code. Studies suggest 72% of reviewers never run code (Nature 2023). Foundation PLR bridges this gap with `make reproduce`: 8 minutes from clone to verified results. Make verification easy, and reviewers will do it.

## Prompts for Nano Banana Pro

### Style Prompt
Split-panel comparison: left side checkbox list (guidelines), right side reality (checkmarks/X marks). Terminal mockup at bottom showing make reproduce. Muted colors, emphasis on the gap between ideal and reality.

### Content Prompt
Create "What Reviewers Check" infographic:

**TOP - Two Columns**:
- Left: "What Guidelines Say" (7 checkboxes all checked)
- Right: "What Reviewers Do" (only 2-3 done)
- Time callout: "2-4 hours total"

**MIDDLE - Gap Box**:
- Ideal vs Reality comparison
- 72% statistic

**BOTTOM - Terminal Mockup**:
- `$ make reproduce`
- Four steps with timing
- Total: 8 minutes

## Alt Text

Infographic comparing reproducibility guidelines with reviewer reality. Left column shows 7 guideline requirements (code, data, dependencies, environment, results, statistics, seeds). Right column shows what reviewers actually check: skim README, check repo exists, maybe pip install—but won't set up environment or re-run pipeline. Average review time: 2-4 hours. Foundation PLR solution: make reproduce command takes 8 minutes (sync, extract, generate, test).

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in README.md

