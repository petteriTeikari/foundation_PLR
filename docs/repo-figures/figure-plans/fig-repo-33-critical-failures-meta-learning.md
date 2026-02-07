# fig-repo-33: Critical Failures: Meta-Learning Documentation

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-33 |
| **Title** | Critical Failures: Meta-Learning Documentation |
| **Complexity Level** | L2 (Process documentation) |
| **Target Persona** | Software Engineer, ML Engineer |
| **Location** | .claude/docs/, CONTRIBUTING.md |
| **Priority** | P2 |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain the meta-learning failure documentation system—how critical failures are captured to prevent recurrence.

## Key Message

"Every critical failure becomes a documented meta-learning with root cause, impact, and prevention rules. These rules are enforced in CLAUDE.md and CI."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    CRITICAL FAILURES: META-LEARNING SYSTEM                      │
│                    Learning from Mistakes to Prevent Recurrence                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  THE PROBLEM                                                                    │
│  ═══════════                                                                    │
│                                                                                 │
│  Without systematic documentation:                                              │
│  • Same mistakes happen again (context loss)                                    │
│  • Root causes aren't addressed (whack-a-mole fixes)                            │
│  • Knowledge stays in one person's head                                         │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  THE SOLUTION: META-LEARNING WORKFLOW                                           │
│  ════════════════════════════════════                                           │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  1. INCIDENT          2. ANALYSIS         3. DOCUMENTATION              │   │
│  │  ══════════           ══════════          ═══════════════              │   │
│  │                                                                         │   │
│  │  Critical bug    ──▶  Root cause     ──▶  meta-learning/                │   │
│  │  discovered           identified          CRITICAL-FAILURE-NNN.md       │   │
│  │                                                                         │   │
│  │       │                    │                        │                   │   │
│  │       ▼                    ▼                        ▼                   │   │
│  │                                                                         │   │
│  │  4. RULES             5. ENFORCEMENT       6. PREVENTION                │   │
│  │  ════════             ═══════════════      ══════════════               │   │
│  │                                                                         │   │
│  │  CLAUDE.md       ──▶  Pre-commit     ──▶  Bug never                     │   │
│  │  rules added          hooks + CI          happens again                 │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  DOCUMENTED CRITICAL FAILURES                                                   │
│  ════════════════════════════                                                   │
│                                                                                 │
│  │ ID              │ Title                          │ Root Cause              │ │
│  │ ─────────────── │ ────────────────────────────── │ ─────────────────────── │ │
│  │ CRITICAL-001    │ Synthetic data in figures      │ Fixed random seed made  │ │
│  │                 │                                │ models look identical   │ │
│  │ ─────────────── │ ────────────────────────────── │ ─────────────────────── │ │
│  │ CRITICAL-002    │ Hardcoding despite systems     │ Claude ignored config   │ │
│  │                 │                                │ infrastructure          │ │
│  │ ─────────────── │ ────────────────────────────── │ ─────────────────────── │ │
│  │ CRITICAL-003    │ Computation in viz code        │ Metrics computed in R   │ │
│  │                 │                                │ instead of extraction   │ │
│  │ ─────────────── │ ────────────────────────────── │ ─────────────────────── │ │
│  │ CRITICAL-004    │ R figure hardcoding            │ Hex colors in R scripts │ │
│  │                 │                                │ instead of YAML refs    │ │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  META-LEARNING FILE STRUCTURE                                                   │
│  ════════════════════════════                                                   │
│                                                                                 │
│  .claude/docs/meta-learnings/                                                   │
│  ├── CRITICAL-FAILURE-001-synthetic-data-in-figures.md                          │
│  ├── CRITICAL-FAILURE-002-hardcoding-despite-existing-systems.md                │
│  ├── CRITICAL-FAILURE-003-computation-decoupling-violation.md                   │
│  ├── CRITICAL-FAILURE-004-r-figure-hardcoding.md                                │
│  └── VIOLATION-001-regex-for-code-parsing.md                                    │
│                                                                                 │
│  Each file contains:                                                            │
│  • Incident description (what happened)                                         │
│  • Root cause analysis (why it happened)                                        │
│  • Impact assessment (what could have gone wrong)                               │
│  • Prevention rules (how to stop recurrence)                                    │
│  • Enforcement mechanism (hooks, CI, CLAUDE.md rules)                           │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ENFORCEMENT LAYERS                                                             │
│  ════════════════                                                               │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                           │ │
│  │  CLAUDE.md Rules       Pre-commit Hooks        CI Pipeline                │ │
│  │  ───────────────       ────────────────        ───────────                │ │
│  │                                                                           │ │
│  │  "NEVER use           check_r_hardcoding.py   pytest test_figure_qa/     │ │
│  │   synthetic data"      ↓                       ↓                          │ │
│  │                        Blocks commit if        Fails PR if                │ │
│  │  "Compute in           hex color found        synthetic data detected     │ │
│  │   extraction ONLY"                                                        │ │
│  │                                                                           │ │
│  │  "Use registry,                                                           │ │
│  │   not parsing"                                                            │ │
│  │                                                                           │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  Multiple layers catch violations at different stages.                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Problem statement**: Why documentation matters
2. **6-step workflow**: Incident → Analysis → Docs → Rules → Enforcement → Prevention
3. **Failure table**: List of documented critical failures
4. **File structure**: Directory listing with contents explanation
5. **Enforcement layers**: CLAUDE.md + hooks + CI

## Text Content

### Title Text
"Critical Failures: Learning from Mistakes Systematically"

### Caption
The meta-learning system documents every critical failure with root cause analysis and prevention rules. When CRITICAL-001 (synthetic data in figures) was discovered, it became a documented meta-learning with enforcement via Figure QA tests. When CRITICAL-004 (R hardcoding) occurred, pre-commit hooks were added. Multiple enforcement layers (CLAUDE.md rules, pre-commit hooks, CI tests) prevent recurrence at different stages.

## Prompts for Nano Banana Pro

### Style Prompt
Process workflow diagram with 6 steps. Table showing documented failures. Directory tree showing meta-learning files. Three-layer enforcement diagram. Clean, technical documentation aesthetic.

### Content Prompt
Create a meta-learning system diagram:

**TOP - Workflow**:
- 6 connected boxes: Incident → Analysis → Documentation → Rules → Enforcement → Prevention

**MIDDLE - Failure Table**:
- 4 rows: ID, Title, Root Cause for each CRITICAL failure

**BOTTOM LEFT - File Structure**:
- Directory tree: .claude/docs/meta-learnings/CRITICAL-*.md

**BOTTOM RIGHT - Enforcement**:
- Three columns: CLAUDE.md rules, Pre-commit hooks, CI tests

## Alt Text

Meta-learning system diagram. Workflow: incident discovered → root cause analysis → document in meta-learning markdown → add rules to CLAUDE.md → enforce via hooks and CI → prevent recurrence. Table shows 4 critical failures: synthetic data, hardcoding, computation in viz, R hardcoding. Files stored in .claude/docs/meta-learnings/. Enforcement at three layers: CLAUDE.md rules, pre-commit hooks, CI pipeline tests.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in CONTRIBUTING.md
