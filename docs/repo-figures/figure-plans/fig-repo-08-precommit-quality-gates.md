# fig-repo-08: Pre-commit Quality Gates

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-08 |
| **Title** | Pre-commit: Automatic Code Quality Checks |
| **Complexity Level** | L2 (Process overview) |
| **Target Persona** | Research Scientist, ML Engineer |
| **Location** | CONTRIBUTING.md, docs/ |
| **Priority** | P3 (Medium) |

## Purpose

Explain what pre-commit hooks do and why they exist, especially the hardcoding checks specific to this repository.

## Key Message

"Pre-commit runs automatic checks before every commit - like a spell-checker for code that catches common mistakes."

## Visual Concept

**Gate metaphor with pass/fail outcomes:**

```
┌─────────────────────────────────────────────────────────────────┐
│                     YOUR CODE CHANGES                           │
│                           │                                     │
│                           ▼                                     │
│              ┌─────────────────────────┐                       │
│              │    PRE-COMMIT HOOKS     │                       │
│              │    (Quality Gate)       │                       │
│              └─────────────────────────┘                       │
│                     │         │                                 │
│         ┌───────────┘         └───────────┐                    │
│         ▼                                 ▼                    │
│  ┌─────────────┐                   ┌─────────────┐            │
│  │ ✓ ruff     │                   │ ✗ Hardcoded │            │
│  │   (format)  │                   │   color     │            │
│  │ ✓ mypy     │                   │   #006BA2   │            │
│  │   (types)   │                   │             │            │
│  │ ✓ pytest   │                   │  BLOCKED!   │            │
│  │   (tests)   │                   │             │            │
│  │ ✓ no-hard  │                   │  Fix & retry│            │
│  │   code     │                   │             │            │
│  └─────────────┘                   └─────────────┘            │
│         │                                 │                    │
│         ▼                                 ▼                    │
│  ┌─────────────┐                   ┌─────────────┐            │
│  │  COMMIT     │                   │  REJECTED   │            │
│  │  ALLOWED    │                   │             │            │
│  └─────────────┘                   └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## Content Elements

### Required Elements
1. Gate/checkpoint metaphor
2. List of checks (ruff, mypy, pytest, hardcoding)
3. Pass path (green) vs fail path (red)
4. Specific example of failure (hardcoded color)

### Optional Elements
1. How to skip (--no-verify, discouraged)
2. How to run manually
3. Config file location (.pre-commit-config.yaml)

## Text Content

### Title Text
"Pre-commit: Quality Gate Before Every Commit"

### Labels/Annotations
- Gate: "Automatic checks run on git commit"
- Checks: "Format, types, tests, no-hardcoding"
- Pass: "All checks pass → commit allowed"
- Fail: "Any check fails → commit blocked"
- Example: "Hardcoded color #006BA2 → BLOCKED"

### Caption (for embedding)
Pre-commit hooks automatically run quality checks before every commit. If any check fails (like finding hardcoded colors), the commit is blocked until the issue is fixed.

## Prompts for Nano Banana Pro

### Style Prompt
Technical but friendly diagram. Gate/checkpoint visual metaphor. Green path for success, red for failure. Include a specific example of what gets caught. Terminal-style output for the checks list.

### Content Prompt
Create a quality gate diagram:
1. TOP: "Your Code Changes" flowing down
2. MIDDLE: A gate labeled "Pre-commit Hooks"
3. BRANCHES: Left side (green) shows passing checks, right side (red) shows blocked commit
4. Include specific example: hardcoded color #006BA2 causing a block

List the checks: ruff, mypy, pytest, no-hardcoding

### Refinement Notes
- The gate metaphor should be clear
- Show that fixing issues allows retry
- Include the specific hardcoding example from this repo

## Alt Text

Diagram showing pre-commit hooks as a quality gate that blocks commits with issues like hardcoded colors, while allowing commits that pass all checks.

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
