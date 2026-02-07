# fig-repo-35: Makefile Commands Overview

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-35 |
| **Title** | Makefile Commands Overview |
| **Complexity Level** | L1 (Quick reference) |
| **Target Persona** | All |
| **Location** | README.md, docs/getting-started/ |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Show the main make commands and what they do—the primary interface for running the pipeline.

## Key Message

"The Makefile provides simple commands for complex operations: `make reproduce` runs everything, `make analyze` generates figures, `make test` runs validation."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MAKEFILE COMMANDS OVERVIEW                                   │
│                    Your interface to the pipeline                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  WHY MAKE?                                                                      │
│  ═════════                                                                      │
│                                                                                 │
│  • One command instead of many                                                  │
│  • Same command works on any machine                                            │
│  • Dependencies handled automatically                                           │
│  • Well-established (50+ years of Unix)                                         │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  MOST COMMON COMMANDS                                                           │
│  ════════════════════                                                           │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │   🚀 make reproduce                                                     │   │
│  │   ════════════════                                                      │   │
│  │   Run the COMPLETE pipeline: extract → analyze → figures                │   │
│  │   Use when: Starting fresh or after new MLflow experiments              │   │
│  │                                                                         │   │
│  │   ┌──────────┐    ┌──────────┐    ┌──────────┐                          │   │
│  │   │  MLflow  │ →  │  DuckDB  │ →  │  Figures │                          │   │
│  │   │  pickles │    │  extract │    │  + JSON  │                          │   │
│  │   └──────────┘    └──────────┘    └──────────┘                          │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │   📊 make analyze                                                       │   │
│  │   ═══════════════                                                       │   │
│  │   Generate figures from existing DuckDB (most common!)                  │   │
│  │   Use when: Tweaking visualizations, no new experiments                 │   │
│  │                                                                         │   │
│  │   ┌──────────┐    ┌──────────┐                                          │   │
│  │   │  DuckDB  │ →  │  Figures │     (Skips extraction)                   │   │
│  │   │  (exists)│    │  + JSON  │                                          │   │
│  │   └──────────┘    └──────────┘                                          │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │   🧪 make test                                                          │   │
│  │   ════════════                                                          │   │
│  │   Run all tests including figure QA                                     │   │
│  │   Use when: Before committing, after any changes                        │   │
│  │                                                                         │   │
│  │   Includes: pytest, figure validation, registry checks                  │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  FULL COMMAND REFERENCE                                                         │
│  ══════════════════════                                                         │
│                                                                                 │
│  │ Command                  │ What it does                                    │ │
│  │ ──────────────────────── │ ───────────────────────────────────────────── │ │
│  │ make reproduce           │ Full pipeline (extract + analyze)              │ │
│  │ make reproduce-from-db   │ Figures from existing DuckDB                   │ │
│  │ make extract             │ MLflow → DuckDB only                           │ │
│  │ make analyze             │ DuckDB → figures only                          │ │
│  │ make test                │ All tests + QA                                 │ │
│  │ make test-figure-qa      │ Figure QA tests only                           │ │
│  │ make lint                │ Code style checks                              │ │
│  │ make clean               │ Remove generated files                         │ │
│  │ make help                │ Show all commands                              │ │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WORKFLOW EXAMPLES                                                              │
│  ═════════════════                                                              │
│                                                                                 │
│  📌 First time setup:                                                           │
│     uv sync                  # Install dependencies                             │
│     make reproduce           # Full pipeline                                    │
│                                                                                 │
│  📌 After editing figure code:                                                  │
│     make analyze             # Regenerate figures                               │
│     make test-figure-qa      # Validate figures                                 │
│                                                                                 │
│  📌 Before committing:                                                          │
│     make test                # Full test suite                                  │
│     make lint                # Style check                                      │
│                                                                                 │
│  📌 After running new experiments:                                              │
│     make reproduce           # Re-extract from MLflow                           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Why Make**: Benefits of using Makefile
2. **Top 3 commands**: reproduce, analyze, test with diagrams
3. **Full reference table**: All commands with descriptions
4. **Workflow examples**: Common scenarios and commands

## Text Content

### Title Text
"Makefile Commands: Your Pipeline Interface"

### Caption
The Makefile provides a consistent interface to complex pipeline operations. `make reproduce` runs the full pipeline (MLflow extraction → DuckDB → figures). `make analyze` is most commonly used—it regenerates figures from existing DuckDB without re-extraction. `make test` runs validation including figure QA. Run `make help` to see all available commands.

## Prompts for Nano Banana Pro

### Style Prompt
Command reference cards with icons. Flow diagrams showing what each command does. Reference table. Workflow examples in code blocks. Clean, command-line aesthetic.

### Content Prompt
Create a Makefile commands diagram:

**TOP - Why Make**:
- 4 bullet points

**MIDDLE - Top 3 Commands**:
- Three cards with icons: reproduce, analyze, test
- Mini flow diagrams showing pipeline steps

**BOTTOM LEFT - Reference Table**:
- Full command list with descriptions

**BOTTOM RIGHT - Workflow Examples**:
- 4 common scenarios with commands

## Alt Text

Makefile commands overview. Three main commands: make reproduce (full pipeline: MLflow → DuckDB → figures), make analyze (figures from existing DuckDB, most common), make test (all tests including figure QA). Full reference table lists 9 commands: reproduce, reproduce-from-db, extract, analyze, test, test-figure-qa, lint, clean, help. Workflow examples for first-time setup, editing figures, committing, and new experiments.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in README.md
