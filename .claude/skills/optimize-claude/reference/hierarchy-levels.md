# CLAUDE.md Hierarchy Levels

## Content Placement Guide

| Level | File(s) | Purpose | Tone | Size Target |
|-------|---------|---------|------|-------------|
| **L1: Root** | `CLAUDE.md` | WHAT the project is, WHY rules exist | Overview + critical don'ts | < 20K chars |
| **L2: Behavior** | `.claude/CLAUDE.md` | HOW to behave, patterns to follow | Imperative contract | < 10K chars |
| **L3: Rules** | `.claude/rules/*.md` | MUST-follow rules (1 topic per file) | Binary MUST/NEVER | < 2K each |
| **L4: Guardrails** | `.claude/GUARDRAILS.md` | Quick-reference cheatsheet | Minimal, scannable | < 1K chars |
| **L5: Domains** | `.claude/domains/*.md` | WHEN-NEEDED context | Reference material | Any size |
| **L6: Auto-context** | `.claude/auto-context.yaml` | File-pattern-triggered rules | Terse imperatives | < 3K chars |

## What Goes Where

### L1 Root CLAUDE.md (Always loaded, most visible)

- Research question (1 paragraph)
- Data provenance table (compact)
- Pipeline diagram (ASCII, no prose)
- Critical rules summary table (1-line per rule, link to details)
- Subject counts table
- Key findings table
- Sister repositories paths

**NOT here**: Code examples, detailed "why" explanations, full metric tables

### L2 .claude/CLAUDE.md (Always loaded, behavior contract)

- Quick reference table (rule → enforcement)
- Anti-hardcoding patterns (CORRECT vs WRONG, 1 example each)
- Computation decoupling architecture
- Directory structure rules
- Figure generation workflow
- Context awareness strategy
- Combo enforcement (reference YAML, don't list all combos)

**NOT here**: Research background, data provenance, STRATOS theory

### L3 .claude/rules/ (Always loaded, one topic per file)

Each file: 1 rule, imperative, < 2K chars:
- `00-research-question.md` - What we study
- `05-registry-source-of-truth.md` - Method validation
- `10-figures.md` - Figure requirements
- `15-stratos-metrics.md` - STRATOS compliance
- `20-package-management.md` - uv/conda rules
- `25-no-reimplementation.md` - Use verified code

### L4 .claude/GUARDRAILS.md (Always loaded, cheatsheet)

5-7 rules, each 1-2 lines. No examples, no rationale. Just the rule.

### L5 .claude/domains/ (Loaded on demand)

Detailed reference for specific work:
- `visualization.md` - Plot styling, D3 patterns
- `mlflow-experiments.md` - Run IDs, experiment structure
- `stratos-detailed.md` - Full metric tables, JSON schemas
- `manuscript.md` - Paper structure, LaTeX conventions

### L6 .claude/auto-context.yaml (Triggered by file patterns)

Terse imperatives triggered by file glob patterns. Max 3 lines per injection.
