# Progressive Disclosure Documentation System

This directory contains hierarchical documentation for AI assistants (Claude, etc.)
working on the Foundation PLR project.

## QUICK START FOR CLAUDE

1. **Rules** (always loaded): `.claude/rules/` - research question, figure rules
2. **Skills**: Use `/figures`, `/validate`, `/manuscript` for common tasks
3. **Planning docs**: `docs/planning/` - ACTION_PLAN.xml, external AI feedback

## Quick Navigation

### Level 1: Overview (Start Here)
- **`/CLAUDE.md`** (root) - Research question, key findings, critical rules

### Level 2: Behavior Contract
- **`.claude/CLAUDE.md`** - Specific rules for Claude Code behavior
  - Combo validation rules
  - Method name verification
  - Figure generation rules

### Level 3: Domain-Specific Context
Load these when working on specific areas:

| Domain | File | Load When |
|--------|------|-----------|
| Figures | `configs/VISUALIZATION/figure_registry.yaml` | Generating any figure |
| Combos | `configs/VISUALIZATION/plot_hyperparam_combos.yaml` | Using hyperparam combos |
| Methods | `.claude/domains/mlflow-experiments.md` | Verifying method names |
| Visualization | `.claude/domains/visualization.md` | Styling figures |
| Manuscript | `.claude/domains/manuscript.md` | Writing/editing paper |

### Level 4: External AI Context
- **`.claude/CONTEXT_FOR_EXTERNAL_AI.md`** - Complete context for Gemini/OpenAI
  - Copy this when seeking second opinions
  - Includes data infrastructure, config system, current problems

## How to Use This System

### For Claude Code Sessions

1. **Start**: Claude reads `/CLAUDE.md` automatically
2. **Task-specific**: Load relevant domain context from `.claude/domains/`
3. **Figure tasks**: Always check `configs/VISUALIZATION/figure_registry.yaml` first
4. **Combo tasks**: Always check `configs/VISUALIZATION/plot_hyperparam_combos.yaml` first

### For External AI Consultation

1. Copy entire contents of `.claude/CONTEXT_FOR_EXTERNAL_AI.md`
2. Add specific code snippets relevant to your question
3. Ask for architectural review, implementation advice, etc.

## Documentation Hierarchy

```
LEVEL 1: OVERVIEW
┌─────────────────────────────────────────────────────────────────┐
│ /CLAUDE.md                                                      │
│ - Research question (preprocessing → classification)            │
│ - NOT about comparing classifiers                               │
│ - Key findings (0.913 AUROC, 9pp embeddings gap)               │
│ - Subject counts (507 preprocess, 208 classify)                │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
LEVEL 2: BEHAVIOR CONTRACT
┌─────────────────────────────────────────────────────────────────┐
│ .claude/CLAUDE.md                                               │
│ - Quick reference table                                         │
│ - NEVER hallucinate combos rule                                │
│ - Available methods (verified from MLflow)                     │
│ - Standard vs extended combos                                   │
│ - Figure generation rules                                       │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
LEVEL 3: DOMAIN CONTEXT (Load as needed)
┌─────────────────────────────────────────────────────────────────┐
│ .claude/domains/visualization.md                                │
│ - Combo reference table                                         │
│ - Visual hierarchy (line styles, colors)                       │
│ - CI band rules                                                 │
│ - Figure type specifics                                         │
├─────────────────────────────────────────────────────────────────┤
│ .claude/domains/mlflow-experiments.md                           │
│ - Experiment locations                                          │
│ - Available outlier methods (17)                               │
│ - Available imputation methods (8)                             │
│ - Classifier list (6)                                          │
├─────────────────────────────────────────────────────────────────┤
│ configs/VISUALIZATION/figure_registry.yaml                                     │
│ - Complete figure catalog                                       │
│ - Generation scripts and entry points                          │
│ - Combo requirements per figure                                │
│ - Privacy levels (public/PRIVATE)                              │
├─────────────────────────────────────────────────────────────────┤
│ configs/VISUALIZATION/plot_hyperparam_combos.yaml                              │
│ - 4 standard combos (main figures)                             │
│ - 5 extended combos (supplementary)                            │
│ - Color definitions                                             │
│ - Preset groups                                                 │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
LEVEL 4: FULL EXTERNAL CONTEXT
┌─────────────────────────────────────────────────────────────────┐
│ .claude/CONTEXT_FOR_EXTERNAL_AI.md                              │
│ - Complete project overview                                     │
│ - Pipeline diagram                                              │
│ - Data infrastructure (SQLite, MLflow, DuckDB)                 │
│ - Configuration system details                                  │
│ - Visualization infrastructure                                  │
│ - Current problems and questions                               │
│ - File tree reference                                           │
└─────────────────────────────────────────────────────────────────┘
```

## Red Flags That Require Context Loading

| Red Flag | Action |
|----------|--------|
| User mentions "combo" or "hyperparam" | Load `plot_hyperparam_combos.yaml` |
| User mentions specific method name | Verify in `mlflow-experiments.md` |
| User asks about figure generation | Load `figure_registry.yaml` |
| User mentions "styling" or "ggplot2" | Load `visualization.md` |
| User asks "how many subjects" | Check root `CLAUDE.md` (507/208) |
| User mentions classifier comparison | STOP - that's not the research question |

## Maintenance

When updating documentation:

1. **Root CLAUDE.md**: Only for critical research context that never changes
2. **.claude/CLAUDE.md**: Behavior rules and quick reference
3. **Domain files**: Detailed context that changes with implementation
4. **External context**: Update when architecture changes

Keep each level focused. Don't duplicate information across levels.
