# Claude Code Context (`.claude/`)

This directory contains AI assistant context for working with the Foundation PLR codebase.

## Overview

The `.claude/` directory provides structured context to help AI assistants (like Claude) understand and work effectively with this codebase. It contains:

- **Rules and constraints** for code generation
- **Domain-specific knowledge** about the research
- **Configuration** for context loading
- **Institutional knowledge** accumulated during development

## Directory Structure

```
.claude/
├── CLAUDE.md                    # Main behavior contract
├── README.md                    # This file
├── CONTEXT_FOR_EXTERNAL_AI.md   # Condensed context for other AIs
├── GUARDRAILS.md                # Hard constraints and banned patterns
├── README_PROGRESSIVE.md        # Progressive disclosure approach
│
├── rules/                       # Mandatory rules (loaded automatically)
│   └── 00-research-question.md  # Core research context
│
├── domains/                     # Domain-specific context
│   ├── mlflow-experiments.md    # MLflow data and experiments
│   ├── visualization.md         # Figure generation
│   ├── testing.md               # Test organization
│   └── configuration.md         # Hydra config system
│
├── institutional-knowledge/     # Lessons learned
│   └── *.md
│
├── config/                      # Context configuration
│   └── auto-context.yaml
│
├── commands/                    # Custom slash commands
├── hooks/                       # Pre/post hooks
├── scripts/                     # Helper scripts
├── sessions/                    # Session artifacts
├── plans/                       # Planning documents
└── planning/                    # Active plans
```

## Key Files

### CLAUDE.md

The main behavior contract containing:
- Research question focus (preprocessing effects, NOT classifier comparison)
- Standard pipeline combinations
- Metric registry usage
- Figure generation rules

### rules/

Rules in this directory are loaded automatically:
- `00-research-question.md` - Core research context (read every session)

### domains/

Domain-specific context loaded on demand:
- `mlflow-experiments.md` - Available experiments, method names
- `visualization.md` - Figure styling, combinations
- `testing.md` - Test organization, running tests
- `configuration.md` - Hydra configuration system

## Loading Context

Context is loaded progressively based on task:

| Task | Load |
|------|------|
| General questions | `CLAUDE.md` |
| Figure generation | `CLAUDE.md` + `domains/visualization.md` |
| MLflow queries | `CLAUDE.md` + `domains/mlflow-experiments.md` |
| Running tests | `CLAUDE.md` + `domains/testing.md` |
| Configuration | `CLAUDE.md` + `domains/configuration.md` |

## Critical Rules

1. **Research Focus**: This is about preprocessing effects, NOT comparing classifiers
2. **STRATOS Metrics**: Report ALL metrics (AUROC, calibration, clinical utility)
3. **Standard Combos**: Always use `configs/VISUALIZATION/plot_hyperparam_combos.yaml`
4. **Metric Registry**: Use `src/viz/metric_registry.py`, don't hardcode
5. **Ground Truth**: Always include ground_truth combo in comparisons

## For Other AI Assistants

If you're an AI assistant other than Claude:
- Read `CONTEXT_FOR_EXTERNAL_AI.md` for condensed context
- The core research question is in `rules/00-research-question.md`
- Key constraints are in `GUARDRAILS.md`

## Updating Context

When adding new institutional knowledge:

1. Create a new file in `institutional-knowledge/`
2. Update relevant domain files if needed
3. Consider adding to `CLAUDE.md` if it's a persistent rule

## See Also

- [ARCHITECTURE.md](../ARCHITECTURE.md) - Pipeline overview
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guide
