# fig-repo-84: CLAUDE.md: How AI Agent Instructions Compose

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-84 |
| **Title** | CLAUDE.md: How AI Agent Instructions Compose |
| **Complexity Level** | L3 |
| **Target Persona** | ML Engineer (who uses Claude Code) |
| **Location** | `.claude/README.md`, `docs/onboarding/` |
| **Priority** | P3 (Medium) |

## Purpose

Show how the multi-layered CLAUDE.md instruction system works: which files are always loaded, which are loaded on demand, and how per-directory overrides specialize behavior. Developers using Claude Code need to understand this hierarchy to write effective instructions and debug unexpected AI behavior.

## Key Message

AI instructions follow a 3-tier hierarchy: always-loaded (30K chars, project-wide), on-demand (domain-specific context), and per-directory overrides (specialized behavior). Instructions cascade and more specific files override general ones.

## Content Specification

### Panel 1: Three-Tier Instruction Hierarchy

```
TIER 1: ALWAYS LOADED (30K chars total)
========================================
Loaded automatically at the start of every Claude Code session.

┌─────────────────────────────────────────────────────────────────┐
│ CLAUDE.md (root, ~9K chars)                                      │
│   Project overview, data provenance, pipeline description,       │
│   key findings, subject counts, sister repos, dependencies       │
│                                                                   │
│ .claude/CLAUDE.md (~9K chars)                                     │
│   Behavior contract: CRITICAL rules quick-reference table,       │
│   Figure QA (CRITICAL-FAILURE-001), Computation decoupling       │
│   (CRITICAL-FAILURE-003), No shortcuts (CRITICAL-FAILURE-006),   │
│   Anti-hardcoding matrix, Hyperparameter combo rules             │
│                                                                   │
│ .claude/rules/ (6 files, ~7K chars total)                        │
│   ├── 00-research-question.md     Fix classifier, vary preproc   │
│   ├── 05-registry-source-of-truth.md  11/8/5 method counts      │
│   ├── 10-figures.md               Max 4 curves, ground truth     │
│   ├── 15-stratos-metrics.md       5 domains, not AUROC-only      │
│   ├── 20-package-management.md    uv only, conda/pip BANNED      │
│   └── 25-no-reimplementation.md   Use verified libs via interop  │
│                                                                   │
│ .claude/auto-context.yaml (~2K chars)                             │
│   Auto-loaded context selector (triggers domain loading)          │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼ loaded when relevant task detected
TIER 2: LOADED ON DEMAND
========================================
Loaded when Claude Code detects a relevant task.

┌─────────────────────────────────────────────────────────────────┐
│ .claude/domains/                                                  │
│   ├── mlflow-experiments.md    MLflow paths, experiment IDs,     │
│   │                            run naming conventions             │
│   ├── visualization.md        Figure generation patterns,        │
│   │                            color system, save_figure() usage  │
│   ├── configuration.md        Hydra composition, config dirs     │
│   └── testing.md              Test tiers, markers, fixtures      │
│                                                                   │
│ .claude/docs/meta-learnings/                                      │
│   ├── CRITICAL-FAILURE-001    Synthetic data in figures           │
│   ├── CRITICAL-FAILURE-002    Mixed featurization extraction      │
│   ├── CRITICAL-FAILURE-003    Computation in visualization        │
│   ├── CRITICAL-FAILURE-004    R figure hardcoding                 │
│   ├── CRITICAL-FAILURE-005    Stuck extraction undetected         │
│   └── VIOLATION-001/002       Regex for code parsing              │
│                                                                   │
│ .claude/planning/                                                  │
│   └── Active planning docs (15 files)                              │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼ when working in a specific directory
TIER 3: PER-DIRECTORY OVERRIDES
========================================
Override or extend behavior for specific directories.

┌─────────────────────────────────────────────────────────────────┐
│ docs/repo-figures/CLAUDE.md                                       │
│   "Repository figures show CODE not RESULTS"                      │
│   BANNED: AUROC values, method rankings, performance metrics      │
│   REQUIRED: config paths, code paths, extension guides            │
│                                                                   │
│ docs/repo-figures/plans-TODO/CLAUDE.md                            │
│   Allowed: brief theory WITHOUT results                           │
│   Naming: fig-repo-{NN}-{descriptive-name}.md                    │
│                                                                   │
│ (other dirs may have their own CLAUDE.md)                          │
└─────────────────────────────────────────────────────────────────┘
```

### Panel 2: Instruction Cascade Example

```
Developer works in src/viz/calibration_plot.py
  │
  ├── Tier 1 loads automatically:
  │   ├── Root CLAUDE.md: "calibration slope/intercept are in DuckDB"
  │   ├── .claude/CLAUDE.md: "src/viz/ BANNED from importing sklearn"
  │   └── rules/15-stratos-metrics.md: "calibration slope is REQUIRED"
  │
  ├── Tier 2 triggered by task:
  │   └── domains/visualization.md: "use setup_style(), COLORS dict"
  │
  └── Result: Claude Code knows to READ calibration_slope from DuckDB,
      use COLORS for rendering, and include it as a STRATOS requirement
```

### Panel 3: Instruction Budget

```
ALWAYS LOADED             ON DEMAND                  TOTAL CAPACITY
┌─────────────────┐      ┌─────────────────┐       ┌─────────────┐
│ ~30K chars       │      │ Variable         │       │ ~200K chars  │
│ (15% of budget)  │      │ (loaded as       │       │ context      │
│                  │      │  needed)         │       │ window       │
│ Always available │      │ Domain-specific  │       │              │
│ Every session    │      │ Task-triggered   │       │              │
└─────────────────┘      └─────────────────┘       └─────────────┘

Design principle: Keep always-loaded lean (30K).
Move specialist knowledge to on-demand domains.
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `.claude/auto-context.yaml` | Defines triggers for loading domain files |
| `.claude/rules/*.md` | 6 numbered rule files (00-25) |
| `CLAUDE.md` (root) | Project-level overview and data sources |
| `.claude/CLAUDE.md` | Behavior contract and enforcement rules |

## Code Paths

| Module | Role |
|--------|------|
| `CLAUDE.md` | Root project context (~9K chars) |
| `.claude/CLAUDE.md` | Behavior contract (~9K chars) |
| `.claude/rules/00-research-question.md` | Research question (fix classifier, vary preprocessing) |
| `.claude/rules/05-registry-source-of-truth.md` | Registry = 11/8/5 method counts |
| `.claude/rules/10-figures.md` | Figure constraints (max 4 curves, ground truth) |
| `.claude/rules/15-stratos-metrics.md` | STRATOS compliance (5 metric domains) |
| `.claude/rules/20-package-management.md` | uv only, conda/pip banned |
| `.claude/rules/25-no-reimplementation.md` | Use verified libraries via interop |
| `.claude/domains/mlflow-experiments.md` | MLflow paths and experiment documentation |
| `.claude/domains/visualization.md` | Figure generation patterns |
| `.claude/docs/meta-learnings/` | Post-mortem documents for critical failures |
| `docs/repo-figures/CLAUDE.md` | Per-directory override: code not results |

## Extension Guide

To add a new instruction:
1. **Project-wide rule**: Add `.claude/rules/NN-rule-name.md` (keep numbered, 5-char increments)
2. **Domain context**: Add `.claude/domains/domain-name.md` (loaded on demand)
3. **Per-directory override**: Add `CLAUDE.md` in the target directory
4. **Meta-learning**: Add `.claude/docs/meta-learnings/CRITICAL-FAILURE-NNN-description.md`
5. Update `.claude/auto-context.yaml` if new domain needs auto-loading triggers

To reduce always-loaded size:
- Move verbose content from root/`.claude/CLAUDE.md` into rules or domains
- Replace with 1-line cross-references: `**Details**: See .claude/rules/NN-*.md`

Note: This is a repo documentation figure - shows HOW the instruction system works, NOT research results.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "repo-84",
    "title": "CLAUDE.md: How AI Agent Instructions Compose"
  },
  "content_architecture": {
    "primary_message": "AI instructions follow a 3-tier hierarchy: always-loaded (30K chars), on-demand (domain-specific), and per-directory overrides.",
    "layout_flow": "Top-down three-tier stack with cascade arrows and instruction budget sidebar",
    "spatial_anchors": {
      "tier1": {"x": 0.1, "y": 0.05, "width": 0.6, "height": 0.35},
      "tier2": {"x": 0.1, "y": 0.4, "width": 0.6, "height": 0.25},
      "tier3": {"x": 0.1, "y": 0.65, "width": 0.6, "height": 0.15},
      "cascade_example": {"x": 0.75, "y": 0.05, "width": 0.2, "height": 0.45},
      "budget": {"x": 0.75, "y": 0.55, "width": 0.2, "height": 0.25}
    },
    "key_structures": [
      {
        "name": "Tier 1: Always Loaded",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["30K chars", "Every session"]
      },
      {
        "name": "Tier 2: On Demand",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["Domain-specific", "Task-triggered"]
      },
      {
        "name": "Tier 3: Per-Directory",
        "role": "highlight_accent",
        "is_highlighted": true,
        "labels": ["Override behavior", "Most specific wins"]
      }
    ],
    "callout_boxes": [
      {"heading": "DESIGN PRINCIPLE", "body_text": "Keep always-loaded lean (30K). Move specialist knowledge to on-demand domains."}
    ]
  }
}
```

## Alt Text

Three-tier diagram showing CLAUDE.md instruction hierarchy: always-loaded rules at top, on-demand domain context in middle, and per-directory overrides at bottom, with cascade arrows showing how instructions compose.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
