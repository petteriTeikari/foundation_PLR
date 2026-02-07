# fig-trans-16: Configuration vs Hardcoding

**Status**: 📋 PLANNED
**Tier**: 4 - Repository Patterns
**Target Persona**: Software engineers, data scientists, anyone who maintains code

---

## 1. Metadata

| Field | Value |
|-------|-------|
| Figure ID | fig-trans-16 |
| Type | Anti-pattern / best practice diagram |
| Style | 75% manuscript + 25% Economist |
| Dimensions | 12" × 10" |
| Format | PDF (vector) + PNG (300 DPI) |

---

## 2. Purpose

Demonstrate why hardcoding values in scientific code leads to reproducibility failures, and how the PLR repository uses YAML configuration files as single sources of truth.

---

## 3. Key Message

> "Every hardcoded value is a reproducibility bug waiting to happen. We learned this the hard way: one figure used prevalence=0.04, another used 0.035. Configuration files fix this."

---

## 4. Context

This figure addresses a critical anti-pattern documented in:
- `.claude/docs/meta-learnings/CRITICAL-FAILURE-002-hardcoding-despite-existing-systems.md`
- `.claude/docs/meta-learnings/CRITICAL-FAILURE-004-r-figure-hardcoding.md`

---

## 5. Visual Concept

```
┌────────────────────────────────────────────────────────────────────────────┐
│  CONFIGURATION vs HARDCODING                                               │
│  Why Every Magic Number Is a Bug                                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  THE PROBLEM                                                               │
│  ───────────                                                               │
│                                                                            │
│  ❌ BAD: Hardcoded values scattered across files                          │
│                                                                            │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐   │
│  │ figure_auroc.py    │  │ figure_dca.R       │  │ analysis.py        │   │
│  │ ──────────────     │  │ ────────────       │  │ ──────────         │   │
│  │ prevalence = 0.035 │  │ prev <- 0.04       │  │ PREVALENCE = 3.5%  │   │
│  │                    │  │    ↑               │  │                    │   │
│  │                    │  │    BUG!            │  │                    │   │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘   │
│                                                                            │
│  Problem: Which is correct? 0.035? 0.04? 3.5%?                            │
│  Answer: Nobody knows without reading the source paper.                    │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  THE SOLUTION                                                              │
│  ────────────                                                              │
│                                                                            │
│  ✓ GOOD: Single source of truth in YAML                                   │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │ configs/defaults.yaml  (THE ONLY SOURCE)                           │   │
│  │ ─────────────────────────────────────────                          │   │
│  │                                                                     │   │
│  │ glaucoma_params:                                                    │   │
│  │   prevalence: 0.0354          # Tham 2014 global estimate          │   │
│  │   target_sensitivity: 0.862   # Najjar 2023                        │   │
│  │   target_specificity: 0.821   # Najjar 2023                        │   │
│  │                                                                     │   │
│  │ bootstrap:                                                          │   │
│  │   n_iterations: 1000                                                │   │
│  │   alpha_ci: 0.95                                                    │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐   │
│  │ figure_auroc.py    │  │ figure_dca.R       │  │ analysis.py        │   │
│  │ ──────────────     │  │ ────────────       │  │ ──────────         │   │
│  │ cfg = load_config()│  │ cfg <- load_cfg()  │  │ cfg = load()       │   │
│  │ p = cfg.prevalence │  │ p <- cfg$prevalence│  │ p = cfg.prevalence │   │
│  │        ↓           │  │        ↓           │  │        ↓           │   │
│  │    ALL 0.0354      │  │    ALL 0.0354      │  │    ALL 0.0354      │   │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘   │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  CONFIGURATION HIERARCHY (PLR Repository)                                  │
│  ────────────────────────────────────────                                  │
│                                                                            │
│  configs/                                                                  │
│  ├── defaults.yaml           ← Global parameters (prevalence, etc.)       │
│  ├── mlflow_registry/        ← Method names (SINGLE SOURCE OF TRUTH)      │
│  │   └── parameters/                                                       │
│  │       └── classification.yaml  ← 11 outlier, 8 imputation methods     │
│  └── VISUALIZATION/          ← Figure configs                             │
│      ├── plot_hyperparam_combos.yaml  ← Standard 4 combos                 │
│      ├── figure_registry.yaml         ← All figure specs                  │
│      └── colors.yaml                  ← Color palette                     │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ENFORCEMENT (Pre-commit Hooks)                                            │
│  ──────────────────────────────                                            │
│                                                                            │
│  scripts/check_r_hardcoding.py  →  BLOCKS commits with:                   │
│    • Hex colors (#RRGGBB) in R files                                       │
│    • ggsave() instead of save_publication_figure()                         │
│    • Hardcoded dimensions                                                  │
│                                                                            │
│  "Mistakes that are easy to make should be hard to commit."               │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Text Content

### Title
"Configuration vs Hardcoding"

### Caption
"Every hardcoded value is a reproducibility bug waiting to happen. The PLR repository uses YAML configuration files as single sources of truth: `defaults.yaml` for global parameters (prevalence, bootstrap settings), `mlflow_registry/` for method names, and `VISUALIZATION/` for figure configs. Pre-commit hooks enforce this by blocking commits with hardcoded colors, dimensions, or method names. Rule: if a value appears in more than one file, it belongs in config."

---

## 7. Prompts for Nano Banana Pro

### Primary Prompt
```
Create a before/after diagram showing configuration vs hardcoding.

TOP - The Problem:
Three code files with different hardcoded prevalence values
(0.035, 0.04, 3.5%) - highlight the inconsistency

MIDDLE - The Solution:
Central YAML config file (configs/defaults.yaml)
Show same three files all reading from the config
All get consistent 0.0354

BOTTOM LEFT - Config hierarchy:
Tree showing configs/ structure
defaults.yaml, mlflow_registry/, VISUALIZATION/

BOTTOM RIGHT - Enforcement:
Pre-commit hooks that block hardcoding
"Mistakes easy to make should be hard to commit"

Style: Clear before/after, emphasis on consistency
```

---

## 8. Alt Text

"Diagram contrasting hardcoding versus configuration. Top shows three files with inconsistent prevalence values (0.035, 0.04, 3.5%). Middle shows solution: central defaults.yaml config file with prevalence 0.0354 and citation, with all three files reading from it consistently. Bottom left shows configuration hierarchy: defaults.yaml for parameters, mlflow_registry for method names, VISUALIZATION for figure configs. Bottom right describes pre-commit hooks that block hardcoded values."

---

## 9. Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in documentation
