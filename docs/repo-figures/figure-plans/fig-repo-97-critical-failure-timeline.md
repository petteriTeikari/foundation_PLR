# fig-repo-97: 6 Critical Failures That Shaped Our Guardrails

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-97 |
| **Title** | 6 Critical Failures That Shaped Our Guardrails |
| **Complexity Level** | L2 |
| **Target Persona** | All (explains why the codebase is strict) |
| **Location** | `CONTRIBUTING.md`, `docs/explanation/guardrails.md`, root `README.md` |
| **Priority** | P2 (High) |

## Purpose

Explain the origin story behind every major guardrail in the codebase. New contributors often wonder why the pre-commit hooks, import bans, and test requirements are so strict. Each guardrail exists because a specific failure happened, was documented, and triggered the creation of automated prevention. The codebase's strictness is not over-engineering -- it is hard-won defensive coding.

## Key Message

Every guardrail in this codebase was created in response to a specific, documented critical failure. The strictness is earned, not arbitrary.

## Content Specification

### Panel 1: Timeline (Cause to Effect)

```
+======================================================================+
||  6 CRITICAL FAILURES --> 6 GUARDRAIL SYSTEMS                        ||
||                                                                      ||
||  Each failure was documented, analyzed, and prevented from recurring ||
+======================================================================+

2026-01-25
+---------------------------------------------------------------------------+
| CRITICAL-FAILURE-001: Synthetic Data in Scientific Figures                 |
|                                                                            |
| WHAT HAPPENED:                                                             |
|   Claude generated calibration plots using SYNTHETIC data instead of       |
|   real model predictions. The figures looked plausible but were            |
|   scientifically meaningless. Discovered during user review.              |
|                                                                            |
| GUARDRAILS CREATED:                                                        |
|   +-- tests/test_figure_qa/test_data_provenance.py                        |
|   |     Verifies JSON sidecar exists for every PNG, traces data origin    |
|   +-- .pre-commit-config.yaml: figure-isolation-check                     |
|   |     Prevents synthetic data from appearing in figures/generated/      |
|   +-- src/utils/data_mode.py                                              |
|         4-gate isolation architecture (synthetic vs production)            |
|                                                                            |
| DOCS: .claude/docs/meta-learnings/CRITICAL-FAILURE-001-synthetic-data-    |
|       in-figures.md                                                        |
+---------------------------------------------------------------------------+

2026-01-26
+---------------------------------------------------------------------------+
| CRITICAL-FAILURE-002: Mixed Featurization in Extraction                    |
|                                                                            |
| WHAT HAPPENED:                                                             |
|   Extraction pulled ALL MLflow runs including orphan runs with garbage     |
|   method names ("anomaly", "exclude") and wrong featurization types.      |
|   Method count jumped from 11 to 17. Data corruption propagated to        |
|   all downstream figures and analyses.                                     |
|                                                                            |
| GUARDRAILS CREATED:                                                        |
|   +-- configs/mlflow_registry/ (Registry as single source of truth)       |
|   |     YAML definitions: exactly 11 outlier, 8 imputation, 5 classifier  |
|   +-- src/data_io/registry.py (EXPECTED_*_COUNT constants)                |
|   +-- .pre-commit-config.yaml: registry-integrity + registry-validation   |
|   |     5-layer anti-cheat verification                                   |
|   +-- .pre-commit-config.yaml: extraction-isolation-check                 |
|         Prevents synthetic data in extraction paths                        |
|                                                                            |
| DOCS: .claude/docs/meta-learnings/CRITICAL-FAILURE-002-mixed-             |
|       featurization-in-extraction.md (root CLAUDE.md reference)           |
+---------------------------------------------------------------------------+

2026-01-27
+---------------------------------------------------------------------------+
| CRITICAL-FAILURE-003: Computation in Visualization Code                    |
|                                                                            |
| WHAT HAPPENED:                                                             |
|   src/viz/ files imported sklearn.metrics and scipy.stats to compute      |
|   metrics on the fly. This violated the two-block architecture            |
|   (extraction computes, visualization reads). Metrics could differ from   |
|   those in DuckDB, creating inconsistencies.                              |
|                                                                            |
| GUARDRAILS CREATED:                                                        |
|   +-- .pre-commit-config.yaml: computation-decoupling                     |
|   |     scripts/check_computation_decoupling.py scans src/viz/ for        |
|   |     banned imports (sklearn, scipy, src/stats)                        |
|   +-- tests/test_no_hardcoding/test_computation_decoupling.py             |
|   |     Test-level enforcement of the import ban                          |
|   +-- DuckDB tables: essential_metrics, calibration_curves, dca_curves,   |
|         retention_metrics, cohort_metrics, distribution_stats             |
|         (all metrics pre-computed and stored)                             |
|                                                                            |
| DOCS: .claude/docs/meta-learnings/CRITICAL-FAILURE-003-computation-       |
|       decoupling-violation.md                                              |
+---------------------------------------------------------------------------+

2026-01-29
+---------------------------------------------------------------------------+
| CRITICAL-FAILURE-004: Hardcoded Values Everywhere                          |
|                                                                            |
| WHAT HAPPENED:                                                             |
|   R figure scripts contained hardcoded hex colors ("#006BA2"),            |
|   hardcoded ggsave() paths, hardcoded dimensions, and custom themes.     |
|   Systematic, recurring violation across all R files. Style changes       |
|   required editing every file individually.                               |
|                                                                            |
| GUARDRAILS CREATED:                                                        |
|   +-- .pre-commit-config.yaml: r-hardcoding-check                        |
|   |     scripts/check_r_hardcoding.py scans .R files for violations       |
|   +-- tests/test_no_hardcoding/test_absolute_paths.py                     |
|   +-- tests/test_no_hardcoding/test_computation_decoupling.py             |
|   +-- src/r/figure_system/ (shared helpers)                               |
|   |     theme_foundation_plr(), load_color_definitions(),                 |
|   |     save_publication_figure()                                         |
|   +-- configs/VISUALIZATION/colors.yaml (single color source)            |
|                                                                            |
| DOCS: .claude/docs/meta-learnings/CRITICAL-FAILURE-004-r-figure-         |
|       hardcoding.md                                                        |
+---------------------------------------------------------------------------+

2026-01-31
+---------------------------------------------------------------------------+
| CRITICAL-FAILURE-005: Visual Bug Priority / Stuck Extraction               |
|                                                                            |
| WHAT HAPPENED:                                                             |
|   An extraction script ran for 24 hours stuck in swap thrashing without   |
|   any heartbeat or progress indication. Also: visual bugs in figures      |
|   were deprioritized behind infrastructure work, leading to repeated      |
|   user frustration (bug mentioned 2+ times = CRITICAL).                   |
|                                                                            |
| GUARDRAILS CREATED:                                                        |
|   +-- src/extraction/guardrails.py                                        |
|   |     ExtractionGuardrails: MemoryMonitor, DiskMonitor, StallDetector,  |
|   |     ProgressTracker (heartbeat every 60s)                             |
|   +-- Bug-First Rule (in .claude/CLAUDE.md)                               |
|   |     "Fix visual bugs IMMEDIATELY. Infrastructure waits."              |
|   +-- Verify-Output Rule                                                   |
|         "After modifying figure code, VIEW the regenerated PNG."          |
|                                                                            |
| DOCS: .claude/docs/meta-learnings/CRITICAL-FAILURE-005-stuck-extraction-  |
|       without-detection.md                                                 |
+---------------------------------------------------------------------------+

2026-02 (ongoing)
+---------------------------------------------------------------------------+
| CRITICAL-FAILURE-006: Shortcuts in Academic Code                           |
|                                                                            |
| WHAT HAPPENED:                                                             |
|   "Quick wins" and shortcuts were proposed that bypassed existing          |
|   verified implementations. Implementation without reading existing       |
|   modules led to duplicated logic, inconsistent behavior, and             |
|   reimplementation of verified libraries.                                 |
|                                                                            |
| GUARDRAILS CREATED:                                                        |
|   +-- Pre-Implementation Checklist (in .claude/CLAUDE.md)                 |
|   |     1. Search existing code first                                     |
|   |     2. Read related modules                                           |
|   |     3. Check test files for patterns                                  |
|   |     4. Follow reviewer recommendations                               |
|   +-- .claude/rules/25-no-reimplementation.md                             |
|   |     Use verified libraries (pminternal, dcurves, SmoothECE) via       |
|   |     interop. Creating src/stats/*_reimplementation.py is FORBIDDEN.   |
|   +-- Code review discipline                                              |
|         Read existing code before writing new code                         |
|                                                                            |
| DOCS: Referenced in .claude/CLAUDE.md (CRITICAL-FAILURE-006 section)      |
+---------------------------------------------------------------------------+
```

### Panel 2: The Pattern (Callout)

```
+----------------------------------------------------------------------+
|  THE PATTERN: Every Guardrail Has a Scar                               |
|                                                                        |
|  Failure Happened --> Documented in .claude/docs/meta-learnings/       |
|                   --> Automated prevention created (pre-commit/test)    |
|                   --> Rule codified in .claude/rules/*.md               |
|                   --> AI agent instructions updated in CLAUDE.md        |
|                                                                        |
|  Result: The same failure CANNOT happen again.                         |
|  New failures create new guardrails. The system only gets stricter.    |
+----------------------------------------------------------------------+
```

### Panel 3: Guardrail Coverage Map

```
Failure Category          Automated Prevention          Manual Rule
+----------------------+  +------------------------+  +------------------+
| Synthetic data       |  | figure-isolation-check |  | Figure QA tests  |
| in figures (#001)    |  | extraction-isolation    |  | before commit    |
+----------------------+  +------------------------+  +------------------+
| Wrong method counts  |  | registry-integrity     |  | 5-layer anti-    |
| in extraction (#002) |  | registry-validation    |  | cheat system     |
+----------------------+  +------------------------+  +------------------+
| Computation in viz   |  | computation-decoupling |  | Two-block arch   |
| code (#003)          |  | test_computation_*     |  | documentation    |
+----------------------+  +------------------------+  +------------------+
| Hardcoded values     |  | r-hardcoding-check     |  | Self-check       |
| (#004)               |  | test_no_hardcoding/*   |  | before code      |
+----------------------+  +------------------------+  +------------------+
| Stuck extraction     |  | ExtractionGuardrails   |  | Bug-First rule   |
| (#005)               |  | ProgressTracker        |  | Verify output    |
+----------------------+  +------------------------+  +------------------+
| Academic shortcuts   |  | No automated hook yet  |  | Pre-implement    |
| (#006)               |  | (enforced by review)   |  | checklist        |
+----------------------+  +------------------------+  +------------------+
```

## Spatial Anchors

```yaml
layout_flow: "Top-down: timeline (6 entries), pattern callout, coverage map"
spatial_anchors:
  timeline:
    x: 0.5
    y: 0.4
    content: "6 critical failures in chronological order with guardrails"
  pattern:
    x: 0.5
    y: 0.78
    content: "The pattern: failure -> document -> automate -> codify"
  coverage_map:
    x: 0.5
    y: 0.92
    content: "Which guardrails cover which failure categories"
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `.pre-commit-config.yaml` | All automated hooks (registry, decoupling, isolation, R hardcoding) |
| `.claude/CLAUDE.md` | Behavior contract with CRITICAL-FAILURE references |
| `.claude/rules/*.md` | 6 rule files codifying lessons learned |
| `configs/registry_canary.yaml` | Anti-cheat reference values (11/8/5) |

## Code Paths

| Module | Role |
|--------|------|
| `.claude/docs/meta-learnings/CRITICAL-FAILURE-001-*.md` | Synthetic data incident report |
| `.claude/docs/meta-learnings/CRITICAL-FAILURE-003-*.md` | Computation decoupling incident |
| `.claude/docs/meta-learnings/CRITICAL-FAILURE-004-*.md` | R hardcoding incident |
| `.claude/docs/meta-learnings/CRITICAL-FAILURE-005-*.md` | Stuck extraction incident |
| `scripts/verify_registry_integrity.py` | Registry anti-cheat (responds to #002) |
| `scripts/check_computation_decoupling.py` | Import ban (responds to #003) |
| `scripts/check_r_hardcoding.py` | R hardcoding prevention (responds to #004) |
| `src/extraction/guardrails.py` | Extraction protection (responds to #005) |
| `tests/test_figure_qa/test_data_provenance.py` | Figure provenance (responds to #001) |

## Extension Guide

To document a new critical failure:
1. Create `.claude/docs/meta-learnings/CRITICAL-FAILURE-NNN-description.md`
2. Include: What happened, Root cause, Impact, Guardrail created
3. Add automated prevention (pre-commit hook, test, or both)
4. Update `.claude/CLAUDE.md` with the new rule
5. Create a `.claude/rules/*.md` file if the rule is broadly applicable
6. Update this figure plan (Panel 1) with the new timeline entry

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-97",
    "title": "6 Critical Failures That Shaped Our Guardrails"
  },
  "content_architecture": {
    "primary_message": "Every guardrail in this codebase was created in response to a specific, documented critical failure. The strictness is earned, not arbitrary.",
    "layout_flow": "Top-down: chronological timeline, pattern callout, coverage map",
    "spatial_anchors": {
      "timeline": {"x": 0.5, "y": 0.4},
      "pattern": {"x": 0.5, "y": 0.78},
      "coverage_map": {"x": 0.5, "y": 0.92}
    },
    "key_structures": [
      {
        "name": "CRITICAL-FAILURE-001",
        "role": "abnormal_warning",
        "is_highlighted": true,
        "labels": ["Synthetic data in figures", "test_data_provenance.py", "figure-isolation-check"]
      },
      {
        "name": "CRITICAL-FAILURE-002",
        "role": "abnormal_warning",
        "is_highlighted": true,
        "labels": ["Mixed featurization", "Registry validation", "5-layer anti-cheat"]
      },
      {
        "name": "CRITICAL-FAILURE-003",
        "role": "abnormal_warning",
        "is_highlighted": true,
        "labels": ["Computation in viz", "computation-decoupling hook", "Import ban"]
      },
      {
        "name": "CRITICAL-FAILURE-004",
        "role": "abnormal_warning",
        "is_highlighted": true,
        "labels": ["Hardcoded values", "r-hardcoding-check", "colors.yaml"]
      },
      {
        "name": "CRITICAL-FAILURE-005",
        "role": "abnormal_warning",
        "is_highlighted": true,
        "labels": ["Stuck extraction", "ExtractionGuardrails", "Bug-First rule"]
      },
      {
        "name": "CRITICAL-FAILURE-006",
        "role": "abnormal_warning",
        "is_highlighted": true,
        "labels": ["Academic shortcuts", "Pre-implementation checklist", "No reimplementation rule"]
      }
    ],
    "callout_boxes": [
      {"heading": "THE PATTERN", "body_text": "Failure -> Document in meta-learnings -> Automate prevention -> Codify in rules. The same failure cannot happen again."},
      {"heading": "KEY INSIGHT", "body_text": "The codebase's strictness comes from hard-won lessons, not over-engineering."}
    ]
  }
}
```

## Alt Text

Timeline of 6 critical failures (synthetic data, mixed featurization, computation in viz, hardcoding, stuck extraction, shortcuts) with the guardrails each one created.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
