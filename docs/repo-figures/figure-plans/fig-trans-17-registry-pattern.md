# fig-trans-17: The Registry Pattern

**Status**: 📋 PLANNED
**Tier**: 4 - Repository Patterns
**Target Persona**: Software engineers, ML engineers, data platform developers

---

## 1. Metadata

| Field | Value |
|-------|-------|
| Figure ID | fig-trans-17 |
| Type | Software architecture pattern diagram |
| Style | 75% manuscript + 25% Economist |
| Dimensions | 14" × 10" |
| Format | PDF (vector) + PNG (300 DPI) |

---

## 2. Purpose

Explain the "registry pattern" used in the PLR repository to maintain a single source of truth for method names, preventing the common bug of parsing experiment names and getting garbage like "anomaly" or "exclude".

---

## 3. Key Message

> "MLflow had 17 unique outlier method strings. The registry has 11. The difference? Orphan runs, test experiments, and typos. The registry pattern is: define valid values ONCE, validate EVERYWHERE."

---

## 4. Context

This pattern was developed after discovering that parsing MLflow run names produced garbage method names. Documented in:
- `.claude/docs/meta-learnings/` - Multiple failures from parsing run names
- `configs/mlflow_registry/README.md` - The registry specification

---

## 5. Visual Concept

```
┌────────────────────────────────────────────────────────────────────────────┐
│  THE REGISTRY PATTERN                                                      │
│  Single Source of Truth for Experiment Parameters                          │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  THE PROBLEM: Parsing Experiment Names                                     │
│  ─────────────────────────────────────                                     │
│                                                                            │
│  ❌ Common (broken) approach:                                              │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │                                                                   │     │
│  │  # Extract methods from MLflow run names                          │     │
│  │  methods = set()                                                  │     │
│  │  for run in mlflow.search_runs():                                 │     │
│  │      outlier = run.data.tags["mlflow.runName"].split("__")[3]    │     │
│  │      methods.add(outlier)                                         │     │
│  │                                                                   │     │
│  │  # Result: {'LOF', 'MOMENT-gt-finetune', 'anomaly', 'exclude',   │     │
│  │  #          'test_run', 'MOMENT-gt-fnetune', ...}                │     │
│  │  #          ↑           ↑              ↑                         │     │
│  │  #          garbage    test runs      typo                       │     │
│  │                                                                   │     │
│  │  print(len(methods))  # 17 (should be 11!)                       │     │
│  │                                                                   │     │
│  └──────────────────────────────────────────────────────────────────┘     │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  THE SOLUTION: Registry Pattern                                            │
│  ──────────────────────────────                                            │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  configs/mlflow_registry/parameters/classification.yaml            │   │
│  │  ──────────────────────────────────────────────────────            │   │
│  │                                                                     │   │
│  │  outlier_methods:                    # THE ONLY 11                 │   │
│  │    - pupil-gt                        # Ground truth                │   │
│  │    - MOMENT-gt-finetune              # Foundation model            │   │
│  │    - MOMENT-gt-zeroshot                                            │   │
│  │    - UniTS-gt-finetune                                             │   │
│  │    - TimesNet-gt                     # Deep learning               │   │
│  │    - LOF                             # Traditional                 │   │
│  │    - OneClassSVM                                                   │   │
│  │    - PROPHET                                                       │   │
│  │    - SubPCA                                                        │   │
│  │    - ensemble-LOF-MOMENT-...         # Ensemble                    │   │
│  │    - ensembleThresholded-...                                       │   │
│  │                                                                     │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  # src/data_io/registry.py                                         │   │
│  │                                                                     │   │
│  │  def get_valid_outlier_methods() -> list[str]:                     │   │
│  │      """Returns EXACTLY 11 methods from YAML."""                   │   │
│  │      cfg = yaml.safe_load(open(REGISTRY_PATH))                     │   │
│  │      return cfg["outlier_methods"]                                 │   │
│  │                                                                     │   │
│  │  def validate_outlier_method(method: str) -> bool:                 │   │
│  │      """Raises ValueError if method not in registry."""            │   │
│  │      if method not in get_valid_outlier_methods():                 │   │
│  │          raise ValueError(f"Invalid: {method}")                    │   │
│  │      return True                                                   │   │
│  │                                                                     │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  THE PATTERN                                                               │
│  ───────────                                                               │
│                                                                            │
│  ┌─────────────┐                                                          │
│  │   YAML      │  ← DEFINE valid values (single source of truth)          │
│  │  Registry   │                                                          │
│  └──────┬──────┘                                                          │
│         │                                                                  │
│         ▼                                                                  │
│  ┌─────────────┐                                                          │
│  │   Python    │  ← LOAD from YAML (never hardcode)                       │
│  │  get_valid* │                                                          │
│  └──────┬──────┘                                                          │
│         │                                                                  │
│    ┌────┴────┬─────────┬─────────┐                                        │
│    ▼         ▼         ▼         ▼                                        │
│  ┌─────┐ ┌─────┐ ┌───────┐ ┌───────┐                                      │
│  │ Viz │ │ Ext │ │ Tests │ │ CI/CD │  ← VALIDATE everywhere               │
│  └─────┘ └─────┘ └───────┘ └───────┘                                      │
│                                                                            │
│  Rule: If a count differs from registry count → CODE IS BROKEN            │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  WHY THIS MATTERS                                                          │
│  ────────────────                                                          │
│                                                                            │
│  Without registry:                                                         │
│  • Figure A shows 17 methods, Figure B shows 15 → inconsistent paper       │
│  • "anomaly" appears in heatmap → reviewer asks "what's anomaly?"         │
│  • Typo in run name → ghost method in analysis                             │
│                                                                            │
│  With registry:                                                            │
│  • All figures show exactly 11 methods → consistent                        │
│  • Invalid methods rejected at load time → fail fast                       │
│  • Single place to update → easy maintenance                               │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Text Content

### Title
"The Registry Pattern"

### Caption
"MLflow contained 17 unique outlier method strings; our registry defines exactly 11. The difference: orphan runs, test experiments, and typos. The registry pattern solves this: define valid values ONCE in YAML, load via accessor functions, validate EVERYWHERE. If code produces a different count than the registry, the code is broken. This pattern prevents inconsistent figures, ghost methods in analyses, and reviewer questions about undefined values."

---

## 7. Prompts for Nano Banana Pro

### Primary Prompt
```
Create a software pattern diagram for the registry approach.

TOP - The Problem:
Show code parsing MLflow run names
Result: 17 methods including 'anomaly', 'exclude', typos
Mark these as garbage/broken

MIDDLE - The Solution:
YAML registry file with exactly 11 methods
Python accessor functions (get_valid_*, validate_*)
Show clean, validated output

BOTTOM - The Pattern:
Flow diagram: YAML → Python accessor → Multiple consumers (Viz, Extraction, Tests, CI)
Arrow labels: DEFINE, LOAD, VALIDATE

FOOTER:
"If count differs from registry → code is broken"

Style: Software engineering diagram, clear hierarchy
```

---

## 8. Alt Text

"Software pattern diagram for registry approach. Top shows broken approach: parsing MLflow names produces 17 methods including garbage like 'anomaly'. Middle shows solution: YAML registry with exactly 11 methods and Python accessor functions. Bottom shows pattern flow: YAML defines truth, Python loads it, multiple consumers (visualization, extraction, tests) validate against it. Footer states rule: if count differs from registry, code is broken."

---

## 9. Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in documentation
