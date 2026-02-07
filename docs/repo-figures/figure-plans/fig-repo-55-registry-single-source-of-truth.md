# Figure Plan: fig-repo-55-registry-single-source-of-truth

**Target**: Repository documentation infographic
**Section**: `configs/mlflow_registry/`
**Purpose**: Explain the Registry Pattern and why it's critical
**Version**: 1.0

---

## Title

**Registry as Single Source of Truth: Data Flow and Validation**

---

## Purpose

Help developers understand:
1. WHY the registry pattern exists (prevent garbage from MLflow parsing)
2. WHERE the registry files are
3. HOW data flows from registry → code → validation
4. WHAT happens when you violate the pattern

---

## Visual Layout (Data Flow Diagram)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FIGURE TITLE: Registry as Single Source of Truth                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  THE PROBLEM                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                          ││
│  │   MLflow Runs          Parsing Run Names           Result                ││
│  │   ┌──────────┐         ┌──────────────┐           ┌──────────┐          ││
│  │   │ run_001  │         │              │           │ GARBAGE  │          ││
│  │   │ run_002  │  ───▷   │ run.split()  │   ───▷    │ "anomaly"│          ││
│  │   │ orphan_x │         │              │           │ "exclude"│          ││
│  │   │ test_run │         │              │           │ 17 methods│         ││
│  │   └──────────┘         └──────────────┘           └──────────┘          ││
│  │                                                                          ││
│  │   ❌ WRONG: Parsing run names gives orphan/test data                     ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  THE SOLUTION: REGISTRY PATTERN                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                          ││
│  │   ┌──────────────────────────────────────────────────────────────────┐  ││
│  │   │                         REGISTRY                                  │  ││
│  │   │  configs/mlflow_registry/parameters/classification.yaml           │  ││
│  │   │                                                                   │  ││
│  │   │  outlier_methods:           imputation_methods:                   │  ││
│  │   │    - pupil-gt                 - pupil-gt                          │  ││
│  │   │    - MOMENT-gt-finetune       - SAITS                             │  ││
│  │   │    - MOMENT-gt-zeroshot       - CSDI                              │  ││
│  │   │    - UniTS-gt-finetune        - linear                            │  ││
│  │   │    - TimesNet-gt              - MOMENT-finetune                   │  ││
│  │   │    - LOF                      - MOMENT-zeroshot                   │  ││
│  │   │    - OneClassSVM              - TimesNet                          │  ││
│  │   │    - PROPHET                  - ensemble-CSDI-...                 │  ││
│  │   │    - SubPCA                                                       │  ││
│  │   │    - ensemble-LOF-...         classifiers:                        │  ││
│  │   │    - ensembleThresholded-...    - CatBoost                        │  ││
│  │   │                                 - XGBoost                         │  ││
│  │   │  EXACTLY 11 outlier methods     - TabPFN                          │  ││
│  │   │  EXACTLY 8 imputation methods   - TabM                            │  ││
│  │   │  EXACTLY 5 classifiers          - LogisticRegression              │  ││
│  │   └──────────────────────────────────────────────────────────────────┘  ││
│  │                           │                                              ││
│  │                           ▼                                              ││
│  │   ┌──────────────────────────────────────────────────────────────────┐  ││
│  │   │                     PYTHON CODE                                   │  ││
│  │   │  src/data_io/registry.py                                          │  ││
│  │   │                                                                   │  ││
│  │   │  from src.data_io.registry import get_valid_outlier_methods       │  ││
│  │   │  valid = get_valid_outlier_methods()  # Returns EXACTLY 11        │  ││
│  │   │                                                                   │  ││
│  │   │  # VALIDATE before use                                            │  ││
│  │   │  if method not in valid:                                          │  ││
│  │   │      raise ValueError(f"Invalid: {method}")                       │  ││
│  │   └──────────────────────────────────────────────────────────────────┘  ││
│  │                           │                                              ││
│  │                           ▼                                              ││
│  │   ┌──────────────────────────────────────────────────────────────────┐  ││
│  │   │                     VALIDATION                                    │  ││
│  │   │  tests/test_registry.py                                           │  ││
│  │   │                                                                   │  ││
│  │   │  def test_outlier_count():                                        │  ││
│  │   │      assert len(get_valid_outlier_methods()) == 11                │  ││
│  │   │                                                                   │  ││
│  │   │  def test_no_garbage():                                           │  ││
│  │   │      for m in ["anomaly", "exclude"]:                             │  ││
│  │   │          assert m not in get_valid_outlier_methods()              │  ││
│  │   └──────────────────────────────────────────────────────────────────┘  ││
│  │                                                                          ││
│  │   ✅ CORRECT: Registry → Code → Validation = EXACTLY 11                  ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  WHAT HAPPENS WHEN YOU VIOLATE THIS PATTERN                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                          ││
│  │  ❌ Parsing MLflow run names                                             ││
│  │     → Gets orphan runs, test runs, malformed names                      ││
│  │     → "anomaly", "exclude" garbage methods                              ││
│  │     → 17 methods instead of 11                                          ││
│  │     → WRONG figures, WRONG analysis                                     ││
│  │                                                                          ││
│  │  ❌ Hardcoding method names in scripts                                   ││
│  │     → Drift from registry over time                                     ││
│  │     → Inconsistent across codebase                                      ││
│  │     → Review failures                                                   ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Content Elements

### File Locations

| File | Purpose |
|------|---------|
| `configs/mlflow_registry/parameters/classification.yaml` | Master registry |
| `src/data_io/registry.py` | Python interface |
| `tests/test_registry.py` | Validation tests |

### Exact Counts (MEMORIZE)

| Parameter | Count | If Different = BROKEN |
|-----------|-------|----------------------|
| Outlier methods | **11** | |
| Imputation methods | **8** | |
| Classifiers | **5** | |

### Invalid Methods (NEVER USE)

- `anomaly` - garbage placeholder
- `exclude` - garbage placeholder
- `MOMENT-orig-finetune` - not in registry
- `UniTS-orig-*` - not in registry
- `TimesNet-orig` - not in registry

---

## Key Messages

1. **NEVER parse MLflow run names**: They contain orphan/test data
2. **ALWAYS use the registry**: `get_valid_outlier_methods()` returns exactly 11
3. **Validate method names**: Reject anything not in registry
4. **Tests enforce counts**: CI fails if registry contract broken

---

## Technical Specifications

- **Aspect ratio**: 16:12 (taller for data flow)
- **Resolution**: 300 DPI
- **Background**: #FBF9F3 (Economist off-white)
- **Typography**: Sans-serif, dark grey (#333333)
- **Colour coding**: Red for wrong, green for correct

---

## Related Documentation

- **README**: `configs/mlflow_registry/README.md`
- **Related infographic**: fig-repo-53 (Outlier Detection Methods - shows the 11)

---

*Figure plan created: 2026-02-02*
*For: configs/mlflow_registry/ documentation - CRITICAL pattern*
