# Experiment Parameters Guardrail System

**Created:** 2026-01-26
**Status:** IMPLEMENTED
**Purpose:** Prevent invalid experiment parameters from appearing in scientific figures

---

## ELI5: What Is This System?

Imagine you have a recipe book (the registry) that says you can only use 11 specific spices. But sometimes the chef (Claude Code or a developer) might accidentally grab the wrong spice from the pantry (MLflow data) because the pantry has extra spices that look similar but aren't in the recipe.

This system makes sure:
1. **Everyone agrees on the 11 spices** - The recipe book, the shopping list, the tests, and the chef all have the same list
2. **Wrong spices get rejected** - If someone tries to use "anomaly spice" which isn't in the recipe, alarms go off
3. **No one can cheat** - Even if someone tries to secretly change the shopping list to match wrong spices, we catch it because we check multiple sources

---

## ELI5: Why Does This Exist?

We ran 400+ machine learning experiments with different combinations of:
- **11 outlier detection methods** (ways to find bad data points)
- **8 imputation methods** (ways to fill in missing data)
- **5 classifiers** (ML models)

The problem: Our MLflow database (where experiments are stored) contains **extra garbage** - test runs, failed experiments, and placeholder values like "anomaly" and "exclude" that should never appear in scientific figures.

**What went wrong before:**
- Scripts would scan MLflow and find 17 "methods" instead of 11
- Garbage like "anomaly" appeared in figures
- Ground truth AUROC showed as 0.88 instead of 0.911 (WRONG!)

**The fix:** Create a single source of truth (the registry) and make EVERYTHING validate against it.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SINGLE SOURCE OF TRUTH                                │
│                                                                          │
│   configs/mlflow_registry/parameters/classification.yaml                 │
│   ├── 11 outlier methods (anomaly_source)                               │
│   ├── 8 imputation methods (imputation_source)                          │
│   └── 5 classifiers (model_name)                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Validates against
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         VERIFICATION LAYER                               │
│                                                                          │
│   scripts/verify_registry_integrity.py                                   │
│   ├── Reads registry YAML                                               │
│   ├── Reads canary file (expected counts + checksum)                    │
│   ├── Reads registry.py (Python constants)                              │
│   ├── Reads test file (hardcoded assertions)                            │
│   └── FAILS if ANY source disagrees                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Runs on every commit
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          ENFORCEMENT LAYER                               │
│                                                                          │
│   .pre-commit-config.yaml                                                │
│   ├── registry-integrity hook (runs verify script)                      │
│   └── registry-validation hook (runs 53 pytest tests)                   │
│                                                                          │
│   Makefile                                                               │
│   ├── make test-registry                                                │
│   ├── make verify-registry-integrity                                    │
│   └── make check-registry (both)                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Used by
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION LAYER                                │
│                                                                          │
│   src/data_io/registry.py                                                │
│   ├── get_valid_outlier_methods() → exactly 11 methods                  │
│   ├── get_valid_imputation_methods() → exactly 8 methods                │
│   ├── validate_outlier_method("anomaly") → False (O(1) lookup)          │
│   └── Raises RegistryError if counts don't match                        │
│                                                                          │
│   Usage in extraction/analysis scripts:                                  │
│   from src.data_io.registry import validate_outlier_method              │
│   if not validate_outlier_method(some_method):                          │
│       raise ValueError("Invalid method!")                                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The Four Files That Must Agree (Anti-Cheat Mechanism)

To prevent anyone (including AI assistants) from temporarily modifying tests to make them pass, we require **four separate files** to all contain consistent values:

| File | What It Contains | Why It Exists |
|------|------------------|---------------|
| `configs/mlflow_registry/parameters/classification.yaml` | The actual method lists (11 + 8 + 5) | **Source of truth** |
| `configs/registry_canary.yaml` | Expected counts + SHA256 checksum | **Tamper detection** |
| `src/data_io/registry.py` | `EXPECTED_*_COUNT` constants | **Runtime validation** |
| `tests/test_registry.py` | Hardcoded test assertions | **CI enforcement** |

**How anti-cheat works:**
1. If Claude Code modifies tests to make them pass, the canary file will still have the original counts
2. If Claude Code modifies both tests AND canary, the registry YAML checksum won't match
3. If Claude Code modifies all three, CI will catch it because CI reads from the registry YAML independently

---

## What Each Component Does

### 1. Registry YAML (`configs/mlflow_registry/parameters/classification.yaml`)

The **ground truth** for all experiment parameters. This was generated from MLflow and manually curated to remove garbage.

```yaml
anomaly_source:
  values:
    - pupil-gt                    # Ground truth (human annotation)
    - MOMENT-gt-finetune          # Foundation model
    - MOMENT-gt-zeroshot
    - UniTS-gt-finetune
    - TimesNet-gt
    - LOF                         # Traditional methods
    - OneClassSVM
    - PROPHET
    - SubPCA
    - ensemble-LOF-MOMENT-...     # Ensembles
    - ensembleThresholded-...
  count: 11  # EXACTLY 11 - not 17, not 15
```

### 2. Canary File (`configs/registry_canary.yaml`)

A **sentinel file** that detects tampering. Contains:
- Expected counts (11, 8, 5)
- SHA256 checksum of the registry YAML
- List of INVALID methods that must never appear

```yaml
expected_counts:
  outlier_methods: 11
  imputation_methods: 8
  classifiers: 5

source_checksums:
  classification_yaml: "67a44d69200746bcfb1cf566aa292e8612c39e1c651193665026df1ad85c2201"

invalid_methods:
  outlier:
    - anomaly       # GARBAGE - must never appear
    - exclude       # GARBAGE
    - MOMENT-orig-finetune  # Not in registry
```

### 3. Registry Module (`src/data_io/registry.py`)

Python API for accessing valid methods with **O(1) validation**:

```python
from src.data_io.registry import (
    get_valid_outlier_methods,    # Returns exactly 11
    validate_outlier_method,       # O(1) frozenset lookup
)

# In extraction scripts:
methods = get_valid_outlier_methods()
assert len(methods) == 11  # Always true

# Validate before use:
if not validate_outlier_method("anomaly"):
    raise ValueError("Invalid!")  # This WILL raise
```

### 4. Test File (`tests/test_registry.py`)

53 pytest tests that enforce:
- Correct counts (11, 8, 5)
- All expected methods exist
- All invalid methods are rejected
- No duplicates
- Ground truth appears in both outlier and imputation

### 5. Integrity Verification Script (`scripts/verify_registry_integrity.py`)

Cross-checks all four files and fails if ANY disagree:

```bash
$ python scripts/verify_registry_integrity.py

============================================================
Registry Integrity Verification
============================================================

Checking consistency across:
  1. Canary file:     configs/registry_canary.yaml
  2. Registry YAML:   configs/mlflow_registry/parameters/classification.yaml
  3. Registry module: src/data_io/registry.py
  4. Test file:       tests/test_registry.py

✓ ALL INTEGRITY CHECKS PASSED

Summary:
  - Outlier methods:    11
  - Imputation methods: 8
  - Classifiers:        5
  - Checksum verified:  ✓
```

---

## How To Use This System

### For Developers: Validating Data

```python
from src.data_io.registry import (
    get_valid_outlier_methods,
    validate_outlier_method,
    RegistryError,
)

# Get all valid methods
valid_outliers = get_valid_outlier_methods()  # Exactly 11

# Validate before using a method from external data
def extract_from_mlflow(run):
    method = run.data.params["anomaly_source"]
    if not validate_outlier_method(method):
        logger.warning(f"Skipping invalid method: {method}")
        return None
    return method
```

### For CI: Running Checks

```bash
# Run all registry checks
make check-registry

# Or individually:
make verify-registry-integrity  # Anti-cheat verification
make test-registry              # Run 53 tests
```

### For Expanding the Experiment Design

If you ever need to add new methods (unlikely - this is for publication):

1. **Update registry YAML first:**
   ```yaml
   # configs/mlflow_registry/parameters/classification.yaml
   anomaly_source:
     values:
       - ...existing methods...
       - NEW-METHOD-NAME  # Add here
     count: 12  # Update count
   ```

2. **Regenerate checksum:**
   ```bash
   sha256sum configs/mlflow_registry/parameters/classification.yaml
   ```

3. **Update canary file:**
   ```yaml
   # configs/registry_canary.yaml
   expected_counts:
     outlier_methods: 12  # Update
   source_checksums:
     classification_yaml: "NEW_HASH_HERE"
   ```

4. **Update registry.py:**
   ```python
   EXPECTED_OUTLIER_COUNT = 12  # Update
   ```

5. **Update tests (if needed)**

6. **Commit all changes together** - pre-commit will verify consistency

---

## Code Review Findings & Improvements Made

### Original Issues Found (2026-01-26)

| Issue | Severity | Root Cause |
|-------|----------|------------|
| Pre-commit hooks ran on EVERY commit | CRITICAL | `always_run: true` overrode `files:` pattern |
| **Regex used for Python parsing** | CRITICAL | Violated explicit CLAUDE.md ban on regex for code analysis |
| Loguru warning on every import | HIGH | Warning at module level polluted logs |
| O(n) validation lookups | MEDIUM | `method in list` instead of `method in set` |
| Canary duplicated entire method lists | MEDIUM | 90 lines of redundant data |
| Checksum was placeholder | MEDIUM | `TO_BE_COMPUTED` provided no value |
| Ruff version outdated | LOW | v0.6.5 vs v0.9.1 installed |

### Improvements Implemented

| Change | File | Impact |
|--------|------|--------|
| Removed `always_run: true` | `.pre-commit-config.yaml` | Hooks now skip for non-registry files |
| **Replaced regex with AST parsing** | `verify_registry_integrity.py` | Robust Python code analysis (VIOLATION-001 fix) |
| Added O(1) frozenset validation | `registry.py` | Constant-time lookups |
| Removed import-time warning | `registry.py` | Clean imports, no log pollution |
| Added `__all__` | `registry.py` | Explicit public API |
| Simplified canary file | `registry_canary.yaml` | 90 → 60 lines |
| Computed real SHA256 | `registry_canary.yaml` | Actual integrity verification |
| Added YAML error handling | `verify_registry_integrity.py` | Graceful failures |
| Fixed exit codes | `verify_registry_integrity.py` | Correct semantics |
| Updated Ruff | `.pre-commit-config.yaml` | Consistent linting |
| Fixed deprecated stages | `.pre-commit-config.yaml` | `commit` → `pre-commit` |

### Verification Results

```
$ make check-registry
✓ Registry Integrity Check passed
✓ 53 tests passed in 0.06s
✓ All registry checks passed
```

---

## Related Documents

| Document | Purpose |
|----------|---------|
| `.claude/rules/05-registry-source-of-truth.md` | Rule file Claude Code reads |
| `CLAUDE.md` (root) | Main behavior contract |
| `.claude/CLAUDE.md` | Additional behavior rules |
| `configs/mlflow_registry/README.md` | Registry documentation |
| `docs/planning/pipeline-robustification-plan.md` | Parent planning document |

---

## Glossary

| Term | Definition |
|------|------------|
| **Registry** | The YAML file defining valid experiment parameters |
| **Canary** | A sentinel file used to detect tampering |
| **Outlier method** | Algorithm for detecting bad data points (e.g., LOF, MOMENT) |
| **Imputation method** | Algorithm for filling missing data (e.g., SAITS, CSDI) |
| **Ground truth** | Human-annotated data (`pupil-gt`) |
| **AUROC** | Area Under ROC Curve - classification performance metric |
| **Frozenset** | Immutable Python set for O(1) membership testing |

---

## FAQ

**Q: Why not just filter in the extraction script?**
A: We tried that. It failed repeatedly because there was no validation against expected counts. The registry system ensures we KNOW when something is wrong.

**Q: Why four files? Isn't that overkill?**
A: It's specifically to prevent AI assistants from "cheating" by temporarily modifying tests. With four files that must agree, any tampering is caught.

**Q: What if someone legitimately needs to add a new method?**
A: Follow the "Expanding the Experiment Design" section. All four files must be updated together, and pre-commit will verify consistency.

**Q: Why is the checksum important?**
A: It detects if the registry YAML was modified without updating the canary. Even if someone updates the counts correctly, a different checksum indicates the actual method list changed.

**Q: How fast are the validation functions?**
A: O(1) constant time using cached frozensets. The frozenset is computed once on first call and cached forever.
