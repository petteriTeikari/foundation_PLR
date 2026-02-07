# Display Name Lookup Table Implementation Plan

**Status**: READY FOR IMPLEMENTATION
**Created**: 2026-01-27
**Last Updated**: 2026-01-27
**Goal**: Map raw MLflow method names to publication-friendly display names

## Problem Statement

Current visualization labels use raw MLflow names which are:
- Technical and cryptic (e.g., `MOMENT-gt-finetune`, `pupil-gt`)
- Inconsistent formatting (hyphens, abbreviations)
- Not suitable for publication figures

## RESOLVED DECISIONS

All open questions have been resolved before implementation:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| File location | `configs/mlflow_registry/display_names.yaml` | Alongside other registry files |
| File format | YAML | Allows comments, consistent with registry |
| `-gt` handling | Remove from display names | Internal abbreviation, not meaningful to readers |
| Hyphenation | `Fine-tuned`, `Zero-shot` | Standard English compound adjective hyphenation |
| Ensemble names | Component list for small (≤3), "Full" for large | Clarity vs brevity tradeoff |
| Foundation model variants | Add suffix `(GT-based)` or `(Raw-based)` | Disambiguate training data source |
| Prophet capitalization | `Prophet` (not PROPHET) | It's a proper noun (Meta's library) |
| Fallback behavior | Return raw name with WARNING log | Never fail silently, never crash |

## Complete Method List from Registry

### Outlier Detection Methods (11)

| MLflow Name | Display Name | Category |
|-------------|--------------|----------|
| `pupil-gt` | Ground Truth | ground_truth |
| `MOMENT-gt-finetune` | MOMENT Fine-tuned | foundation_model |
| `MOMENT-gt-zeroshot` | MOMENT Zero-shot | foundation_model |
| `UniTS-gt-finetune` | UniTS Fine-tuned | foundation_model |
| `TimesNet-gt` | TimesNet | deep_learning |
| `LOF` | LOF | traditional |
| `OneClassSVM` | One-Class SVM | traditional |
| `PROPHET` | Prophet | traditional |
| `SubPCA` | Subspace PCA | traditional |
| `ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune` | Ensemble (Full) | ensemble |
| `ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune` | Ensemble (Thresholded) | ensemble |

### Imputation Methods (8)

| MLflow Name | Display Name | Category |
|-------------|--------------|----------|
| `pupil-gt` | Ground Truth | ground_truth |
| `MOMENT-finetune` | MOMENT Fine-tuned | foundation_model |
| `MOMENT-zeroshot` | MOMENT Zero-shot | foundation_model |
| `CSDI` | CSDI | deep_learning |
| `SAITS` | SAITS | deep_learning |
| `TimesNet` | TimesNet | deep_learning |
| `ensemble-CSDI-MOMENT-SAITS` | Ensemble (CSDI+MOMENT+SAITS) | ensemble |
| `ensemble-CSDI-MOMENT-SAITS-TimesNet` | Ensemble (Deep Learning) | ensemble |

### Classifiers (5)

| MLflow Name | Display Name |
|-------------|--------------|
| `CatBoost` | CatBoost |
| `XGBoost` | XGBoost |
| `TabPFN` | TabPFN |
| `TabM` | TabM |
| `LogisticRegression` | Logistic Regression |

### Category Display Names

| Internal Category | Display Name |
|-------------------|--------------|
| `ground_truth` | Ground Truth |
| `foundation_model` | Foundation Model |
| `deep_learning` | Deep Learning |
| `traditional` | Traditional |
| `ensemble` | Ensemble |

## File Structure

```
configs/mlflow_registry/
├── parameters/
│   └── classification.yaml     # Existing - defines valid methods
├── display_names.yaml          # NEW - maps to display names
└── README.md                   # Update with display_names docs
```

## YAML Schema

### `configs/mlflow_registry/display_names.yaml`

```yaml
# Display Names for Publication Figures
# =====================================
# Maps raw MLflow method names to publication-friendly display names.
# Used by both Python (src/data_io/display_names.py) and R (src/r/utils/load_display_names.R).
#
# RULES:
# 1. Every method in classification.yaml MUST have a display name here
# 2. Display names should be:
#    - Capitalized appropriately (MOMENT, LOF, not moment, lof)
#    - Free of internal abbreviations (-gt, -orig)
#    - Hyphenated for compound terms (Zero-shot, Fine-tuned)
#    - Suitable for publication figures
#
# VALIDATION: Run `pytest tests/unit/test_display_names.py -v`

version: "1.0.0"

outlier_methods:
  # Ground Truth
  pupil-gt: "Ground Truth"

  # Foundation Models - MOMENT
  MOMENT-gt-finetune: "MOMENT Fine-tuned"
  MOMENT-gt-zeroshot: "MOMENT Zero-shot"

  # Foundation Models - UniTS
  UniTS-gt-finetune: "UniTS Fine-tuned"

  # Deep Learning - TimesNet
  TimesNet-gt: "TimesNet"

  # Traditional Methods
  LOF: "LOF"
  OneClassSVM: "One-Class SVM"
  PROPHET: "Prophet"
  SubPCA: "Subspace PCA"

  # Ensembles
  ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune: "Ensemble (Full)"
  ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune: "Ensemble (Thresholded)"

imputation_methods:
  # Ground Truth
  pupil-gt: "Ground Truth"

  # Foundation Models
  MOMENT-finetune: "MOMENT Fine-tuned"
  MOMENT-zeroshot: "MOMENT Zero-shot"

  # Deep Learning
  CSDI: "CSDI"
  SAITS: "SAITS"
  TimesNet: "TimesNet"

  # Ensembles
  ensemble-CSDI-MOMENT-SAITS: "Ensemble (CSDI+MOMENT+SAITS)"
  ensemble-CSDI-MOMENT-SAITS-TimesNet: "Ensemble (Deep Learning)"

classifiers:
  CatBoost: "CatBoost"
  XGBoost: "XGBoost"
  TabPFN: "TabPFN"
  TabM: "TabM"
  LogisticRegression: "Logistic Regression"

# Category display names (for legends and grouping)
categories:
  ground_truth: "Ground Truth"
  foundation_model: "Foundation Model"
  deep_learning: "Deep Learning"
  traditional: "Traditional"
  ensemble: "Ensemble"
```

## TDD Implementation Plan

### Cycle 1: Create Test File (RED)

Create `tests/unit/test_display_names.py` with tests that WILL FAIL initially.

**Test Cases (24 total):**

| Test # | Test Name | Assertion |
|--------|-----------|-----------|
| 1 | `test_display_names_yaml_exists` | File exists at expected path |
| 2 | `test_yaml_is_valid` | YAML parses without error |
| 3 | `test_has_version` | `version` field exists |
| 4 | `test_has_outlier_methods_section` | `outlier_methods` section exists |
| 5 | `test_has_imputation_methods_section` | `imputation_methods` section exists |
| 6 | `test_has_classifiers_section` | `classifiers` section exists |
| 7 | `test_all_outlier_methods_covered` | All 11 registry methods have display names |
| 8 | `test_all_imputation_methods_covered` | All 8 registry methods have display names |
| 9 | `test_all_classifiers_covered` | All 5 registry classifiers have display names |
| 10 | `test_no_extra_outlier_methods` | No display names for non-existent methods |
| 11 | `test_no_extra_imputation_methods` | No display names for non-existent methods |
| 12 | `test_no_gt_in_display_names` | No display name contains "-gt" |
| 13 | `test_no_orig_in_display_names` | No display name contains "-orig" |
| 14 | `test_ground_truth_display` | `pupil-gt` → "Ground Truth" |
| 15 | `test_moment_finetuned_display` | Contains "Fine-tuned" (hyphenated) |
| 16 | `test_moment_zeroshot_display` | Contains "Zero-shot" (hyphenated) |
| 17 | `test_prophet_capitalization` | `PROPHET` → "Prophet" |
| 18 | `test_lof_uppercase` | `LOF` remains "LOF" |
| 19 | `test_svm_expansion` | `OneClassSVM` → "One-Class SVM" |
| 20 | `test_ensemble_full_name` | Full ensemble → "Ensemble (Full)" |
| 21 | `test_ensemble_thresholded_name` | Thresholded → "Ensemble (Thresholded)" |
| 22 | `test_python_module_loads` | `from src.data_io.display_names import get_outlier_display_name` works |
| 23 | `test_fallback_returns_raw_name` | Unknown method returns raw name + logs WARNING |
| 24 | `test_get_all_display_names_returns_dict` | Combined dict of all display names |

### Cycle 2: Create YAML File (GREEN for tests 1-6)

Create the YAML file with schema structure (no content yet).

### Cycle 3: Populate Display Names (GREEN for tests 7-21)

Fill in all display name mappings.

### Cycle 4: Create Python Module (GREEN for tests 22-24)

Create `src/data_io/display_names.py` with:
- `get_outlier_display_name(method: str) -> str`
- `get_imputation_display_name(method: str) -> str`
- `get_classifier_display_name(classifier: str) -> str`
- `get_category_display_name(category: str) -> str`
- `get_all_display_names() -> dict[str, str]`

**Error Handling Specification:**
- If YAML file missing: `raise FileNotFoundError` with clear message
- If method not found: return raw name, log `logger.warning(f"No display name for: {method}")`
- If YAML parse error: `raise yaml.YAMLError` with context

### Cycle 5: Create R Module (Integration)

Create `src/r/utils/load_display_names.R` with:
- `load_display_names()` - returns named list
- `get_outlier_display_name(method)` - returns display name or raw
- `apply_display_names(df)` - adds `*_display` columns

### Cycle 6: Update Forest Plot (Integration)

Modify `src/r/figures/fig02_forest_outlier.R` to use display names.

## Verification Commands

```bash
# Run all display name tests
pytest tests/unit/test_display_names.py -v

# Verify YAML is valid
python -c "import yaml; yaml.safe_load(open('configs/mlflow_registry/display_names.yaml'))"

# Check all methods have display names
python -c "from src.data_io.display_names import get_all_display_names; print(len(get_all_display_names()))"

# Regenerate forest plot with display names
Rscript src/r/figures/fig02_forest_outlier.R
```

## Success Criteria

- [ ] `display_names.yaml` exists with all 24 method mappings
- [ ] Every method in registry has a display name (11 + 8 + 5 = 24)
- [ ] No display name contains "-gt" or "-orig"
- [ ] Python module provides O(1) lookup via `@lru_cache`
- [ ] R module loads same YAML file
- [ ] Forest plot uses display names
- [ ] JSON exports include display names
- [ ] All 24 tests pass
- [ ] Unknown methods return raw name with WARNING log (no crash)

## Implementation Order

1. **Create tests first** (`tests/unit/test_display_names.py`) - ALL TESTS WILL FAIL
2. **Create YAML file** (`configs/mlflow_registry/display_names.yaml`)
3. **Create Python module** (`src/data_io/display_names.py`)
4. **Run tests** - verify all pass
5. **Create R module** (`src/r/utils/load_display_names.R`)
6. **Update forest plot** - verify display names appear
7. **Update JSON exports** - verify display names in metadata
