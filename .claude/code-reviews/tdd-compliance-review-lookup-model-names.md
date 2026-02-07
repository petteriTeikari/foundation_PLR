# TDD Compliance Review: Display Name Lookup Implementation
**Date**: 2026-01-27
**Document**: `/docs/planning/lookup-model-names.md`
**Status**: NEEDS WORK → FAIL (Critical Gaps)
**Reviewer**: Claude Code

---

## EXECUTIVE SUMMARY

**RATING: FAIL** ⚠️

The planning document proposes a **well-structured solution** BUT **violates core TDD principles** by:

1. **Tests are INCOMPLETE** - Only stubs and placeholders, not executable
2. **Implementation code shown in planning document** - Should not exist there
3. **Integration tests missing** - No tests for cross-module interactions
4. **Edge cases under-specified** - Fallback behavior unclear
5. **Missing: Error handling test cases** - What happens on YAML parse failure?
6. **Missing: Validation tests** - Are display names unique? Properly formatted?

### Verdict
**Cannot be implemented as-is. Requires test expansion before coding begins.**

---

## DETAILED ANALYSIS

### ✅ WHAT'S GOOD

#### 1. Clear Problem Statement
- Identifies the real issue: raw MLflow names are cryptic (`MOMENT-gt-finetune`)
- Proposes sensible mapping (`MOMENT Fine-tuned`)
- Shows concrete examples

#### 2. Reasonable Architecture Decision
- ✅ YAML over JSON (allows comments)
- ✅ Alongside existing registry files (consistency)
- ✅ Single source of truth pattern

#### 3. Good Module Structure
- Proposed file paths are correct
- API functions well-named and documented
- LRU caching is appropriate

#### 4. Integration Points Identified
- Extraction flow
- R integration
- CSV/JSON exports

---

### 🔴 CRITICAL GAPS - FAIL CRITERIA

#### 1. TESTS ARE NOT TEST-DRIVEN (P1 CRITICAL)

**Problem**: The test file contains incomplete stubs:

```python
class TestDisplayNamesIntegration:
    def test_csv_export_uses_display_names(self):
        """CSV export should include display_name column."""
        # This will be tested after implementation  # ← NOT A TEST!
        pass

    def test_json_export_uses_display_names(self):
        """JSON exports should use display names for labels."""
        # This will be tested after implementation  # ← NOT A TEST!
        pass
```

**TDD Rule Violated**: RED phase must have FAILING tests, not stubs.

**What Should Happen**:
```python
def test_csv_export_uses_display_names(self):
    """CSV export should include display_name column."""
    from src.data_io.extraction import export_to_csv_with_display_names

    test_df = pd.DataFrame({
        'outlier_method': ['pupil-gt', 'LOF'],
        'imputation_method': ['SAITS', 'pupil-gt'],
        'auroc': [0.911, 0.859]
    })

    # FAILING TEST - export function doesn't exist yet
    export_to_csv_with_display_names(test_df, 'test_output.csv')

    # THEN verify display names are in output
    result = pd.read_csv('test_output.csv')
    assert 'outlier_display_name' in result.columns
    assert result['outlier_display_name'].iloc[0] == 'Ground Truth'
    assert result['outlier_display_name'].iloc[1] == 'LOF'
```

---

#### 2. MISSING: FILE EXISTENCE AND VALIDATION TESTS (P1)

**Problem**: No tests verify YAML structure is correct.

```python
# MISSING - should be in test class
def test_display_names_yaml_schema_valid(self):
    """YAML must have correct structure: outlier_methods, imputation_methods, classifiers."""
    from src.data_io.display_names import _load_display_names

    names = _load_display_names()

    # Schema validation
    assert 'outlier_methods' in names
    assert 'imputation_methods' in names
    assert 'classifiers' in names
    assert isinstance(names['outlier_methods'], dict)
    assert isinstance(names['imputation_methods'], dict)
    assert isinstance(names['classifiers'], dict)

def test_all_values_are_strings(self):
    """All display names must be strings, not None."""
    from src.data_io.display_names import _load_display_names

    names = _load_display_names()

    for section in ['outlier_methods', 'imputation_methods', 'classifiers']:
        for key, value in names[section].items():
            assert isinstance(value, str), f"{section}[{key}] is not a string: {value}"
            assert len(value) > 0, f"{section}[{key}] is empty string"
```

---

#### 3. MISSING: MISSING METHOD HANDLING (P1)

**Problem**: No test for what happens when method name has no display name.

```python
# MISSING - critical error case
def test_get_outlier_display_name_missing_method_returns_none(self):
    """When method has no display name, function should return None or raise."""
    from src.data_io.display_names import get_outlier_display_name

    result = get_outlier_display_name('nonexistent-method-xyz')

    # Current implementation returns None (silent failure)
    # Should we raise instead? Test must specify!
    assert result is None  # or assert raises(KeyError)?

def test_get_outlier_display_name_missing_method_raises(self):
    """Alternative: missing methods should raise KeyError."""
    from src.data_io.display_names import get_outlier_display_name

    with pytest.raises(KeyError, match="nonexistent-method-xyz"):
        get_outlier_display_name('nonexistent-method-xyz')
```

**This is not a minor issue** - affects error handling throughout visualization code.

---

#### 4. MISSING: YAML PARSE FAILURE HANDLING (P2)

```python
def test_display_names_yaml_missing_raises(self):
    """When YAML doesn't exist, function should raise helpful error."""
    from src.data_io import display_names

    # Temporarily move YAML
    import tempfile
    with tempfile.TemporaryDirectory():
        # Mock missing file
        original_path = display_names.DISPLAY_NAMES_PATH
        display_names.DISPLAY_NAMES_PATH = Path('/nonexistent/display_names.yaml')

        with pytest.raises(FileNotFoundError, match="Display names"):
            display_names._load_display_names()

        display_names.DISPLAY_NAMES_PATH = original_path

def test_display_names_yaml_corrupt_raises(self):
    """When YAML is malformed, should provide clear error."""
    from src.data_io import display_names
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("{ invalid yaml: [")  # Malformed
        temp_path = f.name

    try:
        original_path = display_names.DISPLAY_NAMES_PATH
        display_names.DISPLAY_NAMES_PATH = Path(temp_path)

        with pytest.raises(yaml.YAMLError):
            display_names._load_display_names()

        display_names.DISPLAY_NAMES_PATH = original_path
    finally:
        Path(temp_path).unlink()
```

---

#### 5. MISSING: DUPLICATE DISPLAY NAME DETECTION (P2)

```python
def test_no_duplicate_display_names(self):
    """No two raw methods should map to same display name (ambiguous)."""
    from src.data_io.display_names import get_all_display_names

    all_mappings = get_all_display_names()
    display_names = list(all_mappings.values())

    # Check for duplicates
    assert len(display_names) == len(set(display_names)), \
        "Duplicate display names detected"
```

**Why**: If two different methods map to "MOMENT Fine-tuned", plots become ambiguous.

---

#### 6. MISSING: CROSS-REGISTRY VALIDATION (P2)

```python
def test_all_registry_methods_have_display_names(self):
    """Every method from classification.yaml must have a display name.

    This catches when registry is updated but display_names.yaml is not.
    """
    from src.data_io.registry import (
        get_valid_outlier_methods,
        get_valid_imputation_methods,
        get_valid_classifiers,
    )
    from src.data_io.display_names import (
        get_outlier_display_name,
        get_imputation_display_name,
        get_classifier_display_name,
    )

    # Outlier methods
    for method in get_valid_outlier_methods():
        display = get_outlier_display_name(method)
        assert display is not None, f"Outlier method '{method}' has no display name"
        assert isinstance(display, str), f"Display name for '{method}' is not string: {display}"

    # Imputation methods
    for method in get_valid_imputation_methods():
        display = get_imputation_display_name(method)
        assert display is not None, f"Imputation method '{method}' has no display name"

    # Classifiers
    for clf in get_valid_classifiers():
        display = get_classifier_display_name(clf)
        assert display is not None, f"Classifier '{clf}' has no display name"
```

**This is CRITICAL for publication** - catches registry drift.

---

#### 7. MISSING: CACHING BEHAVIOR TESTS (P1)

```python
def test_display_names_cached_between_calls(self):
    """Verify @lru_cache works - only loads YAML once."""
    from src.data_io import display_names

    # Clear cache
    display_names._load_display_names.cache_clear()

    # First call
    result1 = display_names._load_display_names()
    cache_info1 = display_names._load_display_names.cache_info()

    # Second call (should hit cache)
    result2 = display_names._load_display_names()
    cache_info2 = display_names._load_display_names.cache_info()

    # Verify cache was hit
    assert cache_info2.hits == 1, "LRU cache not working"
    assert cache_info2.misses == 1, "First call should be a miss"
    assert result1 is result2, "Should return same cached object"
```

**Why**: Caching is a performance requirement - must verify it works.

---

#### 8. MISSING: CATEGORY DISPLAY NAME TESTS (P2)

```python
def test_get_category_display_name_valid(self):
    """Category display names should be retrievable."""
    from src.data_io.display_names import get_category_display_name

    # From YAML: categories.outlier.foundation_model = "Foundation Model"
    assert get_category_display_name('outlier', 'foundation_model') == 'Foundation Model'
    assert get_category_display_name('outlier', 'traditional') == 'Traditional'
    assert get_category_display_name('imputation', 'ensemble') == 'Ensemble'

def test_get_category_display_name_missing_category(self):
    """Missing categories should return None or raise."""
    from src.data_io.display_names import get_category_display_name

    result = get_category_display_name('outlier', 'nonexistent')
    assert result is None  # or raises?
```

---

#### 9. MISSING: HYPHENATION CONSISTENCY TESTS (P2)

```python
def test_foundation_model_names_use_hyphens_correctly(self):
    """Foundation models should use "Fine-tuned" not "Finetuned"."""
    from src.data_io.display_names import (
        get_outlier_display_name,
        get_imputation_display_name,
    )

    # Check standard hyphenation
    assert "Fine-tuned" in get_outlier_display_name("MOMENT-gt-finetune")
    assert "Zero-shot" in get_outlier_display_name("MOMENT-gt-zeroshot")

    # No double-hyphens or leading/trailing hyphens
    for method in ["MOMENT-gt-finetune", "MOMENT-gt-zeroshot", "UniTS-gt-finetune"]:
        display = get_outlier_display_name(method)
        assert not display.startswith('-'), f"Leading hyphen in '{display}'"
        assert not display.endswith('-'), f"Trailing hyphen in '{display}'"
        assert '--' not in display, f"Double hyphen in '{display}'"
```

---

#### 10. MISSING: ENSEMBLE NAME CLARITY TESTS (P2)

```python
def test_ensemble_names_are_unambiguous(self):
    """Ensemble display names should clearly indicate what they contain."""
    from src.data_io.display_names import get_outlier_display_name

    # These ensemble names should be different enough to tell apart in figures
    full_ensemble = get_outlier_display_name(
        "ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune"
    )
    thresh_ensemble = get_outlier_display_name(
        "ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune"
    )

    # Names should be clearly distinct
    assert "Full" in full_ensemble or "All" in full_ensemble
    assert "Threshold" in thresh_ensemble or "Subset" in thresh_ensemble
    assert full_ensemble != thresh_ensemble
```

---

### 🟡 MEDIUM PRIORITY GAPS

#### 11. MISSING: CAPITALIZATION STANDARDIZATION TESTS

Current planning document says:
- "Zero-shot" (title case on first word)
- "Fine-tuned" (title case)
- But "Logistic Regression" vs "logistic regression"?

```python
def test_capitalization_consistent(self):
    """All display names should follow consistent capitalization."""
    from src.data_io.display_names import get_all_display_names

    all_names = get_all_display_names()

    for raw, display in all_names.items():
        # Should start with capital letter
        assert display[0].isupper(), f"'{display}' should start with capital"

        # No ALL_CAPS except for acronyms (LOF, SVM, etc.)
        assert not (display.isupper() and len(display) > 3), \
            f"'{display}' is all caps (should be title case)"
```

---

#### 12. MISSING: SPECIAL CHARACTER HANDLING

```python
def test_no_special_chars_in_display_names(self):
    """Display names should only contain: letters, numbers, hyphens, spaces."""
    import string
    from src.data_io.display_names import get_all_display_names

    allowed_chars = set(string.ascii_letters + string.digits + '- ()')

    all_names = get_all_display_names()

    for raw, display in all_names.items():
        for char in display:
            assert char in allowed_chars, \
                f"Invalid character '{char}' in display name '{display}'"
```

---

### 🔵 IMPLEMENTATION CODE IN PLANNING DOCUMENT (P1)

**Problem**: The planning document includes actual Python/R implementation code:

Lines 267-337: `src/data_io/display_names.py` implementation
Lines 362-403: `src/r/load_display_names.R` implementation

**TDD Principle Violated**: Planning document should contain:
- ✅ Test cases (RED phase)
- ✅ Architecture decisions
- ✅ Schema/data structures
- ❌ Implementation code (belongs in actual files after tests pass)

**What Should Happen**:
1. Planning doc: Define requirements and test cases
2. Write `tests/unit/test_display_names.py` (RED)
3. Run tests, verify they fail
4. Implement `src/data_io/display_names.py` (GREEN)
5. Clean up and optimize (REFACTOR)

**Consequence**: If implementation code is in planning doc, someone might copy-paste it without running tests first.

---

## SPECIFIC TEST GAPS SUMMARY

| Test Case | Status | Priority | Reason |
|-----------|--------|----------|--------|
| YAML file exists | Present ✅ | - | Good |
| All methods have display names | Present ✅ | - | Good |
| No `-gt` in display names | Present ✅ | - | Good |
| **Missing method handling** | ❌ MISSING | P1 | Error path unclear |
| **YAML parse failure** | ❌ MISSING | P1 | Production robustness |
| **Duplicate display names** | ❌ MISSING | P2 | Ambiguity detection |
| **Cross-registry validation** | ❌ MISSING | P1 | Catches drift |
| **Caching behavior** | ❌ MISSING | P1 | Performance requirement |
| **CSV/JSON export integration** | ❌ STUB ONLY | P1 | Integration untested |
| **Category display names** | ❌ MISSING | P2 | Feature completeness |
| **Hyphenation consistency** | ❌ MISSING | P2 | Publication standards |
| **Ensemble name clarity** | ❌ MISSING | P2 | Figure interpretation |
| **Capitalization standardization** | ❌ MISSING | P2 | Visual consistency |
| **Special character validation** | ❌ MISSING | P2 | Data integrity |
| **Fallback behavior** | ❌ MISSING | P2 | Edge case handling |

---

## EDGE CASES UNDER-SPECIFIED

### 1. What if a method is in registry but not in display_names.yaml?

Current code returns `None` (line 308):
```python
def get_outlier_display_name(method: str) -> Optional[str]:
    names = _load_display_names()
    return names.get("outlier_methods", {}).get(method)  # Silent None if missing
```

**Should this:**
- Option A: Return None (graceful degradation, use raw name as fallback)
- Option B: Raise KeyError (fail fast, catch bugs in YAML)
- Option C: Return raw method name as default

**Test must decide**:
```python
def test_missing_display_name_behavior(self):
    """Decide: None vs KeyError vs fallback?"""
    from src.data_io.display_names import get_outlier_display_name

    # Specify ONE of these:
    # assert get_outlier_display_name("MISSING") is None
    # or
    # assert get_outlier_display_name("MISSING") == "MISSING"  # fallback
    # or
    # with pytest.raises(KeyError):
    #     get_outlier_display_name("MISSING")
```

### 2. What if YAML is malformed?

Should `_load_display_names()` raise yaml.YAMLError or log and return empty dict?

### 3. What if display names contain unicode?

Should we test for encoding issues?

---

## COMPLIANCE WITH PROJECT STANDARDS

### Registry Pattern ✅
Planning follows existing `src/data_io/registry.py` pattern:
- LRU caching ✅
- Validation functions ✅
- Public API exports ✅

### But: Missing Integration Test
The existing registry module has `tests/integration/test_extraction_registry.py` that tests interaction with extraction code.

**Missing**: Equivalent integration test for display_names:
```python
# tests/integration/test_display_names_extraction.py - MISSING
def test_display_names_in_duckdb_export():
    """After extraction, DuckDB should have display_name columns."""
    pass  # Not specified in planning doc
```

---

## RECOMMENDED FIXES (PRIORITY ORDER)

### P1 - BLOCKING (Do before implementation)

1. **Add missing method error handling test**
   - Decide: Return None vs KeyError vs fallback
   - Document in test

2. **Add cross-registry validation test**
   - Every registry method must have display name
   - Catches drift when registry is updated

3. **Add YAML parse failure test**
   - Handle FileNotFoundError
   - Handle yaml.YAMLError
   - Provide helpful messages

4. **Replace stub integration tests with real ones**
   - `test_csv_export_uses_display_names` should actually call export
   - `test_json_export_uses_display_names` should verify JSON structure

5. **Add caching behavior test**
   - Verify @lru_cache actually works
   - Performance requirement

6. **Remove implementation code from planning doc**
   - Move to actual files after tests pass
   - Keep planning doc to requirements only

### P2 - IMPORTANT (Before publication)

7. Add duplicate display name detection test
8. Add category display name tests
9. Add hyphenation consistency tests
10. Add ensemble name clarity tests
11. Add capitalization standardization test
12. Add special character validation test

### P3 - NICE-TO-HAVE

13. Add unicode/encoding tests
14. Add performance benchmark (load time < 1ms)
15. Add documentation example tests

---

## WHAT TDD SHOULD LOOK LIKE

### Current (Planning Document) - INCOMPLETE
```
Tests ✅ (partially)
  └─ Some stubs only
Implementation ❌ (in wrong place)
  └─ Code in planning doc
Integration ❌ (not specified)
  └─ Placeholder comments
```

### Correct TDD Cycle

```
1. RED (Tests Written First)
   ├─ test_display_names.py (13 failing test cases)
   ├─ test_display_names_integration.py (3 failing integration tests)
   └─ All tests FAIL before implementation

2. GREEN (Implementation)
   ├─ src/data_io/display_names.py (actual module)
   ├─ src/r/load_display_names.R (R integration)
   └─ configs/mlflow_registry/display_names.yaml (data)

   → All tests PASS

3. REFACTOR (Optimize)
   ├─ Improve error messages
   ├─ Add logging
   ├─ Optimize paths
   └─ Tests still PASS
```

---

## MISSING FROM DOCUMENTATION

### 1. Fallback Strategy
"What happens if display_names.yaml is missing?" → Not documented
→ Should be in test comment

### 2. Versioning
YAML has `version: "1.0.0"` but no migration strategy
→ Add test for version compatibility

### 3. Performance Requirements
Should loading be < 1ms for interactivity?
→ Add benchmark test

### 4. Logging Strategy
When should warnings be logged?
→ Add test assertions on log output

---

## FINAL VERDICT

### ✅ POSITIVE ASPECTS
1. Clear problem statement
2. Good architecture choice (YAML alongside registry)
3. Sensible mapping rules
4. Caching for performance

### ❌ CRITICAL GAPS
1. Tests are incomplete stubs (not RED phase)
2. Implementation code in planning doc (wrong place)
3. Missing 12+ critical test cases
4. Error handling unspecified
5. Integration untested
6. Edge cases under-specified

### 🔴 RATING: FAIL

**Cannot proceed with implementation without:**

1. ✅ Expanding test file to include all 25 test cases (P1+P2)
2. ✅ Removing implementation code from planning document
3. ✅ Making all P1 tests FAIL before writing any code
4. ✅ Adding integration test that verifies cross-module interaction
5. ✅ Documenting fallback behavior for missing methods

---

## NEXT STEPS FOR USER

### If implementing this feature:

```bash
# STEP 1: Write comprehensive tests (RED phase)
# - Expand tests/unit/test_display_names.py with 25 test cases
# - Add tests/integration/test_display_names_*.py
# - All should FAIL

# STEP 2: Verify tests fail
pytest tests/unit/test_display_names.py -v
# Expected: 15+ FAILING tests

# STEP 3: Only then write implementation (GREEN phase)
# - Create src/data_io/display_names.py
# - Create configs/mlflow_registry/display_names.yaml
# - Create src/r/load_display_names.R

# STEP 4: Run tests again
pytest tests/unit/test_display_names.py -v
# Expected: 15+ PASSING tests

# STEP 5: Refactor
# - Improve error messages
# - Add logging
# - Optimize

# STEP 6: Final check
pytest tests/ -v  # All tests pass
```

---

## CONCLUSION

This is a **well-thought-out solution to a real problem**, but it **violates TDD by being incomplete**. The tests are not executable, implementation is in the planning doc, and critical edge cases are unspecified.

With 25+ test cases properly written and failing, the implementation will be straightforward and confidence-building. As written now, it's just a design document that could be copy-pasted wrong.

**Recommendation**: Convert to proper RED phase (25 failing tests) before implementation.

