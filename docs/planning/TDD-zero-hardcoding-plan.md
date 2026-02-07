# TDD Plan: ZERO Hardcoding (Not Close to Zero - ZERO)

## Audit Summary

| Category | Count | Files |
|----------|-------|-------|
| `case_when()` regex categorization | 8 blocks | 7 R files |
| Method abbreviation hardcodes | 73 | `cd_diagram.R` |
| Hex color hardcodes | 100+ | 12+ R files, 5+ Python files |
| Display name hardcodes | 57 | Multiple R and Python files |

---

## TDD Execution Order

```
1. WRITE TESTS FIRST (all will FAIL)
2. Run tests → see RED (failures prove hardcoding exists)
3. Implement fixes one by one
4. Run tests → see GREEN (zero hardcoding achieved)
5. Never allow tests to regress
```

---

## PHASE 1: TEST SPECIFICATIONS (Write These FIRST)

### Test 1.1: No Hardcoded Display Names in R

**File**: `tests/test_no_hardcoding/test_r_display_names.py`

```python
"""
TDD Test: R files must NOT contain hardcoded display names.
This test MUST FAIL initially - proving hardcoding exists.
When it passes, hardcoding is ZERO.
"""
import re
from pathlib import Path
import pytest

# BANNED display name strings (from audit)
BANNED_DISPLAY_NAMES = [
    r'"Ground Truth"',
    r'"Ground truth"',
    r'"Foundation Model"',
    r'"Traditional"',
    r'"Ensemble"',
    r'"Deep Learning"',
    r'"MOMENT Fine-tuned"',
    r'"MOMENT Zeroshot"',
    r'"Ensemble \+ CSDI"',
    r'"MOMENT \+ SAITS"',
    r'"LOF \+ SAITS"',
    r'"Ground truth \+ Ground truth"',
]

# Files allowed to define display names (SINGLE SOURCE)
ALLOWED_FILES = [
    "configs/VISUALIZATION/",
    "configs/mlflow_registry/",
]

def get_r_files():
    """Get all R files that should NOT have hardcoded names."""
    r_files = list(Path("src/r").rglob("*.R"))
    # Filter out test files
    return [f for f in r_files if "test" not in str(f).lower()]

@pytest.mark.parametrize("pattern", BANNED_DISPLAY_NAMES)
def test_no_hardcoded_display_names_in_r(pattern):
    """Each banned pattern must NOT appear in R source files."""
    violations = []

    for r_file in get_r_files():
        content = r_file.read_text()
        # Skip comment lines
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            # Skip if line is a comment
            if line.strip().startswith('#'):
                continue
            matches = re.findall(pattern, line)
            if matches:
                violations.append(f"{r_file}:{i}: {line.strip()[:80]}")

    assert not violations, (
        f"HARDCODING DETECTED for pattern {pattern}:\n" +
        "\n".join(violations[:10]) +
        (f"\n... and {len(violations)-10} more" if len(violations) > 10 else "")
    )
```

### Test 1.2: No `case_when()` Categorization in R

**File**: `tests/test_no_hardcoding/test_r_no_case_when_categorization.py`

```python
"""
TDD Test: R files must NOT use case_when() for method categorization.
This pattern is the ROOT CAUSE of hardcoding drift.
"""
import re
from pathlib import Path
import pytest

# Pattern that detects case_when with grepl for categorization
BANNED_PATTERNS = [
    # case_when with grepl and display name assignment
    r'case_when\s*\([^)]*grepl\s*\([^)]*\)\s*~\s*"(Ground Truth|Traditional|Foundation Model|Ensemble|Deep Learning)"',
    # Direct pipeline_type or category assignment via case_when
    r'(pipeline_type|category)\s*=\s*case_when',
    # grepl with method name patterns followed by category
    r'grepl\s*\(\s*"(pupil-gt|MOMENT|UniTS|TimesNet|LOF|ensemble).*~\s*"',
]

def get_r_figure_files():
    """Get R figure files."""
    return list(Path("src/r/figures").rglob("*.R"))

@pytest.mark.parametrize("pattern", BANNED_PATTERNS)
def test_no_case_when_categorization(pattern):
    """case_when() must not be used for method categorization."""
    violations = []

    for r_file in get_r_figure_files():
        content = r_file.read_text()
        # Search across lines (case_when often spans multiple lines)
        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
        if matches:
            violations.append(f"{r_file.name}: contains banned case_when pattern")

    assert not violations, (
        f"CASE_WHEN CATEGORIZATION DETECTED:\n" +
        "\n".join(violations) +
        "\n\nUse load_method_category() from config instead!"
    )
```

### Test 1.3: No Hardcoded Hex Colors in R

**File**: `tests/test_no_hardcoding/test_r_no_hex_colors.py`

```python
"""
TDD Test: R files must NOT contain hardcoded hex colors.
All colors must come from YAML config.
"""
import re
from pathlib import Path
import pytest

# Hex color pattern
HEX_COLOR_PATTERN = r'["\']#[0-9A-Fa-f]{6}["\']'

# Files ALLOWED to define colors (SINGLE SOURCE)
ALLOWED_COLOR_DEFINITION_FILES = [
    "color_palettes.R",  # Will be refactored to load from YAML
    "theme_foundation_plr.R",  # Theme definition
]

def get_r_files_for_color_check():
    """Get R files that should NOT have hardcoded colors."""
    all_r = list(Path("src/r").rglob("*.R"))
    return [f for f in all_r
            if f.name not in ALLOWED_COLOR_DEFINITION_FILES
            and "test" not in str(f).lower()]

def test_no_hardcoded_hex_colors():
    """R figure files must not contain hardcoded hex colors."""
    violations = []

    for r_file in get_r_files_for_color_check():
        content = r_file.read_text()
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if line.strip().startswith('#'):  # Skip comments
                continue
            matches = re.findall(HEX_COLOR_PATTERN, line)
            if matches:
                violations.append(f"{r_file.name}:{i}: {matches}")

    assert not violations, (
        f"HARDCODED HEX COLORS DETECTED ({len(violations)} instances):\n" +
        "\n".join(violations[:20]) +
        (f"\n... and {len(violations)-20} more" if len(violations) > 20 else "") +
        "\n\nUse resolve_color('--color-xxx') instead!"
    )
```

### Test 1.4: No Hardcoded Display Names in Python

**File**: `tests/test_no_hardcoding/test_python_display_names.py`

```python
"""
TDD Test: Python files must NOT contain hardcoded display names.
"""
import re
from pathlib import Path
import pytest

BANNED_PATTERNS = [
    r'label\s*=\s*["\']Ground Truth["\']',
    r'label\s*=\s*["\']Ensemble["\']',
    r'["\']Ground Truth["\']',  # Any context
    r'\.replace\([^)]*["\']Ground Truth["\']',  # String replacement
]

# Allowed files (config loading, tests)
ALLOWED_PYTHON_FILES = [
    "test_",
    "conftest.py",
    "display_names.py",  # This is the loader
    "combo_loader.py",   # This is the loader
]

def get_python_files():
    """Get Python files that should NOT have hardcoded names."""
    all_py = list(Path("src").rglob("*.py")) + list(Path("scripts").rglob("*.py"))
    return [f for f in all_py
            if not any(allowed in f.name for allowed in ALLOWED_PYTHON_FILES)]

@pytest.mark.parametrize("pattern", BANNED_PATTERNS)
def test_no_hardcoded_display_names_python(pattern):
    """Python files must not hardcode display names."""
    violations = []

    for py_file in get_python_files():
        content = py_file.read_text()
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if line.strip().startswith('#'):
                continue
            if 'BANNED' in line or 'test' in line.lower():  # Skip test patterns
                continue
            matches = re.findall(pattern, line)
            if matches:
                violations.append(f"{py_file}:{i}")

    assert not violations, (
        f"HARDCODING DETECTED for {pattern}:\n" +
        "\n".join(violations[:10])
    )
```

### Test 1.5: Method Abbreviations from YAML Only

**File**: `tests/test_no_hardcoding/test_method_abbreviations.py`

```python
"""
TDD Test: Method abbreviations must come from YAML, not hardcoded.
"""
import re
from pathlib import Path
import pytest

def test_cd_diagram_no_hardcoded_abbreviations():
    """cd_diagram.R must not have 73 hardcoded abbreviations."""
    cd_diagram = Path("src/r/figure_system/cd_diagram.R")
    content = cd_diagram.read_text()

    # Count hardcoded abbreviation mappings (pattern: "xxx" = "YYY")
    abbreviation_pattern = r'"\S+"\s*=\s*"\S+"'
    matches = re.findall(abbreviation_pattern, content)

    # Allow at most 5 (for genuine fallbacks)
    max_allowed = 5

    assert len(matches) <= max_allowed, (
        f"cd_diagram.R has {len(matches)} hardcoded abbreviations (max allowed: {max_allowed}).\n"
        f"Move to configs/mlflow_registry/method_abbreviations.yaml!"
    )
```

### Test 1.6: YAML is Single Source of Truth

**File**: `tests/test_no_hardcoding/test_yaml_single_source.py`

```python
"""
TDD Test: Verify YAML files are the ONLY source of truth.
"""
import yaml
from pathlib import Path
import pytest

def test_combos_yaml_has_all_required_fields():
    """combos.yaml must have all required fields for each combo."""
    combos_path = Path("configs/VISUALIZATION/combos.yaml")
    config = yaml.safe_load(combos_path.read_text())

    required_fields = ["id", "name", "short_name", "outlier_method",
                       "imputation_method", "classifier"]

    for combo in config["standard_combos"]:
        for field in required_fields:
            assert field in combo, f"Combo {combo.get('id', '?')} missing {field}"

    for combo in config["extended_combos"]:
        for field in required_fields:
            assert field in combo, f"Combo {combo.get('id', '?')} missing {field}"

def test_display_names_yaml_has_all_methods():
    """display_names.yaml must define all 11 outlier and 8 imputation methods."""
    display_path = Path("configs/mlflow_registry/display_names.yaml")
    config = yaml.safe_load(display_path.read_text())

    # Expected counts from registry
    assert len(config["outlier_methods"]) >= 11, "Missing outlier method display names"
    assert len(config["imputation_methods"]) >= 8, "Missing imputation method display names"
    assert len(config["classifiers"]) >= 5, "Missing classifier display names"

def test_no_duplicate_combo_definitions():
    """There should be only ONE combos file, not two."""
    viz_config = Path("configs/VISUALIZATION")
    combo_files = list(viz_config.glob("*combo*.yaml"))

    # After consolidation, should only be combos.yaml
    assert len(combo_files) == 1, (
        f"Found {len(combo_files)} combo files: {[f.name for f in combo_files]}. "
        f"Should only have combos.yaml!"
    )
```

---

## PHASE 2: CREATE INFRASTRUCTURE (Before Fixing Violations)

### Task 2.1: Create `configs/mlflow_registry/method_abbreviations.yaml`

```yaml
# Method Abbreviations for CD Diagrams
# SINGLE SOURCE OF TRUTH - no hardcoding in R!

version: "1.0"

outlier_method_abbreviations:
  "pupil-gt": "GT"
  "MOMENT-gt-finetune": "MOM-ft"
  "MOMENT-gt-zeroshot": "MOM-zs"
  "MOMENT-orig-finetune": "MOM-o-ft"
  "UniTS-gt-finetune": "UniTS-ft"
  "UniTS-orig-finetune": "UniTS-o"
  "UniTS-orig-zeroshot": "UniTS-zs"
  "TimesNet-gt": "TN-gt"
  "TimesNet-orig": "TN-o"
  "LOF": "LOF"
  "OneClassSVM": "OCSVM"
  "SubPCA": "SubPCA"
  "PROPHET": "PROPHET"
  "ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune": "Ens-Full"
  "ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune": "Ens-Thresh"

imputation_method_abbreviations:
  "pupil-gt": "GT"
  "SAITS": "SAITS"
  "CSDI": "CSDI"
  "TimesNet": "TN"
  "MOMENT-finetune": "MOM-ft"
  "MOMENT-zeroshot": "MOM-zs"
  "linear": "Linear"
  "ensemble-CSDI-MOMENT-SAITS-TimesNet": "Ens"

# Combined pipeline abbreviations (for CD diagrams)
pipeline_abbreviations:
  "pupil-gt + pupil-gt": "GT+GT"
  "MOMENT-gt-finetune + SAITS": "MOM+SAITS"
  "LOF + SAITS": "LOF+SAITS"
  # ... etc
```

### Task 2.2: Create `configs/mlflow_registry/category_mapping.yaml`

```yaml
# Category Mapping for Methods
# SINGLE SOURCE - replaces all case_when() in R

version: "1.0"

# Map outlier method patterns to categories
outlier_method_categories:
  # Exact matches first
  "pupil-gt": "Ground Truth"

  # Pattern matches (processed in order)
  patterns:
    - pattern: "^ensemble"
      category: "Ensemble"
    - pattern: "MOMENT|UniTS|TimesNet"
      category: "Foundation Model"
    - pattern: ".*"  # Fallback
      category: "Traditional"

# Map imputation method patterns to categories
imputation_method_categories:
  "pupil-gt": "Ground Truth"

  patterns:
    - pattern: "^ensemble"
      category: "Ensemble"
    - pattern: "MOMENT"
      category: "Foundation Model"
    - pattern: "SAITS|CSDI|TimesNet"
      category: "Deep Learning"
    - pattern: ".*"
      category: "Traditional"
```

### Task 2.3: Create `src/r/figure_system/category_loader.R`

```r
#' Load category mapping from YAML
#' SINGLE SOURCE OF TRUTH - replaces ALL case_when() blocks!
#'
#' @export
load_category_mapping <- function() {
  project_root <- find_project_root()
  path <- file.path(project_root, "configs/mlflow_registry/category_mapping.yaml")
  yaml::read_yaml(path)
}

#' Get category for an outlier method
#' Uses YAML config, NOT hardcoded patterns!
#'
#' @param method Outlier method name
#' @return Category display name
#' @export
get_outlier_category <- function(method) {
  config <- load_category_mapping()

  # Check exact matches first
  if (method %in% names(config$outlier_method_categories)) {
    return(config$outlier_method_categories[[method]])
  }

  # Check patterns
  for (rule in config$outlier_method_categories$patterns) {
    if (grepl(rule$pattern, method)) {
      return(rule$category)
    }
  }

  return("Unknown")
}

#' Vectorized version for use in dplyr pipelines
#' @export
categorize_outlier_methods <- function(methods) {
  sapply(methods, get_outlier_category)
}
```

### Task 2.4: Consolidate YAML Files

**Delete**: `configs/VISUALIZATION/plot_hyperparam_combos.yaml`
**Keep**: `configs/VISUALIZATION/combos.yaml` (add missing fields from deleted file)

---

## PHASE 3: FIX VIOLATIONS (One by One Until Tests Pass)

### Fix 3.1: Replace ALL `case_when()` Categorizations

**Files to fix** (from audit):
1. `src/r/figures/fig_raincloud_auroc.R:71-76`
2. `src/r/figure_system/demo_usage.R:77-82, 97-103`
3. `src/r/figures/fig_fm_dashboard.R:72-76`
4. `src/r/figures/fig_specification_curve.R:60-72`
5. `src/r/figures/generate_all_r_figures.R:76-82, 95-102`
6. `src/r/figures/fig_cd_preprocessing.R:120-126`
7. `src/r/figures/fig_multi_metric_raincloud.R:56-62`

**Before** (BANNED):
```r
pipeline_type = case_when(
  grepl("pupil-gt", outlier_method) ~ "Ground Truth",
  grepl("^ensemble", outlier_method) ~ "Ensemble",
  grepl("MOMENT|UniTS|TimesNet", outlier_method) ~ "Foundation Model",
  TRUE ~ "Traditional"
)
```

**After** (CORRECT):
```r
source(file.path(PROJECT_ROOT, "src/r/figure_system/category_loader.R"))
data <- data %>%
  mutate(pipeline_type = categorize_outlier_methods(outlier_method))
```

### Fix 3.2: Replace cd_diagram.R Abbreviations

**Before** (73 hardcoded lines):
```r
abbreviate_method <- function(name) {
  abbrevs <- c(
    "pupil-gt" = "GT",
    "MOMENT-gt-finetune" = "MOMENT-ft",
    # ... 70 more lines
  )
  ...
}
```

**After** (loads from YAML):
```r
load_method_abbreviations <- function() {
  path <- file.path(find_project_root(),
                    "configs/mlflow_registry/method_abbreviations.yaml")
  yaml::read_yaml(path)
}

abbreviate_method <- function(name) {
  abbrevs <- load_method_abbreviations()

  # Check outlier abbreviations
  if (name %in% names(abbrevs$outlier_method_abbreviations)) {
    return(abbrevs$outlier_method_abbreviations[[name]])
  }

  # Check pipeline abbreviations
  if (name %in% names(abbrevs$pipeline_abbreviations)) {
    return(abbrevs$pipeline_abbreviations[[name]])
  }

  # Fallback: truncate
  return(substr(name, 1, 15))
}
```

### Fix 3.3: Replace Hardcoded Hex Colors

**Before** (BANNED):
```r
colors <- c(
  "Ground Truth" = "#FFD700",
  "Ensemble" = "#006BA2",
  "Foundation Model" = "#3EBCD2",
  "Traditional" = "#999999"
)
```

**After** (CORRECT):
```r
source(file.path(PROJECT_ROOT, "src/r/figure_system/config_loader.R"))
colors <- get_category_colors()  # Loads from YAML
```

### Fix 3.4: Python Display Names

**Before** (BANNED):
```python
label = "Ground Truth"
```

**After** (CORRECT):
```python
from src.data_io.display_names import get_category_display_name
label = get_category_display_name("ground_truth")
```

---

## PHASE 4: VERIFICATION

### Run All Tests

```bash
# Run hardcoding tests - ALL must pass
pytest tests/test_no_hardcoding/ -v

# Expected output after all fixes:
# test_r_display_names.py::test_no_hardcoded_display_names_in_r[pattern0] PASSED
# test_r_display_names.py::test_no_hardcoded_display_names_in_r[pattern1] PASSED
# ...
# test_r_no_case_when_categorization.py::test_no_case_when_categorization PASSED
# test_r_no_hex_colors.py::test_no_hardcoded_hex_colors PASSED
# test_python_display_names.py::test_no_hardcoded_display_names_python PASSED
# test_method_abbreviations.py::test_cd_diagram_no_hardcoded_abbreviations PASSED
# test_yaml_single_source.py::test_combos_yaml_has_all_required_fields PASSED
# test_yaml_single_source.py::test_no_duplicate_combo_definitions PASSED
```

### Data Flow Verification Test

```bash
# This test proves the entire flow works:

# 1. Change a name in YAML
sed -i 's/"Ground Truth"/"Human Baseline"/g' configs/mlflow_registry/display_names.yaml

# 2. Regenerate JSON
python scripts/export_predictions_for_r.py

# 3. Regenerate figures
Rscript src/r/figures/generate_all_r_figures.R

# 4. Verify figure legends show "Human Baseline"
# If any figure shows "Ground Truth", TEST FAILED

# 5. Revert
git checkout configs/
```

---

## EXECUTION CHECKLIST

### Step 1: Write Tests (Day 1)
- [ ] Create `tests/test_no_hardcoding/` directory
- [ ] Write `test_r_display_names.py`
- [ ] Write `test_r_no_case_when_categorization.py`
- [ ] Write `test_r_no_hex_colors.py`
- [ ] Write `test_python_display_names.py`
- [ ] Write `test_method_abbreviations.py`
- [ ] Write `test_yaml_single_source.py`
- [ ] Run tests → ALL MUST FAIL (proves hardcoding exists)

### Step 2: Create Infrastructure (Day 1)
- [ ] Create `method_abbreviations.yaml`
- [ ] Create `category_mapping.yaml`
- [ ] Create `category_loader.R`
- [ ] Consolidate combos YAML files (delete duplicate)

### Step 3: Fix R Files (Day 2)
- [ ] Fix `fig_raincloud_auroc.R` (case_when + colors)
- [ ] Fix `fig_multi_metric_raincloud.R` (case_when)
- [ ] Fix `fig_cd_preprocessing.R` (case_when + colors)
- [ ] Fix `fig_specification_curve.R` (case_when + colors)
- [ ] Fix `fig_fm_dashboard.R` (case_when + colors)
- [ ] Fix `fig_shap_gt_vs_ensemble.R` (case_when + colors)
- [ ] Fix `generate_all_r_figures.R` (case_when)
- [ ] Fix `demo_usage.R` (case_when)
- [ ] Fix `cd_diagram.R` (73 abbreviations)
- [ ] Fix `color_palettes.R` (load from YAML)
- [ ] Fix `figure_factory.R` (fallback colors)
- [ ] Fix `common.R` (fallback colors)

### Step 4: Fix Python Files (Day 2)
- [ ] Fix `generate_shap_figures.py`
- [ ] Fix `figure_and_stats_generation.py`
- [ ] Fix `individual_subject_traces.py`
- [ ] Fix `foundation_model_dashboard.py`
- [ ] Fix `factorial_matrix.py`

### Step 5: Verify (Day 3)
- [ ] Run `pytest tests/test_no_hardcoding/ -v` → ALL PASS
- [ ] Run data flow verification test
- [ ] Add pre-commit hook

---

## SUCCESS CRITERIA

**ZERO means ZERO:**

| Metric | Target |
|--------|--------|
| Hardcoded display names in R | 0 |
| Hardcoded display names in Python | 0 |
| `case_when()` categorization blocks | 0 |
| Hardcoded hex colors in figure files | 0 |
| Method abbreviation hardcodes | 0 (max 5 fallbacks) |
| Duplicate YAML definition files | 0 |
| Failing hardcoding tests | 0 |

**If ANY test fails, we have NOT achieved ZERO.**
