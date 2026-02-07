# PLR Decomposition Pipeline: Production-Grade Double-Checking Plan

**Status:** Ready for implementation
**Purpose:** Ensure decomposition pipeline is production-grade with excellent test coverage
**Cross-reference:** See `plr-decomposition-plan.md` for original implementation plan

---

## Executive Summary

This plan provides comprehensive validation for the PLR decomposition pipeline, ensuring:
1. **No hardcoded values** (colors, dimensions, method names)
2. **Excellent test coverage** (unit, integration, E2E)
3. **Data provenance verification** (no synthetic data per CRITICAL-FAILURE-001)
4. **Publication-quality figures** (DPI, colorblind-safe, dimensions)

---

## 1. Critical Issues Found in Current Code

### 1.1 Hardcoded Colors (HIGH PRIORITY)

**File:** `src/viz/fig_decomposition_grid.py`

| Lines | Issue | Fix Required |
|-------|-------|--------------|
| 67-77 | `COMPONENT_COLORS` dict with hex values | Load from `configs/VISUALIZATION/colors.yaml` |
| 116-120 | `color="#888888"`, `color="blue"`, `color="red"` | Use `COLORS` dict from `plot_config.py` |

**Violations:**
```python
# CURRENT (WRONG):
COMPONENT_COLORS = {
    "phasic": "#E69F00",      # Hardcoded
    "sustained": "#56B4E9",   # Hardcoded
    "pipr": "#009E73",        # Hardcoded
}

# REQUIRED (CORRECT):
from src.viz.plot_config import COLORS
COMPONENT_COLORS = {
    "phasic": COLORS["decomp_component_1"],
    "sustained": COLORS["decomp_component_2"],
    "pipr": COLORS["decomp_component_3"],
}
```

### 1.2 Hardcoded Method/Category Lists (MEDIUM)

**File:** `src/viz/fig_decomposition_grid.py` lines 34-57

```python
# CURRENT (WRONG):
METHOD_ORDER = ["template", "pca", "rotated_pca", "sparse_pca", "ged"]
CATEGORY_ORDER = ["Ground Truth", "Foundation Model", ...]

# REQUIRED (CORRECT):
# Load from configs/mlflow_registry/ or figure_registry.yaml
```

### 1.3 Fallback Mapping in Aggregation (MEDIUM)

**File:** `src/decomposition/aggregation.py` lines 42-53

The fallback category mapping should be removed - config should be **required**, not optional.

---

## 2. Test Coverage Checklist

### 2.1 Unit Tests (Decomposition Methods)

| Method | Test File | Tests Required | Status |
|--------|-----------|----------------|--------|
| Template Fitting | `test_template_fitting.py` | Stimulus timing edge cases, negative amplitudes | Partial |
| Standard PCA | `test_pca_methods.py` | Sign consistency across bootstrap | Good |
| Rotated PCA | `test_pca_methods.py` | Promax convergence edge cases | Good |
| Sparse PCA | `test_pca_methods.py` | Alpha=0 equals standard PCA, high alpha produces zeros | Partial |
| GED | `test_ged.py` | Minimal timepoints, regularization effect | Partial |

**Commands:**
```bash
# Run unit tests
pytest tests/test_decomposition/test_decomposition_methods.py -v

# With coverage
pytest tests/test_decomposition/ --cov=src/decomposition --cov-report=html
```

**Success Criteria:**
- [ ] All 5 decomposition methods have >90% line coverage
- [ ] Edge cases documented and tested
- [ ] No flaky tests (run 3x with different seeds)

### 2.2 Integration Tests (Aggregation)

**New File Required:** `tests/test_decomposition/test_aggregation.py`

```python
class TestDecompositionAggregator:
    def test_load_signals_by_category_returns_correct_shape(self):
        """Loading signals returns (n_subjects, n_timepoints) array."""

    def test_all_five_categories_have_data(self):
        """All 5 preprocessing categories have signals."""

    def test_bootstrap_ci_width_reasonable(self):
        """CI width is 5-30% of signal range."""

    def test_compute_all_decompositions_returns_25_results(self):
        """5 methods x 5 categories = 25 results."""
```

### 2.3 End-to-End Tests (Figure Generation)

**New File Required:** `tests/test_figure_generation/test_decomposition_grid.py`

```python
class TestDecompositionGridFigure:
    def test_figure_generates_without_error(self):
        """Figure generation completes without exceptions."""

    def test_figure_has_25_subplots(self):
        """Output has 5x5 = 25 subplots."""

    def test_json_data_saved(self):
        """Accompanying JSON data is saved."""

    def test_no_synthetic_marker_in_json(self):
        """JSON does not contain 'synthetic' key (unless test mode)."""
```

---

## 3. Data Validation Checklist

### 3.1 Schema Validation

**DuckDB Table:** `preprocessed_signals`

| Column | Type | Constraints |
|--------|------|-------------|
| `config_id` | INTEGER | NOT NULL |
| `outlier_method` | VARCHAR | NOT NULL |
| `imputation_method` | VARCHAR | NOT NULL |
| `preprocessing_category` | VARCHAR | IN ('Ground Truth', 'Foundation Model', 'Deep Learning', 'Traditional', 'Ensemble') |
| `subject_code` | VARCHAR | NOT NULL |
| `signal` | DOUBLE[] | Length ~ 200 |
| `time_vector` | DOUBLE[] | Range [0, 66] |

**Test:**
```python
def test_preprocessed_signals_schema():
    """Verify DuckDB table has correct schema."""
    conn = duckdb.connect(DB_PATH, read_only=True)

    # Check all 5 categories present
    categories = conn.execute(
        "SELECT DISTINCT preprocessing_category FROM preprocessed_signals"
    ).fetchall()
    actual = {c[0] for c in categories}
    expected = {"Ground Truth", "Foundation Model", "Deep Learning",
                "Traditional", "Ensemble"}
    assert actual == expected
```

### 3.2 Data Provenance (CRITICAL-FAILURE-001)

**Checks Required:**

| Check | Test | Expected |
|-------|------|----------|
| No synthetic data | `test_no_synthetic_markers()` | No "synthetic", "fake", "mock" keywords |
| Categories differ | `test_categories_have_different_waveforms()` | Cross-correlation < 0.99 |
| Bootstrap varies | `test_ci_widths_positive()` | CI width > 0 for all cells |

```python
def test_categories_have_different_mean_waveforms():
    """Different preprocessing categories produce different waveforms."""
    # Load mean waveforms for each category
    # Compute pairwise correlations
    # Assert all correlations < 0.99
```

---

## 4. Figure QA Checklist

### 4.1 Publication Standards

| Property | Required | How to Check |
|----------|----------|--------------|
| DPI | >= 300 | `PIL.Image.info.get("dpi")` |
| Width | 14 inches (4200px @ 300dpi) | `img.width >= 4000` |
| Height | 10 inches (3000px @ 300dpi) | `img.height >= 2800` |
| JSON data | Required | File exists |
| No synthetic | No "synthetic" key | JSON check |

### 4.2 Colorblind Accessibility

**Current Colors (Paul Tol palette - GOOD):**
| Component | Color | Hex | Status |
|-----------|-------|-----|--------|
| phasic/PC1 | Orange | #E69F00 | OK |
| sustained/PC2 | Sky blue | #56B4E9 | OK |
| pipr/PC3 | Bluish green | #009E73 | OK |

**BUT:** These must be moved to config, not hardcoded!

### 4.3 Visual Regression

```bash
# Generate golden images (once)
python src/viz/fig_decomposition_grid.py --test --output tests/golden/

# Compare against golden
pytest tests/test_figure_qa/test_visual_rendering.py -v
```

---

## 5. Code Fixes Required

### 5.1 Fix Hardcoded Colors in fig_decomposition_grid.py

**Step 1:** Add colors to `configs/VISUALIZATION/colors.yaml`:
```yaml
decomposition:
  component_1: "#E69F00"  # Orange (phasic/PC1)
  component_2: "#56B4E9"  # Sky blue (sustained/PC2)
  component_3: "#009E73"  # Bluish green (pipr/PC3)
  mean_waveform: "#888888"
  stimulus_blue: "rgba(0, 0, 255, 0.1)"
  stimulus_red: "rgba(255, 0, 0, 0.1)"
```

**Step 2:** Load in Python:
```python
from src.viz.plot_config import COLORS

COMPONENT_COLORS = {
    "phasic": COLORS["decomp_component_1"],
    "sustained": COLORS["decomp_component_2"],
    "pipr": COLORS["decomp_component_3"],
    "1": COLORS["decomp_component_1"],
    "2": COLORS["decomp_component_2"],
    "3": COLORS["decomp_component_3"],
}
```

### 5.2 Fix Hardcoded Method Lists

**Step 1:** Add to `configs/VISUALIZATION/figure_registry.yaml` (already done):
```yaml
grid_spec:
  rows: ["template", "pca", "rotated_pca", "sparse_pca", "ged"]
  columns: ["Ground Truth", "Foundation Model", "Deep Learning", "Traditional", "Ensemble"]
```

**Step 2:** Load in Python:
```python
import yaml
with open("configs/VISUALIZATION/figure_registry.yaml") as f:
    reg = yaml.safe_load(f)

grid_spec = reg["supplementary_figures"]["fig_decomposition_grid"]["grid_spec"]
METHOD_ORDER = grid_spec["rows"]
CATEGORY_ORDER = grid_spec["columns"]
```

### 5.3 Remove Fallback Mapping in aggregation.py

```python
# CURRENT (lines 38-53):
def load_category_mapping() -> dict[str, str]:
    if CATEGORY_MAPPING_PATH.exists():
        # load from config
    # Fallback mapping if file not found  <-- REMOVE THIS
    return {...}

# REQUIRED:
def load_category_mapping() -> dict[str, str]:
    if not CATEGORY_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Category mapping required: {CATEGORY_MAPPING_PATH}"
        )
    with open(CATEGORY_MAPPING_PATH) as f:
        return yaml.safe_load(f)
```

---

## 6. Validation Commands

```bash
# Full validation suite
pytest tests/test_decomposition/ -v
pytest tests/test_figure_qa/ -v -k decomposition

# Code quality (no hardcoded colors)
grep -n "#[0-9A-Fa-f]\{6\}" src/viz/fig_decomposition_grid.py
# Should return nothing after fixes

# Generate test figure
uv run python src/viz/fig_decomposition_grid.py --test

# Generate real figure (after extraction)
./scripts/generate_decomposition_figure.sh
```

---

## 7. Complete Pre-Flight Checklist

### Phase 1: Code Quality
- [ ] No hardcoded hex colors in `fig_decomposition_grid.py`
- [ ] No hardcoded method lists (load from config)
- [ ] No fallback mapping in `aggregation.py`
- [ ] Uses `save_figure()` not `plt.savefig()`

### Phase 2: Test Coverage
- [ ] Unit tests for all 5 decomposition methods
- [ ] Integration tests for aggregation module
- [ ] E2E tests for figure generation
- [ ] Data provenance tests (no synthetic data)

### Phase 3: Data Validation
- [ ] DuckDB has correct schema
- [ ] All 5 categories present
- [ ] Categories produce different waveforms (correlation < 0.99)
- [ ] CI widths are positive

### Phase 4: Figure QA
- [ ] DPI >= 300
- [ ] Dimensions match registry (14x10)
- [ ] All 25 cells have data
- [ ] JSON metadata saved
- [ ] Colorblind-safe palette verified

### Phase 5: Documentation
- [ ] Figure plan exists (`fig-repo-33-decomposition-grid.md`)
- [ ] Figure registered in `figure_registry.yaml`
- [ ] Runner script documented

---

## 8. Files Modified (COMPLETED)

| File | Changes | Status |
|------|---------|--------|
| `src/viz/fig_decomposition_grid.py` | Remove hardcoded colors, load from config | DONE |
| `src/decomposition/aggregation.py` | Remove fallback mapping, require config | DONE |
| `src/viz/plot_config.py` | Add decomposition colors to COLORS dict | DONE |
| `tests/test_decomposition/test_aggregation.py` | Create new integration tests (17 tests) | DONE |
| `tests/test_figure_generation/test_decomposition_grid.py` | Create new E2E tests (12 tests) | DONE |

## 9. Test Results (2026-02-01)

```
tests/test_decomposition/         - 51 passed, 6 skipped
tests/test_figure_generation/     - 12 passed, 2 skipped
Total: 63 passed, 8 skipped (skipped tests require real data)
```

---

## Distilled Essentials from Original Plan

From `plr-decomposition-plan.md`:

1. **5 Decomposition Methods:** Template Fitting, PCA, Rotated PCA (Promax), Sparse PCA, GED
2. **5 Preprocessing Categories:** Ground Truth, FM, DL, Traditional, Ensemble
3. **Data Source:** MLflow experiment 940304421003085572 (imputation artifacts)
4. **Output:** 5×5 grid figure with bootstrap 95% CIs
5. **Key Bug Fixed:** Original attempt used SIMULATED data, now uses REAL MLflow artifacts

**Critical Learning:** The bug where "subplots all look the same" was caused by synthetic data simulation. Tests must detect this pattern:
```python
def test_categories_differ():
    """Cross-category correlation < 0.99 proves real data."""
```
