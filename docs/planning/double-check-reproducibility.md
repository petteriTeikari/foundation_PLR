# Double-Check Reproducibility Plan

## Status: PLANNING (AUDIT PHASE)

## Purpose

Audit and fix the entire DuckDB → JSON → ggplot2 → PNG pipeline to ensure **100% deterministic, idiotproof reproducibility** with guardrails that prevent ANY hallucination or guesswork.

---

## AUDIT RESULTS (2026-01-27)

### CRITICAL FINDINGS (Must Fix IMMEDIATELY)

| ID | Location | Issue | Reproducibility Risk |
|----|----------|-------|---------------------|
| **C1** | `src/viz/plot_config.py:202-214` | **HARDCODED AUROC values** in `KEY_STATS` dict | Stale values if MLflow re-run |
| **C2** | `src/viz/plot_config.py:84-100` | **5-path fallback** for DB selection | Silently loads wrong database |
| **C3** | Python figure scripts | **Method names not validated** against registry | Typos silently fall back |
| **C4** | `src/r/figures/*.R` | **HARDCODED numbers** instead of loading from JSON/YAML | Not reproducible |
| **C5** | `configs/VISUALIZATION/plot_hyperparam_combos.yaml:306-354` | **Raw hex colors** in `shap_figure_combos` instead of `color_ref` | Inconsistent with color_definitions |

### HIGH FINDINGS

| ID | Location | Issue |
|----|----------|-------|
| H1 | JSON exports | Incomplete metadata (no query params, row counts, script commit) |
| H2 | `figure_registry.yaml` | Registry NOT enforced - "broken" figures can still generate |
| H3 | `extract_all_configs_to_duckdb.py` | Hardcoded `CLASSIFIER_NAME_MAP` instead of from registry |
| H4 | DuckDB schema | No version tracking - schema changes silently ignored |
| H5 | `simple_baseline` combo | MISNOMER: uses MOMENT (FM) but name says "simple/baseline" |

### MEDIUM FINDINGS

| ID | Location | Issue |
|----|----------|-------|
| M1 | `figure_registry.yaml` | JSON privacy setting NOT enforced in code |
| M2 | Colors | DUPLICATED in 3 places: `colors.yaml`, `plot_hyperparam_combos.yaml`, `plot_config.py` |
| M3 | Featurization | Display names HARDCODED in Python, not in `display_names.yaml` |
| M4 | `load_display_names.R` | YAML re-parsed on every call, no caching |

---

## THE PROBLEM: CONFIGURATION AS DOCUMENTATION vs CONTRACT

**Current state:** YAML files exist but are treated as **suggestions**, not **enforced contracts**.

```
                    YAML Config
                        │
                        │ (should enforce)
                        ▼
    ┌─────────────────────────────────────────┐
    │  RUNTIME CODE                           │
    │                                         │
    │  ❌ Hardcoded values                    │
    │  ❌ Silent fallbacks                    │
    │  ❌ No validation against registry      │
    │  ❌ Multiple sources of truth           │
    └─────────────────────────────────────────┘
                        │
                        ▼
              Unreproducible Output
```

**Required state:** YAML configs are **executable contracts** that code MUST validate against.

```
                    YAML Config
                        │
                        │ (ENFORCED)
                        ▼
    ┌─────────────────────────────────────────┐
    │  RUNTIME VALIDATOR                      │
    │                                         │
    │  ✅ Loads ALL values from YAML/DB       │
    │  ✅ FAILS if config missing/invalid     │
    │  ✅ Validates against registry          │
    │  ✅ Single source of truth              │
    │  ✅ Logs provenance metadata            │
    └─────────────────────────────────────────┘
                        │
                        ▼
              Reproducible Output
```

---

## GUARDRAIL DESIGN

### Guardrail 1: NO HARDCODED VALUES IN CODE

**Rule:** Any value that could change (AUROC, method names, colors, paths) MUST come from YAML/DB.

**Enforcement:**

```python
# src/guardrails/no_hardcoded_values.py

BANNED_PATTERNS = [
    r'auroc\s*=\s*0\.\d+',           # Hardcoded AUROC
    r'["\'](pupil-gt|MOMENT-|LOF)',  # Hardcoded method names
    r'#[0-9A-Fa-f]{6}',              # Hardcoded hex colors
]

def scan_file_for_hardcoded_values(filepath):
    """Scan Python/R file for banned patterns."""
    content = filepath.read_text()
    violations = []
    for pattern in BANNED_PATTERNS:
        if re.search(pattern, content):
            violations.append((filepath, pattern))
    return violations
```

**Test:**
```bash
pytest tests/test_guardrails/test_no_hardcoded_values.py
```

### Guardrail 2: EXPLICIT PATH SELECTION

**Rule:** Database path MUST be explicitly configured, no fallback chains.

**Enforcement:**

```python
# src/guardrails/explicit_paths.py

def get_database_path():
    """Get database path from explicit config only."""
    # Priority 1: Environment variable
    if env_path := os.environ.get("FOUNDATION_PLR_DB_PATH"):
        path = Path(env_path)
        if not path.exists():
            raise FileNotFoundError(f"FOUNDATION_PLR_DB_PATH set but file not found: {path}")
        return path

    # Priority 2: Canonical location (ONE path, no fallback)
    canonical = Path("outputs/foundation_plr_results.db")
    if not canonical.exists():
        raise FileNotFoundError(
            f"Database not found at canonical location: {canonical}\n"
            "Either:\n"
            "  1. Set FOUNDATION_PLR_DB_PATH environment variable, OR\n"
            "  2. Run: python scripts/extract_all_configs_to_duckdb.py"
        )
    return canonical
```

### Guardrail 3: REGISTRY VALIDATION

**Rule:** All method names MUST exist in registry before use.

**Enforcement:**

```python
# src/guardrails/registry_validator.py

def validate_method_name(method_name: str, method_type: str) -> str:
    """Validate method name exists in registry, return display name."""
    registry = load_registry()

    valid_methods = registry.get(method_type, {})
    if method_name not in valid_methods:
        raise ValueError(
            f"GUARDRAIL VIOLATION: Unknown {method_type} method '{method_name}'\n"
            f"Valid methods: {list(valid_methods.keys())}\n"
            f"If this is a new method, add it to configs/mlflow_registry/{method_type}.yaml"
        )

    return valid_methods[method_name]["display_name"]
```

### Guardrail 4: JSON PROVENANCE METADATA

**Rule:** All JSON exports MUST include complete provenance metadata.

**Required fields:**
```json
{
  "metadata": {
    "created": "ISO8601 timestamp",
    "generator": {
      "script": "scripts/export_X.py",
      "git_commit": "abc123",
      "version": "1.0.0"
    },
    "data_source": {
      "database": "outputs/foundation_plr_results.db",
      "db_hash": "xxhash64 of file",
      "query": "SELECT ... (exact SQL)",
      "rows_before_filter": 410,
      "rows_after_filter": 10
    },
    "config_source": {
      "yaml_file": "configs/VISUALIZATION/plot_hyperparam_combos.yaml",
      "yaml_hash": "xxhash64 of file",
      "section_used": "standard_combos"
    }
  }
}
```

### Guardrail 5: COLOR CONSISTENCY

**Rule:** Colors defined in ONE place only, referenced everywhere else.

**Single source:** `configs/VISUALIZATION/colors.yaml`

**Enforcement:**
```python
# All code must use:
from src.viz.colors import get_color

color = get_color("ground_truth")  # Loads from YAML
# NOT: color = "#666666"
```

### Guardrail 6: R SCRIPTS LOAD FROM JSON

**Rule:** R scripts MUST NOT contain hardcoded metric values.

**Pattern:**
```r
# BANNED:
auroc <- 0.9110

# REQUIRED:
data <- load_validated_json("outputs/r_data/metrics.json")
auroc <- data$configs$ground_truth$auroc
```

---

## SINGLE SOURCES OF TRUTH

| Data Type | Source | Location |
|-----------|--------|----------|
| **Top-10 CatBoost configs** | DATABASE VIEW | `top10_catboost` in `foundation_plr_results.db` |
| **Method display names** | YAML | `configs/mlflow_registry/display_names.yaml` |
| **Colors** | YAML | `configs/VISUALIZATION/colors.yaml` |
| **Hyperparam combos** | YAML | `configs/VISUALIZATION/plot_hyperparam_combos.yaml` |
| **Figure specifications** | YAML | `configs/VISUALIZATION/figure_layouts.yaml` |
| **AUROC and metrics** | DATABASE | `essential_metrics` table |

**NOTHING computed, guessed, or hardcoded.**

---

## FIX PRIORITY ORDER

### Phase 1: CRITICAL (Blocks Everything)

1. **Remove hardcoded `KEY_STATS`** from `plot_config.py`
   - Load from YAML or query from DB
   - Add validator comparing to DB

2. **Remove 5-path fallback** for DB selection
   - Single canonical path
   - Fail loudly if not found

3. **Fix `shap_figure_combos`** raw hex colors
   - Change `color: "#666666"` to `color_ref: "--color-ground-truth"`

4. **Fix R hardcoded values**
   - Create JSON export for each R figure
   - R scripts load from JSON only

### Phase 2: HIGH (Needed for Full Reproducibility)

5. **Add JSON provenance metadata**
   - Update all export scripts
   - Include query, row counts, git commit

6. **Enforce figure registry at runtime**
   - Validator before figure generation
   - Block "broken" status figures

7. **Fix `simple_baseline` naming**
   - Rename or remove from `extended_combos`

### Phase 3: MEDIUM (Cleanup)

8. Consolidate color definitions to one file
9. Add featurization to `display_names.yaml`
10. Cache display names in R

---

## VALIDATION TESTS

### Test 1: No Hardcoded Values
```python
def test_no_hardcoded_auroc_in_python():
    """Scan all Python files for hardcoded AUROC values."""
    for py_file in Path("src").rglob("*.py"):
        content = py_file.read_text()
        # Allowed in test files only
        if "test_" in py_file.name:
            continue
        assert not re.search(r'auroc\s*=\s*0\.\d{3,}', content), \
            f"Hardcoded AUROC found in {py_file}"
```

### Test 2: All Combos From YAML
```python
def test_combos_loaded_from_yaml():
    """Verify no on-the-fly combo selection."""
    figure_scripts = Path("src/r/figures").glob("*.R")
    for script in figure_scripts:
        content = script.read_text()
        assert "configs[1:4]" not in content, f"Hardcoded slice in {script}"
        assert "head(configs" not in content, f"Hardcoded head() in {script}"
```

### Test 3: JSON Has Provenance
```python
def test_json_has_provenance():
    """All JSON exports must have metadata."""
    for json_file in Path("outputs/r_data").glob("*.json"):
        data = json.load(json_file.open())
        assert "metadata" in data, f"Missing metadata in {json_file}"
        assert "data_source" in data["metadata"], f"Missing data_source in {json_file}"
        assert "db_hash" in data["metadata"]["data_source"], f"Missing db_hash in {json_file}"
```

### Test 4: Colors From Single Source
```python
def test_colors_from_single_source():
    """All colors must reference colors.yaml."""
    # Check plot_hyperparam_combos.yaml has no raw hex in combo configs
    config = yaml.safe_load(open("configs/VISUALIZATION/plot_hyperparam_combos.yaml"))
    for combo in config.get("shap_figure_combos", {}).get("configs", []):
        assert "color" not in combo or combo["color"].startswith("--"), \
            f"Raw hex color in {combo['id']}, use color_ref instead"
```

---

## MAKEFILE TARGET

```makefile
validate-reproducibility:
	@echo "=== Phase 1: Check for hardcoded values ==="
	python -m pytest tests/test_guardrails/test_no_hardcoded_values.py -v
	@echo "=== Phase 2: Check JSON provenance ==="
	python -m pytest tests/test_guardrails/test_json_provenance.py -v
	@echo "=== Phase 3: Check YAML consistency ==="
	python -m pytest tests/test_guardrails/test_yaml_consistency.py -v
	@echo "=== Phase 4: Check R scripts ==="
	Rscript tests/test_r_no_hardcoded.R
	@echo "=== All reproducibility checks passed ==="
```

---

## AWAITING APPROVAL

Before execution:

1. **Phase 1 fixes** - Remove hardcoded values, fix paths, fix colors - OK?
2. **Guardrail tests** - Add pytest tests that block future violations - OK?
3. **`simple_baseline` resolution** - Rename or remove? - NEEDS USER INPUT
4. **JSON provenance enhancement** - Add git commit, query params - OK?

---

## APPENDIX: Full Audit Output

See: Agent audit (aa9d5a2) for complete file-by-file analysis.
