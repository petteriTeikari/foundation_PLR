# Figure Data Flow Architecture: Single Source of Truth

## Problem Statement

Current state has **DUPLICATE DEFINITIONS** scattered across:
- `configs/VISUALIZATION/combos.yaml` - Combo definitions (version 1)
- `configs/VISUALIZATION/plot_hyperparam_combos.yaml` - Combo definitions (version 2, with colors)
- `scripts/export_predictions_for_r.py` lines 253-262 - **HARDCODED** combo mappings
- `scripts/export_predictions_for_r.py` lines 215-222 - **HARDCODED** fallback names
- R scripts - **HARDCODED** display names in `case_when()` blocks

This violates the **SINGLE SOURCE OF TRUTH** principle and causes:
1. Names can drift between sources
2. Changes require multiple file edits
3. Tests can't catch hardcoding without parsing all files

---

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SINGLE SOURCE OF TRUTH                                    │
│                                                                             │
│  configs/VISUALIZATION/combos.yaml                                          │
│  ├── standard_combos:                                                       │
│  │   ├── id: "ground_truth"                                                │
│  │   │   name: "Ground Truth"           ← DISPLAY NAME                     │
│  │   │   short_name: "GT"               ← LEGEND SHORT                     │
│  │   │   outlier_method: "pupil-gt"     ← DB FILTER                        │
│  │   │   imputation_method: "pupil-gt"  ← DB FILTER                        │
│  │   │   color_ref: "--color-ground-truth"  ← COLOR REF                    │
│  │   └── ...                                                               │
│  ├── extended_combos: [...]                                                 │
│  └── color_definitions:                                                     │
│      "--color-ground-truth": "#666666"                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Python reads YAML
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PYTHON EXPORT LAYER                                       │
│                                                                             │
│  scripts/export_predictions_for_r.py                                        │
│  scripts/export_data_for_r.py                                               │
│                                                                             │
│  1. Load combos from YAML:                                                  │
│     combos = yaml.safe_load(open("configs/VISUALIZATION/combos.yaml"))      │
│                                                                             │
│  2. Query DB using outlier_method + imputation_method from YAML             │
│                                                                             │
│  3. Write JSON with ALL metadata from YAML:                                 │
│     {                                                                       │
│       "metadata": {...},                                                    │
│       "data": {                                                             │
│         "configs": [                                                        │
│           {                                                                 │
│             "id": "ground_truth",           ← FROM YAML                    │
│             "name": "Ground Truth",         ← FROM YAML                    │
│             "short_name": "GT",             ← FROM YAML                    │
│             "color_ref": "--color-ground-truth",  ← FROM YAML              │
│             "roc": {"fpr": [...], "tpr": [...], "auroc": 0.911},           │
│             "rc": {"coverage": [...], "risk": [...], "aurc": 0.085}        │
│           },                                                               │
│           ...                                                               │
│         ]                                                                   │
│       }                                                                     │
│     }                                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ JSON files
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DATA LAYER (data/r_data/)                                 │
│                                                                             │
│  roc_rc_data.json          - ROC/RC curves with names, color_refs, metrics  │
│  calibration_data.json     - Calibration data with names, color_refs        │
│  predictions_top4.json     - Predictions with names, color_refs             │
│  dca_data.json             - DCA curves with names, color_refs              │
│                                                                             │
│  ALL JSON files contain:                                                    │
│  - id (for joining)                                                         │
│  - name (display name from YAML)                                            │
│  - short_name (for legends)                                                 │
│  - color_ref (resolved by R)                                                │
│  - Metric values computed from DB                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ R reads JSON + YAML (colors only)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    R FIGURE LAYER                                            │
│                                                                             │
│  src/r/figures/fig_roc_rc_combined.R                                        │
│                                                                             │
│  1. Load data with validation:                                              │
│     data <- validate_data_source("roc_rc_data.json")                        │
│                                                                             │
│  2. Names come FROM JSON (which came from YAML):                            │
│     legend_labels <- sapply(data$data$configs, function(cfg) {              │
│       sprintf("%s (AUROC: %.3f)", cfg$name, cfg$roc$auroc)                  │
│     })                                                                      │
│                                                                             │
│  3. Colors resolved from YAML:                                              │
│     color_defs <- load_color_definitions()                                  │
│     colors <- sapply(data$data$configs, function(cfg) {                     │
│       resolve_color(cfg$color_ref, color_defs)                              │
│     })                                                                      │
│                                                                             │
│  4. ZERO HARDCODING - cfg$name, cfg$color_ref all from JSON                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Rendered figure
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FIGURE OUTPUT                                             │
│                                                                             │
│  figures/generated/ggplot2/main/fig_roc_rc_combined.png                     │
│                                                                             │
│  Legend shows:                                                              │
│  "Ground Truth (AUROC: 0.911)"  ← name from YAML → JSON → R                │
│  "Ensemble + CSDI (AUROC: 0.913)"                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Tasks

### Phase 1: Consolidate YAML (CRITICAL)

**Task 1.1: Remove duplicate `plot_hyperparam_combos.yaml`**
- Merge unique content into `combos.yaml`
- Delete `plot_hyperparam_combos.yaml`
- Update all imports

**Task 1.2: Ensure `combos.yaml` has ALL required fields per combo:**
```yaml
- id: "ground_truth"           # Required: unique identifier
  name: "Ground Truth"         # Required: display name for legends
  short_name: "GT"             # Required: short form for cramped legends
  description: "..."           # Optional: for documentation
  outlier_method: "pupil-gt"   # Required: exact DB value
  imputation_method: "pupil-gt" # Required: exact DB value
  classifier: "CatBoost"       # Required: exact DB value
  color_ref: "--color-ground-truth"  # Required: color reference
  auroc: 0.9110               # Optional: expected value for verification
```

### Phase 2: Fix Python Export Scripts

**Task 2.1: Create `src/data_io/combo_loader.py`**
```python
"""
Load combo definitions from SINGLE SOURCE OF TRUTH.
GUARDRAIL: This is the ONLY place combos are loaded.
"""
import yaml
from pathlib import Path

COMBOS_PATH = Path(__file__).parent.parent.parent / "configs/VISUALIZATION/combos.yaml"

def load_all_combos() -> dict:
    """Load all combos from YAML."""
    with open(COMBOS_PATH) as f:
        return yaml.safe_load(f)

def get_standard_combos() -> list[dict]:
    """Get the 4 standard combos."""
    return load_all_combos()["standard_combos"]

def get_combo_by_id(combo_id: str) -> dict:
    """Get a specific combo by ID."""
    all_combos = load_all_combos()
    for combo in all_combos["standard_combos"] + all_combos["extended_combos"]:
        if combo["id"] == combo_id:
            return combo
    raise ValueError(f"Unknown combo: {combo_id}")

def get_combo_db_filter(combo_id: str) -> tuple[str, str]:
    """Get (outlier_method, imputation_method) for DB query."""
    combo = get_combo_by_id(combo_id)
    return combo["outlier_method"], combo["imputation_method"]
```

**Task 2.2: Refactor `export_predictions_for_r.py`**
- Remove hardcoded `_get_combo_filter()` (lines 251-262)
- Remove hardcoded `_generate_fallback_configs()` (lines 211-235)
- Use `combo_loader.get_combo_db_filter()` instead
- Include `name`, `short_name`, `color_ref` in JSON output

**Task 2.3: Refactor all export scripts to use `combo_loader`**

### Phase 3: Update JSON Schema

**Task 3.1: Define JSON schema for combo data**
```json
{
  "$schema": "...",
  "metadata": {
    "generated_at": "2026-01-28T...",
    "generator": "export_predictions_for_r.py",
    "combos_yaml_hash": "abc123..."  // For cache invalidation
  },
  "data": {
    "n_configs": 4,
    "configs": [
      {
        "id": "ground_truth",
        "name": "Ground Truth",
        "short_name": "GT",
        "color_ref": "--color-ground-truth",
        "outlier_method": "pupil-gt",
        "imputation_method": "pupil-gt",
        "roc": {
          "fpr": [...],
          "tpr": [...],
          "auroc": 0.911,
          "auroc_ci_lo": 0.85,
          "auroc_ci_hi": 0.95
        },
        "rc": {
          "coverage": [...],
          "risk": [...],
          "aurc": 0.085
        }
      }
    ]
  }
}
```

### Phase 4: Update R Scripts

**Task 4.1: Update `config_loader.R` to prefer JSON names**
```r
#' Get display names for configs from JSON data
#' Names come FROM JSON (which came from YAML) - NOT hardcoded!
get_config_names <- function(data) {
  sapply(data$data$configs, function(cfg) cfg$name)
}
```

**Task 4.2: Remove ALL hardcoded `case_when()` blocks**

Files to fix:
- `fig_raincloud_auroc.R` - Remove `case_when(grepl("pupil-gt"...))`
- `fig_multi_metric_raincloud.R` - Same
- `fig_cd_preprocessing.R` - Same

**Task 4.3: All R figure scripts follow pattern:**
```r
# Load data (names are IN the JSON)
data <- validate_data_source("roc_rc_data.json")

# Names from JSON (which came from YAML)
names <- sapply(data$data$configs, function(cfg) cfg$name)

# Colors from YAML via resolve_color
colors <- sapply(data$data$configs, function(cfg) {
  resolve_color(cfg$color_ref, load_color_definitions())
})

# ZERO hardcoding of "Ground Truth", "Ensemble", etc.
```

### Phase 5: Add Tests to Catch Hardcoding

**Task 5.1: Create `tests/test_no_hardcoded_names.py`**
```python
"""
Test that no figure scripts contain hardcoded combo names.
GUARDRAIL: Catches any attempt to hardcode "Ground Truth", etc.
"""
import re
from pathlib import Path

BANNED_STRINGS = [
    r'"Ground Truth"',
    r'"Ensemble \+ CSDI"',
    r'"MOMENT \+ SAITS"',
    r'"LOF \+ SAITS"',
    r'"Traditional"',
]

def test_r_scripts_no_hardcoded_names():
    """R figure scripts must not hardcode display names."""
    r_scripts = Path("src/r/figures").glob("*.R")

    for script in r_scripts:
        content = script.read_text()
        for pattern in BANNED_STRINGS:
            matches = re.findall(pattern, content)
            if matches:
                assert False, f"{script.name} contains hardcoded name: {matches}"

def test_python_export_no_hardcoded_names():
    """Python export scripts must not hardcode display names."""
    export_scripts = Path("scripts").glob("export_*.py")

    for script in export_scripts:
        content = script.read_text()
        # Allow in comments but not in code
        lines = [l for l in content.split('\n') if not l.strip().startswith('#')]
        code = '\n'.join(lines)
        for pattern in BANNED_STRINGS:
            matches = re.findall(pattern, code)
            if matches:
                assert False, f"{script.name} contains hardcoded name: {matches}"
```

**Task 5.2: Add pre-commit hook**
```yaml
- repo: local
  hooks:
    - id: no-hardcoded-combo-names
      name: No Hardcoded Combo Names
      entry: pytest tests/test_no_hardcoded_names.py -v
      language: system
      files: \.(py|R)$
```

---

## Verification Checklist

After implementation, verify:

- [ ] `configs/VISUALIZATION/combos.yaml` is the ONLY place combo names are defined
- [ ] `configs/VISUALIZATION/plot_hyperparam_combos.yaml` is DELETED
- [ ] All Python export scripts use `combo_loader.py`
- [ ] All JSON files in `data/r_data/` include `name`, `short_name`, `color_ref`
- [ ] All R figure scripts get names from JSON, not hardcoded
- [ ] `tests/test_no_hardcoded_names.py` passes
- [ ] Pre-commit hook catches any new hardcoding

---

## Data Flow Test

To verify the complete flow works:

```bash
# 1. Change a name in combos.yaml
sed -i 's/Ground Truth/Human Baseline/' configs/VISUALIZATION/combos.yaml

# 2. Regenerate JSON
python scripts/export_predictions_for_r.py

# 3. Regenerate figures
Rscript src/r/figures/generate_all_r_figures.R

# 4. Verify figure legends show "Human Baseline" (not "Ground Truth")
# If they still show "Ground Truth", there's hardcoding somewhere!

# 5. Revert
git checkout configs/VISUALIZATION/combos.yaml
```

---

## Summary

| Layer | Source of Truth | What It Contains |
|-------|-----------------|------------------|
| **YAML** | `combos.yaml` | names, color_refs, DB filters |
| **Python** | Reads YAML via `combo_loader.py` | NO hardcoding |
| **JSON** | Written by Python | names, color_refs, metric values |
| **R** | Reads JSON + YAML | names from JSON, colors from YAML |
| **Tests** | `test_no_hardcoded_names.py` | Catches violations |

**EVERYTHING flows from `combos.yaml`. NOTHING is hardcoded.**
