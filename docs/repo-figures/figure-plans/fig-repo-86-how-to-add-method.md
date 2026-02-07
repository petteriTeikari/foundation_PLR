# fig-repo-86: Adding a New Outlier/Imputation Method: Step-by-Step

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-86 |
| **Title** | Adding a New Outlier/Imputation Method: Step-by-Step |
| **Complexity Level** | L3 |
| **Target Persona** | Research Scientist / ML Engineer |
| **Location** | `src/anomaly_detection/README.md`, `docs/contributing/` |
| **Priority** | P2 (High) |

## Purpose

Document the exact sequence of file changes required to add a new outlier detection or imputation method. The anti-cheat system (5-layer registry verification) means a developer cannot simply add code -- they must update multiple coordinated files. This figure prevents the common mistake of updating one file and getting pre-commit failures on the others.

## Key Message

Adding a new method requires synchronized changes in 6 places: registry YAML, config file, source module, tests, pipeline re-extraction, and anti-cheat layer updates. All 5 anti-cheat layers must agree on the new count.

## Content Specification

### Panel 1: Six-Step Checklist Flowchart

```
STEP 1: REGISTRY YAML (Source of Truth)
═══════════════════════════════════════════
File: configs/mlflow_registry/parameters/classification.yaml

  outlier_methods:
    count: 12                              ← was 11, now 12
    methods:
      - pupil-gt
      - MOMENT-gt-finetune
      - ...existing 10 methods...
      - NewMethod                          ← ADD HERE
              │
              ▼
STEP 2: CONFIG FILE (Hyperparameters)
═══════════════════════════════════════════
File: configs/OUTLIER_MODELS/NewMethod.yaml

  _target_: src.anomaly_detection.new_method.detect_outliers
  threshold: 0.5                            ← method-specific params
  window_size: 100
  model_path: null                          ← optional pretrained model
              │
              ▼
STEP 3: SOURCE MODULE (Implementation)
═══════════════════════════════════════════
Directory: src/anomaly_detection/new_method/

  src/anomaly_detection/
  ├── new_method/
  │   ├── __init__.py
  │   ├── new_method_main.py              ← detect_outliers(signal) → mask
  │   └── new_method_utils.py             ← helper functions
  │
  Interface contract:
    def detect_outliers(signal: np.ndarray, **kwargs) -> np.ndarray:
        """Returns boolean mask: True = outlier, False = clean."""
        ...
              │
              ▼
STEP 4: TESTS (Verify correctness)
═══════════════════════════════════════════
Files:
  tests/unit/test_new_method.py             ← Pure function tests
  tests/integration/test_anomaly_detection.py ← Add new method to integration

  # Unit test pattern:
  def test_new_method_returns_boolean_mask():
      signal = np.random.randn(1000)
      mask = detect_outliers(signal)
      assert mask.dtype == bool
      assert len(mask) == len(signal)

  def test_new_method_detects_known_outliers():
      signal = np.zeros(100)
      signal[50] = 100  # obvious outlier
      mask = detect_outliers(signal)
      assert mask[50] == True
              │
              ▼
STEP 5: RE-EXTRACT (Run pipeline)
═══════════════════════════════════════════
Commands:
  make extract           Block 1: MLflow → DuckDB (new method included)
  make analyze           Block 2: DuckDB → updated figures/stats

  Note: This requires the new method to have MLflow runs.
  If adding a new method to an existing experiment:
    1. Run the experiment with the new method first
    2. Then extract the results
              │
              ▼
STEP 6: UPDATE ANTI-CHEAT (5 layers MUST agree)
═══════════════════════════════════════════
All 5 layers must show the new count (12 for outlier methods):

  Layer 1: configs/registry_canary.yaml
           outlier_methods:
             expected_count: 12              ← was 11

  Layer 2: configs/mlflow_registry/parameters/classification.yaml
           (already updated in Step 1)

  Layer 3: src/data_io/registry.py
           EXPECTED_OUTLIER_COUNT = 12       ← was 11

  Layer 4: tests/test_registry.py
           assert len(methods) == 12         ← was 11

  Layer 5: .pre-commit-config.yaml
           - id: registry-integrity
             args: ['--expected-outlier', '12']  ← was 11
```

### Panel 2: Anti-Cheat Verification Flow

```
On every git commit, pre-commit runs:

  registry_canary.yaml  ──┐
  mlflow_registry/     ──┤
  registry.py          ──├── ALL MUST AGREE ── Pass ✓
  test_registry.py     ──┤                     or
  .pre-commit-config   ──┘                     Fail ✗

  If ANY layer disagrees → commit BLOCKED
  This prevents accidentally changing one file but not others.
```

### Panel 3: Imputation Method Variant

```
For IMPUTATION methods, the pattern is identical but uses different paths:

  Step 1: configs/mlflow_registry/parameters/classification.yaml
          imputation_methods: count: 9     ← was 8

  Step 2: configs/MODELS/NewImputation.yaml
          _target_: src.imputation.new_imputation.impute

  Step 3: src/imputation/new_imputation/
          def impute(signal, mask) -> np.ndarray:
              """Returns reconstructed signal."""

  Steps 4-6: Same pattern with updated counts (8 → 9)
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `configs/mlflow_registry/parameters/classification.yaml` | Master registry (add method + increment count) |
| `configs/registry_canary.yaml` | Anti-cheat reference (update count) |
| `configs/OUTLIER_MODELS/NewMethod.yaml` | New method hyperparameters |
| `configs/MODELS/NewImputation.yaml` | New imputation hyperparameters |
| `.pre-commit-config.yaml` | Pre-commit hook args (update count) |

## Code Paths

| Module | Role |
|--------|------|
| `src/anomaly_detection/` | Outlier detection implementations |
| `src/imputation/` | Imputation implementations |
| `src/data_io/registry.py` | `EXPECTED_OUTLIER_COUNT`, `EXPECTED_IMPUTATION_COUNT` |
| `src/data_io/registry.py` | `get_valid_outlier_methods()`, `validate_outlier_method()` |
| `tests/test_registry.py` | Count assertions for all method types |
| `scripts/verify_registry_integrity.py` | Cross-layer verification script |
| `src/ensemble/ensemble_anomaly_detection.py` | Ensemble construction (may need update) |

## Extension Guide

To add a new method type entirely (beyond outlier/imputation):
1. Define in `configs/mlflow_registry/parameters/classification.yaml`
2. Add `EXPECTED_{TYPE}_COUNT` to `src/data_io/registry.py`
3. Add validation function: `get_valid_{type}_methods()`, `validate_{type}_method()`
4. Add canary entry in `configs/registry_canary.yaml`
5. Add count assertion in `tests/test_registry.py`
6. Add pre-commit hook check in `.pre-commit-config.yaml`

Common mistakes:
- Forgetting to update the canary file (Layer 1) -- pre-commit will catch this
- Using a method name not in the registry -- `validate_outlier_method()` will raise ValueError
- Forgetting to update `EXPECTED_OUTLIER_COUNT` constant -- test_registry.py will fail

Note: This is a repo documentation figure - shows HOW to extend the codebase, NOT research results.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "repo-86",
    "title": "Adding a New Outlier/Imputation Method: Step-by-Step"
  },
  "content_architecture": {
    "primary_message": "Adding a new method requires synchronized changes in 6 places: registry YAML, config, source, tests, re-extraction, and anti-cheat updates.",
    "layout_flow": "Top-down 6-step vertical flowchart with anti-cheat verification panel on the right",
    "spatial_anchors": {
      "step1": {"x": 0.1, "y": 0.02},
      "step2": {"x": 0.1, "y": 0.17},
      "step3": {"x": 0.1, "y": 0.32},
      "step4": {"x": 0.1, "y": 0.47},
      "step5": {"x": 0.1, "y": 0.62},
      "step6": {"x": 0.1, "y": 0.77},
      "anticheat": {"x": 0.7, "y": 0.6, "width": 0.25, "height": 0.35}
    },
    "key_structures": [
      {
        "name": "Registry YAML",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Source of truth", "Step 1"]
      },
      {
        "name": "Anti-Cheat Layers",
        "role": "abnormal_warning",
        "is_highlighted": true,
        "labels": ["All 5 must agree"]
      },
      {
        "name": "detect_outliers()",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["Interface contract"]
      }
    ],
    "callout_boxes": [
      {"heading": "ALL 5 LAYERS MUST AGREE", "body_text": "Registry canary, YAML, Python constant, test assertion, and pre-commit hook must all show the same count."},
      {"heading": "INTERFACE CONTRACT", "body_text": "detect_outliers(signal) returns boolean mask. impute(signal, mask) returns reconstructed signal."}
    ]
  }
}
```

## Alt Text

Six-step vertical flowchart showing how to add a new outlier or imputation method: registry YAML, config file, source module, tests, re-extraction, and anti-cheat updates. Side panel shows 5-layer verification.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
