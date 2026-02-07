# fig-repo-73: 5-Layer Registry Verification: How Method Counts Stay Honest

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-73 |
| **Title** | 5-Layer Registry Verification: How Method Counts Stay Honest |
| **Complexity Level** | L4 |
| **Target Persona** | ML Engineer |
| **Location** | `configs/mlflow_registry/README.md`, `docs/explanation/registry-integrity.md` |
| **Priority** | P2 (High) |

## Purpose

Show how method counts (11 outlier, 8 imputation, 5 classifiers) are enforced by 5 independent verification layers that must ALL agree. Developers need to understand that changing a count in one place requires synchronized updates across all 5, and that tampering with one layer is caught by the others.

## Key Message

Five independent verification systems cross-check the same method counts (11/8/5). Tampering with one layer triggers failures in the remaining four.

## Content Specification

### Panel 1: The 5 Concentric Verification Layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 (outermost): CANARY FILE                                       │
│  configs/registry_canary.yaml                                            │
│  ├── expected_counts: {outlier: 11, imputation: 8, classifiers: 5}      │
│  ├── source_checksums: SHA256 of classification.yaml                     │
│  ├── invalid_methods: ["anomaly", "exclude", ...]                        │
│  └── frozen_for_publication: true                                        │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │  LAYER 2: REGISTRY YAML                                           │   │
│  │  configs/mlflow_registry/parameters/classification.yaml           │   │
│  │  ├── outlier_methods: [pupil-gt, MOMENT-gt-finetune, ...]  (11)  │   │
│  │  ├── imputation_methods: [pupil-gt, CSDI, SAITS, ...]      (8)   │   │
│  │  └── classifiers: [CatBoost, XGBoost, ...]                 (5)   │   │
│  │                                                                    │   │
│  │  ┌────────────────────────────────────────────────────────────┐   │   │
│  │  │  LAYER 3: PYTHON MODULE                                    │   │   │
│  │  │  src/data_io/registry.py                                   │   │   │
│  │  │  ├── EXPECTED_OUTLIER_COUNT = 11                           │   │   │
│  │  │  ├── EXPECTED_IMPUTATION_COUNT = 8                         │   │   │
│  │  │  ├── EXPECTED_CLASSIFIER_COUNT = 5                         │   │   │
│  │  │  └── get_valid_outlier_methods() → validates len == 11     │   │   │
│  │  │                                                             │   │   │
│  │  │  ┌─────────────────────────────────────────────────────┐   │   │   │
│  │  │  │  LAYER 4: PYTEST ASSERTIONS                          │   │   │   │
│  │  │  │  tests/test_registry.py                              │   │   │   │
│  │  │  │  ├── assert len(outlier_methods) == 11               │   │   │   │
│  │  │  │  ├── assert len(imputation_methods) == 8             │   │   │   │
│  │  │  │  ├── assert "anomaly" not in methods                 │   │   │   │
│  │  │  │  └── Cross-checks canary vs module constants         │   │   │   │
│  │  │  │                                                      │   │   │   │
│  │  │  │  ┌──────────────────────────────────────────────┐   │   │   │   │
│  │  │  │  │  LAYER 5 (innermost): PRE-COMMIT HOOK        │   │   │   │   │
│  │  │  │  │  .pre-commit-config.yaml                     │   │   │   │   │
│  │  │  │  │  ├── registry-integrity hook                 │   │   │   │   │
│  │  │  │  │  │   verify_registry_integrity.py            │   │   │   │   │
│  │  │  │  │  │   (cross-checks ALL 4 layers above)      │   │   │   │   │
│  │  │  │  │  └── registry-validation hook                │   │   │   │   │
│  │  │  │  │      pytest tests/test_registry.py           │   │   │   │   │
│  │  │  │  └──────────────────────────────────────────────┘   │   │   │   │
│  │  │  └─────────────────────────────────────────────────────┘   │   │   │
│  │  └────────────────────────────────────────────────────────────┘   │   │
│  └───────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 2: Tamper Detection Matrix

```
SCENARIO: What if you change the count in ONLY ONE layer?

┌─────────────────────────────────────────────────────────────────────┐
│  Tampered Layer        │ Detected By                                │
├────────────────────────┼────────────────────────────────────────────┤
│ Canary YAML            │ verify_registry_integrity.py (SHA mismatch)│
│ Registry YAML          │ Canary checksum mismatch + module count    │
│ Python constants       │ Canary cross-check + pytest assertions     │
│ Test assertions        │ verify_registry_integrity.py (test parse)  │
│ Pre-commit disabled    │ CI pipeline runs same checks independently │
└────────────────────────┴────────────────────────────────────────────┘

RESULT: Changing one requires changing ALL FIVE simultaneously
        → Single-file tampering is always caught
```

### Panel 3: Correct Update Workflow

```
"I need to add a 12th outlier method"

Step 1: classification.yaml        → Add method to outlier_methods list
Step 2: registry_canary.yaml       → Update count: 12, regenerate SHA256
Step 3: src/data_io/registry.py    → EXPECTED_OUTLIER_COUNT = 12
Step 4: tests/test_registry.py     → assert len(methods) == 12
Step 5: git commit (all 4 files)   → Pre-commit verifies consistency
                                      ↓
                              ALL 5 AGREE → commit succeeds
```

### Panel 4: Additional Safeguards

```
┌────────────────────────────────────────────────────────────────┐
│  SUPPLEMENTARY VERIFICATION TOOLS                               │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  scripts/verify_registry_integrity.py                           │
│  ├── Loads canary, registry YAML, Python module (AST), tests   │
│  ├── Cross-checks counts, checksums, invalid method lists      │
│  ├── --ci mode (stricter, used in GitHub Actions)              │
│  └── --fix mode (shows what needs updating)                     │
│                                                                  │
│  configs/_version_manifest.yaml                                 │
│  ├── Content hashes for all 68 config files                    │
│  ├── Detects silent config changes                              │
│  └── Generated by scripts/auto_version_configs.py              │
│                                                                  │
│  .github/workflows/config-integrity.yml                        │
│  └── CI job that runs verify_registry_integrity.py --ci        │
│                                                                  │
│  FROZEN COUNTS (publication):                                    │
│  frozen_for_publication: true in canary                          │
│  → Any count change during publication = HARD FAILURE           │
└────────────────────────────────────────────────────────────────┘
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `configs/registry_canary.yaml` | Layer 1: Reference counts, SHA256 checksum, invalid methods |
| `configs/mlflow_registry/parameters/classification.yaml` | Layer 2: YAML method definitions (single source of truth) |
| `configs/_version_manifest.yaml` | Content hashes for all 68 config files |

## Code Paths

| Module | Role |
|--------|------|
| `src/data_io/registry.py` | Layer 3: EXPECTED_*_COUNT constants, validation functions |
| `tests/test_registry.py` | Layer 4: pytest assertions on counts and invalid methods |
| `scripts/verify_registry_integrity.py` | Layer 5: Cross-checks all 4 layers (AST + YAML parsing) |
| `.pre-commit-config.yaml` | Hook definitions: registry-integrity, registry-validation |
| `.github/workflows/config-integrity.yml` | CI-level registry verification |
| `scripts/auto_version_configs.py` | Generates content hashes for version manifest |

## Extension Guide

To add a new method to the registry:
1. Add the method to `configs/mlflow_registry/parameters/classification.yaml`
2. Update `configs/registry_canary.yaml`: increment count, regenerate SHA256 (`sha256sum classification.yaml`)
3. Update `EXPECTED_*_COUNT` in `src/data_io/registry.py`
4. Update assertions in `tests/test_registry.py`
5. Commit ALL changes together (pre-commit verifies consistency)
6. If `frozen_for_publication: true` in canary, you must set it to `false` first (and justify the change)

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "repo-73",
    "title": "5-Layer Registry Verification: How Method Counts Stay Honest"
  },
  "content_architecture": {
    "primary_message": "Five independent verification systems cross-check method counts (11/8/5). Tampering with one layer triggers failures in the remaining four.",
    "layout_flow": "Concentric rings from outermost (canary) to innermost (pre-commit), with tamper detection matrix below",
    "spatial_anchors": {
      "concentric_layers": {"x": 0.5, "y": 0.35},
      "tamper_matrix": {"x": 0.5, "y": 0.7},
      "update_workflow": {"x": 0.5, "y": 0.9}
    },
    "key_structures": [
      {
        "name": "Layer 1: Canary YAML",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Reference counts", "SHA256 checksum"]
      },
      {
        "name": "Layer 2: Registry YAML",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Method definitions"]
      },
      {
        "name": "Layer 3: Python constants",
        "role": "primary_pathway",
        "is_highlighted": false,
        "labels": ["EXPECTED_*_COUNT"]
      },
      {
        "name": "Layer 4: pytest assertions",
        "role": "primary_pathway",
        "is_highlighted": false,
        "labels": ["assert len == 11"]
      },
      {
        "name": "Layer 5: Pre-commit hook",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Cross-checks all layers"]
      }
    ],
    "callout_boxes": [
      {"heading": "ANTI-CHEAT", "body_text": "Changing one layer without the others triggers verification failures in all remaining layers."}
    ]
  }
}
```

## Alt Text

Diagram of 5 concentric verification layers for method count integrity, from canary YAML to pre-commit hooks.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
