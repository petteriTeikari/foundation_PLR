# fig-repo-91: How 7 Methods Become One Ensemble

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-91 |
| **Title** | How 7 Methods Become One Ensemble |
| **Complexity Level** | L3 |
| **Target Persona** | Research Scientist / ML Engineer |
| **Location** | `src/ensemble/README.md`, `docs/explanation/ensemble.md` |
| **Priority** | P3 (Medium) |

## Purpose

Document how the two ensemble outlier detection methods are constructed from base methods. The ensemble names in the registry encode their member methods (e.g., `ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune`), but developers need to understand the voting mechanism and the distinction between the full ensemble (7 methods, including traditional) and the thresholded ensemble (foundation models only).

## Key Message

Two ensemble variants exist: the full ensemble (7 base methods, majority voting) and the thresholded ensemble (4 foundation model methods, threshold voting with 2-or-more agreement). Both are registered as individual methods in the 11-method outlier registry.

## Content Specification

### Panel 1: Full Ensemble (7 Methods)

```
BASE METHODS (7)                              VOTING                  ENSEMBLE OUTPUT
═══════════════════════════════════           ═══════════            ═══════════════════

TRADITIONAL METHODS (4):
┌──────────────┐
│ LOF           │──┐
│ (Local Outlier│  │
│  Factor)      │  │
└──────────────┘  │
┌──────────────┐  │
│ OneClassSVM   │──┤
│               │  │      ┌─────────────────────────┐
└──────────────┘  │      │   MAJORITY VOTING         │     ┌───────────────────────────────┐
┌──────────────┐  ├──────│                           │─────│ ensemble-LOF-MOMENT-          │
│ PROPHET       │──┤      │   For each timepoint t:   │     │ OneClassSVM-PROPHET-SubPCA-   │
│               │  │      │   outlier(t) = True if     │     │ TimesNet-UniTS-gt-finetune    │
└──────────────┘  │      │     count(votes=True) ≥ 4  │     │                               │
┌──────────────┐  │      │     (strict majority of 7) │     │ Registry entry #10 of 11      │
│ SubPCA        │──┤      │                           │     └───────────────────────────────┘
│               │  │      └─────────────────────────┘
└──────────────┘  │
                  │
FOUNDATION MODEL METHODS (3):
┌──────────────┐  │
│ MOMENT-gt-    │──┤
│ finetune      │  │
└──────────────┘  │
┌──────────────┐  │
│ TimesNet-gt   │──┤
│               │  │
└──────────────┘  │
┌──────────────┐  │
│ UniTS-gt-     │──┘
│ finetune      │
└──────────────┘
```

### Panel 2: Thresholded Ensemble (FM-Only, 4 Methods)

```
FM-ONLY BASE METHODS (4):                    VOTING                  ENSEMBLE OUTPUT
═══════════════════════════════════          ═══════════            ═══════════════════

┌──────────────┐
│ MOMENT-gt-    │──┐
│ finetune      │  │
└──────────────┘  │      ┌─────────────────────────┐
┌──────────────┐  │      │  THRESHOLD VOTING         │     ┌───────────────────────────────┐
│ TimesNet-gt   │──├──────│                           │─────│ ensembleThresholded-MOMENT-   │
│               │  │      │  For each timepoint t:    │     │ TimesNet-UniTS-gt-finetune    │
└──────────────┘  │      │  outlier(t) = True if      │     │                               │
┌──────────────┐  │      │    count(votes=True) ≥ 2   │     │ Registry entry #11 of 11      │
│ UniTS-gt-     │──┤      │    (2 out of 4 agree)     │     └───────────────────────────────┘
│ finetune      │  │      │                           │
└──────────────┘  │      └─────────────────────────┘
┌──────────────┐  │
│ (MOMENT also  │──┘
│  contributes  │
│  via zeroshot │          Note: The thresholded ensemble uses
│  weighting)   │          a lower agreement threshold because
└──────────────┘          fewer methods are available.
```

### Panel 3: Method Categorization

```
THE 11 OUTLIER METHODS IN THE REGISTRY
═══════════════════════════════════════

  GROUND TRUTH (1):
  ┌────────────────────────────┐
  │ pupil-gt                    │  Human-annotated outlier mask
  └────────────────────────────┘

  FOUNDATION MODELS (4 individual):
  ┌────────────────────────────┐
  │ MOMENT-gt-finetune          │  Time-series FM, finetuned
  │ MOMENT-gt-zeroshot          │  Time-series FM, zero-shot
  │ UniTS-gt-finetune           │  Unified Time Series, finetuned
  │ TimesNet-gt                 │  TimesNet temporal 2D variation
  └────────────────────────────┘

  TRADITIONAL (4 individual):
  ┌────────────────────────────┐
  │ LOF                         │  Local Outlier Factor
  │ OneClassSVM                 │  One-Class SVM
  │ PROPHET                     │  Facebook Prophet decomposition
  │ SubPCA                      │  Subspace PCA
  └────────────────────────────┘

  ENSEMBLES (2):
  ┌────────────────────────────┐
  │ ensemble-LOF-MOMENT-...     │  Full: 7 methods, majority vote
  │ ensembleThresholded-...     │  FM-only: 4 methods, ≥2 agree
  └────────────────────────────┘

  Total: 1 + 4 + 4 + 2 = 11 methods ← MUST match registry count
```

### Panel 4: Why Ensembles

```
RATIONALE (conceptual, no performance numbers)
═════════════════════════════════════════════

  Individual methods have biases:
  ├── LOF: sensitive to local density → misses global patterns
  ├── MOMENT: learned patterns → may hallucinate in OOD regions
  └── PROPHET: assumes trend/seasonality → may miss impulse outliers

  Ensemble reduces individual bias:
  ├── Majority voting: outlier only if multiple detectors agree
  ├── False positive reduction: spurious detections filtered out
  └── Coverage: different methods catch different outlier types

  Two ensemble strategies:
  ├── Full (7): Maximum diversity, traditional + FM
  └── Thresholded (4): FM-only, for evaluating FM consensus
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `configs/mlflow_registry/parameters/classification.yaml` | Registry listing both ensemble names |
| `configs/registry_canary.yaml` | Expected count: 11 (includes both ensembles) |
| `configs/OUTLIER_MODELS/` | Individual method configs used by ensemble |

## Code Paths

| Module | Role |
|--------|------|
| `src/ensemble/ensemble_anomaly_detection.py` | Ensemble construction and voting logic |
| `src/ensemble/ensemble_utils.py` | Voting utilities (majority, threshold) |
| `src/ensemble/tasks_ensembling.py` | Prefect tasks for ensemble execution |
| `src/anomaly_detection/anomaly_detection.py` | Base method dispatch |
| `src/anomaly_detection/momentfm_outlier/` | MOMENT outlier detection |
| `src/anomaly_detection/units/` | UniTS outlier detection |
| `src/anomaly_detection/timesnet_wrapper.py` | TimesNet outlier detection |
| `src/anomaly_detection/outlier_sklearn.py` | LOF, OneClassSVM |
| `src/anomaly_detection/outlier_prophet.py` | PROPHET |
| `src/data_io/registry.py` | `get_valid_outlier_methods()` returns all 11 |

## Extension Guide

To add a new base method to the full ensemble:
1. Implement the base method (see fig-repo-86)
2. Update `src/ensemble/ensemble_anomaly_detection.py`:
   - Add new method to the `base_methods` list
   - Update voting threshold if needed (currently strict majority: `>= ceil(N/2)`)
3. Update ensemble name in registry (append new method name, alphabetical)
4. Update registry count: 11 -> 12 (the ensemble counts as one method)
5. Re-extract: `make extract` to regenerate ensemble results

To create a new ensemble variant:
1. Define member methods and voting strategy
2. Add to `src/ensemble/ensemble_anomaly_detection.py`
3. Register as new method in `configs/mlflow_registry/parameters/classification.yaml`
4. Update all 5 anti-cheat layers (registry count changes)

Note: This is a repo documentation figure - shows HOW ensembles are constructed, NOT their comparative performance.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "repo-91",
    "title": "How 7 Methods Become One Ensemble"
  },
  "content_architecture": {
    "primary_message": "Two ensemble variants: full (7 methods, majority vote) and thresholded (4 FM methods, >=2 agree). Both registered in the 11-method outlier registry.",
    "layout_flow": "Two-row layout: full ensemble (top) and thresholded ensemble (bottom), with method categorization sidebar",
    "spatial_anchors": {
      "full_ensemble": {"x": 0.05, "y": 0.05, "width": 0.6, "height": 0.4},
      "thresholded": {"x": 0.05, "y": 0.5, "width": 0.6, "height": 0.3},
      "categorization": {"x": 0.7, "y": 0.05, "width": 0.25, "height": 0.5},
      "rationale": {"x": 0.7, "y": 0.6, "width": 0.25, "height": 0.3}
    },
    "key_structures": [
      {
        "name": "Full Ensemble",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["7 methods", "Majority voting"]
      },
      {
        "name": "Thresholded Ensemble",
        "role": "foundation_model",
        "is_highlighted": true,
        "labels": ["4 FM methods", ">=2 agree"]
      },
      {
        "name": "Traditional Methods",
        "role": "traditional_method",
        "is_highlighted": false,
        "labels": ["LOF, OneClassSVM, PROPHET, SubPCA"]
      },
      {
        "name": "Foundation Models",
        "role": "foundation_model",
        "is_highlighted": false,
        "labels": ["MOMENT, TimesNet, UniTS"]
      }
    ],
    "callout_boxes": [
      {"heading": "REGISTRY", "body_text": "Both ensembles are individual entries in the 11-method registry. Their names encode member methods."},
      {"heading": "VOTING", "body_text": "Full: strict majority (>=4 of 7). Thresholded: lower bar (>=2 of 4) for FM consensus."}
    ]
  }
}
```

## Alt Text

Diagram showing two ensemble outlier detectors: full ensemble combining 7 methods via majority voting, and thresholded ensemble combining 4 foundation model methods with 2-or-more agreement. Method categorization shows traditional versus foundation model grouping.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
