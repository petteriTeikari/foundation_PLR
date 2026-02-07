# fig-repo-96: Abstain When Uncertain: Risk-Coverage Analysis

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-96 |
| **Title** | Abstain When Uncertain: Risk-Coverage Analysis |
| **Complexity Level** | L3 |
| **Target Persona** | Biostatistician / Research Scientist |
| **Location** | `src/stats/README.md`, `docs/explanation/advanced-analyses.md` |
| **Priority** | P3 (Medium) |

## Purpose

Explain the selective classification concept implemented in `src/stats/decision_uncertainty.py`. In clinical screening, a model that says "I don't know" for uncertain cases (referring them to a specialist) can be more valuable than one that forces a binary decision for every patient. This figure shows the conceptual framework without any performance numbers.

## Key Message

Selective classification trades coverage for accuracy: by abstaining on uncertain predictions, the model's error rate on accepted cases decreases. AURC (Area Under Risk-Coverage curve) quantifies this trade-off.

## Content Specification

### Panel 1: The Selective Classification Concept

```
+----------------------------------------------------------------------+
|  SELECTIVE CLASSIFICATION: "I Don't Know" Is a Valid Answer            |
+----------------------------------------------------------------------+
|                                                                        |
|  Standard Classification:                                              |
|  All 208 subjects --> forced binary decision (glaucoma / control)     |
|                       Every subject gets a label, even uncertain ones  |
|                                                                        |
|  Selective Classification:                                             |
|  All 208 subjects --> predictions sorted by confidence                 |
|  |                                                                     |
|  +----[HIGH CONFIDENCE]----+----------[LOW CONFIDENCE]--------+       |
|  |                         |                                   |       |
|  v                         |                                   v       |
|  ACCEPT                    |                             REJECT        |
|  Model makes a             |                             "I don't know"|
|  definitive prediction     |                             Refer to      |
|  (glaucoma or control)     |                             specialist    |
|  |                         |                                   |       |
|  v                         v                                   v       |
|  Lower error rate          The threshold determines          Patient   |
|  on accepted cases         the split point                   safety    |
|                                                              preserved |
+----------------------------------------------------------------------+
```

### Panel 2: Decision Uncertainty (DU) Per Subject

```
For each subject, across 1000 bootstrap predictions:

  Subject A (certain):
  |||||||||||||||||||||||||||||||||||||||||||||| all above threshold
  DU = min(P(above), P(below)) = min(1.0, 0.0) = 0.0  (certain)

  Subject B (uncertain):
  |||||||||||||||||||  above     |||||||||||||| below
  DU = min(P(above), P(below)) = min(0.55, 0.45) = 0.45  (uncertain)

  Subject C (very uncertain):
  ||||||||||| above  ||||||||||||| below
  DU = min(P(above), P(below)) = min(0.48, 0.52) = 0.48  (near max)

  DU Range: [0.0, 0.5]
  DU = 0.0  --> Complete certainty about treatment decision
  DU = 0.5  --> Maximum uncertainty (50/50 chance of crossing threshold)
```

### Panel 3: Risk-Coverage Curve (Conceptual)

```
  Risk (error rate)
  ^
  |
  |  *
  |   *
  |    *
  |     **
  |       ***
  |          ****
  |              ******
  |                    **********
  |                              **********************
  +-----------------------------------------------------> Coverage (%)
  0%    20%    40%    60%    80%    100%

  Coverage = fraction of subjects the model makes a prediction for
  Risk = error rate on the accepted (predicted) subjects

  AURC = Area Under this curve (shaded area)
  Lower AURC = better selective classification

  At 100% coverage: risk = overall error rate (standard classification)
  At 0% coverage:   risk = 0 (model predicts nothing, no errors)

  The interesting region: where does risk drop sharply?
  That tells you how many subjects you can safely classify.
```

### Panel 4: Clinical Interpretation

```
+----------------------------------------------------------------------+
|  CLINICAL CONTEXT                                                      |
|                                                                        |
|  Glaucoma screening scenario:                                          |
|                                                                        |
|  208 patients attend a PLR-based screening clinic                     |
|                                                                        |
|  Without selective classification:                                     |
|    All 208 get a "glaucoma" or "control" label                        |
|    Some borderline cases get wrong labels                              |
|                                                                        |
|  With selective classification:                                        |
|    ~170 patients: confident prediction (accept)                       |
|    ~38 patients:  uncertain (reject, refer to ophthalmologist)        |
|                                                                        |
|  The rejected patients are those near the decision boundary            |
|  where bootstrap predictions straddle the threshold.                   |
|                                                                        |
|  Trade-off: fewer automated decisions, but much higher accuracy       |
|  on the decisions that ARE made.                                       |
+----------------------------------------------------------------------+
```

### Panel 5: Implementation Architecture

```
src/stats/decision_uncertainty.py
|
+-- decision_uncertainty(bootstrap_samples, threshold) --> float
|     Per-subject DU: min(P(p > t), P(p < t))
|
+-- decision_uncertainty_per_subject(bootstrap_matrix, threshold) --> array
|     Vectorized: (n_subjects, n_bootstrap) --> (n_subjects,)
|
+-- decision_uncertainty_summary(bootstrap_matrix, threshold, du_threshold=0.3) --> dict
      Returns: mean_du, median_du, std_du, pct_above_threshold, n_uncertain

Config: Threshold sweep from 0.0 to 1.0
        du_threshold = 0.3 (default for "high uncertainty" classification)
        bootstrap_matrix shape: (208 subjects, 1000 bootstraps)
```

## Spatial Anchors

```yaml
layout_flow: "Top-down: concept, DU per subject, risk-coverage curve, clinical context, code"
spatial_anchors:
  concept:
    x: 0.5
    y: 0.15
    content: "Accept (high confidence) vs Reject (uncertain)"
  du_per_subject:
    x: 0.5
    y: 0.35
    content: "DU = min(P(above), P(below)) for each subject"
  risk_coverage:
    x: 0.5
    y: 0.55
    content: "Conceptual risk-coverage curve with AURC"
  clinical:
    x: 0.5
    y: 0.75
    content: "Glaucoma screening interpretation"
  implementation:
    x: 0.5
    y: 0.92
    content: "Python API and config"
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `configs/CLS_EVALUATION.yaml` | `BOOTSTRAP.n_iterations=1000` (bootstrap sample count) |
| `configs/CLS_EVALUATION.yaml` | `glaucoma_params.prevalence=0.0354` (disease prevalence) |
| `src/stats/_defaults.py` | Default constants (`N_BOOTSTRAP`, `CI_LEVEL`) |

## Code Paths

| Module | Role |
|--------|------|
| `src/stats/decision_uncertainty.py` | `decision_uncertainty()`, `decision_uncertainty_per_subject()`, `decision_uncertainty_summary()` |
| `src/stats/_exceptions.py` | `ValidationError` for input validation |
| `src/stats/classifier_metrics.py` | Bootstrap loop that generates the `(n_subjects, n_bootstrap)` prediction matrix |

## Extension Guide

To add a new selective classification metric:
1. Implement the metric function in `src/stats/decision_uncertainty.py`
2. Add to `__all__` list for public API
3. Include input validation using `ValidationError` from `_exceptions.py`
4. Add extraction to `src/data_io/streaming_duckdb_export.py` (if storing in DuckDB)
5. Register display name in `src/viz/metric_registry.py`
6. Add unit test in `tests/unit/test_decision_uncertainty.py`

References:
- Barrenada et al. (2025). The fundamental problem of providing uncertainty in individual risk predictions. BMJ Medicine.
- Geifman & El-Yaniv (2017). Selective prediction: A cost-effective method for confident deep learning.

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-96",
    "title": "Abstain When Uncertain: Risk-Coverage Analysis"
  },
  "content_architecture": {
    "primary_message": "Selective classification trades coverage for accuracy: by abstaining on uncertain predictions, the model's error rate on accepted cases decreases.",
    "layout_flow": "Top-down: concept, DU per subject, risk-coverage curve, clinical context, implementation",
    "spatial_anchors": {
      "concept": {"x": 0.5, "y": 0.15},
      "du_per_subject": {"x": 0.5, "y": 0.35},
      "risk_coverage": {"x": 0.5, "y": 0.55},
      "clinical": {"x": 0.5, "y": 0.75},
      "implementation": {"x": 0.5, "y": 0.92}
    },
    "key_structures": [
      {
        "name": "Accept/Reject Split",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["ACCEPT (high confidence)", "REJECT (uncertain)", "Refer to specialist"]
      },
      {
        "name": "Decision Uncertainty Formula",
        "role": "highlight_accent",
        "is_highlighted": true,
        "labels": ["DU = min(P(above), P(below))", "Range [0.0, 0.5]"]
      },
      {
        "name": "Risk-Coverage Curve",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["AURC = area under curve", "Lower = better"]
      },
      {
        "name": "Clinical Context",
        "role": "callout_box",
        "is_highlighted": false,
        "labels": ["208 patients", "~170 accept", "~38 refer"]
      }
    ],
    "callout_boxes": [
      {"heading": "KEY CONCEPT", "body_text": "A model that says 'I don't know' for uncertain cases can be more valuable than one that forces a decision for every patient."},
      {"heading": "METRIC", "body_text": "AURC (Area Under Risk-Coverage curve): lower = better selective classification."}
    ]
  }
}
```

## Alt Text

Conceptual diagram of selective classification: subjects sorted by prediction confidence into accept (high) and reject (uncertain) groups, with a risk-coverage curve showing AURC.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
