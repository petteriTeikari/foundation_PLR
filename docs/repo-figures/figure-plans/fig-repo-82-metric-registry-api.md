# fig-repo-82: MetricRegistry: The Code That Knows Every Metric

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-82 |
| **Title** | MetricRegistry: The Code That Knows Every Metric |
| **Complexity Level** | L4 |
| **Target Persona** | ML Engineer |
| **Location** | `src/viz/README.md`, `docs/explanation/metrics.md` |
| **Priority** | P3 (Medium) |

## Purpose

Show the class architecture of MetricRegistry from `src/viz/metric_registry.py`, including the MetricDefinition dataclass, the registry's class methods, the STRATOS metric sets, and how visualization modules use it. Developers need to understand how to look up metric metadata without computing metrics, and how STRATOS groupings are defined.

## Key Message

MetricRegistry is a singleton-like class that groups all STRATOS metrics by domain and provides display names, formatting, DuckDB column lookups, and validation. Visualization code reads metadata only; compute functions are reserved for extraction.

## Content Specification

### Panel 1: Class Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│              MetricRegistry CLASS ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  @dataclass                                                             │
│  MetricDefinition                                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  name: str                  # Internal key (e.g., "auroc")        │  │
│  │  display_name: str          # For plots (e.g., "AUROC")           │  │
│  │  higher_is_better: bool     # True = higher is better             │  │
│  │  unit: str                  # e.g., "%", "ms"                     │  │
│  │  format_str: str            # e.g., ".3f", ".1%"                  │  │
│  │  value_range: (float,float) # Expected (min, max)                 │  │
│  │  compute_fn: Callable       # EXTRACTION ONLY (banned in viz)     │  │
│  │                                                                    │  │
│  │  Methods:                                                          │  │
│  │  ├── format_value(value) → str     # "0.911"                     │  │
│  │  └── is_better(a, b) → bool        # a > b if higher_is_better   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  class MetricRegistry                                                   │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  _metrics: Dict[str, MetricDefinition]     (class-level store)    │  │
│  │                                                                    │  │
│  │  @classmethod register(metric)      # Add a metric definition     │  │
│  │  @classmethod get(name) → def       # Get metric (KeyError if     │  │
│  │                                     #   not found)                │  │
│  │  @classmethod get_or_default(name)  # Get metric or create       │  │
│  │                                     #   default from name         │  │
│  │  @classmethod list_metrics() → list # All registered metric names │  │
│  │  @classmethod has(name) → bool      # Check if metric exists     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 2: STRATOS Metric Sets

```
┌─────────────────────────────────────────────────────────────────────────┐
│  STRATOS_METRIC_SETS (Van Calster 2024)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  "stratos_core" (MANDATORY for all reporting):                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  auroc                    Discrimination                          │  │
│  │  brier                    Overall performance                     │  │
│  │  scaled_brier             Overall (interpretable, IPA)            │  │
│  │  calibration_slope        Calibration (weak)                      │  │
│  │  calibration_intercept    Calibration (mean)                      │  │
│  │  o_e_ratio                Calibration (mean)                      │  │
│  │  net_benefit              Clinical utility                        │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  "discrimination":                                                      │
│  ┌────────────────────────────────┐                                     │
│  │  auroc, sensitivity,           │                                     │
│  │  specificity                   │                                     │
│  └────────────────────────────────┘                                     │
│                                                                          │
│  "calibration":                                                         │
│  ┌────────────────────────────────┐                                     │
│  │  brier, scaled_brier,          │                                     │
│  │  calibration_slope,            │                                     │
│  │  calibration_intercept,        │                                     │
│  │  o_e_ratio                     │                                     │
│  └────────────────────────────────┘                                     │
│                                                                          │
│  "clinical_utility":                                                    │
│  ┌────────────────────────────────┐                                     │
│  │  net_benefit                   │                                     │
│  └────────────────────────────────┘                                     │
│                                                                          │
│  "outlier_detection":                                                   │
│  ┌────────────────────────────────┐                                     │
│  │  outlier_f1, outlier_precision,│                                     │
│  │  outlier_recall                │                                     │
│  └────────────────────────────────┘                                     │
│                                                                          │
│  "imputation":                                                          │
│  ┌────────────────────────────────┐                                     │
│  │  mae, rmse                     │                                     │
│  └────────────────────────────────┘                                     │
│                                                                          │
│  "manuscript_full":                                                     │
│  ┌────────────────────────────────┐                                     │
│  │  auroc, brier, scaled_brier,   │                                     │
│  │  calibration_slope, o_e_ratio, │                                     │
│  │  net_benefit, sensitivity,     │                                     │
│  │  specificity                   │                                     │
│  └────────────────────────────────┘                                     │
│                                                                          │
│  Access: get_metric_set("stratos_core") → [MetricDefinition, ...]      │
│  List:   list_metric_sets() → ["stratos_core", "discrimination", ...]  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 3: Usage Patterns

```
┌─────────────────────────────────────────────────────────────────────────┐
│  HOW VIZ MODULES USE MetricRegistry                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  CORRECT (visualization — read metadata only):                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  from src.viz.metric_registry import MetricRegistry               │  │
│  │                                                                    │  │
│  │  # Check if metric exists                                         │  │
│  │  MetricRegistry.has("auroc")            → True                    │  │
│  │                                                                    │  │
│  │  # Get display name for plot labels                               │  │
│  │  metric = MetricRegistry.get("auroc")                             │  │
│  │  metric.display_name                    → "AUROC"                 │  │
│  │  metric.higher_is_better                → True                    │  │
│  │  metric.format_value(0.911)             → "0.911"                 │  │
│  │                                                                    │  │
│  │  # Get all metrics in a set                                       │  │
│  │  from src.viz.metric_registry import get_metric_set               │  │
│  │  core = get_metric_set("stratos_core")  → [MetricDef, ...]       │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  BANNED (visualization — never compute):                                │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  # NEVER do this in src/viz/:                                     │  │
│  │  metric = MetricRegistry.get("auroc")                             │  │
│  │  value = metric.compute_fn(y_true, y_prob)   ← BANNED            │  │
│  │  # compute_fn imports sklearn internally                          │  │
│  │  # Caught by: computation-decoupling pre-commit hook              │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ALLOWED (extraction — compute and store):                              │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  # In scripts/extract_*.py ONLY:                                  │  │
│  │  metric = MetricRegistry.get("auroc")                             │  │
│  │  value = metric.compute_fn(y_true, y_prob)   ← OK in extraction  │  │
│  │  conn.execute("INSERT INTO metrics ...")                          │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 4: Registered Metrics Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ALL PRE-REGISTERED METRICS                                              │
├──────────────────────┬───────────────────┬──────────────┬───────────────┤
│  name                │  display_name     │  higher_is   │  format       │
│                      │                   │  _better     │               │
├──────────────────────┼───────────────────┼──────────────┼───────────────┤
│  auroc               │  AUROC            │  True        │  .3f          │
│  brier               │  Brier Score      │  False       │  .3f          │
│  scaled_brier        │  Scaled Brier     │  True        │  .3f          │
│  calibration_slope   │  Cal. Slope       │  None*       │  .3f          │
│  calibration_interc  │  Cal. Intercept   │  None*       │  .3f          │
│  o_e_ratio           │  O:E Ratio        │  None*       │  .3f          │
│  net_benefit         │  Net Benefit      │  True        │  .4f          │
│  sensitivity         │  Sensitivity      │  True        │  .3f          │
│  specificity         │  Specificity      │  True        │  .3f          │
│  mae                 │  MAE              │  False       │  .4f          │
│  rmse                │  RMSE             │  False       │  .4f          │
│  outlier_f1          │  Outlier F1       │  True        │  .3f          │
│  outlier_precision   │  Outlier Prec.    │  True        │  .3f          │
│  outlier_recall      │  Outlier Recall   │  True        │  .3f          │
└──────────────────────┴───────────────────┴──────────────┴───────────────┘
│  * Calibration metrics: ideal = 1.0 (slope, O:E) or 0.0 (intercept)    │
│    Not simply "higher is better" — closer to ideal is better            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `configs/CLS_EVALUATION.yaml` | Bootstrap params used by compute_fn (n_iterations, alpha_CI) |
| `src/stats/_defaults.py` | Default constants (n_bootstrap, ci_level, prevalence) |

## Code Paths

| Module | Role |
|--------|------|
| `src/viz/metric_registry.py` | MetricDefinition dataclass, MetricRegistry class, STRATOS_METRIC_SETS, get_metric_set(), list_metric_sets() |
| `src/viz/plot_config.py` | Uses MetricRegistry for display names in plot labels |
| `src/viz/fig_decomposition_grid.py` | Uses MetricRegistry.get() for metric metadata |
| `scripts/extract_all_configs_to_duckdb.py` | Uses MetricRegistry.get().compute_fn for extraction |
| `src/stats/_defaults.py` | Default constants referenced by compute functions |

## Extension Guide

To add a new metric to the registry:
1. Define the metric in `src/viz/metric_registry.py`:
   ```python
   MetricRegistry.register(MetricDefinition(
       name="new_metric",
       display_name="New Metric",
       higher_is_better=True,
       format_str=".3f",
       value_range=(0.0, 1.0),
       compute_fn=_compute_new_metric,  # Only for extraction
   ))
   ```
2. Add compute function (prefixed with `_compute_`) in the same file
3. Add to appropriate STRATOS_METRIC_SETS group
4. Add DuckDB column in `src/data_io/streaming_duckdb_export.py`
5. Use in viz: `MetricRegistry.get("new_metric").display_name` (metadata only)

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "repo-82",
    "title": "MetricRegistry: The Code That Knows Every Metric"
  },
  "content_architecture": {
    "primary_message": "MetricRegistry groups all STRATOS metrics by domain, provides display names and formatting, and enforces the computation boundary between extraction and visualization.",
    "layout_flow": "Class diagram at top, STRATOS sets in middle, usage patterns below",
    "spatial_anchors": {
      "metric_definition": {"x": 0.3, "y": 0.15},
      "metric_registry": {"x": 0.7, "y": 0.15},
      "stratos_sets": {"x": 0.5, "y": 0.5},
      "usage_patterns": {"x": 0.5, "y": 0.8}
    },
    "key_structures": [
      {
        "name": "MetricDefinition",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["name, display_name, compute_fn"]
      },
      {
        "name": "MetricRegistry",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["register, get, has, list_metrics"]
      },
      {
        "name": "STRATOS_METRIC_SETS",
        "role": "secondary_pathway",
        "is_highlighted": true,
        "labels": ["7 predefined groups"]
      },
      {
        "name": "compute_fn boundary",
        "role": "abnormal_warning",
        "is_highlighted": true,
        "labels": ["Extraction ONLY"]
      }
    ],
    "callout_boxes": [
      {"heading": "DUAL-USE MODULE", "body_text": "Viz reads metadata (display_name, format). Extraction calls compute_fn. The pre-commit hook enforces this boundary."}
    ]
  }
}
```

## Alt Text

Class diagram of MetricRegistry showing MetricDefinition dataclass, registry methods, STRATOS metric set groupings, and the extraction-only compute boundary.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
