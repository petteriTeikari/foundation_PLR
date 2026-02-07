# fig-repo-90: pminternal: Measuring Model Instability (Riley 2023)

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-90 |
| **Title** | pminternal: Measuring Model Instability (Riley 2023) |
| **Complexity Level** | L4 |
| **Target Persona** | Biostatistician |
| **Location** | `src/stats/README.md`, `docs/explanation/model-stability.md` |
| **Priority** | P3 (Medium) |

## Purpose

Document the Python-R interop flow for model stability analysis using the pminternal R package (Rhodes 2025, implementing Riley 2023). This is a canonical example of the "no reimplementation" rule: complex statistical methods are called from Python via subprocess, not reimplemented in Python. Biostatisticians need to understand the data flow, what pminternal produces, and how results are consumed.

## Key Message

Model instability is assessed via the pminternal R package (Rhodes 2025), wrapped in Python via subprocess. Python sends predictions as CSV, R runs `validate()` with bootstrap resampling, and results return as CSV/JSON for figure generation.

## Content Specification

### Panel 1: Python-R Interop Architecture

```
PYTHON SIDE                                  R SIDE
═══════════════════════════                 ═══════════════════════════

src/stats/pminternal_wrapper.py             R (library(pminternal))
┌────────────────────────────┐              ┌────────────────────────────┐
│                            │              │                            │
│ class PmInternalWrapper:   │              │ # Called via subprocess     │
│                            │              │                            │
│   def run_stability(       │  subprocess  │ library(pminternal)        │
│     y_true,                │─────────────▶│ library(rms)               │
│     y_prob,                │  CSV input   │                            │
│     n_bootstrap=200        │              │ data <- read.csv(input)    │
│   ):                       │              │                            │
│     # 1. Export to CSV     │              │ # Fit logistic model       │
│     # 2. Call Rscript      │              │ model <- lrm(y ~ x, data) │
│     # 3. Parse results     │              │                            │
│                            │              │ # Run validation           │
│   Inputs:                  │              │ result <- validate(        │
│     y_true (N,)            │              │   model,                   │
│     y_prob (N,)            │              │   method = "boot",         │
│     n_bootstrap: int       │              │   B = 200                  │
│                            │              │ )                          │
│   Outputs:                 │  CSV/JSON    │                            │
│     instability_index      │◀─────────────│ # Extract:                 │
│     optimism_corrected     │  results     │ # - Optimism-corrected     │
│     calibration_stats      │              │ #   performance estimates  │
│     forest_plot_data       │              │ # - Instability index      │
│                            │              │ # - Calibration stats      │
└────────────────────────────┘              │ # - Bootstrap distributions│
                                            │                            │
                                            │ write.csv(results, output) │
                                            └────────────────────────────┘
```

### Panel 2: What pminternal Produces

```
validate() OUTPUT
═══════════════════

1. Optimism-Corrected Performance Estimates (OPE)
   ┌───────────────────────────────────────────┐
   │ For each metric (AUROC, Cal.Slope, Brier):│
   │   apparent  = metric on full data          │
   │   optimism  = mean(boot_metric - test_metric)│
   │   corrected = apparent - optimism           │
   └───────────────────────────────────────────┘

2. Instability Index
   ┌───────────────────────────────────────────┐
   │ Measures how much predictions change       │
   │ across bootstrap resamples.                │
   │ Higher = more unstable model.              │
   │                                            │
   │ instability_index = mean |p_boot - p_orig| │
   └───────────────────────────────────────────┘

3. Forest Plot Data
   ┌───────────────────────────────────────────┐
   │ Per-metric:                                │
   │   point_estimate                           │
   │   ci_lower (2.5th percentile)              │
   │   ci_upper (97.5th percentile)             │
   │   optimism                                 │
   └───────────────────────────────────────────┘

4. Bootstrap Distributions
   ┌───────────────────────────────────────────┐
   │ Raw bootstrap values for each metric       │
   │ (B=200 values per metric)                  │
   │ Used for histograms and density plots      │
   └───────────────────────────────────────────┘
```

### Panel 3: Data Extraction & Figure Generation

```
EXTRACTION (Block 1)                    VISUALIZATION (Block 2)
═══════════════════                     ═══════════════════════

scripts/extract_pminternal_data.py      src/viz/fig_instability_plots.py
        │                                       │
        ▼                                       ▼
For each (outlier x imputation) config: DuckDB READ ONLY
  1. Load y_true, y_prob from MLflow    ├── Instability forest plots
  2. Call PmInternalWrapper             ├── OPE comparison plots
  3. Store in DuckDB                    ├── Bootstrap distributions
        │                               └── Calibration stability
        ▼
DuckDB tables:
  pminternal_results
  ├── outlier_method
  ├── imputation_method
  ├── instability_index
  ├── optimism_auroc
  ├── optimism_cal_slope
  ├── corrected_auroc
  ├── corrected_cal_slope
  └── bootstrap_distributions (JSON)

Output figures: R1-R6 (model stability series)
```

### Panel 4: Why Not Reimplement

```
THE NO-REIMPLEMENTATION RULE
════════════════════════════

  ✗ BANNED: src/stats/pminternal_reimplementation.py
    Subtle bugs in optimism correction, instability calculation,
    bootstrap stratification. Reviewers expect canonical implementation.

  ✓ ALLOWED: src/stats/pminternal_wrapper.py
    Calls the verified R package via subprocess.
    Authors have validated the code (Rhodes 2025).
    Published-method reproducibility guaranteed.

  Pattern:
    Python → subprocess.run(["Rscript", script.R, input.csv, output.json])
    Parse JSON/CSV output back into Python objects.

  Also used for:
    ├── dcurves (DCA computation)
    ├── pmcalibration (calibration curves)
    └── pROC (partial AUROC, if needed)
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `renv.lock` | Pins pminternal version |
| `configs/CLS_EVALUATION.yaml` | `BOOTSTRAP.n_iterations=1000` for main analysis |
| `src/stats/_defaults.py` | Default `n_bootstrap=1000`, `ci_level=0.95` |

## Code Paths

| Module | Role |
|--------|------|
| `src/stats/pminternal_wrapper.py` | Python wrapper for pminternal R package |
| `src/stats/pminternal_analysis.py` | Analysis orchestration |
| `src/viz/fig_instability_plots.py` | Instability figure generation (DuckDB read-only) |
| `src/viz/generate_instability_figures.py` | Instability figure entry point |
| `scripts/extract_pminternal_data.py` | Extraction script for pminternal results |
| `tests/test_pminternal_extraction.py` | Tests for extraction correctness |
| `tests/test_pminternal_json.py` | Tests for JSON output format |
| `tests/test_pminternal_figure.R` | R-side tests for pminternal figures |

## Extension Guide

To add a new R-based statistical method:
1. Install R package: `renv::install("package")` then `renv::snapshot()`
2. Create wrapper: `src/stats/new_method_wrapper.py`
3. Implement subprocess call pattern:
   ```python
   result = subprocess.run(
       ["Rscript", "src/r/scripts/new_method.R", input_csv, output_json],
       capture_output=True, text=True
   )
   ```
4. Parse output JSON/CSV back into Python
5. Add extraction in `scripts/extract_new_method_data.py`
6. Store results in DuckDB table
7. Create visualization in `src/viz/` (read-only from DuckDB)
8. Add tests for both the R script and the Python wrapper

References:
- Riley RD et al. 2023 BMC Medicine (model instability framework)
- Rhodes PJ et al. 2025 (pminternal R package)

Note: This is a repo documentation figure - shows HOW the Python-R interop works, NOT model stability results.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "repo-90",
    "title": "pminternal: Measuring Model Instability (Riley 2023)"
  },
  "content_architecture": {
    "primary_message": "Model instability is assessed via the pminternal R package, wrapped in Python via subprocess. No reimplementation -- verified library interop only.",
    "layout_flow": "Left-right Python-R interop flow with extraction and figure generation below",
    "spatial_anchors": {
      "python_side": {"x": 0.05, "y": 0.05, "width": 0.4, "height": 0.45},
      "r_side": {"x": 0.55, "y": 0.05, "width": 0.4, "height": 0.45},
      "outputs": {"x": 0.05, "y": 0.55, "width": 0.4, "height": 0.35},
      "no_reimplement": {"x": 0.55, "y": 0.55, "width": 0.4, "height": 0.35}
    },
    "key_structures": [
      {
        "name": "PmInternalWrapper",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Python subprocess call"]
      },
      {
        "name": "pminternal::validate()",
        "role": "foundation_model",
        "is_highlighted": true,
        "labels": ["R verified implementation"]
      },
      {
        "name": "Instability Index",
        "role": "highlight_accent",
        "is_highlighted": true,
        "labels": ["Key output metric"]
      }
    ],
    "callout_boxes": [
      {"heading": "NO REIMPLEMENTATION", "body_text": "Reimplementing pminternal in Python is BANNED. Use verified R package via subprocess."},
      {"heading": "REFERENCES", "body_text": "Riley 2023 BMC Medicine (framework). Rhodes 2025 (pminternal R package)."}
    ]
  }
}
```

## Alt Text

Two-column diagram showing Python-R interop for model stability analysis: Python wrapper sends predictions via CSV to R pminternal package, which returns optimism-corrected estimates and instability indices via JSON.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
