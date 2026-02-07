# End-to-End Pipeline for Visualization: COMPREHENSIVE PLAN

**Created**: 2026-01-25
**Status**: READY FOR EXECUTION
**Related**: GitHub Issue #5, figure-improvement-plan-with-ggplot2.md

---

## Executive Summary

This plan ensures **scientific reproducibility** by establishing a complete data pipeline:

```
MLflow → DuckDB (source of truth) → JSON (figure data) → Visualizations
```

**Core Principle**: ALL numerical claims must be traceable from visualization back to MLflow.

---

## FINAL Top-10 CatBoost Configurations

### Exclusion Criteria (Programmatic Rule)

Configurations are **EXCLUDED** from the top-10 if ANY of the following conditions are true:

| Condition | Field | Values | Reason |
|-----------|-------|--------|--------|
| **Unknown OD source** | `outlier_method` | `"anomaly"`, `"exclude"` | Legacy placeholder from parsing code |
| **No linked OD run** | `mlflow_run_outlier_detection` | `None` | Actual preprocessing unclear |
| **Missing OD metrics** | `Outlier_f1` | `NaN` | No outlier detection was evaluated |

**Implementation**: The DuckDB column `outlier_source_known` is `false` for excluded configs.

```sql
-- Query for valid top-10 CatBoost
SELECT * FROM top10_catboost;
-- This view already excludes configs where outlier_source_known = false
```

### Top-10 Selection

| New Rank | Outlier Detection | Imputation | AUROC Mean | 95% CI |
|----------|-------------------|------------|------------|--------|
| 1 | ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune | CSDI | 0.913 | (0.904, 0.919) |
| 2 | ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune | TimesNet | 0.912 | (0.902, 0.921) |
| 3 | ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune | CSDI | 0.912 | (0.904, 0.918) |
| 4 | ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune | TimesNet | 0.911 | (0.903, 0.919) |
| 5 | pupil-gt | pupil-gt | 0.911 | (0.903, 0.918) |
| 6 | pupil-gt | ensemble-CSDI-MOMENT-SAITS | 0.911 | (0.903, 0.918) |
| 7 | ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune | SAITS | 0.910 | (0.903, 0.917) |
| 8 | pupil-gt | CSDI | 0.910 | (0.903, 0.917) |
| 9 | MOMENT-gt-finetune | SAITS | 0.910 | (0.899, 0.921) |
| 10 | pupil-gt | TimesNet | 0.909 | (0.902, 0.916) |

### Excluded Configurations

The following configs were automatically excluded by the `outlier_source_known = false` rule:

| Original Rank | Outlier Method | Reason |
|---------------|----------------|--------|
| 5 | `anomaly` | `mlflow_run_outlier_detection = None`, `Outlier_f1 = NaN` |

**Documentation**: See `docs/mlflow-naming-convention.md` for full explanation of "anomaly" as a legacy placeholder.

---

## DuckDB Schema Design

### Database 1: `outputs/foundation_plr_results.db` (PUBLIC)

Aggregate metrics and summaries - shareable.

```sql
-- Essential metrics (one row per configuration)
CREATE TABLE essential_metrics (
    run_id VARCHAR PRIMARY KEY,
    outlier_method VARCHAR,
    imputation_method VARCHAR,
    featurization VARCHAR,
    classifier VARCHAR,
    auroc DOUBLE,
    auroc_ci_lo DOUBLE,
    auroc_ci_hi DOUBLE,
    brier DOUBLE,
    scaled_brier DOUBLE,
    calibration_slope DOUBLE,
    calibration_intercept DOUBLE,
    o_e_ratio DOUBLE,
    net_benefit_5pct DOUBLE,
    net_benefit_10pct DOUBLE,
    net_benefit_15pct DOUBLE,
    net_benefit_20pct DOUBLE,
    outlier_f1 DOUBLE,
    n_bootstrap INTEGER
);

-- SHAP feature importance (one row per config × feature)
CREATE TABLE shap_feature_importance (
    config_rank INTEGER,
    run_id VARCHAR,
    feature_name VARCHAR,
    mean_abs_shap DOUBLE,
    std_abs_shap DOUBLE,
    ci_lo DOUBLE,
    ci_hi DOUBLE,
    PRIMARY KEY (config_rank, feature_name)
);

-- VIF analysis
CREATE TABLE vif_analysis (
    config_rank INTEGER,
    feature_name VARCHAR,
    vif DOUBLE,
    is_collinear BOOLEAN,  -- VIF > 5
    is_highly_collinear BOOLEAN,  -- VIF > 10
    PRIMARY KEY (config_rank, feature_name)
);

-- Ensemble feature importance (aggregated across top-10)
CREATE TABLE ensemble_importance (
    feature_name VARCHAR PRIMARY KEY,
    mean_importance_equal DOUBLE,  -- Equal weighting
    std_importance_equal DOUBLE,
    mean_importance_auroc DOUBLE,  -- AUROC-weighted
    std_importance_auroc DOUBLE,
    ci_lo_equal DOUBLE,
    ci_hi_equal DOUBLE,
    ci_lo_auroc DOUBLE,
    ci_hi_auroc DOUBLE
);

-- Variance decomposition
CREATE TABLE variance_decomposition (
    factor VARCHAR PRIMARY KEY,
    eta_squared DOUBLE,
    partial_eta_squared DOUBLE,
    f_statistic DOUBLE,
    p_value DOUBLE,
    df_numerator INTEGER,
    df_denominator INTEGER
);

-- CD diagram data
CREATE TABLE cd_diagram_data (
    comparison_type VARCHAR,  -- 'outlier', 'imputation', 'full_pipeline'
    method VARCHAR,
    mean_rank DOUBLE,
    n_configs INTEGER
);
```

### Database 2: `outputs/foundation_plr_bootstrap.db` (PRIVATE - LOCAL ONLY)

Per-bootstrap data - too large for git, contains individual-level data.

```sql
-- Per-bootstrap AUROC values (1000 × 407 configs)
CREATE TABLE bootstrap_auroc (
    run_id VARCHAR,
    bootstrap_idx INTEGER,
    auroc DOUBLE,
    PRIMARY KEY (run_id, bootstrap_idx)
);

-- Per-bootstrap SHAP values (1000 × 10 configs × features × samples)
CREATE TABLE bootstrap_shap (
    config_rank INTEGER,
    bootstrap_idx INTEGER,
    sample_idx INTEGER,
    feature_name VARCHAR,
    shap_value DOUBLE
);

-- Per-bootstrap predictions (for calibration/DCA curves)
CREATE TABLE bootstrap_predictions (
    run_id VARCHAR,
    bootstrap_idx INTEGER,
    sample_idx INTEGER,
    y_true INTEGER,
    y_prob DOUBLE
);

-- ROC curves per bootstrap
CREATE TABLE bootstrap_roc (
    run_id VARCHAR,
    bootstrap_idx INTEGER,
    fpr DOUBLE[],
    tpr DOUBLE[],
    thresholds DOUBLE[]
);

-- DCA curves per bootstrap
CREATE TABLE bootstrap_dca (
    run_id VARCHAR,
    bootstrap_idx INTEGER,
    thresholds DOUBLE[],
    nb_model DOUBLE[],
    nb_all DOUBLE[],
    nb_none DOUBLE[]
);
```

### Database 3: `outputs/shap_analysis.db` (PUBLIC)

SHAP-specific results - shareable summaries.

```sql
-- SHAP config summary (one row per config)
CREATE TABLE shap_config_summary (
    config_rank INTEGER PRIMARY KEY,
    run_id VARCHAR,
    outlier_method VARCHAR,
    imputation_method VARCHAR,
    auroc_mean DOUBLE,
    auroc_ci_lo DOUBLE,
    auroc_ci_hi DOUBLE,
    n_bootstrap INTEGER,
    n_features INTEGER,
    n_samples_test INTEGER
);

-- SHAP feature importance with full uncertainty
CREATE TABLE shap_importance_detailed (
    config_rank INTEGER,
    feature_name VARCHAR,
    mean_abs_shap DOUBLE,
    std_abs_shap DOUBLE,
    median_abs_shap DOUBLE,
    q025 DOUBLE,
    q975 DOUBLE,
    min_abs_shap DOUBLE,
    max_abs_shap DOUBLE,
    n_bootstrap INTEGER,
    PRIMARY KEY (config_rank, feature_name)
);
```

---

## Phase 1: Data Extraction Pipeline

### Task 1.1: Extract ALL Configurations to DuckDB

**Script**: `scripts/extract_all_configs_to_duckdb.py`

```python
"""Extract ALL 407 CatBoost configurations to DuckDB."""
# Input: /home/petteri/mlruns/253031330985650090
# Output: outputs/foundation_plr_results.db
```

**Progress Tracking**:
- [x] Extract essential_metrics table (328 rows - all classifiers with model pickles)
- [ ] Extract variance_decomposition table
- [ ] Extract cd_diagram_data table
- [x] Verify row counts match expected (328 total, 81 CatBoost)

### Task 1.2: Extract Top-10 CatBoost Models + Artifacts

**Script**: `scripts/extract_top10_models_with_artifacts.py`

```python
"""Extract top-10 CatBoost models with full artifacts for SHAP."""
# Input: MLflow runs for ranks 1-4, 6-11 (excluding rank 5)
# Output:
#   - outputs/top10_catboost_models.pkl (models + features)
#   - outputs/shap_analysis.db (shap_config_summary)
```

**Progress Tracking**:
- [x] Identify correct run_ids for 10 configs
- [x] Extract CatBoost models (10 configs × 1000 bootstrap = 10,000 models)
- [x] Extract X_train, X_test, y_train, y_test, feature_names (10 sets)
- [x] Verify model loading works
- [ ] Write shap_config_summary table

### Task 1.3: Extract Bootstrap Data

**Script**: `scripts/extract_bootstrap_data.py`

```python
"""Extract per-bootstrap metrics for uncertainty quantification."""
# Input: MLflow artifacts/*_metrics_CATBOOST_*.pickle
# Output: outputs/foundation_plr_bootstrap.db (LOCAL ONLY)
```

**Progress Tracking**:
- [ ] Extract bootstrap_auroc (407 configs × 1000 iterations)
- [ ] Extract bootstrap_predictions for top-10 configs
- [ ] Verify bootstrap count per config

---

## Phase 2: SHAP Analysis Pipeline

### Task 2.1: Compute SHAP Values for Top-10 Configs

**Script**: `scripts/compute_shap_values.py`

For each of 10 configs × 1000 bootstrap iterations = 10,000 SHAP computations.

```python
"""Compute per-bootstrap SHAP values for top-10 configs."""
import shap
from catboost import CatBoostClassifier

for config in top_10_configs:
    model = load_model(config)
    X_test = load_features(config)

    # TreeExplainer is fast for CatBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Store in DuckDB
    save_shap_to_db(config, shap_values)
```

**Progress Tracking**:
- [ ] Config 1 (AUROC 0.913): 1000 bootstrap SHAP computed
- [ ] Config 2 (AUROC 0.912): 1000 bootstrap SHAP computed
- [ ] Config 3 (AUROC 0.912): 1000 bootstrap SHAP computed
- [ ] Config 4 (AUROC 0.911): 1000 bootstrap SHAP computed
- [ ] Config 6 (ground truth): 1000 bootstrap SHAP computed
- [ ] Config 7 (AUROC 0.911): 1000 bootstrap SHAP computed
- [ ] Config 8 (AUROC 0.910): 1000 bootstrap SHAP computed
- [ ] Config 9 (AUROC 0.910): 1000 bootstrap SHAP computed
- [ ] Config 10 (AUROC 0.910): 1000 bootstrap SHAP computed
- [ ] Config 11 (AUROC 0.909): 1000 bootstrap SHAP computed

### Task 2.2: Aggregate SHAP Statistics

**Script**: `scripts/aggregate_shap_statistics.py`

```python
"""Aggregate SHAP values to feature importance with uncertainty."""
# Per config:
#   - mean_abs_shap[feature] = mean over (bootstrap × samples)
#   - std_abs_shap[feature] = std over (bootstrap × samples)
#   - ci_lo, ci_hi = 2.5th, 97.5th percentile

# Output: shap_importance_detailed table
```

**Progress Tracking**:
- [ ] Compute per-config feature importance (10 rows × n_features)
- [ ] Compute ensemble importance (equal weighting)
- [ ] Compute ensemble importance (AUROC weighting)
- [ ] Write to DuckDB

### Task 2.3: Compute VIF Analysis

**Script**: `scripts/compute_vif_analysis.py`

```python
"""Compute Variance Inflation Factor for feature collinearity."""
from statsmodels.stats.outliers_influence import variance_inflation_factor

for config in top_10_configs:
    X = load_X_train(config)
    vif_scores = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    save_vif_to_db(config, feature_names, vif_scores)
```

**Progress Tracking**:
- [ ] Compute VIF for all 10 configs
- [ ] Write vif_analysis table
- [ ] Flag features with VIF > 5 (collinear) and VIF > 10 (highly collinear)

---

## Phase 3: Visualization Data Generation

### Static Publication-Quality Figures

| Figure ID | Source Table | JSON Output | Status |
|-----------|--------------|-------------|--------|
| fig_shap_summary | shap_importance_detailed | fig_shap_summary_data.json | [ ] |
| fig_shap_beeswarm | bootstrap_shap | fig_shap_beeswarm_data.json | [ ] |
| fig_feature_importance_bar | ensemble_importance | fig_feature_importance_bar_data.json | [ ] |
| fig_vif_bar | vif_analysis | fig_vif_bar_data.json | [ ] |
| fig_raincloud_auroc | bootstrap_auroc | fig_raincloud_auroc_data.json | [ ] |
| fig_calibration_enhanced | bootstrap_predictions | fig_calibration_enhanced_data.json | [ ] |
| fig_pdp | bootstrap_predictions + models | fig_pdp_data.json | [ ] |
| fig_upset_classifier | bootstrap_predictions | fig_upset_classifier_data.json | [ ] |

### Interactive D3.js Figures

| Figure ID | Source Table | JSON Output | Status |
|-----------|--------------|-------------|--------|
| fig_sankey_pipeline | essential_metrics | sankey_pipeline_data.json | [ ] |
| fig_parallel_coords | essential_metrics | parallel_coords_data.json | [ ] |
| fig_linked_brushing | essential_metrics + bootstrap | linked_brushing_data.json | [ ] |
| fig_bootstrap_animation | bootstrap_auroc | bootstrap_animation_data.json | [ ] |
| fig_spec_curve_interactive | essential_metrics | spec_curve_interactive_data.json | [ ] |
| fig_plr_viewer | PRIVATE: raw signals | plr_viewer_data.json | [ ] |

### Existing Figures (Verify Data Flow)

| Figure ID | Source Table | JSON Output | Status |
|-----------|--------------|-------------|--------|
| fig01_variance_decomposition | variance_decomposition | fig01_variance_decomposition_data.json | [ ] Verify |
| fig02_forest_outlier | essential_metrics | fig02_forest_outlier_data.json | [ ] Verify |
| fig03_forest_imputation | essential_metrics | fig03_forest_imputation_data.json | [ ] Verify |
| fig06_specification_curve | essential_metrics | fig06_specification_curve_data.json | [ ] Verify |
| cd_preprocessing_comparison | cd_diagram_data | cd_preprocessing_comparison_data.json | [ ] Verify |

---

## Phase 4: Visualization Generation

### Task 4.1: SHAP Figures (NEW)

**Script**: `src/viz/shap_figures.py`

```python
"""Generate SHAP-based figures."""
# fig_shap_summary: SHAP summary plot (beeswarm) for top-10 mean
# fig_shap_beeswarm: Per-config beeswarm with bootstrap uncertainty
# fig_feature_importance_bar: Bar chart with CI error bars
# fig_feature_importance_ensemble: Ensemble aggregated importance
```

**Progress Tracking**:
- [ ] fig_shap_summary.pdf generated
- [ ] fig_shap_beeswarm.pdf generated (10 subplots)
- [ ] fig_feature_importance_bar.pdf generated
- [ ] fig_feature_importance_ensemble.pdf generated
- [ ] All JSON data files saved

### Task 4.2: VIF Figure (NEW)

**Script**: `src/viz/vif_figure.py`

```python
"""Generate VIF bar chart."""
# Horizontal bar chart
# Red bars for VIF > 5, dark red for VIF > 10
# Vertical reference lines at 5 and 10
```

**Progress Tracking**:
- [ ] fig_vif_bar.pdf generated
- [ ] JSON data saved

### Task 4.3: Raincloud Plots (NEW)

**Script**: `src/viz/raincloud_plots.py`

```python
"""Generate raincloud plots for AUROC distributions."""
# Half-violin + box plot + jittered points
# For top-10 configs showing bootstrap distributions
```

**Progress Tracking**:
- [ ] fig_raincloud_auroc.pdf generated
- [ ] JSON data saved

### Task 4.4: Update Existing Figures

**Script**: `src/viz/generate_all_figures.py` (EXISTING)

Verify all existing figures pull from DuckDB, not hardcoded values.

**Progress Tracking**:
- [ ] fig01_variance_decomposition verified
- [ ] fig02_forest_outlier verified
- [ ] fig03_forest_imputation verified
- [ ] fig06_specification_curve verified
- [ ] cd_preprocessing_comparison verified

---

## Phase 5: D3.js Interactive Figures

### Task 5.1: Interactive Sankey Diagram

**Files**:
- `apps/visualization/src/components/SankeyDiagram.tsx`
- `figures/generated/data/sankey_pipeline_data.json`

**Progress Tracking**:
- [ ] JSON data generated
- [ ] D3 component implemented
- [ ] Hover interactions work
- [ ] Click filtering works

### Task 5.2: Linked Brushing Dashboard

**Files**:
- `apps/visualization/src/components/LinkedBrushing.tsx`
- `figures/generated/data/linked_brushing_data.json`

**Progress Tracking**:
- [ ] JSON data generated
- [ ] AUROC forest plot component
- [ ] Calibration scatter component
- [ ] DCA curves component
- [ ] Cross-highlighting works

### Task 5.3: PLR Signal Viewer

**Files**:
- `apps/visualization/src/components/PLRViewer.tsx`
- `figures/generated/data/plr_viewer_data.json` (PRIVATE)

**Progress Tracking**:
- [ ] JSON data generated (12 demo subjects)
- [ ] Signal rendering works
- [ ] Artifact highlighting works
- [ ] Imputation toggle works

---

## Verification Checkpoints

### Checkpoint 1: Data Integrity

Run: `pytest tests/test_extraction_verification.py -v`

- [ ] Config count = 407
- [ ] AUROC range = [0.500, 0.913]
- [ ] Top-10 configs correctly identified (excluding rank 5)
- [ ] All required columns present

### Checkpoint 2: SHAP Completeness

Run: `pytest tests/test_shap_extraction.py -v`

- [ ] 10 configs × n_features rows in shap_importance_detailed
- [ ] Bootstrap variance captured (std > 0)
- [ ] VIF computed for all features

### Checkpoint 3: Figure Reproducibility

Run: `python src/viz/verify_figure_data.py`

- [ ] All figures have corresponding JSON files
- [ ] JSON matches DuckDB values (within tolerance)
- [ ] No hardcoded values in figure code

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| SHAP computation too slow | Medium | High | Use TreeExplainer (fast); parallelize |
| Memory constraints | Medium | Medium | Process sequentially; clear memory |
| Model loading failures | Low | Medium | Error handling; skip + report |
| Bootstrap count mismatch | Low | High | Verify counts at extraction |

---

## File Inventory (To Be Created)

### Scripts (Phase 1-2)

- [ ] `scripts/extract_all_configs_to_duckdb.py`
- [ ] `scripts/extract_top10_models_with_artifacts.py`
- [ ] `scripts/extract_bootstrap_data.py`
- [ ] `scripts/compute_shap_values.py`
- [ ] `scripts/aggregate_shap_statistics.py`
- [ ] `scripts/compute_vif_analysis.py`

### Visualization Code (Phase 3-4)

- [ ] `src/viz/shap_figures.py`
- [ ] `src/viz/vif_figure.py`
- [ ] `src/viz/raincloud_plots.py`
- [ ] `src/viz/pdp_plots.py`

### D3.js Components (Phase 5)

- [ ] `apps/visualization/src/components/SankeyDiagram.tsx`
- [ ] `apps/visualization/src/components/LinkedBrushing.tsx`
- [ ] `apps/visualization/src/components/PLRViewer.tsx`
- [ ] `apps/visualization/src/components/BootstrapAnimation.tsx`

### Tests

- [x] `tests/test_extraction_verification.py` (EXISTS)
- [x] `tests/test_shap_extraction.py` (UPDATED - 9 tests passing)
  - `TestTop10ArtifactIntegrity`: 8 tests verifying artifact structure
  - `TestSplitAgnosticAPI`: 1 test for split selection
- [ ] `tests/test_figure_data_integrity.py`

---

## Execution Order

```
Phase 1: Data Extraction
├── 1.1 Extract ALL configs → essential_metrics
├── 1.2 Extract top-10 models → top10_catboost_models.pkl
└── 1.3 Extract bootstrap data → bootstrap tables

Phase 2: SHAP Analysis
├── 2.1 Compute SHAP (10 × 1000 = 10,000 computations)
├── 2.2 Aggregate statistics → shap_importance_detailed
└── 2.3 Compute VIF → vif_analysis

Phase 3: Data to JSON
├── Generate JSON for each figure
└── Verify JSON matches DuckDB

Phase 4: Static Figures
├── 4.1 SHAP figures (4 figures)
├── 4.2 VIF figure
├── 4.3 Raincloud plots
└── 4.4 Verify existing figures

Phase 5: Interactive Figures
├── 5.1 Sankey diagram
├── 5.2 Linked brushing
└── 5.3 PLR viewer

Verification: Run all checkpoints
```

---

## Reviewer Agent Evaluation

### Reviewer 1: Statistical Rigor

**Concerns Addressed**:
- Bootstrap uncertainty propagated to SHAP
- VIF for collinearity assessment
- Both equal and AUROC-weighted ensemble aggregation

### Reviewer 2: Reproducibility

**Concerns Addressed**:
- DuckDB as single source of truth
- JSON data accompanies every figure
- Public/Private separation for patient data

### Reviewer 3: Completeness

**Concerns Addressed**:
- All 25+ visualizations from suggested-visualizations.md covered
- Existing figures verified against DuckDB
- Progress tracking with checkboxes

---

## Success Criteria

1. **Data Integrity**: `pytest tests/test_extraction_verification.py` PASSES
2. **SHAP Complete**: 10 configs × all features have SHAP importance
3. **Figures Reproducible**: Every PDF has matching JSON
4. **No Hardcoding**: Zero hardcoded values in visualization code
5. **Trust**: User can verify any claim by querying DuckDB

---

*Plan created: 2026-01-25*
*Status: READY FOR EXECUTION*
*Next: Begin Phase 1.1 - Extract ALL configs to DuckDB*
