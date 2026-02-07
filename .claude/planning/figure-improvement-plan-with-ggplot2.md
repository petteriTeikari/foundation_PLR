# Figure Improvement Plan with ggplot2 and Feature Importance Analysis

**Created**: 2026-01-25
**Status**: READY FOR EXECUTION - User decisions recorded
**Related Issue**: GitHub Issue #5 orchestration pipeline complete

### Session Progress (2026-01-25)
- [x] Fixed extraction pipeline (CI values were NaN due to numpy array bug)
- [x] Re-extracted data with correct CIs
- [x] Fixed manuscript Table 1 with correct AUROC values (0.913 top, not 0.889)
- [x] Verified top-10 CatBoost configurations from MLflow
- [x] Recorded user decisions on approach
- [ ] **NEXT**: Extract top-10 models with full bootstrap data for SHAP
- [ ] Compute per-bootstrap SHAP values (10,000 computations)
- [ ] Compute VIF analysis
- [ ] Generate ggplot2 figures

---

## User Prompt (Verbatim)

> Now that we have the end-to-end pipeline working we could work on the insides of the last visualization plot! Let's create a plan to /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/planning/figure-improvement-plan-with-ggplot2.md ! Analyze these ideas /home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/iterated-llm-council/03-kb-restructure-evaluation/suggested-visualizations.md on new visualizations, and we definitely at least need to do feature importance analysis with various methods (VIF, SHAP, and SHAP variant is now for cool kids? https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Catboost%20tutorial.html as the vanilla one?). You should create a .db artifact or .pkl (whatever is easier and more reproducible for the top-10 Catboost models from all 1000 bootrap itearation so that we could get the feature importances, e.g. SHAP mean+std both from the 1000 boostrap iterations for each of those 10 hyperparam combos, and then propagate those invididual uncertainties to the "ensemble top-10" to have an idea what the non-winner rules it all feature importance looks like for the hand-crafted PLR features (which is itself is somewhat interesting for the readers of this article). Save my prompt verbatim to the plan, and then evaluate the plan with reviewer agents to optimize it for the execution. Pause before execution, and ask any clarification in multi-choice manner if there any!

---

## Executive Summary

This plan addresses feature importance analysis for the Foundation PLR manuscript, focusing on:

1. **SHAP Analysis** with uncertainty quantification from bootstrap iterations
2. **VIF (Variance Inflation Factor)** for feature collinearity assessment
3. **Ensemble Top-10 Feature Importance** aggregating across hyperparameter combos
4. **Data Artifact Creation** for reproducibility

---

## Existing Infrastructure

### Already Available

| Asset | Path | Description |
|-------|------|-------------|
| Extraction script | `scripts/extract_top_models_from_mlflow.py` | Top-N model extraction |
| MLflow runs | `/home/petteri/mlruns/253031330985650090` | 410 classification runs |
| Bootstrap metrics | Per-run pickle files | 1000 iterations each |
| CatBoost models | `artifacts/model/*.pickle` | Trained models |
| Feature arrays | `artifacts/dict_arrays/*.pickle` | X_train, X_test, feature_names |

### Missing (To Be Created)

| Asset | Format | Description |
|-------|--------|-------------|
| Top-10 model artifact | `.pkl` | Models + features for SHAP |
| SHAP values database | `.db` or `.pkl` | Per-bootstrap SHAP with uncertainty |
| VIF analysis results | `.json` | VIF scores per feature |
| Ensemble importance | `.json` | Aggregated across top-10 configs |

---

## Task Breakdown

### Phase 1: Data Artifact Creation

**Task 1.1: Extract Top-10 CatBoost Models with Full Data**

```python
# Expected output structure
top10_artifact = {
    "configs": [
        {
            "rank": 1,
            "run_id": "abc123",
            "config": {"outlier": "...", "imputation": "...", "classifier": "CatBoost"},
            "auroc_mean": 0.913,
            "auroc_std": 0.025,
            "model": <CatBoostClassifier>,
            "X_train": np.array(...),  # For SHAP background
            "X_test": np.array(...),   # For SHAP explanation
            "y_test": np.array(...),
            "feature_names": ["amp_bin_0", "amp_bin_1", ..., "PIPR"],
        },
        # ... 9 more configs
    ],
    "metadata": {
        "created": "2026-01-25",
        "n_bootstrap": 1000,
        "selection_criterion": "AUROC_mean",
    }
}
```

**Task 1.2: Bootstrap SHAP Extraction**

For each of the top-10 configs × 1000 bootstrap iterations:
- Extract SHAP values for test set predictions
- Aggregate: mean, std, percentiles (2.5, 97.5)

**Challenge**: 1000 bootstrap models per config = 10,000 SHAP computations
**Solution**:
- Option A: Sample N bootstrap iterations (e.g., 100)
- Option B: Use SHAP approximations (TreeExplainer is fast for CatBoost)

---

### Phase 2: Feature Importance Analysis

**Task 2.1: SHAP Summary Plot (Standard)**

```python
import shap

# Per-config SHAP
for config in top10_artifact["configs"]:
    explainer = shap.TreeExplainer(config["model"])
    shap_values = explainer.shap_values(config["X_test"])

    # Save summary statistics
    shap_stats = {
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "feature_names": config["feature_names"],
    }
```

**Task 2.2: SHAP with Bootstrap Uncertainty**

```python
# Aggregate across bootstrap iterations
shap_all_boots = []
for boot_idx in range(n_bootstrap_samples):
    boot_model = load_bootstrap_model(config, boot_idx)
    explainer = shap.TreeExplainer(boot_model)
    shap_values = explainer.shap_values(X_test)
    shap_all_boots.append(shap_values)

# Uncertainty: mean ± std across bootstrap
shap_mean = np.mean(shap_all_boots, axis=0)
shap_std = np.std(shap_all_boots, axis=0)
```

**Task 2.3: VIF Analysis**

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def compute_vif(X, feature_names):
    vif_data = []
    for i, name in enumerate(feature_names):
        vif = variance_inflation_factor(X, i)
        vif_data.append({"feature": name, "VIF": vif})
    return pd.DataFrame(vif_data)
```

**Task 2.4: Ensemble Top-10 Feature Importance**

Aggregate SHAP across all top-10 configs:

```python
# Weighted by AUROC or equal weights
ensemble_shap = {}
for feature in feature_names:
    values_across_configs = []
    for config in top10_artifact["configs"]:
        values_across_configs.append(config["shap_stats"][feature])

    ensemble_shap[feature] = {
        "mean": np.mean(values_across_configs),
        "std": np.std(values_across_configs),
        "min": np.min(values_across_configs),
        "max": np.max(values_across_configs),
    }
```

---

### Phase 3: Visualization Generation

**Task 3.1: SHAP Beeswarm Plot (Standard)**

Using vanilla SHAP as referenced in user prompt:
- https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Catboost%20tutorial.html

```python
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

**Task 3.2: Feature Importance Forest Plot with Uncertainty**

Using ggplot2 (R) for publication quality:

```r
library(ggplot2)
library(ggdist)

# Feature importance with bootstrap CI
ggplot(importance_df, aes(x = mean_importance, y = reorder(feature, mean_importance))) +
  geom_pointrange(aes(xmin = ci_lower, xmax = ci_upper)) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(x = "SHAP Feature Importance (mean |SHAP|)",
       y = "Feature",
       title = "Feature Importance with Bootstrap Uncertainty") +
  theme_minimal()
```

**Task 3.3: VIF Bar Chart**

```r
ggplot(vif_df, aes(x = VIF, y = reorder(feature, VIF))) +
  geom_col(fill = ifelse(vif_df$VIF > 5, "red", "steelblue")) +
  geom_vline(xintercept = 5, linetype = "dashed", color = "red") +
  geom_vline(xintercept = 10, linetype = "dashed", color = "darkred") +
  annotate("text", x = 5.5, y = 1, label = "Concern", color = "red") +
  annotate("text", x = 10.5, y = 1, label = "High", color = "darkred") +
  labs(title = "Variance Inflation Factor (VIF) by Feature") +
  theme_minimal()
```

**Task 3.4: Ensemble Top-10 Comparison**

```r
# Faceted by config or stacked
ggplot(ensemble_df, aes(x = importance, y = feature, fill = config)) +
  geom_col(position = "dodge") +
  facet_wrap(~ config, ncol = 2) +
  theme_minimal()
```

---

## Output Artifacts

| Artifact | Path | Format | Privacy |
|----------|------|--------|---------|
| Top-10 models | `data/artifacts/top10_catboost_models.pkl` | Pickle | PUBLIC |
| SHAP values DB | `data/artifacts/shap_values_bootstrap.db` | DuckDB | PUBLIC |
| VIF analysis | `data/artifacts/vif_analysis.json` | JSON | PUBLIC |
| Feature importance | `figures/generated/fig_shap_importance.pdf` | PDF | PUBLIC |
| Ensemble comparison | `figures/generated/fig_ensemble_feature_importance.pdf` | PDF | PUBLIC |

---

## Visualization Suggestions Analysis

From `suggested-visualizations.md`, the most relevant for feature importance:

| Visualization | Priority | Difficulty | Status |
|---------------|----------|------------|--------|
| **SHAP summary plots** | HIGH | Low | Primary task |
| **Feature importance bars** | HIGH | Low | Primary task |
| Raincloud plots | Medium | Low | Consider for SHAP distributions |
| Partial Dependence Plots | Medium | Medium | Nice-to-have |
| VIF bar chart | HIGH | Low | Primary task |

### Ideas to Defer

- Interactive D3.js visualizations (Phase 2)
- Chord/Sankey diagrams (graphical abstract phase)
- t-SNE embeddings (supplementary)

---

## SHAP Variant Consideration

The user asked: "SHAP variant is now for cool kids?"

### Options

1. **TreeSHAP** (Standard for tree models)
   - Fast, exact for tree models
   - Used in referenced CatBoost tutorial
   - **Recommended for this analysis**

2. **KernelSHAP** (Model-agnostic)
   - Slower but works with any model
   - Not needed for CatBoost

3. **SHAP Interaction Values** (Cool extension)
   - Shows feature interactions
   - Computationally expensive
   - **Consider if time permits**

4. **Approximate SHAP** (Fast sampling)
   - Faster but less accurate
   - Use if 10,000 computations too slow

---

## Execution Plan

### Step 1: Clarify Requirements (CURRENT)

See questions below.

### Step 2: Extract Top-10 Models

```bash
python scripts/extract_top_models_from_mlflow.py \
    --output data/artifacts/top10_catboost_models.pkl \
    --top-n 10 \
    --include-features \
    --include-models
```

### Step 3: Compute SHAP Values

New script: `scripts/compute_shap_with_uncertainty.py`

### Step 4: Compute VIF

Add to existing analysis flow or new script.

### Step 5: Generate Figures

Use ggplot2 (R) or matplotlib for publication quality.

---

## FINAL Top-10 Configurations (Updated 2026-01-25)

**DECISION**: Rank 5 EXCLUDED (unknown "anomaly" OD source), Rank 11 promoted.

| Rank | Outlier Detection | Imputation | AUROC (95% CI) | Status |
|------|-------------------|------------|----------------|--------|
| 1 | Ensemble (all methods) | CSDI | 0.913 (0.904--0.919) | ✅ INCLUDED |
| 2 | Ensemble (all methods) | TimesNet | 0.912 (0.902--0.921) | ✅ INCLUDED |
| 3 | EnsembleThresh (M+T+U) | CSDI | 0.912 (0.904--0.918) | ✅ INCLUDED |
| 4 | EnsembleThresh (M+T+U) | TimesNet | 0.911 (0.903--0.919) | ✅ INCLUDED |
| ~~5~~ | ~~anomaly (UNKNOWN)~~ | ~~Ensemble (C+M+S+T)~~ | ~~0.911~~ | ❌ EXCLUDED |
| 6 | **Ground-truth** | **Ground-truth** | 0.911 (0.903--0.918) | ✅ INCLUDED |
| 7 | Ground-truth | Ensemble (C+M+S) | 0.911 (0.903--0.918) | ✅ INCLUDED |
| 8 | Ensemble (all methods) | SAITS | 0.910 (0.903--0.917) | ✅ INCLUDED |
| 9 | Ground-truth | CSDI | 0.910 (0.903--0.917) | ✅ INCLUDED |
| 10 | MOMENT fine-tuned | SAITS | 0.910 (0.899--0.921) | ✅ INCLUDED |
| 11→10 | pupil-gt | TimesNet | 0.909 (0.902--0.916) | ✅ PROMOTED |

**Why Rank 5 excluded**: `mlflow_run_outlier_detection: None`, `Outlier_f1: nan` - actual OD source unclear.
This is a legacy placeholder from parsing code (see `docs/mlflow-naming-convention.md`).

**All 10 configs use CatBoost classifier and handcrafted features.**
(M=MOMENT, T=TimesNet, U=UniTS, C=CSDI, S=SAITS)

**Key observation**: Ground-truth preprocessing (rank 6) is matched or exceeded by
ensemble foundation model approaches (ranks 1-4), demonstrating FM effectiveness.

---

## User Decisions (2026-01-25)

| Question | Answer |
|----------|--------|
| Bootstrap sampling | **All 1000 iterations x 10 configs** - no shortcuts, save raw to .db |
| Artifact format | **DuckDB .db** |
| Visualization | **R + ggplot2** with JSON data for language-agnostic reproducibility |
| Ensemble weighting | **Both equal and AUROC-weighted** |

---

## Clarification Questions

Before proceeding, the following questions need answers:

### Q1: Bootstrap Sampling Strategy

For 10 configs × 1000 bootstrap iterations = 10,000 SHAP computations:

- **Option A**: Use all 1000 bootstrap iterations (slow but complete)
- **Option B**: Sample 100 iterations per config (faster, still captures uncertainty)
- **Option C**: Use single "best" model per config + standard SHAP (no uncertainty)

### Q2: Artifact Format Preference

For the top-10 model artifact:

- **Option A**: Single `.pkl` file with all data (simple, large)
- **Option B**: DuckDB `.db` with structured tables (queryable)
- **Option C**: Split files: `models.pkl` + `features.parquet` (modular)

### Q3: ggplot2 vs Python Visualization

For publication figures:

- **Option A**: Use R + ggplot2 (publication standard, requires R interop)
- **Option B**: Use Python matplotlib/seaborn (simpler, already in pipeline)
- **Option C**: Hybrid: ggplot2 for main, matplotlib for supplementary

### Q4: Ensemble Weighting

For aggregating across top-10 configs:

- **Option A**: Equal weights (all configs contribute equally)
- **Option B**: AUROC-weighted (better configs contribute more)
- **Option C**: Both (show comparison)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SHAP computation too slow | Medium | High | Sample bootstrap iterations |
| Memory constraints | Medium | Medium | Process configs sequentially |
| Model loading failures | Low | Medium | Error handling, skip missing |
| VIF collinearity issues | Low | Low | Document if features are collinear |

---

## Review Agent Evaluation

### Reviewer 1: Statistical Rigor

**Concern**: Propagating SHAP uncertainty across bootstrap requires careful aggregation.

**Recommendation**:
- Use hierarchical model: Config → Bootstrap → SHAP
- Report both within-config and across-config variance

### Reviewer 2: Reproducibility

**Concern**: Large artifacts may not version well.

**Recommendation**:
- Store in `data/artifacts/` not `figures/generated/`
- Add to `.gitignore` if >100MB
- Document regeneration command

### Reviewer 3: Visualization Quality

**Concern**: ggplot2 requires R interop.

**Recommendation**:
- Use existing `rpy2` infrastructure
- Or use `plotnine` (Python ggplot2 clone)
- Ensure consistent styling with existing figures

---

## Next Steps

1. **Answer clarification questions** (multi-choice format provided)
2. **Validate extraction script** works for top-10
3. **Implement SHAP computation** with selected sampling strategy
4. **Generate figures** in chosen framework
5. **Integrate into analysis flow** (Block 2)

---

*Document generated: 2026-01-25*
*Plan status: AWAITING CLARIFICATION*
