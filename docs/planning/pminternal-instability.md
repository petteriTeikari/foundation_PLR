# Planning: pminternal Prediction Instability Plots

## Overview

Implement Riley 2023-style prediction instability plots using the `pminternal` R package with Economist-style aesthetics.

**Reference**: Riley RD et al. (2023) "Clinical prediction models and the multiverse of madness." BMC Medicine.

**R Package**: https://cran.r-project.org/web/packages/pminternal/

## What Prediction Instability Plots Show

The plot visualizes how individual predictions vary across bootstrap samples:

- **X-axis**: Predicted risk from the developed (original) model
- **Y-axis**: Predicted risk from bootstrap models (one point per bootstrap per patient)
- **Diagonal line**: Perfect agreement (prediction unchanged across bootstraps)
- **Dashed lines**: 2.5th and 97.5th percentiles (95% CI of instability)
- **Vertical spread**: Indicates prediction instability for that patient

### Interpretation

- **Narrow vertical spread** → Stable predictions (trustworthy)
- **Wide vertical spread** → Unstable predictions (flag for clinical review)
- Points far from diagonal → Predictions differ substantially between original and bootstrap models

## Data Requirements

We need per-patient predictions from:
1. **Original model**: The model trained on full data
2. **Bootstrap models**: Models trained on B bootstrap resamples

### Current Data Availability

From MLflow, we have:
- `y_true`: Ground truth labels per patient
- `y_prob`: Predictions from the developed model
- **Missing**: Per-bootstrap predictions

### Data Extraction Plan

**Option A**: Re-run bootstrap predictions (expensive but complete)
```python
# For each bootstrap iteration:
for b in range(n_bootstrap):
    X_boot, y_boot = resample(X, y)
    model_boot = train_model(X_boot, y_boot)
    y_prob_boot[b, :] = model_boot.predict_proba(X)[:, 1]  # Predict on ORIGINAL X
```

**Option B**: Use existing bootstrap data from MLflow
- Check if per-iteration predictions were stored
- Path: `/home/petteri/mlruns/253031330985650090/*/artifacts/`

## Implementation Plan

### Phase 1: Data Extraction

1. **Check existing MLflow data** for per-bootstrap predictions
   ```bash
   # Check structure of MLflow artifacts
   ls /home/petteri/mlruns/253031330985650090/*/artifacts/
   ```

2. **If not available**, create extraction script:
   ```python
   # scripts/extract_bootstrap_predictions.py
   # - Load trained model from MLflow
   # - Perform B bootstrap resamples
   # - Store predictions matrix (n_patients × B)
   ```

3. **Export to JSON** for R consumption:
   ```json
   {
     "config_id": "ground_truth",
     "n_patients": 208,
     "n_bootstrap": 200,
     "y_prob_original": [...],
     "y_prob_bootstrap": [[...], [...], ...]  // n_bootstrap × n_patients
   }
   ```

### Phase 2: R Implementation

1. **Create R script**: `src/r/figures/fig_instability_plot.R`

2. **Key components**:
   ```r
   # Load pminternal
   library(pminternal)

   # Or create custom ggplot2 version for Economist styling
   create_instability_plot <- function(y_prob_original, y_prob_bootstrap) {
     # y_prob_original: vector of length n_patients
     # y_prob_bootstrap: matrix (n_bootstrap × n_patients)

     # Create long-format data for ggplot
     df <- data.frame(
       original = rep(y_prob_original, each = nrow(y_prob_bootstrap)),
       bootstrap = as.vector(t(y_prob_bootstrap)),
       patient_id = rep(1:length(y_prob_original), each = nrow(y_prob_bootstrap))
     )

     # Compute percentile bands
     percentiles <- apply(y_prob_bootstrap, 2, function(x) {
       c(lo = quantile(x, 0.025), hi = quantile(x, 0.975))
     })

     # Create Economist-style plot
     ggplot(df, aes(x = original, y = bootstrap)) +
       # Diagonal reference
       geom_abline(slope = 1, intercept = 0, color = "#333333", linewidth = 0.5) +
       # Bootstrap predictions (gray points)
       geom_point(alpha = 0.05, size = 0.5, color = "#666666") +
       # 95% CI bands
       geom_line(data = percentile_df, aes(y = lo), linetype = "dashed", color = "#006BA2") +
       geom_line(data = percentile_df, aes(y = hi), linetype = "dashed", color = "#006BA2") +
       # Economist theme
       theme_foundation_plr() +
       labs(
         x = "Predicted risk from developed model",
         y = "Predicted risk from bootstrap models"
       )
   }
   ```

### Phase 3: Multi-Panel Figure

Create 2-panel figure comparing:
- **Panel A**: Ground truth preprocessing (should show stability)
- **Panel B**: Best automated preprocessing (may show more instability)

```r
# src/r/figures/fig_instability_combined.R
p_gt <- create_instability_plot(gt_data) +
  labs(subtitle = "Ground Truth Preprocessing")

p_auto <- create_instability_plot(auto_data) +
  labs(subtitle = "Automated Preprocessing")

combined <- (p_gt | p_auto) +
  plot_annotation(tag_levels = "A")

save_publication_figure(combined, "fig_instability_combined", width = 12, height = 6)
```

## pminternal Package Functions

Key functions from pminternal:

| Function | Purpose |
|----------|---------|
| `pminternal::validate()` | Main validation function |
| `pminternal::pred_instability()` | Compute prediction instability metrics |
| `pminternal::plot_pred_instability()` | Create instability plot |
| `pminternal::mape_summary()` | Mean Absolute Prediction Error |

### Using pminternal Directly

```r
library(pminternal)

# If we have the model object and data
result <- validate(
  model = trained_model,
  data = validation_data,
  outcome = "class_label",
  B = 200  # Bootstrap iterations
)

# Extract instability plot
plot_pred_instability(result)
```

### Custom Implementation (Preferred for Styling)

Since pminternal uses base R graphics, we'll create a custom ggplot2 version that:
1. Takes the same data structure as pminternal
2. Produces Economist-style output
3. Integrates with our figure system

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `fig_instability_combined.png` | `figures/generated/ggplot2/supplementary/` | 2-panel instability plot |
| `instability_data.json` | `outputs/r_data/` | Bootstrap predictions data |
| `fig_instability_combined.md` | `figures/generated/ggplot2/supplementary/` | Explainer document |

## Timeline

1. **Data check** (30 min): Verify if bootstrap predictions exist in MLflow
2. **Extraction script** (2 hrs): If needed, create bootstrap prediction extractor
3. **R implementation** (2 hrs): Create ggplot2 instability plot with Economist styling
4. **Integration** (1 hr): Add to figure generation pipeline and YAML config

## Success Criteria

- [ ] Bootstrap predictions extracted for ≥2 configs (GT, best automated)
- [ ] Instability plot matches Riley 2023 Figure 2 style
- [ ] Economist colors (off-white bg, blue accents)
- [ ] 2-panel figure comparing GT vs automated preprocessing
- [ ] JSON data saved for reproducibility
- [ ] Added to figure_layouts.yaml under supplementary

## References

1. Riley RD et al. (2023) "Clinical prediction models and the multiverse of madness." BMC Medicine. DOI: 10.1186/s12916-023-02849-7

2. Rhodes SA et al. (2025) "pminternal: Internal Validation of Clinical Prediction Models." CRAN. https://cran.r-project.org/package=pminternal

3. Van Calster B et al. (2024) "Performance evaluation of predictive AI models." STRATOS TG6.
