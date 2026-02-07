# fig-repo-27e: How to Read Mean |SHAP| Importance Plots

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-27e |
| **Title** | How to Read Mean |SHAP| Importance Plots |
| **Complexity Level** | L2-L3 (Statistical visualization + methodology) |
| **Target Persona** | All |
| **Location** | docs/concepts-for-researchers.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Teach readers how to interpret Mean |SHAP| (absolute SHAP value) bar plots with bootstrap confidence intervals—the format used in our paper for feature importance analysis.

## Key Message

"Mean |SHAP| bar plots show average feature importance across all predictions. Unlike beeswarm plots, they aggregate to a single importance value per feature with confidence intervals. Higher mean |SHAP| = more influential feature. But beware: correlated features still share credit unpredictably."

## Literature Foundation

| Source | Key Contribution |
|--------|------------------|
| Lundberg & Lee 2017 | NIPS - SHAP (SHapley Additive exPlanations) |
| Molnar 2019 | Interpretable Machine Learning book |
| TreeExplainer | Efficient exact SHAP for tree-based models |

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   HOW TO READ MEAN |SHAP| IMPORTANCE PLOTS                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  WHAT IS A MEAN |SHAP| PLOT?                                                    │
│  ═══════════════════════════                                                    │
│                                                                                 │
│  A bar chart showing the AVERAGE ABSOLUTE contribution of each feature         │
│  to model predictions, with confidence intervals from bootstrap resampling.    │
│                                                                                 │
│  KEY INSIGHT: Unlike beeswarm plots, this AGGREGATES across all samples:       │
│  • Beeswarm: Shows EVERY sample as a dot (detailed but complex)                │
│  • Mean |SHAP| bar: Shows ONE number per feature (simplified summary)          │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ANATOMY OF A MEAN |SHAP| PLOT                                                  │
│  ═════════════════════════════                                                  │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  A  Ground Truth Pipeline                                               │   │
│  │                                                                         │   │
│  │  Blue_PHASIC_value      ●────────────────────┤     ← Most important    │   │
│  │                                                                         │   │
│  │  Red_PHASIC_value       ●───────────────────┤      ← 95% CI error bar  │   │
│  │                                                                         │   │
│  │  Blue_SUSTAINED_value   ●──────────────────┤                           │   │
│  │                                                                         │   │
│  │  Blue_MAX_CONSTRICTION  ●─────────────────┤                            │   │
│  │                                                                         │   │
│  │  Red_SUSTAINED_value    ●────────────────┤                             │   │
│  │                                                                         │   │
│  │  ...                                                                    │   │
│  │                                                                         │   │
│  │  Red_PIPR_AUC_value     ●────┤                  ← Least important      │   │
│  │                                                                         │   │
│  │                         ├────┼────┼────┼────┤                          │   │
│  │                        0.00 0.05 0.10 0.15 0.20                         │   │
│  │                                                                         │   │
│  │                         Mean |SHAP| value                               │   │
│  │                                                                         │   │
│  │  Legend: ● Blue (469nm) - Melanopsin pathway                           │   │
│  │          ● Red (640nm)  - Cone pathway                                 │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  READING THE PLOT                                                               │
│  ════════════════                                                               │
│                                                                                 │
│  1. BAR LENGTH (Feature Importance)                                             │
│     ───────────────────────────────                                             │
│     • Longer bar = higher mean |SHAP| = more important feature                  │
│     • Importance = average of absolute contributions across ALL predictions    │
│     • Uses ABSOLUTE value because positive/negative both matter                 │
│                                                                                 │
│  2. ERROR BARS (Confidence Intervals)                                           │
│     ─────────────────────────────────                                           │
│     • Show 95% CI from bootstrap resampling (B=1000 iterations)                │
│     • Overlapping CIs → features NOT significantly different in importance     │
│     • Non-overlapping CIs → significantly different importance                  │
│                                                                                 │
│  3. FEATURE ORDERING (Y-axis)                                                   │
│     ──────────────────────────                                                  │
│     • Features sorted by mean |SHAP| (most important at top)                   │
│     • Ordering may differ between pipelines (compare both panels!)             │
│                                                                                 │
│  4. COLOR CODING (Physiological Grouping)                                       │
│     ─────────────────────────────────────                                       │
│     • Blue (469nm): Melanopsin pathway features (ipRGC response)               │
│     • Red (640nm): Cone pathway features (photoreceptor response)              │
│     • Grouping reveals which pathway drives glaucoma detection                  │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHY ABSOLUTE VALUE |SHAP|?                                                     │
│  ═════════════════════════                                                      │
│                                                                                 │
│  SHAP values can be positive or negative:                                       │
│  • Positive SHAP: Feature pushes prediction TOWARD glaucoma                    │
│  • Negative SHAP: Feature pushes prediction TOWARD control                     │
│                                                                                 │
│  For IMPORTANCE ranking, we care about MAGNITUDE, not direction:               │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  Patient A:  SHAP = +0.15  (pushes toward glaucoma)                    │   │
│  │  Patient B:  SHAP = -0.12  (pushes toward control)                     │   │
│  │  Patient C:  SHAP = +0.08  (mild push toward glaucoma)                 │   │
│  │                                                                         │   │
│  │  Mean SHAP = (+0.15 - 0.12 + 0.08) / 3 = +0.037  ← Underestimates!    │   │
│  │  Mean |SHAP| = (0.15 + 0.12 + 0.08) / 3 = 0.117  ← True importance    │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Absolute values prevent positive and negative contributions from canceling.   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  COMPARING PIPELINES (Multi-Panel Figures)                                      │
│  ═════════════════════════════════════════                                      │
│                                                                                 │
│  When comparing across preprocessing pipelines:                                 │
│                                                                                 │
│  ┌────────────────────────────────┐  ┌────────────────────────────────┐       │
│  │  A  Ground Truth Pipeline      │  │  B  Ensemble Pipeline          │       │
│  │                                │  │                                │       │
│  │  Blue_PHASIC      ●─────────┤  │  │  Blue_SUSTAINED  ●─────────┤  │       │
│  │  Red_PHASIC       ●────────┤   │  │  Blue_MAX_CONSTR ●────────┤   │       │
│  │  Blue_SUSTAINED   ●───────┤    │  │  Red_PHASIC      ●───────┤    │       │
│  │  ...                          │  │  ...                          │       │
│  └────────────────────────────────┘  └────────────────────────────────┘       │
│                                                                                 │
│  What to compare:                                                               │
│  • Do the SAME features dominate both pipelines? (robustness)                  │
│  • Does ordering change? (preprocessing affects feature importance)            │
│  • Are CI widths similar? (similar uncertainty)                                │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ⚠️  CRITICAL ASSUMPTION: MULTICOLLINEARITY                                     │
│  ═══════════════════════════════════════════                                    │
│                                                                                 │
│  SHAP has a hidden assumption that is often violated:                           │
│                                                                                 │
│  PROBLEM: When features are correlated, SHAP values are DISTRIBUTED             │
│  between them in arbitrary ways depending on model structure.                   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  Example: Blue_PHASIC and Blue_SUSTAINED are correlated (r = 0.7)      │   │
│  │                                                                         │   │
│  │  True importance:   Blue_PHASIC    ━━━━━━━━━━━━━━━━━━━━                │   │
│  │                     Blue_SUSTAINED ━━━━━━━━━━━━━━━━━━━━                │   │
│  │                     (Both equally important physiologically)            │   │
│  │                                                                         │   │
│  │  SHAP might show:   Blue_PHASIC    ━━━━━━━━━━━━━━━━━━━━━━━━━           │   │
│  │                     Blue_SUSTAINED ━━━━━━━━━━━                         │   │
│  │                     (Credit arbitrarily assigned to one)                │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  SOLUTION: Check VIF (Variance Inflation Factor) BEFORE interpreting SHAP      │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  VIF Threshold    │ Interpretation           │ SHAP Reliability        │   │
│  │  ─────────────────┼──────────────────────────┼─────────────────────────│   │
│  │  VIF < 5          │ Low collinearity         │ ✓ SHAP values reliable  │   │
│  │  VIF 5-10         │ Moderate collinearity    │ ⚠ Interpret with caution│   │
│  │  VIF > 10         │ High collinearity        │ ✗ SHAP values unreliable│   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  COMMON MISTAKES                                                                │
│  ═══════════════                                                                │
│                                                                                 │
│  ❌ "The top feature is the 'true' cause of glaucoma"                           │
│     → SHAP shows predictive importance, not causal importance                   │
│                                                                                 │
│  ❌ "Comparing bar lengths of correlated features"                              │
│     → Credit is arbitrarily split; check VIF first                              │
│                                                                                 │
│  ❌ "Features with overlapping CIs are 'equally important'"                     │
│     → Overlapping CIs mean NOT significantly DIFFERENT, not equal               │
│                                                                                 │
│  ❌ "Mean |SHAP| tells you effect direction"                                    │
│     → Use raw SHAP (not absolute) or beeswarm for direction                    │
│                                                                                 │
│  ❌ "Comparing mean |SHAP| across different models/datasets"                    │
│     → Scale depends on outcome variance; normalize or use same dataset          │
│                                                                                 │
│  ❌ "Low mean |SHAP| = feature should be removed"                               │
│     → Low importance doesn't mean unnecessary; may interact with others         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Implementation in This Repository

### Python Code for Computing Mean |SHAP| with Bootstrap CIs
```python
import shap
import numpy as np
from scipy import stats

def compute_mean_shap_with_ci(model, X, n_bootstrap=1000, ci=0.95):
    """
    Compute mean |SHAP| values with bootstrap confidence intervals.

    Args:
        model: Trained CatBoost/XGBoost model
        X: Feature matrix (pandas DataFrame)
        n_bootstrap: Number of bootstrap iterations
        ci: Confidence interval level

    Returns:
        DataFrame with mean_shap, ci_lo, ci_hi per feature
    """
    # Get SHAP values using TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classification, use class 1 SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Compute mean |SHAP| per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Bootstrap for confidence intervals
    bootstrap_means = []
    n_samples = len(X)

    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        boot_shap = shap_values[idx]
        boot_mean = np.abs(boot_shap).mean(axis=0)
        bootstrap_means.append(boot_mean)

    bootstrap_means = np.array(bootstrap_means)

    # Compute percentile CIs
    alpha = 1 - ci
    ci_lo = np.percentile(bootstrap_means, 100 * alpha / 2, axis=0)
    ci_hi = np.percentile(bootstrap_means, 100 * (1 - alpha / 2), axis=0)

    return pd.DataFrame({
        'feature': X.columns,
        'mean_shap': mean_abs_shap,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi
    }).sort_values('mean_shap', ascending=False)
```

### R Code for Plotting
```r
# Load SHAP summary data
shap_summary <- read_csv("data/shap_importance.csv")

# Add stimulus color
shap_summary <- shap_summary %>%
  mutate(stimulus = case_when(
    str_detect(feature, "^Blue_") ~ "Blue (469nm)",
    str_detect(feature, "^Red_") ~ "Red (640nm)",
    TRUE ~ "Other"
  ))

# Create horizontal bar plot with error bars
ggplot(shap_summary, aes(x = mean_shap, y = reorder(feature, mean_shap))) +
  geom_errorbarh(aes(xmin = ci_lo, xmax = ci_hi, color = stimulus),
                 height = 0.3) +
  geom_point(aes(color = stimulus), size = 3) +
  scale_color_manual(values = c("Blue (469nm)" = "#0072B2",
                                 "Red (640nm)" = "#D55E00")) +
  labs(x = "Mean |SHAP| value",
       y = NULL,
       color = "Stimulus") +
  theme_minimal()
```

## Content Elements

1. **Mean |SHAP| anatomy**: Labeled horizontal bar plot with error bars
2. **Reading guide**: 4 numbered steps (bar length, error bars, ordering, color)
3. **Why absolute value**: Explanation with numeric example
4. **Multi-panel comparison**: How to compare across pipelines
5. **Multicollinearity warning**: VIF check requirement
6. **Common mistakes**: What NOT to conclude

## Text Content

### Title Text
"How to Read Mean |SHAP| Feature Importance Plots"

### Caption
Mean |SHAP| plots show the average absolute SHAP value per feature, sorted by importance. Error bars indicate 95% bootstrap CIs (B=1000). Longer bars = more important features. Color coding groups features by stimulus wavelength (blue 469nm = melanopsin pathway, red 640nm = cone pathway). CRITICAL: Correlated features split SHAP credit unpredictably—check VIF before comparing importance of related features.

## Prompts for Nano Banana Pro

### Style Prompt
Educational diagram explaining Mean |SHAP| bar plot interpretation. Horizontal bars with confidence interval error bars. Multi-panel comparison example. Color-coded by physiological pathway. Clean instructional style.

### Content Prompt
Create a Mean |SHAP| plot reading guide:

**TOP - Anatomy**:
- Horizontal bar chart with labeled parts
- Error bars for 95% CI
- Color-coded by stimulus wavelength

**MIDDLE - Reading Steps**:
- 4 numbered steps with icons
- 1: Bar length = importance
- 2: Error bars = uncertainty
- 3: Y-axis ordering = ranking
- 4: Color = physiological grouping

**MIDDLE - Why |SHAP|**:
- Numeric example showing why absolute value needed
- Positive/negative cancellation problem

**BOTTOM - Multi-Panel Comparison**:
- Side-by-side Ground Truth vs Ensemble
- What to compare across pipelines

**SIDEBAR - VIF Warning**:
- Multicollinearity caution
- VIF threshold table

## Alt Text

Educational diagram explaining Mean |SHAP| feature importance plot interpretation. Shows horizontal bar chart anatomy with bars representing average absolute SHAP values, error bars for 95% bootstrap confidence intervals, and features sorted by importance. Reading guide: (1) bar length shows importance, (2) error bars show uncertainty, (3) y-axis ordering shows ranking, (4) color coding groups by stimulus wavelength (blue 469nm melanopsin vs red 640nm cone pathway). Numeric example demonstrates why absolute values prevent positive/negative cancellation. Multi-panel comparison shows how to compare Ground Truth vs Ensemble pipelines. VIF warning table shows thresholds for interpreting correlated features.

## Status

- [x] Draft created
- [x] Updated to match actual paper figure (Mean |SHAP| bar plot, not beeswarm)
- [ ] Generated
- [ ] Placed in docs/concepts-for-researchers.md

## References

- Lundberg SM, Lee SI (2017) A unified approach to interpreting model predictions. NeurIPS.
- Molnar C (2019) Interpretable Machine Learning. https://christophm.github.io/interpretable-ml-book/
- Lundberg SM, Erion G, et al. (2020) From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence.

## Mean |SHAP| vs Beeswarm: When to Use Which

| Plot Type | Shows | Best For |
|-----------|-------|----------|
| **Mean \|SHAP\| bars** | Single importance value per feature | Publication figures, summary |
| **Beeswarm** | Every sample as a dot with feature value | Detailed exploration, effect direction |
| **Waterfall** | Single prediction breakdown | Explaining individual cases |
| **Dependence** | Feature value vs SHAP value | Non-linear relationships |

**Our paper uses Mean |SHAP| bars** because:
1. Cleaner for publication (less visual clutter)
2. Directly answers "which features are most important?"
3. Bootstrap CIs enable statistical comparison
4. Color coding reveals pathway-level patterns

## PLR Feature Interpretation (Domain Context)

| Feature | Pathway | What It Measures |
|---------|---------|------------------|
| Blue_PHASIC_value | Melanopsin | Initial rapid pupil constriction to 469nm |
| Blue_SUSTAINED_value | Melanopsin | Maintained pupil constriction after stimulus |
| Blue_MAX_CONSTRICTION | Melanopsin | Maximum pupil diameter change |
| Blue_PIPR_AUC_value | Melanopsin | Post-illumination pupil response area |
| Red_* variants | Cone | Same metrics for 640nm (cone pathway) |

**Clinical significance**: Melanopsin-expressing intrinsically photosensitive retinal ganglion cells (ipRGCs) are preferentially affected in glaucoma. Higher importance of Blue_* features aligns with known pathophysiology.
