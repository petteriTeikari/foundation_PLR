# Variance Decomposition Figure Redesign

## Status: PLANNED

## Current State

`fig04_variance_decomposition.png` shows η² for preprocessing factors (CatBoost fixed):
- Outlier Method: 69.0%
- Imputation Method: 5.0%
- Interaction: 26.0%

**Problem**: Vertical bar chart is not optimal for showing variance decomposition.

## User Request

> "The use of bars is not very useful I think for fig04_variance_decomposition.png and what do you think if we modify the plotting approach (any ideas), and if we make it two-column as well if on the left if we only address the variance from outlier detection method to AUROC, and on the right include the effect of classifier model which should show then that the classifier model choice dominates"

## Proposed Redesign

### Two-Panel Layout (1x2)

**Panel A (Left): Preprocessing Effects Only** (CatBoost fixed)
- Factors: Outlier Method, Imputation Method, Interaction
- Shows: Within-preprocessing variance decomposition
- Message: "Outlier method choice dominates within preprocessing"

**Panel B (Right): Full Pipeline Including Classifier**
- Factors: Classifier, Outlier Method, Imputation Method, interactions
- Shows: Full variance decomposition including classifier
- Message: "Classifier choice dominates overall, but preprocessing still matters"

### Visualization Alternatives (Ranked by Preference)

#### Option 1: Horizontal Lollipop Chart (RECOMMENDED)
```
                               η² (% variance explained)
                    0%   20%   40%   60%   80%  100%
                    |----|----|----|----|----|----|
Outlier Method      ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━●  69.0%
Interaction         ●━━━━━━━━━━━━━━━━━━━●                26.0%
Imputation Method   ●━━●                                  5.0%
```
Pros: Clean, easy to compare, publication-standard
Cons: None significant

#### Option 2: Treemap / Proportional Area
```
┌──────────────────────────────────┬────────┐
│                                  │        │
│       Outlier Method             │ Inter- │
│          69.0%                   │ action │
│                                  │ 26.0%  │
│                                  │        │
├──────────────────────────────────┼────────┤
│ Imputation 5.0%                  │        │
└──────────────────────────────────┴────────┘
```
Pros: Intuitive proportional representation
Cons: Harder to read exact values

#### Option 3: Stacked Horizontal Bar (100%)
```
┌─────────────────────────────────────────────────────┐
│███████████████████████████████████░░░░░░░░░▒▒▒▒▒▒▒▒│
│    Outlier Method (69%)           │Int (26%)│Imp(5%)│
└─────────────────────────────────────────────────────┘
```
Pros: Shows total = 100% explicitly
Cons: Small segments hard to read

#### Option 4: Donut/Pie Chart
Pros: Familiar
Cons: Criticized in scientific literature, hard to compare precisely

### Implementation Plan

1. **Compute full variance decomposition** including classifier:
   ```r
   # Full ANOVA with classifier as factor
   model_full <- aov(auroc ~ classifier * outlier_method * imputation_method, data = metrics_all)
   ```

2. **Create horizontal lollipop visualization**:
   ```r
   # Use geom_point + geom_segment for lollipop
   ggplot(eta_sq, aes(x = eta_squared, y = reorder(factor, eta_squared))) +
     geom_segment(aes(x = 0, xend = eta_squared, yend = factor)) +
     geom_point(size = 4, color = factor_color) +
     geom_text(aes(label = sprintf("%.1f%%", eta_squared * 100)), hjust = -0.3)
   ```

3. **Compose 1x2 combined figure**:
   ```r
   fig_variance_combined <- compose_figures(
     list(p_preprocessing, p_full),
     layout = "1x2",
     tag_levels = "A",
     panel_titles = c("Preprocessing Effects (CatBoost)", "Full Pipeline")
   )
   ```

### Expected Message

- **Panel A**: "Within CatBoost, outlier method choice explains 69% of AUROC variance"
- **Panel B**: "Across all pipelines, classifier choice dominates (e.g., η²~50%), but preprocessing still explains significant variance"

This two-panel design directly supports the manuscript narrative:
> "Classifier choice matters most, but preprocessing choices propagate meaningful effects to downstream performance"

## Files to Modify

1. `src/r/figures/fig04_variance_decomposition.R` - Complete rewrite
2. `configs/VISUALIZATION/figure_layouts.yaml` - Add `fig_variance_combined` entry

## Data Requirements

- `outputs/r_data/essential_metrics.csv` - Already exists, contains all classifiers
- Need to ensure classifier column is properly populated

## Validation

- [ ] Panel A values match current figure (69%, 5%, 26%)
- [ ] Panel B shows classifier effect > preprocessing effects
- [ ] Lollipop chart is readable at publication size
- [ ] Figure passes QA tests
