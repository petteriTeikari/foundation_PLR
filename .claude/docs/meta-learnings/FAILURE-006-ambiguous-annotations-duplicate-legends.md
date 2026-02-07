# FAILURE-006: Ambiguous Annotations and Duplicate Legends

## Severity: HIGH

## The Failures

### 1. Duplicate Legends
`fig_calibration_dca_combined.png` has TWO separate legends (one per panel) instead of ONE shared legend at the bottom.

### 2. Ambiguous Annotation Box
The calibration panel shows:
```
Slope: 0.52, O:E: 0.82
Brier: 0.135, IPA: 0.32
```

**Problem:** With 4 curves on the plot, which curve do these metrics belong to?

This annotation is MISLEADING because:
- It shows metrics for only ONE curve (probably ground truth)
- Reader has no way to know which curve it refers to
- Other curves have different metrics but aren't shown

### 3. Featurization Not From YAML
The featurization filter (`simple1.0` vs `MOMENT-embedding`) should be defined in YAML config, not hardcoded in scripts. This is part of the recurring CRITICAL-FAILURE-002 pattern.

## Evidence

```
fig_calibration_dca_combined.png shows:
- Panel A: 4 curves + 1 annotation box with unknown reference
- Panel B: 4 curves + separate legend
- Each panel has its own legend instead of shared bottom legend
```

## Root Causes

### 1. Legends Not Collected
The patchwork composition doesn't use `guides = "collect"` properly:

```r
# WRONG - separate legends
composed <- (p_cal | p_dca)

# CORRECT - shared legend at bottom
composed <- (p_cal | p_dca) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")
```

### 2. Annotation Box Shows Single Curve Metrics
The annotation was designed for single-curve plots. With multi-curve plots, it becomes ambiguous.

**Options:**
1. Remove the annotation entirely (metrics in table elsewhere)
2. Add per-curve annotations (cluttered)
3. Only annotate ground truth with clear label
4. Use separate metrics panel/table

### 3. No YAML Config for Featurization
Should have:
```yaml
# configs/VISUALIZATION/data_filters.yaml
classification_figures:
  featurization: "simple1.0"  # Handcrafted features
  classifier: "CatBoost"
```

Instead, scripts have hardcoded or missing filters.

## Impact

- Misleading figure with ambiguous metrics
- Duplicate visual elements (legends)
- Data integrity compromised by missing featurization filter
- User frustration at repeated failures

## Fixes Required

### 1. Fix Legend Composition
```r
# In fig_calibration_dca_combined.R
composed <- (p_cal_titled | p_dca_titled) +
  plot_layout(guides = "collect") &
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal"
  )
```

### 2. Fix Annotation Box
Either:
- **Option A**: Remove annotation entirely (recommended for multi-curve)
- **Option B**: Change to "Ground Truth: Slope=0.52, O:E=0.82"
- **Option C**: Add small metrics table below figure

### 3. Create YAML Config for Data Filters
```yaml
# configs/VISUALIZATION/data_filters.yaml
default_filters:
  featurization: "simple1.0"
  classifier: "CatBoost"

# Override per-figure if needed
fig_embedding_comparison:
  featurization: ["simple1.0", "MOMENT-embedding"]
```

## Checklist

- [x] Fix legend collection in fig_calibration_dca_combined.R (already had correct `guides = "collect"`)
- [x] Remove or clarify ambiguous annotation box - **FIXED: Now shows "Ground Truth:" prefix**
- [x] Create configs/VISUALIZATION/data_filters.yaml
- [x] Update ALL export scripts to read featurization from YAML
- [x] Regenerate figure
- [x] Verify single shared legend at bottom

## Resolution (2026-01-28)

The annotation box in `fig_calibration_dca_combined.R` was updated to clearly label
which curve's metrics are being shown:

```r
# Before: Ambiguous - didn't say which curve
annotation_text <- sprintf(
  "Slope: %.2f, O:E: %.2f\nBrier: %.3f, IPA: %.2f", ...
)

# After: Clear - identifies Ground Truth
annotation_text <- sprintf(
  "Ground Truth:\nSlope: %.2f, O:E: %.2f\nBrier: %.3f, IPA: %.2f", ...
)
```

See: `.claude/planning/data-integrity-fixes-2026-01-28.md` for full audit trail.

## Cross-Reference

- CRITICAL-FAILURE-002: Mixed featurization in extraction
- FAILURE-005: Featurization not filtered (recurring)
- Planning: `yet-another-reproducibility-pipeline-data-check-not-probably-the-last-one.md`
