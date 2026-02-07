# CRITICAL-FAILURE-003: Legend Takes 50% of Figure Width

## Incident Date: 2026-01-27

## Summary

Claude generated `fig_selective_classification.png` where the legend occupies approximately 50% of the total figure width, leaving insufficient space for the actual data visualization. User response: "What the fuck serious Claude Opus 4.5?"

## Root Cause Analysis

1. **Default ggplot2 legend placement**: Legend placed to the right with automatic sizing
2. **Long legend labels**: Pipeline names like "ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune" are extremely long
3. **No legend proportion check**: No validation that legend doesn't dominate the figure
4. **Fixed figure width**: Width was set without accounting for legend space requirements

## What Went Wrong

```r
# Problematic pattern:
p <- ggplot(data, aes(x = coverage, y = auroc, color = config)) +
  geom_line() +
  theme(legend.position = "right")  # Default, no size constraint

# Long config names create wide legend:
# "ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune"
# This text pushes the legend to take ~50% of horizontal space
```

## Visual Example

```
|<------ Figure Width ------>|
|  Plot Area   |    Legend   |
|     50%      |     50%     |  <- WRONG: Legend too wide
|              |             |
```

Should be:
```
|<------ Figure Width ------>|
|      Plot Area       | Leg |
|         85%          | 15% |  <- CORRECT: Legend compact
```

## Why This Is Critical

- **Wastes visual real estate**: Half the figure shows no data
- **Poor publication quality**: Reviewers would reject this layout
- **User frustration**: Repeated issues indicate pattern of carelessness

## Required Fixes

### 1. Add Legend Proportion Test

```r
test_that("Legends do not exceed 25% of figure width", {
  # This requires analyzing the saved plot or using ggplot internals
  # Approximation: check for very long legend labels

  figure_scripts <- list.files("src/r/figures", pattern = "^fig_.*\\.R$", full.names = TRUE)

  for (script in figure_scripts) {
    content <- paste(readLines(script), collapse = "\n")

    # Check for legend position handling
    has_legend_control <- grepl("legend\\.position.*=.*(\"bottom\"|\"none\"|\"inside\")", content) ||
                          grepl("guides\\(.*=.*guide_legend\\(ncol", content) ||
                          grepl("legend\\.key\\.width", content)

    # If figure has color/fill aesthetics, it should have legend control
    has_color_aes <- grepl("aes\\([^)]*color\\s*=", content) ||
                     grepl("aes\\([^)]*fill\\s*=", content)

    if (has_color_aes) {
      expect_true(has_legend_control,
        info = paste("Script has color/fill but no legend size control:", basename(script)))
    }
  }
})
```

### 2. Legend Best Practices for This Project

```r
# OPTION 1: Bottom legend with multiple columns
theme(legend.position = "bottom") +
guides(color = guide_legend(ncol = 2))

# OPTION 2: Abbreviated legend labels
scale_color_manual(
  values = colors,
  labels = c(
    "ensemble-LOF-MOMENT-..." = "Ensemble (Full)",
    "pupil-gt" = "Ground Truth",
    ...
  )
)

# OPTION 3: Inside plot legend (if space allows)
theme(
  legend.position = c(0.85, 0.15),  # Bottom-right inside plot
  legend.background = element_rect(fill = "white", alpha = 0.8)
)

# OPTION 4: No legend, use direct labels
library(ggrepel)
geom_text_repel(aes(label = config), data = endpoints_only)
```

### 3. Project-Wide Legend Guidelines

Add to `src/r/theme_foundation_plr.R`:

```r
# Standard legend configurations
LEGEND_BOTTOM_2COL <- theme(legend.position = "bottom") +
  guides(color = guide_legend(ncol = 2), fill = guide_legend(ncol = 2))

LEGEND_INSIDE_BR <- theme(
  legend.position = c(0.85, 0.15),
  legend.background = element_rect(fill = alpha("white", 0.8)),
  legend.key.size = unit(0.8, "lines")
)

# Use abbreviated names for pipelines
PIPELINE_LABELS_SHORT <- c(
  "pupil-gt" = "GT",
  "ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune" = "Ensemble",
  "MOMENT-gt-finetune" = "MOMENT",
  ...
)
```

## Behavioral Changes Required

1. **ALWAYS consider legend placement** when creating figures with multiple categories
2. **USE abbreviated labels** for long pipeline/config names
3. **PREFER bottom legends** for figures with many categories
4. **TEST visually** before declaring figure complete
5. **CHECK figure proportions** - data should dominate, not chrome

## Related Issues

- Similar problem likely in other figures with pipeline comparisons
- Need audit of all figures with color legends

## Lessons Learned

- Default ggplot2 settings are not publication-ready
- Long categorical labels need abbreviation strategy
- Legend placement is a design decision, not an afterthought
- "It generated without errors" is NOT a quality bar
