# CRITICAL-FAILURE-002: Blank Figure Generation

## Incident Date: 2026-01-27

## Summary

Claude generated `fig_shap_importance_multi.png` which was completely blank (all white, no content). This figure passed through all existing validation and was committed to the repository.

## Root Cause Analysis

1. **No content validation**: The figure generation script ran without error, but produced an empty plot
2. **Silent failure**: No error was raised when data was missing or empty
3. **No visual inspection**: Claude did not verify the output before declaring success
4. **Inadequate tests**: Existing TDD tests only checked for:
   - File existence
   - Correct directory routing
   - Color compliance
   - They did NOT check for actual visual content

## What Went Wrong

```r
# Likely scenario in fig_shap_importance.R:
# Data was empty or filtered to zero rows, but ggplot silently created empty plot
shap_data <- load_shap_data() %>% filter(config %in% selected_configs)
# If selected_configs doesn't match any data, shap_data has 0 rows
# ggplot still creates a valid (but empty) figure
p <- ggplot(shap_data, aes(...)) + geom_point()  # Empty plot, no error
ggsave("fig_shap_importance_multi.png", p)  # Saves blank figure
```

## Why This Is Critical

- **Scientific fraud risk**: A blank figure in a publication would be unacceptable
- **Trust erosion**: User explicitly asked "How is it possible that you let through a figure like this?"
- **Quality gate failure**: All validation passed for a completely unusable figure

## Required Fixes

### 1. Add Content Validation Test

```r
test_that("All PNG figures have visual content (not blank)", {
  png_files <- list.files("figures/generated/ggplot2",
                          pattern = "\\.png$",
                          recursive = TRUE,
                          full.names = TRUE)

  for (png_file in png_files) {
    # Check file size (blank PNGs are typically < 5KB)
    file_size <- file.info(png_file)$size
    expect_gt(file_size, 10000,  # 10KB minimum
      info = paste("Figure appears blank (too small):", basename(png_file)))

    # Check pixel variance (requires png package)
    if (requireNamespace("png", quietly = TRUE)) {
      img <- png::readPNG(png_file)
      pixel_variance <- var(as.vector(img))
      expect_gt(pixel_variance, 0.001,
        info = paste("Figure appears blank (no pixel variance):", basename(png_file)))
    }
  }
})
```

### 2. Add Data Validation in Figure Scripts

```r
# BEFORE creating any figure:
if (nrow(data) == 0) {
  stop("ERROR: No data available for figure. Check data loading and filtering.")
}

# Or use assertthat:
assertthat::assert_that(nrow(data) > 0,
  msg = "Cannot create figure: data has zero rows")
```

### 3. Add Pre-Commit Hook for Figure Validation

```bash
# .git/hooks/pre-commit addition
for fig in figures/generated/ggplot2/**/*.png; do
  size=$(stat -f%z "$fig" 2>/dev/null || stat -c%s "$fig")
  if [ "$size" -lt 10000 ]; then
    echo "ERROR: $fig appears to be blank (size: $size bytes)"
    exit 1
  fi
done
```

## Behavioral Changes Required

1. **NEVER declare figure generation complete without visual inspection**
2. **ALWAYS verify data exists before plotting**
3. **ALWAYS check generated file sizes are reasonable**
4. **ADD assertion for minimum data rows in every figure script**

## Lessons Learned

- Silent failures are worse than loud failures
- Tests must verify OUTPUT quality, not just OUTPUT existence
- "Script ran without errors" ≠ "Figure is correct"
- Visual inspection is mandatory, not optional

## Related Incidents

- CRITICAL-FAILURE-001: Synthetic data in calibration figures (similar pattern of silent failure)
