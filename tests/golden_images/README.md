# Golden Images for Visual Regression Testing

This directory contains approved baseline figures for visual regression testing.

## Baseline Creation

- **Date**: 2026-01-29
- **Source**: `figures/generated/`
- **Created by**: Automated Phase 2 execution

## Contents

| File | Description | Last Updated |
|------|-------------|--------------|
| fig_imputation_quality.png | Imputation method comparison | 2026-01-29 |
| fig_outlier_detection_quality.png | Outlier detection quality | 2026-01-29 |
| fig_preprocessing_summary.png | Preprocessing pipeline summary | 2026-01-29 |

## Usage

Visual regression tests compare newly generated figures against these baselines
using perceptual hashing (phash). If a figure differs by more than the threshold
(Hamming distance > 15), the test fails.

## Updating Golden Images

To update a golden image after an intentional change:

```bash
# Update specific figure
cp figures/generated/fig_name.png tests/golden_images/

# Or update all
cp figures/generated/*.png tests/golden_images/
```

After updating, document the reason in this README.

## Why Golden Images?

1. **Detect unintentional changes**: Code changes may silently affect figures
2. **Catch rendering failures**: Font issues, color changes, dimension problems
3. **Enforce review**: Any visual change requires explicit approval

## Notes

- PDFs are not included (perceptual hash works on raster images)
- Threshold of 15 allows for minor anti-aliasing differences
- New figures won't fail (no golden = skip test)
