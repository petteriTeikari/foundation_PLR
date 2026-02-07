# Imputation Methods for PLR Ground Truth

R implementations for imputing missing values in PLR signals after outlier removal.

## Overview

After outlier detection marks blinks and artifacts as missing (NA), imputation reconstructs the underlying pupil size trend. The ground truth imputation used **MissForest** with human verification.

## Primary Method: MissForest

### Algorithm Description

MissForest is a non-parametric imputation method using Random Forests:

1. **Initialization**: Fill missing values with column means
2. **Iteration**: For each variable with missing values:
   - Train Random Forest on observed values
   - Predict missing values using the trained model
3. **Convergence**: Repeat until OOB error stabilizes or max iterations reached

### Ground Truth Parameters

```r
impute.with.MissForest(
  vars_as_matrices,
  pupil_col = 'pupil_toBeImputed',
  miss_forest_parallelize = 'variables'
)
```

Parameters used for ground truth creation:
- `maxiter = 10` - Maximum iterations (default in missForest)
- `ntree = 100` - Trees per forest (default)
- `parallelize = 'variables'` - Parallel processing mode

### Execution Time

MissForest is computationally expensive:
- ~2 hours for the full PLR dataset (507 subjects)
- Processing done subject-by-subject

### OOB Error Interpretation

The Out-of-Bag (OOB) error indicates imputation quality:

| OOB Error | Quality | Interpretation |
|-----------|---------|----------------|
| < 0.05 | Excellent | Imputation closely matches true values |
| 0.05 - 0.10 | Good | Acceptable reconstruction |
| 0.10 - 0.20 | Moderate | May introduce artifacts |
| > 0.20 | Poor | Consider alternative methods |

Example from our dataset: `OOBerror = 0.0566` (Good quality)

### Code Location

```r
# Main wrapper function
source("lowLevel_imputation_wrappers.R")

# Usage
library(missForest)
library(doParallel)

result <- impute.with.MissForest(data_matrices)
imputed_data <- result[[1]]
oob_error <- result[[2]]
```

## Alternative Methods (imputeTS)

For simpler or faster imputation, the `imputeTS` package offers:

### Kalman Smoothing (Recommended Alternative)

```r
library(imputeTS)

# Using StructTS model (structural time series)
y_imputed <- na.kalman(ts_pupil, model = "StructTS")

# Using auto.arima model
y_imputed <- na.kalman(ts_pupil, model = "auto.arima")
```

### Linear Interpolation

```r
y_imputed <- na.interpolation(ts_pupil)
```

### Seasonal Decomposition

```r
y_imputed <- na.seadec(ts_pupil)
```

### Method Selection Guidance

From imputeTS documentation:
> "In general, for most time series one algorithm out of `na.kalman`, `na.interpolation` and `na.seadec` will yield the best results."

| Method | Speed | Quality | Best For |
|--------|-------|---------|----------|
| MissForest | Slow | Best | Ground truth creation |
| Kalman | Medium | Good | Production pipelines |
| Interpolation | Fast | Moderate | Short gaps |

## Human Verification Process

MissForest output was manually verified using the Shiny app:

1. **Visual inspection**: Check imputed segments for physiological plausibility
2. **Exclude artifacts**: Mark imputation errors for re-processing
3. **Include missed points**: Restore over-aggressive outlier removal

See `../shiny-apps/README.md` for details on the verification workflow.

## Batch Processing

For batch imputation analysis:

```r
source("batch_AnalyzeAndReImpute.R")
# Processes all files in input directory
# Compares multiple imputation methods
```

## Data Flow

```
Raw PLR Signal
    |
    v
Outlier Detection --> outlier_free/
    |
    v
MissForest Imputation --> imputation_raw/
    |
    v
Human Verification (Shiny) --> imputation_final/
    |
    v
Ground Truth for Foundation Model Training
```

## Dependencies

```r
install.packages(c("missForest", "imputeTS", "doParallel"))
```

## References

- Waljee AK, Mukherjee A, Singal AG, et al. (2013). "Comparison of imputation methods for missing laboratory data in medicine." BMJ Open 3:e002847. doi:10.1136/bmjopen-2013-002847

- Stekhoven DJ, Buhlmann P (2012). "MissForest - non-parametric missing value imputation for mixed-type data." Bioinformatics 28(1):112-118.

- Moritz S, Bartz-Beielstein T (2017). "imputeTS: Time Series Missing Value Imputation in R." The R Journal 9(1):207-218.
