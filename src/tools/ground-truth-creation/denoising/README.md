# Denoising Methods for PLR Ground Truth

R implementations for decomposing and denoising PLR signals using Empirical Mode Decomposition (EMD) variants.

## Overview

PLR signals contain multiple oscillatory components at different frequencies. EMD-based methods decompose the signal into Intrinsic Mode Functions (IMFs) that can be classified and recombined to extract the underlying pupil trend.

## Primary Method: CEEMD

### Algorithm Description

Complete Ensemble Empirical Mode Decomposition (CEEMD) is a noise-assisted EMD variant:

1. **Ensemble generation**: Add white noise to the signal multiple times
2. **EMD decomposition**: Decompose each noisy version into IMFs
3. **Ensemble averaging**: Average IMFs across all trials to cancel noise
4. **Result**: Stable IMF decomposition resistant to mode mixing

### Ground Truth Parameters

From `lowLevel_decomposition_wrappers.R`:

```r
# Estimate noise amplitude from LOESS residuals
model <- loess(y ~ t, span = 0.1)
noise.amp <- sd(model$residuals)

# CEEMD parameters
trials <- 100  # Number of ensemble members
nimf <- 10     # Maximum number of IMFs

# Run CEEMD
ceemd.result <- CEEMD(y, t, noise.amp, trials, verbose = FALSE)
```

### Parameters Explained

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `trials` | 100 | Balance between stability and computation time |
| `nimf` | 10 | Sufficient for PLR signal bandwidth |
| `noise.amp` | From LOESS | Adaptive to signal characteristics |
| `span` | 0.1 | ~3 second window at 30 fps for local mean |

### Execution Time

CEEMD is computationally expensive:
- ~2.8 minutes per subject
- ~24 hours for full 507-subject dataset

Compare to basic EMD (~1.8 seconds per subject).

## IMF Classification Scheme

After CEEMD decomposition, IMFs are classified into signal components:

### Categories

| Category | Description | Typical IMFs | Physiological Source |
|----------|-------------|--------------|---------------------|
| `noiseNorm` | Gaussian noise | IMF 1-2 | Measurement noise |
| `noiseNonNorm` | Non-Gaussian artifacts | IMF 2-3 | Spikes, blinks |
| `hiFreq` | High frequency oscillations | IMF 3-5 | Hippus, microsaccades |
| `loFreq` | Low frequency oscillations | IMF 5-8 | Respiration, fatigue |
| `base` | Baseline trend | IMF 8-10 + residue | PLR response |

### Signal Reconstruction

```r
# Denoised signal = all components except noise
smooth_indices <- c(FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE)
smooth_signal <- rowSums(df_IMFs[smooth_indices])

# Or combine specific components
hiFreq <- rowSums(df_IMFs[3:5])
loFreq <- rowSums(df_IMFs[5:8])
base <- rowSums(df_IMFs[8:10]) + residue
```

### Manual Classification

IMF classification is performed using the `inspect_EMD` Shiny app:
1. View all IMFs with baseline offsets
2. Select component category for each IMF via radio buttons
3. Verify combined signals in right panel
4. Save mapping to disk

## Alternative Denoising Methods

### LOESS Decomposition

Simple two-stage LOESS filtering for quick preprocessing:

```r
loess.decomposition <- function(t, y, span1 = 0.1, span2 = 0.3) {
  # High frequency extraction (2nd degree LOESS)
  loess_model <- loess(y ~ t, span = span1, degree = 2)
  fit <- loess_model$fitted
  hiFreq <- y - fit

  # Low frequency extraction (1st degree LOWESS)
  loess_model <- loess(fit ~ t, span = span2, degree = 1)
  base <- loess_model$fitted
  loFreq <- y - base

  return(list(base, loFreq, hiFreq))
}
```

### Wavelet Shrinkage (wmtsa)

```r
library(wmtsa)
y_denoised <- wavShrink(
  data_per_subject,
  wavelet = "s8",           # Symlet 8 mother wavelet
  n.level = ilogb(length(data), base = 2),
  shrink.fun = "hard",
  thresh.fun = "universal",
  xform = "modwt",
  reflect = TRUE
)
```

## R Package Comparison

### hht Package (Used for Ground Truth)

- Implements CEEMD, EEMD, EMD
- Includes Hilbert-Huang Transform
- Slower but more features
- Documentation: https://cran.r-project.org/web/packages/hht/hht.pdf

### Rlibeemd Package (Alternative)

- Faster C implementation
- CEEMDAN support (improved CEEMD)
- Recommended for production pipelines
- Reference: Luukko et al. (2017)

### Performance Comparison

| Package | Method | Time per Subject |
|---------|--------|------------------|
| hht | EMD | ~1.8 seconds |
| hht | CEEMD | ~2.8 minutes |
| Rlibeemd | CEEMDAN | ~30 seconds |

## Data Flow

```
Imputed PLR Signal
    |
    v
CEEMD Decomposition --> recon_EMD/
    |
    v
IMF Classification (Shiny) --> IMF_fusion/
    |                           |
    v                           v
{filecode}_mapping.csv      {filecode}_signals.csv
    |
    v
Ground Truth Denoised Signal
```

## Dependencies

```r
# CRAN packages
install.packages(c("hht", "wmtsa", "tvd", "bssm"))

# For faster CEEMD (recommended)
install.packages("Rlibeemd")
```

## References

- Luukko PJ, Helske J, Rasanen E (2016). "Introducing libeemd: A program package for performing the ensemble empirical mode decomposition." Computational Statistics 31(2):545-557.

- Wu Z, Huang NE (2009). "Ensemble Empirical Mode Decomposition: A Noise-Assisted Data Analysis Method." Advances in Adaptive Data Analysis 1(1):1-41.

- Huang NE, et al. (1998). "The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis." Proceedings of the Royal Society A 454:903-995.
