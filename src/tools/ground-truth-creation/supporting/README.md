# Supporting R Functions for PLR Analysis

Utility functions for changepoint detection, feature extraction, and data augmentation.

## Files

### changepoint_detection.R

Detects abrupt changes in PLR signals using the PELT algorithm.

**Algorithm**: Pruned Exact Linear Time (PELT) from the `changepoint` package.

**Usage**:
```r
source("changepoint_detection.R")

df_out <- changepoint.detection(
  df = data.frame(time = t, pupil = y),
  method = 'meanvar'  # Options: 'meanvar', 'mean', 'var'
)
```

**Methods**:
| Method | Function | Detects |
|--------|----------|---------|
| `meanvar` | `cpt.meanvar()` | Changes in both mean and variance |
| `mean` | `cpt.mean()` | Changes in mean only |
| `var` | `cpt.var()` | Changes in variance only |

**Behavior**:
- Identifies changepoints as segment boundaries
- Sets values between consecutive changepoints to NA (for imputation)
- Returns dataframe with detected artifacts masked

**Application**: Used in `change.detect.augmentation()` for cleaning signals before decomposition.

**Reference**: Killick R, Eckley IA (2014). "changepoint: An R Package for Changepoint Analysis." Journal of Statistical Software 58(3):1-19.

---

### compute_PLR_features.R

Extracts physiological features from PLR signals for classification.

**Main Function**:
```r
result <- compute.PLR.features(
  data_frame_in,        # PLR data with time, pupil columns
  bins,                 # Feature definition table (from bins.csv)
  color,                # Light color (e.g., "blue", "red")
  normalize_on,         # Normalization flag
  normalize_method,     # Normalization method
  normalize_indiv_colors,
  baseline_period       # Time range for baseline
)

features <- result[[1]]   # List of computed features
data_bins <- result[[2]]  # Data points per bin
```

**Bin-Based Feature Extraction**:

Features are defined in a `bins.csv` configuration file:
- `Name`: Feature identifier (e.g., "PIPR", "MAX_CONSTRICTION")
- `Method`: Computation method (e.g., "mean", "min", "latency")
- `StartString`: Timing reference ("lightOn", "lightOff", "absolute")
- `Start`, `End`: Time boundaries in seconds

**Example Feature Definitions**:
| Name | Method | Start | End | Description |
|------|--------|-------|-----|-------------|
| BASELINE | mean | -2.0 | 0.0 | Pre-stimulus baseline |
| MAX_CONSTRICTION | min | 0.5 | 3.0 | Maximum pupil constriction |
| PIPR | mean | 10.0 | 30.0 | Post-illumination pupil response |

**Workflow**:
1. Define light onset/offset times
2. For each bin definition:
   - Extract data points in time window
   - Apply computation method
   - Store feature value

---

### PLR_augmentation.R

Data augmentation for deep learning using CEEMD components.

**Purpose**: Generate synthetic PLR traces to increase training data size while preserving physiological characteristics.

**Main Function**:
```r
augmented <- augment.traces.for.deep.learning(
  list_in,              # Input component lists
  list_full,            # Full decomposition (hiFreq, loFreq, noise, base)
  t, y, error,          # Time, pupil, error vectors
  classes,              # Class labels
  subject_code_augm,    # Subject identifiers
  method = '1st_try',   # Augmentation method
  increase_by_nr = 10,  # Augmentation factor
  range = c(-0.3, 0.3), # Random weight range
  train_indices,        # Training set indices
  stat_correction_augm = 'normalize_mean_keepMaxConstrTheSame'
)
```

**Augmentation Strategies**:

| Method | Formula | Description |
|--------|---------|-------------|
| loFreq | `y + r1*loFreq` | Add scaled low frequency |
| loFreq_hiFreq | `y + r1*loFreq + r2*hiFreq` | Add both frequency bands |
| loFreq_noise | `y + r1*loFreq + r2*noise` | Add low freq + noise |
| hifreq | `y + r1*hiFreq` | Add high frequency only |
| hifreq_noise | `y + r1*hiFreq + r2*noise` | Add high freq + noise |
| hifreq_lofreq_noise | `y + r1*hiFreq + r2*loFreq + r3*noise` | Three-component |
| hifreq_lofreq_noise_base | `y + r1*hiFreq + r2*loFreq + r3*noise + r4*base` | Full |
| loess_smooth1-3 | LOESS residual subtraction | Smoothing variants |

Where `r1, r2, r3, r4` are random weights sampled from `range`.

**Statistical Corrections**:

| Correction | Description |
|------------|-------------|
| `globalZ` | Z-standardize over all augmented versions |
| `indivZ` | Z-standardize each synthetic subject |
| `meanNorm` | Match mean to original subject |
| `meanNorm_keepMaxConstrTheSame` | Match mean + preserve max constriction |
| `noCorr` | No correction |

**Output**:
```r
# Returns list with:
augmented$y            # Synthetic traces matrix
augmented$labels       # Class labels (repeated)
augmented$subjectCodes # Subject codes (repeated)
augmented$method_strings # Augmentation method per trace
```

**Augmentation Factor**:
- 11x increase with R methods alone
- Additional methods from Matlab preprocessing (optional)

**Example**:
- Input: 241 subjects
- Output: 2651 synthetic subjects (11x)

## Dependencies

```r
install.packages(c(
  "changepoint",  # PELT changepoint detection
  "ggplot2",      # Visualization
  "metafor"       # Meta-analysis (for feature uncertainty)
))
```

## Usage Notes

1. **Feature extraction** requires a `bins.csv` configuration file defining the time windows and computation methods for each feature.

2. **Data augmentation** requires pre-computed CEEMD decomposition with components stored in `list_full`:
   - `hiFreq`: High frequency oscillations
   - `loFreq`: Low frequency oscillations
   - `noiseNorm`: Gaussian noise
   - `noiseNonNorm`: Non-Gaussian artifacts
   - `base_osc`: Baseline oscillations

3. **Changepoint detection** is primarily used as a preprocessing step for other methods, not as a standalone outlier detection approach.
