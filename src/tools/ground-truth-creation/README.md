# Ground Truth Creation Pipeline

> **LEGACY CODE**: These R scripts were used for the original ground truth
> annotation workflow. They contain hardcoded absolute paths specific to the
> original development machine. The ground truth data has already been created
> and is stored in `SERI_PLR_GLAUCOMA.db`. These scripts are preserved for
> reference and reproducibility documentation only.

## Overview

This pipeline creates human-verified ground truth annotations for PLR signals through a multi-stage process combining automated algorithms with manual verification.

```
Raw PLR Signal (with blinks, artifacts, noise)
    │
    ├──► [1] Automated Outlier Detection
    │         │  (changepoint, LOF, etc.)
    │         ▼
    ├──► [2] Human Verification (Shiny: inspect_outliers)
    │         │  • Add false negatives
    │         │  • Remove false positives
    │         ▼
    ├──► [3] MissForest Imputation
    │         │  • Random Forest-based
    │         │  • Multivariate across subjects
    │         ▼
    ├──► [4] Human Verification (Shiny: inspect_outliers, imputation mode)
    │         │  • Check for hallucinated values
    │         │  • Re-annotate problematic segments
    │         ▼
    ├──► [5] CEEMD Denoising
    │         │  • Decompose into IMFs
    │         │  • trials=100 ensemble
    │         ▼
    └──► [6] Human IMF Selection (Shiny: inspect_EMD)
              │  • Classify each IMF (signal vs noise)
              │  • Reconstruct denoised signal
              ▼
        Ground Truth Output (pupil_gt)
```

## The 8-Step Workflow

Based on the original R-PLR wiki documentation:

### Step 1: Import the traces
- Load raw PLR recordings from video segmentation output
- Format: CSV with columns `time`, `pupil_raw`

### Step 2: Reduce artifacts (Outlier Detection)
- Apply changepoint detection (PELT algorithm)
- Mark blinks, saccades, tracking failures as NA

### Step 3: Resample to same time vector
- Align all subjects to common time grid
- Handle variable recording lengths

### Step 4: Impute missing values (MissForest)
- Replace NAs with imputed values
- Multivariate approach across subjects
- **Human verification via Shiny app**

### Step 5: Decompose traces with EMD (CEEMD)
- Complete Ensemble EMD decomposition
- 100 ensemble trials for noise averaging
- Output: IMF_1 through IMF_n + residue

### Step 6: Combine different files together
- Merge all variants into single dataset
- Track provenance of each column

### Step 7: Augment data for ML and compute derivatives
- Generate synthetic variations (optional)
- Compute velocity, acceleration

### Step 8: Compute hand-crafted features
- Extract PIPR, MAX_CONSTRICTION, etc.
- Normalize to baseline

## Output Columns

| Column | Description |
|--------|-------------|
| `pupil_raw` | Original unprocessed signal |
| `outlier_mask` | Boolean mask (TRUE = outlier) |
| `pupil_outlier_corrected` | After outlier removal (NA for outliers) |
| `missForest` | After MissForest imputation |
| `CEEMD_IMF_1` ... `CEEMD_IMF_n` | Individual IMFs |
| `CEEMD_residue` | Low-frequency trend |
| `pupil_gt` | Final denoised ground truth |

## Data Available in DuckDB

The processed ground truth data is available in `/home/petteri/Dropbox/github-personal/foundation-PLR/SERI_PLR_GLAUCOMA.db`:

| Column | Description |
|--------|-------------|
| `time` | Time in seconds |
| `pupil_orig` | Original (unnormalized) pupil measurement |
| `pupil_raw` | Normalized raw signal |
| `pupil_gt` | Ground truth (denoised) signal |
| `outlier_mask` | Integer (1 = outlier, 0 = valid) |
| `subject_code` | Subject identifier (PLRxxxx) |
| `class_label` | 'control', 'glaucoma', or None |
| `Red`, `Blue` | Light stimulus channels |

**Note:** If someone wishes to re-inspect the data using the Shiny apps, they would need to:
1. Modify the apps to read from DuckDB (using R's `duckdb` package)
2. Recompute boolean outlier/inlier masks from the stored `outlier_mask` column
3. Adapt the UI to work with the database format instead of file-based input

## Key Parameters Used

### MissForest Imputation

```r
missForest(
  xmis = data_matrix,
  maxiter = 10,        # Max iterations
  ntree = 100,         # Trees per forest
  parallelize = 'variables'
)
# Iterated until visual smoothness
```

### CEEMD Denoising

```r
CEEMD(
  y = imputed_signal,
  t = time_vector,
  noise.amp = sd(loess_residuals),  # span=0.1
  trials = 100                       # Ensemble size
)
# IMF selection: typically keep 2-10 + residue, drop IMF1 (noise)
```

### Changepoint Detection

```r
cpt.meanvar(
  signal,
  method = "PELT",
  penalty = "MBIC"
)
```

## Processing Timeline

- **Annotator**: PT (single annotator)
- **Software**: R-PLR Shiny apps
- **Period**: 2018
- **Subjects processed**: 507 × 2 eyes

## Directory Contents

```
ground-truth-creation/
├── README.md               # This file
├── shiny-apps/
│   ├── inspect_outliers/   # Outlier/imputation verification
│   └── inspect_EMD/        # IMF selection
├── imputation/
│   ├── lowLevel_imputation_wrappers.R
│   └── batch_AnalyzeAndReImpute.R
├── denoising/
│   ├── lowLevel_decomposition_wrappers.R
│   └── lowLevel_denoising_wrappers.R
└── supporting/
    ├── changepoint_detection.R
    ├── compute_PLR_features.R
    └── PLR_augmentation.R
```

## Prerequisites

### R Packages

```r
install.packages(c(
  "shiny", "rstudioapi",
  "missForest", "doParallel",
  "hht",  # or "Rlibeemd" (faster)
  "changepoint",
  "imputeTS",
  "data.table", "ggplot2",
  "Cairo", "moments"
))
```

### System Dependencies (for Cairo)

```bash
sudo apt-get install libcairo2-dev libgtk2.0-dev xvfb xauth xfonts-base libxt-dev
```

## Performance Notes

From the wiki:
- **hht CEEMD**: ~2.84 minutes per file (1981 samples)
- **Rlibeemd (1 thread)**: ~1 second per file
- **Rlibeemd (parallel)**: ~300 milliseconds per file

**Recommendation**: Use `Rlibeemd` package for faster processing.

## References

### MissForest
Waljee AK et al. (2013) "Comparison of imputation methods for missing laboratory data in medicine." BMJ Open. doi:10.1136/bmjopen-2013-002847

### CEEMD/libeemd
Luukko PJJ, Helske J, Räsänen E (2017) "Introducing libeemd: A program package for performing the ensemble empirical mode decomposition." arXiv:1707.00487
