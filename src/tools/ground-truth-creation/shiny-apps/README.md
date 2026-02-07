# Shiny Apps for Ground Truth Creation

Interactive R Shiny applications for manual annotation and quality control of PLR signal preprocessing.

## Overview

These apps were used to create the ground truth labels for:
1. **Outlier detection** - Marking blinks and artifacts in raw PLR signals
2. **Imputation validation** - Verifying reconstructed signal segments
3. **EMD decomposition** - Classifying IMF components into signal categories

## Applications

### 1. inspect_outliers

**Purpose**: Manual correction of automated outlier detection results.

**Running the App**:
```r
# In RStudio, open server.R and click "Run App"
# OR from R console:
library(shiny)
runApp("inspect_outliers")
```

**Interface Layout**:

| Panel | Name | Function |
|-------|------|----------|
| Left | ROI Selection | Brush to select region of interest for zooming |
| Center | Include/Exclude | Double-click with brush to include or exclude points |
| Right | Visualize the Fit | Shows denoised signal overlaid on corrections |

**Controls**:
- **Mode Radio Buttons**: Switch between Exclude (remove false inliers) and Include (restore false outliers)
- **Reset Button**: Clear all selections for current file (TODO: not fully implemented)
- **Visualize Button**: Re-fit spline to current selection (TODO: not fully implemented)
- **Save to Disk**: Write corrected CSV and move to next file

**Mode Switching** (server.R lines 61-62):
```r
# Manually toggle between modes:
# mode = 'outlier'    # For correcting outlier detection
mode = 'imputation'   # For correcting imputation results
```

**Important**: You must manually edit `server.R` to switch modes before running. The app will display a warning confirming the current mode.

**Input/Output Paths**:
- Reads from `config/paths.csv` for platform-specific paths
- Outlier mode: `DATA_OUT/outlier_free/` -> `outlier_free_corrected/`
- Imputation mode: `DATA_OUT/imputation_final/` -> `recon_imputation_correction/`

### 2. inspect_EMD

**Purpose**: Classify CEEMD Intrinsic Mode Functions (IMFs) into signal component categories.

**Running the App**:
```r
library(shiny)
runApp("inspect_EMD")
```

**Interface Layout**:

| Panel | Name | Function |
|-------|------|----------|
| Left | Select Components | Radio buttons for each IMF classification |
| Center | IMFs | All IMFs plotted with baseline offsets |
| Right | Composites | Combined signals per category |

**IMF Classification Categories**:
- `noiseNorm` - Gaussian noise (typically IMF1-2)
- `noiseNonNorm` - Non-Gaussian noise, spikes
- `hiFreq` - High frequency oscillations (hippus, blinks)
- `loFreq` - Low frequency oscillations (fatigue, respiration)
- `base` - Baseline trend (residue + low IMFs)

**Workflow**:
1. App loads CEEMD decomposition CSV from `recon_EMD/` folder
2. Algorithm estimates initial IMF classifications
3. Annotator adjusts classifications via radio buttons
4. Click SAVE to export mapping and signals to `IMF_fusion/` folder

**Output Files**:
- `{filecode}_mapping.csv` - IMF to component classification
- `{filecode}_signals.csv` - Combined signals per category

## Common Troubleshooting

### App Crashes on Startup

**Symptom**: "Error: cannot open connection" or path-related errors

**Solution**:
1. Verify `config/paths.csv` exists and has correct paths for your OS
2. Ensure input directories contain CSV files to process
3. Check that R working directory is set to the shiny-apps folder

### Path Configuration

The apps read paths from `config/paths.csv`:
```csv
data_in,windows,C:\path\to\DATA_IN
data_in,unix,/home/user/DATA_IN
data_out,windows,C:\path\to\DATA_OUT
data_out,unix,/home/user/DATA_OUT
```

### No Files to Review

If you see "YOU DO NOT HAVE ANY FILES TO REVIEW ANYMORE!":
- All files in input directory have been processed
- Check the output directory for completed files

### Cairo Graphics Issues on Linux

Install system dependencies:
```bash
sudo apt-get install libcairo2-dev libgtk2.0-dev xvfb xauth xfonts-base libxt-dev
```

## Docker vs RStudio Usage

### RStudio (Recommended for Annotation)
- Open the project in RStudio
- Navigate to the shiny-apps folder
- Open `server.R` and click "Run App"
- App opens in RStudio's built-in browser

### Docker
Not recommended for interactive annotation due to:
- File I/O complexity with mounted volumes
- Browser interaction through port forwarding
- Difficulty with `rstudioapi::getActiveDocumentContext()`

## Demo Video

See `src/tools/media/inspect-outliers-demo-2018.mp4` for a demonstration of the outlier correction workflow.

## Dependencies

```r
install.packages(c("shiny", "ggplot2", "Cairo", "reshape2", "moments"))
```

## References

- Shiny documentation: https://shiny.rstudio.com/
- Cairo for R graphics: https://www.cairographics.org/
