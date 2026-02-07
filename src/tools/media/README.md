# Media Assets

Demo videos and visual documentation for the ground truth creation tools.

## Files

### inspect-outliers-demo-2018.mp4

**Description**: Screen recording demonstrating the outlier correction workflow using the `inspect_outliers` Shiny app.

**File Details**:
- Size: 10.8 MB
- Format: MP4 video
- Created: 2018

**What It Shows**:

1. **Loading a PLR file**: Opening a subject's PLR recording with automated outlier detection results
2. **ROI selection**: Using the left panel to zoom into regions of interest
3. **Outlier correction**:
   - Excluding false inliers (points incorrectly kept by the algorithm)
   - Including false outliers (points incorrectly marked as artifacts)
4. **Mode switching**: Toggling between Exclude and Include modes
5. **Saving results**: Writing the corrected CSV to disk and loading the next file

**Context**:

This video was created during the initial ground truth annotation campaign in 2018. The workflow shown was used to manually verify and correct automated outlier detection for all 507 subjects in the SERI PLR dataset.

**Viewing**:

Any standard video player (VLC, Windows Media Player, QuickTime) can play this file.

```bash
# Linux
vlc inspect-outliers-demo-2018.mp4

# macOS
open inspect-outliers-demo-2018.mp4

# Windows
start inspect-outliers-demo-2018.mp4
```

## Historical Note

The 2018 ground truth creation process involved:
- ~507 subjects reviewed
- ~2000 timepoints per subject
- Manual correction of blink detection, artifact removal
- Multiple passes for outlier detection, imputation, and denoising

This ground truth dataset forms the basis for evaluating foundation models vs traditional methods in the current research.
