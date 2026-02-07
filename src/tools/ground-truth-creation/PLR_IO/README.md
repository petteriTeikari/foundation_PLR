# PLR I/O Utilities

## Overview

This directory contains utility functions for reading and writing PLR data files during the ground truth annotation process.

## Files

| File | Purpose |
|------|---------|
| `check_for_done_filecodes.R` | Track which recordings have been annotated |
| `export_pupil_dataframe_toDisk.R` | Save annotated data to disk |

## Note

These are **placeholder stubs** preserved for documentation purposes. The original implementations read/wrote data from the Singapore Eye Research Institute (SERI) dataset format.

The actual I/O logic was specific to the file naming conventions and directory structure used in the 2018 annotation project.

## Original Directory Structure

```
data_in/
├── PLR0001/
│   ├── left_eye.csv
│   └── right_eye.csv
├── PLR0002/
...

data_out/
├── outlier_free/
│   ├── PLR0001_left.csv
│   └── PLR0001_left_corrected/
├── imputation_final/
└── decomposed/
```
