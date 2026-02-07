# PLR Signal Reconstruction

## Overview

This directory contains utilities for reconstructing denoised PLR signals from CEEMD decomposition results.

## Files

| File | Purpose |
|------|---------|
| `post_process_decomposition_IMFs.R` | Reconstruct signal from selected IMF components |

## CEEMD Reconstruction Process

1. **Input**: IMF matrix from CEEMD decomposition
   - Columns: IMF_1, IMF_2, ..., IMF_n, residue
   - Rows: Timepoints

2. **Human Assignment**: Annotator classifies each IMF:
   - `noise`: High-frequency noise (typically IMF_1, sometimes IMF_2)
   - `hiFreq`: Fast physiological oscillations
   - `loFreq`: Slow physiological trends
   - `base`: Baseline drift / residue

3. **Reconstruction**: Sum of non-noise IMFs
   ```
   pupil_gt = sum(IMFs classified as hiFreq, loFreq, base)
   ```

## Note

These are **placeholder stubs** preserved for documentation. The actual reconstruction logic was integrated into the inspect_EMD Shiny app interface.

## References

- Wu & Huang (2009) "Ensemble empirical mode decomposition"
- Luukko et al. (2016) "Introducing libeemd"
