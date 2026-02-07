# Legacy R Tools for Ground Truth Creation

## Purpose

These tools were used to create the **ground truth preprocessing annotations** for the 507 PLR (Pupillary Light Reflex) recordings evaluated in this study. Human annotators used interactive R Shiny applications to verify and correct automated preprocessing.

> **Note:** These tools are archived for **transparency and reproducibility**. They are NOT integrated into the main analysis pipeline.

## Relationship to the Paper

The foundation model evaluation benchmarks automated methods against ground truth created using:

1. **Outlier Detection**: Automated algorithms + human verification via Shiny app
2. **Imputation**: MissForest random forest imputation + human correction
3. **Denoising**: CEEMD (Complementary Ensemble EMD) + human IMF selection

### Manuscript Reference (methods.tex)

> "Ground truth artifact annotations were created through a hybrid human-algorithmic process by a single annotator (PT) using custom R Shiny applications... Marked outlier segments were then removed and imputed using the MissForest algorithm... Complete ensemble empirical mode decomposition (cEEMD) was subsequently applied to remove high-frequency instrumentation noise while preserving physiologically relevant temporal dynamics."

## Tool Categories

| Category | Directory | Purpose |
|----------|-----------|---------|
| **Shiny Apps** | `ground-truth-creation/shiny-apps/` | Interactive proofreading |
| **Imputation** | `ground-truth-creation/imputation/` | MissForest RF imputation |
| **Denoising** | `ground-truth-creation/denoising/` | cEEMD decomposition |
| **Supporting** | `ground-truth-creation/supporting/` | Feature extraction, changepoint |
| **Documentation** | `docs/` | Original wiki tutorials |
| **Media** | `media/` | Demo video |

## Demo Video

A video demonstration of the outlier correction workflow is available:

📹 **`media/inspect-outliers-demo-2018.mp4`** (11.3 MB)

This shows the interactive process of reviewing automated outlier detection and correcting false positives/negatives.

## Subject Counts

| Stage | N Subjects | Notes |
|-------|------------|-------|
| **Outlier Detection** | 507 | All subjects have ground truth masks |
| **Imputation** | 507 | All subjects have denoised signals |
| **Classification** | 208 | Only labeled subjects (152 control + 56 glaucoma) |

## Quick Start

### Prerequisites

- R >= 4.4
- RStudio (for Shiny apps) OR Docker

### Required R Packages

```r
install.packages(c(
  "shiny",
  "missForest",
  "hht",          # or "Rlibeemd" for faster CEEMD
  "changepoint",
  "imputeTS",
  "data.table",
  "ggplot2",
  "Cairo",
  "doParallel"
))
```

### Running Shiny Apps (RStudio)

1. Open `ground-truth-creation/shiny-apps/inspect_outliers/server.R` in RStudio
2. Click "Run App"
3. Configure mode (outlier vs imputation) on line 61-62

### Running with Docker

```bash
# Build the Shiny image
docker build -t foundation-plr-shiny -f Dockerfile.shiny .

# Run Shiny Server
docker-compose --profile shiny up -d

# Access at: http://localhost:3838/inspect_outliers/
```

## Original Repositories

These tools were originally developed in:

- **R-PLR**: https://github.com/petteriTeikari/R-PLR
- **deepPLR**: https://github.com/petteriTeikari/deepPLR

## See Also

- [Ground Truth Creation Pipeline](ground-truth-creation/README.md)
- [Original Wiki Tutorial](docs/) (HTML export)
- [Main Project ARCHITECTURE.md](../../ARCHITECTURE.md)

## References

### MissForest
Waljee AK et al. (2013) "Comparison of imputation methods for missing laboratory data in medicine." BMJ Open. doi:10.1136/bmjopen-2013-002847

### CEEMD/libeemd
Luukko PJJ, Helske J, Räsänen E (2017) "Introducing libeemd: A program package for performing the ensemble empirical mode decomposition." arXiv:1707.00487

### Original PLR Data
Najjar RP et al. (2023) "Handheld chromatic pupillometry can accurately and rapidly reveal functional loss in glaucoma." Br J Ophthalmol.
