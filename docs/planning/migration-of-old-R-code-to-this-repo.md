# Migration Plan: Legacy R-PLR and deepPLR Tools for Manuscript Transparency

**Created:** 2026-01-31
**Status:** Planning
**Priority:** High (manuscript reproducibility requirement)

## Purpose

**Goal:** Copy relevant ground truth creation tools from legacy repositories into this repo for **transparency and reproducibility**. Readers of the academic paper should have confidence in our preprocessing ground truth methodology.

**This is NOT about:**
- Integrating into Prefect flows
- Running these tools in the main pipeline
- Active development of legacy code

**This IS about:**
- Archiving the tools that created ground truth
- Documenting how they were used
- Providing complete transparency for peer review

---

## Source Repositories

| Repository | URL | Local Path |
|------------|-----|------------|
| **R-PLR** | https://github.com/petteriTeikari/R-PLR | `/home/petteri/Dropbox/github-personal/foundation-PLR/3rdParty_repos/R-PLR-master` |
| **deepPLR** | https://github.com/petteriTeikari/deepPLR | `/home/petteri/Dropbox/github-personal/foundation-PLR/3rdParty_repos/deepPLR-master` |

---

## Existing Documentation Assets

Already downloaded and available:

| Asset | Location | Contents |
|-------|----------|----------|
| **Wiki Tutorial** | `src/tools/docs/Clean the recordings (single file) · petteriTeikari_R-PLR Wiki.html` | Step-by-step Shiny app usage |
| **Screenshots** | `src/tools/docs/.../*.PNG` | outlier_mode, imputation_mode, missForest, CEEMD |
| **Video Demo** | `/home/petteri/.../inspect outliers SERI2018.mp4` (11.3 MB) | Live demo of outlier annotation process |

---

## Proposed Directory Structure

```
src/tools/
├── README.md                           # Overview and usage guide
├── docs/                               # ✅ EXISTING - wiki docs
│   └── [existing wiki HTML + images]
│
├── ground-truth-creation/              # NEW - copied R code
│   ├── README.md                       # How these tools were used
│   │
│   ├── shiny-apps/                     # Interactive proofreading tools
│   │   ├── inspect_outliers/           # Outlier annotation app
│   │   │   ├── ui.R
│   │   │   ├── server.R
│   │   │   └── README.md
│   │   └── inspect_EMD/                # cEEMD IMF selection app
│   │       ├── ui.R
│   │       ├── server.R
│   │       └── README.md
│   │
│   ├── imputation/                     # MissForest implementation
│   │   ├── lowLevel_imputation_wrappers.R
│   │   ├── batch_AnalyzeAndReImpute.R
│   │   └── README.md
│   │
│   ├── denoising/                      # cEEMD denoising
│   │   ├── lowLevel_decomposition_wrappers.R
│   │   ├── data_denoising_wrapper.R
│   │   └── README.md
│   │
│   └── supporting/                     # Other relevant utilities
│       ├── changepoint_detection.R     # PELT-based artifact detection
│       ├── compute_PLR_features.R      # Feature extraction template
│       └── README.md
│
├── media/                              # Demo videos/images
│   ├── inspect-outliers-demo.mp4       # Copy from manuscript folder
│   └── README.md                       # Video descriptions
│
└── parameters-used.md                  # CRITICAL: exact parameters for ground truth
```

---

## Files to Copy

### From R-PLR (`3rdParty_repos/R-PLR-master/`)

| Source Path | Target Path | Purpose |
|-------------|-------------|---------|
| `Apps_Shiny/inspect_outliers/ui.R` | `ground-truth-creation/shiny-apps/inspect_outliers/` | Outlier annotation UI |
| `Apps_Shiny/inspect_outliers/server.R` | `ground-truth-creation/shiny-apps/inspect_outliers/` | Outlier annotation logic |
| `Apps_Shiny/inspect_EMD/ui.R` | `ground-truth-creation/shiny-apps/inspect_EMD/` | cEEMD IMF selection UI |
| `Apps_Shiny/inspect_EMD/server.R` | `ground-truth-creation/shiny-apps/inspect_EMD/` | cEEMD IMF selection logic |
| `PLR_reconstruction/subfunctions/lowLevel_imputation_wrappers.R` | `ground-truth-creation/imputation/` | MissForest wrapper |
| `PLR_reconstruction/batch_AnalyzeAndReImpute.R` | `ground-truth-creation/imputation/` | Imputation orchestrator |
| `PLR_reconstruction/subfunctions/lowLevel_decomposition_wrappers.R` | `ground-truth-creation/denoising/` | cEEMD implementation |
| `PLR_reconstruction/subfunctions/data_denoising_wrapper.R` | `ground-truth-creation/denoising/` | Denoising orchestrator |
| `PLR_artifacts/subfunctions/changepoint_detection.R` | `ground-truth-creation/supporting/` | PELT detection |
| `PLR_analysis/subfunctions/compute_PLR_features.R` | `ground-truth-creation/supporting/` | Feature extraction |

### From deepPLR (`3rdParty_repos/deepPLR-master/`)

| Source Path | Target Path | Purpose |
|-------------|-------------|---------|
| `PLR_dataAugmentation/R_CEEMD_augm/PLR_augmentation.R` | `ground-truth-creation/supporting/` | cEEMD augmentation reference |

### Media

| Source Path | Target Path |
|-------------|-------------|
| `/home/petteri/.../inspect outliers SERI2018.mp4` | `src/tools/media/inspect-outliers-demo.mp4` |

---

## Documentation to Create

### 1. `src/tools/README.md` - Main Overview

```markdown
# Ground Truth Creation Tools

This directory contains the legacy R tools used to create ground truth
preprocessing annotations for the Foundation PLR study.

## Why These Tools Exist

The ground truth (pupil_gt) signals were created through a hybrid
human-algorithmic process:

1. **Changepoint Detection** - Automated blink/artifact detection (PELT)
2. **Manual Annotation** - R Shiny app for human review and correction
3. **MissForest Imputation** - Random forest-based multivariate imputation
4. **cEEMD Denoising** - Ensemble Empirical Mode Decomposition

These tools are archived here for **transparency and reproducibility**.
They are NOT integrated into the main pipeline.

## Video Demo

See `media/inspect-outliers-demo.mp4` for a live demonstration of the
annotation workflow.

## External Documentation

- Original Wiki: https://github.com/petteriTeikari/R-PLR/wiki
- Local copy: `docs/Clean the recordings (single file)...html`
```

### 2. `src/tools/ground-truth-creation/parameters-used.md` - CRITICAL

```markdown
# Ground Truth Creation Parameters

These are the exact parameters used to create the ground truth (pupil_gt)
signals in SERI_PLR_GLAUCOMA.db.

## MissForest Imputation

```r
library(missForest)
result <- missForest(
  xmis = data_matrix,       # Matrix with NA values
  parallelize = 'forests',  # Parallelize across forests
  verbose = TRUE
)
# Iterated until visual smoothness achieved (typically 2-4 iterations)
```

## cEEMD Denoising

```r
library(EMD)

# Noise estimation
model <- loess(signal ~ time, span = 0.1, degree = 2)
noise_amplitude <- sd(model$residuals)

# CEEMD decomposition
ceemd_result <- CEEMD(
  y = imputed_signal,
  t = time_vector,
  noise.amp = noise_amplitude,
  trials = 100              # Ensemble size
)

# IMF selection: keep 2-10 + residue, drop IMF1 (high-freq noise)
imf_selection <- 2:10
imfs_to_keep <- ceemd_result$imf[, imf_selection]
denoised <- rowSums(imfs_to_keep) + ceemd_result$residue
```

## Changepoint Detection

```r
library(changepoint)

# PELT algorithm for mean+variance changepoints
changepoints <- cpt.meanvar(
  signal,
  method = "PELT",
  penalty = "MBIC"
)
```

## Processing Timeline

- **Annotator**: PT (single annotator)
- **Software**: R-PLR (custom R Shiny apps)
- **Period**: 2018 (see video demo timestamp)
- **Files processed**: 507 subjects × 2 eyes
```

### 3. Per-directory READMEs

Each subdirectory should have a brief README explaining:
- What the code does
- How it was used in ground truth creation
- Key functions/entry points
- Dependencies required

---

## Tasks Checklist

### Phase 1: Copy Files

- [ ] Create `src/tools/ground-truth-creation/` directory structure
- [ ] Copy Shiny apps (inspect_outliers, inspect_EMD)
- [ ] Copy imputation code (missForest wrappers)
- [ ] Copy denoising code (cEEMD wrappers)
- [ ] Copy supporting utilities (changepoint, features)
- [ ] Copy video demo to `src/tools/media/`

### Phase 2: Documentation

- [ ] Create `src/tools/README.md` (main overview)
- [ ] Create `src/tools/ground-truth-creation/parameters-used.md`
- [ ] Create README.md for each subdirectory
- [ ] Add header comments to copied R files explaining their role

### Phase 3: Manuscript Integration

- [ ] Update methods.tex supplementary materials with:
  - Link to `src/tools/` in GitHub repo
  - Reference to wiki documentation
  - Note about video demo
- [ ] Verify all external links (GitHub repos) are accessible

### Phase 4: Validation

- [ ] Ensure copied code has no hardcoded local paths
- [ ] Add `.gitignore` entries if needed (large files, data)
- [ ] Verify wiki HTML renders correctly
- [ ] Test that video plays

---

## Manuscript Text Suggestion

For methods.tex supplementary materials:

```latex
\section{Ground Truth Creation Tools}

The ground truth preprocessing pipeline was implemented using custom R tools
available at \url{https://github.com/petteriTeikari/foundation_PLR/tree/main/src/tools}.

The annotation workflow used R Shiny applications for interactive visual
inspection of outlier detection and cEEMD denoising results. A video
demonstration of this process is included in the repository
(\texttt{src/tools/media/inspect-outliers-demo.mp4}).

Original tool repositories:
\begin{itemize}
  \item R-PLR: \url{https://github.com/petteriTeikari/R-PLR}
  \item deepPLR: \url{https://github.com/petteriTeikari/deepPLR}
\end{itemize}

Key parameters used for ground truth creation are documented in
\texttt{src/tools/ground-truth-creation/parameters-used.md}.
```

---

## R Dependencies for Legacy Tools

If anyone wants to actually run these tools:

```r
# Core dependencies
install.packages(c(
  "shiny",
  "missForest",
  "EMD",           # or "hht" for CEEMD
  "changepoint",
  "imputeTS",
  "data.table",
  "ggplot2"
))
```

**Note:** These tools are archived for reference, not active use. Running them would require the original data files and directory structure.

---

## Timeline

| Task | Effort | Priority |
|------|--------|----------|
| Copy files | 30 min | High |
| Create READMEs | 1-2 hours | High |
| Update methods.tex | 30 min | Medium |
| Validation | 30 min | Medium |

**Total estimated effort: 3-4 hours**

---

## References

1. **MissForest:** Stekhoven DJ, Bühlmann P. MissForest—non-parametric missing value imputation for mixed-type data. Bioinformatics. 2012;28(1):112-118.

2. **cEEMD:** Torres ME, et al. A complete ensemble empirical mode decomposition with adaptive noise. ICASSP 2011.

3. **Changepoint:** Killick R, Eckley IA. changepoint: An R Package for Changepoint Analysis. J Stat Softw. 2014;58(3):1-19.
