# Open-Source Pupillometry Software Ecosystem: Mini Literature Review

> **Purpose**: Comprehensive research for fig-repo-99 (Pupillometry Software Ecosystem figure).
> **Date**: 2026-02-08
> **Method**: GitHub repository analysis of all identifiable open-source pupillometry tools.

## Executive Summary

The open-source pupillometry ecosystem comprises **9 actively accessible tools** spanning 4 programming languages (Python, R, MATLAB, C++). The ecosystem is fragmented along three axes:

1. **Hardware coupling**: From tightly coupled (PupilEXT+Basler, PyPlr+Pupil Core) to fully device-agnostic
2. **Functional scope**: From acquisition-only to full preprocessing pipelines
3. **License restrictions**: From fully permissive (MIT) to non-commercial only (CC-BY-NC)

**Critical gap identified**: No existing tool provides TSFM-based (foundation model) preprocessing for pupillometry data. All tools use traditional signal processing (filtering, interpolation, threshold-based artifact detection).

---

## Tool-by-Tool Analysis

### 1. PupilEXT

| Attribute | Value |
|-----------|-------|
| **Repository** | [openPupil/Open-PupilEXT](https://github.com/openPupil/Open-PupilEXT) |
| **Language** | C++ / Qt 5.15 |
| **License** | GPLv3 + **non-commercial restriction** on pupil detection |
| **Stars** | 135 |
| **Last commit** | September 2024 |
| **Status** | Active (beta v0.1.1, community branch v0.1.2) |
| **Publication** | Frontiers in Neuroscience |
| **Commercial use** | **NO** (pupil detection algorithms restricted to academic use) |

**Capabilities**: Real-time pupil diameter measurement with 6 detection algorithms (Starburst, Swirski2D, ExCuSe, ElSe, PuRe, PuReST). Supports stereo camera setups for mm-scale measurements. Offline batch processing of recorded images.

**Hardware coupling**: **Tight** — requires Basler industrial cameras (acA2040-120um, acA1300-200um). Needs Pylon Camera Software Suite.

**Preprocessing scope**: Pupil detection only (no blink removal, no interpolation, no baseline correction). Output is raw pupil diameter time series.

**Dependencies**: Boost 1.75, Ceres 2.0, Eigen 3.3.9, OpenCV 4.5.1, Qt 5.15, TBB. ~6 GB disk for build dependencies.

---

### 2. PyPlr

| Attribute | Value |
|-----------|-------|
| **Repository** | [PyPlr/cvd_pupillometry](https://github.com/PyPlr/cvd_pupillometry) |
| **Language** | Python (+ Jupyter notebooks 96.6%) |
| **License** | MIT |
| **Stars** | 14 |
| **Last commit** | November 2022 |
| **Status** | Inactive (no activity in ~3 years) |
| **Publication** | Zenodo DOI |
| **Commercial use** | **YES** (MIT) |

**Capabilities**: PLR research toolkit with stimulus design, optimization, and delivery scripting. Interfaces to Spectra Tune Lab (STLAB) light engines and Ocean Optics spectrometers. Basic pupil data extraction and cleaning.

**Hardware coupling**: **Tight** — requires Pupil Core eye tracking platform (Pupil Labs GmbH) + STLAB light engine (LEDMOTIVE).

**Preprocessing scope**: Data extraction from Pupil Core exports, basic cleaning. Primarily an acquisition/stimulus control toolkit.

**Dependencies**: matplotlib, numpy, pandas, scipy, seaborn, pyzmq, seabreeze. Install via `pip install pyplr`.

---

### 3. PupilMetrics

| Attribute | Value |
|-----------|-------|
| **Repository** | [JulesGoninRIO/PupilMetrics](https://github.com/JulesGoninRIO/PupilMetrics) |
| **Language** | Python |
| **License** | GPLv3 |
| **Stars** | 1 |
| **Last commit** | April 2023 |
| **Status** | Inactive (14 commits total) |
| **Publication** | Nature Scientific Reports (2024) |
| **Commercial use** | **Yes with copyleft** (GPLv3 — must open-source derivatives) |

**Capabilities**: Interactive GUI for artifact validation/correction in pupillometric recordings. Features: flash-level grouping, software-generated fits visualization, undo/redo, Excel export. Configurable parameters for delete threshold, drop detection sensitivity, fit precision.

**Hardware coupling**: **Tight** — supports NeuroLight and Diagnosys clinical pupillometers only.

**Preprocessing scope**: Artifact detection + interactive correction + outcome measure extraction. Full clinical preprocessing pipeline but device-specific.

**Dependencies**: Not formally documented (likely matplotlib, scipy, tkinter). Run via `python app.py`.

---

### 4. GazeR

| Attribute | Value |
|-----------|-------|
| **Repository** | [dmirman/gazer](https://github.com/dmirman/gazer) |
| **Language** | R |
| **License** | GPL-3 |
| **Stars** | 52 |
| **Last commit** | October 2025 |
| **Status** | **Actively maintained** (508+ commits) |
| **Publication** | Geller et al. (2020), Behavior Research Methods |
| **Commercial use** | **Yes with copyleft** (GPL-3) |

**Capabilities**: Comprehensive eye-tracking data processing package. Blink detection (Hershman algorithm based on pupillary noise), saccade/fixation identification, interpolation, baseline correction, pupil scaling, data binification, AOI conversion, upsampling (added 2025), growth curve formatting.

**Hardware coupling**: **Multi-device** — EyeLink (EDF/ASC), Tobii X2-30, Pupil Labs Neon (added Oct 2025), generic data.frame input.

**Preprocessing scope**: Full pipeline (blink detection → interpolation → baseline correction → visualization). No TSFM/ML-based methods.

**Dependencies**: ggplot2, tidyverse, lme4, data.table, saccades (GitHub). Install via `remotes::install_github("dmirman/gazer")`.

---

### 5. eyeris

| Attribute | Value |
|-----------|-------|
| **Repository** | [shawntz/eyeris](https://github.com/shawntz/eyeris) |
| **Language** | R |
| **License** | MIT |
| **Stars** | 5 |
| **Last commit** | February 2026 |
| **Status** | **Actively maintained** (CRAN v3.0.1) |
| **Publication** | CRAN package |
| **Commercial use** | **YES** (MIT) |

**Capabilities**: Most comprehensive preprocessing pipeline among all tools. The `glassbox()` function implements 10 modular stages: (1) load ASC, (2) deblink, (3) detransient (MAD-based spike removal), (4) interpolate, (5) lowpass Butterworth filter, (6) downsample, (7) bin, (8) detrend, (9) z-score, (10) confound visualization. Also: interactive HTML QC reports, gaze heatmaps, binocular correlation, BIDS-like file organization.

**Hardware coupling**: **Partially device-agnostic** — primary input is EyeLink .asc files, but pipeline logic is generic. DuckDB integration for scalable storage.

**Preprocessing scope**: Full pipeline (10 stages). FAIR/BIDS conventions. Traditional signal processing only (no ML/TSFM).

**Dependencies**: R base, optional DuckDB and Apache Arrow. Install via `install.packages("eyeris")`.

---

### 6. PuPl

| Attribute | Value |
|-----------|-------|
| **Repository** | [kinleyid/PuPl](https://github.com/kinleyid/PuPl) |
| **Language** | MATLAB (97.1%) / GNU Octave compatible |
| **License** | **CC-BY-NC-4.0** (Non-commercial only) |
| **Stars** | 11 |
| **Last commit** | January 2025 |
| **Status** | Active (v2.1.10, 254 commits) |
| **Publication** | — |
| **Commercial use** | **NO** (CC-BY-NC explicitly prohibits commercial use) |

**Capabilities**: GUI-based pipeliner for pupillometry data. Interactive processing + segmentation with reproducible pipeline export via `pupl history`. Supports SMI .txt, BIDS-compliant TSV, and custom text formats. Batch processing via command-line scripting.

**Hardware coupling**: **Partially device-agnostic** — native SMI support, BIDS format support, extensible via add-ons for other formats.

**Preprocessing scope**: Full pipeline (file I/O → preprocessing → processing → plotting). Module structure: prep/, process/, edit/, plot/, file/, tools/, add-ons/.

**Dependencies**: MATLAB or GNU Octave (no external toolboxes required).

---

### 7. PupEyes

| Attribute | Value |
|-----------|-------|
| **Repository** | [HanZhang-psych/pupeyes](https://github.com/HanZhang-psych/pupeyes) |
| **Language** | Python |
| **License** | GPL-3.0 |
| **Stars** | 8 |
| **Last commit** | January 2026 |
| **Status** | Active (v0.3.7) |
| **Publication** | Zhang & Jonides (2025), OSF Preprint |
| **Commercial use** | **Yes with copyleft** (GPL-3) |

**Capabilities**: Pupil preprocessing and interactive visualization. Blink detection & removal, artifact rejection, smoothing, baseline correction. Interactive Plotly Dash viewers: Pupil Viewer, Fixation Viewer, AOI Drawing Tool. Blink detection method comparison tools.

**Hardware coupling**: **Multi-device** — EyeLink (native .asc), Tobii (via Titta/Pro Lab), generic data.frame input.

**Preprocessing scope**: Full pipeline (blink detection → artifact rejection → smoothing → baseline correction → visualization). Traditional signal processing only.

**Dependencies**: pandas, scipy, dash, opencv-python-headless, h5py. Install via `pip install pupeyes`. Built with Poetry.

---

### 8. PupilSense

| Attribute | Value |
|-----------|-------|
| **Repository** | [stevenshci/PupilSense](https://github.com/stevenshci/PupilSense) |
| **Language** | Python (93.6%) |
| **License** | MIT |
| **Stars** | 66 |
| **Last commit** | April 2024 |
| **Status** | Active (companion projects at ACM MobileHCI 2024, IEEE BSN 2024) |
| **Publication** | Multiple conference papers |
| **Commercial use** | **YES** (MIT) |

**Capabilities**: Deep learning-based pupillometry from smartphone eye images. Pupil and iris segmentation using Detectron2. Pupil-to-iris ratio (PIR) estimation. Designed for naturalistic (real-world) settings with varying lighting and head positions. Fine-tunable models for custom datasets. Application: depressive episode detection via pupillary response.

**Hardware coupling**: **Smartphone-only** — uses smartphone front-facing camera for acquisition. No traditional eye tracker required.

**Preprocessing scope**: Acquisition + segmentation + ratio estimation. Not a preprocessing pipeline for time-series data — operates on individual images.

**Dependencies**: Detectron2, PyTorch, standard scientific Python. Deep learning focused.

---

### 9. PySilSub

| Attribute | Value |
|-----------|-------|
| **Repository** | [PySilentSubstitution/pysilsub](https://github.com/PySilentSubstitution/pysilsub) |
| **Language** | Python / Jupyter |
| **License** | MIT |
| **Stars** | 13 |
| **Last commit** | July 2023 |
| **Status** | Stable (v0.1.1, low-frequency updates) |
| **Publication** | Journal of Vision (2023) |
| **Commercial use** | **YES** (MIT) |

**Capabilities**: Silent substitution stimulus computation. Not a preprocessing tool — computes observer- and device-specific solutions for photoreceptor-targeted stimulation. Supports melanopsin, S-cones, L/M cones, rods targeting. CIEPO06/CIES026-compliant action spectra.

**Hardware coupling**: **Device-agnostic** for computation, but designed to work with multiprimary stimulation devices.

**Preprocessing scope**: **None** — this is a stimulus design tool, not a signal preprocessing tool. Included in ecosystem because it is used alongside PyPlr for PLR research.

**Dependencies**: Standard scientific Python. Install via `pip install pysilsub`.

---

### 10. MTSC

| Attribute | Value |
|-----------|-------|
| **Repository** | `MTSC-PLR/MTSC` (URL returns 404) |
| **Status** | **DOES NOT EXIST** — repository deleted, made private, or never created |

**Excluded from analysis.**

---

## Comparative Analysis

### Capability Matrix

| Tool | Blink Detection | Interpolation | Baseline Corr. | Filtering | Artifact QC | ML/TSFM | Visualization |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **PupilEXT** | - | - | - | - | - | - | Real-time |
| **PyPlr** | - | - | - | - | - | - | Basic |
| **PupilMetrics** | Interactive | - | - | - | Interactive | - | GUI |
| **GazeR** | Hershman | Linear | Subtractive | - | - | - | ggplot2 |
| **eyeris** | Extend+mask | Linear | Detrend | Butterworth | HTML reports | - | Interactive |
| **PuPl** | Via pipeline | Via pipeline | Via pipeline | Via pipeline | - | - | GUI |
| **PupEyes** | Automated | Smoothing | Subtractive | Smoothing | - | - | Plotly Dash |
| **PupilSense** | - | - | - | - | - | **DL** (Detectron2) | - |
| **PySilSub** | - | - | - | - | - | - | Plots |
| **foundation_PLR** | **MOMENT/LOF/Ensemble** | **SAITS/CSDI/MOMENT** | cEEMD | - | Automated QA | **TSFM** | Matplotlib+R |

### Hardware Support Matrix

| Tool | EyeLink | Tobii | Pupil Labs | Basler | NeuroLight | Diagnosys | Smartphone | SMI | Generic CSV |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **PupilEXT** | - | - | - | **Primary** | - | - | - | - | - |
| **PyPlr** | - | - | **Primary** | - | - | - | - | - | - |
| **PupilMetrics** | - | - | - | - | **Primary** | **Primary** | - | - | - |
| **GazeR** | EDF/ASC | X2-30 | Neon | - | - | - | - | - | data.frame |
| **eyeris** | **ASC** | - | - | - | - | - | - | - | - |
| **PuPl** | - | - | - | - | - | - | - | **Primary** | BIDS/txt |
| **PupEyes** | **ASC** | Pro Lab | - | - | - | - | - | - | data.frame |
| **PupilSense** | - | - | - | - | - | - | **Primary** | - | - |
| **foundation_PLR** | - | - | - | - | - | - | - | - | **Any 30Hz** |

### License Summary

| License | Tools | Commercial OK? |
|---------|-------|:-:|
| **MIT** (most permissive) | PyPlr, eyeris, PupilSense, PySilSub | **YES** |
| **GPL-3** (copyleft) | GazeR, PupilMetrics, PupEyes | Conditional (must open-source derivatives) |
| **GPLv3 + non-commercial** | PupilEXT | **NO** (detection algorithms restricted) |
| **CC-BY-NC-4.0** | PuPl | **NO** (explicitly prohibited) |

### Programming Language Distribution

| Language | Tools | Count |
|----------|-------|:-----:|
| **Python** | PyPlr, PupilMetrics, PupEyes, PupilSense, PySilSub | 5 |
| **R** | GazeR, eyeris | 2 |
| **MATLAB** | PuPl | 1 |
| **C++** | PupilEXT | 1 |

### Activity & Community

| Tool | Stars | Commits | Last Active | Peer-Reviewed |
|------|------:|--------:|-------------|:--:|
| **PupilEXT** | 135 | 100+ | Sep 2024 | Frontiers |
| **PupilSense** | 66 | 100+ | Apr 2024 | MobileHCI/BSN |
| **GazeR** | 52 | 508+ | Oct 2025 | BRM 2020 |
| **PyPlr** | 14 | 198 | Nov 2022 | Zenodo |
| **PySilSub** | 13 | 82 | Jul 2023 | J. Vision |
| **PuPl** | 11 | 254 | Jan 2025 | — |
| **PupEyes** | 8 | 50+ | Jan 2026 | OSF Preprint |
| **eyeris** | 5 | 100+ | Feb 2026 | CRAN |
| **PupilMetrics** | 1 | 14 | Apr 2023 | Nat. Sci. Rep. |

---

## Key Insights for fig-repo-99

### 1. The Preprocessing Gap

Every existing tool uses **traditional signal processing** (threshold-based blink detection, linear/cubic interpolation, Butterworth filtering, subtractive baseline correction). **No tool applies foundation models or deep learning to preprocessing** (PupilSense uses DL for acquisition/segmentation, not preprocessing).

### 2. Hardware Fragmentation

Tools cluster into three tiers:
- **Tightly coupled**: PupilEXT (Basler), PyPlr (Pupil Core), PupilMetrics (NeuroLight/Diagnosys)
- **Multi-device**: GazeR (EyeLink+Tobii+Neon), PupEyes (EyeLink+Tobii)
- **Device-agnostic**: eyeris (EyeLink primary but generic pipeline), PuPl (SMI+BIDS+custom)

### 3. Functional Scope Spectrum

```
Acquisition        Preprocessing       Full Pipeline        Clinical
    Only               Only            (Preproc+Analysis)   Application
    |                   |                    |                    |
PupilEXT          foundation_PLR        GazeR                PupilMetrics
PyPlr                                   eyeris               PupilSense
PySilSub                                PuPl
                                        PupEyes
```

### 4. License Landscape

- 4/9 tools are MIT (fully permissive for commercial use)
- 3/9 tools are GPL-3 (copyleft — commercial with conditions)
- 2/9 tools restrict commercial use (PupilEXT detection + PuPl entirely)

### 5. The R vs Python Divide

R tools (GazeR, eyeris) focus on **statistical analysis integration** (growth curve modeling, ggplot2 visualization, BIDS compliance). Python tools span wider (acquisition, DL, GUI, clinical). MATLAB (PuPl) is a declining but still-used niche.

### 6. Our Unique Position (foundation_PLR)

| Dimension | foundation_PLR | Nearest Alternative |
|-----------|---------------|-------------------|
| **Preprocessing method** | TSFM (MOMENT, UniTS) + traditional (LOF) | None (all traditional only) |
| **Hardware coupling** | Fully agnostic (any 30 Hz signal) | eyeris (generic pipeline, EyeLink input) |
| **Imputation** | SAITS, CSDI, MOMENT zero-shot | Linear interpolation (GazeR, eyeris) |
| **Outlier detection** | 11 methods including ensembles | Threshold-based (all tools) |
| **Evaluation** | STRATOS-compliant (5 metric domains) | None (no standardized eval framework) |
| **Reproducibility** | MLflow + DuckDB + Prefect | None at this level |

---

## Visual Design Recommendations for fig-repo-99

### Axes
- **X-axis**: Functional scope (Acquisition → Preprocessing → Full Pipeline → Clinical)
- **Y-axis**: Hardware coupling (Tightly Coupled → Multi-Device → Device-Agnostic)

### Visual Elements
- **Card per tool**: Name, language icon (Python snake, R logo, MATLAB diamond, C++ brackets), license badge
- **License badges**: Green shield = MIT, Yellow shield = GPL-3, Red shield = Non-commercial
- **Star indicator**: Small GitHub star icon with count
- **Connection lines**: Tools from same ecosystem (PyPlr ↔ PySilSub)
- **foundation_PLR**: Gold-highlighted star in the "Device-Agnostic + Preprocessing" quadrant with "TSFM" badge

### Key Callouts
1. "THE GAP" — No TSFM-based preprocessing tool exists
2. "ALL TRADITIONAL" — Every existing tool uses threshold/filter methods
3. "OUR CONTRIBUTION" — First TSFM-based, device-agnostic preprocessing evaluation

---

## References

1. PupilEXT: Santini et al. (2021), Frontiers in Neuroscience
2. PyPlr: Martin & Spitschan, Zenodo DOI
3. PupilMetrics: Nature Scientific Reports (2024)
4. GazeR: Geller et al. (2020), Behavior Research Methods
5. eyeris: Schwartz, CRAN package (2025)
6. PupEyes: Zhang & Jonides (2025), OSF Preprint
7. PySilSub: Martin et al. (2023), Journal of Vision
8. PupilSense: Stevens et al., ACM MobileHCI (2024)
