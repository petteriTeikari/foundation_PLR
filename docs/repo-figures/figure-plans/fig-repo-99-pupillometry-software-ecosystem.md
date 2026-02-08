# fig-repo-99: The Pupillometry Software Ecosystem — A Library Hierarchy

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-99 |
| **Title** | The Pupillometry Software Ecosystem: From Experiment Design to TSFM Preprocessing |
| **Complexity Level** | L2 (Landscape overview) |
| **Target Persona** | Research Scientist, PI |
| **Location** | Root README.md (Open-Source Pupillometry Libraries section) |
| **Priority** | P2 (High) |
| **Aspect Ratio** | 16:9 |

## Purpose

Map the open-source pupillometry ecosystem as a three-layer library hierarchy: experiment design (top), signal analysis (middle), and preprocessing (bottom). Reveals that (1) all existing analysis tools use traditional signal processing, (2) no TSFM-based preprocessing exists, and (3) our repository fills the preprocessing layer gap so that cognitive neuroscientists, chronobiologists, and psychologists can design experiments without worrying about TSFM intricacies.

## Key Message

"The pupillometry ecosystem has mature experiment design tools (PsychoPy, PySilSub) and multiple signal analysis libraries — but ALL use traditional preprocessing (threshold + interpolation). foundation_PLR introduces TSFM-based preprocessing as a drop-in upgrade."

## Visual Concept

Three-tier horizontal hierarchy with tool cards positioned by layer and hardware coupling:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  THE PUPILLOMETRY SOFTWARE ECOSYSTEM                                              │
│                                                                                   │
│  ╔══════════════════════════════════════════════════════════════════════════════╗  │
│  ║  LAYER 3: EXPERIMENT DESIGN — "What stimulus? What protocol?"              ║  │
│  ║  Users: Cognitive neuroscientists, chronobiologists, psychologists          ║  │
│  ║                                                                             ║  │
│  ║  ┌─────────────────┐          ┌──────────────────────┐                     ║  │
│  ║  │ 🐍 PsychoPy     │          │ 🐍 PySilSub          │                     ║  │
│  ║  │ MIT  ★1900      │          │ MIT  ★13             │                     ║  │
│  ║  │ Stimulus timing  │  ←───→  │ Silent substitution  │                     ║  │
│  ║  │ LMS color spaces │          │ Cone-isolating       │                     ║  │
│  ║  │ Hardware sync    │          │ stimulus computation │                     ║  │
│  ║  └─────────────────┘          └──────────────────────┘                     ║  │
│  ╚══════════════════════════════════════════════════════════════════════════════╝  │
│       │                                                                           │
│       │ Delivers stimulus → Records pupil response                                │
│       ▼                                                                           │
│  ╔══════════════════════════════════════════════════════════════════════════════╗  │
│  ║  LAYER 2: SIGNAL ANALYSIS — "Process the pupil time series"                ║  │
│  ║  Users: Signal processing experts, vision researchers                       ║  │
│  ║                                                                             ║  │
│  ║  HARDWARE-COUPLED                 MULTI-DEVICE / DEVICE-AGNOSTIC           ║  │
│  ║  ┌──────────────┐ ┌───────────┐  ┌──────────┐ ┌────────┐ ┌───────────┐   ║  │
│  ║  │ C++ PupilEXT │ │🐍 PyPlr   │  │ R GazeR  │ │R eyeris│ │🐍 PupEyes│   ║  │
│  ║  │ GPLv3+NC ★135│ │ MIT  ★14  │  │ GPL3 ★52 │ │MIT  ★5 │ │ GPL3  ★8 │   ║  │
│  ║  │ Basler cams  │ │ Pupil Core│  │ EyeLink  │ │EyeLink │ │ EyeLink  │   ║  │
│  ║  │ 6 detection  │ │ STLAB     │  │ Tobii    │ │BIDS/   │ │ Tobii    │   ║  │
│  ║  │ algorithms   │ │ acq+stim  │  │ Neon     │ │DuckDB  │ │ Dash viz │   ║  │
│  ║  └──────────────┘ └───────────┘  │ data.frame│ │10-stage│ │ baseline │   ║  │
│  ║                                   └──────────┘ │pipeline│ │ correct. │   ║  │
│  ║  ┌──────────────────┐  ┌─────────────────┐    └────────┘ └───────────┘   ║  │
│  ║  │🐍 PupilMetrics   │  │ MATLAB PuPl     │                               ║  │
│  ║  │ GPLv3  ★1        │  │ CC-NC   ★11     │    ┌──────────────┐           ║  │
│  ║  │ NeuroLight /     │  │ GUI pipeliner   │    │🐍 PupilSense │           ║  │
│  ║  │ Diagnosys        │  │ SMI / BIDS      │    │ MIT    ★66   │           ║  │
│  ║  │ Nature pub.      │  │ Octave compat.  │    │ Smartphone   │           ║  │
│  ║  └──────────────────┘  └─────────────────┘    │ DL segm.     │           ║  │
│  ║                                                └──────────────┘           ║  │
│  ║  ⚠ ALL tools use TRADITIONAL preprocessing:                              ║  │
│  ║    threshold blink detection, linear interpolation, Butterworth filter    ║  │
│  ╚══════════════════════════════════════════════════════════════════════════════╝  │
│       │                                                                           │
│       │ Raw pupil signal → Needs preprocessing                                    │
│       ▼                                                                           │
│  ╔══════════════════════════════════════════════════════════════════════════════╗  │
│  ║  LAYER 1: PREPROCESSING — "Clean the signal robustly"           ★ THE GAP ║  │
│  ║                                                                             ║  │
│  ║  ┌────────────────────────────────────────────────────────────────────┐    ║  │
│  ║  │  ★ foundation_PLR (THIS REPO)                       MIT (planned) │    ║  │
│  ║  │                                                                    │    ║  │
│  ║  │  TSFM-based preprocessing for ANY pupillometer:                    │    ║  │
│  ║  │  • 11 outlier detection methods (MOMENT, UniTS, LOF, Ensembles)   │    ║  │
│  ║  │  • 8 imputation methods (SAITS, CSDI, MOMENT zero-shot)          │    ║  │
│  ║  │  • STRATOS-compliant evaluation (5 metric domains)               │    ║  │
│  ║  │  • Device-agnostic: any 30 Hz pupil signal                       │    ║  │
│  ║  │                                                                    │    ║  │
│  ║  │  Researchers design experiments above ↑ without worrying about     │    ║  │
│  ║  │  TSFM intricacies — this layer handles robust preprocessing.      │    ║  │
│  ║  └────────────────────────────────────────────────────────────────────┘    ║  │
│  ╚══════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                   │
│  LICENSE LEGEND:  🟢 MIT (commercial OK)  🟡 GPL-3 (copyleft)  🔴 Non-commercial  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Spatial Anchors

```yaml
layout_flow: "Top-to-bottom three-layer hierarchy: experiment design → signal analysis → TSFM preprocessing"
spatial_anchors:
  experiment_layer:
    x: 0.5
    y: 0.15
    content: "PsychoPy + PySilSub as experiment design layer"
  analysis_layer:
    x: 0.5
    y: 0.50
    content: "9 existing tools positioned by hardware coupling"
  hardware_coupled_zone:
    x: 0.25
    y: 0.45
    content: "PupilEXT, PyPlr, PupilMetrics on left (device-specific)"
  device_agnostic_zone:
    x: 0.72
    y: 0.45
    content: "GazeR, eyeris, PupEyes, PuPl, PupilSense on right"
  traditional_warning:
    x: 0.5
    y: 0.65
    content: "Warning: all tools use traditional preprocessing"
  preprocessing_layer:
    x: 0.5
    y: 0.85
    content: "foundation_PLR as TSFM preprocessing layer (THE GAP)"
  license_legend:
    x: 0.5
    y: 0.96
    content: "MIT / GPL-3 / Non-commercial license badges"
```

## Content Elements

### Key Structures
| Name | Role (Semantic Tag) | Description |
|------|---------------------|-------------|
| PsychoPy | `highlight_accent` | Python, MIT, 1900 stars. Experiment platform: stimulus timing, LMS color spaces, hardware sync. 40K+ users, 4K+ citations |
| PySilSub | `secondary_pathway` | Python, MIT, 13 stars. Silent substitution stimulus computation. Photoreceptor targeting. J. Vision 2023 |
| PupilEXT | `secondary_pathway` | C++/Qt, GPLv3+NC, 135 stars. 6 pupil detection algorithms. Basler cameras. Frontiers in Neuroscience |
| PyPlr | `secondary_pathway` | Python, MIT, 14 stars. Pupil Core + STLAB acquisition. PLR stimulus delivery. Inactive since 2022 |
| PupilMetrics | `secondary_pathway` | Python, GPLv3, 1 star. NeuroLight/Diagnosys clinical. Interactive GUI artifact correction. Nature Sci. Rep. 2024 |
| GazeR | `secondary_pathway` | R, GPL-3, 52 stars. EyeLink+Tobii+Neon. Hershman blink detection, interpolation, baseline correction. BRM 2020 |
| eyeris | `secondary_pathway` | R, MIT, 5 stars. CRAN v3.0.1. 10-stage pipeline (deblink→detransient→interpolate→filter→downsample→bin→detrend→z-score). DuckDB/BIDS |
| PuPl | `secondary_pathway` | MATLAB/Octave, CC-BY-NC, 11 stars. GUI pipeliner. SMI/BIDS. Reproducible pipeline export |
| PupEyes | `secondary_pathway` | Python, GPL-3, 8 stars. EyeLink+Tobii. Plotly Dash interactive viz. Blink detection comparison tools |
| PupilSense | `secondary_pathway` | Python, MIT, 66 stars. Smartphone camera pupillometry. Detectron2 DL segmentation. Depression screening |
| foundation_PLR | `highlight_accent` | Python, TSFM-based, device-agnostic. 11 outlier + 8 imputation methods. STRATOS evaluation. THE GAP |
| Layer arrows | `primary_pathway` | Vertical flow: experiment design → signal analysis → preprocessing |
| Traditional warning | `abnormal_warning` | "ALL tools use traditional preprocessing" banner across analysis layer |
| License legend | `annotation` | Color-coded: MIT=green, GPL-3=yellow, Non-commercial=red |

### Relationships/Connections
| From | To | Type | Label |
|------|-----|------|-------|
| PsychoPy | PySilSub | Bidirectional arrow | "Stimulus computation ↔ delivery" |
| Experiment layer | Analysis layer | Downward arrow | "Delivers stimulus → records response" |
| Analysis layer | Preprocessing layer | Downward arrow | "Raw signal → needs preprocessing" |
| All analysis tools | Traditional warning | Connection | "Threshold + interpolation + filter" |
| foundation_PLR | Gap indicator | Star highlight | "First TSFM-based preprocessing" |
| PsychoPy | PyPlr | Dashed line | "Same ecosystem (Pupil Labs)" |

### Callout Boxes
| Title | Content | Location |
|-------|---------|----------|
| "EXPERIMENT DESIGN" | PsychoPy (40K+ users) and PySilSub enable photoreceptor-selective stimuli with precise timing | Top layer |
| "ALL TRADITIONAL" | Every existing analysis tool uses threshold-based blink detection + linear interpolation — no ML/TSFM | Middle layer warning |
| "THE GAP" | No TSFM-based preprocessing exists. foundation_PLR is the first to evaluate MOMENT, UniTS, SAITS for pupillometry | Bottom layer |
| "ABSTRACTION BENEFIT" | Chronobiologists and psychologists design experiments without worrying about TSFM — this repo handles it | Bottom strip |

## Text Content

### Labels (Max 30 chars each)
- Label 1: PsychoPy (Experiment Platform)
- Label 2: PySilSub (Silent Subst.)
- Label 3: PupilEXT (C++, Basler)
- Label 4: PyPlr (Pupil Core)
- Label 5: PupilMetrics (Clinical)
- Label 6: GazeR (R, Multi-device)
- Label 7: eyeris (R, CRAN, 10-stage)
- Label 8: PuPl (MATLAB, GUI)
- Label 9: PupEyes (Python, Dash)
- Label 10: PupilSense (Smartphone)
- Label 11: foundation_PLR (TSFM)
- Label 12: Traditional Preprocessing
- Label 13: TSFM Preprocessing (NEW)

### Caption (for embedding)
The pupillometry software ecosystem as a three-layer hierarchy: experiment design (PsychoPy, PySilSub), signal analysis (9 tools spanning Python/R/MATLAB/C++), and preprocessing. All existing tools use traditional signal processing. foundation_PLR introduces TSFM-based preprocessing as a device-agnostic drop-in layer.

## Prompts for Nano Banana Pro

### Style Prompt
Medical illustration quality, ray-traced ambient occlusion, soft volumetric lighting, Economist off-white background (#FBF9F3), elegant scientific illustration, clean editorial layout, professional data visualization, hairline vector callout lines, Scientific American infographic style. Subtle eye anatomy background elements (iris, pupil) as watermark-level decoration.

### Content Prompt
Create a three-layer vertical hierarchy infographic showing the pupillometry software ecosystem:

TOP LAYER ("Experiment Design") — Warm cream background panel with gold accent border. Two tool cards:
- PsychoPy: Large card with Python snake icon, green MIT license shield, "1900 stars" badge. Features: "Stimulus timing, LMS colors, Hardware sync". Prominent because it's the ecosystem anchor (40K+ users).
- PySilSub: Smaller card with Python icon, green MIT shield, "13 stars". Features: "Silent substitution, Cone-isolating stimuli". Connected to PsychoPy with a bidirectional arrow labeled "Stimulus computation ↔ Delivery".
A subtle dashed line connects PySilSub to PyPlr below (same research group).
Caption strip: "Cognitive neuroscientists, chronobiologists, psychologists design experiments here."

MIDDLE LAYER ("Signal Analysis") — Larger panel, light gray background. Nine tool cards arranged in two groups:

Left group (hardware-coupled, slightly faded):
- PupilEXT: C++ brackets icon, red non-commercial shield, "135 stars". "6 detection algorithms, Basler cameras"
- PyPlr: Python icon, green MIT shield, "14 stars". "Pupil Core, STLAB, Inactive 2022"
- PupilMetrics: Python icon, yellow GPL-3 shield, "1 star". "NeuroLight/Diagnosys, Nature pub."

Right group (multi-device / device-agnostic, slightly brighter):
- GazeR: R logo, yellow GPL-3 shield, "52 stars". "EyeLink+Tobii+Neon, Hershman blink detection"
- eyeris: R logo, green MIT shield, "5 stars". "CRAN, 10-stage pipeline, DuckDB/BIDS"
- PuPl: MATLAB diamond, red non-commercial shield, "11 stars". "GUI pipeliner, SMI/BIDS"
- PupEyes: Python icon, yellow GPL-3 shield, "8 stars". "EyeLink+Tobii, Plotly Dash interactive viz"
- PupilSense: Python icon, green MIT shield, "66 stars". "Smartphone camera, DL segmentation"

A prominent warning banner stretches across the bottom of this layer in muted red: "ALL tools use TRADITIONAL preprocessing: threshold detection + linear interpolation + Butterworth filter"

BOTTOM LAYER ("TSFM Preprocessing — THE GAP") — Gold-accented panel, highlighted as the key contribution. Single large card for foundation_PLR with gold border and star icon:
- "11 outlier detection methods (MOMENT, UniTS, LOF, Ensembles)"
- "8 imputation methods (SAITS, CSDI, MOMENT zero-shot)"
- "STRATOS evaluation (5 metric domains)"
- "Device-agnostic: any 30 Hz pupil signal"
A callout reads: "Researchers design experiments above without worrying about TSFM intricacies."

Vertical downward arrows connect the three layers with labels: "Delivers stimulus → Records response" (top→middle) and "Raw signal → Needs robust preprocessing" (middle→bottom).

BOTTOM STRIP — License legend with three colored shields: Green = MIT (commercial OK), Yellow = GPL-3 (copyleft), Red = Non-commercial.

### Refinement Notes
- The three-layer hierarchy should be the dominant visual structure
- PsychoPy should be visually prominent (largest experiment-layer card) given its 40K+ user base
- Hardware-coupled tools should feel slightly "constrained" vs device-agnostic tools
- The "ALL TRADITIONAL" warning should be visually striking but not garish
- foundation_PLR in the bottom layer should have gold highlighting as the ecosystem contribution
- Language icons (Python snake, R logo, MATLAB diamond, C++ brackets) should be recognizable at small scale
- License shields should be small but color-coded and readable
- Star counts as small GitHub-style badges
- The vertical flow should feel natural: design → analyze → preprocess
- Include subtle eye anatomy elements (iris, pupil) as background watermark

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-99",
    "title": "The Pupillometry Software Ecosystem: From Experiment Design to TSFM Preprocessing"
  },
  "content_architecture": {
    "primary_message": "All existing pupillometry tools use traditional preprocessing. foundation_PLR introduces TSFM-based preprocessing as a device-agnostic drop-in layer for the ecosystem.",
    "layout_flow": "Top-to-bottom three-layer hierarchy: experiment design → signal analysis → TSFM preprocessing",
    "spatial_anchors": {
      "experiment_layer": {"x": 0.5, "y": 0.15},
      "analysis_layer": {"x": 0.5, "y": 0.50},
      "hardware_coupled": {"x": 0.25, "y": 0.45},
      "device_agnostic": {"x": 0.72, "y": 0.45},
      "traditional_warning": {"x": 0.5, "y": 0.65},
      "preprocessing_layer": {"x": 0.5, "y": 0.85},
      "license_legend": {"x": 0.5, "y": 0.96}
    },
    "key_structures": [
      {"name": "PsychoPy", "role": "highlight_accent", "is_highlighted": true, "labels": ["Python", "MIT", "1900 stars", "Experiment platform"]},
      {"name": "PySilSub", "role": "secondary_pathway", "is_highlighted": false, "labels": ["Python", "MIT", "13 stars", "Silent substitution"]},
      {"name": "PupilEXT", "role": "secondary_pathway", "is_highlighted": false, "labels": ["C++", "GPLv3+NC", "135 stars", "Basler cameras"]},
      {"name": "PyPlr", "role": "secondary_pathway", "is_highlighted": false, "labels": ["Python", "MIT", "14 stars", "Pupil Core"]},
      {"name": "PupilMetrics", "role": "secondary_pathway", "is_highlighted": false, "labels": ["Python", "GPLv3", "1 star", "NeuroLight/Diagnosys"]},
      {"name": "GazeR", "role": "secondary_pathway", "is_highlighted": false, "labels": ["R", "GPL-3", "52 stars", "EyeLink+Tobii+Neon"]},
      {"name": "eyeris", "role": "secondary_pathway", "is_highlighted": false, "labels": ["R", "MIT", "5 stars", "CRAN, 10-stage pipeline"]},
      {"name": "PuPl", "role": "secondary_pathway", "is_highlighted": false, "labels": ["MATLAB", "CC-BY-NC", "11 stars", "GUI pipeliner"]},
      {"name": "PupEyes", "role": "secondary_pathway", "is_highlighted": false, "labels": ["Python", "GPL-3", "8 stars", "Plotly Dash viz"]},
      {"name": "PupilSense", "role": "secondary_pathway", "is_highlighted": false, "labels": ["Python", "MIT", "66 stars", "Smartphone DL"]},
      {"name": "foundation_PLR", "role": "highlight_accent", "is_highlighted": true, "labels": ["Python", "TSFM", "Device-agnostic", "11 outlier + 8 imputation"]}
    ],
    "callout_boxes": [
      {"heading": "EXPERIMENT DESIGN", "body_text": "PsychoPy (40K+ users) + PySilSub enable photoreceptor-selective stimuli with precise timing"},
      {"heading": "ALL TRADITIONAL", "body_text": "Every analysis tool uses threshold blink detection + linear interpolation. No ML/TSFM."},
      {"heading": "THE GAP", "body_text": "No TSFM-based preprocessing exists. foundation_PLR evaluates MOMENT, UniTS, SAITS for pupillometry."},
      {"heading": "ABSTRACTION", "body_text": "Chronobiologists and psychologists design experiments without worrying about TSFM intricacies."}
    ],
    "tool_details": [
      {
        "name": "PsychoPy",
        "language": "Python",
        "license": "MIT",
        "license_commercial": true,
        "stars": 1900,
        "layer": "experiment_design",
        "hardware_coupling": "device-agnostic",
        "capabilities": ["stimulus timing", "LMS color spaces", "hardware synchronization", "Pupil Labs integration"],
        "last_active": "2026-01",
        "publication": "Peirce et al. (2019) Behavior Research Methods"
      },
      {
        "name": "PySilSub",
        "language": "Python",
        "license": "MIT",
        "license_commercial": true,
        "stars": 13,
        "layer": "experiment_design",
        "hardware_coupling": "device-agnostic",
        "capabilities": ["silent substitution computation", "photoreceptor targeting", "observer models"],
        "last_active": "2023-07",
        "publication": "Martin et al. (2023) Journal of Vision"
      },
      {
        "name": "PupilEXT",
        "language": "C++",
        "license": "GPLv3 + non-commercial",
        "license_commercial": false,
        "stars": 135,
        "layer": "signal_analysis",
        "hardware_coupling": "tight (Basler cameras)",
        "capabilities": ["6 pupil detection algorithms", "stereo camera mm-scale", "real-time processing", "offline batch"],
        "last_active": "2024-09",
        "publication": "Santini et al. (2021) Frontiers in Neuroscience"
      },
      {
        "name": "PyPlr",
        "language": "Python",
        "license": "MIT",
        "license_commercial": true,
        "stars": 14,
        "layer": "signal_analysis",
        "hardware_coupling": "tight (Pupil Core + STLAB)",
        "capabilities": ["stimulus design", "data extraction", "basic cleaning"],
        "last_active": "2022-11",
        "publication": "Martin & Spitschan, Zenodo"
      },
      {
        "name": "PupilMetrics",
        "language": "Python",
        "license": "GPLv3",
        "license_commercial": "copyleft",
        "stars": 1,
        "layer": "signal_analysis",
        "hardware_coupling": "tight (NeuroLight, Diagnosys)",
        "capabilities": ["interactive artifact correction", "flash-level analysis", "clinical outcome measures"],
        "last_active": "2023-04",
        "publication": "Nature Scientific Reports (2024)"
      },
      {
        "name": "GazeR",
        "language": "R",
        "license": "GPL-3",
        "license_commercial": "copyleft",
        "stars": 52,
        "layer": "signal_analysis",
        "hardware_coupling": "multi-device (EyeLink, Tobii, Neon)",
        "capabilities": ["Hershman blink detection", "interpolation", "baseline correction", "pupil scaling", "growth curve"],
        "last_active": "2025-10",
        "publication": "Geller et al. (2020) Behavior Research Methods"
      },
      {
        "name": "eyeris",
        "language": "R",
        "license": "MIT",
        "license_commercial": true,
        "stars": 5,
        "layer": "signal_analysis",
        "hardware_coupling": "partially agnostic (EyeLink primary, generic pipeline)",
        "capabilities": ["10-stage modular pipeline", "BIDS compliance", "DuckDB storage", "HTML QC reports", "MAD-based spike removal"],
        "last_active": "2026-02",
        "publication": "CRAN v3.0.1"
      },
      {
        "name": "PuPl",
        "language": "MATLAB/Octave",
        "license": "CC-BY-NC-4.0",
        "license_commercial": false,
        "stars": 11,
        "layer": "signal_analysis",
        "hardware_coupling": "partially agnostic (SMI, BIDS, custom)",
        "capabilities": ["GUI pipeliner", "reproducible pipeline export", "batch processing"],
        "last_active": "2025-01",
        "publication": null
      },
      {
        "name": "PupEyes",
        "language": "Python",
        "license": "GPL-3",
        "license_commercial": "copyleft",
        "stars": 8,
        "layer": "signal_analysis",
        "hardware_coupling": "multi-device (EyeLink, Tobii)",
        "capabilities": ["blink detection", "artifact rejection", "baseline correction", "Plotly Dash interactive viz"],
        "last_active": "2026-01",
        "publication": "Zhang & Jonides (2025) OSF Preprint"
      },
      {
        "name": "PupilSense",
        "language": "Python",
        "license": "MIT",
        "license_commercial": true,
        "stars": 66,
        "layer": "signal_analysis",
        "hardware_coupling": "smartphone only",
        "capabilities": ["Detectron2 pupil segmentation", "pupil-to-iris ratio", "naturalistic conditions"],
        "last_active": "2024-04",
        "publication": "Stevens et al. ACM MobileHCI (2024)"
      }
    ]
  }
}
```

## Alt Text

Three-layer hierarchy of the pupillometry software ecosystem. Top layer: experiment design tools (PsychoPy with 1900 stars, PySilSub for silent substitution). Middle layer: 9 signal analysis tools spanning Python, R, MATLAB, and C++, split between hardware-coupled (PupilEXT/Basler, PyPlr/Pupil Core, PupilMetrics/NeuroLight) and device-agnostic (GazeR, eyeris, PupEyes, PuPl, PupilSense). All use traditional preprocessing. Bottom layer: foundation_PLR fills the gap as the first TSFM-based preprocessing tool. License badges show MIT (green), GPL-3 (yellow), and non-commercial (red).

## Research Source

Full literature review with verified facts: `docs/planning/plr-repo-research.md`

## Status

- [x] Draft created
- [x] Literature review completed (10 repos)
- [x] Factual review passed (reviewer agent)
- [ ] Generated
- [ ] Placed in README/docs
